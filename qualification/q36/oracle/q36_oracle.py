"""Independent f64 oracle for the Q36 host reference tests (packet Q36-HR).

Written from the architecture rules of mlx-lm 0.31.0 (qwen3_5.py, qwen3_next.py, gated_delta.py),
not from the Rust code. Weights come from the same integer hash as the Rust tests, so both sides
hold identical (dyadic) values. Run with `python3 -I q36_oracle.py`; it prints the Rust constants.
"""
import math

MASK = 0xFFFFFFFF


def val(tag, i):
    h = (i * 2654435761 + tag * 40503 + 12345) & MASK
    return (((h >> 7) % 17) - 8) / 8.0


def vec(tag, n):
    return [val(tag, i) for i in range(n)]


def mat(tag, rows, cols):
    return [vec_row(tag, r * cols, cols) for r in range(rows)]


def vec_row(tag, start, n):
    return [val(tag, start + c) for c in range(n)]


def stack(tag, depth, rows, cols):
    return [mat_off(tag, d * rows * cols, rows, cols) for d in range(depth)]


def mat_off(tag, off, rows, cols):
    return [vec_row(tag, off + r * cols, cols) for r in range(rows)]


def norm_w(tag, n):
    return [1.0 + val(tag, i) / 4.0 for i in range(n)]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def linear(w, x):
    return [sum(a * b for a, b in zip(row, x)) for row in w]


def rms(x, w, eps):
    m = sum(v * v for v in x) / len(x)
    s = 1.0 / math.sqrt(m + eps)
    return [v * s * wi for v, wi in zip(x, w)]


def rms_none(x, eps):
    return rms(x, [1.0] * len(x), eps)


def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]


def rope(x, rot, pos, theta):
    x = list(x)
    half = rot // 2
    for i in range(half):
        ang = pos / theta ** (2.0 * i / rot)
        c, s = math.cos(ang), math.sin(ang)
        a, b = x[i], x[i + half]
        x[i] = a * c - b * s
        x[i + half] = b * c + a * s
    return x


class Delta:
    def __init__(self, tag, hidden, kh, kd, vh, vd, ck):
        self.kh, self.kd, self.vh, self.vd, self.ck = kh, kd, vh, vd, ck
        self.conv_dim = 2 * kh * kd + vh * vd
        self.qkv = mat(tag + 1, self.conv_dim, hidden)
        self.z = mat(tag + 2, vh * vd, hidden)
        self.b = mat(tag + 3, vh, hidden)
        self.a = mat(tag + 4, vh, hidden)
        self.out = mat(tag + 5, hidden, vh * vd)
        self.conv = [vec_row(tag + 6, c * ck, ck) for c in range(self.conv_dim)]
        self.a_log = [v / 2 for v in vec(tag + 7, vh)]
        self.dt_bias = [v / 2 for v in vec(tag + 8, vh)]
        self.norm = norm_w(tag + 9, vd)
        self.ring = [[0.0] * (ck - 1) for _ in range(self.conv_dim)]
        self.state = [[[0.0] * kd for _ in range(vd)] for _ in range(vh)]  # [h][j][i]

    def step(self, x, eps):
        kh, kd, vh, vd, ck = self.kh, self.kd, self.vh, self.vd, self.ck
        qkv = linear(self.qkv, x)
        z = linear(self.z, x)
        b = linear(self.b, x)
        a = linear(self.a, x)
        y = []
        for c in range(self.conv_dim):
            window = self.ring[c] + [qkv[c]]
            y.append(silu(sum(w * v for w, v in zip(self.conv[c], window))))
            self.ring[c] = window[1:]
        kw = kh * kd
        q = [y[g * kd:(g + 1) * kd] for g in range(kh)]
        k = [y[kw + g * kd:kw + (g + 1) * kd] for g in range(kh)]
        v = [y[2 * kw + h * vd:2 * kw + (h + 1) * vd] for h in range(vh)]
        inv = kd ** -0.5
        q = [[inv ** 2 * t for t in rms_none(r, 1e-6)] for r in q]
        k = [[inv * t for t in rms_none(r, 1e-6)] for r in k]
        rep = vh // kh
        out = []
        for h in range(vh):
            qh, kk = q[h // rep], k[h // rep]
            beta = sigmoid(b[h])
            g = math.exp(-math.exp(self.a_log[h]) * softplus(a[h] + self.dt_bias[h]))
            s = self.state[h]
            yh = []
            for j in range(vd):
                s[j] = [t * g for t in s[j]]
                kv = sum(t * u for t, u in zip(s[j], kk))
                d = (v[h][j] - kv) * beta
                s[j] = [t + u * d for t, u in zip(s[j], kk)]
                yh.append(sum(t * u for t, u in zip(s[j], qh)))
            normed = rms(yh, self.norm, eps)
            out += [silu(z[h * vd + j]) * normed[j] for j in range(vd)]
        return linear(self.out, out)


class Attn:
    def __init__(self, tag, hidden, hq, hkv, hd, rot, theta):
        self.hq, self.hkv, self.hd, self.rot, self.theta = hq, hkv, hd, rot, theta
        self.q = mat(tag + 1, hq * hd * 2, hidden)
        self.k = mat(tag + 2, hkv * hd, hidden)
        self.v = mat(tag + 3, hkv * hd, hidden)
        self.o = mat(tag + 4, hidden, hq * hd)
        self.q_norm = norm_w(tag + 5, hd)
        self.k_norm = norm_w(tag + 6, hd)
        self.keys, self.values = [], []

    def step(self, x, eps, pos):
        hq, hkv, hd = self.hq, self.hkv, self.hd
        qg = linear(self.q, x)
        k = linear(self.k, x)
        v = linear(self.v, x)
        qs, gates = [], []
        for h in range(hq):
            base = h * 2 * hd
            qs.append(rope(rms(qg[base:base + hd], self.q_norm, eps), self.rot, pos, self.theta))
            gates += qg[base + hd:base + 2 * hd]
        kn = [rope(rms(k[g * hd:(g + 1) * hd], self.k_norm, eps), self.rot, pos, self.theta) for g in range(hkv)]
        self.keys.append(kn)
        self.values.append([v[g * hd:(g + 1) * hd] for g in range(hkv)])
        rep = hq // hkv
        out = []
        for h in range(hq):
            g = h // rep
            sc = softmax([sum(a * b for a, b in zip(qs[h], self.keys[t][g])) * hd ** -0.5 for t in range(len(self.keys))])
            out += [sum(sc[t] * self.values[t][g][d] for t in range(len(sc))) for d in range(hd)]
        out = [o * sigmoid(g) for o, g in zip(out, gates)]
        return linear(self.o, out)


class Moe:
    def __init__(self, tag, hidden, experts, top_k, inter, shared):
        self.top_k = top_k
        self.router = mat(tag + 1, experts, hidden)
        self.gate = stack(tag + 2, experts, inter, hidden)
        self.up = stack(tag + 3, experts, inter, hidden)
        self.down = stack(tag + 4, experts, hidden, inter)
        self.sg = mat(tag + 5, shared, hidden)
        self.su = mat(tag + 6, shared, hidden)
        self.sd = mat(tag + 7, hidden, shared)
        self.sgate = mat(tag + 8, 1, hidden)
        self.margin = 1.0

    def step(self, x):
        p = softmax(linear(self.router, x))
        order = sorted(range(len(p)), key=lambda e: (-p[e], e))
        sel = order[:self.top_k]
        self.margin = min(self.margin, p[order[self.top_k - 1]] - p[order[self.top_k]])
        den = sum(p[e] for e in sel)
        out = [0.0] * len(x)
        for e in sel:
            g = linear(self.gate[e], x)
            u = linear(self.up[e], x)
            h = [silu(a) * b for a, b in zip(g, u)]
            o = linear(self.down[e], h)
            out = [t + p[e] / den * s for t, s in zip(out, o)]
        sg = linear(self.sg, x)
        su = linear(self.su, x)
        so = linear(self.sd, [silu(a) * b for a, b in zip(sg, su)])
        gate = sigmoid(linear(self.sgate, x)[0])
        return [t + gate * s for t, s in zip(out, so)], sel


def fmt(xs):
    return "[" + ", ".join(repr(round(v, 9)) for v in xs) + "]"


H = 4
EPS = 1e-6


def delta_scenario():
    d = Delta(0, H, 2, 2, 4, 2, 4)
    res = []
    for t in range(3):
        res.append(d.step(vec(100 + t, H), EPS))
    state = [t for h in d.state for row in h for t in row]
    return res, state


def attn_scenario(rot=2):
    a = Attn(10, H, 4, 2, 4, rot, 100.0)
    return [a.step(vec(110 + t, H), EPS, t) for t in range(3)]


def moe_scenario():
    m = Moe(30, H, 4, 2, 3, 2)
    outs, sels = [], []
    for t in range(3):
        o, s = m.step(vec(120 + t, H))
        outs.append(o)
        sels.append(s)
    return outs, sels, m.margin


VOCAB = 6


def model_scenario(tokens):
    embed = mat(50, VOCAB, H)
    layers = [
        (norm_w(51, H), norm_w(52, H), Delta(60, H, 2, 2, 4, 2, 4), Moe(70, H, 4, 2, 3, 2)),
        (norm_w(53, H), norm_w(54, H), Attn(80, H, 4, 2, 4, 2, 100.0), Moe(90, H, 4, 2, 3, 2)),
    ]
    final = norm_w(55, H)
    head = mat(56, VOCAB, H)
    logits_all, layer_all, margin = [], [], 1.0
    for pos, tok in enumerate(tokens):
        x = embed[tok]
        outs = []
        for (n1, n2, mixer, moe) in layers:
            nx = rms(x, n1, EPS)
            r = mixer.step(nx, EPS) if isinstance(mixer, Delta) else mixer.step(nx, EPS, pos)
            h = [a + b for a, b in zip(x, r)]
            m, _ = moe.step(rms(h, n2, EPS))
            margin = min(margin, moe.margin)
            x = [a + b for a, b in zip(h, m)]
            outs.append(x)
        logits_all.append(linear(head, rms(x, final, EPS)))
        layer_all.append(outs)
    return logits_all, layer_all, margin


if __name__ == "__main__":
    out, state = delta_scenario()
    print("DELTA_OUT", [fmt(o) for o in out])
    print("DELTA_STATE", fmt(state))
    print("ATTN_OUT", [fmt(o) for o in attn_scenario()])
    print("ATTN_OUT_ROT4", [fmt(o) for o in attn_scenario(4)])
    o, s, margin = moe_scenario()
    print("MOE_OUT", [fmt(x) for x in o], s, "margin", margin)
    lg, la, margin = model_scenario([1, 4, 2])
    print("MODEL_LOGITS", [fmt(x) for x in lg])
    print("MODEL_LAYERS", [[fmt(l) for l in step] for step in la], "margin", margin)
