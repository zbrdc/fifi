// web-model-router.mjs — loopback-only /v1 model router for the LAN web GUI.
// Merges /v1/models across engine backends and routes each request by its
// "model" field, injecting the matching backend's API key. The vite preview
// proxy (lan-vite.config.mjs) points FIFI_BACKEND here; nothing on the LAN
// talks to this process directly.
import { createServer, request } from 'node:http';
import { readFileSync } from 'node:fs';

const RUN = process.env.FIFI_PRIMARY_RUN ?? '/home/ben/fifi-scratch/a81ebdf26a2a/qwen36-local';
const RUN_2080TI = process.env.FIFI_SECONDARY_RUN ?? '/home/ben/fifi-scratch/a81ebdf26a2a/qwen36-2080ti';
const HOST = '127.0.0.1';
const PORT = Number(process.env.FIFI_MODEL_ROUTER_PORT ?? 18795);
const MAX_BODY = 20 * 1024 * 1024;
const CLOSE_BACKEND_CONNECTIONS = process.env.FIFI_ROUTER_CLOSE_BACKEND_CONNECTIONS === '1';

// ponytail: static backend table; add a third engine by appending here.
const BACKENDS = [
  { model: 'fifi-q36-2080ti', origin: 'http://127.0.0.1:18793', key: readFileSync(`${RUN_2080TI}/api-key`, 'utf8').trim() },
  { model: 'fifi-q36', origin: 'http://127.0.0.1:18791', key: readFileSync(`${RUN}/api-key`, 'utf8').trim() },
];
// Modelless requests (GET /v1/system, /ready, /health, monitor panels) are
// pair-by-design: the pair is the primary engine and its behavior is unchanged.
const DEFAULT = BACKENDS[BACKENDS.length - 1];

function pick(model) {
  return BACKENDS.find(b => b.model === model) ?? DEFAULT;
}

function forward(req, body, backend, res) {
  const { pathname, search } = new URL(req.url, 'http://x');
  const headers = { 'content-type': req.headers['content-type'] ?? 'application/json' };
  if (req.headers.accept) headers['accept'] = req.headers.accept;
  if (body.length) headers['content-length'] = String(body.length);
  headers['authorization'] = 'Bearer ' + backend.key;
  // The opt-in policy avoids pooling completed sockets; streams stay open until completion.
  const preq = request(backend.origin + pathname + (search ?? ''), {
    method: req.method, headers, ...(CLOSE_BACKEND_CONNECTIONS ? { agent: false } : {}),
  }, pres => {
    const hop = { connection: 1, 'transfer-encoding': 1, 'keep-alive': 1 };
    const out = {}; for (const [k, v] of Object.entries(pres.headers)) if (!hop[k.toLowerCase()]) out[k] = v;
    res.writeHead(pres.statusCode ?? 502, out);
    pres.on('error', () => res.destroy());
    pres.pipe(res);
  });
  preq.on('error', () => {
    if (!res.headersSent) { res.writeHead(502, { 'content-type': 'application/json' }); res.end(JSON.stringify({ error: 'backend unreachable' })); }
    else res.destroy();
  });
  // GUI stop button: client gone mid-stream -> cancel the upstream generation too.
  res.on('close', () => { if (!res.writableEnded) preq.destroy(); });
  preq.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = []; let size = 0; let oversize = false;
    req.on('data', c => { size += c.length; if (size > MAX_BODY) { oversize = true; req.removeAllListeners('data'); return; } chunks.push(c); });
    req.on('end', () => oversize ? reject(new Error('too large')) : resolve(Buffer.concat(chunks)));
    req.on('error', () => reject(new Error('client aborted'))); // res is dead; nothing to answer
  });
}

async function mergedModels(res) {
  // The opt-in policy avoids retaining idle model-list sockets on each backend.
  const lists = await Promise.all(BACKENDS.map(b =>
    fetch(b.origin + '/v1/models', {
      headers: { authorization: 'Bearer ' + b.key, ...(CLOSE_BACKEND_CONNECTIONS ? { connection: 'close' } : {}) },
      signal: AbortSignal.timeout(4000),
    })
      .then(r => r.json()).catch(() => null)));
  const data = lists.flatMap((l, i) => l?.data ?? [{ id: BACKENDS[i].model, object: 'model', owned_by: 'fifi' }]);
  // The GUI selects the first model when it has no saved choice. Prefer the P2P engine.
  data.sort((a, b) => Number(b.id === DEFAULT.model) - Number(a.id === DEFAULT.model));
  if (!lists.some(l => l?.data?.length)) { res.writeHead(502, { 'content-type': 'application/json' }); res.end(JSON.stringify({ error: 'no backends reachable' })); return; }
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ object: 'list', data }));
}

createServer((req, res) => {
  let url;
  try { url = new URL(req.url, 'http://x'); } catch { res.writeHead(400, { 'content-type': 'application/json' }); res.end('{"error":"bad request target"}'); return; }
  if (req.method === 'GET' && url.pathname === '/v1/models') { mergedModels(res); return; }
  let traceBackend;
  if (req.method === 'GET' && url.pathname === '/v1/sampler-trace' && 'x-fifi-trace-model' in req.headers) {
    traceBackend = BACKENDS.find(b => b.model === req.headers['x-fifi-trace-model']);
    if (!traceBackend) { res.writeHead(400, { 'content-type': 'application/json' }); res.end('{"error":"unknown trace model"}'); return; }
  }
  readBody(req).then(body => {
    let model = '';
    if (body.length && (req.headers['content-type'] ?? '').includes('application/json')) {
      try { model = String(JSON.parse(body.toString())?.model ?? ''); } catch { /* route to default */ }
    }
    forward(req, body, traceBackend ?? pick(model), res);
  }).catch(e => {
    if (e.message === 'too large') { res.writeHead(413, { 'content-type': 'application/json' }); res.end('{"error":"body too large"}'); res.once('finish', () => req.destroy()); }
  });
}).listen(PORT, HOST, () => console.log(`model router on ${HOST}:${PORT} -> ${BACKENDS.map(b => b.model).join(', ')}`));
