"""CPU FP32 block oracle. PYTHONPATH points to z-lab/dflash; argv[1] = HF directory.
Synthetic target taps, normalized noise inputs and a compact shared-head test double.
Only head rows [0,64) are eligible (exercise selector independently of target quantization).
"""
import json
import sys
from pathlib import Path
import torch
from transformers import Qwen3Config
from safetensors.torch import load_file
from dflash.model import DFlash2DraftModel


def run(root):
    torch.set_num_threads(4)
    torch.manual_seed(9378)
    cfg = Qwen3Config.from_json_file(str(root / 'config.json'))
    cfg.rope_theta = 10000.0
    with torch.device('meta'):
        model = DFlash2DraftModel(cfg)
    state = load_file(str(root / 'model.safetensors'))
    for name in ['predecessor_codebook', 'successor_codebook']:
        key = 'candidate_selector.' + name
        state[key + '.weight'] = state.pop(key)
    model.load_state_dict(state, assign=True, strict=True)
    model.float().eval()
    # RoPE inv_freq is nonpersistent, so initialize outside meta device.
    model.rotary_emb = type(model.rotary_emb)(cfg)
    noise = torch.randn(1, 2, 2304)[:, [0,1,1,1,1,1,1,1]].contiguous()
    taps = torch.randn(1, 3, 9216)
    head = torch.randn(64, 2304) / (2304 ** .5)
    hidden = model(position_ids=torch.arange(11)[None], noise_embedding=noise, target_hidden=taps)[:, 1:]
    raw, ids = (hidden @ head.T).topk(16, sorted=True)
    selector = model.candidate_selector
    projected = selector.hidden_projection(hidden)
    pred = torch.tensor([33]); tokens, scores = [], []
    for i in range(7):
        correction = ((projected[:,i] * selector.predecessor_codebook(pred))[:,None] * selector.successor_codebook(ids[:,i])).sum(-1)
        corrected = 30 * torch.tanh((raw[:,i] + correction) / 30)
        pred = ids[:,i].gather(-1, corrected.argmax(-1,keepdim=True))[:,0]
        tokens.append(pred.item()); scores.append(corrected[0].tolist())
    # Post-correction tanh is monotonic: the uncapped published selector must choose
    # the same path for these finite, non-saturated scores. Keep this independent check.
    reference_path, _, _ = selector.select(hidden, hidden @ head.T, torch.tensor([33]), 0.0)
    assert reference_path[0].tolist() == tokens
    out = Path(__file__).parent
    for name, x in [('taps', taps), ('noise', noise), ('head', head)]:
        x.numpy().astype('<f4').tofile(out / (name + '.f32le'))
    (out/'expected.json').write_text(json.dumps(dict(anchor=33,candidates=ids[0].tolist(),tokens=tokens,scores=scores),indent=2)+'\n')
    print(tokens)


with torch.inference_mode():
    run(Path(sys.argv[1]))
