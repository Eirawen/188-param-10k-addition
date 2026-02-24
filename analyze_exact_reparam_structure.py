"""Quick preflight audit for exact reparameterization assumptions."""

from __future__ import annotations

import importlib
import torch

TARGETS = [
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.v_proj",
    "layers.0.self_attn.o_proj",
    "layers.0.mlp.up_proj",
    "layers.0.mlp.down_proj",
    "layers.1.self_attn.q_proj",
    "layers.1.self_attn.k_proj",
    "layers.1.self_attn.v_proj",
    "layers.1.self_attn.o_proj",
    "layers.1.mlp.up_proj",
    "layers.1.mlp.down_proj",
]

def resolve(obj, path):
    cur = obj
    for p in path.split('.'):
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    return cur

def main(module_name: str):
    mod = importlib.import_module(module_name)
    model = mod.build_magic_model()
    print(f'module={module_name}')
    emb = model.embed_tokens.weight.detach() if hasattr(model.embed_tokens, 'weight') else model.embed_tokens.weight
    print('embed_rank', int(torch.linalg.matrix_rank(emb.float()).item()), 'shape', tuple(emb.shape))
    print('lm_head_rank', int(torch.linalg.matrix_rank(model.lm_head.weight.detach().float()).item()), 'shape', tuple(model.lm_head.weight.shape))
    for t in TARGETS:
        w = resolve(model, t).weight.detach()
        print(t, 'rank', int(torch.linalg.matrix_rank(w.float()).item()), 'shape', tuple(w.shape))
    g0 = model.layers[0].mlp.gate_proj.weight.detach()
    print('layer0_gate_nonzero_idx', (g0 != 0).nonzero(as_tuple=False).tolist())
    print('v_proj_equal', torch.equal(model.layers[0].self_attn.v_proj.weight.detach(), model.layers[1].self_attn.v_proj.weight.detach()))

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--module', default='param_343')
    args = ap.parse_args()
    main(args.module)
