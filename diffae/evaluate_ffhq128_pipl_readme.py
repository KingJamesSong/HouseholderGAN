#!/usr/bin/env python
"""PIPL evaluation following README_LR_HOP_PPL_PIPL.md.

Protocol highlights:
  - FFHQ test[0:2000], no shuffle, seed=0
  - eps=1e-4, DDIM T=100, batch=16 (125 batches / 2000 pairs)
  - Directions: SVD of regional projector W; max 10 effective dirs / region
  - Batch cycling: enc:0.., mid:0.., dec:0..
  - Same shared noise for base / edited; inject eps*v before style_r
  - LPIPS/eps^2, 1–99 percentile trim, report mean

Usage:
  python evaluate_ffhq128_pipl_readme.py --arm householder5
  python evaluate_ffhq128_pipl_readme.py --arm householder10
  python evaluate_ffhq128_pipl_readme.py --arm householder512
  python evaluate_ffhq128_pipl_readme.py --arm lora10
  python evaluate_ffhq128_pipl_readme.py --arm vanilla
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Tuple

import lpips
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import FFHQlmdb
from experiment import LitModel


ARMS = {
    'householder5': dict(kind='householder', rank=5,
                         ckpt='checkpoints/0927_ffhq128_autoenc_rank5_svd_random/checkpoints/step=50000.ckpt'),
    'householder10': dict(kind='householder', rank=10,
                          ckpt='checkpoints/0927_ffhq128_autoenc_rank10_svd_random/checkpoints/step=50000.ckpt'),
    'householder512': dict(kind='householder', rank=512,
                           ckpt='checkpoints/0927_ffhq128_autoenc_rank512_svd_random/checkpoints/step=50000.ckpt'),
    'lora10': dict(kind='lora', rank=10,
                   ckpt='checkpoints/0927_ffhq128_autoenc_lora10_random/checkpoints/step=50000.ckpt'),
    'vanilla': dict(kind='vanilla', rank=None,
                    ckpt='checkpoints/ffhq128_autoenc_130M/last.ckpt'),
}


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(arm: dict):
    kind = arm['kind']
    ckpt = arm['ckpt']
    if kind == 'vanilla':
        from run_ffhq128_base_eval import load_vanilla
        conf, model, _ = load_vanilla(ckpt)
    elif kind == 'lora':
        from run_ffhq128_projector_ablation import build_conf
        conf = build_conf('lora', arm['rank'])
        model = LitModel(conf)
        state = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'], strict=True)
    else:
        from run_ffhq128_projector_ablation import build_conf
        conf = build_conf('householder', arm['rank'])
        model = LitModel(conf)
        state = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'], strict=True)
    model.ema_model.eval()
    model.ema_model.to('cuda:0')
    return conf, model


def projector_weight(module) -> torch.Tensor:
    """Rebuild effective W matching forward() (CPU float32)."""
    if hasattr(module, 'A') and hasattr(module, 'B'):
        # LoRA: W = A @ B
        return (module.A.detach().float().cpu() @
                module.B.detach().float().cpu())
    if getattr(module, 'is_ortho', False):
        U = module.U.detach().float().cpu()
        V = module.V.detach().float().cpu()
        S = module.S.detach().float().cpu()
        QU = module.fasthpp(U)
        QV = module.fasthpp(V)
        if module.in_dim < module.out_dim:
            W = QU.mm(S).mm(QV)
        else:
            W = QV.mm(S).mm(QU)
        return W * float(module.scale)
    if hasattr(module, 'weight'):
        return module.weight.detach().float().cpu() * float(module.scale)
    raise TypeError(f'unsupported projector: {type(module)}')


def svd_directions(W: torch.Tensor, max_dirs: int = 10,
                   rel_thresh: float = 1e-6) -> torch.Tensor:
    """Return (k, D) unit right-singular vectors (Vh rows), k <= max_dirs."""
    # torch.linalg.svd -> U, S, Vh
    _, S, Vh = torch.linalg.svd(W, full_matrices=False)
    if S.numel() == 0:
        raise RuntimeError('empty singular values')
    thresh = float(S.max()) * rel_thresh
    n_eff = int((S > thresh).sum().item())
    k = max(1, min(max_dirs, n_eff, Vh.shape[0]))
    return Vh[:k].contiguous()


def extract_region_directions(ema, max_dirs: int = 10, vanilla: bool = False
                              ) -> List[Tuple[str, int, torch.Tensor]]:
    """List of (region_name, layer_index, directions[k,D])."""
    if vanilla:
        # No regional projectors: directions from a fixed orthonormal basis
        # of the semantic code (seeded QR), injected by adding to z.
        g = torch.Generator().manual_seed(0)
        M = torch.randn(512, 512, generator=g)
        Q, _ = torch.linalg.qr(M)
        dirs = Q[:, :max_dirs].T.contiguous()  # (max_dirs, 512)
        # single synthetic "global" region; layer_index unused
        return [('global', -1, dirs)]

    regions = [
        ('enc', 0, ema.style_enc),
        ('mid', 1, ema.style_mid),
        ('dec', 2, ema.style_dec),
    ]
    out = []
    for name, idx, mod in regions:
        W = projector_weight(mod)
        dirs = svd_directions(W, max_dirs=max_dirs)
        print(f'[dirs] {name}: W{tuple(W.shape)} n_dir={dirs.shape[0]} '
              f'||d0||={dirs[0].norm():.4f}')
        out.append((name, idx, dirs))
    return out


def build_group_schedule(region_dirs, n_batches: int):
    """Cycle enc:0.. mid:0.. dec:0.. across batches (README §4.3)."""
    groups = []
    for name, layer_index, dirs in region_dirs:
        for j in range(dirs.shape[0]):
            groups.append((name, layer_index, j, dirs[j]))
    assert groups, 'no PIPL direction groups'
    schedule = [groups[b % len(groups)] for b in range(n_batches)]
    # counts
    counts = {}
    for name, _, j, _ in schedule:
        key = f'{name}:{j}'
        counts[key] = counts.get(key, 0) + 1
    print(f'[schedule] n_groups={len(groups)} n_batches={n_batches}')
    for k, v in counts.items():
        print(f'  {k}: {v} batches ({v * 16} pairs)')
    return schedule


def trim_mean(distances: np.ndarray):
    # numpy<1.22 uses interpolation=; newer uses method=
    try:
        lo = np.percentile(distances, 1, method='lower')
        hi = np.percentile(distances, 99, method='higher')
    except TypeError:
        lo = np.percentile(distances, 1, interpolation='lower')
        hi = np.percentile(distances, 99, interpolation='higher')
    kept = distances[(distances >= lo) & (distances <= hi)]
    return {
        'mean': float(kept.mean()),
        'std': float(kept.std()),  # population std (numpy default)
        'raw_count': int(distances.size),
        'count': int(kept.size),
        'trim_lower': float(lo),
        'trim_upper': float(hi),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True, choices=sorted(ARMS.keys()))
    p.add_argument('--samples', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epsilon', type=float, default=1e-4)
    p.add_argument('--ddim-steps', type=int, default=100)
    p.add_argument('--pipl-directions', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data-path', type=str, default='datasets/ffhq256.lmdb')
    p.add_argument('--out-dir', type=str, default='evals/readme_pipl')
    return p.parse_args()


def main():
    args = parse_args()
    arm = ARMS[args.arm]
    if not os.path.exists(arm['ckpt']):
        raise FileNotFoundError(arm['ckpt'])
    assert args.samples % args.batch_size == 0
    n_batches = args.samples // args.batch_size

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'[pipl] arm={args.arm} ckpt={arm["ckpt"]}')
    print(f'[pipl] samples={args.samples} batch={args.batch_size} '
          f'eps={args.epsilon} T={args.ddim_steps} seed={args.seed}')

    conf, model = load_model(arm)
    region_dirs = extract_region_directions(
        model.ema_model,
        max_dirs=args.pipl_directions,
        vanilla=(arm['kind'] == 'vanilla'),
    )
    schedule = build_group_schedule(region_dirs, n_batches)

    data = FFHQlmdb(path=args.data_path, image_size=conf.img_size,
                    split='test', do_augment=False)
    subset = Subset(data, list(range(args.samples)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, drop_last=False)

    percept = lpips.LPIPS(net='vgg').to('cuda:0').eval()
    distances = []
    per_group = {}

    with torch.no_grad():
        for b_idx, batch in enumerate(tqdm(loader, total=n_batches)):
            imgs = batch['img'].to('cuda:0')
            assert imgs.size(0) == args.batch_size
            name, layer_index, j, v = schedule[b_idx]
            group_key = f'{name}:{j}'

            cond = model.encode(imgs)
            noise = torch.randn_like(imgs)
            direction = (args.epsilon * v.to('cuda:0')).unsqueeze(0)
            direction = direction.expand(imgs.size(0), -1)

            base = model.render(noise, cond=cond, T=args.ddim_steps)
            if arm['kind'] == 'vanilla':
                edited = model.render(
                    noise, cond=cond + direction, T=args.ddim_steps)
            else:
                edited = model.render(
                    noise, cond=cond, T=args.ddim_steps,
                    layer_index=layer_index, direction=direction)

            # render returns [0,1]; LPIPS expects roughly [-1,1]
            base_in = base * 2 - 1
            edit_in = edited * 2 - 1
            raw = percept(base_in, edit_in).view(-1)
            scaled = (raw / (args.epsilon ** 2)).detach().cpu().numpy()
            distances.append(scaled)
            per_group.setdefault(group_key, []).append(scaled)

    distances = np.concatenate(distances, 0)
    assert distances.shape[0] == args.samples, distances.shape

    summary = trim_mean(distances)
    group_stats = {
        k: trim_mean(np.concatenate(vs, 0)) for k, vs in per_group.items()
    }

    out = {
        'arm': args.arm,
        'kind': arm['kind'],
        'rank': arm['rank'],
        'ckpt': arm['ckpt'],
        'epsilon': args.epsilon,
        'samples': args.samples,
        'batch_size': args.batch_size,
        'ddim_steps': args.ddim_steps,
        'seed': args.seed,
        'pipl_directions_cap': args.pipl_directions,
        'protocol': 'README_LR_HOP_PPL_PIPL.md §4',
        'pipl': summary,
        'per_group': group_stats,
    }
    tag = (f'{args.arm}_step50000_eps'
           f'{str(args.epsilon).replace(".", "p")}_n{args.samples}')
    out_json = os.path.join(args.out_dir, f'{tag}_pipl.json')
    out_txt = os.path.join('evals', f'readme_{tag}_pipl.txt')
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    with open(out_txt, 'w') as f:
        f.write(json.dumps({
            'pipl': summary['mean'],
            'std': summary['std'],
            'raw_count': summary['raw_count'],
            'count': summary['count'],
            'eps': args.epsilon,
            'n_sample': args.samples,
            'arm': args.arm,
            'ckpt': arm['ckpt'],
            'protocol': 'README_LR_HOP_PPL_PIPL.md',
        }) + '\n')
    print(f'[pipl] {args.arm}: mean={summary["mean"]:.6f} '
          f'std={summary["std"]:.6f} kept={summary["count"]}/{summary["raw_count"]}')
    print(f'[pipl] wrote {out_json}')
    print(f'[pipl] wrote {out_txt}')


if __name__ == '__main__':
    main()
