#!/usr/bin/env python
"""PPL evaluation following README_LR_HOP_PPL_PIPL.md §3.

Protocol:
  - FFHQ test[0:2000], seed=0
  - Partner pairing via np.random.default_rng(0) circular permutation
  - ppl-sampling=end → t=0:
        x0 = G(n, z1)
        x1 = G(n, z1 + eps*(z2-z1))
  - Shared noise n; DDIM T=100; eps=1e-4
  - LPIPS/eps^2, 1–99 percentile trim mean
"""
from __future__ import annotations

import argparse
import json
import os

import lpips
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import FFHQlmdb
from evaluate_ffhq128_pipl_readme import ARMS, load_model, seed_everything, trim_mean


def build_partner_indices(n: int, seed: int = 0) -> np.ndarray:
    """Circular partners from a seeded permutation (README §3.1)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    partner = np.empty(n, dtype=np.int64)
    for i in range(n):
        start = int(perm[i])
        end = int(perm[i - 1])
        partner[start] = end
    assert not np.any(partner == np.arange(n))
    return partner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True, choices=sorted(ARMS.keys()))
    p.add_argument('--samples', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epsilon', type=float, default=1e-4)
    p.add_argument('--ddim-steps', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data-path', type=str, default='datasets/ffhq256.lmdb')
    p.add_argument('--out-dir', type=str, default='evals/readme_ppl')
    p.add_argument('--ppl-sampling', type=str, default='end', choices=['end'])
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

    print(f'[ppl] arm={args.arm} ckpt={arm["ckpt"]}')
    print(f'[ppl] samples={args.samples} batch={args.batch_size} '
          f'eps={args.epsilon} T={args.ddim_steps} sampling={args.ppl_sampling}')

    conf, model = load_model(arm)

    data = FFHQlmdb(path=args.data_path, image_size=conf.img_size,
                    split='test', do_augment=False)
    subset = Subset(data, list(range(args.samples)))
    loader_all = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, drop_last=False)
    imgs_all = []
    for batch in loader_all:
        imgs_all.append(batch['img'])
    imgs_all = torch.cat(imgs_all, dim=0)
    assert imgs_all.size(0) == args.samples

    partner = build_partner_indices(args.samples, seed=args.seed)
    print(f'[ppl] partner[:8]={partner[:8].tolist()}')

    with torch.no_grad():
        zs = []
        for i in range(n_batches):
            sl = slice(i * args.batch_size, (i + 1) * args.batch_size)
            zs.append(model.encode(imgs_all[sl].to('cuda:0')).cpu())
        z_all = torch.cat(zs, dim=0)

    percept = lpips.LPIPS(net='vgg').to('cuda:0').eval()
    distances = []

    # Reset seed before generation (README: reset RNG at start of PPL)
    seed_everything(args.seed)

    with torch.no_grad():
        for i in tqdm(range(n_batches), total=n_batches):
            idx = np.arange(i * args.batch_size, (i + 1) * args.batch_size)
            z1 = z_all[idx].to('cuda:0')
            z2 = z_all[partner[idx]].to('cuda:0')

            noise = torch.randn(
                args.batch_size, 3, conf.img_size, conf.img_size,
                device='cuda:0')
            z_a = z1
            z_b = z1 + args.epsilon * (z2 - z1)

            img_a = model.render(noise, cond=z_a, T=args.ddim_steps)
            img_b = model.render(noise, cond=z_b, T=args.ddim_steps)

            raw = percept(img_a * 2 - 1, img_b * 2 - 1).view(-1)
            scaled = (raw / (args.epsilon ** 2)).detach().cpu().numpy()
            distances.append(scaled)

    distances = np.concatenate(distances, 0)
    assert distances.shape[0] == args.samples, distances.shape
    summary = trim_mean(distances)

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
        'ppl_sampling': args.ppl_sampling,
        'ppl_partner_indices': partner.tolist(),
        'protocol': 'README_LR_HOP_PPL_PIPL.md §3',
        'ppl': summary,
    }
    tag = (f'{args.arm}_step50000_eps'
           f'{str(args.epsilon).replace(".", "p")}_n{args.samples}')
    out_json = os.path.join(args.out_dir, f'{tag}_ppl.json')
    out_txt = os.path.join('evals', f'readme_{tag}_ppl.txt')
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    with open(out_txt, 'w') as f:
        f.write(json.dumps({
            'ppl': summary['mean'],
            'std': summary['std'],
            'raw_count': summary['raw_count'],
            'count': summary['count'],
            'eps': args.epsilon,
            'n_sample': args.samples,
            'ppl_sampling': args.ppl_sampling,
            'arm': args.arm,
            'ckpt': arm['ckpt'],
            'protocol': 'README_LR_HOP_PPL_PIPL.md',
        }) + '\n')
    print(f'[ppl] {args.arm}: mean={summary["mean"]:.6f} '
          f'std={summary["std"]:.6f} kept={summary["count"]}/{summary["raw_count"]}')
    print(f'[ppl] wrote {out_json}')
    print(f'[ppl] wrote {out_txt}')


if __name__ == '__main__':
    main()
