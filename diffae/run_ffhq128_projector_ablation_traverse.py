#!/usr/bin/env python
"""Paper-style latent traversal for Householder / LoRA projectors.

Layout matches Fig.13-style grids: columns = different faces, rows = alphas
along one closed-form eigenvector (injected via layer_index + direction).

Usage:
  python run_ffhq128_projector_ablation_traverse.py \
    --method householder --rank 512 --step 50000 \
    --n_sample 7 --n_steps 4 --degree 5 --indices 0 1 2 3 4 5 6 7
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from dataset import FFHQlmdb
from experiment import LitModel
from run_ffhq128_projector_ablation import build_conf, exp_name, step_ckpt_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['householder', 'lora'])
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--factor', type=str, default=None)
    p.add_argument('--n_sample', type=int, default=7,
                   help='number of face columns')
    p.add_argument('--n_steps', type=int, default=4,
                   help='number of traversal rows (alphas)')
    p.add_argument('--degree', type=float, default=5.0,
                   help='alpha range is [-degree, +degree]')
    p.add_argument('--T', type=int, default=20,
                   help='DDIM steps for editing render (apply_factor uses 20)')
    p.add_argument('--T_encode', type=int, default=100,
                   help='DDIM steps for stochastic encode')
    p.add_argument('--indices', type=int, nargs='+', default=None,
                   help='eigenvector indices to visualize; default first 8')
    p.add_argument('--layers', type=int, nargs='+', default=[0, 1, 2],
                   help='0=style_enc, 1=style_mid, 2=style_dec')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data_path', type=str, default='datasets/ffhq256.lmdb')
    p.add_argument('--out_dir', type=str, default=None)
    return p.parse_args()


def load_ema_factors(factor_path):
    eigvec_dict = torch.load(factor_path, map_location='cpu')
    ema_items = [(k, v) for k, v in eigvec_dict.items() if k.startswith('ema_')]
    if not ema_items:
        ema_items = list(eigvec_dict.items())
    # stable order: enc / mid / dec if present
    def sort_key(item):
        k = item[0]
        if 'style_enc' in k:
            return 0
        if 'style_mid' in k:
            return 1
        if 'style_dec' in k:
            return 2
        return 99
    ema_items.sort(key=sort_key)
    return ema_items


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt = step_ckpt_path(args.method, args.rank, args.step)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    name = exp_name(args.method, args.rank)
    factor = args.factor or f'factors/{name}_step{int(args.step)}.pt'
    if not os.path.exists(factor):
        raise FileNotFoundError(
            f'missing factor {factor}; run closed_form_factorization first')

    out_dir = args.out_dir or os.path.join(
        'checkpoints', name, f'traverse_step{int(args.step)}')
    os.makedirs(out_dir, exist_ok=True)

    print(f'[traverse] method={args.method} rank={args.rank} step={args.step}')
    print(f'[traverse] ckpt={ckpt}')
    print(f'[traverse] factor={factor}')
    print(f'[traverse] out_dir={out_dir}')

    conf = build_conf(args.method, args.rank)
    model = LitModel(conf)
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=True)
    model.ema_model.eval()
    model.ema_model.to('cuda:0')

    ema_items = load_ema_factors(factor)
    print('[traverse] factor keys:', [k for k, _ in ema_items])

    data = FFHQlmdb(path=args.data_path, image_size=conf.img_size, split='test')
    loader = DataLoader(data, batch_size=args.n_sample, shuffle=False,
                        num_workers=4)
    batch = next(iter(loader))
    imgs = batch['img'][:args.n_sample].to('cuda:0')

    cond = model.encode(imgs)
    xT = model.encode_stochastic(imgs, cond=cond, T=args.T_encode)

    # baseline recon
    recon = model.render(xT, cond=cond, T=args.T_encode)
    save_image(make_grid(recon, nrow=args.n_sample),
               os.path.join(out_dir, 'baseline_recon.png'))
    save_image(make_grid((imgs + 1) / 2, nrow=args.n_sample),
               os.path.join(out_dir, 'baseline_real.png'))

    alphas = np.linspace(-args.degree, args.degree, args.n_steps)
    indices = args.indices
    if indices is None:
        # for full-rank, also sample sparse mid/high indices (often near-null)
        if args.rank >= 512:
            indices = [0, 1, 2, 3, 4, 5, 6, 7, 16, 32, 64, 128, 256, 511]
        else:
            indices = list(range(min(8, args.rank)))

    for layer_index in args.layers:
        if layer_index < 0 or layer_index >= len(ema_items):
            print(f'[traverse] skip invalid layer_index={layer_index}')
            continue
        key, eigvec = ema_items[layer_index]
        n_dir = eigvec.shape[1]
        print(f'[traverse] layer={layer_index} key={key} n_dir={n_dir}')

        for j in indices:
            if j < 0 or j >= n_dir:
                print(f'  skip index={j} (n_dir={n_dir})')
                continue
            direction = eigvec[:, j].unsqueeze(0).to('cuda:0')
            direction = direction / direction.norm()

            rows = []
            for a in alphas:
                # broadcast same scaled direction to the whole batch
                d = (float(a) * direction).expand(imgs.size(0), -1)
                edited = model.render(
                    xT, cond=cond, T=args.T,
                    layer_index=layer_index, direction=d)
                rows.append(edited.cpu())

            # rows = alphas, cols = faces  -> stack then rearrange
            # make_grid with nrow=n_sample puts consecutive imgs in a row
            grid_tensor = torch.cat(rows, dim=0)  # (n_steps * N, 3, H, W)
            path = os.path.join(
                out_dir,
                f'layer{layer_index}_eig{j:03d}_deg{args.degree:g}.png')
            save_image(make_grid(grid_tensor, nrow=args.n_sample, padding=2),
                       path)
            print(f'  saved {path}')

    print('[traverse] done')


if __name__ == '__main__':
    main()
