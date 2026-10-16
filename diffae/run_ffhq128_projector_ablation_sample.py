#!/usr/bin/env python
"""Generate reconstruction sample grids for 0927 projector ablations.

Usage:
  python run_ffhq128_projector_ablation_sample.py \
    --method householder --rank 512 --step 50000 --n_sample 64 --T 100
"""
import argparse
import os

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
    p.add_argument('--n_sample', type=int, default=64)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--T', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data_path', type=str, default='datasets/ffhq256.lmdb')
    p.add_argument('--out_dir', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    ckpt = step_ckpt_path(args.method, args.rank, args.step)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    conf = build_conf(args.method, args.rank)
    name = exp_name(args.method, args.rank)
    out_dir = args.out_dir or os.path.join(
        'checkpoints', name, f'sample_ema_step{int(args.step)}')
    os.makedirs(out_dir, exist_ok=True)

    print(f'[sample] method={args.method} rank={args.rank} step={args.step}')
    print(f'[sample] ckpt={ckpt}')
    print(f'[sample] out_dir={out_dir} n={args.n_sample} T={args.T}')

    model = LitModel(conf)
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=True)
    model.ema_model.eval()
    model.ema_model.to('cuda:0')

    data = FFHQlmdb(path=args.data_path, image_size=conf.img_size, split='test')
    loader = DataLoader(data,
                        batch_size=args.batch,
                        shuffle=False,
                        num_workers=4,
                        drop_last=False)

    reals = []
    recons = []
    n_need = args.n_sample
    for batch in loader:
        imgs = batch['img'].to('cuda:0')
        take = min(n_need - len(reals), imgs.size(0))
        imgs = imgs[:take]
        cond = model.encode(imgs)
        xT = model.encode_stochastic(imgs, cond=cond, T=args.T)
        recon = model.render(xT, cond=cond, T=args.T)
        # real in [-1,1] -> [0,1]; render already returns [0,1]
        reals.append((imgs + 1) / 2)
        recons.append(recon)
        if len(torch.cat(reals)) >= n_need:
            break

    real = torch.cat(reals)[:n_need].cpu()
    recon = torch.cat(recons)[:n_need].cpu()
    paired = torch.stack([real, recon], dim=1).flatten(0, 1)

    nrow = 8 if n_need >= 8 else n_need
    save_image(make_grid(real, nrow=nrow),
               os.path.join(out_dir, 'real.png'))
    save_image(make_grid(recon, nrow=nrow),
               os.path.join(out_dir, 'recon.png'))
    # rows: real / recon pairs
    save_image(make_grid(paired, nrow=2),
               os.path.join(out_dir, 'real_recon_pairs.png'))
    print(f'[sample] saved grids under {out_dir}')


if __name__ == '__main__':
    main()
