import argparse
import glob
import os

from templates import *
from templates_latent import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True, choices=[5, 20, 30])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    return parser.parse_args()


def find_latest_ckpt(logdir):
    candidates = []
    for pattern in (f'{logdir}/last.ckpt', f'{logdir}/checkpoints/*.ckpt'):
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


if __name__ == '__main__':
    args = parse_args()
    conf = ffhq128_autoenc_rank_ablation(diag_size=args.rank)

    if args.mode == 'eval':
        conf.eval_programs = ['fid(10,10)']
        ckpt = find_latest_ckpt(conf.logdir)
        if ckpt is None:
            raise FileNotFoundError(f'no checkpoint found under {conf.logdir}')
        conf.eval_path = ckpt

    train(conf, gpus=args.gpus, mode=args.mode)
