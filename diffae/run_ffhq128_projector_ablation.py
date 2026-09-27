#!/usr/bin/env python
"""Projector ablation (date-prefixed experiments).

Three arms (random projector init, cold-start from DiffAE backbone):
  1) Householder / SVD-style ortho projector, rank=10
  2) Householder / SVD-style ortho projector, rank=512
  3) LoRA-style low-rank projector, A:512x10, B:10x512

Save at 20k and 50k (+ step=0).

Usage:
  python run_ffhq128_projector_ablation.py --method householder --rank 10
  python run_ffhq128_projector_ablation.py --method householder --rank 512
  python run_ffhq128_projector_ablation.py --method lora --rank 10
  python run_ffhq128_projector_ablation.py --mode eval --method lora --rank 10 --step 20000
"""
import argparse
import glob
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback, LearningRateMonitor

from templates import *

DATE_TAG = '0927'  # experiment start date
SAVE_STEPS = (20_000, 50_000)
MAX_STEPS = 50_000
INIT_CKPT = 'checkpoints/ffhq128_autoenc_130M/last.ckpt'


def exp_name(method: str, rank: int) -> str:
    if method == 'householder':
        return f'{DATE_TAG}_ffhq128_autoenc_rank{int(rank)}_svd_random'
    if method == 'lora':
        return f'{DATE_TAG}_ffhq128_autoenc_lora{int(rank)}_random'
    raise ValueError(method)


class SaveAtStepsCallback(Callback):
    def __init__(self, steps, dirpath, tag='proj-ablation'):
        super().__init__()
        self.steps = set(int(s) for s in steps if int(s) > 0)
        self.dirpath = dirpath
        self.tag = tag
        self._saved = set()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step not in self.steps or step in self._saved:
            return
        if not trainer.is_global_zero:
            return
        os.makedirs(self.dirpath, exist_ok=True)
        path = os.path.join(self.dirpath, f'step={step}.ckpt')
        if os.path.exists(path):
            self._saved.add(step)
            return
        try:
            trainer.save_checkpoint(path)
            print(f'[{self.tag}] saved {path}', flush=True)
        except OSError as e:
            light = os.path.join(self.dirpath, f'step={step}_weights.ckpt')
            print(f'[{self.tag}] full save failed ({e}); trying weights-only -> {light}',
                  flush=True)
            payload = {
                'state_dict': pl_module.state_dict(),
                'global_step': step,
                'pytorch-lightning_version': pl.__version__,
            }
            torch.save(payload, light)
            print(f'[{self.tag}] saved {light}', flush=True)
        self._saved.add(step)


def build_conf(method: str, rank: int):
    conf = ffhq128_autoenc_base()
    conf.total_samples = 130_000_000
    conf.batch_size = 32
    conf.batch_size_eval = 64
    conf.model_conf.is_ortho = False
    conf.model_conf.use_mlp_multi = True
    conf.model_conf.diag_size = int(rank)
    if method == 'householder':
        conf.model_conf.is_ortho_multi = True
        conf.model_conf.use_low_rank_multi = False
    elif method == 'lora':
        conf.model_conf.is_ortho_multi = False
        conf.model_conf.use_low_rank_multi = True
    else:
        raise ValueError(method)
    conf.sample_every_samples = 10**12
    conf.eval_every_samples = 10**12
    conf.eval_ema_every_samples = 10**12
    conf.save_every_samples = 10**12
    conf.name = exp_name(method, rank)
    return conf


def find_latest_own_ckpt(logdir):
    step_ckpts = []
    for path in glob.glob(f'{logdir}/checkpoints/step=*.ckpt'):
        base = os.path.basename(path)
        if base == 'step=0.ckpt' or base.endswith('_weights.ckpt'):
            continue
        try:
            step = int(base[len('step='):-len('.ckpt')])
        except ValueError:
            continue
        step_ckpts.append((step, path))
    if step_ckpts:
        step_ckpts.sort(key=lambda x: x[0])
        return step_ckpts[-1][1]
    last = f'{logdir}/checkpoints/last.ckpt'
    if os.path.exists(last):
        return last
    if os.path.exists(f'{logdir}/last.ckpt'):
        return f'{logdir}/last.ckpt'
    return None


def save_step0(model, ckpt_dir, tag='proj-ablation'):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, 'step=0.ckpt')
    if os.path.exists(path):
        print(f'[{tag}] step=0 already exists: {path}')
        return path
    payload = {
        'state_dict': model.state_dict(),
        'global_step': 0,
        'pytorch-lightning_version': pl.__version__,
    }
    torch.save(payload, path)
    print(f'[{tag}] saved {path}')
    return path


def train_one(method: str, rank: int):
    from experiment import LitModel

    conf = build_conf(method, rank)
    tag = f'{method}-r{rank}'
    print('conf:', conf.name, 'method:', method, 'rank:', rank)
    print('[init] DiffAE backbone + RANDOM projector params (no SVD-identity init)')
    model = LitModel(conf)
    os.makedirs(conf.logdir, exist_ok=True)
    ckpt_dir = os.path.join(conf.logdir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # sanity print projector type
    pe = model.model.style_enc
    print(f'[{tag}] style_enc={type(pe).__name__}', flush=True)
    if hasattr(pe, 'A'):
        print(f'[{tag}] LoRA A{tuple(pe.A.shape)} B{tuple(pe.B.shape)}', flush=True)
    if hasattr(pe, 'U'):
        print(f'[{tag}] Householder U{tuple(pe.U.shape)} V{tuple(pe.V.shape)} '
              f'diag_size={conf.model_conf.diag_size}', flush=True)

    resume_ckpt = find_latest_own_ckpt(conf.logdir)
    if resume_ckpt is not None:
        print('resuming from:', resume_ckpt)
    else:
        print('cold-start init from:', INIT_CKPT)
        state = torch.load(INIT_CKPT, map_location='cpu')
        missing, unexpected = model.load_state_dict(state['state_dict'],
                                                    strict=False)
        print(f'loaded base ckpt; missing={len(missing)} unexpected={len(unexpected)}')
        # projectors stay at constructor random init (Householder U/V or LoRA A/B)
        save_step0(model, ckpt_dir, tag=tag)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')
    save_cb = SaveAtStepsCallback(SAVE_STEPS, ckpt_dir, tag=tag)
    trainer = pl.Trainer(
        max_steps=MAX_STEPS,
        num_nodes=1,
        accelerator='auto',
        precision=16 if conf.fp16 else 32,
        callbacks=[save_cb, LearningRateMonitor()],
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        strategy='ddp',
        enable_checkpointing=False,
    )
    trainer.fit(model, ckpt_path=resume_ckpt)


def step_ckpt_path(method: str, rank: int, step: int) -> str:
    return (f'checkpoints/{exp_name(method, rank)}/checkpoints/'
            f'step={int(step)}.ckpt')


def eval_step(method: str, rank: int, step: int, gpus):
    from experiment import train

    ckpt = step_ckpt_path(method, rank, step)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    conf = build_conf(method, rank)
    conf.name = f'{exp_name(method, rank)}_step{int(step)}'
    conf.eval_programs = ['fid(10,10)']
    conf.eval_path = ckpt
    conf.batch_size = 32
    conf.batch_size_eval = 64
    print(f'[eval] method={method} rank={rank} step={step} ckpt={ckpt}')
    train(conf, gpus=gpus, mode='eval')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['householder', 'lora'])
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--mode', choices=['train', 'eval'], default='train')
    p.add_argument('--step', type=int, default=None)
    p.add_argument('--gpus', type=int, nargs='+', default=[0])
    args = p.parse_args()
    if args.method == 'householder' and args.rank not in (10, 512):
        raise SystemExit('householder rank must be 10 or 512 for this ablation')
    if args.method == 'lora' and args.rank != 10:
        raise SystemExit('lora ablation uses rank=10 (A:512x10, B:10x512)')
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        train_one(args.method, args.rank)
    else:
        if args.step is None:
            raise SystemExit('--step is required for --mode eval')
        eval_step(args.method, args.rank, args.step, args.gpus)
