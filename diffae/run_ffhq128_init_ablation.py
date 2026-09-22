#!/usr/bin/env python
"""Initialization ablation for Householder multi-projector finetuning.

Same recipe as 0214 steps ablation (base DiffAE cold-start, diag_size=10,
save at 0/10k/25k/50k/100k), but projector init is selectable:

  random       — U,V ~ N(0, 0.05)  (same as 0214; already have those ckpts)
  svd_identity — projection_layer.intialize(I) via SVD + geqrf

Usage:
  python run_ffhq128_init_ablation.py --init svd_identity
  python run_ffhq128_init_ablation.py --mode eval --init svd_identity --step 10000
"""
import argparse
import glob
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback, LearningRateMonitor

from templates import *

SAVE_STEPS = (0, 10_000, 25_000, 50_000, 100_000)
MAX_STEPS = 100_000
INIT_CKPT = 'checkpoints/ffhq128_autoenc_130M/last.ckpt'
STYLE_NAMES = ('style_enc', 'style_mid', 'style_dec')
VALID_INITS = ('random', 'svd_identity')


def exp_name(init_mode: str) -> str:
    return f'0222_ffhq128_autoenc_130M_multi_projector_init_{init_mode}'


class SaveAtStepsCallback(Callback):
    """Save a checkpoint exactly when trainer.global_step hits one of SAVE_STEPS."""

    def __init__(self, steps, dirpath):
        super().__init__()
        self.steps = set(int(s) for s in steps if int(s) > 0)
        self.dirpath = dirpath
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
            print(f'[init-ablation] saved {path}', flush=True)
        except OSError as e:
            light = os.path.join(self.dirpath, f'step={step}_weights.ckpt')
            print(f'[init-ablation] full save failed ({e}); trying weights-only -> {light}',
                  flush=True)
            payload = {
                'state_dict': pl_module.state_dict(),
                'global_step': step,
                'pytorch-lightning_version': pl.__version__,
            }
            torch.save(payload, light)
            print(f'[init-ablation] saved {light}', flush=True)
        self._saved.add(step)


def build_conf(init_mode: str):
    conf = ffhq128_autoenc_base()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.sample_every_samples = 80_000
    conf.save_every_samples = 100_000
    conf.batch_size = 32
    conf.batch_size_eval = 64
    conf.model_conf.is_ortho = False
    conf.model_conf.is_ortho_multi = True
    conf.model_conf.use_mlp_multi = True
    conf.model_conf.diag_size = 10
    # keep mid-run sampling/eval rare so 100k steps fit in one day
    conf.sample_every_samples = 10**12
    conf.eval_every_samples = 10**12
    conf.eval_ema_every_samples = 10**12
    conf.name = exp_name(init_mode)
    return conf


def find_latest_own_ckpt(logdir):
    step_ckpts = []
    for path in glob.glob(f'{logdir}/checkpoints/step=*.ckpt'):
        base = os.path.basename(path)
        if base == 'step=0.ckpt':
            continue
        if base.endswith('_weights.ckpt'):
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


def apply_projector_init(model, init_mode: str):
    """Apply Householder projector init on model + ema_model style_* layers."""
    if init_mode == 'random':
        print('[init-ablation] projector init = random N(0, 0.05) (constructor default)')
        return
    if init_mode != 'svd_identity':
        raise ValueError(f'unknown init_mode={init_mode}')

    eye = torch.eye(512)
    n = 0
    for net_name in ('model', 'ema_model'):
        net = getattr(model, net_name)
        for style_name in STYLE_NAMES:
            layer = getattr(net, style_name, None)
            if layer is None or not hasattr(layer, 'intialize'):
                raise RuntimeError(f'missing projector {net_name}.{style_name}')
            layer.intialize(eye.clone())
            n += 1
    print(f'[init-ablation] projector init = svd_identity on {n} layers '
          f'({STYLE_NAMES})')


def save_step0(model, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, 'step=0.ckpt')
    if os.path.exists(path):
        print(f'[init-ablation] step=0 already exists: {path}')
        return path
    payload = {
        'state_dict': model.state_dict(),
        'global_step': 0,
        'pytorch-lightning_version': pl.__version__,
    }
    torch.save(payload, path)
    print(f'[init-ablation] saved {path}')
    return path


def train_init_ablation(init_mode: str):
    from experiment import LitModel

    conf = build_conf(init_mode)
    print('conf:', conf.name, 'init:', init_mode)
    model = LitModel(conf)
    os.makedirs(conf.logdir, exist_ok=True)
    ckpt_dir = os.path.join(conf.logdir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    resume_ckpt = find_latest_own_ckpt(conf.logdir)
    if resume_ckpt is not None:
        print('resuming from:', resume_ckpt)
    else:
        print('cold-start init from:', INIT_CKPT)
        state = torch.load(INIT_CKPT, map_location='cpu')
        missing, unexpected = model.load_state_dict(state['state_dict'],
                                                    strict=False)
        print(f'loaded base ckpt; missing={len(missing)} unexpected={len(unexpected)}')
        apply_projector_init(model, init_mode)
        save_step0(model, ckpt_dir)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')
    save_cb = SaveAtStepsCallback(SAVE_STEPS, ckpt_dir)
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


def step_ckpt_path(init_mode: str, step: int) -> str:
    return (f'checkpoints/{exp_name(init_mode)}/checkpoints/'
            f'step={int(step)}.ckpt')


def eval_step(init_mode: str, step: int, gpus):
    from experiment import train

    ckpt = step_ckpt_path(init_mode, step)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    conf = build_conf(init_mode)
    conf.name = f'{exp_name(init_mode)}_step{int(step)}'
    conf.eval_programs = ['fid(10,10)']
    conf.eval_path = ckpt
    conf.batch_size = 32
    conf.batch_size_eval = 64
    print(f'[init-ablation] eval init={init_mode} step={step} ckpt={ckpt}')
    train(conf, gpus=gpus, mode='eval')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--init', choices=VALID_INITS, required=True)
    p.add_argument('--mode', choices=['train', 'eval'], default='train')
    p.add_argument('--step', type=int, default=None,
                   help='checkpoint step to eval (required for --mode eval)')
    p.add_argument('--gpus', type=int, nargs='+', default=[0])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        train_init_ablation(args.init)
    else:
        if args.step is None:
            raise SystemExit('--step is required for --mode eval')
        eval_step(args.init, args.step, args.gpus)
