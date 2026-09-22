#!/usr/bin/env python
"""Finetuning-steps ablation for 0214_ffhq128_autoenc_130M_multi_projector.

Cold-start from base DiffAE (ffhq128_autoenc_130M/last.ckpt), then finetune
Householder multi-projectors and save at steps 0 / 10k / 25k / 50k / 100k.

Eval (FID):
  python run_ffhq128_steps_ablation.py --mode eval --step 10000
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
EXP_NAME = '0214_ffhq128_autoenc_130M_multi_projector'


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
            print(f'[steps-ablation] saved {path}', flush=True)
        except OSError as e:
            # Disk quota: fall back to a lighter weights-only dump (~1G vs ~2G).
            light = os.path.join(self.dirpath, f'step={step}_weights.ckpt')
            print(f'[steps-ablation] full save failed ({e}); trying weights-only -> {light}',
                  flush=True)
            payload = {
                'state_dict': pl_module.state_dict(),
                'global_step': step,
                'pytorch-lightning_version': pl.__version__,
            }
            torch.save(payload, light)
            print(f'[steps-ablation] saved {light}', flush=True)
        self._saved.add(step)


def build_conf():
    conf = ffhq128_autoenc_base()
    # match 0214 hparams.yaml
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
    conf.name = EXP_NAME
    return conf


def find_latest_own_ckpt(logdir):
    """Prefer the highest step=N.ckpt with optimizer state; fall back to last.ckpt."""
    step_ckpts = []
    for path in glob.glob(f'{logdir}/checkpoints/step=*.ckpt'):
        base = os.path.basename(path)
        if base == 'step=0.ckpt':
            continue  # lightweight dump, no optimizer
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


def save_step0(model, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, 'step=0.ckpt')
    if os.path.exists(path):
        print(f'[steps-ablation] step=0 already exists: {path}')
        return path
    # Lightning-compatible dict so later eval/resume can load it the same way
    payload = {
        'state_dict': model.state_dict(),
        'global_step': 0,
        'pytorch-lightning_version': pl.__version__,
    }
    torch.save(payload, path)
    print(f'[steps-ablation] saved {path}')
    return path


def train_steps_ablation(conf):
    from experiment import LitModel

    print('conf:', conf.name)
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
        # projectors (style_*) are newly initialized; persist as finetune step 0
        save_step0(model, ckpt_dir)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')
    # Only save at the ablation milestones (0/10k/25k/50k/100k).
    # Avoid writing last.ckpt every 5k — that previously hit disk quota.
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


def step_ckpt_path(step: int) -> str:
    return f'checkpoints/{EXP_NAME}/checkpoints/step={int(step)}.ckpt'


def eval_step(step: int, gpus):
    """Run FID eval for one saved step checkpoint via experiment.train(mode=eval)."""
    from experiment import train

    ckpt = step_ckpt_path(step)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    conf = build_conf()
    # unique name -> separate FID cache + evals/*.txt per step
    conf.name = f'{EXP_NAME}_step{int(step)}'
    conf.eval_programs = ['fid(10,10)']
    conf.eval_path = ckpt
    # restore reasonable eval batching (build_conf disables mid-train eval)
    conf.batch_size = 32
    conf.batch_size_eval = 64
    print(f'[steps-ablation] eval step={step} ckpt={ckpt}')
    train(conf, gpus=gpus, mode='eval')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train', 'eval'], default='train')
    p.add_argument('--step', type=int, default=None,
                   help='checkpoint step to eval (required for --mode eval)')
    p.add_argument('--gpus', type=int, nargs='+', default=[0])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        train_steps_ablation(build_conf())
    else:
        if args.step is None:
            raise SystemExit('--step is required for --mode eval')
        eval_step(args.step, gpus=args.gpus)
