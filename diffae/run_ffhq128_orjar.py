#!/usr/bin/env python
"""Train 0125_ffhq128_autoenc_130M_multi_mlp_OrJaR from saved hparams settings."""
import glob
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from templates import *


def build_conf():
    conf = ffhq128_autoenc_base()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.sample_every_samples = 80_000
    conf.save_every_samples = 100_000
    conf.batch_size = 32
    conf.batch_size_eval = 64
    conf.model_conf.is_ortho = False
    conf.model_conf.is_ortho_multi = False
    conf.model_conf.use_mlp_multi = True
    conf.name = '0125_ffhq128_autoenc_130M_multi_mlp_OrJaR'
    return conf


def enable_orjar(conf):
    orig_make = conf._make_diffusion_conf

    def _make(T=None):
        dconf = orig_make(T)
        dconf.use_ortho_jacob = True
        dconf.ortho_weight = 0.2
        dconf.use_hessian_penalty = False
        return dconf

    conf._make_diffusion_conf = _make
    return conf


def train_orjar(conf):
    from experiment import LitModel
    print('conf:', conf.name)
    conf = enable_orjar(conf)
    model = LitModel(conf)

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)

    resume_ckpt = None
    own = []
    for pattern in (f'{conf.logdir}/last.ckpt', f'{conf.logdir}/checkpoints/*.ckpt'):
        own.extend(glob.glob(pattern))
    if own:
        resume_ckpt = max(own, key=os.path.getmtime)
        print('resuming from:', resume_ckpt)
    else:
        init_ckpt = 'checkpoints/ffhq128_autoenc_130M/last.ckpt'
        print('cold-start init from:', init_ckpt)
        state = torch.load(init_ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'], strict=False)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir, name=None, version='')
    ckpt_cb = ModelCheckpoint(
        dirpath=f'{conf.logdir}/checkpoints',
        save_last=True,
        save_top_k=1,
        every_n_train_steps=max(1, conf.save_every_samples // conf.batch_size_effective),
    )
    trainer = pl.Trainer(
        max_steps=conf.total_samples // conf.batch_size_effective,
        num_nodes=1,
        accelerator='auto',
        precision=16 if conf.fp16 else 32,
        callbacks=[ckpt_cb, LearningRateMonitor()],
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        strategy='ddp',
    )
    trainer.fit(model, ckpt_path=resume_ckpt)


if __name__ == '__main__':
    train_orjar(build_conf())
