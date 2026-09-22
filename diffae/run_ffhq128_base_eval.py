#!/usr/bin/env python
"""Eval original DiffAE: checkpoints/ffhq128_autoenc_130M/last.ckpt

The current codebase defaults to multi_projector layers, but the original
ffhq128_autoenc_130M checkpoint has no style projectors / time_embed.style
params (style path was effectively Identity). This script patches the model
to match before loading weights.
"""
import argparse
import json
import os

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import loggers as pl_loggers

from templates import *


CKPT = 'checkpoints/ffhq128_autoenc_130M/last.ckpt'
EXP_NAME = 'ffhq128_autoenc_130M'


def build_conf():
    conf = ffhq128_autoenc_base()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.batch_size = 32
    conf.batch_size_eval = 64
    conf.name = EXP_NAME
    return conf


def patch_to_vanilla(lit_model):
    """Match original DiffAE architecture used by ffhq128_autoenc_130M."""
    for net in (lit_model.model, lit_model.ema_model):
        net.conf.use_mlp_multi = False
        net.conf.is_ortho = False
        net.conf.is_ortho_multi = False
        # original checkpoint has no time_embed.style.* params
        net.time_embed.style = nn.Identity()
    return lit_model


def load_vanilla(ckpt=CKPT):
    from experiment import LitModel
    conf = build_conf()
    model = LitModel(conf)
    model = patch_to_vanilla(model)
    state = torch.load(ckpt, map_location='cpu')
    missing, unexpected = model.load_state_dict(state['state_dict'], strict=False)
    print(f'loaded {ckpt}; step={state.get("global_step")} '
          f'missing={len(missing)} unexpected={len(unexpected)}')
    # only time_embed.style / unused projector params should be missing
    return conf, model, state


def eval_fid(ckpt=CKPT):
    from experiment import LitModel
    from dist_utils import get_rank

    conf = build_conf()
    conf.eval_programs = ['fid(10,10)']
    conf.eval_path = ckpt

    model = LitModel(conf)
    model = patch_to_vanilla(model)

    os.makedirs(conf.logdir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')
    trainer = pl.Trainer(
        max_steps=1,
        num_nodes=1,
        accelerator='auto',
        precision=16 if conf.fp16 else 32,
        logger=tb_logger,
        strategy='ddp',
    )

    print('loading from:', ckpt)
    state = torch.load(ckpt, map_location='cpu')
    print('step:', state.get('global_step'))
    model.load_state_dict(state['state_dict'], strict=False)

    dummy = DataLoader(TensorDataset(torch.tensor([0.] * conf.batch_size)),
                       batch_size=conf.batch_size)
    out = trainer.test(model, dataloaders=dummy)[0]
    print(out)

    if get_rank() == 0:
        tgt = f'evals/{EXP_NAME}.txt'
        os.makedirs('evals', exist_ok=True)
        with open(tgt, 'w') as f:
            payload = dict(out)
            payload['ckpt'] = ckpt
            payload['global_step'] = state.get('global_step')
            f.write(json.dumps(payload) + '\n')
        print('wrote', tgt)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['fid'], default='fid')
    p.add_argument('--ckpt', default=CKPT)
    args = p.parse_args()
    if args.mode == 'fid':
        eval_fid(args.ckpt)
