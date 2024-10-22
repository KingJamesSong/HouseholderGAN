wandb: Appending key for api.wandb.ai to your netrc file: /home/chenyu.zhang/.netrc
Global seed set to 0
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:446: UserWarning: Checkpoint directory checkpoints/ffhq128_autoenc_130M exists and is not empty.
  rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:432: UserWarning: ModelCheckpoint(save_last=True, save_top_k=None, monitor=None) is a redundant configuration. You can save the last checkpoint with ModelCheckpoint(save_top_k=None, monitor=None).
  rank_zero_warn(
Using native 16bit precision.
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
Restoring states from the checkpoint file at checkpoints/ffhq128_autoenc_130M/last.ckpt
Global seed set to 0
Global seed set to 0
initializing ddp: GLOBAL_RANK: 0, MEMBER: 1/2
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:446: UserWarning: Checkpoint directory checkpoints/ffhq128_autoenc_130M exists and is not empty.
  rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:432: UserWarning: ModelCheckpoint(save_last=True, save_top_k=None, monitor=None) is a redundant configuration. You can save the last checkpoint with ModelCheckpoint(save_top_k=None, monitor=None).
  rank_zero_warn(
Using native 16bit precision.
Global seed set to 0
initializing ddp: GLOBAL_RANK: 1, MEMBER: 2/2
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All DDP processes registered. Starting ddp with 2 processes
----------------------------------------------------------------------------------------------------

LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1]
Set SLURM handle signals.
Set SLURM handle signals.
Restored all states from the checkpoint file at checkpoints/ffhq128_autoenc_130M/last.ckpt

  | Name      | Type                 | Params
---------------------------------------------------
0 | model     | BeatGANsAutoencModel | 128 M 
1 | ema_model | BeatGANsAutoencModel | 128 M 
---------------------------------------------------
128 M     Trainable params
128 M     Non-trainable params
257 M     Total params
1,028.361 Total estimated model params size (MB)
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/trainer/data_loading.py:105: UserWarning: The dataloader, train dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 64 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/callbacks/lr_monitor.py:112: RuntimeWarning: You are using `LearningRateMonitor` callback with models that have no learning rate schedulers. Please see documentation for `configure_optimizers` method.
  rank_zero_warn(
slurmstepd: error: *** JOB 22972 ON dsml-chaos.disi.unitn.it CANCELLED AT 2024-10-20T22:22:20 DUE TO TIME LIMIT ***
