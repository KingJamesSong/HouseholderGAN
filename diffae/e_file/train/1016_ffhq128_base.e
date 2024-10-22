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
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torch/cuda/__init__.py:104: UserWarning: 
NVIDIA RTX A6000 with CUDA capability sm_86 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
If you want to use the NVIDIA RTX A6000 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

  warnings.warn(incompatible_device_warn.format(device_name, capability, " ".join(arch_list), device_name))
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
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
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/trainer/data_loading.py:105: UserWarning: The dataloader, train dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 128 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/callbacks/lr_monitor.py:112: RuntimeWarning: You are using `LearningRateMonitor` callback with models that have no learning rate schedulers. Please see documentation for `configure_optimizers` method.
  rank_zero_warn(
Traceback (most recent call last):
  File "/nfs/data_chaos/czhang/HouseholderGAN/diffae/run_ffhq128.py", line 9, in <module>
    train(conf, gpus=gpus)
  File "/nfs/data_chaos/czhang/HouseholderGAN/diffae/experiment.py", line 941, in train
    trainer.fit(model)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 552, in fit
    self._run(model)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 917, in _run
    self._dispatch()
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 985, in _dispatch
    self.accelerator.start_training(self)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/accelerators/accelerator.py", line 92, in start_training
    self.training_type_plugin.start_training(trainer)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/plugins/training_type/training_type_plugin.py", line 161, in start_training
    self._results = trainer.run_stage()
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 995, in run_stage
    return self._run_train()
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1044, in _run_train
    self.fit_loop.run()
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/loops/base.py", line 111, in run
    self.advance(*args, **kwargs)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/loops/fit_loop.py", line 200, in advance
    epoch_output = self.epoch_loop.run(train_dataloader)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/loops/base.py", line 111, in run
    self.advance(*args, **kwargs)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/loops/epoch/training_epoch_loop.py", line 130, in advance
    batch_output = self.batch_loop.run(batch, self.iteration_count, self._dataloader_idx)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/loops/batch/training_batch_loop.py", line 100, in run
    super().run(batch, batch_idx, dataloader_idx)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/loops/base.py", line 111, in run
    self.advance(*args, **kwargs)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/loops/batch/training_batch_loop.py", line 147, in advance
    result = self._run_optimization(batch_idx, split_batch, opt_idx, optimizer)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/loops/batch/training_batch_loop.py", line 201, in _run_optimization
    self._optimizer_step(optimizer, opt_idx, batch_idx, closure)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/loops/batch/training_batch_loop.py", line 395, in _optimizer_step
    model_ref.optimizer_step(
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/core/lightning.py", line 1618, in optimizer_step
    optimizer.step(closure=optimizer_closure)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/core/optimizer.py", line 209, in step
    self.__optimizer_step(*args, closure=closure, profiler_name=profiler_name, **kwargs)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/core/optimizer.py", line 129, in __optimizer_step
    trainer.accelerator.optimizer_step(optimizer, self._optimizer_idx, lambda_closure=closure, **kwargs)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/accelerators/accelerator.py", line 292, in optimizer_step
    make_optimizer_step = self.precision_plugin.pre_optimizer_step(
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/plugins/precision/native_amp.py", line 59, in pre_optimizer_step
    result = lambda_closure()
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/loops/batch/training_batch_loop.py", line 235, in _training_step_and_backward_closure
    result = self.training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, hiddens)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/loops/batch/training_batch_loop.py", line 536, in training_step_and_backward
    result = self._training_step(split_batch, batch_idx, opt_idx, hiddens)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/loops/batch/training_batch_loop.py", line 306, in _training_step
    training_step_output = self.trainer.accelerator.training_step(step_kwargs)
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/accelerators/accelerator.py", line 193, in training_step
    return self.training_type_plugin.training_step(*step_kwargs.values())
  File "/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/plugins/training_type/training_type_plugin.py", line 172, in training_step
    return self.model.training_step(*args, **kwargs)
  File "/nfs/data_chaos/czhang/HouseholderGAN/diffae/experiment.py", line 377, in training_step
    losses = self.sampler.training_losses(model=self.model,
  File "/nfs/data_chaos/czhang/HouseholderGAN/diffae/diffusion/diffusion.py", line 100, in training_losses
    return super().training_losses(self._wrap_model(model), *args,
  File "/nfs/data_chaos/czhang/HouseholderGAN/diffae/diffusion/base.py", line 121, in training_losses
    noise = th.randn_like(x_start)
RuntimeError: CUDA error: no kernel image is available for execution on the device
