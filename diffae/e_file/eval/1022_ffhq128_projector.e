wandb: Appending key for api.wandb.ai to your netrc file: /home/chenyu.zhang/.netrc
[rank: 0] Seed set to 0
/nfs/data_chaos/czhang/HouseholderGAN/diffae/experiment.py:71: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state = torch.load(conf.pretrain.path, map_location='cpu')
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/lightning_fabric/connector.py:571: `precision=16` is supported for historical reasons but its usage is discouraged. Please set your precision to 16-mixed instead!
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/lightning_fabric/plugins/environments/slurm.py:204: The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: srun python run_ffhq128.py ...
Using 16bit Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
/nfs/data_chaos/czhang/HouseholderGAN/diffae/experiment.py:963: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state = torch.load(eval_path, map_location='cpu')
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2
[rank: 1] Seed set to 0
/nfs/data_chaos/czhang/HouseholderGAN/diffae/experiment.py:71: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state = torch.load(conf.pretrain.path, map_location='cpu')
/nfs/data_chaos/czhang/HouseholderGAN/diffae/experiment.py:963: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state = torch.load(eval_path, map_location='cpu')
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/2
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 2 processes
----------------------------------------------------------------------------------------------------

LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1]
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:215: Using `DistributedSampler` with the dataloaders. During `trainer.test()`, it is recommended to use `Trainer(devices=1, num_nodes=1)` to ensure each sample/batch gets evaluated exactly once. Otherwise, multi-device settings use `DistributedSampler` that replicates some samples to make sure all devices have same batch size in case of uneven inputs.
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:424: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=3` in the `DataLoader` to improve performance.

copy images:   0%|          | 0/781 [00:00<?, ?it/s][A/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

copy images:   0%|          | 1/781 [00:01<21:21,  1.64s/it][A
copy images:   0%|          | 2/781 [00:02<12:08,  1.07it/s][A
copy images:   0%|          | 3/781 [00:02<09:08,  1.42it/s][A
copy images:   1%|          | 4/781 [00:02<07:24,  1.75it/s][A
copy images:   1%|          | 5/781 [00:03<06:31,  1.98it/s][A
copy images:   1%|          | 6/781 [00:03<06:08,  2.10it/s][A
copy images:   1%|          | 7/781 [00:04<05:36,  2.30it/s][A
copy images:   1%|          | 8/781 [00:04<05:14,  2.46it/s][A
copy images:   1%|          | 9/781 [00:04<05:12,  2.47it/s][A
copy images:   1%|▏         | 10/781 [00:05<05:10,  2.49it/s][A
copy images:   1%|▏         | 11/781 [00:05<04:58,  2.58it/s][A
copy images:   2%|▏         | 12/781 [00:05<04:54,  2.61it/s][A
copy images:   2%|▏         | 13/781 [00:06<04:39,  2.74it/s][A
copy images:   2%|▏         | 14/781 [00:06<04:39,  2.74it/s][A
copy images:   2%|▏         | 15/781 [00:06<04:39,  2.74it/s][A
copy images:   2%|▏         | 16/781 [00:07<04:38,  2.74it/s][A
copy images:   2%|▏         | 17/781 [00:07<04:41,  2.71it/s][A
copy images:   2%|▏         | 18/781 [00:08<04:46,  2.67it/s][A
copy images:   2%|▏         | 19/781 [00:08<04:33,  2.79it/s][A
copy images:   3%|▎         | 20/781 [00:08<04:37,  2.75it/s][A
copy images:   3%|▎         | 21/781 [00:09<04:31,  2.80it/s][A
copy images:   3%|▎         | 22/781 [00:09<04:32,  2.79it/s][A
copy images:   3%|▎         | 23/781 [00:09<04:47,  2.64it/s][A
copy images:   3%|▎         | 24/781 [00:10<04:55,  2.56it/s][A
copy images:   3%|▎         | 25/781 [00:10<05:01,  2.50it/s][A
copy images:   3%|▎         | 26/781 [00:11<05:06,  2.47it/s][A
copy images:   3%|▎         | 27/781 [00:11<05:05,  2.47it/s][A
copy images:   4%|▎         | 28/781 [00:11<04:52,  2.58it/s][A
copy images:   4%|▎         | 29/781 [00:12<04:43,  2.65it/s][A
copy images:   4%|▍         | 30/781 [00:12<04:43,  2.65it/s][A
copy images:   4%|▍         | 31/781 [00:13<04:53,  2.56it/s][A
copy images:   4%|▍         | 32/781 [00:13<04:59,  2.50it/s][A
copy images:   4%|▍         | 33/781 [00:13<05:06,  2.44it/s][A
copy images:   4%|▍         | 34/781 [00:14<05:07,  2.43it/s][A
copy images:   4%|▍         | 35/781 [00:14<05:07,  2.42it/s][A
copy images:   5%|▍         | 36/781 [00:15<05:11,  2.40it/s][A
copy images:   5%|▍         | 37/781 [00:15<05:10,  2.40it/s][A
copy images:   5%|▍         | 38/781 [00:16<05:06,  2.42it/s][A
copy images:   5%|▍         | 39/781 [00:16<05:09,  2.40it/s][A
copy images:   5%|▌         | 40/781 [00:16<05:10,  2.39it/s][A
copy images:   5%|▌         | 41/781 [00:17<05:12,  2.37it/s][A
copy images:   5%|▌         | 42/781 [00:17<05:11,  2.37it/s][A
copy images:   6%|▌         | 43/781 [00:18<05:08,  2.39it/s][A
copy images:   6%|▌         | 44/781 [00:18<04:58,  2.47it/s][A
copy images:   6%|▌         | 45/781 [00:18<04:48,  2.55it/s][A
copy images:   6%|▌         | 46/781 [00:19<04:45,  2.58it/s][A
copy images:   6%|▌         | 47/781 [00:19<04:27,  2.74it/s][A
copy images:   6%|▌         | 48/781 [00:19<04:17,  2.85it/s][A
copy images:   6%|▋         | 49/781 [00:20<04:08,  2.94it/s][A
copy images:   6%|▋         | 50/781 [00:20<04:08,  2.94it/s][A
copy images:   7%|▋         | 51/781 [00:20<04:17,  2.84it/s][A
copy images:   7%|▋         | 52/781 [00:21<04:08,  2.93it/s][A
copy images:   7%|▋         | 53/781 [00:21<04:21,  2.78it/s][A
copy images:   7%|▋         | 54/781 [00:21<04:20,  2.79it/s][A
copy images:   7%|▋         | 55/781 [00:22<04:15,  2.84it/s][A
copy images:   7%|▋         | 56/781 [00:22<04:10,  2.90it/s][A
copy images:   7%|▋         | 57/781 [00:22<04:07,  2.92it/s][A
copy images:   7%|▋         | 58/781 [00:23<04:02,  2.98it/s][A
copy images:   8%|▊         | 59/781 [00:23<04:12,  2.86it/s][A
copy images:   8%|▊         | 60/781 [00:24<04:05,  2.94it/s][A
copy images:   8%|▊         | 61/781 [00:24<04:13,  2.84it/s][A
copy images:   8%|▊         | 62/781 [00:24<04:07,  2.90it/s][A
copy images:   8%|▊         | 63/781 [00:25<04:02,  2.96it/s][A
copy images:   8%|▊         | 64/781 [00:25<03:59,  2.99it/s][A
copy images:   8%|▊         | 65/781 [00:25<03:59,  2.99it/s][A
copy images:   8%|▊         | 66/781 [00:26<03:56,  3.02it/s][A
copy images:   9%|▊         | 67/781 [00:26<04:07,  2.89it/s][A
copy images:   9%|▊         | 68/781 [00:26<04:00,  2.97it/s][A
copy images:   9%|▉         | 69/781 [00:27<04:09,  2.85it/s][A
copy images:   9%|▉         | 70/781 [00:27<04:01,  2.94it/s][A
copy images:   9%|▉         | 71/781 [00:27<03:58,  2.98it/s][A
copy images:   9%|▉         | 72/781 [00:28<03:54,  3.02it/s][A
copy images:   9%|▉         | 73/781 [00:28<03:50,  3.07it/s][A
copy images:   9%|▉         | 74/781 [00:28<03:48,  3.09it/s][A
copy images:  10%|▉         | 75/781 [00:29<04:01,  2.93it/s][A
copy images:  10%|▉         | 76/781 [00:29<03:57,  2.97it/s][A
copy images:  10%|▉         | 77/781 [00:29<04:06,  2.86it/s][A
copy images:  10%|▉         | 78/781 [00:30<03:59,  2.93it/s][A
copy images:  10%|█         | 79/781 [00:30<03:54,  2.99it/s][A
copy images:  10%|█         | 80/781 [00:30<03:52,  3.01it/s][A
copy images:  10%|█         | 81/781 [00:31<03:52,  3.02it/s][A
copy images:  10%|█         | 82/781 [00:31<03:49,  3.05it/s][A
copy images:  11%|█         | 83/781 [00:31<04:00,  2.90it/s][A
copy images:  11%|█         | 84/781 [00:32<03:55,  2.95it/s][A
copy images:  11%|█         | 85/781 [00:32<04:04,  2.84it/s][A
copy images:  11%|█         | 86/781 [00:32<03:57,  2.92it/s][A
copy images:  11%|█         | 87/781 [00:33<03:54,  2.96it/s][A
copy images:  11%|█▏        | 88/781 [00:33<03:50,  3.00it/s][A
copy images:  11%|█▏        | 89/781 [00:33<03:47,  3.04it/s][A
copy images:  12%|█▏        | 90/781 [00:34<03:48,  3.03it/s][A
copy images:  12%|█▏        | 91/781 [00:34<03:58,  2.90it/s][A
copy images:  12%|█▏        | 92/781 [00:34<03:53,  2.95it/s][A
copy images:  12%|█▏        | 93/781 [00:35<04:00,  2.86it/s][A
copy images:  12%|█▏        | 94/781 [00:35<03:52,  2.96it/s][A
copy images:  12%|█▏        | 95/781 [00:35<03:48,  3.01it/s][A
copy images:  12%|█▏        | 96/781 [00:36<03:44,  3.06it/s][A
copy images:  12%|█▏        | 97/781 [00:36<03:41,  3.09it/s][A
copy images:  13%|█▎        | 98/781 [00:36<03:40,  3.09it/s][A
copy images:  13%|█▎        | 99/781 [00:37<03:50,  2.95it/s][A
copy images:  13%|█▎        | 100/781 [00:37<03:45,  3.03it/s][A
copy images:  13%|█▎        | 101/781 [00:37<03:53,  2.91it/s][A
copy images:  13%|█▎        | 102/781 [00:38<03:47,  2.99it/s][A
copy images:  13%|█▎        | 103/781 [00:38<03:44,  3.02it/s][A
copy images:  13%|█▎        | 104/781 [00:38<03:42,  3.05it/s][A
copy images:  13%|█▎        | 105/781 [00:39<03:38,  3.09it/s][A
copy images:  14%|█▎        | 106/781 [00:39<03:37,  3.10it/s][A
copy images:  14%|█▎        | 107/781 [00:39<03:47,  2.96it/s][A
copy images:  14%|█▍        | 108/781 [00:40<03:42,  3.03it/s][A
copy images:  14%|█▍        | 109/781 [00:40<03:51,  2.91it/s][A
copy images:  14%|█▍        | 110/781 [00:40<03:45,  2.98it/s][A
copy images:  14%|█▍        | 111/781 [00:41<03:42,  3.01it/s][A
copy images:  14%|█▍        | 112/781 [00:41<03:41,  3.02it/s][A
copy images:  14%|█▍        | 113/781 [00:41<03:38,  3.06it/s][A
copy images:  15%|█▍        | 114/781 [00:42<03:38,  3.06it/s][A
copy images:  15%|█▍        | 115/781 [00:42<03:47,  2.92it/s][A
copy images:  15%|█▍        | 116/781 [00:42<03:42,  2.99it/s][A
copy images:  15%|█▍        | 117/781 [00:43<03:47,  2.92it/s][A
copy images:  15%|█▌        | 118/781 [00:43<03:40,  3.01it/s][A
copy images:  15%|█▌        | 119/781 [00:43<03:35,  3.07it/s][A
copy images:  15%|█▌        | 120/781 [00:44<03:38,  3.02it/s][A
copy images:  15%|█▌        | 121/781 [00:44<03:41,  2.98it/s][A
copy images:  16%|█▌        | 122/781 [00:44<03:44,  2.94it/s][A
copy images:  16%|█▌        | 123/781 [00:45<03:48,  2.88it/s][A
copy images:  16%|█▌        | 124/781 [00:45<03:39,  2.99it/s][A
copy images:  16%|█▌        | 125/781 [00:45<03:44,  2.92it/s][A
copy images:  16%|█▌        | 126/781 [00:46<03:39,  2.98it/s][A
copy images:  16%|█▋        | 127/781 [00:46<03:34,  3.05it/s][A
copy images:  16%|█▋        | 128/781 [00:46<03:31,  3.09it/s][A
copy images:  17%|█▋        | 129/781 [00:47<03:29,  3.11it/s][A
copy images:  17%|█▋        | 130/781 [00:47<03:27,  3.14it/s][A
copy images:  17%|█▋        | 131/781 [00:47<03:36,  3.00it/s][A
copy images:  17%|█▋        | 132/781 [00:48<03:34,  3.03it/s][A
copy images:  17%|█▋        | 133/781 [00:48<03:43,  2.89it/s][A
copy images:  17%|█▋        | 134/781 [00:48<03:45,  2.87it/s][A
copy images:  17%|█▋        | 135/781 [00:49<03:44,  2.88it/s][A
copy images:  17%|█▋        | 136/781 [00:49<03:43,  2.89it/s][A
copy images:  18%|█▊        | 137/781 [00:49<03:45,  2.85it/s][A
copy images:  18%|█▊        | 138/781 [00:50<03:45,  2.85it/s][A
copy images:  18%|█▊        | 139/781 [00:50<03:46,  2.83it/s][A
copy images:  18%|█▊        | 140/781 [00:50<03:44,  2.85it/s][A
copy images:  18%|█▊        | 141/781 [00:51<03:47,  2.81it/s][A
copy images:  18%|█▊        | 142/781 [00:51<03:45,  2.83it/s][A
copy images:  18%|█▊        | 143/781 [00:51<03:43,  2.86it/s][A
copy images:  18%|█▊        | 144/781 [00:52<03:42,  2.86it/s][A
copy images:  19%|█▊        | 145/781 [00:52<03:43,  2.85it/s][A
copy images:  19%|█▊        | 146/781 [00:53<03:42,  2.86it/s][A
copy images:  19%|█▉        | 147/781 [00:53<03:40,  2.87it/s][A
copy images:  19%|█▉        | 148/781 [00:53<03:41,  2.86it/s][A
copy images:  19%|█▉        | 149/781 [00:54<03:39,  2.87it/s][A
copy images:  19%|█▉        | 150/781 [00:54<03:41,  2.85it/s][A
copy images:  19%|█▉        | 151/781 [00:54<03:38,  2.88it/s][A
copy images:  19%|█▉        | 152/781 [00:55<03:39,  2.86it/s][A
copy images:  20%|█▉        | 153/781 [00:55<03:38,  2.88it/s][A
copy images:  20%|█▉        | 154/781 [00:55<03:36,  2.89it/s][A
copy images:  20%|█▉        | 155/781 [00:56<03:37,  2.88it/s][A
copy images:  20%|█▉        | 156/781 [00:56<03:34,  2.91it/s][A
copy images:  20%|██        | 157/781 [00:56<03:34,  2.91it/s][A
copy images:  20%|██        | 158/781 [00:57<03:33,  2.91it/s][A
copy images:  20%|██        | 159/781 [00:57<03:33,  2.91it/s][A
copy images:  20%|██        | 160/781 [00:57<03:32,  2.92it/s][A
copy images:  21%|██        | 161/781 [00:58<03:34,  2.89it/s][A
copy images:  21%|██        | 162/781 [00:58<03:34,  2.88it/s][A
copy images:  21%|██        | 163/781 [00:58<03:36,  2.85it/s][A
copy images:  21%|██        | 164/781 [00:59<03:34,  2.87it/s][A
copy images:  21%|██        | 165/781 [00:59<03:34,  2.88it/s][A
copy images:  21%|██▏       | 166/781 [00:59<03:33,  2.88it/s][A
copy images:  21%|██▏       | 167/781 [01:00<03:34,  2.87it/s][A
copy images:  22%|██▏       | 168/781 [01:00<03:33,  2.87it/s][A
copy images:  22%|██▏       | 169/781 [01:01<03:32,  2.88it/s][A
copy images:  22%|██▏       | 170/781 [01:01<03:31,  2.89it/s][A
copy images:  22%|██▏       | 171/781 [01:01<03:32,  2.87it/s][A
copy images:  22%|██▏       | 172/781 [01:02<03:30,  2.89it/s][A
copy images:  22%|██▏       | 173/781 [01:02<03:31,  2.87it/s][A
copy images:  22%|██▏       | 174/781 [01:02<03:31,  2.87it/s][A
copy images:  22%|██▏       | 175/781 [01:03<03:30,  2.88it/s][A
copy images:  23%|██▎       | 176/781 [01:03<03:32,  2.84it/s][A
copy images:  23%|██▎       | 177/781 [01:03<03:30,  2.87it/s][A
copy images:  23%|██▎       | 178/781 [01:04<03:28,  2.89it/s][A
copy images:  23%|██▎       | 179/781 [01:04<03:30,  2.86it/s][A
copy images:  23%|██▎       | 180/781 [01:04<03:29,  2.86it/s][A
copy images:  23%|██▎       | 181/781 [01:05<03:30,  2.84it/s][A
copy images:  23%|██▎       | 182/781 [01:05<03:30,  2.85it/s][A
copy images:  23%|██▎       | 183/781 [01:05<03:29,  2.86it/s][A
copy images:  24%|██▎       | 184/781 [01:06<03:28,  2.87it/s][A
copy images:  24%|██▎       | 185/781 [01:06<03:28,  2.86it/s][A
copy images:  24%|██▍       | 186/781 [01:06<03:28,  2.86it/s][A
copy images:  24%|██▍       | 187/781 [01:07<03:37,  2.73it/s][A
copy images:  24%|██▍       | 188/781 [01:07<03:34,  2.77it/s][A
copy images:  24%|██▍       | 189/781 [01:08<03:30,  2.81it/s][A
copy images:  24%|██▍       | 190/781 [01:08<03:28,  2.84it/s][A
copy images:  24%|██▍       | 191/781 [01:08<03:25,  2.87it/s][A
copy images:  25%|██▍       | 192/781 [01:09<03:26,  2.86it/s][A
copy images:  25%|██▍       | 193/781 [01:09<03:24,  2.87it/s][A
copy images:  25%|██▍       | 194/781 [01:09<03:24,  2.87it/s][A
copy images:  25%|██▍       | 195/781 [01:10<03:27,  2.82it/s][A
copy images:  25%|██▌       | 196/781 [01:10<03:25,  2.85it/s][A
copy images:  25%|██▌       | 197/781 [01:10<03:25,  2.84it/s][A
copy images:  25%|██▌       | 198/781 [01:11<03:23,  2.86it/s][A
copy images:  25%|██▌       | 199/781 [01:11<03:22,  2.87it/s][A
copy images:  26%|██▌       | 200/781 [01:11<03:22,  2.87it/s][A
copy images:  26%|██▌       | 201/781 [01:12<03:21,  2.89it/s][A
copy images:  26%|██▌       | 202/781 [01:12<03:21,  2.87it/s][A
copy images:  26%|██▌       | 203/781 [01:12<03:21,  2.87it/s][A
copy images:  26%|██▌       | 204/781 [01:13<03:21,  2.86it/s][A
copy images:  26%|██▌       | 205/781 [01:13<03:21,  2.86it/s][A
copy images:  26%|██▋       | 206/781 [01:13<03:20,  2.87it/s][A
copy images:  27%|██▋       | 207/781 [01:14<03:19,  2.88it/s][A
copy images:  27%|██▋       | 208/781 [01:14<03:19,  2.87it/s][A
copy images:  27%|██▋       | 209/781 [01:15<03:19,  2.87it/s][A
copy images:  27%|██▋       | 210/781 [01:15<03:19,  2.87it/s][A
copy images:  27%|██▋       | 211/781 [01:15<03:20,  2.85it/s][A
copy images:  27%|██▋       | 212/781 [01:16<03:20,  2.84it/s][A
copy images:  27%|██▋       | 213/781 [01:16<03:20,  2.84it/s][A
copy images:  27%|██▋       | 214/781 [01:16<03:19,  2.84it/s][A
copy images:  28%|██▊       | 215/781 [01:17<03:18,  2.85it/s][A
copy images:  28%|██▊       | 216/781 [01:17<03:17,  2.86it/s][A
copy images:  28%|██▊       | 217/781 [01:17<03:17,  2.85it/s][A
copy images:  28%|██▊       | 218/781 [01:18<03:15,  2.88it/s][A
copy images:  28%|██▊       | 219/781 [01:18<03:16,  2.85it/s][A
copy images:  28%|██▊       | 220/781 [01:18<03:17,  2.84it/s][A
copy images:  28%|██▊       | 221/781 [01:19<03:14,  2.88it/s][A
copy images:  28%|██▊       | 222/781 [01:19<03:15,  2.86it/s][A
copy images:  29%|██▊       | 223/781 [01:19<03:15,  2.86it/s][A
copy images:  29%|██▊       | 224/781 [01:20<03:13,  2.87it/s][A
copy images:  29%|██▉       | 225/781 [01:20<03:15,  2.84it/s][A
copy images:  29%|██▉       | 226/781 [01:20<03:15,  2.83it/s][A
copy images:  29%|██▉       | 227/781 [01:21<03:13,  2.86it/s][A
copy images:  29%|██▉       | 228/781 [01:21<03:12,  2.87it/s][A
copy images:  29%|██▉       | 229/781 [01:22<03:13,  2.85it/s][A
copy images:  29%|██▉       | 230/781 [01:22<03:13,  2.85it/s][A
copy images:  30%|██▉       | 231/781 [01:22<03:12,  2.86it/s][A
copy images:  30%|██▉       | 232/781 [01:23<03:12,  2.86it/s][A
copy images:  30%|██▉       | 233/781 [01:23<03:11,  2.86it/s][A
copy images:  30%|██▉       | 234/781 [01:23<03:11,  2.86it/s][A
copy images:  30%|███       | 235/781 [01:24<03:10,  2.86it/s][A
copy images:  30%|███       | 236/781 [01:24<03:10,  2.86it/s][A
copy images:  30%|███       | 237/781 [01:24<03:09,  2.87it/s][A
copy images:  30%|███       | 238/781 [01:25<03:08,  2.89it/s][A
copy images:  31%|███       | 239/781 [01:25<03:08,  2.87it/s][A
copy images:  31%|███       | 240/781 [01:25<03:06,  2.90it/s][A
copy images:  31%|███       | 241/781 [01:26<03:06,  2.90it/s][A
copy images:  31%|███       | 242/781 [01:26<03:05,  2.90it/s][A
copy images:  31%|███       | 243/781 [01:26<03:05,  2.90it/s][A
copy images:  31%|███       | 244/781 [01:27<03:06,  2.88it/s][A
copy images:  31%|███▏      | 245/781 [01:27<03:05,  2.89it/s][A
copy images:  31%|███▏      | 246/781 [01:27<03:06,  2.87it/s][A
copy images:  32%|███▏      | 247/781 [01:28<03:05,  2.87it/s][A
copy images:  32%|███▏      | 248/781 [01:28<03:06,  2.85it/s][A
copy images:  32%|███▏      | 249/781 [01:28<03:05,  2.87it/s][A
copy images:  32%|███▏      | 250/781 [01:29<03:03,  2.89it/s][A
copy images:  32%|███▏      | 251/781 [01:29<03:06,  2.85it/s][A
copy images:  32%|███▏      | 252/781 [01:30<03:04,  2.87it/s][A
copy images:  32%|███▏      | 253/781 [01:30<03:03,  2.88it/s][A
copy images:  33%|███▎      | 254/781 [01:30<03:02,  2.89it/s][A
copy images:  33%|███▎      | 255/781 [01:31<03:01,  2.91it/s][A
copy images:  33%|███▎      | 256/781 [01:31<03:00,  2.91it/s][A
copy images:  33%|███▎      | 257/781 [01:31<03:00,  2.90it/s][A
copy images:  33%|███▎      | 258/781 [01:32<03:00,  2.89it/s][A
copy images:  33%|███▎      | 259/781 [01:32<03:00,  2.90it/s][A
copy images:  33%|███▎      | 260/781 [01:32<03:00,  2.88it/s][A
copy images:  33%|███▎      | 261/781 [01:33<03:00,  2.89it/s][A
copy images:  34%|███▎      | 262/781 [01:33<03:00,  2.88it/s][A
copy images:  34%|███▎      | 263/781 [01:33<03:01,  2.85it/s][A
copy images:  34%|███▍      | 264/781 [01:34<02:59,  2.88it/s][A
copy images:  34%|███▍      | 265/781 [01:34<02:58,  2.88it/s][A
copy images:  34%|███▍      | 266/781 [01:34<02:59,  2.87it/s][A
copy images:  34%|███▍      | 267/781 [01:35<02:59,  2.87it/s][A
copy images:  34%|███▍      | 268/781 [01:35<02:57,  2.88it/s][A
copy images:  34%|███▍      | 269/781 [01:35<02:58,  2.87it/s][A
copy images:  35%|███▍      | 270/781 [01:36<02:58,  2.86it/s][A
copy images:  35%|███▍      | 271/781 [01:36<02:58,  2.85it/s][A
copy images:  35%|███▍      | 272/781 [01:36<02:58,  2.86it/s][A
copy images:  35%|███▍      | 273/781 [01:37<02:57,  2.86it/s][A
copy images:  35%|███▌      | 274/781 [01:37<02:57,  2.85it/s][A
copy images:  35%|███▌      | 275/781 [01:38<02:56,  2.87it/s][A
copy images:  35%|███▌      | 276/781 [01:38<02:57,  2.84it/s][A
copy images:  35%|███▌      | 277/781 [01:38<02:56,  2.85it/s][A
copy images:  36%|███▌      | 278/781 [01:39<02:56,  2.85it/s][A
copy images:  36%|███▌      | 279/781 [01:39<02:55,  2.85it/s][A
copy images:  36%|███▌      | 280/781 [01:39<02:55,  2.86it/s][A
copy images:  36%|███▌      | 281/781 [01:40<02:56,  2.83it/s][A
copy images:  36%|███▌      | 282/781 [01:40<02:55,  2.84it/s][A
copy images:  36%|███▌      | 283/781 [01:40<02:54,  2.85it/s][A
copy images:  36%|███▋      | 284/781 [01:41<02:54,  2.84it/s][A
copy images:  36%|███▋      | 285/781 [01:41<02:55,  2.83it/s][A
copy images:  37%|███▋      | 286/781 [01:41<02:52,  2.86it/s][A
copy images:  37%|███▋      | 287/781 [01:42<02:52,  2.86it/s][A
copy images:  37%|███▋      | 288/781 [01:42<02:52,  2.86it/s][A
copy images:  37%|███▋      | 289/781 [01:42<02:51,  2.86it/s][A
copy images:  37%|███▋      | 290/781 [01:43<02:52,  2.85it/s][A
copy images:  37%|███▋      | 291/781 [01:43<02:51,  2.86it/s][A
copy images:  37%|███▋      | 292/781 [01:43<02:51,  2.86it/s][A
copy images:  38%|███▊      | 293/781 [01:44<02:49,  2.87it/s][A
copy images:  38%|███▊      | 294/781 [01:44<02:48,  2.89it/s][A
copy images:  38%|███▊      | 295/781 [01:45<02:49,  2.87it/s][A
copy images:  38%|███▊      | 296/781 [01:45<02:49,  2.87it/s][A
copy images:  38%|███▊      | 297/781 [01:45<02:47,  2.88it/s][A
copy images:  38%|███▊      | 298/781 [01:46<02:46,  2.89it/s][A
copy images:  38%|███▊      | 299/781 [01:46<02:47,  2.87it/s][A
copy images:  38%|███▊      | 300/781 [01:46<02:47,  2.88it/s][A
copy images:  39%|███▊      | 301/781 [01:47<02:47,  2.87it/s][A
copy images:  39%|███▊      | 302/781 [01:47<02:45,  2.89it/s][A
copy images:  39%|███▉      | 303/781 [01:47<02:44,  2.90it/s][A
copy images:  39%|███▉      | 304/781 [01:48<02:44,  2.89it/s][A
copy images:  39%|███▉      | 305/781 [01:48<02:44,  2.90it/s][A
copy images:  39%|███▉      | 306/781 [01:48<02:43,  2.90it/s][A
copy images:  39%|███▉      | 307/781 [01:49<02:44,  2.88it/s][A
copy images:  39%|███▉      | 308/781 [01:49<02:43,  2.89it/s][A
copy images:  40%|███▉      | 309/781 [01:49<02:42,  2.90it/s][A
copy images:  40%|███▉      | 310/781 [01:50<02:42,  2.90it/s][A
copy images:  40%|███▉      | 311/781 [01:50<02:41,  2.90it/s][A
copy images:  40%|███▉      | 312/781 [01:50<02:42,  2.89it/s][A
copy images:  40%|████      | 313/781 [01:51<02:41,  2.90it/s][A
copy images:  40%|████      | 314/781 [01:51<02:41,  2.89it/s][A
copy images:  40%|████      | 315/781 [01:51<02:41,  2.89it/s][A
copy images:  40%|████      | 316/781 [01:52<02:41,  2.88it/s][A
copy images:  41%|████      | 317/781 [01:52<02:41,  2.88it/s][A
copy images:  41%|████      | 318/781 [01:53<02:41,  2.87it/s][A
copy images:  41%|████      | 319/781 [01:53<02:39,  2.89it/s][A
copy images:  41%|████      | 320/781 [01:53<02:39,  2.88it/s][A
copy images:  41%|████      | 321/781 [01:54<02:39,  2.89it/s][A
copy images:  41%|████      | 322/781 [01:54<02:38,  2.90it/s][A
copy images:  41%|████▏     | 323/781 [01:54<02:38,  2.90it/s][A
copy images:  41%|████▏     | 324/781 [01:55<02:37,  2.91it/s][A
copy images:  42%|████▏     | 325/781 [01:55<02:37,  2.89it/s][A
copy images:  42%|████▏     | 326/781 [01:55<02:37,  2.89it/s][A
copy images:  42%|████▏     | 327/781 [01:56<02:37,  2.89it/s][A
copy images:  42%|████▏     | 328/781 [01:56<02:36,  2.90it/s][A
copy images:  42%|████▏     | 329/781 [01:56<02:37,  2.88it/s][A
copy images:  42%|████▏     | 330/781 [01:57<02:35,  2.91it/s][A
copy images:  42%|████▏     | 331/781 [01:57<02:34,  2.91it/s][A
copy images:  43%|████▎     | 332/781 [01:57<02:35,  2.89it/s][A
copy images:  43%|████▎     | 333/781 [01:58<02:34,  2.91it/s][A
copy images:  43%|████▎     | 334/781 [01:58<02:34,  2.89it/s][A
copy images:  43%|████▎     | 335/781 [01:58<02:34,  2.89it/s][A
copy images:  43%|████▎     | 336/781 [01:59<02:33,  2.90it/s][A
copy images:  43%|████▎     | 337/781 [01:59<02:34,  2.88it/s][A
copy images:  43%|████▎     | 338/781 [01:59<02:34,  2.87it/s][A
copy images:  43%|████▎     | 339/781 [02:00<02:33,  2.88it/s][A
copy images:  44%|████▎     | 340/781 [02:00<02:32,  2.89it/s][A
copy images:  44%|████▎     | 341/781 [02:00<02:31,  2.90it/s][A
copy images:  44%|████▍     | 342/781 [02:01<02:31,  2.90it/s][A
copy images:  44%|████▍     | 343/781 [02:01<02:30,  2.91it/s][A
copy images:  44%|████▍     | 344/781 [02:01<02:30,  2.90it/s][A
copy images:  44%|████▍     | 345/781 [02:02<02:29,  2.91it/s][A
copy images:  44%|████▍     | 346/781 [02:02<02:29,  2.91it/s][A
copy images:  44%|████▍     | 347/781 [02:03<02:31,  2.87it/s][A
copy images:  45%|████▍     | 348/781 [02:03<02:30,  2.87it/s][A
copy images:  45%|████▍     | 349/781 [02:03<02:30,  2.87it/s][A
copy images:  45%|████▍     | 350/781 [02:04<02:29,  2.89it/s][A
copy images:  45%|████▍     | 351/781 [02:04<02:28,  2.90it/s][A
copy images:  45%|████▌     | 352/781 [02:04<02:28,  2.89it/s][A
copy images:  45%|████▌     | 353/781 [02:05<02:27,  2.89it/s][A
copy images:  45%|████▌     | 354/781 [02:05<02:27,  2.89it/s][A
copy images:  45%|████▌     | 355/781 [02:05<02:26,  2.90it/s][A
copy images:  46%|████▌     | 356/781 [02:06<02:28,  2.87it/s][A
copy images:  46%|████▌     | 357/781 [02:06<02:27,  2.87it/s][A
copy images:  46%|████▌     | 358/781 [02:06<02:26,  2.88it/s][A
copy images:  46%|████▌     | 359/781 [02:07<02:26,  2.89it/s][A
copy images:  46%|████▌     | 360/781 [02:07<02:25,  2.89it/s][A
copy images:  46%|████▌     | 361/781 [02:07<02:25,  2.88it/s][A
copy images:  46%|████▋     | 362/781 [02:08<02:25,  2.88it/s][A
copy images:  46%|████▋     | 363/781 [02:08<02:24,  2.90it/s][A
copy images:  47%|████▋     | 364/781 [02:08<02:24,  2.88it/s][A
copy images:  47%|████▋     | 365/781 [02:09<02:24,  2.88it/s][A
copy images:  47%|████▋     | 366/781 [02:09<02:25,  2.86it/s][A
copy images:  47%|████▋     | 367/781 [02:09<02:24,  2.87it/s][A
copy images:  47%|████▋     | 368/781 [02:10<02:24,  2.86it/s][A
copy images:  47%|████▋     | 369/781 [02:10<02:21,  2.91it/s][A
copy images:  47%|████▋     | 370/781 [02:10<02:22,  2.89it/s][A
copy images:  48%|████▊     | 371/781 [02:11<02:21,  2.89it/s][A
copy images:  48%|████▊     | 372/781 [02:11<02:22,  2.87it/s][A
copy images:  48%|████▊     | 373/781 [02:12<02:21,  2.89it/s][A
copy images:  48%|████▊     | 374/781 [02:12<02:21,  2.87it/s][A
copy images:  48%|████▊     | 375/781 [02:12<02:21,  2.86it/s][A
copy images:  48%|████▊     | 376/781 [02:13<02:20,  2.87it/s][A
copy images:  48%|████▊     | 377/781 [02:13<02:20,  2.88it/s][A
copy images:  48%|████▊     | 378/781 [02:13<02:20,  2.87it/s][A
copy images:  49%|████▊     | 379/781 [02:14<02:20,  2.86it/s][A
copy images:  49%|████▊     | 380/781 [02:14<02:20,  2.86it/s][A
copy images:  49%|████▉     | 381/781 [02:14<02:18,  2.89it/s][A
copy images:  49%|████▉     | 382/781 [02:15<02:18,  2.89it/s][A
copy images:  49%|████▉     | 383/781 [02:15<02:18,  2.88it/s][A
copy images:  49%|████▉     | 384/781 [02:15<02:16,  2.91it/s][A
copy images:  49%|████▉     | 385/781 [02:16<02:15,  2.92it/s][A
copy images:  49%|████▉     | 386/781 [02:16<02:15,  2.92it/s][A
copy images:  50%|████▉     | 387/781 [02:16<02:15,  2.91it/s][A
copy images:  50%|████▉     | 388/781 [02:17<02:15,  2.89it/s][A
copy images:  50%|████▉     | 389/781 [02:17<02:15,  2.90it/s][A
copy images:  50%|████▉     | 390/781 [02:17<02:15,  2.89it/s][A
copy images:  50%|█████     | 391/781 [02:18<02:14,  2.91it/s][A
copy images:  50%|█████     | 392/781 [02:18<02:14,  2.89it/s][A
copy images:  50%|█████     | 393/781 [02:18<02:13,  2.90it/s][A
copy images:  50%|█████     | 394/781 [02:19<02:13,  2.91it/s][A
copy images:  51%|█████     | 395/781 [02:19<02:13,  2.90it/s][A
copy images:  51%|█████     | 396/781 [02:19<02:12,  2.91it/s][A
copy images:  51%|█████     | 397/781 [02:20<02:12,  2.90it/s][A
copy images:  51%|█████     | 398/781 [02:20<02:11,  2.91it/s][A
copy images:  51%|█████     | 399/781 [02:21<02:10,  2.92it/s][A
copy images:  51%|█████     | 400/781 [02:21<02:09,  2.93it/s][A
copy images:  51%|█████▏    | 401/781 [02:21<02:10,  2.91it/s][A
copy images:  51%|█████▏    | 402/781 [02:22<02:11,  2.89it/s][A
copy images:  52%|█████▏    | 403/781 [02:22<02:10,  2.90it/s][A
copy images:  52%|█████▏    | 404/781 [02:22<02:09,  2.92it/s][A
copy images:  52%|█████▏    | 405/781 [02:23<02:09,  2.90it/s][A
copy images:  52%|█████▏    | 406/781 [02:23<02:09,  2.89it/s][A
copy images:  52%|█████▏    | 407/781 [02:23<02:09,  2.89it/s][A
copy images:  52%|█████▏    | 408/781 [02:24<02:08,  2.89it/s][A
copy images:  52%|█████▏    | 409/781 [02:24<02:08,  2.89it/s][A
copy images:  52%|█████▏    | 410/781 [02:24<02:07,  2.90it/s][A
copy images:  53%|█████▎    | 411/781 [02:25<02:08,  2.89it/s][A
copy images:  53%|█████▎    | 412/781 [02:25<02:06,  2.91it/s][A
copy images:  53%|█████▎    | 413/781 [02:25<02:06,  2.91it/s][A
copy images:  53%|█████▎    | 414/781 [02:26<02:05,  2.92it/s][A
copy images:  53%|█████▎    | 415/781 [02:26<02:06,  2.90it/s][A
copy images:  53%|█████▎    | 416/781 [02:26<02:06,  2.89it/s][A
copy images:  53%|█████▎    | 417/781 [02:27<02:06,  2.88it/s][A
copy images:  54%|█████▎    | 418/781 [02:27<02:05,  2.88it/s][A
copy images:  54%|█████▎    | 419/781 [02:27<02:05,  2.89it/s][A
copy images:  54%|█████▍    | 420/781 [02:28<02:05,  2.88it/s][A
copy images:  54%|█████▍    | 421/781 [02:28<02:04,  2.89it/s][A
copy images:  54%|█████▍    | 422/781 [02:28<02:03,  2.91it/s][A
copy images:  54%|█████▍    | 423/781 [02:29<02:02,  2.93it/s][A
copy images:  54%|█████▍    | 424/781 [02:29<02:03,  2.90it/s][A
copy images:  54%|█████▍    | 425/781 [02:29<02:03,  2.88it/s][A
copy images:  55%|█████▍    | 426/781 [02:30<02:02,  2.91it/s][A
copy images:  55%|█████▍    | 427/781 [02:30<02:01,  2.91it/s][A
copy images:  55%|█████▍    | 428/781 [02:31<02:02,  2.89it/s][A
copy images:  55%|█████▍    | 429/781 [02:31<02:01,  2.89it/s][A
copy images:  55%|█████▌    | 430/781 [02:31<02:01,  2.88it/s][A
copy images:  55%|█████▌    | 431/781 [02:32<02:01,  2.89it/s][A
copy images:  55%|█████▌    | 432/781 [02:32<02:00,  2.90it/s][A
copy images:  55%|█████▌    | 433/781 [02:32<01:59,  2.90it/s][A
copy images:  56%|█████▌    | 434/781 [02:33<01:59,  2.89it/s][A
copy images:  56%|█████▌    | 435/781 [02:33<02:00,  2.87it/s][A
copy images:  56%|█████▌    | 436/781 [02:33<02:00,  2.87it/s][A
copy images:  56%|█████▌    | 437/781 [02:34<02:00,  2.87it/s][A
copy images:  56%|█████▌    | 438/781 [02:34<02:00,  2.85it/s][A
copy images:  56%|█████▌    | 439/781 [02:34<01:59,  2.87it/s][A
copy images:  56%|█████▋    | 440/781 [02:35<01:58,  2.87it/s][A
copy images:  56%|█████▋    | 441/781 [02:35<01:58,  2.86it/s][A
copy images:  57%|█████▋    | 442/781 [02:35<01:58,  2.85it/s][A
copy images:  57%|█████▋    | 443/781 [02:36<01:58,  2.86it/s][A
copy images:  57%|█████▋    | 444/781 [02:36<01:56,  2.89it/s][A
copy images:  57%|█████▋    | 445/781 [02:36<01:57,  2.87it/s][A
copy images:  57%|█████▋    | 446/781 [02:37<01:57,  2.86it/s][A
copy images:  57%|█████▋    | 447/781 [02:37<01:56,  2.86it/s][A
copy images:  57%|█████▋    | 448/781 [02:37<01:56,  2.86it/s][A
copy images:  57%|█████▋    | 449/781 [02:38<01:55,  2.88it/s][A
copy images:  58%|█████▊    | 450/781 [02:38<01:55,  2.87it/s][A
copy images:  58%|█████▊    | 451/781 [02:39<01:54,  2.88it/s][A
copy images:  58%|█████▊    | 452/781 [02:39<01:54,  2.88it/s][A
copy images:  58%|█████▊    | 453/781 [02:39<01:54,  2.86it/s][A
copy images:  58%|█████▊    | 454/781 [02:40<01:54,  2.86it/s][A
copy images:  58%|█████▊    | 455/781 [02:40<01:53,  2.88it/s][A
copy images:  58%|█████▊    | 456/781 [02:40<01:52,  2.90it/s][A
copy images:  59%|█████▊    | 457/781 [02:41<01:52,  2.88it/s][A
copy images:  59%|█████▊    | 458/781 [02:41<01:51,  2.90it/s][A
copy images:  59%|█████▉    | 459/781 [02:41<01:51,  2.88it/s][A
copy images:  59%|█████▉    | 460/781 [02:42<01:51,  2.89it/s][A
copy images:  59%|█████▉    | 461/781 [02:42<01:50,  2.90it/s][A
copy images:  59%|█████▉    | 462/781 [02:42<01:49,  2.92it/s][A
copy images:  59%|█████▉    | 463/781 [02:43<01:48,  2.92it/s][A
copy images:  59%|█████▉    | 464/781 [02:43<01:48,  2.93it/s][A
copy images:  60%|█████▉    | 465/781 [02:43<01:48,  2.90it/s][A
copy images:  60%|█████▉    | 466/781 [02:44<01:49,  2.88it/s][A
copy images:  60%|█████▉    | 467/781 [02:44<01:48,  2.89it/s][A
copy images:  60%|█████▉    | 468/781 [02:44<01:47,  2.92it/s][A
copy images:  60%|██████    | 469/781 [02:45<01:46,  2.92it/s][A
copy images:  60%|██████    | 470/781 [02:45<01:46,  2.91it/s][A
copy images:  60%|██████    | 471/781 [02:45<01:46,  2.92it/s][A
copy images:  60%|██████    | 472/781 [02:46<01:45,  2.92it/s][A
copy images:  61%|██████    | 473/781 [02:46<01:45,  2.91it/s][A
copy images:  61%|██████    | 474/781 [02:46<01:45,  2.92it/s][A
copy images:  61%|██████    | 475/781 [02:47<01:45,  2.91it/s][A
copy images:  61%|██████    | 476/781 [02:47<01:44,  2.91it/s][A
copy images:  61%|██████    | 477/781 [02:47<01:44,  2.91it/s][A
copy images:  61%|██████    | 478/781 [02:48<01:44,  2.89it/s][A
copy images:  61%|██████▏   | 479/781 [02:48<01:44,  2.89it/s][A
copy images:  61%|██████▏   | 480/781 [02:49<01:43,  2.91it/s][A
copy images:  62%|██████▏   | 481/781 [02:49<01:43,  2.90it/s][A
copy images:  62%|██████▏   | 482/781 [02:49<01:43,  2.88it/s][A
copy images:  62%|██████▏   | 483/781 [02:50<01:43,  2.88it/s][A
copy images:  62%|██████▏   | 484/781 [02:50<01:42,  2.89it/s][A
copy images:  62%|██████▏   | 485/781 [02:50<01:41,  2.91it/s][A
copy images:  62%|██████▏   | 486/781 [02:51<01:40,  2.93it/s][A
copy images:  62%|██████▏   | 487/781 [02:51<01:40,  2.92it/s][A
copy images:  62%|██████▏   | 488/781 [02:51<01:40,  2.92it/s][A
copy images:  63%|██████▎   | 489/781 [02:52<01:40,  2.91it/s][A
copy images:  63%|██████▎   | 490/781 [02:52<01:40,  2.89it/s][A
copy images:  63%|██████▎   | 491/781 [02:52<01:40,  2.89it/s][A
copy images:  63%|██████▎   | 492/781 [02:53<01:40,  2.89it/s][A
copy images:  63%|██████▎   | 493/781 [02:53<01:39,  2.90it/s][A
copy images:  63%|██████▎   | 494/781 [02:53<01:39,  2.88it/s][A
copy images:  63%|██████▎   | 495/781 [02:54<01:38,  2.89it/s][A
copy images:  64%|██████▎   | 496/781 [02:54<01:38,  2.90it/s][A
copy images:  64%|██████▎   | 497/781 [02:54<01:38,  2.87it/s][A
copy images:  64%|██████▍   | 498/781 [02:55<01:38,  2.88it/s][A
copy images:  64%|██████▍   | 499/781 [02:55<01:37,  2.88it/s][A
copy images:  64%|██████▍   | 500/781 [02:55<01:37,  2.89it/s][A
copy images:  64%|██████▍   | 501/781 [02:56<01:36,  2.90it/s][A
copy images:  64%|██████▍   | 502/781 [02:56<01:35,  2.91it/s][A
copy images:  64%|██████▍   | 503/781 [02:56<01:36,  2.89it/s][A
copy images:  65%|██████▍   | 504/781 [02:57<01:35,  2.90it/s][A
copy images:  65%|██████▍   | 505/781 [02:57<01:34,  2.91it/s][A
copy images:  65%|██████▍   | 506/781 [02:57<01:34,  2.92it/s][A
copy images:  65%|██████▍   | 507/781 [02:58<01:34,  2.90it/s][A
copy images:  65%|██████▌   | 508/781 [02:58<01:34,  2.90it/s][A
copy images:  65%|██████▌   | 509/781 [02:59<01:34,  2.87it/s][A
copy images:  65%|██████▌   | 510/781 [02:59<01:34,  2.87it/s][A
copy images:  65%|██████▌   | 511/781 [02:59<01:34,  2.85it/s][A
copy images:  66%|██████▌   | 512/781 [03:00<01:34,  2.85it/s][A
copy images:  66%|██████▌   | 513/781 [03:00<01:33,  2.85it/s][A
copy images:  66%|██████▌   | 514/781 [03:00<01:33,  2.85it/s][A
copy images:  66%|██████▌   | 515/781 [03:01<01:32,  2.87it/s][A
copy images:  66%|██████▌   | 516/781 [03:01<01:32,  2.87it/s][A
copy images:  66%|██████▌   | 517/781 [03:01<01:32,  2.85it/s][A
copy images:  66%|██████▋   | 518/781 [03:02<01:31,  2.87it/s][A
copy images:  66%|██████▋   | 519/781 [03:02<01:31,  2.88it/s][A
copy images:  67%|██████▋   | 520/781 [03:02<01:31,  2.85it/s][A
copy images:  67%|██████▋   | 521/781 [03:03<01:30,  2.87it/s][A
copy images:  67%|██████▋   | 522/781 [03:03<01:29,  2.88it/s][A
copy images:  67%|██████▋   | 523/781 [03:03<01:29,  2.90it/s][A
copy images:  67%|██████▋   | 524/781 [03:04<01:29,  2.88it/s][A
copy images:  67%|██████▋   | 525/781 [03:04<01:28,  2.88it/s][A
copy images:  67%|██████▋   | 526/781 [03:04<01:28,  2.87it/s][A
copy images:  67%|██████▋   | 527/781 [03:05<01:27,  2.89it/s][A
copy images:  68%|██████▊   | 528/781 [03:05<01:27,  2.89it/s][A
copy images:  68%|██████▊   | 529/781 [03:06<01:27,  2.87it/s][A
copy images:  68%|██████▊   | 530/781 [03:06<01:27,  2.87it/s][A
copy images:  68%|██████▊   | 531/781 [03:06<01:27,  2.87it/s][A
copy images:  68%|██████▊   | 532/781 [03:07<01:26,  2.87it/s][A
copy images:  68%|██████▊   | 533/781 [03:07<01:26,  2.88it/s][A
copy images:  68%|██████▊   | 534/781 [03:07<01:25,  2.89it/s][A
copy images:  69%|██████▊   | 535/781 [03:08<01:24,  2.90it/s][A
copy images:  69%|██████▊   | 536/781 [03:08<01:25,  2.88it/s][A
copy images:  69%|██████▉   | 537/781 [03:08<01:24,  2.88it/s][A
copy images:  69%|██████▉   | 538/781 [03:09<01:23,  2.90it/s][A
copy images:  69%|██████▉   | 539/781 [03:09<01:22,  2.92it/s][A
copy images:  69%|██████▉   | 540/781 [03:09<01:22,  2.92it/s][A
copy images:  69%|██████▉   | 541/781 [03:10<01:22,  2.91it/s][A
copy images:  69%|██████▉   | 542/781 [03:10<01:22,  2.90it/s][A
copy images:  70%|██████▉   | 543/781 [03:10<01:21,  2.91it/s][A
copy images:  70%|██████▉   | 544/781 [03:11<01:22,  2.89it/s][A
copy images:  70%|██████▉   | 545/781 [03:11<01:21,  2.88it/s][A
copy images:  70%|██████▉   | 546/781 [03:11<01:21,  2.89it/s][A
copy images:  70%|███████   | 547/781 [03:12<01:20,  2.90it/s][A
copy images:  70%|███████   | 548/781 [03:12<01:20,  2.88it/s][A
copy images:  70%|███████   | 549/781 [03:12<01:20,  2.88it/s][A
copy images:  70%|███████   | 550/781 [03:13<01:20,  2.87it/s][A
copy images:  71%|███████   | 551/781 [03:13<01:20,  2.87it/s][A
copy images:  71%|███████   | 552/781 [03:13<01:19,  2.86it/s][A
copy images:  71%|███████   | 553/781 [03:14<01:19,  2.88it/s][A
copy images:  71%|███████   | 554/781 [03:14<01:18,  2.88it/s][A
copy images:  71%|███████   | 555/781 [03:15<01:18,  2.87it/s][A
copy images:  71%|███████   | 556/781 [03:15<01:18,  2.87it/s][A
copy images:  71%|███████▏  | 557/781 [03:15<01:18,  2.86it/s][A
copy images:  71%|███████▏  | 558/781 [03:16<01:17,  2.86it/s][A
copy images:  72%|███████▏  | 559/781 [03:16<01:17,  2.87it/s][A
copy images:  72%|███████▏  | 560/781 [03:16<01:17,  2.86it/s][A
copy images:  72%|███████▏  | 561/781 [03:17<01:16,  2.86it/s][A
copy images:  72%|███████▏  | 562/781 [03:17<01:16,  2.86it/s][A
copy images:  72%|███████▏  | 563/781 [03:17<01:16,  2.87it/s][A
copy images:  72%|███████▏  | 564/781 [03:18<01:14,  2.90it/s][A
copy images:  72%|███████▏  | 565/781 [03:18<01:14,  2.91it/s][A
copy images:  72%|███████▏  | 566/781 [03:18<01:13,  2.91it/s][A
copy images:  73%|███████▎  | 567/781 [03:19<01:13,  2.91it/s][A
copy images:  73%|███████▎  | 568/781 [03:19<01:13,  2.92it/s][A
copy images:  73%|███████▎  | 569/781 [03:19<01:12,  2.91it/s][A
copy images:  73%|███████▎  | 570/781 [03:20<01:12,  2.90it/s][A
copy images:  73%|███████▎  | 571/781 [03:20<01:12,  2.88it/s][A
copy images:  73%|███████▎  | 572/781 [03:20<01:12,  2.88it/s][A
copy images:  73%|███████▎  | 573/781 [03:21<01:11,  2.89it/s][A
copy images:  73%|███████▎  | 574/781 [03:21<01:11,  2.89it/s][A
copy images:  74%|███████▎  | 575/781 [03:21<01:11,  2.89it/s][A
copy images:  74%|███████▍  | 576/781 [03:22<01:11,  2.87it/s][A
copy images:  74%|███████▍  | 577/781 [03:22<01:11,  2.86it/s][A
copy images:  74%|███████▍  | 578/781 [03:23<01:10,  2.88it/s][A
copy images:  74%|███████▍  | 579/781 [03:23<01:10,  2.88it/s][A
copy images:  74%|███████▍  | 580/781 [03:23<01:10,  2.86it/s][A
copy images:  74%|███████▍  | 581/781 [03:24<01:08,  2.90it/s][A
copy images:  75%|███████▍  | 582/781 [03:24<01:08,  2.92it/s][A
copy images:  75%|███████▍  | 583/781 [03:24<01:08,  2.90it/s][A
copy images:  75%|███████▍  | 584/781 [03:25<01:07,  2.91it/s][A
copy images:  75%|███████▍  | 585/781 [03:25<01:07,  2.91it/s][A
copy images:  75%|███████▌  | 586/781 [03:25<01:07,  2.88it/s][A
copy images:  75%|███████▌  | 587/781 [03:26<01:06,  2.90it/s][A
copy images:  75%|███████▌  | 588/781 [03:26<01:06,  2.88it/s][A
copy images:  75%|███████▌  | 589/781 [03:26<01:06,  2.89it/s][A
copy images:  76%|███████▌  | 590/781 [03:27<01:05,  2.90it/s][A
copy images:  76%|███████▌  | 591/781 [03:27<01:05,  2.89it/s][A
copy images:  76%|███████▌  | 592/781 [03:27<01:06,  2.85it/s][A
copy images:  76%|███████▌  | 593/781 [03:28<01:05,  2.87it/s][A
copy images:  76%|███████▌  | 594/781 [03:28<01:04,  2.89it/s][A
copy images:  76%|███████▌  | 595/781 [03:28<01:03,  2.91it/s][A
copy images:  76%|███████▋  | 596/781 [03:29<01:03,  2.90it/s][A
copy images:  76%|███████▋  | 597/781 [03:29<01:03,  2.90it/s][A
copy images:  77%|███████▋  | 598/781 [03:29<01:03,  2.90it/s][A
copy images:  77%|███████▋  | 599/781 [03:30<01:02,  2.90it/s][A
copy images:  77%|███████▋  | 600/781 [03:30<01:02,  2.89it/s][A
copy images:  77%|███████▋  | 601/781 [03:30<01:02,  2.89it/s][A
copy images:  77%|███████▋  | 602/781 [03:31<01:02,  2.88it/s][A
copy images:  77%|███████▋  | 603/781 [03:31<01:01,  2.88it/s][A
copy images:  77%|███████▋  | 604/781 [03:31<01:01,  2.88it/s][A
copy images:  77%|███████▋  | 605/781 [03:32<01:00,  2.91it/s][A
copy images:  78%|███████▊  | 606/781 [03:32<01:00,  2.91it/s][A
copy images:  78%|███████▊  | 607/781 [03:33<00:59,  2.91it/s][A
copy images:  78%|███████▊  | 608/781 [03:33<00:59,  2.92it/s][A
copy images:  78%|███████▊  | 609/781 [03:33<00:58,  2.92it/s][A
copy images:  78%|███████▊  | 610/781 [03:34<00:58,  2.91it/s][A
copy images:  78%|███████▊  | 611/781 [03:34<00:58,  2.91it/s][A
copy images:  78%|███████▊  | 612/781 [03:34<00:58,  2.91it/s][A
copy images:  78%|███████▊  | 613/781 [03:35<00:57,  2.90it/s][A
copy images:  79%|███████▊  | 614/781 [03:35<00:57,  2.91it/s][A
copy images:  79%|███████▊  | 615/781 [03:35<00:56,  2.92it/s][A
copy images:  79%|███████▉  | 616/781 [03:36<00:56,  2.91it/s][A
copy images:  79%|███████▉  | 617/781 [03:36<00:55,  2.94it/s][A
copy images:  79%|███████▉  | 618/781 [03:36<00:55,  2.92it/s][A
copy images:  79%|███████▉  | 619/781 [03:37<00:55,  2.94it/s][A
copy images:  79%|███████▉  | 620/781 [03:37<00:55,  2.91it/s][A
copy images:  80%|███████▉  | 621/781 [03:37<00:54,  2.92it/s][A
copy images:  80%|███████▉  | 622/781 [03:38<00:54,  2.92it/s][A
copy images:  80%|███████▉  | 623/781 [03:38<00:54,  2.89it/s][A
copy images:  80%|███████▉  | 624/781 [03:38<00:54,  2.90it/s][A
copy images:  80%|████████  | 625/781 [03:39<00:53,  2.93it/s][A
copy images:  80%|████████  | 626/781 [03:39<00:53,  2.89it/s][A
copy images:  80%|████████  | 627/781 [03:39<00:53,  2.90it/s][A
copy images:  80%|████████  | 628/781 [03:40<00:53,  2.87it/s][A
copy images:  81%|████████  | 629/781 [03:40<00:52,  2.88it/s][A
copy images:  81%|████████  | 630/781 [03:40<00:52,  2.88it/s][A
copy images:  81%|████████  | 631/781 [03:41<00:51,  2.89it/s][A
copy images:  81%|████████  | 632/781 [03:41<00:51,  2.90it/s][A
copy images:  81%|████████  | 633/781 [03:41<00:51,  2.89it/s][A
copy images:  81%|████████  | 634/781 [03:42<00:51,  2.88it/s][A
copy images:  81%|████████▏ | 635/781 [03:42<00:51,  2.86it/s][A
copy images:  81%|████████▏ | 636/781 [03:43<00:50,  2.85it/s][A
copy images:  82%|████████▏ | 637/781 [03:43<00:50,  2.84it/s][A
copy images:  82%|████████▏ | 638/781 [03:43<00:49,  2.88it/s][A
copy images:  82%|████████▏ | 639/781 [03:44<00:49,  2.88it/s][A
copy images:  82%|████████▏ | 640/781 [03:44<00:49,  2.85it/s][A
copy images:  82%|████████▏ | 641/781 [03:44<00:49,  2.84it/s][A
copy images:  82%|████████▏ | 642/781 [03:45<00:48,  2.85it/s][A
copy images:  82%|████████▏ | 643/781 [03:45<00:48,  2.85it/s][A
copy images:  82%|████████▏ | 644/781 [03:45<00:47,  2.87it/s][A
copy images:  83%|████████▎ | 645/781 [03:46<00:46,  2.91it/s][A
copy images:  83%|████████▎ | 646/781 [03:46<00:46,  2.90it/s][A
copy images:  83%|████████▎ | 647/781 [03:46<00:46,  2.88it/s][A
copy images:  83%|████████▎ | 648/781 [03:47<00:46,  2.88it/s][A
copy images:  83%|████████▎ | 649/781 [03:47<00:45,  2.91it/s][A
copy images:  83%|████████▎ | 650/781 [03:47<00:45,  2.89it/s][A
copy images:  83%|████████▎ | 651/781 [03:48<00:44,  2.90it/s][A
copy images:  83%|████████▎ | 652/781 [03:48<00:44,  2.91it/s][A
copy images:  84%|████████▎ | 653/781 [03:48<00:44,  2.91it/s][A
copy images:  84%|████████▎ | 654/781 [03:49<00:43,  2.92it/s][A
copy images:  84%|████████▍ | 655/781 [03:49<00:43,  2.90it/s][A
copy images:  84%|████████▍ | 656/781 [03:49<00:43,  2.91it/s][A
copy images:  84%|████████▍ | 657/781 [03:50<00:42,  2.91it/s][A
copy images:  84%|████████▍ | 658/781 [03:50<00:42,  2.91it/s][A
copy images:  84%|████████▍ | 659/781 [03:50<00:41,  2.91it/s][A
copy images:  85%|████████▍ | 660/781 [03:51<00:41,  2.90it/s][A
copy images:  85%|████████▍ | 661/781 [03:51<00:41,  2.90it/s][A
copy images:  85%|████████▍ | 662/781 [03:52<00:41,  2.89it/s][A
copy images:  85%|████████▍ | 663/781 [03:52<00:40,  2.88it/s][A
copy images:  85%|████████▌ | 664/781 [03:52<00:40,  2.88it/s][A
copy images:  85%|████████▌ | 665/781 [03:53<00:39,  2.91it/s][A
copy images:  85%|████████▌ | 666/781 [03:53<00:39,  2.91it/s][A
copy images:  85%|████████▌ | 667/781 [03:53<00:39,  2.92it/s][A
copy images:  86%|████████▌ | 668/781 [03:54<00:38,  2.92it/s][A
copy images:  86%|████████▌ | 669/781 [03:54<00:38,  2.91it/s][A
copy images:  86%|████████▌ | 670/781 [03:54<00:38,  2.91it/s][A
copy images:  86%|████████▌ | 671/781 [03:55<00:38,  2.88it/s][A
copy images:  86%|████████▌ | 672/781 [03:55<00:37,  2.89it/s][A
copy images:  86%|████████▌ | 673/781 [03:55<00:37,  2.89it/s][A
copy images:  86%|████████▋ | 674/781 [03:56<00:36,  2.91it/s][A
copy images:  86%|████████▋ | 675/781 [03:56<00:36,  2.90it/s][A
copy images:  87%|████████▋ | 676/781 [03:56<00:36,  2.89it/s][A
copy images:  87%|████████▋ | 677/781 [03:57<00:35,  2.92it/s][A
copy images:  87%|████████▋ | 678/781 [03:57<00:35,  2.92it/s][A
copy images:  87%|████████▋ | 679/781 [03:57<00:35,  2.90it/s][A
copy images:  87%|████████▋ | 680/781 [03:58<00:34,  2.90it/s][A
copy images:  87%|████████▋ | 681/781 [03:58<00:34,  2.88it/s][A
copy images:  87%|████████▋ | 682/781 [03:58<00:34,  2.90it/s][A
copy images:  87%|████████▋ | 683/781 [03:59<00:34,  2.88it/s][A
copy images:  88%|████████▊ | 684/781 [03:59<00:33,  2.89it/s][A
copy images:  88%|████████▊ | 685/781 [03:59<00:32,  2.92it/s][A
copy images:  88%|████████▊ | 686/781 [04:00<00:32,  2.91it/s][A
copy images:  88%|████████▊ | 687/781 [04:00<00:32,  2.89it/s][A
copy images:  88%|████████▊ | 688/781 [04:00<00:32,  2.89it/s][A
copy images:  88%|████████▊ | 689/781 [04:01<00:31,  2.91it/s][A
copy images:  88%|████████▊ | 690/781 [04:01<00:31,  2.90it/s][A
copy images:  88%|████████▊ | 691/781 [04:02<00:31,  2.90it/s][A
copy images:  89%|████████▊ | 692/781 [04:02<00:30,  2.90it/s][A
copy images:  89%|████████▊ | 693/781 [04:02<00:30,  2.89it/s][A
copy images:  89%|████████▉ | 694/781 [04:03<00:30,  2.89it/s][A
copy images:  89%|████████▉ | 695/781 [04:03<00:29,  2.90it/s][A
copy images:  89%|████████▉ | 696/781 [04:03<00:29,  2.88it/s][A
copy images:  89%|████████▉ | 697/781 [04:04<00:29,  2.88it/s][A
copy images:  89%|████████▉ | 698/781 [04:04<00:28,  2.89it/s][A
copy images:  90%|████████▉ | 699/781 [04:04<00:28,  2.90it/s][A
copy images:  90%|████████▉ | 700/781 [04:05<00:27,  2.91it/s][A
copy images:  90%|████████▉ | 701/781 [04:05<00:27,  2.89it/s][A
copy images:  90%|████████▉ | 702/781 [04:05<00:27,  2.86it/s][A
copy images:  90%|█████████ | 703/781 [04:06<00:27,  2.86it/s][A
copy images:  90%|█████████ | 704/781 [04:06<00:26,  2.87it/s][A
copy images:  90%|█████████ | 705/781 [04:06<00:26,  2.88it/s][A
copy images:  90%|█████████ | 706/781 [04:07<00:25,  2.90it/s][A
copy images:  91%|█████████ | 707/781 [04:07<00:25,  2.89it/s][A
copy images:  91%|█████████ | 708/781 [04:07<00:25,  2.89it/s][A
copy images:  91%|█████████ | 709/781 [04:08<00:24,  2.90it/s][A
copy images:  91%|█████████ | 710/781 [04:08<00:24,  2.89it/s][A
copy images:  91%|█████████ | 711/781 [04:08<00:24,  2.89it/s][A
copy images:  91%|█████████ | 712/781 [04:09<00:23,  2.88it/s][A
copy images:  91%|█████████▏| 713/781 [04:09<00:23,  2.89it/s][A
copy images:  91%|█████████▏| 714/781 [04:09<00:23,  2.87it/s][A
copy images:  92%|█████████▏| 715/781 [04:10<00:23,  2.86it/s][A
copy images:  92%|█████████▏| 716/781 [04:10<00:22,  2.84it/s][A
copy images:  92%|█████████▏| 717/781 [04:11<00:22,  2.85it/s][A
copy images:  92%|█████████▏| 718/781 [04:11<00:21,  2.87it/s][A
copy images:  92%|█████████▏| 719/781 [04:11<00:21,  2.86it/s][A
copy images:  92%|█████████▏| 720/781 [04:12<00:21,  2.87it/s][A
copy images:  92%|█████████▏| 721/781 [04:12<00:20,  2.89it/s][A
copy images:  92%|█████████▏| 722/781 [04:12<00:20,  2.90it/s][A
copy images:  93%|█████████▎| 723/781 [04:13<00:19,  2.92it/s][A
copy images:  93%|█████████▎| 724/781 [04:13<00:19,  2.91it/s][A
copy images:  93%|█████████▎| 725/781 [04:13<00:19,  2.90it/s][A
copy images:  93%|█████████▎| 726/781 [04:14<00:18,  2.93it/s][A
copy images:  93%|█████████▎| 727/781 [04:14<00:18,  2.93it/s][A
copy images:  93%|█████████▎| 728/781 [04:14<00:18,  2.90it/s][A
copy images:  93%|█████████▎| 729/781 [04:15<00:18,  2.87it/s][A
copy images:  93%|█████████▎| 730/781 [04:15<00:17,  2.88it/s][A
copy images:  94%|█████████▎| 731/781 [04:15<00:17,  2.89it/s][A
copy images:  94%|█████████▎| 732/781 [04:16<00:17,  2.88it/s][A
copy images:  94%|█████████▍| 733/781 [04:16<00:16,  2.90it/s][A
copy images:  94%|█████████▍| 734/781 [04:16<00:16,  2.91it/s][A
copy images:  94%|█████████▍| 735/781 [04:17<00:15,  2.92it/s][A
copy images:  94%|█████████▍| 736/781 [04:17<00:15,  2.90it/s][A
copy images:  94%|█████████▍| 737/781 [04:17<00:15,  2.90it/s][A
copy images:  94%|█████████▍| 738/781 [04:18<00:14,  2.93it/s][A
copy images:  95%|█████████▍| 739/781 [04:18<00:14,  2.91it/s][A
copy images:  95%|█████████▍| 740/781 [04:18<00:14,  2.89it/s][A
copy images:  95%|█████████▍| 741/781 [04:19<00:13,  2.91it/s][A
copy images:  95%|█████████▌| 742/781 [04:19<00:13,  2.89it/s][A
copy images:  95%|█████████▌| 743/781 [04:20<00:13,  2.91it/s][A
copy images:  95%|█████████▌| 744/781 [04:20<00:12,  2.94it/s][A
copy images:  95%|█████████▌| 745/781 [04:20<00:12,  2.92it/s][A
copy images:  96%|█████████▌| 746/781 [04:21<00:11,  2.92it/s][A
copy images:  96%|█████████▌| 747/781 [04:21<00:11,  2.90it/s][A
copy images:  96%|█████████▌| 748/781 [04:21<00:11,  2.89it/s][A
copy images:  96%|█████████▌| 749/781 [04:22<00:11,  2.87it/s][A
copy images:  96%|█████████▌| 750/781 [04:22<00:10,  2.87it/s][A
copy images:  96%|█████████▌| 751/781 [04:22<00:10,  2.88it/s][A
copy images:  96%|█████████▋| 752/781 [04:23<00:10,  2.89it/s][A
copy images:  96%|█████████▋| 753/781 [04:23<00:09,  2.88it/s][A
copy images:  97%|█████████▋| 754/781 [04:23<00:09,  2.88it/s][A
copy images:  97%|█████████▋| 755/781 [04:24<00:09,  2.87it/s][A
copy images:  97%|█████████▋| 756/781 [04:24<00:08,  2.87it/s][A
copy images:  97%|█████████▋| 757/781 [04:24<00:08,  2.89it/s][A
copy images:  97%|█████████▋| 758/781 [04:25<00:07,  2.93it/s][A
copy images:  97%|█████████▋| 759/781 [04:25<00:07,  2.90it/s][A
copy images:  97%|█████████▋| 760/781 [04:25<00:07,  2.91it/s][A
copy images:  97%|█████████▋| 761/781 [04:26<00:06,  2.90it/s][A
copy images:  98%|█████████▊| 762/781 [04:26<00:06,  2.90it/s][A
copy images:  98%|█████████▊| 763/781 [04:26<00:06,  2.92it/s][A
copy images:  98%|█████████▊| 764/781 [04:27<00:05,  2.92it/s][A
copy images:  98%|█████████▊| 765/781 [04:27<00:05,  2.93it/s][A
copy images:  98%|█████████▊| 766/781 [04:27<00:05,  2.90it/s][A
copy images:  98%|█████████▊| 767/781 [04:28<00:04,  2.87it/s][A
copy images:  98%|█████████▊| 768/781 [04:28<00:04,  2.87it/s][A
copy images:  98%|█████████▊| 769/781 [04:29<00:05,  2.13it/s][A
copy images:  99%|█████████▊| 770/781 [04:29<00:04,  2.31it/s][A
copy images:  99%|█████████▊| 771/781 [04:30<00:04,  2.46it/s][A
copy images:  99%|█████████▉| 772/781 [04:30<00:03,  2.58it/s][A
copy images:  99%|█████████▉| 773/781 [04:30<00:03,  2.64it/s][A
copy images:  99%|█████████▉| 774/781 [04:31<00:02,  2.70it/s][A
copy images:  99%|█████████▉| 775/781 [04:31<00:02,  2.76it/s][A
copy images:  99%|█████████▉| 776/781 [04:31<00:01,  2.78it/s][A
copy images:  99%|█████████▉| 777/781 [04:32<00:01,  2.79it/s][A
copy images: 100%|█████████▉| 778/781 [04:32<00:01,  2.82it/s][A
copy images: 100%|█████████▉| 779/781 [04:32<00:00,  2.85it/s][A
copy images: 100%|█████████▉| 780/781 [04:33<00:00,  2.85it/s][A
copy images: 100%|██████████| 781/781 [04:33<00:00,  2.87it/s][Acopy images: 100%|██████████| 781/781 [04:33<00:00,  2.85it/s]

generating images:   0%|          | 0/781 [00:00<?, ?it/s][Agenerating images:   0%|          | 0/781 [00:00<?, ?it/s]/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/HouseholderGAN/diffae/diffusion/base.py:306: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast(self.conf.fp16):
/nfs/data_chaos/czhang/HouseholderGAN/diffae/diffusion/base.py:306: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast(self.conf.fp16):
generating images:   0%|          | 1/781 [00:05<1:05:23,  5.03s/it]
generating images:   0%|          | 1/781 [00:05<1:06:29,  5.12s/it][Agenerating images:   0%|          | 2/781 [00:08<54:45,  4.22s/it]  
generating images:   0%|          | 2/781 [00:08<55:28,  4.27s/it]  [Agenerating images:   0%|          | 3/781 [00:12<51:20,  3.96s/it]
generating images:   0%|          | 3/781 [00:12<51:48,  4.00s/it][Agenerating images:   1%|          | 4/781 [00:16<49:47,  3.85s/it]
generating images:   1%|          | 4/781 [00:16<50:29,  3.90s/it][Agenerating images:   1%|          | 5/781 [00:19<48:44,  3.77s/it]
generating images:   1%|          | 5/781 [00:19<49:19,  3.81s/it][Agenerating images:   1%|          | 6/781 [00:23<48:06,  3.72s/it]
generating images:   1%|          | 6/781 [00:23<48:44,  3.77s/it][Agenerating images:   1%|          | 7/781 [00:26<47:40,  3.70s/it]
generating images:   1%|          | 7/781 [00:27<48:11,  3.74s/it][Agenerating images:   1%|          | 8/781 [00:30<47:26,  3.68s/it]
generating images:   1%|          | 8/781 [00:30<47:50,  3.71s/it][Agenerating images:   1%|          | 9/781 [00:34<47:15,  3.67s/it]
generating images:   1%|          | 9/781 [00:34<47:37,  3.70s/it][Agenerating images:   1%|▏         | 10/781 [00:37<47:08,  3.67s/it]
generating images:   1%|▏         | 10/781 [00:38<47:26,  3.69s/it][Agenerating images:   1%|▏         | 11/781 [00:41<47:02,  3.67s/it]
generating images:   1%|▏         | 11/781 [00:41<47:17,  3.69s/it][Agenerating images:   2%|▏         | 12/781 [00:45<46:56,  3.66s/it]
generating images:   2%|▏         | 12/781 [00:45<47:10,  3.68s/it][Agenerating images:   2%|▏         | 13/781 [00:48<46:51,  3.66s/it]
generating images:   2%|▏         | 13/781 [00:49<47:04,  3.68s/it][Agenerating images:   2%|▏         | 14/781 [00:52<46:46,  3.66s/it]
generating images:   2%|▏         | 14/781 [00:52<46:58,  3.67s/it][Agenerating images:   2%|▏         | 15/781 [00:56<46:43,  3.66s/it]
generating images:   2%|▏         | 15/781 [00:56<46:53,  3.67s/it][Agenerating images:   2%|▏         | 16/781 [00:59<46:39,  3.66s/it]
generating images:   2%|▏         | 16/781 [01:00<46:49,  3.67s/it][Agenerating images:   2%|▏         | 17/781 [01:03<46:34,  3.66s/it]
generating images:   2%|▏         | 17/781 [01:03<46:44,  3.67s/it][Agenerating images:   2%|▏         | 18/781 [01:07<46:30,  3.66s/it]
generating images:   2%|▏         | 18/781 [01:07<46:39,  3.67s/it][Agenerating images:   2%|▏         | 19/781 [01:10<46:25,  3.65s/it]
generating images:   2%|▏         | 19/781 [01:11<46:35,  3.67s/it][Agenerating images:   3%|▎         | 20/781 [01:14<46:22,  3.66s/it]
generating images:   3%|▎         | 20/781 [01:14<46:30,  3.67s/it][Agenerating images:   3%|▎         | 21/781 [01:18<46:17,  3.66s/it]
generating images:   3%|▎         | 21/781 [01:18<46:27,  3.67s/it][Agenerating images:   3%|▎         | 22/781 [01:21<46:13,  3.65s/it]
generating images:   3%|▎         | 22/781 [01:22<46:24,  3.67s/it][Agenerating images:   3%|▎         | 23/781 [01:25<46:09,  3.65s/it]
generating images:   3%|▎         | 23/781 [01:25<46:21,  3.67s/it][Agenerating images:   3%|▎         | 24/781 [01:29<46:07,  3.66s/it]
generating images:   3%|▎         | 24/781 [01:29<46:18,  3.67s/it][Agenerating images:   3%|▎         | 25/781 [01:32<46:03,  3.66s/it]
generating images:   3%|▎         | 25/781 [01:33<46:10,  3.66s/it][Agenerating images:   3%|▎         | 26/781 [01:36<45:59,  3.65s/it]
generating images:   3%|▎         | 26/781 [01:36<46:07,  3.67s/it][Agenerating images:   3%|▎         | 27/781 [01:40<45:51,  3.65s/it]
generating images:   3%|▎         | 27/781 [01:40<46:04,  3.67s/it][Agenerating images:   4%|▎         | 28/781 [01:43<45:50,  3.65s/it]
generating images:   4%|▎         | 28/781 [01:44<46:02,  3.67s/it][Agenerating images:   4%|▎         | 29/781 [01:47<45:47,  3.65s/it]
generating images:   4%|▎         | 29/781 [01:47<45:58,  3.67s/it][Agenerating images:   4%|▍         | 30/781 [01:50<45:44,  3.65s/it]
generating images:   4%|▍         | 30/781 [01:51<45:56,  3.67s/it][Agenerating images:   4%|▍         | 31/781 [01:54<45:42,  3.66s/it]
generating images:   4%|▍         | 31/781 [01:55<45:53,  3.67s/it][Agenerating images:   4%|▍         | 32/781 [01:58<45:39,  3.66s/it]
generating images:   4%|▍         | 32/781 [01:58<45:50,  3.67s/it][Agenerating images:   4%|▍         | 33/781 [02:01<45:35,  3.66s/it]
generating images:   4%|▍         | 33/781 [02:02<45:47,  3.67s/it][Agenerating images:   4%|▍         | 34/781 [02:05<45:31,  3.66s/it]
generating images:   4%|▍         | 34/781 [02:06<45:44,  3.67s/it][Agenerating images:   4%|▍         | 35/781 [02:09<45:24,  3.65s/it]
generating images:   4%|▍         | 35/781 [02:09<45:38,  3.67s/it][Agenerating images:   5%|▍         | 36/781 [02:12<45:21,  3.65s/it]
generating images:   5%|▍         | 36/781 [02:13<45:35,  3.67s/it][Agenerating images:   5%|▍         | 37/781 [02:16<45:15,  3.65s/it]
generating images:   5%|▍         | 37/781 [02:17<45:30,  3.67s/it][Agenerating images:   5%|▍         | 38/781 [02:20<45:15,  3.65s/it]
generating images:   5%|▍         | 38/781 [02:20<45:27,  3.67s/it][Agenerating images:   5%|▍         | 39/781 [02:23<45:12,  3.66s/it]
generating images:   5%|▍         | 39/781 [02:24<45:23,  3.67s/it][Agenerating images:   5%|▌         | 40/781 [02:27<45:08,  3.65s/it]
generating images:   5%|▌         | 40/781 [02:28<45:19,  3.67s/it][Agenerating images:   5%|▌         | 41/781 [02:31<45:04,  3.65s/it]
generating images:   5%|▌         | 41/781 [02:31<45:15,  3.67s/it][Agenerating images:   5%|▌         | 42/781 [02:34<45:01,  3.66s/it]
generating images:   5%|▌         | 42/781 [02:35<45:11,  3.67s/it][Agenerating images:   6%|▌         | 43/781 [02:38<44:58,  3.66s/it]
generating images:   6%|▌         | 43/781 [02:39<45:07,  3.67s/it][Agenerating images:   6%|▌         | 44/781 [02:42<44:54,  3.66s/it]
generating images:   6%|▌         | 44/781 [02:42<45:04,  3.67s/it][Agenerating images:   6%|▌         | 45/781 [02:45<44:51,  3.66s/it]
generating images:   6%|▌         | 45/781 [02:46<45:00,  3.67s/it][Agenerating images:   6%|▌         | 46/781 [02:49<44:47,  3.66s/it]
generating images:   6%|▌         | 46/781 [02:50<44:57,  3.67s/it][Agenerating images:   6%|▌         | 47/781 [02:53<44:42,  3.66s/it]
generating images:   6%|▌         | 47/781 [02:54<44:54,  3.67s/it][Agenerating images:   6%|▌         | 48/781 [02:56<44:40,  3.66s/it]
generating images:   6%|▌         | 48/781 [02:57<44:51,  3.67s/it][Agenerating images:   6%|▋         | 49/781 [03:00<44:37,  3.66s/it]
generating images:   6%|▋         | 49/781 [03:01<44:48,  3.67s/it][Agenerating images:   6%|▋         | 50/781 [03:04<44:34,  3.66s/it]
generating images:   6%|▋         | 50/781 [03:05<44:43,  3.67s/it][Agenerating images:   7%|▋         | 51/781 [03:07<44:30,  3.66s/it]
generating images:   7%|▋         | 51/781 [03:08<44:39,  3.67s/it][Agenerating images:   7%|▋         | 52/781 [03:11<44:26,  3.66s/it]
generating images:   7%|▋         | 52/781 [03:12<44:36,  3.67s/it][Agenerating images:   7%|▋         | 53/781 [03:15<44:22,  3.66s/it]
generating images:   7%|▋         | 53/781 [03:16<44:33,  3.67s/it][Agenerating images:   7%|▋         | 54/781 [03:18<44:18,  3.66s/it]
generating images:   7%|▋         | 54/781 [03:19<44:29,  3.67s/it][Agenerating images:   7%|▋         | 55/781 [03:22<44:14,  3.66s/it]
generating images:   7%|▋         | 55/781 [03:23<44:25,  3.67s/it][Agenerating images:   7%|▋         | 56/781 [03:26<44:10,  3.66s/it]
generating images:   7%|▋         | 56/781 [03:27<44:21,  3.67s/it][Agenerating images:   7%|▋         | 57/781 [03:29<44:08,  3.66s/it]
generating images:   7%|▋         | 57/781 [03:30<44:18,  3.67s/it][Agenerating images:   7%|▋         | 58/781 [03:33<44:04,  3.66s/it]
generating images:   7%|▋         | 58/781 [03:34<44:14,  3.67s/it][Agenerating images:   8%|▊         | 59/781 [03:37<44:02,  3.66s/it]
generating images:   8%|▊         | 59/781 [03:38<44:12,  3.67s/it][Agenerating images:   8%|▊         | 60/781 [03:40<43:57,  3.66s/it]
generating images:   8%|▊         | 60/781 [03:41<44:09,  3.67s/it][Agenerating images:   8%|▊         | 61/781 [03:44<43:53,  3.66s/it]
generating images:   8%|▊         | 61/781 [03:45<44:05,  3.67s/it][Agenerating images:   8%|▊         | 62/781 [03:47<43:49,  3.66s/it]
generating images:   8%|▊         | 62/781 [03:49<44:02,  3.67s/it][Agenerating images:   8%|▊         | 63/781 [03:51<43:45,  3.66s/it]
generating images:   8%|▊         | 63/781 [03:52<43:57,  3.67s/it][Agenerating images:   8%|▊         | 64/781 [03:55<43:41,  3.66s/it]
generating images:   8%|▊         | 64/781 [03:56<43:52,  3.67s/it][Agenerating images:   8%|▊         | 65/781 [03:58<43:37,  3.66s/it]
generating images:   8%|▊         | 65/781 [04:00<43:47,  3.67s/it][Agenerating images:   8%|▊         | 66/781 [04:02<43:35,  3.66s/it]
generating images:   8%|▊         | 66/781 [04:03<43:44,  3.67s/it][Agenerating images:   9%|▊         | 67/781 [04:06<43:30,  3.66s/it]
generating images:   9%|▊         | 67/781 [04:07<43:40,  3.67s/it][Agenerating images:   9%|▊         | 68/781 [04:09<43:27,  3.66s/it]
generating images:   9%|▊         | 68/781 [04:11<43:36,  3.67s/it][Agenerating images:   9%|▉         | 69/781 [04:13<43:23,  3.66s/it]
generating images:   9%|▉         | 69/781 [04:14<43:32,  3.67s/it][Agenerating images:   9%|▉         | 70/781 [04:17<43:20,  3.66s/it]
generating images:   9%|▉         | 70/781 [04:18<43:31,  3.67s/it][Agenerating images:   9%|▉         | 71/781 [04:20<43:18,  3.66s/it]
generating images:   9%|▉         | 71/781 [04:22<43:26,  3.67s/it][Agenerating images:   9%|▉         | 72/781 [04:24<43:14,  3.66s/it]
generating images:   9%|▉         | 72/781 [04:25<43:22,  3.67s/it][Agenerating images:   9%|▉         | 73/781 [04:28<43:10,  3.66s/it]
generating images:   9%|▉         | 73/781 [04:29<43:20,  3.67s/it][Agenerating images:   9%|▉         | 74/781 [04:31<43:05,  3.66s/it]
generating images:   9%|▉         | 74/781 [04:33<43:15,  3.67s/it][Agenerating images:  10%|▉         | 75/781 [04:35<43:01,  3.66s/it]
generating images:  10%|▉         | 75/781 [04:36<43:12,  3.67s/it][Agenerating images:  10%|▉         | 76/781 [04:39<42:57,  3.66s/it]
generating images:  10%|▉         | 76/781 [04:40<43:08,  3.67s/it][Agenerating images:  10%|▉         | 77/781 [04:42<42:52,  3.65s/it]
generating images:  10%|▉         | 77/781 [04:44<43:04,  3.67s/it][Agenerating images:  10%|▉         | 78/781 [04:46<42:45,  3.65s/it]
generating images:  10%|▉         | 78/781 [04:47<43:02,  3.67s/it][Agenerating images:  10%|█         | 79/781 [04:50<42:40,  3.65s/it]
generating images:  10%|█         | 79/781 [04:51<42:57,  3.67s/it][Agenerating images:  10%|█         | 80/781 [04:53<42:39,  3.65s/it]
generating images:  10%|█         | 80/781 [04:55<42:53,  3.67s/it][Agenerating images:  10%|█         | 81/781 [04:57<42:37,  3.65s/it]
generating images:  10%|█         | 81/781 [04:58<42:50,  3.67s/it][Agenerating images:  10%|█         | 82/781 [05:01<42:34,  3.65s/it]
generating images:  10%|█         | 82/781 [05:02<42:47,  3.67s/it][Agenerating images:  11%|█         | 83/781 [05:04<42:31,  3.66s/it]
generating images:  11%|█         | 83/781 [05:06<42:42,  3.67s/it][Agenerating images:  11%|█         | 84/781 [05:08<42:27,  3.65s/it]
generating images:  11%|█         | 84/781 [05:09<42:38,  3.67s/it][Agenerating images:  11%|█         | 85/781 [05:12<42:22,  3.65s/it]
generating images:  11%|█         | 85/781 [05:13<42:35,  3.67s/it][Agenerating images:  11%|█         | 86/781 [05:15<42:20,  3.66s/it]
generating images:  11%|█         | 86/781 [05:17<42:33,  3.67s/it][Agenerating images:  11%|█         | 87/781 [05:19<42:17,  3.66s/it]
generating images:  11%|█         | 87/781 [05:20<42:29,  3.67s/it][Agenerating images:  11%|█▏        | 88/781 [05:23<42:13,  3.66s/it]
generating images:  11%|█▏        | 88/781 [05:24<42:24,  3.67s/it][Agenerating images:  11%|█▏        | 89/781 [05:26<42:11,  3.66s/it]
generating images:  11%|█▏        | 89/781 [05:28<42:20,  3.67s/it][Agenerating images:  12%|█▏        | 90/781 [05:30<42:06,  3.66s/it]
generating images:  12%|█▏        | 90/781 [05:31<42:16,  3.67s/it][Agenerating images:  12%|█▏        | 91/781 [05:33<42:03,  3.66s/it]
generating images:  12%|█▏        | 91/781 [05:35<42:14,  3.67s/it][Agenerating images:  12%|█▏        | 92/781 [05:37<41:59,  3.66s/it]
generating images:  12%|█▏        | 92/781 [05:39<42:10,  3.67s/it][Agenerating images:  12%|█▏        | 93/781 [05:41<41:56,  3.66s/it]
generating images:  12%|█▏        | 93/781 [05:42<42:06,  3.67s/it][Agenerating images:  12%|█▏        | 94/781 [05:44<41:52,  3.66s/it]
generating images:  12%|█▏        | 94/781 [05:46<42:03,  3.67s/it][Agenerating images:  12%|█▏        | 95/781 [05:48<41:48,  3.66s/it]
generating images:  12%|█▏        | 95/781 [05:50<41:55,  3.67s/it][Agenerating images:  12%|█▏        | 96/781 [05:52<41:45,  3.66s/it]
generating images:  12%|█▏        | 96/781 [05:53<41:48,  3.66s/it][Agenerating images:  12%|█▏        | 97/781 [05:55<41:40,  3.66s/it]
generating images:  12%|█▏        | 97/781 [05:57<41:43,  3.66s/it][Agenerating images:  13%|█▎        | 98/781 [05:59<41:36,  3.65s/it]
generating images:  13%|█▎        | 98/781 [06:01<41:38,  3.66s/it][Agenerating images:  13%|█▎        | 99/781 [06:03<41:32,  3.65s/it]
generating images:  13%|█▎        | 99/781 [06:04<41:34,  3.66s/it][Agenerating images:  13%|█▎        | 100/781 [06:06<41:28,  3.65s/it]
generating images:  13%|█▎        | 100/781 [06:08<41:30,  3.66s/it][Agenerating images:  13%|█▎        | 101/781 [06:10<41:25,  3.66s/it]
generating images:  13%|█▎        | 101/781 [06:12<41:30,  3.66s/it][Agenerating images:  13%|█▎        | 102/781 [06:14<41:22,  3.66s/it]
generating images:  13%|█▎        | 102/781 [06:15<41:28,  3.66s/it][Agenerating images:  13%|█▎        | 103/781 [06:17<41:19,  3.66s/it]
generating images:  13%|█▎        | 103/781 [06:19<41:24,  3.67s/it][Agenerating images:  13%|█▎        | 104/781 [06:21<41:15,  3.66s/it]
generating images:  13%|█▎        | 104/781 [06:23<41:22,  3.67s/it][Agenerating images:  13%|█▎        | 105/781 [06:25<41:12,  3.66s/it]
generating images:  13%|█▎        | 105/781 [06:26<41:19,  3.67s/it][Agenerating images:  14%|█▎        | 106/781 [06:28<41:09,  3.66s/it]
generating images:  14%|█▎        | 106/781 [06:30<41:15,  3.67s/it][Agenerating images:  14%|█▎        | 107/781 [06:32<41:04,  3.66s/it]
generating images:  14%|█▎        | 107/781 [06:34<41:10,  3.67s/it][Agenerating images:  14%|█▍        | 108/781 [06:36<41:00,  3.66s/it]
generating images:  14%|█▍        | 108/781 [06:37<41:07,  3.67s/it][Agenerating images:  14%|█▍        | 109/781 [06:39<40:56,  3.66s/it]
generating images:  14%|█▍        | 109/781 [06:41<41:05,  3.67s/it][Agenerating images:  14%|█▍        | 110/781 [06:43<40:54,  3.66s/it]
generating images:  14%|█▍        | 110/781 [06:45<41:02,  3.67s/it][Agenerating images:  14%|█▍        | 111/781 [06:47<40:50,  3.66s/it]
generating images:  14%|█▍        | 111/781 [06:48<40:59,  3.67s/it][Agenerating images:  14%|█▍        | 112/781 [06:50<40:46,  3.66s/it]
generating images:  14%|█▍        | 112/781 [06:52<40:55,  3.67s/it][Agenerating images:  14%|█▍        | 113/781 [06:54<40:42,  3.66s/it]
generating images:  14%|█▍        | 113/781 [06:56<40:52,  3.67s/it][Agenerating images:  15%|█▍        | 114/781 [06:58<40:38,  3.66s/it]
generating images:  15%|█▍        | 114/781 [06:59<40:48,  3.67s/it][Agenerating images:  15%|█▍        | 115/781 [07:01<40:35,  3.66s/it]
generating images:  15%|█▍        | 115/781 [07:03<40:45,  3.67s/it][Agenerating images:  15%|█▍        | 116/781 [07:05<40:31,  3.66s/it]
generating images:  15%|█▍        | 116/781 [07:07<40:41,  3.67s/it][Agenerating images:  15%|█▍        | 117/781 [07:09<40:29,  3.66s/it]
generating images:  15%|█▍        | 117/781 [07:10<40:37,  3.67s/it][Agenerating images:  15%|█▌        | 118/781 [07:12<40:25,  3.66s/it]
generating images:  15%|█▌        | 118/781 [07:14<40:34,  3.67s/it][Agenerating images:  15%|█▌        | 119/781 [07:16<40:21,  3.66s/it]
generating images:  15%|█▌        | 119/781 [07:18<40:29,  3.67s/it][Agenerating images:  15%|█▌        | 120/781 [07:20<40:18,  3.66s/it]
generating images:  15%|█▌        | 120/781 [07:21<40:25,  3.67s/it][Agenerating images:  15%|█▌        | 121/781 [07:23<40:13,  3.66s/it]
generating images:  15%|█▌        | 121/781 [07:25<40:21,  3.67s/it][Agenerating images:  16%|█▌        | 122/781 [07:27<40:09,  3.66s/it]
generating images:  16%|█▌        | 122/781 [07:29<40:18,  3.67s/it][Agenerating images:  16%|█▌        | 123/781 [07:31<40:05,  3.66s/it]
generating images:  16%|█▌        | 123/781 [07:32<40:13,  3.67s/it][Agenerating images:  16%|█▌        | 124/781 [07:34<40:01,  3.66s/it]
generating images:  16%|█▌        | 124/781 [07:36<40:10,  3.67s/it][Agenerating images:  16%|█▌        | 125/781 [07:38<39:58,  3.66s/it]
generating images:  16%|█▌        | 125/781 [07:40<40:08,  3.67s/it][Agenerating images:  16%|█▌        | 126/781 [07:41<39:55,  3.66s/it]
generating images:  16%|█▌        | 126/781 [07:43<40:04,  3.67s/it][Agenerating images:  16%|█▋        | 127/781 [07:45<39:53,  3.66s/it]
generating images:  16%|█▋        | 127/781 [07:47<39:59,  3.67s/it][Agenerating images:  16%|█▋        | 128/781 [07:49<39:49,  3.66s/it]
generating images:  16%|█▋        | 128/781 [07:51<39:57,  3.67s/it][Agenerating images:  17%|█▋        | 129/781 [07:52<39:46,  3.66s/it]
generating images:  17%|█▋        | 129/781 [07:54<39:54,  3.67s/it][Agenerating images:  17%|█▋        | 130/781 [07:56<39:41,  3.66s/it]
generating images:  17%|█▋        | 130/781 [07:58<39:49,  3.67s/it][Agenerating images:  17%|█▋        | 131/781 [08:00<39:37,  3.66s/it]
generating images:  17%|█▋        | 131/781 [08:02<39:46,  3.67s/it][Agenerating images:  17%|█▋        | 132/781 [08:03<39:33,  3.66s/it]
generating images:  17%|█▋        | 132/781 [08:05<39:41,  3.67s/it][Agenerating images:  17%|█▋        | 133/781 [08:07<39:28,  3.65s/it]
generating images:  17%|█▋        | 133/781 [08:09<39:38,  3.67s/it][Agenerating images:  17%|█▋        | 134/781 [08:11<39:20,  3.65s/it]
generating images:  17%|█▋        | 134/781 [08:13<39:35,  3.67s/it][Agenerating images:  17%|█▋        | 135/781 [08:14<39:15,  3.65s/it]
generating images:  17%|█▋        | 135/781 [08:16<39:31,  3.67s/it][Agenerating images:  17%|█▋        | 136/781 [08:18<39:09,  3.64s/it]
generating images:  17%|█▋        | 136/781 [08:20<39:27,  3.67s/it][Agenerating images:  18%|█▊        | 137/781 [08:22<39:09,  3.65s/it]
generating images:  18%|█▊        | 137/781 [08:24<39:25,  3.67s/it][Agenerating images:  18%|█▊        | 138/781 [08:25<39:02,  3.64s/it]
generating images:  18%|█▊        | 138/781 [08:28<39:21,  3.67s/it][Agenerating images:  18%|█▊        | 139/781 [08:29<39:02,  3.65s/it]
generating images:  18%|█▊        | 139/781 [08:31<39:17,  3.67s/it][Agenerating images:  18%|█▊        | 140/781 [08:33<38:56,  3.65s/it]
generating images:  18%|█▊        | 140/781 [08:35<39:13,  3.67s/it][Agenerating images:  18%|█▊        | 141/781 [08:36<38:55,  3.65s/it]
generating images:  18%|█▊        | 141/781 [08:39<39:09,  3.67s/it][Agenerating images:  18%|█▊        | 142/781 [08:40<38:52,  3.65s/it]
generating images:  18%|█▊        | 142/781 [08:42<39:05,  3.67s/it][Agenerating images:  18%|█▊        | 143/781 [08:44<38:50,  3.65s/it]
generating images:  18%|█▊        | 143/781 [08:46<39:01,  3.67s/it][Agenerating images:  18%|█▊        | 144/781 [08:47<38:47,  3.65s/it]
generating images:  18%|█▊        | 144/781 [08:50<38:54,  3.66s/it][Agenerating images:  19%|█▊        | 145/781 [08:51<38:44,  3.66s/it]
generating images:  19%|█▊        | 145/781 [08:53<38:52,  3.67s/it][Agenerating images:  19%|█▊        | 146/781 [08:55<38:41,  3.66s/it]
generating images:  19%|█▊        | 146/781 [08:57<38:53,  3.67s/it][Agenerating images:  19%|█▉        | 147/781 [08:58<38:38,  3.66s/it]
generating images:  19%|█▉        | 147/781 [09:01<38:46,  3.67s/it][Agenerating images:  19%|█▉        | 148/781 [09:02<38:34,  3.66s/it]
generating images:  19%|█▉        | 148/781 [09:04<38:43,  3.67s/it][Agenerating images:  19%|█▉        | 149/781 [09:05<38:30,  3.66s/it]
generating images:  19%|█▉        | 149/781 [09:08<38:41,  3.67s/it][Agenerating images:  19%|█▉        | 150/781 [09:09<38:28,  3.66s/it]
generating images:  19%|█▉        | 150/781 [09:12<38:38,  3.67s/it][Agenerating images:  19%|█▉        | 151/781 [09:13<38:24,  3.66s/it]
generating images:  19%|█▉        | 151/781 [09:15<38:34,  3.67s/it][Agenerating images:  19%|█▉        | 152/781 [09:16<38:20,  3.66s/it]
generating images:  19%|█▉        | 152/781 [09:19<38:30,  3.67s/it][Agenerating images:  20%|█▉        | 153/781 [09:20<38:15,  3.66s/it]
generating images:  20%|█▉        | 153/781 [09:23<38:27,  3.67s/it][Agenerating images:  20%|█▉        | 154/781 [09:24<38:11,  3.66s/it]
generating images:  20%|█▉        | 154/781 [09:26<38:24,  3.67s/it][Agenerating images:  20%|█▉        | 155/781 [09:27<38:07,  3.65s/it]
generating images:  20%|█▉        | 155/781 [09:30<38:20,  3.67s/it][Agenerating images:  20%|█▉        | 156/781 [09:31<38:03,  3.65s/it]
generating images:  20%|█▉        | 156/781 [09:34<38:16,  3.67s/it][Agenerating images:  20%|██        | 157/781 [09:35<37:59,  3.65s/it]
generating images:  20%|██        | 157/781 [09:37<38:13,  3.68s/it][Agenerating images:  20%|██        | 158/781 [09:38<37:57,  3.66s/it]
generating images:  20%|██        | 158/781 [09:41<38:06,  3.67s/it][Agenerating images:  20%|██        | 159/781 [09:42<37:53,  3.66s/it]
generating images:  20%|██        | 159/781 [09:45<38:02,  3.67s/it][Agenerating images:  20%|██        | 160/781 [09:46<37:50,  3.66s/it]
generating images:  20%|██        | 160/781 [09:48<37:59,  3.67s/it][Agenerating images:  21%|██        | 161/781 [09:49<37:46,  3.65s/it]
generating images:  21%|██        | 161/781 [09:52<37:56,  3.67s/it][Agenerating images:  21%|██        | 162/781 [09:53<37:43,  3.66s/it]
generating images:  21%|██        | 162/781 [09:56<37:52,  3.67s/it][Agenerating images:  21%|██        | 163/781 [09:57<37:40,  3.66s/it]
generating images:  21%|██        | 163/781 [09:59<37:48,  3.67s/it][Agenerating images:  21%|██        | 164/781 [10:00<37:36,  3.66s/it]
generating images:  21%|██        | 164/781 [10:03<37:45,  3.67s/it][Agenerating images:  21%|██        | 165/781 [10:04<37:32,  3.66s/it]
generating images:  21%|██        | 165/781 [10:07<37:42,  3.67s/it][Agenerating images:  21%|██▏       | 166/781 [10:08<37:29,  3.66s/it]
generating images:  21%|██▏       | 166/781 [10:10<37:35,  3.67s/it][Agenerating images:  21%|██▏       | 167/781 [10:11<37:25,  3.66s/it]
generating images:  21%|██▏       | 167/781 [10:14<37:32,  3.67s/it][Agenerating images:  22%|██▏       | 168/781 [10:15<37:21,  3.66s/it]
generating images:  22%|██▏       | 168/781 [10:18<37:29,  3.67s/it][Agenerating images:  22%|██▏       | 169/781 [10:19<37:16,  3.65s/it]
generating images:  22%|██▏       | 169/781 [10:21<37:26,  3.67s/it][Agenerating images:  22%|██▏       | 170/781 [10:22<37:12,  3.65s/it]
generating images:  22%|██▏       | 170/781 [10:25<37:22,  3.67s/it][Agenerating images:  22%|██▏       | 171/781 [10:26<37:09,  3.66s/it]
generating images:  22%|██▏       | 171/781 [10:29<37:16,  3.67s/it][Agenerating images:  22%|██▏       | 172/781 [10:30<37:06,  3.66s/it]
generating images:  22%|██▏       | 172/781 [10:32<37:14,  3.67s/it][Agenerating images:  22%|██▏       | 173/781 [10:33<37:01,  3.65s/it]
generating images:  22%|██▏       | 173/781 [10:36<37:10,  3.67s/it][Agenerating images:  22%|██▏       | 174/781 [10:37<36:58,  3.65s/it]
generating images:  22%|██▏       | 174/781 [10:40<37:07,  3.67s/it][Agenerating images:  22%|██▏       | 175/781 [10:41<36:55,  3.66s/it]
generating images:  22%|██▏       | 175/781 [10:43<37:03,  3.67s/it][Agenerating images:  23%|██▎       | 176/781 [10:44<36:51,  3.66s/it]
generating images:  23%|██▎       | 176/781 [10:47<36:59,  3.67s/it][Agenerating images:  23%|██▎       | 177/781 [10:48<36:47,  3.66s/it]
generating images:  23%|██▎       | 177/781 [10:51<36:52,  3.66s/it][Agenerating images:  23%|██▎       | 178/781 [10:52<36:43,  3.65s/it]
generating images:  23%|██▎       | 178/781 [10:54<36:50,  3.67s/it][Agenerating images:  23%|██▎       | 179/781 [10:55<36:41,  3.66s/it]
generating images:  23%|██▎       | 179/781 [10:58<36:46,  3.67s/it][Agenerating images:  23%|██▎       | 180/781 [10:59<36:37,  3.66s/it]
generating images:  23%|██▎       | 180/781 [11:02<36:43,  3.67s/it][Agenerating images:  23%|██▎       | 181/781 [11:02<36:33,  3.66s/it]
generating images:  23%|██▎       | 181/781 [11:05<36:40,  3.67s/it][Agenerating images:  23%|██▎       | 182/781 [11:06<36:29,  3.66s/it]
generating images:  23%|██▎       | 182/781 [11:09<36:36,  3.67s/it][Agenerating images:  23%|██▎       | 183/781 [11:10<36:26,  3.66s/it]
generating images:  23%|██▎       | 183/781 [11:13<36:30,  3.66s/it][Agenerating images:  24%|██▎       | 184/781 [11:13<36:23,  3.66s/it]
generating images:  24%|██▎       | 184/781 [11:16<36:28,  3.67s/it][Agenerating images:  24%|██▎       | 185/781 [11:17<36:19,  3.66s/it]
generating images:  24%|██▎       | 185/781 [11:20<36:26,  3.67s/it][Agenerating images:  24%|██▍       | 186/781 [11:21<36:15,  3.66s/it]
generating images:  24%|██▍       | 186/781 [11:24<36:22,  3.67s/it][Agenerating images:  24%|██▍       | 187/781 [11:24<36:11,  3.66s/it]
generating images:  24%|██▍       | 187/781 [11:27<36:20,  3.67s/it][Agenerating images:  24%|██▍       | 188/781 [11:28<36:07,  3.65s/it]
generating images:  24%|██▍       | 188/781 [11:31<36:16,  3.67s/it][Agenerating images:  24%|██▍       | 189/781 [11:32<36:03,  3.65s/it]
generating images:  24%|██▍       | 189/781 [11:35<36:09,  3.67s/it][Agenerating images:  24%|██▍       | 190/781 [11:35<35:59,  3.65s/it]
generating images:  24%|██▍       | 190/781 [11:38<36:04,  3.66s/it][Agenerating images:  24%|██▍       | 191/781 [11:39<35:57,  3.66s/it]
generating images:  24%|██▍       | 191/781 [11:42<36:01,  3.66s/it][Agenerating images:  25%|██▍       | 192/781 [11:43<35:54,  3.66s/it]
generating images:  25%|██▍       | 192/781 [11:46<35:59,  3.67s/it][Agenerating images:  25%|██▍       | 193/781 [11:46<35:50,  3.66s/it]
generating images:  25%|██▍       | 193/781 [11:49<35:57,  3.67s/it][Agenerating images:  25%|██▍       | 194/781 [11:50<35:46,  3.66s/it]
generating images:  25%|██▍       | 194/781 [11:53<35:54,  3.67s/it][Agenerating images:  25%|██▍       | 195/781 [11:54<35:43,  3.66s/it]
generating images:  25%|██▍       | 195/781 [11:57<35:51,  3.67s/it][Agenerating images:  25%|██▌       | 196/781 [11:57<35:40,  3.66s/it]
generating images:  25%|██▌       | 196/781 [12:00<35:47,  3.67s/it][Agenerating images:  25%|██▌       | 197/781 [12:01<35:36,  3.66s/it]
generating images:  25%|██▌       | 197/781 [12:04<35:44,  3.67s/it][Agenerating images:  25%|██▌       | 198/781 [12:05<35:32,  3.66s/it]
generating images:  25%|██▌       | 198/781 [12:08<35:40,  3.67s/it][Agenerating images:  25%|██▌       | 199/781 [12:08<35:28,  3.66s/it]
generating images:  25%|██▌       | 199/781 [12:11<35:36,  3.67s/it][Agenerating images:  26%|██▌       | 200/781 [12:12<35:24,  3.66s/it]
generating images:  26%|██▌       | 200/781 [12:15<35:32,  3.67s/it][Agenerating images:  26%|██▌       | 201/781 [12:16<35:20,  3.66s/it]
generating images:  26%|██▌       | 201/781 [12:19<35:29,  3.67s/it][Agenerating images:  26%|██▌       | 202/781 [12:19<35:16,  3.66s/it]
generating images:  26%|██▌       | 202/781 [12:22<35:26,  3.67s/it][Agenerating images:  26%|██▌       | 203/781 [12:23<35:12,  3.66s/it]
generating images:  26%|██▌       | 203/781 [12:26<35:21,  3.67s/it][Agenerating images:  26%|██▌       | 204/781 [12:27<35:10,  3.66s/it]
generating images:  26%|██▌       | 204/781 [12:30<35:18,  3.67s/it][Agenerating images:  26%|██▌       | 205/781 [12:30<35:06,  3.66s/it]
generating images:  26%|██▌       | 205/781 [12:33<35:15,  3.67s/it][Agenerating images:  26%|██▋       | 206/781 [12:34<35:04,  3.66s/it]
generating images:  26%|██▋       | 206/781 [12:37<35:11,  3.67s/it][Agenerating images:  27%|██▋       | 207/781 [12:38<35:00,  3.66s/it]
generating images:  27%|██▋       | 207/781 [12:41<35:06,  3.67s/it][Agenerating images:  27%|██▋       | 208/781 [12:41<34:56,  3.66s/it]
generating images:  27%|██▋       | 208/781 [12:44<35:03,  3.67s/it][Agenerating images:  27%|██▋       | 209/781 [12:45<34:52,  3.66s/it]
generating images:  27%|██▋       | 209/781 [12:48<34:59,  3.67s/it][Agenerating images:  27%|██▋       | 210/781 [12:49<34:48,  3.66s/it]
generating images:  27%|██▋       | 210/781 [12:52<34:56,  3.67s/it][Agenerating images:  27%|██▋       | 211/781 [12:52<34:43,  3.66s/it]
generating images:  27%|██▋       | 211/781 [12:55<34:54,  3.67s/it][Agenerating images:  27%|██▋       | 212/781 [12:56<34:40,  3.66s/it]
generating images:  27%|██▋       | 212/781 [12:59<34:46,  3.67s/it][Agenerating images:  27%|██▋       | 213/781 [12:59<34:36,  3.66s/it]
generating images:  27%|██▋       | 213/781 [13:03<34:44,  3.67s/it][Agenerating images:  27%|██▋       | 214/781 [13:03<34:31,  3.65s/it]
generating images:  27%|██▋       | 214/781 [13:06<34:40,  3.67s/it][Agenerating images:  28%|██▊       | 215/781 [13:07<34:28,  3.65s/it]
generating images:  28%|██▊       | 215/781 [13:10<34:36,  3.67s/it][Agenerating images:  28%|██▊       | 216/781 [13:10<34:25,  3.66s/it]
generating images:  28%|██▊       | 216/781 [13:14<34:33,  3.67s/it][Agenerating images:  28%|██▊       | 217/781 [13:14<34:20,  3.65s/it]
generating images:  28%|██▊       | 217/781 [13:17<34:29,  3.67s/it][Agenerating images:  28%|██▊       | 218/781 [13:18<34:14,  3.65s/it]
generating images:  28%|██▊       | 218/781 [13:21<34:26,  3.67s/it][Agenerating images:  28%|██▊       | 219/781 [13:21<34:12,  3.65s/it]
generating images:  28%|██▊       | 219/781 [13:25<34:22,  3.67s/it][Agenerating images:  28%|██▊       | 220/781 [13:25<34:10,  3.65s/it]
generating images:  28%|██▊       | 220/781 [13:28<34:18,  3.67s/it][Agenerating images:  28%|██▊       | 221/781 [13:29<34:06,  3.66s/it]
generating images:  28%|██▊       | 221/781 [13:32<34:15,  3.67s/it][Agenerating images:  28%|██▊       | 222/781 [13:32<34:04,  3.66s/it]
generating images:  28%|██▊       | 222/781 [13:36<34:12,  3.67s/it][Agenerating images:  29%|██▊       | 223/781 [13:36<33:59,  3.66s/it]
generating images:  29%|██▊       | 223/781 [13:39<34:08,  3.67s/it][Agenerating images:  29%|██▊       | 224/781 [13:40<33:55,  3.65s/it]
generating images:  29%|██▊       | 224/781 [13:43<34:03,  3.67s/it][Agenerating images:  29%|██▉       | 225/781 [13:43<33:50,  3.65s/it]
generating images:  29%|██▉       | 225/781 [13:47<34:00,  3.67s/it][Agenerating images:  29%|██▉       | 226/781 [13:47<33:44,  3.65s/it]
generating images:  29%|██▉       | 226/781 [13:50<33:56,  3.67s/it][Agenerating images:  29%|██▉       | 227/781 [13:51<33:42,  3.65s/it]
generating images:  29%|██▉       | 227/781 [13:54<33:53,  3.67s/it][Agenerating images:  29%|██▉       | 228/781 [13:54<33:39,  3.65s/it]
generating images:  29%|██▉       | 228/781 [13:58<33:49,  3.67s/it][Agenerating images:  29%|██▉       | 229/781 [13:58<33:36,  3.65s/it]
generating images:  29%|██▉       | 229/781 [14:01<33:45,  3.67s/it][Agenerating images:  29%|██▉       | 230/781 [14:02<33:33,  3.65s/it]
generating images:  29%|██▉       | 230/781 [14:05<33:42,  3.67s/it][Agenerating images:  30%|██▉       | 231/781 [14:05<33:30,  3.66s/it]
generating images:  30%|██▉       | 231/781 [14:09<33:38,  3.67s/it][Agenerating images:  30%|██▉       | 232/781 [14:09<33:28,  3.66s/it]
generating images:  30%|██▉       | 232/781 [14:12<33:33,  3.67s/it][Agenerating images:  30%|██▉       | 233/781 [14:13<33:24,  3.66s/it]
generating images:  30%|██▉       | 233/781 [14:16<33:28,  3.67s/it][Agenerating images:  30%|██▉       | 234/781 [14:16<33:20,  3.66s/it]
generating images:  30%|██▉       | 234/781 [14:20<33:26,  3.67s/it][Agenerating images:  30%|███       | 235/781 [14:20<33:16,  3.66s/it]
generating images:  30%|███       | 235/781 [14:23<33:23,  3.67s/it][Agenerating images:  30%|███       | 236/781 [14:24<33:12,  3.66s/it]
generating images:  30%|███       | 236/781 [14:27<33:19,  3.67s/it][Agenerating images:  30%|███       | 237/781 [14:27<33:08,  3.66s/it]
generating images:  30%|███       | 237/781 [14:31<33:16,  3.67s/it][Agenerating images:  30%|███       | 238/781 [14:31<33:05,  3.66s/it]
generating images:  30%|███       | 238/781 [14:34<33:12,  3.67s/it][Agenerating images:  31%|███       | 239/781 [14:35<33:01,  3.66s/it]
generating images:  31%|███       | 239/781 [14:38<33:09,  3.67s/it][Agenerating images:  31%|███       | 240/781 [14:38<32:59,  3.66s/it]generating images:  31%|███       | 241/781 [14:42<32:55,  3.66s/it]
generating images:  31%|███       | 240/781 [14:42<33:06,  3.67s/it][Agenerating images:  31%|███       | 242/781 [14:45<32:51,  3.66s/it]
generating images:  31%|███       | 241/781 [14:46<33:02,  3.67s/it][Agenerating images:  31%|███       | 243/781 [14:49<32:47,  3.66s/it]
generating images:  31%|███       | 242/781 [14:49<32:59,  3.67s/it][Agenerating images:  31%|███       | 244/781 [14:53<32:43,  3.66s/it]
generating images:  31%|███       | 243/781 [14:53<32:55,  3.67s/it][Agenerating images:  31%|███▏      | 245/781 [14:56<32:39,  3.66s/it]
generating images:  31%|███       | 244/781 [14:57<32:51,  3.67s/it][Agenerating images:  31%|███▏      | 246/781 [15:00<32:36,  3.66s/it]
generating images:  31%|███▏      | 245/781 [15:00<32:47,  3.67s/it][Agenerating images:  32%|███▏      | 247/781 [15:04<32:32,  3.66s/it]
generating images:  31%|███▏      | 246/781 [15:04<32:44,  3.67s/it][Agenerating images:  32%|███▏      | 248/781 [15:07<32:28,  3.66s/it]
generating images:  32%|███▏      | 247/781 [15:08<32:39,  3.67s/it][Agenerating images:  32%|███▏      | 249/781 [15:11<32:25,  3.66s/it]
generating images:  32%|███▏      | 248/781 [15:11<32:36,  3.67s/it][Agenerating images:  32%|███▏      | 250/781 [15:15<32:22,  3.66s/it]
generating images:  32%|███▏      | 249/781 [15:15<32:32,  3.67s/it][Agenerating images:  32%|███▏      | 251/781 [15:18<32:16,  3.65s/it]
generating images:  32%|███▏      | 250/781 [15:19<32:29,  3.67s/it][Agenerating images:  32%|███▏      | 252/781 [15:22<32:13,  3.66s/it]
generating images:  32%|███▏      | 251/781 [15:22<32:25,  3.67s/it][Agenerating images:  32%|███▏      | 253/781 [15:26<32:07,  3.65s/it]
generating images:  32%|███▏      | 252/781 [15:26<32:19,  3.67s/it][Agenerating images:  33%|███▎      | 254/781 [15:29<32:04,  3.65s/it]
generating images:  32%|███▏      | 253/781 [15:30<32:16,  3.67s/it][Agenerating images:  33%|███▎      | 255/781 [15:33<32:02,  3.65s/it]
generating images:  33%|███▎      | 254/781 [15:33<32:12,  3.67s/it][Agenerating images:  33%|███▎      | 256/781 [15:37<31:59,  3.66s/it]
generating images:  33%|███▎      | 255/781 [15:37<32:09,  3.67s/it][Agenerating images:  33%|███▎      | 257/781 [15:40<31:53,  3.65s/it]
generating images:  33%|███▎      | 256/781 [15:41<32:06,  3.67s/it][Agenerating images:  33%|███▎      | 258/781 [15:44<31:50,  3.65s/it]
generating images:  33%|███▎      | 257/781 [15:44<32:03,  3.67s/it][Agenerating images:  33%|███▎      | 259/781 [15:48<31:47,  3.65s/it]
generating images:  33%|███▎      | 258/781 [15:48<31:59,  3.67s/it][Agenerating images:  33%|███▎      | 260/781 [15:51<31:45,  3.66s/it]
generating images:  33%|███▎      | 259/781 [15:52<31:56,  3.67s/it][Agenerating images:  33%|███▎      | 261/781 [15:55<31:40,  3.66s/it]
generating images:  33%|███▎      | 260/781 [15:55<31:49,  3.67s/it][Agenerating images:  34%|███▎      | 262/781 [15:59<31:36,  3.65s/it]
generating images:  33%|███▎      | 261/781 [15:59<31:47,  3.67s/it][Agenerating images:  34%|███▎      | 263/781 [16:02<31:33,  3.66s/it]
generating images:  34%|███▎      | 262/781 [16:03<31:41,  3.66s/it][Agenerating images:  34%|███▍      | 264/781 [16:06<31:30,  3.66s/it]
generating images:  34%|███▎      | 263/781 [16:06<31:38,  3.67s/it][Agenerating images:  34%|███▍      | 265/781 [16:10<31:26,  3.66s/it]
generating images:  34%|███▍      | 264/781 [16:10<31:35,  3.67s/it][Agenerating images:  34%|███▍      | 266/781 [16:13<31:22,  3.66s/it]
generating images:  34%|███▍      | 265/781 [16:14<31:32,  3.67s/it][Agenerating images:  34%|███▍      | 267/781 [16:17<31:18,  3.66s/it]
generating images:  34%|███▍      | 266/781 [16:17<31:29,  3.67s/it][Agenerating images:  34%|███▍      | 268/781 [16:21<31:15,  3.66s/it]
generating images:  34%|███▍      | 267/781 [16:21<31:26,  3.67s/it][Agenerating images:  34%|███▍      | 269/781 [16:24<31:11,  3.65s/it]
generating images:  34%|███▍      | 268/781 [16:25<31:23,  3.67s/it][Agenerating images:  35%|███▍      | 270/781 [16:28<31:05,  3.65s/it]
generating images:  34%|███▍      | 269/781 [16:28<31:19,  3.67s/it][Agenerating images:  35%|███▍      | 271/781 [16:31<31:03,  3.65s/it]
generating images:  35%|███▍      | 270/781 [16:32<31:15,  3.67s/it][Agenerating images:  35%|███▍      | 272/781 [16:35<30:59,  3.65s/it]
generating images:  35%|███▍      | 271/781 [16:36<31:11,  3.67s/it][Agenerating images:  35%|███▍      | 273/781 [16:39<30:56,  3.65s/it]
generating images:  35%|███▍      | 272/781 [16:39<31:08,  3.67s/it][Agenerating images:  35%|███▌      | 274/781 [16:42<30:53,  3.66s/it]
generating images:  35%|███▍      | 273/781 [16:43<31:01,  3.66s/it][Agenerating images:  35%|███▌      | 275/781 [16:46<30:49,  3.66s/it]
generating images:  35%|███▌      | 274/781 [16:47<30:59,  3.67s/it][Agenerating images:  35%|███▌      | 276/781 [16:50<30:46,  3.66s/it]
generating images:  35%|███▌      | 275/781 [16:50<30:56,  3.67s/it][Agenerating images:  35%|███▌      | 277/781 [16:53<30:38,  3.65s/it]
generating images:  35%|███▌      | 276/781 [16:54<30:51,  3.67s/it][Agenerating images:  36%|███▌      | 278/781 [16:57<30:33,  3.64s/it]
generating images:  35%|███▌      | 277/781 [16:58<30:48,  3.67s/it][Agenerating images:  36%|███▌      | 279/781 [17:01<30:29,  3.64s/it]
generating images:  36%|███▌      | 278/781 [17:01<30:45,  3.67s/it][Agenerating images:  36%|███▌      | 280/781 [17:04<30:27,  3.65s/it]
generating images:  36%|███▌      | 279/781 [17:05<30:42,  3.67s/it][Agenerating images:  36%|███▌      | 281/781 [17:08<30:22,  3.65s/it]
generating images:  36%|███▌      | 280/781 [17:09<30:39,  3.67s/it][Agenerating images:  36%|███▌      | 282/781 [17:12<30:20,  3.65s/it]
generating images:  36%|███▌      | 281/781 [17:12<30:35,  3.67s/it][Agenerating images:  36%|███▌      | 283/781 [17:15<30:18,  3.65s/it]
generating images:  36%|███▌      | 282/781 [17:16<30:32,  3.67s/it][Agenerating images:  36%|███▋      | 284/781 [17:19<30:15,  3.65s/it]
generating images:  36%|███▌      | 283/781 [17:20<30:28,  3.67s/it][Agenerating images:  36%|███▋      | 285/781 [17:23<30:12,  3.65s/it]
generating images:  36%|███▋      | 284/781 [17:23<30:22,  3.67s/it][Agenerating images:  37%|███▋      | 286/781 [17:26<30:10,  3.66s/it]
generating images:  36%|███▋      | 285/781 [17:27<30:19,  3.67s/it][Agenerating images:  37%|███▋      | 287/781 [17:30<30:06,  3.66s/it]
generating images:  37%|███▋      | 286/781 [17:31<30:17,  3.67s/it][Agenerating images:  37%|███▋      | 288/781 [17:34<30:02,  3.66s/it]
generating images:  37%|███▋      | 287/781 [17:34<30:13,  3.67s/it][Agenerating images:  37%|███▋      | 289/781 [17:37<29:58,  3.66s/it]
generating images:  37%|███▋      | 288/781 [17:38<30:09,  3.67s/it][Agenerating images:  37%|███▋      | 290/781 [17:41<29:54,  3.66s/it]
generating images:  37%|███▋      | 289/781 [17:42<30:05,  3.67s/it][Agenerating images:  37%|███▋      | 291/781 [17:45<29:50,  3.65s/it]
generating images:  37%|███▋      | 290/781 [17:45<30:02,  3.67s/it][Agenerating images:  37%|███▋      | 292/781 [17:48<29:47,  3.65s/it]
generating images:  37%|███▋      | 291/781 [17:49<29:58,  3.67s/it][Agenerating images:  38%|███▊      | 293/781 [17:52<29:44,  3.66s/it]
generating images:  37%|███▋      | 292/781 [17:53<29:55,  3.67s/it][Agenerating images:  38%|███▊      | 294/781 [17:55<29:40,  3.66s/it]
generating images:  38%|███▊      | 293/781 [17:56<29:49,  3.67s/it][Agenerating images:  38%|███▊      | 295/781 [17:59<29:36,  3.66s/it]
generating images:  38%|███▊      | 294/781 [18:00<29:44,  3.66s/it][Agenerating images:  38%|███▊      | 296/781 [18:03<29:32,  3.66s/it]
generating images:  38%|███▊      | 295/781 [18:04<29:41,  3.66s/it][Agenerating images:  38%|███▊      | 297/781 [18:06<29:30,  3.66s/it]
generating images:  38%|███▊      | 296/781 [18:07<29:35,  3.66s/it][Agenerating images:  38%|███▊      | 298/781 [18:10<29:24,  3.65s/it]
generating images:  38%|███▊      | 297/781 [18:11<29:33,  3.66s/it][Agenerating images:  38%|███▊      | 299/781 [18:14<29:22,  3.66s/it]
generating images:  38%|███▊      | 298/781 [18:15<29:30,  3.67s/it][Agenerating images:  38%|███▊      | 300/781 [18:17<29:17,  3.65s/it]
generating images:  38%|███▊      | 299/781 [18:18<29:28,  3.67s/it][Agenerating images:  39%|███▊      | 301/781 [18:21<29:13,  3.65s/it]
generating images:  38%|███▊      | 300/781 [18:22<29:22,  3.66s/it][Agenerating images:  39%|███▊      | 302/781 [18:25<29:10,  3.66s/it]
generating images:  39%|███▊      | 301/781 [18:26<29:16,  3.66s/it][Agenerating images:  39%|███▉      | 303/781 [18:28<29:07,  3.66s/it]
generating images:  39%|███▊      | 302/781 [18:29<29:11,  3.66s/it][Agenerating images:  39%|███▉      | 304/781 [18:32<29:03,  3.66s/it]
generating images:  39%|███▉      | 303/781 [18:33<29:08,  3.66s/it][Agenerating images:  39%|███▉      | 305/781 [18:36<29:00,  3.66s/it]
generating images:  39%|███▉      | 304/781 [18:37<29:03,  3.66s/it][Agenerating images:  39%|███▉      | 306/781 [18:39<28:56,  3.66s/it]
generating images:  39%|███▉      | 305/781 [18:40<29:00,  3.66s/it][Agenerating images:  39%|███▉      | 307/781 [18:43<28:53,  3.66s/it]
generating images:  39%|███▉      | 306/781 [18:44<28:56,  3.66s/it][Agenerating images:  39%|███▉      | 308/781 [18:47<28:51,  3.66s/it]
generating images:  39%|███▉      | 307/781 [18:48<28:55,  3.66s/it][Agenerating images:  40%|███▉      | 309/781 [18:50<28:47,  3.66s/it]
generating images:  39%|███▉      | 308/781 [18:51<28:53,  3.66s/it][Agenerating images:  40%|███▉      | 310/781 [18:54<28:42,  3.66s/it]
generating images:  40%|███▉      | 309/781 [18:55<28:50,  3.67s/it][Agenerating images:  40%|███▉      | 311/781 [18:58<28:38,  3.66s/it]
generating images:  40%|███▉      | 310/781 [18:59<28:47,  3.67s/it][Agenerating images:  40%|███▉      | 312/781 [19:01<28:34,  3.66s/it]
generating images:  40%|███▉      | 311/781 [19:02<28:43,  3.67s/it][Agenerating images:  40%|████      | 313/781 [19:05<28:30,  3.66s/it]
generating images:  40%|███▉      | 312/781 [19:06<28:40,  3.67s/it][Agenerating images:  40%|████      | 314/781 [19:09<28:26,  3.65s/it]
generating images:  40%|████      | 313/781 [19:10<28:35,  3.66s/it][Agenerating images:  40%|████      | 315/781 [19:12<28:23,  3.65s/it]
generating images:  40%|████      | 314/781 [19:13<28:33,  3.67s/it][Agenerating images:  40%|████      | 316/781 [19:16<28:17,  3.65s/it]
generating images:  40%|████      | 315/781 [19:17<28:30,  3.67s/it][Agenerating images:  41%|████      | 317/781 [19:20<28:14,  3.65s/it]
generating images:  40%|████      | 316/781 [19:21<28:27,  3.67s/it][Agenerating images:  41%|████      | 318/781 [19:23<28:08,  3.65s/it]
generating images:  41%|████      | 317/781 [19:24<28:23,  3.67s/it][Agenerating images:  41%|████      | 319/781 [19:27<28:05,  3.65s/it]
generating images:  41%|████      | 318/781 [19:28<28:19,  3.67s/it][Agenerating images:  41%|████      | 320/781 [19:30<28:01,  3.65s/it]
generating images:  41%|████      | 319/781 [19:32<28:15,  3.67s/it][Agenerating images:  41%|████      | 321/781 [19:34<27:59,  3.65s/it]
generating images:  41%|████      | 320/781 [19:35<28:11,  3.67s/it][Agenerating images:  41%|████      | 322/781 [19:38<27:54,  3.65s/it]
generating images:  41%|████      | 321/781 [19:39<28:08,  3.67s/it][Agenerating images:  41%|████▏     | 323/781 [19:41<27:49,  3.64s/it]
generating images:  41%|████      | 322/781 [19:43<28:05,  3.67s/it][Agenerating images:  41%|████▏     | 324/781 [19:45<27:45,  3.64s/it]
generating images:  41%|████▏     | 323/781 [19:46<28:01,  3.67s/it][Agenerating images:  42%|████▏     | 325/781 [19:49<27:43,  3.65s/it]
generating images:  41%|████▏     | 324/781 [19:50<27:57,  3.67s/it][Agenerating images:  42%|████▏     | 326/781 [19:52<27:38,  3.64s/it]
generating images:  42%|████▏     | 325/781 [19:54<27:53,  3.67s/it][Agenerating images:  42%|████▏     | 327/781 [19:56<27:32,  3.64s/it]
generating images:  42%|████▏     | 326/781 [19:57<27:49,  3.67s/it][Agenerating images:  42%|████▏     | 328/781 [20:00<27:31,  3.65s/it]
generating images:  42%|████▏     | 327/781 [20:01<27:46,  3.67s/it][Agenerating images:  42%|████▏     | 329/781 [20:03<27:29,  3.65s/it]
generating images:  42%|████▏     | 328/781 [20:05<27:40,  3.67s/it][Agenerating images:  42%|████▏     | 330/781 [20:07<27:24,  3.65s/it]
generating images:  42%|████▏     | 329/781 [20:08<27:37,  3.67s/it][Agenerating images:  42%|████▏     | 331/781 [20:11<27:20,  3.64s/it]
generating images:  42%|████▏     | 330/781 [20:12<27:34,  3.67s/it][Agenerating images:  43%|████▎     | 332/781 [20:14<27:17,  3.65s/it]
generating images:  42%|████▏     | 331/781 [20:16<27:31,  3.67s/it][Agenerating images:  43%|████▎     | 333/781 [20:18<27:15,  3.65s/it]
generating images:  43%|████▎     | 332/781 [20:19<27:26,  3.67s/it][Agenerating images:  43%|████▎     | 334/781 [20:22<27:11,  3.65s/it]
generating images:  43%|████▎     | 333/781 [20:23<27:23,  3.67s/it][Agenerating images:  43%|████▎     | 335/781 [20:25<27:08,  3.65s/it]
generating images:  43%|████▎     | 334/781 [20:27<27:20,  3.67s/it][Agenerating images:  43%|████▎     | 336/781 [20:29<27:05,  3.65s/it]
generating images:  43%|████▎     | 335/781 [20:30<27:16,  3.67s/it][Agenerating images:  43%|████▎     | 337/781 [20:33<27:02,  3.65s/it]
generating images:  43%|████▎     | 336/781 [20:34<27:12,  3.67s/it][Agenerating images:  43%|████▎     | 338/781 [20:36<26:59,  3.66s/it]
generating images:  43%|████▎     | 337/781 [20:38<27:10,  3.67s/it][Agenerating images:  43%|████▎     | 339/781 [20:40<26:53,  3.65s/it]
generating images:  43%|████▎     | 338/781 [20:41<27:07,  3.67s/it][Agenerating images:  44%|████▎     | 340/781 [20:43<26:50,  3.65s/it]
generating images:  43%|████▎     | 339/781 [20:45<27:03,  3.67s/it][Agenerating images:  44%|████▎     | 341/781 [20:47<26:47,  3.65s/it]
generating images:  44%|████▎     | 340/781 [20:49<27:00,  3.67s/it][Agenerating images:  44%|████▍     | 342/781 [20:51<26:44,  3.66s/it]
generating images:  44%|████▎     | 341/781 [20:52<26:56,  3.67s/it][Agenerating images:  44%|████▍     | 343/781 [20:54<26:41,  3.66s/it]
generating images:  44%|████▍     | 342/781 [20:56<26:50,  3.67s/it][Agenerating images:  44%|████▍     | 344/781 [20:58<26:38,  3.66s/it]
generating images:  44%|████▍     | 343/781 [21:00<26:47,  3.67s/it][Agenerating images:  44%|████▍     | 345/781 [21:02<26:34,  3.66s/it]
generating images:  44%|████▍     | 344/781 [21:03<26:43,  3.67s/it][Agenerating images:  44%|████▍     | 346/781 [21:05<26:30,  3.66s/it]
generating images:  44%|████▍     | 345/781 [21:07<26:41,  3.67s/it][Agenerating images:  44%|████▍     | 347/781 [21:09<26:26,  3.66s/it]
generating images:  44%|████▍     | 346/781 [21:11<26:37,  3.67s/it][Agenerating images:  45%|████▍     | 348/781 [21:13<26:22,  3.65s/it]
generating images:  44%|████▍     | 347/781 [21:14<26:33,  3.67s/it][Agenerating images:  45%|████▍     | 349/781 [21:16<26:18,  3.65s/it]
generating images:  45%|████▍     | 348/781 [21:18<26:29,  3.67s/it][Agenerating images:  45%|████▍     | 350/781 [21:20<26:15,  3.66s/it]
generating images:  45%|████▍     | 349/781 [21:22<26:26,  3.67s/it][Agenerating images:  45%|████▍     | 351/781 [21:24<26:12,  3.66s/it]
generating images:  45%|████▍     | 350/781 [21:25<26:22,  3.67s/it][Agenerating images:  45%|████▌     | 352/781 [21:27<26:08,  3.66s/it]
generating images:  45%|████▍     | 351/781 [21:29<26:18,  3.67s/it][Agenerating images:  45%|████▌     | 353/781 [21:31<26:05,  3.66s/it]
generating images:  45%|████▌     | 352/781 [21:33<26:14,  3.67s/it][Agenerating images:  45%|████▌     | 354/781 [21:35<26:01,  3.66s/it]
generating images:  45%|████▌     | 353/781 [21:36<26:10,  3.67s/it][Agenerating images:  45%|████▌     | 355/781 [21:38<25:57,  3.66s/it]
generating images:  45%|████▌     | 354/781 [21:40<26:06,  3.67s/it][Agenerating images:  46%|████▌     | 356/781 [21:42<25:53,  3.66s/it]
generating images:  45%|████▌     | 355/781 [21:44<26:02,  3.67s/it][Agenerating images:  46%|████▌     | 357/781 [21:46<25:49,  3.66s/it]
generating images:  46%|████▌     | 356/781 [21:47<25:59,  3.67s/it][Agenerating images:  46%|████▌     | 358/781 [21:49<25:44,  3.65s/it]
generating images:  46%|████▌     | 357/781 [21:51<25:56,  3.67s/it][Agenerating images:  46%|████▌     | 359/781 [21:53<25:41,  3.65s/it]
generating images:  46%|████▌     | 358/781 [21:55<25:51,  3.67s/it][Agenerating images:  46%|████▌     | 360/781 [21:57<25:37,  3.65s/it]
generating images:  46%|████▌     | 359/781 [21:58<25:48,  3.67s/it][Agenerating images:  46%|████▌     | 361/781 [22:00<25:34,  3.65s/it]
generating images:  46%|████▌     | 360/781 [22:02<25:45,  3.67s/it][Agenerating images:  46%|████▋     | 362/781 [22:04<25:28,  3.65s/it]
generating images:  46%|████▌     | 361/781 [22:06<25:41,  3.67s/it][Agenerating images:  46%|████▋     | 363/781 [22:08<25:25,  3.65s/it]
generating images:  46%|████▋     | 362/781 [22:09<25:38,  3.67s/it][Agenerating images:  47%|████▋     | 364/781 [22:11<25:23,  3.65s/it]
generating images:  46%|████▋     | 363/781 [22:13<25:34,  3.67s/it][Agenerating images:  47%|████▋     | 365/781 [22:15<25:20,  3.65s/it]
generating images:  47%|████▋     | 364/781 [22:17<25:30,  3.67s/it][Agenerating images:  47%|████▋     | 366/781 [22:18<25:17,  3.66s/it]
generating images:  47%|████▋     | 365/781 [22:20<25:26,  3.67s/it][Agenerating images:  47%|████▋     | 367/781 [22:22<25:13,  3.66s/it]
generating images:  47%|████▋     | 366/781 [22:24<25:23,  3.67s/it][Agenerating images:  47%|████▋     | 368/781 [22:26<25:09,  3.66s/it]
generating images:  47%|████▋     | 367/781 [22:28<25:19,  3.67s/it][Agenerating images:  47%|████▋     | 369/781 [22:29<25:05,  3.65s/it]
generating images:  47%|████▋     | 368/781 [22:31<25:15,  3.67s/it][Agenerating images:  47%|████▋     | 370/781 [22:33<25:02,  3.66s/it]
generating images:  47%|████▋     | 369/781 [22:35<25:12,  3.67s/it][Agenerating images:  48%|████▊     | 371/781 [22:37<24:59,  3.66s/it]
generating images:  47%|████▋     | 370/781 [22:39<25:05,  3.66s/it][Agenerating images:  48%|████▊     | 372/781 [22:40<24:55,  3.66s/it]
generating images:  48%|████▊     | 371/781 [22:42<25:03,  3.67s/it][Agenerating images:  48%|████▊     | 373/781 [22:44<24:52,  3.66s/it]
generating images:  48%|████▊     | 372/781 [22:46<24:59,  3.67s/it][Agenerating images:  48%|████▊     | 374/781 [22:48<24:48,  3.66s/it]
generating images:  48%|████▊     | 373/781 [22:50<24:56,  3.67s/it][Agenerating images:  48%|████▊     | 375/781 [22:51<24:46,  3.66s/it]
generating images:  48%|████▊     | 374/781 [22:53<24:53,  3.67s/it][Agenerating images:  48%|████▊     | 376/781 [22:55<24:43,  3.66s/it]
generating images:  48%|████▊     | 375/781 [22:57<24:49,  3.67s/it][Agenerating images:  48%|████▊     | 377/781 [22:59<24:39,  3.66s/it]
generating images:  48%|████▊     | 376/781 [23:01<24:45,  3.67s/it][Agenerating images:  48%|████▊     | 378/781 [23:02<24:33,  3.66s/it]
generating images:  48%|████▊     | 377/781 [23:04<24:42,  3.67s/it][Agenerating images:  49%|████▊     | 379/781 [23:06<24:29,  3.66s/it]
generating images:  48%|████▊     | 378/781 [23:08<24:38,  3.67s/it][Agenerating images:  49%|████▊     | 380/781 [23:10<24:25,  3.66s/it]
generating images:  49%|████▊     | 379/781 [23:12<24:34,  3.67s/it][Agenerating images:  49%|████▉     | 381/781 [23:13<24:21,  3.65s/it]
generating images:  49%|████▊     | 380/781 [23:15<24:31,  3.67s/it][Agenerating images:  49%|████▉     | 382/781 [23:17<24:18,  3.65s/it]
generating images:  49%|████▉     | 381/781 [23:19<24:27,  3.67s/it][Agenerating images:  49%|████▉     | 383/781 [23:21<24:14,  3.65s/it]
generating images:  49%|████▉     | 382/781 [23:23<24:24,  3.67s/it][Agenerating images:  49%|████▉     | 384/781 [23:24<24:11,  3.66s/it]
generating images:  49%|████▉     | 383/781 [23:26<24:20,  3.67s/it][Agenerating images:  49%|████▉     | 385/781 [23:28<24:08,  3.66s/it]
generating images:  49%|████▉     | 384/781 [23:30<24:17,  3.67s/it][Agenerating images:  49%|████▉     | 386/781 [23:32<24:02,  3.65s/it]
generating images:  49%|████▉     | 385/781 [23:34<24:13,  3.67s/it][Agenerating images:  50%|████▉     | 387/781 [23:35<23:57,  3.65s/it]
generating images:  49%|████▉     | 386/781 [23:37<24:10,  3.67s/it][Agenerating images:  50%|████▉     | 388/781 [23:39<23:55,  3.65s/it]
generating images:  50%|████▉     | 387/781 [23:41<24:06,  3.67s/it][Agenerating images:  50%|████▉     | 389/781 [23:43<23:51,  3.65s/it]
generating images:  50%|████▉     | 388/781 [23:45<24:03,  3.67s/it][Agenerating images:  50%|████▉     | 390/781 [23:46<23:48,  3.65s/it]
generating images:  50%|████▉     | 389/781 [23:48<23:59,  3.67s/it][Agenerating images:  50%|█████     | 391/781 [23:50<23:44,  3.65s/it]
generating images:  50%|████▉     | 390/781 [23:52<23:55,  3.67s/it][Agenerating images:  50%|█████     | 392/781 [23:54<23:41,  3.65s/it]
generating images:  50%|█████     | 391/781 [23:56<23:51,  3.67s/it][Agenerating images:  50%|█████     | 393/781 [23:57<23:38,  3.65s/it]
generating images:  50%|█████     | 392/781 [23:59<23:47,  3.67s/it][Agenerating images:  50%|█████     | 394/781 [24:01<23:35,  3.66s/it]
generating images:  50%|█████     | 393/781 [24:03<23:43,  3.67s/it][Agenerating images:  51%|█████     | 395/781 [24:05<23:32,  3.66s/it]
generating images:  50%|█████     | 394/781 [24:07<23:39,  3.67s/it][Agenerating images:  51%|█████     | 396/781 [24:08<23:25,  3.65s/it]
generating images:  51%|█████     | 395/781 [24:11<23:36,  3.67s/it][Agenerating images:  51%|█████     | 397/781 [24:12<23:22,  3.65s/it]
generating images:  51%|█████     | 396/781 [24:14<23:33,  3.67s/it][Agenerating images:  51%|█████     | 398/781 [24:15<23:19,  3.65s/it]
generating images:  51%|█████     | 397/781 [24:18<23:29,  3.67s/it][Agenerating images:  51%|█████     | 399/781 [24:19<23:16,  3.65s/it]
generating images:  51%|█████     | 398/781 [24:22<23:23,  3.67s/it][Agenerating images:  51%|█████     | 400/781 [24:23<23:12,  3.66s/it]
generating images:  51%|█████     | 399/781 [24:25<23:21,  3.67s/it][Agenerating images:  51%|█████▏    | 401/781 [24:26<23:07,  3.65s/it]
generating images:  51%|█████     | 400/781 [24:29<23:17,  3.67s/it][Agenerating images:  51%|█████▏    | 402/781 [24:30<23:01,  3.65s/it]
generating images:  51%|█████▏    | 401/781 [24:33<23:14,  3.67s/it][Agenerating images:  52%|█████▏    | 403/781 [24:34<22:57,  3.64s/it]
generating images:  51%|█████▏    | 402/781 [24:36<23:10,  3.67s/it][Agenerating images:  52%|█████▏    | 404/781 [24:37<22:52,  3.64s/it]
generating images:  52%|█████▏    | 403/781 [24:40<23:06,  3.67s/it][Agenerating images:  52%|█████▏    | 405/781 [24:41<22:48,  3.64s/it]
generating images:  52%|█████▏    | 404/781 [24:44<23:02,  3.67s/it][Agenerating images:  52%|█████▏    | 406/781 [24:45<22:47,  3.65s/it]
generating images:  52%|█████▏    | 405/781 [24:47<22:57,  3.66s/it][Agenerating images:  52%|█████▏    | 407/781 [24:48<22:45,  3.65s/it]
generating images:  52%|█████▏    | 406/781 [24:51<22:52,  3.66s/it][Agenerating images:  52%|█████▏    | 408/781 [24:52<22:42,  3.65s/it]
generating images:  52%|█████▏    | 407/781 [24:55<22:50,  3.67s/it][Agenerating images:  52%|█████▏    | 409/781 [24:56<22:37,  3.65s/it]
generating images:  52%|█████▏    | 408/781 [24:58<22:47,  3.67s/it][Agenerating images:  52%|█████▏    | 410/781 [24:59<22:32,  3.65s/it]
generating images:  52%|█████▏    | 409/781 [25:02<22:44,  3.67s/it][Agenerating images:  53%|█████▎    | 411/781 [25:03<22:30,  3.65s/it]
generating images:  52%|█████▏    | 410/781 [25:06<22:40,  3.67s/it][Agenerating images:  53%|█████▎    | 412/781 [25:07<22:27,  3.65s/it]
generating images:  53%|█████▎    | 411/781 [25:09<22:37,  3.67s/it][Agenerating images:  53%|█████▎    | 413/781 [25:10<22:23,  3.65s/it]
generating images:  53%|█████▎    | 412/781 [25:13<22:34,  3.67s/it][Agenerating images:  53%|█████▎    | 414/781 [25:14<22:21,  3.65s/it]
generating images:  53%|█████▎    | 413/781 [25:17<22:31,  3.67s/it][Agenerating images:  53%|█████▎    | 415/781 [25:18<22:18,  3.66s/it]
generating images:  53%|█████▎    | 414/781 [25:20<22:27,  3.67s/it][Agenerating images:  53%|█████▎    | 416/781 [25:21<22:15,  3.66s/it]
generating images:  53%|█████▎    | 415/781 [25:24<22:23,  3.67s/it][Agenerating images:  53%|█████▎    | 417/781 [25:25<22:11,  3.66s/it]
generating images:  53%|█████▎    | 416/781 [25:28<22:19,  3.67s/it][Agenerating images:  54%|█████▎    | 418/781 [25:28<22:07,  3.66s/it]
generating images:  53%|█████▎    | 417/781 [25:31<22:16,  3.67s/it][Agenerating images:  54%|█████▎    | 419/781 [25:32<22:04,  3.66s/it]
generating images:  54%|█████▎    | 418/781 [25:35<22:12,  3.67s/it][Agenerating images:  54%|█████▍    | 420/781 [25:36<22:00,  3.66s/it]
generating images:  54%|█████▎    | 419/781 [25:39<22:09,  3.67s/it][Agenerating images:  54%|█████▍    | 421/781 [25:39<21:54,  3.65s/it]
generating images:  54%|█████▍    | 420/781 [25:42<22:03,  3.67s/it][Agenerating images:  54%|█████▍    | 422/781 [25:43<21:51,  3.65s/it]
generating images:  54%|█████▍    | 421/781 [25:46<22:00,  3.67s/it][Agenerating images:  54%|█████▍    | 423/781 [25:47<21:48,  3.66s/it]
generating images:  54%|█████▍    | 422/781 [25:50<21:57,  3.67s/it][Agenerating images:  54%|█████▍    | 424/781 [25:50<21:43,  3.65s/it]
generating images:  54%|█████▍    | 423/781 [25:53<21:54,  3.67s/it][Agenerating images:  54%|█████▍    | 425/781 [25:54<21:40,  3.65s/it]
generating images:  54%|█████▍    | 424/781 [25:57<21:50,  3.67s/it][Agenerating images:  55%|█████▍    | 426/781 [25:58<21:35,  3.65s/it]
generating images:  54%|█████▍    | 425/781 [26:01<21:44,  3.67s/it][Agenerating images:  55%|█████▍    | 427/781 [26:01<21:30,  3.65s/it]
generating images:  55%|█████▍    | 426/781 [26:04<21:42,  3.67s/it][Agenerating images:  55%|█████▍    | 428/781 [26:05<21:27,  3.65s/it]
generating images:  55%|█████▍    | 427/781 [26:08<21:39,  3.67s/it][Agenerating images:  55%|█████▍    | 429/781 [26:09<21:25,  3.65s/it]
generating images:  55%|█████▍    | 428/781 [26:12<21:35,  3.67s/it][Agenerating images:  55%|█████▌    | 430/781 [26:12<21:22,  3.65s/it]
generating images:  55%|█████▍    | 429/781 [26:15<21:31,  3.67s/it][Agenerating images:  55%|█████▌    | 431/781 [26:16<21:19,  3.66s/it]
generating images:  55%|█████▌    | 430/781 [26:19<21:27,  3.67s/it][Agenerating images:  55%|█████▌    | 432/781 [26:20<21:16,  3.66s/it]
generating images:  55%|█████▌    | 431/781 [26:23<21:24,  3.67s/it][Agenerating images:  55%|█████▌    | 433/781 [26:23<21:12,  3.66s/it]
generating images:  55%|█████▌    | 432/781 [26:26<21:20,  3.67s/it][Agenerating images:  56%|█████▌    | 434/781 [26:27<21:08,  3.66s/it]
generating images:  55%|█████▌    | 433/781 [26:30<21:16,  3.67s/it][Agenerating images:  56%|█████▌    | 435/781 [26:31<21:05,  3.66s/it]
generating images:  56%|█████▌    | 434/781 [26:34<21:13,  3.67s/it][Agenerating images:  56%|█████▌    | 436/781 [26:34<21:01,  3.66s/it]
generating images:  56%|█████▌    | 435/781 [26:37<21:09,  3.67s/it][Agenerating images:  56%|█████▌    | 437/781 [26:38<20:58,  3.66s/it]
generating images:  56%|█████▌    | 436/781 [26:41<21:06,  3.67s/it][Agenerating images:  56%|█████▌    | 438/781 [26:42<20:54,  3.66s/it]
generating images:  56%|█████▌    | 437/781 [26:45<21:02,  3.67s/it][Agenerating images:  56%|█████▌    | 439/781 [26:45<20:50,  3.66s/it]
generating images:  56%|█████▌    | 438/781 [26:48<20:59,  3.67s/it][Agenerating images:  56%|█████▋    | 440/781 [26:49<20:46,  3.66s/it]
generating images:  56%|█████▌    | 439/781 [26:52<20:56,  3.67s/it][Agenerating images:  56%|█████▋    | 441/781 [26:53<20:41,  3.65s/it]
generating images:  56%|█████▋    | 440/781 [26:56<20:51,  3.67s/it][Agenerating images:  57%|█████▋    | 442/781 [26:56<20:38,  3.65s/it]
generating images:  56%|█████▋    | 441/781 [26:59<20:48,  3.67s/it][Agenerating images:  57%|█████▋    | 443/781 [27:00<20:35,  3.66s/it]
generating images:  57%|█████▋    | 442/781 [27:03<20:44,  3.67s/it][Agenerating images:  57%|█████▋    | 444/781 [27:03<20:32,  3.66s/it]
generating images:  57%|█████▋    | 443/781 [27:07<20:40,  3.67s/it][Agenerating images:  57%|█████▋    | 445/781 [27:07<20:26,  3.65s/it]
generating images:  57%|█████▋    | 444/781 [27:10<20:37,  3.67s/it][Agenerating images:  57%|█████▋    | 446/781 [27:11<20:21,  3.65s/it]
generating images:  57%|█████▋    | 445/781 [27:14<20:33,  3.67s/it][Agenerating images:  57%|█████▋    | 447/781 [27:14<20:19,  3.65s/it]
generating images:  57%|█████▋    | 446/781 [27:18<20:29,  3.67s/it][Agenerating images:  57%|█████▋    | 448/781 [27:18<20:16,  3.65s/it]
generating images:  57%|█████▋    | 447/781 [27:21<20:25,  3.67s/it][Agenerating images:  57%|█████▋    | 449/781 [27:22<20:13,  3.65s/it]
generating images:  57%|█████▋    | 448/781 [27:25<20:21,  3.67s/it][Agenerating images:  58%|█████▊    | 450/781 [27:25<20:10,  3.66s/it]
generating images:  57%|█████▋    | 449/781 [27:29<20:17,  3.67s/it][Agenerating images:  58%|█████▊    | 451/781 [27:29<20:04,  3.65s/it]
generating images:  58%|█████▊    | 450/781 [27:32<20:12,  3.66s/it][Agenerating images:  58%|█████▊    | 452/781 [27:33<20:01,  3.65s/it]
generating images:  58%|█████▊    | 451/781 [27:36<20:09,  3.67s/it][Agenerating images:  58%|█████▊    | 453/781 [27:36<19:58,  3.65s/it]
generating images:  58%|█████▊    | 452/781 [27:40<20:05,  3.66s/it][Agenerating images:  58%|█████▊    | 454/781 [27:40<19:54,  3.65s/it]
generating images:  58%|█████▊    | 453/781 [27:43<20:02,  3.66s/it][Agenerating images:  58%|█████▊    | 455/781 [27:44<19:50,  3.65s/it]
generating images:  58%|█████▊    | 454/781 [27:47<19:59,  3.67s/it][Agenerating images:  58%|█████▊    | 456/781 [27:47<19:47,  3.65s/it]
generating images:  58%|█████▊    | 455/781 [27:51<19:56,  3.67s/it][Agenerating images:  59%|█████▊    | 457/781 [27:51<19:43,  3.65s/it]
generating images:  58%|█████▊    | 456/781 [27:54<19:52,  3.67s/it][Agenerating images:  59%|█████▊    | 458/781 [27:55<19:40,  3.65s/it]
generating images:  59%|█████▊    | 457/781 [27:58<19:49,  3.67s/it][Agenerating images:  59%|█████▉    | 459/781 [27:58<19:37,  3.66s/it]
generating images:  59%|█████▊    | 458/781 [28:02<19:46,  3.67s/it][Agenerating images:  59%|█████▉    | 460/781 [28:02<19:34,  3.66s/it]
generating images:  59%|█████▉    | 459/781 [28:05<19:42,  3.67s/it][Agenerating images:  59%|█████▉    | 461/781 [28:06<19:31,  3.66s/it]
generating images:  59%|█████▉    | 460/781 [28:09<19:38,  3.67s/it][Agenerating images:  59%|█████▉    | 462/781 [28:09<19:27,  3.66s/it]
generating images:  59%|█████▉    | 461/781 [28:13<19:34,  3.67s/it][Agenerating images:  59%|█████▉    | 463/781 [28:13<19:23,  3.66s/it]
generating images:  59%|█████▉    | 462/781 [28:16<19:30,  3.67s/it][Agenerating images:  59%|█████▉    | 464/781 [28:17<19:20,  3.66s/it]
generating images:  59%|█████▉    | 463/781 [28:20<19:25,  3.67s/it][Agenerating images:  60%|█████▉    | 465/781 [28:20<19:14,  3.65s/it]
generating images:  59%|█████▉    | 464/781 [28:24<19:22,  3.67s/it][Agenerating images:  60%|█████▉    | 466/781 [28:24<19:11,  3.66s/it]
generating images:  60%|█████▉    | 465/781 [28:27<19:18,  3.67s/it][Agenerating images:  60%|█████▉    | 467/781 [28:28<19:08,  3.66s/it]
generating images:  60%|█████▉    | 466/781 [28:31<19:13,  3.66s/it][Agenerating images:  60%|█████▉    | 468/781 [28:31<19:02,  3.65s/it]
generating images:  60%|█████▉    | 467/781 [28:35<19:10,  3.67s/it][Agenerating images:  60%|██████    | 469/781 [28:35<18:57,  3.65s/it]
generating images:  60%|█████▉    | 468/781 [28:38<19:07,  3.67s/it][Agenerating images:  60%|██████    | 470/781 [28:38<18:53,  3.64s/it]
generating images:  60%|██████    | 469/781 [28:42<19:04,  3.67s/it][Agenerating images:  60%|██████    | 471/781 [28:42<18:50,  3.65s/it]
generating images:  60%|██████    | 470/781 [28:46<19:00,  3.67s/it][Agenerating images:  60%|██████    | 472/781 [28:46<18:47,  3.65s/it]
generating images:  60%|██████    | 471/781 [28:49<19:04,  3.69s/it][Agenerating images:  61%|██████    | 473/781 [28:49<18:45,  3.65s/it]
generating images:  60%|██████    | 472/781 [28:53<18:58,  3.69s/it][Agenerating images:  61%|██████    | 474/781 [28:53<18:41,  3.65s/it]
generating images:  61%|██████    | 473/781 [28:57<18:53,  3.68s/it][Agenerating images:  61%|██████    | 475/781 [28:57<18:38,  3.65s/it]generating images:  61%|██████    | 476/781 [29:00<18:34,  3.65s/it]
generating images:  61%|██████    | 474/781 [29:00<18:49,  3.68s/it][Agenerating images:  61%|██████    | 477/781 [29:04<18:30,  3.65s/it]
generating images:  61%|██████    | 475/781 [29:04<18:45,  3.68s/it][Agenerating images:  61%|██████    | 478/781 [29:08<18:26,  3.65s/it]
generating images:  61%|██████    | 476/781 [29:08<18:40,  3.68s/it][Agenerating images:  61%|██████▏   | 479/781 [29:11<18:23,  3.65s/it]
generating images:  61%|██████    | 477/781 [29:11<18:36,  3.67s/it][Agenerating images:  61%|██████▏   | 480/781 [29:15<18:19,  3.65s/it]
generating images:  61%|██████    | 478/781 [29:15<18:34,  3.68s/it][Agenerating images:  62%|██████▏   | 481/781 [29:19<18:16,  3.65s/it]
generating images:  61%|██████▏   | 479/781 [29:19<18:29,  3.68s/it][Agenerating images:  62%|██████▏   | 482/781 [29:22<18:12,  3.65s/it]
generating images:  61%|██████▏   | 480/781 [29:22<18:25,  3.67s/it][Agenerating images:  62%|██████▏   | 483/781 [29:26<18:07,  3.65s/it]
generating images:  62%|██████▏   | 481/781 [29:26<18:21,  3.67s/it][Agenerating images:  62%|██████▏   | 484/781 [29:30<18:04,  3.65s/it]
generating images:  62%|██████▏   | 482/781 [29:30<18:18,  3.67s/it][Agenerating images:  62%|██████▏   | 485/781 [29:33<18:01,  3.65s/it]
generating images:  62%|██████▏   | 483/781 [29:33<18:14,  3.67s/it][Agenerating images:  62%|██████▏   | 486/781 [29:37<17:56,  3.65s/it]
generating images:  62%|██████▏   | 484/781 [29:37<18:10,  3.67s/it][Agenerating images:  62%|██████▏   | 487/781 [29:41<17:53,  3.65s/it]
generating images:  62%|██████▏   | 485/781 [29:41<18:07,  3.67s/it][Agenerating images:  62%|██████▏   | 488/781 [29:44<17:48,  3.65s/it]
generating images:  62%|██████▏   | 486/781 [29:45<18:04,  3.67s/it][Agenerating images:  63%|██████▎   | 489/781 [29:48<17:45,  3.65s/it]
generating images:  62%|██████▏   | 487/781 [29:48<18:00,  3.67s/it][Agenerating images:  63%|██████▎   | 490/781 [29:52<17:43,  3.65s/it]
generating images:  62%|██████▏   | 488/781 [29:52<17:55,  3.67s/it][Agenerating images:  63%|██████▎   | 491/781 [29:55<17:38,  3.65s/it]
generating images:  63%|██████▎   | 489/781 [29:56<17:52,  3.67s/it][Agenerating images:  63%|██████▎   | 492/781 [29:59<17:35,  3.65s/it]
generating images:  63%|██████▎   | 490/781 [29:59<17:48,  3.67s/it][Agenerating images:  63%|██████▎   | 493/781 [30:02<17:30,  3.65s/it]
generating images:  63%|██████▎   | 491/781 [30:03<17:44,  3.67s/it][Agenerating images:  63%|██████▎   | 494/781 [30:06<17:26,  3.65s/it]
generating images:  63%|██████▎   | 492/781 [30:07<17:41,  3.67s/it][Agenerating images:  63%|██████▎   | 495/781 [30:10<17:21,  3.64s/it]
generating images:  63%|██████▎   | 493/781 [30:10<17:37,  3.67s/it][Agenerating images:  64%|██████▎   | 496/781 [30:13<17:17,  3.64s/it]
generating images:  63%|██████▎   | 494/781 [30:14<17:31,  3.67s/it][Agenerating images:  64%|██████▎   | 497/781 [30:17<17:13,  3.64s/it]
generating images:  63%|██████▎   | 495/781 [30:18<17:26,  3.66s/it][Agenerating images:  64%|██████▍   | 498/781 [30:21<17:11,  3.64s/it]
generating images:  64%|██████▎   | 496/781 [30:21<17:22,  3.66s/it][Agenerating images:  64%|██████▍   | 499/781 [30:24<17:09,  3.65s/it]
generating images:  64%|██████▎   | 497/781 [30:25<17:20,  3.66s/it][Agenerating images:  64%|██████▍   | 500/781 [30:28<17:05,  3.65s/it]
generating images:  64%|██████▍   | 498/781 [30:28<17:16,  3.66s/it][Agenerating images:  64%|██████▍   | 501/781 [30:32<17:02,  3.65s/it]
generating images:  64%|██████▍   | 499/781 [30:32<17:12,  3.66s/it][Agenerating images:  64%|██████▍   | 502/781 [30:35<16:59,  3.65s/it]
generating images:  64%|██████▍   | 500/781 [30:36<17:09,  3.66s/it][Agenerating images:  64%|██████▍   | 503/781 [30:39<16:55,  3.65s/it]
generating images:  64%|██████▍   | 501/781 [30:39<17:06,  3.67s/it][Agenerating images:  65%|██████▍   | 504/781 [30:43<16:52,  3.65s/it]
generating images:  64%|██████▍   | 502/781 [30:43<17:02,  3.66s/it][Agenerating images:  65%|██████▍   | 505/781 [30:46<16:48,  3.65s/it]
generating images:  64%|██████▍   | 503/781 [30:47<16:58,  3.66s/it][Agenerating images:  65%|██████▍   | 506/781 [30:50<16:44,  3.65s/it]
generating images:  65%|██████▍   | 504/781 [30:50<16:55,  3.67s/it][Agenerating images:  65%|██████▍   | 507/781 [30:54<16:41,  3.66s/it]
generating images:  65%|██████▍   | 505/781 [30:54<16:51,  3.67s/it][Agenerating images:  65%|██████▌   | 508/781 [30:57<16:38,  3.66s/it]
generating images:  65%|██████▍   | 506/781 [30:58<16:48,  3.67s/it][Agenerating images:  65%|██████▌   | 509/781 [31:01<16:34,  3.66s/it]
generating images:  65%|██████▍   | 507/781 [31:01<16:45,  3.67s/it][Agenerating images:  65%|██████▌   | 510/781 [31:05<16:29,  3.65s/it]
generating images:  65%|██████▌   | 508/781 [31:05<16:41,  3.67s/it][Agenerating images:  65%|██████▌   | 511/781 [31:08<16:24,  3.65s/it]
generating images:  65%|██████▌   | 509/781 [31:09<16:38,  3.67s/it][Agenerating images:  66%|██████▌   | 512/781 [31:12<16:21,  3.65s/it]
generating images:  65%|██████▌   | 510/781 [31:12<16:33,  3.66s/it][Agenerating images:  66%|██████▌   | 513/781 [31:15<16:19,  3.65s/it]
generating images:  65%|██████▌   | 511/781 [31:16<16:29,  3.67s/it][Agenerating images:  66%|██████▌   | 514/781 [31:19<16:15,  3.65s/it]
generating images:  66%|██████▌   | 512/781 [31:20<16:26,  3.67s/it][Agenerating images:  66%|██████▌   | 515/781 [31:23<16:12,  3.66s/it]
generating images:  66%|██████▌   | 513/781 [31:23<16:23,  3.67s/it][Agenerating images:  66%|██████▌   | 516/781 [31:26<16:08,  3.66s/it]
generating images:  66%|██████▌   | 514/781 [31:27<16:20,  3.67s/it][Agenerating images:  66%|██████▌   | 517/781 [31:30<16:04,  3.66s/it]
generating images:  66%|██████▌   | 515/781 [31:31<16:16,  3.67s/it][Agenerating images:  66%|██████▋   | 518/781 [31:34<16:00,  3.65s/it]
generating images:  66%|██████▌   | 516/781 [31:35<16:12,  3.67s/it][Agenerating images:  66%|██████▋   | 519/781 [31:37<15:57,  3.66s/it]
generating images:  66%|██████▌   | 517/781 [31:38<16:08,  3.67s/it][Agenerating images:  67%|██████▋   | 520/781 [31:41<15:54,  3.66s/it]
generating images:  66%|██████▋   | 518/781 [31:42<16:05,  3.67s/it][Agenerating images:  67%|██████▋   | 521/781 [31:45<15:50,  3.66s/it]
generating images:  66%|██████▋   | 519/781 [31:46<16:01,  3.67s/it][Agenerating images:  67%|██████▋   | 522/781 [31:48<15:47,  3.66s/it]
generating images:  67%|██████▋   | 520/781 [31:49<15:57,  3.67s/it][Agenerating images:  67%|██████▋   | 523/781 [31:52<15:43,  3.66s/it]
generating images:  67%|██████▋   | 521/781 [31:53<15:54,  3.67s/it][Agenerating images:  67%|██████▋   | 524/781 [31:56<15:40,  3.66s/it]
generating images:  67%|██████▋   | 522/781 [31:57<15:50,  3.67s/it][Agenerating images:  67%|██████▋   | 525/781 [31:59<15:36,  3.66s/it]
generating images:  67%|██████▋   | 523/781 [32:00<15:47,  3.67s/it][Agenerating images:  67%|██████▋   | 526/781 [32:03<15:32,  3.66s/it]
generating images:  67%|██████▋   | 524/781 [32:04<15:43,  3.67s/it][Agenerating images:  67%|██████▋   | 527/781 [32:07<15:28,  3.66s/it]
generating images:  67%|██████▋   | 525/781 [32:08<15:40,  3.67s/it][Agenerating images:  68%|██████▊   | 528/781 [32:10<15:23,  3.65s/it]
generating images:  67%|██████▋   | 526/781 [32:11<15:36,  3.67s/it][Agenerating images:  68%|██████▊   | 529/781 [32:14<15:20,  3.65s/it]
generating images:  67%|██████▋   | 527/781 [32:15<15:32,  3.67s/it][Agenerating images:  68%|██████▊   | 530/781 [32:18<15:17,  3.65s/it]
generating images:  68%|██████▊   | 528/781 [32:19<15:28,  3.67s/it][Agenerating images:  68%|██████▊   | 531/781 [32:21<15:13,  3.65s/it]
generating images:  68%|██████▊   | 529/781 [32:22<15:25,  3.67s/it][Agenerating images:  68%|██████▊   | 532/781 [32:25<15:09,  3.65s/it]
generating images:  68%|██████▊   | 530/781 [32:26<15:21,  3.67s/it][Agenerating images:  68%|██████▊   | 533/781 [32:29<15:06,  3.66s/it]
generating images:  68%|██████▊   | 531/781 [32:30<15:17,  3.67s/it][Agenerating images:  68%|██████▊   | 534/781 [32:32<15:02,  3.65s/it]
generating images:  68%|██████▊   | 532/781 [32:33<15:13,  3.67s/it][Agenerating images:  69%|██████▊   | 535/781 [32:36<14:58,  3.65s/it]
generating images:  68%|██████▊   | 533/781 [32:37<15:10,  3.67s/it][Agenerating images:  69%|██████▊   | 536/781 [32:40<14:54,  3.65s/it]
generating images:  68%|██████▊   | 534/781 [32:41<15:07,  3.67s/it][Agenerating images:  69%|██████▉   | 537/781 [32:43<14:50,  3.65s/it]
generating images:  69%|██████▊   | 535/781 [32:44<15:03,  3.67s/it][Agenerating images:  69%|██████▉   | 538/781 [32:47<14:47,  3.65s/it]
generating images:  69%|██████▊   | 536/781 [32:48<14:59,  3.67s/it][Agenerating images:  69%|██████▉   | 539/781 [32:50<14:43,  3.65s/it]
generating images:  69%|██████▉   | 537/781 [32:52<14:55,  3.67s/it][Agenerating images:  69%|██████▉   | 540/781 [32:54<14:39,  3.65s/it]
generating images:  69%|██████▉   | 538/781 [32:55<14:51,  3.67s/it][Agenerating images:  69%|██████▉   | 541/781 [32:58<14:34,  3.64s/it]
generating images:  69%|██████▉   | 539/781 [32:59<14:48,  3.67s/it][Agenerating images:  69%|██████▉   | 542/781 [33:01<14:31,  3.65s/it]
generating images:  69%|██████▉   | 540/781 [33:03<14:44,  3.67s/it][Agenerating images:  70%|██████▉   | 543/781 [33:05<14:28,  3.65s/it]
generating images:  69%|██████▉   | 541/781 [33:06<14:40,  3.67s/it][Agenerating images:  70%|██████▉   | 544/781 [33:09<14:25,  3.65s/it]
generating images:  69%|██████▉   | 542/781 [33:10<14:36,  3.67s/it][Agenerating images:  70%|██████▉   | 545/781 [33:12<14:22,  3.65s/it]
generating images:  70%|██████▉   | 543/781 [33:14<14:33,  3.67s/it][Agenerating images:  70%|██████▉   | 546/781 [33:16<14:19,  3.66s/it]
generating images:  70%|██████▉   | 544/781 [33:17<14:29,  3.67s/it][Agenerating images:  70%|███████   | 547/781 [33:20<14:16,  3.66s/it]
generating images:  70%|██████▉   | 545/781 [33:21<14:26,  3.67s/it][Agenerating images:  70%|███████   | 548/781 [33:23<14:12,  3.66s/it]
generating images:  70%|██████▉   | 546/781 [33:25<14:22,  3.67s/it][Agenerating images:  70%|███████   | 549/781 [33:27<14:09,  3.66s/it]
generating images:  70%|███████   | 547/781 [33:28<14:19,  3.67s/it][Agenerating images:  70%|███████   | 550/781 [33:31<14:05,  3.66s/it]
generating images:  70%|███████   | 548/781 [33:32<14:15,  3.67s/it][Agenerating images:  71%|███████   | 551/781 [33:34<14:01,  3.66s/it]
generating images:  70%|███████   | 549/781 [33:36<14:11,  3.67s/it][Agenerating images:  71%|███████   | 552/781 [33:38<13:58,  3.66s/it]
generating images:  70%|███████   | 550/781 [33:39<14:08,  3.67s/it][Agenerating images:  71%|███████   | 553/781 [33:42<13:54,  3.66s/it]
generating images:  71%|███████   | 551/781 [33:43<14:04,  3.67s/it][Agenerating images:  71%|███████   | 554/781 [33:45<13:51,  3.66s/it]
generating images:  71%|███████   | 552/781 [33:47<14:01,  3.67s/it][Agenerating images:  71%|███████   | 555/781 [33:49<13:47,  3.66s/it]
generating images:  71%|███████   | 553/781 [33:50<13:57,  3.67s/it][Agenerating images:  71%|███████   | 556/781 [33:53<13:43,  3.66s/it]
generating images:  71%|███████   | 554/781 [33:54<13:53,  3.67s/it][Agenerating images:  71%|███████▏  | 557/781 [33:56<13:39,  3.66s/it]
generating images:  71%|███████   | 555/781 [33:58<13:50,  3.67s/it][Agenerating images:  71%|███████▏  | 558/781 [34:00<13:35,  3.66s/it]
generating images:  71%|███████   | 556/781 [34:01<13:46,  3.67s/it][Agenerating images:  72%|███████▏  | 559/781 [34:04<13:31,  3.66s/it]
generating images:  71%|███████▏  | 557/781 [34:05<13:42,  3.67s/it][Agenerating images:  72%|███████▏  | 560/781 [34:07<13:28,  3.66s/it]
generating images:  71%|███████▏  | 558/781 [34:09<13:38,  3.67s/it][Agenerating images:  72%|███████▏  | 561/781 [34:11<13:24,  3.66s/it]
generating images:  72%|███████▏  | 559/781 [34:12<13:35,  3.67s/it][Agenerating images:  72%|███████▏  | 562/781 [34:15<13:21,  3.66s/it]
generating images:  72%|███████▏  | 560/781 [34:16<13:31,  3.67s/it][Agenerating images:  72%|███████▏  | 563/781 [34:18<13:17,  3.66s/it]
generating images:  72%|███████▏  | 561/781 [34:20<13:28,  3.67s/it][Agenerating images:  72%|███████▏  | 564/781 [34:22<13:14,  3.66s/it]
generating images:  72%|███████▏  | 562/781 [34:23<13:24,  3.67s/it][Agenerating images:  72%|███████▏  | 565/781 [34:26<13:10,  3.66s/it]
generating images:  72%|███████▏  | 563/781 [34:27<13:20,  3.67s/it][Agenerating images:  72%|███████▏  | 566/781 [34:29<13:07,  3.66s/it]
generating images:  72%|███████▏  | 564/781 [34:31<13:17,  3.67s/it][Agenerating images:  73%|███████▎  | 567/781 [34:33<13:03,  3.66s/it]
generating images:  72%|███████▏  | 565/781 [34:34<13:14,  3.68s/it][Agenerating images:  73%|███████▎  | 568/781 [34:37<12:59,  3.66s/it]
generating images:  72%|███████▏  | 566/781 [34:38<13:10,  3.68s/it][Agenerating images:  73%|███████▎  | 569/781 [34:40<12:55,  3.66s/it]
generating images:  73%|███████▎  | 567/781 [34:42<13:06,  3.68s/it][Agenerating images:  73%|███████▎  | 570/781 [34:44<12:52,  3.66s/it]
generating images:  73%|███████▎  | 568/781 [34:45<13:02,  3.67s/it][Agenerating images:  73%|███████▎  | 571/781 [34:48<12:48,  3.66s/it]
generating images:  73%|███████▎  | 569/781 [34:49<12:57,  3.67s/it][Agenerating images:  73%|███████▎  | 572/781 [34:51<12:44,  3.66s/it]
generating images:  73%|███████▎  | 570/781 [34:53<12:54,  3.67s/it][Agenerating images:  73%|███████▎  | 573/781 [34:55<12:41,  3.66s/it]
generating images:  73%|███████▎  | 571/781 [34:56<12:50,  3.67s/it][Agenerating images:  73%|███████▎  | 574/781 [34:59<12:37,  3.66s/it]
generating images:  73%|███████▎  | 572/781 [35:00<12:46,  3.67s/it][Agenerating images:  74%|███████▎  | 575/781 [35:02<12:34,  3.66s/it]
generating images:  73%|███████▎  | 573/781 [35:04<12:43,  3.67s/it][Agenerating images:  74%|███████▍  | 576/781 [35:06<12:29,  3.66s/it]
generating images:  73%|███████▎  | 574/781 [35:07<12:39,  3.67s/it][Agenerating images:  74%|███████▍  | 577/781 [35:10<12:25,  3.66s/it]
generating images:  74%|███████▎  | 575/781 [35:11<12:36,  3.67s/it][Agenerating images:  74%|███████▍  | 578/781 [35:13<12:22,  3.66s/it]
generating images:  74%|███████▍  | 576/781 [35:15<12:32,  3.67s/it][Agenerating images:  74%|███████▍  | 579/781 [35:17<12:19,  3.66s/it]
generating images:  74%|███████▍  | 577/781 [35:18<12:28,  3.67s/it][Agenerating images:  74%|███████▍  | 580/781 [35:21<12:16,  3.66s/it]
generating images:  74%|███████▍  | 578/781 [35:22<12:24,  3.67s/it][Agenerating images:  74%|███████▍  | 581/781 [35:24<12:12,  3.66s/it]
generating images:  74%|███████▍  | 579/781 [35:26<12:20,  3.67s/it][Agenerating images:  75%|███████▍  | 582/781 [35:28<12:08,  3.66s/it]
generating images:  74%|███████▍  | 580/781 [35:29<12:17,  3.67s/it][Agenerating images:  75%|███████▍  | 583/781 [35:31<12:04,  3.66s/it]
generating images:  74%|███████▍  | 581/781 [35:33<12:13,  3.67s/it][Agenerating images:  75%|███████▍  | 584/781 [35:35<12:00,  3.66s/it]
generating images:  75%|███████▍  | 582/781 [35:37<12:10,  3.67s/it][Agenerating images:  75%|███████▍  | 585/781 [35:39<11:55,  3.65s/it]
generating images:  75%|███████▍  | 583/781 [35:40<12:06,  3.67s/it][Agenerating images:  75%|███████▌  | 586/781 [35:42<11:52,  3.66s/it]
generating images:  75%|███████▍  | 584/781 [35:44<12:02,  3.67s/it][Agenerating images:  75%|███████▌  | 587/781 [35:46<11:49,  3.66s/it]
generating images:  75%|███████▍  | 585/781 [35:48<11:59,  3.67s/it][Agenerating images:  75%|███████▌  | 588/781 [35:50<11:45,  3.66s/it]
generating images:  75%|███████▌  | 586/781 [35:51<11:56,  3.67s/it][Agenerating images:  75%|███████▌  | 589/781 [35:53<11:41,  3.66s/it]
generating images:  75%|███████▌  | 587/781 [35:55<11:52,  3.67s/it][Agenerating images:  76%|███████▌  | 590/781 [35:57<11:38,  3.66s/it]
generating images:  75%|███████▌  | 588/781 [35:59<11:49,  3.67s/it][Agenerating images:  76%|███████▌  | 591/781 [36:01<11:34,  3.65s/it]
generating images:  75%|███████▌  | 589/781 [36:03<11:45,  3.67s/it][Agenerating images:  76%|███████▌  | 592/781 [36:04<11:30,  3.65s/it]
generating images:  76%|███████▌  | 590/781 [36:06<11:41,  3.67s/it][Agenerating images:  76%|███████▌  | 593/781 [36:08<11:27,  3.66s/it]
generating images:  76%|███████▌  | 591/781 [36:10<11:37,  3.67s/it][Agenerating images:  76%|███████▌  | 594/781 [36:12<11:23,  3.66s/it]
generating images:  76%|███████▌  | 592/781 [36:14<11:33,  3.67s/it][Agenerating images:  76%|███████▌  | 595/781 [36:15<11:20,  3.66s/it]
generating images:  76%|███████▌  | 593/781 [36:17<11:29,  3.67s/it][Agenerating images:  76%|███████▋  | 596/781 [36:19<11:16,  3.66s/it]
generating images:  76%|███████▌  | 594/781 [36:21<11:26,  3.67s/it][Agenerating images:  76%|███████▋  | 597/781 [36:23<11:12,  3.66s/it]
generating images:  76%|███████▌  | 595/781 [36:25<11:22,  3.67s/it][Agenerating images:  77%|███████▋  | 598/781 [36:26<11:09,  3.66s/it]
generating images:  76%|███████▋  | 596/781 [36:28<11:18,  3.67s/it][Agenerating images:  77%|███████▋  | 599/781 [36:30<11:05,  3.66s/it]
generating images:  76%|███████▋  | 597/781 [36:32<11:15,  3.67s/it][Agenerating images:  77%|███████▋  | 600/781 [36:34<11:01,  3.66s/it]
generating images:  77%|███████▋  | 598/781 [36:36<11:11,  3.67s/it][Agenerating images:  77%|███████▋  | 601/781 [36:37<10:58,  3.66s/it]
generating images:  77%|███████▋  | 599/781 [36:39<11:07,  3.67s/it][Agenerating images:  77%|███████▋  | 602/781 [36:41<10:54,  3.66s/it]
generating images:  77%|███████▋  | 600/781 [36:43<11:04,  3.67s/it][Agenerating images:  77%|███████▋  | 603/781 [36:45<10:51,  3.66s/it]
generating images:  77%|███████▋  | 601/781 [36:47<11:00,  3.67s/it][Agenerating images:  77%|███████▋  | 604/781 [36:48<10:47,  3.66s/it]
generating images:  77%|███████▋  | 602/781 [36:50<10:56,  3.67s/it][Agenerating images:  77%|███████▋  | 605/781 [36:52<10:44,  3.66s/it]
generating images:  77%|███████▋  | 603/781 [36:54<10:53,  3.67s/it][Agenerating images:  78%|███████▊  | 606/781 [36:56<10:40,  3.66s/it]
generating images:  77%|███████▋  | 604/781 [36:58<10:48,  3.67s/it][Agenerating images:  78%|███████▊  | 607/781 [36:59<10:36,  3.66s/it]
generating images:  77%|███████▋  | 605/781 [37:01<10:44,  3.66s/it][Agenerating images:  78%|███████▊  | 608/781 [37:03<10:32,  3.66s/it]
generating images:  78%|███████▊  | 606/781 [37:05<10:40,  3.66s/it][Agenerating images:  78%|███████▊  | 609/781 [37:07<10:28,  3.65s/it]
generating images:  78%|███████▊  | 607/781 [37:08<10:36,  3.66s/it][Agenerating images:  78%|███████▊  | 610/781 [37:10<10:25,  3.66s/it]
generating images:  78%|███████▊  | 608/781 [37:12<10:32,  3.66s/it][Agenerating images:  78%|███████▊  | 611/781 [37:14<10:21,  3.66s/it]
generating images:  78%|███████▊  | 609/781 [37:16<10:28,  3.65s/it][Agenerating images:  78%|███████▊  | 612/781 [37:18<10:17,  3.66s/it]
generating images:  78%|███████▊  | 610/781 [37:19<10:25,  3.66s/it][Agenerating images:  78%|███████▊  | 613/781 [37:21<10:14,  3.66s/it]
generating images:  78%|███████▊  | 611/781 [37:23<10:21,  3.66s/it][Agenerating images:  79%|███████▊  | 614/781 [37:25<10:10,  3.65s/it]
generating images:  78%|███████▊  | 612/781 [37:27<10:18,  3.66s/it][Agenerating images:  79%|███████▊  | 615/781 [37:28<10:05,  3.65s/it]
generating images:  78%|███████▊  | 613/781 [37:30<10:15,  3.66s/it][Agenerating images:  79%|███████▉  | 616/781 [37:32<10:02,  3.65s/it]
generating images:  79%|███████▊  | 614/781 [37:34<10:12,  3.67s/it][Agenerating images:  79%|███████▉  | 617/781 [37:36<09:59,  3.65s/it]
generating images:  79%|███████▊  | 615/781 [37:38<10:08,  3.67s/it][Agenerating images:  79%|███████▉  | 618/781 [37:39<09:55,  3.65s/it]
generating images:  79%|███████▉  | 616/781 [37:41<10:05,  3.67s/it][Agenerating images:  79%|███████▉  | 619/781 [37:43<09:52,  3.65s/it]
generating images:  79%|███████▉  | 617/781 [37:45<10:01,  3.67s/it][Agenerating images:  79%|███████▉  | 620/781 [37:47<09:48,  3.66s/it]
generating images:  79%|███████▉  | 618/781 [37:49<09:57,  3.66s/it][Agenerating images:  80%|███████▉  | 621/781 [37:50<09:44,  3.66s/it]
generating images:  79%|███████▉  | 619/781 [37:52<09:53,  3.67s/it][Agenerating images:  80%|███████▉  | 622/781 [37:54<09:40,  3.65s/it]
generating images:  79%|███████▉  | 620/781 [37:56<09:50,  3.67s/it][Agenerating images:  80%|███████▉  | 623/781 [37:58<09:36,  3.65s/it]
generating images:  80%|███████▉  | 621/781 [38:00<09:47,  3.67s/it][Agenerating images:  80%|███████▉  | 624/781 [38:01<09:33,  3.65s/it]
generating images:  80%|███████▉  | 622/781 [38:03<09:43,  3.67s/it][Agenerating images:  80%|████████  | 625/781 [38:05<09:29,  3.65s/it]
generating images:  80%|███████▉  | 623/781 [38:07<09:39,  3.67s/it][Agenerating images:  80%|████████  | 626/781 [38:09<09:25,  3.65s/it]
generating images:  80%|███████▉  | 624/781 [38:11<09:35,  3.67s/it][Agenerating images:  80%|████████  | 627/781 [38:12<09:21,  3.65s/it]
generating images:  80%|████████  | 625/781 [38:14<09:32,  3.67s/it][Agenerating images:  80%|████████  | 628/781 [38:16<09:18,  3.65s/it]
generating images:  80%|████████  | 626/781 [38:18<09:28,  3.67s/it][Agenerating images:  81%|████████  | 629/781 [38:20<09:15,  3.65s/it]
generating images:  80%|████████  | 627/781 [38:22<09:25,  3.67s/it][Agenerating images:  81%|████████  | 630/781 [38:23<09:11,  3.65s/it]
generating images:  80%|████████  | 628/781 [38:26<09:21,  3.67s/it][Agenerating images:  81%|████████  | 631/781 [38:27<09:07,  3.65s/it]
generating images:  81%|████████  | 629/781 [38:29<09:18,  3.67s/it][Agenerating images:  81%|████████  | 632/781 [38:31<09:04,  3.65s/it]
generating images:  81%|████████  | 630/781 [38:33<09:14,  3.67s/it][Agenerating images:  81%|████████  | 633/781 [38:34<09:01,  3.66s/it]
generating images:  81%|████████  | 631/781 [38:37<09:10,  3.67s/it][Agenerating images:  81%|████████  | 634/781 [38:38<08:57,  3.66s/it]
generating images:  81%|████████  | 632/781 [38:40<09:06,  3.67s/it][Agenerating images:  81%|████████▏ | 635/781 [38:42<08:53,  3.65s/it]
generating images:  81%|████████  | 633/781 [38:44<09:03,  3.67s/it][Agenerating images:  81%|████████▏ | 636/781 [38:45<08:49,  3.65s/it]
generating images:  81%|████████  | 634/781 [38:48<08:59,  3.67s/it][Agenerating images:  82%|████████▏ | 637/781 [38:49<08:46,  3.66s/it]
generating images:  81%|████████▏ | 635/781 [38:51<08:56,  3.67s/it][Agenerating images:  82%|████████▏ | 638/781 [38:52<08:43,  3.66s/it]
generating images:  81%|████████▏ | 636/781 [38:55<08:52,  3.67s/it][Agenerating images:  82%|████████▏ | 639/781 [38:56<08:39,  3.66s/it]
generating images:  82%|████████▏ | 637/781 [38:59<08:48,  3.67s/it][Agenerating images:  82%|████████▏ | 640/781 [39:00<08:36,  3.66s/it]
generating images:  82%|████████▏ | 638/781 [39:02<08:45,  3.67s/it][Agenerating images:  82%|████████▏ | 641/781 [39:03<08:31,  3.65s/it]
generating images:  82%|████████▏ | 639/781 [39:06<08:41,  3.67s/it][Agenerating images:  82%|████████▏ | 642/781 [39:07<08:27,  3.65s/it]
generating images:  82%|████████▏ | 640/781 [39:10<08:36,  3.67s/it][Agenerating images:  82%|████████▏ | 643/781 [39:11<08:23,  3.65s/it]
generating images:  82%|████████▏ | 641/781 [39:13<08:33,  3.67s/it][Agenerating images:  82%|████████▏ | 644/781 [39:14<08:20,  3.65s/it]
generating images:  82%|████████▏ | 642/781 [39:17<08:30,  3.67s/it][Agenerating images:  83%|████████▎ | 645/781 [39:18<08:16,  3.65s/it]
generating images:  82%|████████▏ | 643/781 [39:21<08:26,  3.67s/it][Agenerating images:  83%|████████▎ | 646/781 [39:22<08:13,  3.65s/it]
generating images:  82%|████████▏ | 644/781 [39:24<08:22,  3.67s/it][Agenerating images:  83%|████████▎ | 647/781 [39:25<08:09,  3.65s/it]
generating images:  83%|████████▎ | 645/781 [39:28<08:19,  3.67s/it][Agenerating images:  83%|████████▎ | 648/781 [39:29<08:06,  3.66s/it]
generating images:  83%|████████▎ | 646/781 [39:32<08:16,  3.67s/it][Agenerating images:  83%|████████▎ | 649/781 [39:33<08:02,  3.65s/it]
generating images:  83%|████████▎ | 647/781 [39:35<08:12,  3.67s/it][Agenerating images:  83%|████████▎ | 650/781 [39:36<07:58,  3.65s/it]
generating images:  83%|████████▎ | 648/781 [39:39<08:08,  3.68s/it][Agenerating images:  83%|████████▎ | 651/781 [39:40<07:55,  3.65s/it]
generating images:  83%|████████▎ | 649/781 [39:43<08:04,  3.67s/it][Agenerating images:  83%|████████▎ | 652/781 [39:44<07:51,  3.66s/it]
generating images:  83%|████████▎ | 650/781 [39:46<08:01,  3.67s/it][Agenerating images:  84%|████████▎ | 653/781 [39:47<07:47,  3.65s/it]
generating images:  83%|████████▎ | 651/781 [39:50<07:57,  3.67s/it][Agenerating images:  84%|████████▎ | 654/781 [39:51<07:43,  3.65s/it]
generating images:  83%|████████▎ | 652/781 [39:54<07:54,  3.67s/it][Agenerating images:  84%|████████▍ | 655/781 [39:55<07:40,  3.65s/it]
generating images:  84%|████████▎ | 653/781 [39:57<07:50,  3.67s/it][Agenerating images:  84%|████████▍ | 656/781 [39:58<07:37,  3.66s/it]
generating images:  84%|████████▎ | 654/781 [40:01<07:46,  3.67s/it][Agenerating images:  84%|████████▍ | 657/781 [40:02<07:33,  3.66s/it]
generating images:  84%|████████▍ | 655/781 [40:05<07:42,  3.67s/it][Agenerating images:  84%|████████▍ | 658/781 [40:06<07:30,  3.66s/it]
generating images:  84%|████████▍ | 656/781 [40:08<07:38,  3.67s/it][Agenerating images:  84%|████████▍ | 659/781 [40:09<07:26,  3.66s/it]
generating images:  84%|████████▍ | 657/781 [40:12<07:35,  3.67s/it][Agenerating images:  85%|████████▍ | 660/781 [40:13<07:21,  3.65s/it]
generating images:  84%|████████▍ | 658/781 [40:16<07:31,  3.67s/it][Agenerating images:  85%|████████▍ | 661/781 [40:17<07:17,  3.65s/it]
generating images:  84%|████████▍ | 659/781 [40:19<07:27,  3.67s/it][Agenerating images:  85%|████████▍ | 662/781 [40:20<07:13,  3.64s/it]
generating images:  85%|████████▍ | 660/781 [40:23<07:24,  3.67s/it][Agenerating images:  85%|████████▍ | 663/781 [40:24<07:10,  3.64s/it]
generating images:  85%|████████▍ | 661/781 [40:27<07:20,  3.67s/it][Agenerating images:  85%|████████▌ | 664/781 [40:27<07:06,  3.65s/it]
generating images:  85%|████████▍ | 662/781 [40:30<07:16,  3.67s/it][Agenerating images:  85%|████████▌ | 665/781 [40:31<07:03,  3.65s/it]
generating images:  85%|████████▍ | 663/781 [40:34<07:12,  3.67s/it][Agenerating images:  85%|████████▌ | 666/781 [40:35<06:59,  3.65s/it]
generating images:  85%|████████▌ | 664/781 [40:38<07:09,  3.67s/it][Agenerating images:  85%|████████▌ | 667/781 [40:38<06:56,  3.65s/it]
generating images:  85%|████████▌ | 665/781 [40:41<07:04,  3.66s/it][Agenerating images:  86%|████████▌ | 668/781 [40:42<06:52,  3.65s/it]
generating images:  85%|████████▌ | 666/781 [40:45<07:01,  3.67s/it][Agenerating images:  86%|████████▌ | 669/781 [40:46<06:49,  3.65s/it]
generating images:  85%|████████▌ | 667/781 [40:49<06:57,  3.67s/it][Agenerating images:  86%|████████▌ | 670/781 [40:49<06:45,  3.65s/it]
generating images:  86%|████████▌ | 668/781 [40:52<06:54,  3.66s/it][Agenerating images:  86%|████████▌ | 671/781 [40:53<06:41,  3.65s/it]
generating images:  86%|████████▌ | 669/781 [40:56<06:50,  3.67s/it][Agenerating images:  86%|████████▌ | 672/781 [40:57<06:38,  3.66s/it]
generating images:  86%|████████▌ | 670/781 [41:00<06:47,  3.67s/it][Agenerating images:  86%|████████▌ | 673/781 [41:00<06:35,  3.66s/it]
generating images:  86%|████████▌ | 671/781 [41:03<06:43,  3.67s/it][Agenerating images:  86%|████████▋ | 674/781 [41:04<06:31,  3.66s/it]
generating images:  86%|████████▌ | 672/781 [41:07<06:39,  3.67s/it][Agenerating images:  86%|████████▋ | 675/781 [41:08<06:27,  3.65s/it]
generating images:  86%|████████▌ | 673/781 [41:11<06:36,  3.67s/it][Agenerating images:  87%|████████▋ | 676/781 [41:11<06:23,  3.65s/it]
generating images:  86%|████████▋ | 674/781 [41:14<06:33,  3.67s/it][Agenerating images:  87%|████████▋ | 677/781 [41:15<06:19,  3.65s/it]
generating images:  86%|████████▋ | 675/781 [41:18<06:29,  3.67s/it][Agenerating images:  87%|████████▋ | 678/781 [41:19<06:16,  3.66s/it]
generating images:  87%|████████▋ | 676/781 [41:22<06:25,  3.67s/it][Agenerating images:  87%|████████▋ | 679/781 [41:22<06:12,  3.66s/it]
generating images:  87%|████████▋ | 677/781 [41:25<06:21,  3.67s/it][Agenerating images:  87%|████████▋ | 680/781 [41:26<06:09,  3.66s/it]
generating images:  87%|████████▋ | 678/781 [41:29<06:17,  3.66s/it][Agenerating images:  87%|████████▋ | 681/781 [41:30<06:05,  3.65s/it]
generating images:  87%|████████▋ | 679/781 [41:33<06:13,  3.66s/it][Agenerating images:  87%|████████▋ | 682/781 [41:33<06:01,  3.65s/it]
generating images:  87%|████████▋ | 680/781 [41:36<06:09,  3.66s/it][Agenerating images:  87%|████████▋ | 683/781 [41:37<05:57,  3.65s/it]
generating images:  87%|████████▋ | 681/781 [41:40<06:06,  3.66s/it][Agenerating images:  88%|████████▊ | 684/781 [41:41<05:53,  3.65s/it]
generating images:  87%|████████▋ | 682/781 [41:44<06:02,  3.66s/it][Agenerating images:  88%|████████▊ | 685/781 [41:44<05:50,  3.65s/it]
generating images:  87%|████████▋ | 683/781 [41:47<05:59,  3.67s/it][Agenerating images:  88%|████████▊ | 686/781 [41:48<05:47,  3.65s/it]
generating images:  88%|████████▊ | 684/781 [41:51<05:55,  3.67s/it][Agenerating images:  88%|████████▊ | 687/781 [41:51<05:43,  3.65s/it]
generating images:  88%|████████▊ | 685/781 [41:55<05:52,  3.67s/it][Agenerating images:  88%|████████▊ | 688/781 [41:55<05:39,  3.65s/it]
generating images:  88%|████████▊ | 686/781 [41:58<05:48,  3.67s/it][Agenerating images:  88%|████████▊ | 689/781 [41:59<05:36,  3.65s/it]
generating images:  88%|████████▊ | 687/781 [42:02<05:44,  3.67s/it][Agenerating images:  88%|████████▊ | 690/781 [42:02<05:32,  3.66s/it]
generating images:  88%|████████▊ | 688/781 [42:06<05:41,  3.67s/it][Agenerating images:  88%|████████▊ | 691/781 [42:06<05:29,  3.66s/it]
generating images:  88%|████████▊ | 689/781 [42:09<05:37,  3.67s/it][Agenerating images:  89%|████████▊ | 692/781 [42:10<05:25,  3.66s/it]
generating images:  88%|████████▊ | 690/781 [42:13<05:33,  3.67s/it][Agenerating images:  89%|████████▊ | 693/781 [42:13<05:21,  3.65s/it]
generating images:  88%|████████▊ | 691/781 [42:17<05:30,  3.67s/it][Agenerating images:  89%|████████▉ | 694/781 [42:17<05:17,  3.65s/it]
generating images:  89%|████████▊ | 692/781 [42:20<05:26,  3.67s/it][Agenerating images:  89%|████████▉ | 695/781 [42:21<05:14,  3.65s/it]
generating images:  89%|████████▊ | 693/781 [42:24<05:22,  3.67s/it][Agenerating images:  89%|████████▉ | 696/781 [42:24<05:10,  3.65s/it]
generating images:  89%|████████▉ | 694/781 [42:28<05:19,  3.67s/it][Agenerating images:  89%|████████▉ | 697/781 [42:28<05:06,  3.65s/it]
generating images:  89%|████████▉ | 695/781 [42:31<05:15,  3.67s/it][Agenerating images:  89%|████████▉ | 698/781 [42:32<05:03,  3.65s/it]
generating images:  89%|████████▉ | 696/781 [42:35<05:11,  3.67s/it][Agenerating images:  90%|████████▉ | 699/781 [42:35<04:59,  3.65s/it]
generating images:  89%|████████▉ | 697/781 [42:39<05:08,  3.67s/it][Agenerating images:  90%|████████▉ | 700/781 [42:39<04:55,  3.65s/it]
generating images:  89%|████████▉ | 698/781 [42:42<05:04,  3.67s/it][Agenerating images:  90%|████████▉ | 701/781 [42:43<04:51,  3.65s/it]
generating images:  90%|████████▉ | 699/781 [42:46<05:00,  3.67s/it][Agenerating images:  90%|████████▉ | 702/781 [42:46<04:48,  3.65s/it]
generating images:  90%|████████▉ | 700/781 [42:50<04:56,  3.67s/it][Agenerating images:  90%|█████████ | 703/781 [42:50<04:44,  3.65s/it]
generating images:  90%|████████▉ | 701/781 [42:53<04:53,  3.67s/it][Agenerating images:  90%|█████████ | 704/781 [42:54<04:41,  3.65s/it]
generating images:  90%|████████▉ | 702/781 [42:57<04:49,  3.66s/it][Agenerating images:  90%|█████████ | 705/781 [42:57<04:37,  3.65s/it]
generating images:  90%|█████████ | 703/781 [43:01<04:45,  3.66s/it][Agenerating images:  90%|█████████ | 706/781 [43:01<04:34,  3.66s/it]
generating images:  90%|█████████ | 704/781 [43:04<04:42,  3.66s/it][Agenerating images:  91%|█████████ | 707/781 [43:05<04:30,  3.65s/it]
generating images:  90%|█████████ | 705/781 [43:08<04:38,  3.67s/it][Agenerating images:  91%|█████████ | 708/781 [43:08<04:27,  3.66s/it]
generating images:  90%|█████████ | 706/781 [43:12<04:35,  3.67s/it][Agenerating images:  91%|█████████ | 709/781 [43:12<04:23,  3.66s/it]
generating images:  91%|█████████ | 707/781 [43:15<04:31,  3.67s/it][Agenerating images:  91%|█████████ | 710/781 [43:16<04:19,  3.66s/it]
generating images:  91%|█████████ | 708/781 [43:19<04:28,  3.67s/it][Agenerating images:  91%|█████████ | 711/781 [43:19<04:16,  3.66s/it]
generating images:  91%|█████████ | 709/781 [43:23<04:24,  3.67s/it][Agenerating images:  91%|█████████ | 712/781 [43:23<04:12,  3.66s/it]
generating images:  91%|█████████ | 710/781 [43:26<04:20,  3.67s/it][Agenerating images:  91%|█████████▏| 713/781 [43:26<04:08,  3.66s/it]
generating images:  91%|█████████ | 711/781 [43:30<04:17,  3.67s/it][Agenerating images:  91%|█████████▏| 714/781 [43:30<04:05,  3.66s/it]
generating images:  91%|█████████ | 712/781 [43:34<04:13,  3.67s/it][Agenerating images:  92%|█████████▏| 715/781 [43:34<04:01,  3.66s/it]
generating images:  91%|█████████▏| 713/781 [43:37<04:09,  3.67s/it][Agenerating images:  92%|█████████▏| 716/781 [43:37<03:57,  3.66s/it]
generating images:  91%|█████████▏| 714/781 [43:41<04:06,  3.67s/it][Agenerating images:  92%|█████████▏| 717/781 [43:41<03:54,  3.66s/it]
generating images:  92%|█████████▏| 715/781 [43:45<04:02,  3.67s/it][Agenerating images:  92%|█████████▏| 718/781 [43:45<03:50,  3.66s/it]
generating images:  92%|█████████▏| 716/781 [43:48<03:58,  3.67s/it][Agenerating images:  92%|█████████▏| 719/781 [43:48<03:46,  3.66s/it]
generating images:  92%|█████████▏| 717/781 [43:52<03:54,  3.67s/it][Agenerating images:  92%|█████████▏| 720/781 [43:52<03:43,  3.66s/it]generating images:  92%|█████████▏| 721/781 [43:56<03:39,  3.66s/it]
generating images:  92%|█████████▏| 718/781 [43:56<03:51,  3.67s/it][Agenerating images:  92%|█████████▏| 722/781 [43:59<03:35,  3.65s/it]
generating images:  92%|█████████▏| 719/781 [43:59<03:47,  3.67s/it][Agenerating images:  93%|█████████▎| 723/781 [44:03<03:31,  3.65s/it]
generating images:  92%|█████████▏| 720/781 [44:03<03:44,  3.67s/it][Agenerating images:  93%|█████████▎| 724/781 [44:07<03:28,  3.65s/it]
generating images:  92%|█████████▏| 721/781 [44:07<03:40,  3.67s/it][Agenerating images:  93%|█████████▎| 725/781 [44:10<03:25,  3.68s/it]
generating images:  92%|█████████▏| 722/781 [44:10<03:36,  3.67s/it][Agenerating images:  93%|█████████▎| 726/781 [44:14<03:21,  3.67s/it]
generating images:  93%|█████████▎| 723/781 [44:14<03:32,  3.67s/it][Agenerating images:  93%|█████████▎| 727/781 [44:18<03:17,  3.66s/it]
generating images:  93%|█████████▎| 724/781 [44:18<03:29,  3.67s/it][Agenerating images:  93%|█████████▎| 728/781 [44:21<03:14,  3.66s/it]
generating images:  93%|█████████▎| 725/781 [44:21<03:25,  3.67s/it][Agenerating images:  93%|█████████▎| 729/781 [44:25<03:10,  3.66s/it]
generating images:  93%|█████████▎| 726/781 [44:25<03:21,  3.67s/it][Agenerating images:  93%|█████████▎| 730/781 [44:29<03:06,  3.66s/it]
generating images:  93%|█████████▎| 727/781 [44:29<03:18,  3.67s/it][Agenerating images:  94%|█████████▎| 731/781 [44:32<03:02,  3.66s/it]
generating images:  93%|█████████▎| 728/781 [44:32<03:14,  3.67s/it][Agenerating images:  94%|█████████▎| 732/781 [44:36<02:59,  3.66s/it]
generating images:  93%|█████████▎| 729/781 [44:36<03:11,  3.68s/it][Agenerating images:  94%|█████████▍| 733/781 [44:40<02:55,  3.65s/it]
generating images:  93%|█████████▎| 730/781 [44:40<03:07,  3.68s/it][Agenerating images:  94%|█████████▍| 734/781 [44:43<02:51,  3.65s/it]
generating images:  94%|█████████▎| 731/781 [44:44<03:03,  3.68s/it][Agenerating images:  94%|█████████▍| 735/781 [44:47<02:48,  3.65s/it]
generating images:  94%|█████████▎| 732/781 [44:47<02:59,  3.67s/it][Agenerating images:  94%|█████████▍| 736/781 [44:51<02:44,  3.65s/it]
generating images:  94%|█████████▍| 733/781 [44:51<02:57,  3.69s/it][Agenerating images:  94%|█████████▍| 737/781 [44:54<02:40,  3.65s/it]
generating images:  94%|█████████▍| 734/781 [44:55<02:53,  3.69s/it][Agenerating images:  94%|█████████▍| 738/781 [44:58<02:37,  3.65s/it]
generating images:  94%|█████████▍| 735/781 [44:58<02:49,  3.69s/it][Agenerating images:  95%|█████████▍| 739/781 [45:02<02:33,  3.65s/it]
generating images:  94%|█████████▍| 736/781 [45:02<02:45,  3.68s/it][Agenerating images:  95%|█████████▍| 740/781 [45:05<02:29,  3.66s/it]
generating images:  94%|█████████▍| 737/781 [45:06<02:41,  3.68s/it][Agenerating images:  95%|█████████▍| 741/781 [45:09<02:26,  3.65s/it]
generating images:  94%|█████████▍| 738/781 [45:09<02:38,  3.68s/it][Agenerating images:  95%|█████████▌| 742/781 [45:13<02:22,  3.66s/it]
generating images:  95%|█████████▍| 739/781 [45:13<02:34,  3.68s/it][Agenerating images:  95%|█████████▌| 743/781 [45:16<02:18,  3.66s/it]
generating images:  95%|█████████▍| 740/781 [45:17<02:30,  3.68s/it][Agenerating images:  95%|█████████▌| 744/781 [45:20<02:15,  3.65s/it]
generating images:  95%|█████████▍| 741/781 [45:20<02:26,  3.67s/it][Agenerating images:  95%|█████████▌| 745/781 [45:24<02:11,  3.65s/it]
generating images:  95%|█████████▌| 742/781 [45:24<02:23,  3.67s/it][Agenerating images:  96%|█████████▌| 746/781 [45:27<02:07,  3.66s/it]
generating images:  95%|█████████▌| 743/781 [45:28<02:19,  3.67s/it][Agenerating images:  96%|█████████▌| 747/781 [45:31<02:04,  3.66s/it]
generating images:  95%|█████████▌| 744/781 [45:31<02:15,  3.67s/it][Agenerating images:  96%|█████████▌| 748/781 [45:35<02:00,  3.66s/it]
generating images:  95%|█████████▌| 745/781 [45:35<02:12,  3.67s/it][Agenerating images:  96%|█████████▌| 749/781 [45:38<01:56,  3.65s/it]
generating images:  96%|█████████▌| 746/781 [45:39<02:08,  3.67s/it][Agenerating images:  96%|█████████▌| 750/781 [45:42<01:53,  3.66s/it]
generating images:  96%|█████████▌| 747/781 [45:42<02:04,  3.67s/it][Agenerating images:  96%|█████████▌| 751/781 [45:45<01:49,  3.66s/it]
generating images:  96%|█████████▌| 748/781 [45:46<02:01,  3.67s/it][Agenerating images:  96%|█████████▋| 752/781 [45:49<01:46,  3.66s/it]
generating images:  96%|█████████▌| 749/781 [45:50<01:57,  3.67s/it][Agenerating images:  96%|█████████▋| 753/781 [45:53<01:42,  3.66s/it]
generating images:  96%|█████████▌| 750/781 [45:53<01:53,  3.67s/it][Agenerating images:  97%|█████████▋| 754/781 [45:56<01:38,  3.66s/it]
generating images:  96%|█████████▌| 751/781 [45:57<01:50,  3.67s/it][Agenerating images:  97%|█████████▋| 755/781 [46:00<01:35,  3.66s/it]
generating images:  96%|█████████▋| 752/781 [46:01<01:46,  3.67s/it][Agenerating images:  97%|█████████▋| 756/781 [46:04<01:31,  3.66s/it]
generating images:  96%|█████████▋| 753/781 [46:04<01:42,  3.67s/it][Agenerating images:  97%|█████████▋| 757/781 [46:07<01:27,  3.66s/it]
generating images:  97%|█████████▋| 754/781 [46:08<01:39,  3.67s/it][Agenerating images:  97%|█████████▋| 758/781 [46:11<01:24,  3.66s/it]
generating images:  97%|█████████▋| 755/781 [46:12<01:35,  3.67s/it][Agenerating images:  97%|█████████▋| 759/781 [46:15<01:20,  3.66s/it]
generating images:  97%|█████████▋| 756/781 [46:15<01:31,  3.67s/it][Agenerating images:  97%|█████████▋| 760/781 [46:18<01:16,  3.65s/it]
generating images:  97%|█████████▋| 757/781 [46:19<01:28,  3.67s/it][Agenerating images:  97%|█████████▋| 761/781 [46:22<01:13,  3.66s/it]
generating images:  97%|█████████▋| 758/781 [46:23<01:24,  3.67s/it][Agenerating images:  98%|█████████▊| 762/781 [46:26<01:09,  3.66s/it]
generating images:  97%|█████████▋| 759/781 [46:26<01:20,  3.67s/it][Agenerating images:  98%|█████████▊| 763/781 [46:29<01:05,  3.66s/it]
generating images:  97%|█████████▋| 760/781 [46:30<01:17,  3.67s/it][Agenerating images:  98%|█████████▊| 764/781 [46:33<01:02,  3.66s/it]
generating images:  97%|█████████▋| 761/781 [46:34<01:13,  3.67s/it][Agenerating images:  98%|█████████▊| 765/781 [46:37<00:58,  3.66s/it]
generating images:  98%|█████████▊| 762/781 [46:37<01:09,  3.68s/it][Agenerating images:  98%|█████████▊| 766/781 [46:40<00:54,  3.66s/it]
generating images:  98%|█████████▊| 763/781 [46:41<01:06,  3.67s/it][Agenerating images:  98%|█████████▊| 767/781 [46:44<00:51,  3.66s/it]
generating images:  98%|█████████▊| 764/781 [46:45<01:02,  3.67s/it][Agenerating images:  98%|█████████▊| 768/781 [46:48<00:47,  3.66s/it]
generating images:  98%|█████████▊| 765/781 [46:48<00:58,  3.67s/it][Agenerating images:  98%|█████████▊| 769/781 [46:51<00:43,  3.66s/it]
generating images:  98%|█████████▊| 766/781 [46:52<00:55,  3.67s/it][Agenerating images:  99%|█████████▊| 770/781 [46:55<00:40,  3.66s/it]
generating images:  98%|█████████▊| 767/781 [46:56<00:51,  3.67s/it][Agenerating images:  99%|█████████▊| 771/781 [46:59<00:36,  3.65s/it]
generating images:  98%|█████████▊| 768/781 [46:59<00:47,  3.67s/it][Agenerating images:  99%|█████████▉| 772/781 [47:02<00:32,  3.66s/it]
generating images:  98%|█████████▊| 769/781 [47:03<00:44,  3.67s/it][Agenerating images:  99%|█████████▉| 773/781 [47:06<00:29,  3.66s/it]
generating images:  99%|█████████▊| 770/781 [47:07<00:40,  3.67s/it][Agenerating images:  99%|█████████▉| 774/781 [47:10<00:25,  3.66s/it]
generating images:  99%|█████████▊| 771/781 [47:10<00:36,  3.67s/it][Agenerating images:  99%|█████████▉| 775/781 [47:13<00:21,  3.66s/it]
generating images:  99%|█████████▉| 772/781 [47:14<00:33,  3.67s/it][Agenerating images:  99%|█████████▉| 776/781 [47:17<00:18,  3.66s/it]
generating images:  99%|█████████▉| 773/781 [47:18<00:29,  3.67s/it][Agenerating images:  99%|█████████▉| 777/781 [47:21<00:14,  3.66s/it]
generating images:  99%|█████████▉| 774/781 [47:21<00:25,  3.67s/it][Agenerating images: 100%|█████████▉| 778/781 [47:24<00:10,  3.66s/it]
generating images:  99%|█████████▉| 775/781 [47:25<00:22,  3.67s/it][Agenerating images: 100%|█████████▉| 779/781 [47:28<00:07,  3.66s/it]
generating images:  99%|█████████▉| 776/781 [47:29<00:18,  3.67s/it][Agenerating images: 100%|█████████▉| 780/781 [47:32<00:03,  3.66s/it]
generating images:  99%|█████████▉| 777/781 [47:32<00:14,  3.67s/it][Agenerating images: 100%|██████████| 781/781 [47:35<00:00,  3.66s/it]generating images: 100%|██████████| 781/781 [47:35<00:00,  3.66s/it]

generating images: 100%|█████████▉| 778/781 [47:36<00:11,  3.67s/it][A
generating images: 100%|█████████▉| 779/781 [47:40<00:07,  3.67s/it][A
generating images: 100%|█████████▉| 780/781 [47:43<00:03,  3.67s/it][A
generating images: 100%|██████████| 781/781 [47:47<00:00,  3.67s/it][Agenerating images: 100%|██████████| 781/781 [47:47<00:00,  3.67s/it]
Downloading: "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth" to /home/chenyu.zhang/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth

  0%|          | 0.00/91.2M [00:00<?, ?B/s][A
 11%|█         | 10.1M/91.2M [00:00<00:01, 67.9MB/s][A
 22%|██▏       | 20.1M/91.2M [00:00<00:00, 78.8MB/s][A
 31%|███       | 27.9M/91.2M [00:00<00:00, 72.5MB/s][A
 38%|███▊      | 34.9M/91.2M [00:00<00:00, 68.4MB/s][A
 46%|████▌     | 41.5M/91.2M [00:00<00:00, 65.9MB/s][A
 55%|█████▍    | 50.1M/91.2M [00:00<00:00, 70.6MB/s][A
 66%|██████▌   | 60.1M/91.2M [00:00<00:00, 76.7MB/s][A
 77%|███████▋  | 70.1M/91.2M [00:00<00:00, 78.9MB/s][A
 88%|████████▊ | 80.1M/91.2M [00:01<00:00, 81.2MB/s][A
 99%|█████████▉| 90.1M/91.2M [00:01<00:00, 83.1MB/s][A100%|██████████| 91.2M/91.2M [00:01<00:00, 77.4MB/s]
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torch/utils/data/dataloader.py:617: UserWarning: This DataLoader will create 64 worker processes in total. Our suggested max number of worker in current system is 8, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(

  0%|          | 0/1562 [00:00<?, ?it/s][A/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

  0%|          | 1/1562 [00:04<2:09:05,  4.96s/it][A
  0%|          | 4/1562 [00:05<25:12,  1.03it/s]  [A
  0%|          | 7/1562 [00:05<12:09,  2.13it/s][A
  1%|          | 10/1562 [00:05<07:16,  3.56it/s][A
  1%|          | 13/1562 [00:05<04:49,  5.34it/s][A
  1%|          | 16/1562 [00:05<03:26,  7.47it/s][A
  1%|          | 19/1562 [00:05<02:35,  9.91it/s][A
  1%|▏         | 22/1562 [00:05<02:03, 12.49it/s][A
  2%|▏         | 25/1562 [00:05<01:41, 15.12it/s][A
  2%|▏         | 28/1562 [00:05<01:27, 17.58it/s][A
  2%|▏         | 31/1562 [00:06<01:17, 19.74it/s][A
  2%|▏         | 34/1562 [00:06<01:10, 21.73it/s][A
  2%|▏         | 37/1562 [00:06<01:06, 23.10it/s][A
  3%|▎         | 40/1562 [00:06<01:02, 24.40it/s][A
  3%|▎         | 43/1562 [00:06<01:00, 25.15it/s][A
  3%|▎         | 46/1562 [00:06<00:58, 25.75it/s][A
  3%|▎         | 49/1562 [00:06<00:57, 26.37it/s][A
  3%|▎         | 52/1562 [00:06<00:56, 26.71it/s][A
  4%|▎         | 55/1562 [00:06<00:55, 27.15it/s][A
  4%|▎         | 58/1562 [00:07<00:55, 27.33it/s][A
  4%|▍         | 61/1562 [00:07<00:56, 26.73it/s][A
  4%|▍         | 64/1562 [00:07<00:58, 25.77it/s][A
  4%|▍         | 67/1562 [00:07<00:56, 26.42it/s][A
  4%|▍         | 70/1562 [00:07<00:55, 26.96it/s][A
  5%|▍         | 73/1562 [00:07<00:54, 27.33it/s][A
  5%|▍         | 76/1562 [00:07<00:53, 27.71it/s][A
  5%|▌         | 79/1562 [00:07<00:53, 27.93it/s][A
  5%|▌         | 82/1562 [00:07<00:52, 28.15it/s][A
  5%|▌         | 85/1562 [00:08<00:52, 28.10it/s][A
  6%|▌         | 88/1562 [00:08<00:52, 28.22it/s][A
  6%|▌         | 91/1562 [00:08<00:52, 28.28it/s][A
  6%|▌         | 94/1562 [00:08<00:52, 28.20it/s][A
  6%|▌         | 97/1562 [00:08<00:51, 28.25it/s][A
  6%|▋         | 100/1562 [00:08<00:51, 28.15it/s][A
  7%|▋         | 103/1562 [00:08<00:51, 28.06it/s][A
  7%|▋         | 106/1562 [00:08<00:51, 28.04it/s][A
  7%|▋         | 109/1562 [00:08<00:51, 28.02it/s][A
  7%|▋         | 112/1562 [00:08<00:52, 27.77it/s][A
  7%|▋         | 115/1562 [00:09<00:51, 27.98it/s][A
  8%|▊         | 118/1562 [00:09<00:51, 28.15it/s][A
  8%|▊         | 121/1562 [00:09<00:51, 28.23it/s][A
  8%|▊         | 124/1562 [00:09<00:50, 28.36it/s][A
  8%|▊         | 127/1562 [00:09<00:50, 28.29it/s][A
  8%|▊         | 130/1562 [00:09<00:50, 28.22it/s][A
  9%|▊         | 133/1562 [00:09<00:51, 27.94it/s][A
  9%|▊         | 136/1562 [00:09<00:51, 27.67it/s][A
  9%|▉         | 139/1562 [00:09<00:51, 27.58it/s][A
  9%|▉         | 142/1562 [00:10<00:51, 27.47it/s][A
  9%|▉         | 145/1562 [00:10<00:51, 27.38it/s][A
  9%|▉         | 148/1562 [00:10<00:51, 27.27it/s][A
 10%|▉         | 151/1562 [00:10<00:51, 27.14it/s][A
 10%|▉         | 154/1562 [00:10<00:51, 27.09it/s][A
 10%|█         | 157/1562 [00:10<00:51, 27.08it/s][A
 10%|█         | 160/1562 [00:10<00:52, 26.96it/s][A
 10%|█         | 163/1562 [00:10<00:52, 26.83it/s][A
 11%|█         | 166/1562 [00:10<00:51, 26.92it/s][A
 11%|█         | 169/1562 [00:11<00:51, 26.90it/s][A
 11%|█         | 172/1562 [00:11<00:51, 26.84it/s][A
 11%|█         | 175/1562 [00:11<00:51, 26.88it/s][A
 11%|█▏        | 178/1562 [00:11<00:51, 26.99it/s][A
 12%|█▏        | 181/1562 [00:11<00:50, 27.15it/s][A
 12%|█▏        | 184/1562 [00:11<00:50, 27.22it/s][A
 12%|█▏        | 187/1562 [00:11<00:50, 27.30it/s][A
 12%|█▏        | 190/1562 [00:11<00:50, 27.28it/s][A
 12%|█▏        | 193/1562 [00:11<00:50, 27.33it/s][A
 13%|█▎        | 196/1562 [00:12<00:49, 27.47it/s][A
 13%|█▎        | 199/1562 [00:12<00:49, 27.68it/s][A
 13%|█▎        | 202/1562 [00:12<00:49, 27.70it/s][A
 13%|█▎        | 205/1562 [00:12<00:49, 27.62it/s][A
 13%|█▎        | 208/1562 [00:12<00:49, 27.56it/s][A
 14%|█▎        | 211/1562 [00:12<00:48, 27.71it/s][A
 14%|█▎        | 214/1562 [00:12<00:48, 27.84it/s][A
 14%|█▍        | 217/1562 [00:12<00:48, 27.90it/s][A
 14%|█▍        | 220/1562 [00:12<00:48, 27.76it/s][A
 14%|█▍        | 223/1562 [00:13<00:48, 27.80it/s][A
 14%|█▍        | 226/1562 [00:13<00:47, 27.92it/s][A
 15%|█▍        | 229/1562 [00:13<00:47, 27.96it/s][A
 15%|█▍        | 232/1562 [00:13<00:47, 28.07it/s][A
 15%|█▌        | 235/1562 [00:13<00:47, 28.07it/s][A
 15%|█▌        | 238/1562 [00:13<00:47, 28.13it/s][A
 15%|█▌        | 241/1562 [00:13<00:47, 28.06it/s][A
 16%|█▌        | 244/1562 [00:13<00:47, 27.87it/s][A
 16%|█▌        | 247/1562 [00:13<00:47, 27.44it/s][A
 16%|█▌        | 250/1562 [00:13<00:48, 27.24it/s][A
 16%|█▌        | 253/1562 [00:14<00:48, 27.21it/s][A
 16%|█▋        | 256/1562 [00:14<00:47, 27.30it/s][A
 17%|█▋        | 259/1562 [00:14<00:47, 27.28it/s][A
 17%|█▋        | 262/1562 [00:14<00:47, 27.39it/s][A
 17%|█▋        | 265/1562 [00:14<00:47, 27.44it/s][A
 17%|█▋        | 268/1562 [00:14<00:47, 27.42it/s][A
 17%|█▋        | 271/1562 [00:14<00:47, 27.47it/s][A
 18%|█▊        | 274/1562 [00:14<00:46, 27.49it/s][A
 18%|█▊        | 277/1562 [00:14<00:46, 27.52it/s][A
 18%|█▊        | 280/1562 [00:15<00:46, 27.54it/s][A
 18%|█▊        | 283/1562 [00:15<00:46, 27.60it/s][A
 18%|█▊        | 286/1562 [00:15<00:46, 27.46it/s][A
 19%|█▊        | 289/1562 [00:15<00:46, 27.42it/s][A
 19%|█▊        | 292/1562 [00:15<00:46, 27.45it/s][A
 19%|█▉        | 295/1562 [00:15<00:46, 27.43it/s][A
 19%|█▉        | 298/1562 [00:15<00:46, 27.41it/s][A
 19%|█▉        | 301/1562 [00:15<00:45, 27.46it/s][A
 19%|█▉        | 304/1562 [00:15<00:45, 27.46it/s][A
 20%|█▉        | 307/1562 [00:16<00:45, 27.46it/s][A
 20%|█▉        | 310/1562 [00:16<00:45, 27.45it/s][A
 20%|██        | 313/1562 [00:16<00:45, 27.26it/s][A
 20%|██        | 316/1562 [00:16<00:45, 27.28it/s][A
 20%|██        | 319/1562 [00:16<00:45, 27.17it/s][A
 21%|██        | 322/1562 [00:16<00:45, 27.28it/s][A
 21%|██        | 325/1562 [00:16<00:45, 27.41it/s][A
 21%|██        | 328/1562 [00:16<00:45, 27.39it/s][A
 21%|██        | 331/1562 [00:16<00:44, 27.39it/s][A
 21%|██▏       | 334/1562 [00:17<00:44, 27.38it/s][A
 22%|██▏       | 337/1562 [00:17<00:44, 27.45it/s][A
 22%|██▏       | 340/1562 [00:17<00:44, 27.50it/s][A
 22%|██▏       | 343/1562 [00:17<00:44, 27.33it/s][A
 22%|██▏       | 346/1562 [00:17<00:44, 27.47it/s][A
 22%|██▏       | 349/1562 [00:17<00:44, 27.55it/s][A
 23%|██▎       | 352/1562 [00:17<00:43, 27.58it/s][A
 23%|██▎       | 355/1562 [00:17<00:43, 27.55it/s][A
 23%|██▎       | 358/1562 [00:17<00:43, 27.59it/s][A
 23%|██▎       | 361/1562 [00:18<00:43, 27.63it/s][A
 23%|██▎       | 364/1562 [00:18<00:43, 27.67it/s][A
 23%|██▎       | 367/1562 [00:18<00:43, 27.71it/s][A
 24%|██▎       | 370/1562 [00:18<00:42, 27.74it/s][A
 24%|██▍       | 373/1562 [00:18<00:42, 27.70it/s][A
 24%|██▍       | 376/1562 [00:18<00:42, 27.69it/s][A
 24%|██▍       | 379/1562 [00:18<00:42, 27.74it/s][A
 24%|██▍       | 382/1562 [00:18<00:42, 27.77it/s][A
 25%|██▍       | 385/1562 [00:18<00:42, 27.72it/s][A
 25%|██▍       | 388/1562 [00:19<00:42, 27.73it/s][A
 25%|██▌       | 391/1562 [00:19<00:42, 27.69it/s][A
 25%|██▌       | 394/1562 [00:19<00:42, 27.71it/s][A
 25%|██▌       | 397/1562 [00:19<00:42, 27.72it/s][A
 26%|██▌       | 400/1562 [00:19<00:41, 27.76it/s][A
 26%|██▌       | 403/1562 [00:19<00:41, 27.77it/s][A
 26%|██▌       | 406/1562 [00:19<00:41, 27.83it/s][A
 26%|██▌       | 409/1562 [00:19<00:41, 27.83it/s][A
 26%|██▋       | 412/1562 [00:19<00:41, 27.83it/s][A
 27%|██▋       | 415/1562 [00:19<00:41, 27.78it/s][A
 27%|██▋       | 418/1562 [00:20<00:41, 27.78it/s][A
 27%|██▋       | 421/1562 [00:20<00:41, 27.75it/s][A
 27%|██▋       | 424/1562 [00:20<00:40, 27.77it/s][A
 27%|██▋       | 427/1562 [00:20<00:40, 27.81it/s][A
 28%|██▊       | 430/1562 [00:20<00:40, 27.81it/s][A
 28%|██▊       | 433/1562 [00:20<00:40, 27.73it/s][A
 28%|██▊       | 436/1562 [00:20<00:40, 27.59it/s][A
 28%|██▊       | 439/1562 [00:20<00:40, 27.56it/s][A
 28%|██▊       | 442/1562 [00:20<00:40, 27.45it/s][A
 28%|██▊       | 445/1562 [00:21<00:40, 27.35it/s][A
 29%|██▊       | 448/1562 [00:21<00:40, 27.33it/s][A
 29%|██▉       | 451/1562 [00:21<00:40, 27.46it/s][A
 29%|██▉       | 454/1562 [00:21<00:40, 27.52it/s][A
 29%|██▉       | 457/1562 [00:21<00:40, 27.52it/s][A
 29%|██▉       | 460/1562 [00:21<00:40, 27.41it/s][A
 30%|██▉       | 463/1562 [00:21<00:39, 27.48it/s][A
 30%|██▉       | 466/1562 [00:21<00:39, 27.52it/s][A
 30%|███       | 469/1562 [00:21<00:39, 27.62it/s][A
 30%|███       | 472/1562 [00:22<00:39, 27.50it/s][A
 30%|███       | 475/1562 [00:22<00:39, 27.48it/s][A
 31%|███       | 478/1562 [00:22<00:39, 27.59it/s][A
 31%|███       | 481/1562 [00:22<00:39, 27.64it/s][A
 31%|███       | 484/1562 [00:22<00:39, 27.62it/s][A
 31%|███       | 487/1562 [00:22<00:39, 27.49it/s][A
 31%|███▏      | 490/1562 [00:22<00:38, 27.59it/s][A
 32%|███▏      | 493/1562 [00:22<00:38, 27.58it/s][A
 32%|███▏      | 496/1562 [00:22<00:38, 27.63it/s][A
 32%|███▏      | 499/1562 [00:23<00:38, 27.58it/s][A
 32%|███▏      | 502/1562 [00:23<00:38, 27.58it/s][A
 32%|███▏      | 505/1562 [00:23<00:38, 27.53it/s][A
 33%|███▎      | 508/1562 [00:23<00:38, 27.59it/s][A
 33%|███▎      | 511/1562 [00:23<00:38, 27.59it/s][A
 33%|███▎      | 514/1562 [00:23<00:37, 27.59it/s][A
 33%|███▎      | 517/1562 [00:23<00:37, 27.60it/s][A
 33%|███▎      | 520/1562 [00:23<00:37, 27.65it/s][A
 33%|███▎      | 523/1562 [00:23<00:37, 27.55it/s][A
 34%|███▎      | 526/1562 [00:24<00:37, 27.64it/s][A
 34%|███▍      | 529/1562 [00:24<00:37, 27.74it/s][A
 34%|███▍      | 532/1562 [00:24<00:37, 27.78it/s][A
 34%|███▍      | 535/1562 [00:24<00:37, 27.74it/s][A
 34%|███▍      | 538/1562 [00:24<00:36, 27.70it/s][A
 35%|███▍      | 541/1562 [00:24<00:36, 27.76it/s][A
 35%|███▍      | 544/1562 [00:24<00:36, 27.84it/s][A
 35%|███▌      | 547/1562 [00:24<00:36, 27.79it/s][A
 35%|███▌      | 550/1562 [00:24<00:36, 27.87it/s][A
 35%|███▌      | 553/1562 [00:24<00:36, 27.75it/s][A
 36%|███▌      | 556/1562 [00:25<00:36, 27.74it/s][A
 36%|███▌      | 559/1562 [00:25<00:36, 27.82it/s][A
 36%|███▌      | 562/1562 [00:25<00:36, 27.74it/s][A
 36%|███▌      | 565/1562 [00:25<00:35, 27.84it/s][A
 36%|███▋      | 568/1562 [00:25<00:35, 27.86it/s][A
 37%|███▋      | 571/1562 [00:25<00:35, 27.71it/s][A
 37%|███▋      | 574/1562 [00:25<00:35, 27.69it/s][A
 37%|███▋      | 577/1562 [00:25<00:35, 27.62it/s][A
 37%|███▋      | 580/1562 [00:25<00:35, 27.58it/s][A
 37%|███▋      | 583/1562 [00:26<00:35, 27.56it/s][A
 38%|███▊      | 586/1562 [00:26<00:35, 27.57it/s][A
 38%|███▊      | 589/1562 [00:26<00:35, 27.59it/s][A
 38%|███▊      | 592/1562 [00:26<00:35, 27.62it/s][A
 38%|███▊      | 595/1562 [00:26<00:35, 27.61it/s][A
 38%|███▊      | 598/1562 [00:26<00:34, 27.64it/s][A
 38%|███▊      | 601/1562 [00:26<00:34, 27.55it/s][A
 39%|███▊      | 604/1562 [00:26<00:34, 27.56it/s][A
 39%|███▉      | 607/1562 [00:26<00:34, 27.57it/s][A
 39%|███▉      | 610/1562 [00:27<00:34, 27.53it/s][A
 39%|███▉      | 613/1562 [00:27<00:34, 27.54it/s][A
 39%|███▉      | 616/1562 [00:27<00:34, 27.54it/s][A
 40%|███▉      | 619/1562 [00:27<00:34, 27.46it/s][A
 40%|███▉      | 622/1562 [00:27<00:34, 27.53it/s][A
 40%|████      | 625/1562 [00:27<00:34, 27.54it/s][A
 40%|████      | 628/1562 [00:27<00:34, 27.10it/s][A
 40%|████      | 631/1562 [00:27<00:34, 27.29it/s][A
 41%|████      | 634/1562 [00:27<00:33, 27.31it/s][A
 41%|████      | 637/1562 [00:28<00:33, 27.24it/s][A
 41%|████      | 640/1562 [00:28<00:33, 27.19it/s][A
 41%|████      | 643/1562 [00:28<00:33, 27.22it/s][A
 41%|████▏     | 646/1562 [00:28<00:33, 27.26it/s][A
 42%|████▏     | 649/1562 [00:28<00:33, 27.38it/s][A
 42%|████▏     | 652/1562 [00:28<00:33, 27.31it/s][A
 42%|████▏     | 655/1562 [00:28<00:33, 27.40it/s][A
 42%|████▏     | 658/1562 [00:28<00:32, 27.48it/s][A
 42%|████▏     | 661/1562 [00:28<00:32, 27.55it/s][A
 43%|████▎     | 664/1562 [00:29<00:32, 27.49it/s][A
 43%|████▎     | 667/1562 [00:29<00:32, 27.45it/s][A
 43%|████▎     | 670/1562 [00:29<00:32, 27.47it/s][A
 43%|████▎     | 673/1562 [00:29<00:32, 27.57it/s][A
 43%|████▎     | 676/1562 [00:29<00:32, 27.53it/s][A
 43%|████▎     | 679/1562 [00:29<00:32, 27.49it/s][A
 44%|████▎     | 682/1562 [00:29<00:31, 27.50it/s][A
 44%|████▍     | 685/1562 [00:29<00:31, 27.49it/s][A
 44%|████▍     | 688/1562 [00:29<00:31, 27.52it/s][A
 44%|████▍     | 691/1562 [00:29<00:31, 27.57it/s][A
 44%|████▍     | 694/1562 [00:30<00:31, 27.55it/s][A
 45%|████▍     | 697/1562 [00:30<00:31, 27.38it/s][A
 45%|████▍     | 700/1562 [00:30<00:31, 27.45it/s][A
 45%|████▌     | 703/1562 [00:30<00:31, 27.40it/s][A
 45%|████▌     | 706/1562 [00:30<00:31, 27.26it/s][A
 45%|████▌     | 709/1562 [00:30<00:31, 27.29it/s][A
 46%|████▌     | 712/1562 [00:30<00:31, 27.40it/s][A
 46%|████▌     | 715/1562 [00:30<00:30, 27.42it/s][A
 46%|████▌     | 718/1562 [00:30<00:30, 27.52it/s][A
 46%|████▌     | 721/1562 [00:31<00:30, 27.60it/s][A
 46%|████▋     | 724/1562 [00:31<00:30, 27.62it/s][A
 47%|████▋     | 727/1562 [00:31<00:30, 27.57it/s][A
 47%|████▋     | 730/1562 [00:31<00:30, 27.54it/s][A
 47%|████▋     | 733/1562 [00:31<00:30, 27.53it/s][A
 47%|████▋     | 736/1562 [00:31<00:29, 27.63it/s][A
 47%|████▋     | 739/1562 [00:31<00:29, 27.66it/s][A
 48%|████▊     | 742/1562 [00:31<00:29, 27.62it/s][A
 48%|████▊     | 745/1562 [00:31<00:29, 27.60it/s][A
 48%|████▊     | 748/1562 [00:32<00:29, 27.62it/s][A
 48%|████▊     | 751/1562 [00:32<00:29, 27.69it/s][A
 48%|████▊     | 754/1562 [00:32<00:29, 27.65it/s][A
 48%|████▊     | 757/1562 [00:32<00:29, 27.60it/s][A
 49%|████▊     | 760/1562 [00:32<00:29, 27.63it/s][A
 49%|████▉     | 763/1562 [00:32<00:28, 27.57it/s][A
 49%|████▉     | 766/1562 [00:32<00:28, 27.60it/s][A
 49%|████▉     | 769/1562 [00:32<00:29, 27.29it/s][A
 49%|████▉     | 772/1562 [00:32<00:28, 27.25it/s][A
 50%|████▉     | 775/1562 [00:33<00:28, 27.41it/s][A
 50%|████▉     | 778/1562 [00:33<00:28, 27.48it/s][A
 50%|█████     | 781/1562 [00:33<00:28, 27.51it/s][A
 50%|█████     | 784/1562 [00:33<00:28, 27.40it/s][A
 50%|█████     | 787/1562 [00:33<00:28, 27.48it/s][A
 51%|█████     | 790/1562 [00:33<00:28, 27.54it/s][A
 51%|█████     | 793/1562 [00:33<00:27, 27.61it/s][A
 51%|█████     | 796/1562 [00:33<00:27, 27.65it/s][A
 51%|█████     | 799/1562 [00:33<00:27, 27.64it/s][A
 51%|█████▏    | 802/1562 [00:34<00:27, 27.64it/s][A
 52%|█████▏    | 805/1562 [00:34<00:27, 27.61it/s][A
 52%|█████▏    | 808/1562 [00:34<00:27, 27.67it/s][A
 52%|█████▏    | 811/1562 [00:34<00:27, 27.68it/s][A
 52%|█████▏    | 814/1562 [00:34<00:27, 27.65it/s][A
 52%|█████▏    | 817/1562 [00:34<00:26, 27.63it/s][A
 52%|█████▏    | 820/1562 [00:34<00:26, 27.64it/s][A
 53%|█████▎    | 823/1562 [00:34<00:26, 27.63it/s][A
 53%|█████▎    | 826/1562 [00:34<00:26, 27.70it/s][A
 53%|█████▎    | 829/1562 [00:35<00:26, 27.69it/s][A
 53%|█████▎    | 832/1562 [00:35<00:26, 27.70it/s][A
 53%|█████▎    | 835/1562 [00:35<00:26, 27.61it/s][A
 54%|█████▎    | 838/1562 [00:35<00:26, 27.56it/s][A
 54%|█████▍    | 841/1562 [00:35<00:26, 27.59it/s][A
 54%|█████▍    | 844/1562 [00:35<00:25, 27.62it/s][A
 54%|█████▍    | 847/1562 [00:35<00:25, 27.64it/s][A
 54%|█████▍    | 850/1562 [00:35<00:25, 27.70it/s][A
 55%|█████▍    | 853/1562 [00:35<00:25, 27.67it/s][A
 55%|█████▍    | 856/1562 [00:35<00:25, 27.72it/s][A
 55%|█████▍    | 859/1562 [00:36<00:25, 27.67it/s][A
 55%|█████▌    | 862/1562 [00:36<00:25, 27.68it/s][A
 55%|█████▌    | 865/1562 [00:36<00:25, 27.70it/s][A
 56%|█████▌    | 868/1562 [00:36<00:25, 27.74it/s][A
 56%|█████▌    | 871/1562 [00:36<00:24, 27.69it/s][A
 56%|█████▌    | 874/1562 [00:36<00:24, 27.74it/s][A
 56%|█████▌    | 877/1562 [00:36<00:24, 27.71it/s][A
 56%|█████▋    | 880/1562 [00:36<00:24, 27.69it/s][A
 57%|█████▋    | 883/1562 [00:36<00:24, 27.67it/s][A
 57%|█████▋    | 886/1562 [00:37<00:24, 27.67it/s][A
 57%|█████▋    | 889/1562 [00:37<00:24, 27.67it/s][A
 57%|█████▋    | 892/1562 [00:37<00:24, 27.61it/s][A
 57%|█████▋    | 895/1562 [00:37<00:24, 27.62it/s][A
 57%|█████▋    | 898/1562 [00:37<00:24, 27.59it/s][A
 58%|█████▊    | 901/1562 [00:37<00:24, 27.52it/s][A
 58%|█████▊    | 904/1562 [00:37<00:23, 27.49it/s][A
 58%|█████▊    | 907/1562 [00:37<00:23, 27.55it/s][A
 58%|█████▊    | 910/1562 [00:37<00:23, 27.63it/s][A
 58%|█████▊    | 913/1562 [00:38<00:23, 27.55it/s][A
 59%|█████▊    | 916/1562 [00:38<00:23, 27.56it/s][A
 59%|█████▉    | 919/1562 [00:38<00:23, 27.63it/s][A
 59%|█████▉    | 922/1562 [00:38<00:23, 27.67it/s][A
 59%|█████▉    | 925/1562 [00:38<00:22, 27.71it/s][A
 59%|█████▉    | 928/1562 [00:38<00:22, 27.73it/s][A
 60%|█████▉    | 931/1562 [00:38<00:22, 27.75it/s][A
 60%|█████▉    | 934/1562 [00:38<00:22, 27.76it/s][A
 60%|█████▉    | 937/1562 [00:38<00:22, 27.72it/s][A
 60%|██████    | 940/1562 [00:39<00:22, 27.68it/s][A
 60%|██████    | 943/1562 [00:39<00:22, 27.72it/s][A
 61%|██████    | 946/1562 [00:39<00:22, 27.75it/s][A
 61%|██████    | 949/1562 [00:39<00:22, 27.70it/s][A
 61%|██████    | 952/1562 [00:39<00:22, 27.69it/s][A
 61%|██████    | 955/1562 [00:39<00:21, 27.69it/s][A
 61%|██████▏   | 958/1562 [00:39<00:21, 27.73it/s][A
 62%|██████▏   | 961/1562 [00:39<00:21, 27.72it/s][A
 62%|██████▏   | 964/1562 [00:39<00:21, 27.63it/s][A
 62%|██████▏   | 967/1562 [00:39<00:21, 27.54it/s][A
 62%|██████▏   | 970/1562 [00:40<00:21, 27.60it/s][A
 62%|██████▏   | 973/1562 [00:40<00:21, 27.60it/s][A
 62%|██████▏   | 976/1562 [00:40<00:21, 27.62it/s][A
 63%|██████▎   | 979/1562 [00:40<00:21, 27.59it/s][A
 63%|██████▎   | 982/1562 [00:40<00:21, 27.46it/s][A
 63%|██████▎   | 985/1562 [00:40<00:20, 27.49it/s][A
 63%|██████▎   | 988/1562 [00:40<00:20, 27.50it/s][A
 63%|██████▎   | 991/1562 [00:40<00:20, 27.58it/s][A
 64%|██████▎   | 994/1562 [00:40<00:20, 27.63it/s][A
 64%|██████▍   | 997/1562 [00:41<00:20, 27.71it/s][A
 64%|██████▍   | 1000/1562 [00:41<00:20, 27.67it/s][A
 64%|██████▍   | 1003/1562 [00:41<00:20, 27.64it/s][A
 64%|██████▍   | 1006/1562 [00:41<00:20, 27.62it/s][A
 65%|██████▍   | 1009/1562 [00:41<00:20, 27.61it/s][A
 65%|██████▍   | 1012/1562 [00:41<00:19, 27.66it/s][A
 65%|██████▍   | 1015/1562 [00:41<00:19, 27.68it/s][A
 65%|██████▌   | 1018/1562 [00:41<00:19, 27.72it/s][A
 65%|██████▌   | 1021/1562 [00:41<00:19, 27.69it/s][A
 66%|██████▌   | 1024/1562 [00:42<00:19, 27.51it/s][A
 66%|██████▌   | 1027/1562 [00:42<00:19, 27.46it/s][A
 66%|██████▌   | 1030/1562 [00:42<00:19, 27.49it/s][A
 66%|██████▌   | 1033/1562 [00:42<00:19, 27.56it/s][A
 66%|██████▋   | 1036/1562 [00:42<00:19, 27.59it/s][A
 67%|██████▋   | 1039/1562 [00:42<00:18, 27.63it/s][A
 67%|██████▋   | 1042/1562 [00:42<00:18, 27.63it/s][A
 67%|██████▋   | 1045/1562 [00:42<00:18, 27.63it/s][A
 67%|██████▋   | 1048/1562 [00:42<00:18, 27.67it/s][A
 67%|██████▋   | 1051/1562 [00:43<00:18, 27.64it/s][A
 67%|██████▋   | 1054/1562 [00:43<00:18, 27.67it/s][A
 68%|██████▊   | 1057/1562 [00:43<00:18, 27.64it/s][A
 68%|██████▊   | 1060/1562 [00:43<00:18, 27.72it/s][A
 68%|██████▊   | 1063/1562 [00:43<00:18, 27.70it/s][A
 68%|██████▊   | 1066/1562 [00:43<00:17, 27.73it/s][A
 68%|██████▊   | 1069/1562 [00:43<00:17, 27.74it/s][A
 69%|██████▊   | 1072/1562 [00:43<00:17, 27.68it/s][A
 69%|██████▉   | 1075/1562 [00:43<00:17, 27.63it/s][A
 69%|██████▉   | 1078/1562 [00:44<00:17, 27.65it/s][A
 69%|██████▉   | 1081/1562 [00:44<00:17, 27.71it/s][A
 69%|██████▉   | 1084/1562 [00:44<00:17, 27.75it/s][A
 70%|██████▉   | 1087/1562 [00:44<00:17, 27.71it/s][A
 70%|██████▉   | 1090/1562 [00:44<00:17, 27.73it/s][A
 70%|██████▉   | 1093/1562 [00:44<00:16, 27.74it/s][A
 70%|███████   | 1096/1562 [00:44<00:16, 27.72it/s][A
 70%|███████   | 1099/1562 [00:44<00:16, 27.71it/s][A
 71%|███████   | 1102/1562 [00:44<00:16, 27.73it/s][A
 71%|███████   | 1105/1562 [00:44<00:16, 27.52it/s][A
 71%|███████   | 1108/1562 [00:45<00:16, 27.55it/s][A
 71%|███████   | 1111/1562 [00:45<00:16, 27.55it/s][A
 71%|███████▏  | 1114/1562 [00:45<00:16, 27.58it/s][A
 72%|███████▏  | 1117/1562 [00:45<00:16, 27.62it/s][A
 72%|███████▏  | 1120/1562 [00:45<00:15, 27.64it/s][A
 72%|███████▏  | 1123/1562 [00:45<00:15, 27.69it/s][A
 72%|███████▏  | 1126/1562 [00:45<00:15, 27.72it/s][A
 72%|███████▏  | 1129/1562 [00:45<00:15, 27.68it/s][A
 72%|███████▏  | 1132/1562 [00:45<00:15, 27.68it/s][A
 73%|███████▎  | 1135/1562 [00:46<00:15, 27.71it/s][A
 73%|███████▎  | 1138/1562 [00:46<00:15, 27.72it/s][A
 73%|███████▎  | 1141/1562 [00:46<00:15, 27.68it/s][A
 73%|███████▎  | 1144/1562 [00:46<00:15, 27.72it/s][A
 73%|███████▎  | 1147/1562 [00:46<00:14, 27.67it/s][A
 74%|███████▎  | 1150/1562 [00:46<00:14, 27.76it/s][A
 74%|███████▍  | 1153/1562 [00:46<00:14, 27.74it/s][A
 74%|███████▍  | 1156/1562 [00:46<00:14, 27.72it/s][A
 74%|███████▍  | 1159/1562 [00:46<00:14, 27.74it/s][A
 74%|███████▍  | 1162/1562 [00:47<00:14, 27.70it/s][A
 75%|███████▍  | 1165/1562 [00:47<00:14, 27.73it/s][A
 75%|███████▍  | 1168/1562 [00:47<00:14, 27.74it/s][A
 75%|███████▍  | 1171/1562 [00:47<00:14, 27.71it/s][A
 75%|███████▌  | 1174/1562 [00:47<00:13, 27.75it/s][A
 75%|███████▌  | 1177/1562 [00:47<00:13, 27.74it/s][A
 76%|███████▌  | 1180/1562 [00:47<00:13, 27.72it/s][A
 76%|███████▌  | 1183/1562 [00:47<00:13, 27.73it/s][A
 76%|███████▌  | 1186/1562 [00:47<00:13, 27.71it/s][A
 76%|███████▌  | 1189/1562 [00:48<00:13, 27.70it/s][A
 76%|███████▋  | 1192/1562 [00:48<00:13, 27.76it/s][A
 77%|███████▋  | 1195/1562 [00:48<00:13, 27.74it/s][A
 77%|███████▋  | 1198/1562 [00:48<00:13, 27.74it/s][A
 77%|███████▋  | 1201/1562 [00:48<00:13, 27.72it/s][A
 77%|███████▋  | 1204/1562 [00:48<00:12, 27.70it/s][A
 77%|███████▋  | 1207/1562 [00:48<00:12, 27.71it/s][A
 77%|███████▋  | 1210/1562 [00:48<00:12, 27.74it/s][A
 78%|███████▊  | 1213/1562 [00:48<00:12, 27.79it/s][A
 78%|███████▊  | 1216/1562 [00:48<00:12, 27.78it/s][A
 78%|███████▊  | 1219/1562 [00:49<00:12, 27.75it/s][A
 78%|███████▊  | 1222/1562 [00:49<00:12, 27.72it/s][A
 78%|███████▊  | 1225/1562 [00:49<00:12, 27.70it/s][A
 79%|███████▊  | 1228/1562 [00:49<00:12, 27.72it/s][A
 79%|███████▉  | 1231/1562 [00:49<00:11, 27.68it/s][A
 79%|███████▉  | 1234/1562 [00:49<00:11, 27.67it/s][A
 79%|███████▉  | 1237/1562 [00:49<00:11, 27.64it/s][A
 79%|███████▉  | 1240/1562 [00:49<00:11, 27.68it/s][A
 80%|███████▉  | 1243/1562 [00:49<00:11, 27.60it/s][A
 80%|███████▉  | 1246/1562 [00:50<00:11, 27.70it/s][A
 80%|███████▉  | 1249/1562 [00:50<00:11, 27.76it/s][A
 80%|████████  | 1252/1562 [00:50<00:11, 27.80it/s][A
 80%|████████  | 1255/1562 [00:50<00:11, 27.75it/s][A
 81%|████████  | 1258/1562 [00:50<00:10, 27.69it/s][A
 81%|████████  | 1261/1562 [00:50<00:10, 27.71it/s][A
 81%|████████  | 1264/1562 [00:50<00:10, 27.63it/s][A
 81%|████████  | 1267/1562 [00:50<00:10, 27.60it/s][A
 81%|████████▏ | 1270/1562 [00:50<00:10, 27.62it/s][A
 81%|████████▏ | 1273/1562 [00:51<00:10, 27.65it/s][A
 82%|████████▏ | 1276/1562 [00:51<00:10, 27.72it/s][A
 82%|████████▏ | 1279/1562 [00:51<00:10, 27.72it/s][A
 82%|████████▏ | 1282/1562 [00:51<00:10, 27.64it/s][A
 82%|████████▏ | 1285/1562 [00:51<00:10, 27.66it/s][A
 82%|████████▏ | 1288/1562 [00:51<00:09, 27.67it/s][A
 83%|████████▎ | 1291/1562 [00:51<00:09, 27.69it/s][A
 83%|████████▎ | 1294/1562 [00:51<00:09, 27.68it/s][A
 83%|████████▎ | 1297/1562 [00:51<00:09, 27.68it/s][A
 83%|████████▎ | 1300/1562 [00:52<00:09, 27.64it/s][A
 83%|████████▎ | 1303/1562 [00:52<00:09, 27.63it/s][A
 84%|████████▎ | 1306/1562 [00:52<00:09, 27.70it/s][A
 84%|████████▍ | 1309/1562 [00:52<00:09, 27.72it/s][A
 84%|████████▍ | 1312/1562 [00:52<00:09, 27.74it/s][A
 84%|████████▍ | 1315/1562 [00:52<00:08, 27.68it/s][A
 84%|████████▍ | 1318/1562 [00:52<00:08, 27.68it/s][A
 85%|████████▍ | 1321/1562 [00:52<00:08, 27.66it/s][A
 85%|████████▍ | 1324/1562 [00:52<00:08, 27.70it/s][A
 85%|████████▍ | 1327/1562 [00:53<00:08, 27.68it/s][A
 85%|████████▌ | 1330/1562 [00:53<00:08, 27.64it/s][A
 85%|████████▌ | 1333/1562 [00:53<00:08, 27.63it/s][A
 86%|████████▌ | 1336/1562 [00:53<00:08, 27.66it/s][A
 86%|████████▌ | 1339/1562 [00:53<00:08, 27.50it/s][A
 86%|████████▌ | 1342/1562 [00:53<00:07, 27.60it/s][A
 86%|████████▌ | 1345/1562 [00:53<00:07, 27.63it/s][A
 86%|████████▋ | 1348/1562 [00:53<00:07, 27.62it/s][A
 86%|████████▋ | 1351/1562 [00:53<00:07, 27.62it/s][A
 87%|████████▋ | 1354/1562 [00:53<00:07, 27.68it/s][A
 87%|████████▋ | 1357/1562 [00:54<00:07, 27.67it/s][A
 87%|████████▋ | 1360/1562 [00:54<00:07, 27.65it/s][A
 87%|████████▋ | 1363/1562 [00:54<00:07, 27.63it/s][A
 87%|████████▋ | 1366/1562 [00:54<00:07, 27.63it/s][A
 88%|████████▊ | 1369/1562 [00:54<00:06, 27.69it/s][A
 88%|████████▊ | 1372/1562 [00:54<00:06, 27.64it/s][A
 88%|████████▊ | 1375/1562 [00:54<00:06, 27.65it/s][A
 88%|████████▊ | 1378/1562 [00:54<00:06, 27.64it/s][A
 88%|████████▊ | 1381/1562 [00:54<00:06, 27.70it/s][A
 89%|████████▊ | 1384/1562 [00:55<00:06, 27.54it/s][A
 89%|████████▉ | 1387/1562 [00:55<00:06, 27.61it/s][A
 89%|████████▉ | 1390/1562 [00:55<00:06, 27.65it/s][A
 89%|████████▉ | 1393/1562 [00:55<00:06, 27.68it/s][A
 89%|████████▉ | 1396/1562 [00:55<00:06, 27.64it/s][A
 90%|████████▉ | 1399/1562 [00:55<00:05, 27.67it/s][A
 90%|████████▉ | 1402/1562 [00:55<00:05, 27.70it/s][A
 90%|████████▉ | 1405/1562 [00:55<00:05, 27.71it/s][A
 90%|█████████ | 1408/1562 [00:55<00:05, 27.72it/s][A
 90%|█████████ | 1411/1562 [00:56<00:05, 27.69it/s][A
 91%|█████████ | 1414/1562 [00:56<00:05, 27.64it/s][A
 91%|█████████ | 1417/1562 [00:56<00:05, 27.63it/s][A
 91%|█████████ | 1420/1562 [00:56<00:05, 27.68it/s][A
 91%|█████████ | 1423/1562 [00:56<00:05, 27.66it/s][A
 91%|█████████▏| 1426/1562 [00:56<00:04, 27.67it/s][A
 91%|█████████▏| 1429/1562 [00:56<00:04, 27.63it/s][A
 92%|█████████▏| 1432/1562 [00:56<00:04, 27.71it/s][A
 92%|█████████▏| 1435/1562 [00:56<00:04, 27.70it/s][A
 92%|█████████▏| 1438/1562 [00:57<00:04, 27.87it/s][A
 92%|█████████▏| 1441/1562 [00:57<00:04, 28.00it/s][A
 92%|█████████▏| 1444/1562 [00:57<00:04, 28.06it/s][A
 93%|█████████▎| 1447/1562 [00:57<00:04, 28.11it/s][A
 93%|█████████▎| 1450/1562 [00:57<00:03, 28.15it/s][A
 93%|█████████▎| 1453/1562 [00:57<00:03, 28.19it/s][A
 93%|█████████▎| 1456/1562 [00:57<00:03, 28.20it/s][A
 93%|█████████▎| 1459/1562 [00:57<00:03, 28.22it/s][A
 94%|█████████▎| 1462/1562 [00:57<00:03, 28.25it/s][A
 94%|█████████▍| 1465/1562 [00:57<00:03, 28.25it/s][A
 94%|█████████▍| 1468/1562 [00:58<00:03, 28.25it/s][A
 94%|█████████▍| 1471/1562 [00:58<00:03, 28.24it/s][A
 94%|█████████▍| 1474/1562 [00:58<00:03, 28.25it/s][A
 95%|█████████▍| 1477/1562 [00:58<00:03, 28.19it/s][A
 95%|█████████▍| 1480/1562 [00:58<00:02, 28.17it/s][A
 95%|█████████▍| 1483/1562 [00:58<00:02, 28.19it/s][A
 95%|█████████▌| 1486/1562 [00:58<00:02, 28.22it/s][A
 95%|█████████▌| 1489/1562 [00:58<00:02, 28.25it/s][A
 96%|█████████▌| 1492/1562 [00:58<00:02, 28.28it/s][A
 96%|█████████▌| 1495/1562 [00:59<00:02, 28.27it/s][A
 96%|█████████▌| 1498/1562 [00:59<00:02, 28.28it/s][A
 96%|█████████▌| 1501/1562 [00:59<00:02, 28.26it/s][A
 96%|█████████▋| 1504/1562 [00:59<00:02, 28.26it/s][A
 96%|█████████▋| 1507/1562 [00:59<00:01, 28.26it/s][A
 97%|█████████▋| 1510/1562 [00:59<00:01, 28.29it/s][A
 97%|█████████▋| 1513/1562 [00:59<00:01, 28.29it/s][A
 97%|█████████▋| 1516/1562 [00:59<00:01, 28.30it/s][A
 97%|█████████▋| 1519/1562 [00:59<00:01, 28.30it/s][A
 97%|█████████▋| 1522/1562 [00:59<00:01, 28.29it/s][A
 98%|█████████▊| 1525/1562 [01:00<00:01, 28.29it/s][A
 98%|█████████▊| 1528/1562 [01:00<00:01, 28.29it/s][A
 98%|█████████▊| 1531/1562 [01:00<00:01, 28.29it/s][A
 98%|█████████▊| 1534/1562 [01:00<00:00, 28.30it/s][A
 98%|█████████▊| 1537/1562 [01:00<00:00, 28.29it/s][A
 99%|█████████▊| 1540/1562 [01:00<00:00, 28.28it/s][A
 99%|█████████▉| 1543/1562 [01:00<00:00, 28.12it/s][A
 99%|█████████▉| 1546/1562 [01:00<00:00, 28.17it/s][A
 99%|█████████▉| 1549/1562 [01:00<00:00, 28.20it/s][A
 99%|█████████▉| 1552/1562 [01:01<00:00, 28.22it/s][A
100%|█████████▉| 1555/1562 [01:01<00:00, 28.24it/s][A
100%|█████████▉| 1558/1562 [01:01<00:00, 28.27it/s][A
100%|█████████▉| 1561/1562 [01:01<00:00, 28.28it/s][A100%|██████████| 1562/1562 [01:02<00:00, 25.06it/s]

  0%|          | 0/1562 [00:00<?, ?it/s][A/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

  0%|          | 1/1562 [00:04<2:04:52,  4.80s/it][A
  0%|          | 4/1562 [00:04<24:32,  1.06it/s]  [A
  0%|          | 7/1562 [00:05<11:50,  2.19it/s][A
  1%|          | 10/1562 [00:05<07:05,  3.64it/s][A
  1%|          | 13/1562 [00:05<04:44,  5.45it/s][A
  1%|          | 16/1562 [00:05<03:22,  7.63it/s][A
  1%|          | 19/1562 [00:05<02:32, 10.09it/s][A
  1%|▏         | 22/1562 [00:05<02:01, 12.70it/s][A
  2%|▏         | 25/1562 [00:05<01:40, 15.25it/s][A
  2%|▏         | 28/1562 [00:05<01:26, 17.67it/s][A
  2%|▏         | 31/1562 [00:05<01:16, 19.94it/s][A
  2%|▏         | 34/1562 [00:06<01:10, 21.73it/s][A
  2%|▏         | 37/1562 [00:06<01:05, 23.13it/s][A
  3%|▎         | 40/1562 [00:06<01:02, 24.37it/s][A
  3%|▎         | 43/1562 [00:06<01:00, 25.23it/s][A
  3%|▎         | 46/1562 [00:06<00:58, 25.76it/s][A
  3%|▎         | 49/1562 [00:06<00:57, 26.35it/s][A
  3%|▎         | 52/1562 [00:06<00:56, 26.61it/s][A
  4%|▎         | 55/1562 [00:06<00:55, 26.93it/s][A
  4%|▎         | 58/1562 [00:06<00:55, 26.87it/s][A
  4%|▍         | 61/1562 [00:07<00:56, 26.77it/s][A
  4%|▍         | 64/1562 [00:07<00:58, 25.63it/s][A
  4%|▍         | 67/1562 [00:07<00:56, 26.38it/s][A
  4%|▍         | 70/1562 [00:07<00:55, 26.87it/s][A
  5%|▍         | 73/1562 [00:07<00:54, 27.24it/s][A
  5%|▍         | 76/1562 [00:07<00:54, 27.44it/s][A
  5%|▌         | 79/1562 [00:07<00:53, 27.66it/s][A
  5%|▌         | 82/1562 [00:07<00:53, 27.76it/s][A
  5%|▌         | 85/1562 [00:07<00:52, 27.90it/s][A
  6%|▌         | 88/1562 [00:07<00:52, 27.98it/s][A
  6%|▌         | 91/1562 [00:08<00:52, 27.98it/s][A
  6%|▌         | 94/1562 [00:08<00:52, 28.07it/s][A
  6%|▌         | 97/1562 [00:08<00:52, 28.13it/s][A
  6%|▋         | 100/1562 [00:08<00:52, 28.04it/s][A
  7%|▋         | 103/1562 [00:08<00:52, 27.98it/s][A
  7%|▋         | 106/1562 [00:08<00:52, 27.91it/s][A
  7%|▋         | 109/1562 [00:08<00:51, 27.96it/s][A
  7%|▋         | 112/1562 [00:08<00:51, 28.00it/s][A
  7%|▋         | 115/1562 [00:08<00:51, 27.91it/s][A
  8%|▊         | 118/1562 [00:09<00:51, 27.99it/s][A
  8%|▊         | 121/1562 [00:09<00:51, 28.07it/s][A
  8%|▊         | 124/1562 [00:09<00:51, 27.84it/s][A
  8%|▊         | 127/1562 [00:09<00:51, 27.99it/s][A
  8%|▊         | 130/1562 [00:09<00:51, 28.01it/s][A
  9%|▊         | 133/1562 [00:09<00:51, 27.91it/s][A
  9%|▊         | 136/1562 [00:09<00:51, 27.76it/s][A
  9%|▉         | 139/1562 [00:09<00:51, 27.73it/s][A
  9%|▉         | 142/1562 [00:09<00:51, 27.67it/s][A
  9%|▉         | 145/1562 [00:10<00:51, 27.46it/s][A
  9%|▉         | 148/1562 [00:10<00:51, 27.48it/s][A
 10%|▉         | 151/1562 [00:10<00:51, 27.39it/s][A
 10%|▉         | 154/1562 [00:10<00:51, 27.41it/s][A
 10%|█         | 157/1562 [00:10<00:51, 27.32it/s][A
 10%|█         | 160/1562 [00:10<00:51, 27.38it/s][A
 10%|█         | 163/1562 [00:10<00:50, 27.44it/s][A
 11%|█         | 166/1562 [00:10<00:50, 27.47it/s][A
 11%|█         | 169/1562 [00:10<00:50, 27.40it/s][A
 11%|█         | 172/1562 [00:11<00:50, 27.46it/s][A
 11%|█         | 175/1562 [00:11<00:50, 27.47it/s][A
 11%|█▏        | 178/1562 [00:11<00:50, 27.53it/s][A
 12%|█▏        | 181/1562 [00:11<00:50, 27.51it/s][A
 12%|█▏        | 184/1562 [00:11<00:50, 27.47it/s][A
 12%|█▏        | 187/1562 [00:11<00:50, 27.42it/s][A
 12%|█▏        | 190/1562 [00:11<00:50, 27.38it/s][A
 12%|█▏        | 193/1562 [00:11<00:50, 27.35it/s][A
 13%|█▎        | 196/1562 [00:11<00:49, 27.39it/s][A
 13%|█▎        | 199/1562 [00:12<00:49, 27.46it/s][A
 13%|█▎        | 202/1562 [00:12<00:49, 27.39it/s][A
 13%|█▎        | 205/1562 [00:12<00:49, 27.38it/s][A
 13%|█▎        | 208/1562 [00:12<00:49, 27.40it/s][A
 14%|█▎        | 211/1562 [00:12<00:49, 27.46it/s][A
 14%|█▎        | 214/1562 [00:12<00:48, 27.53it/s][A
 14%|█▍        | 217/1562 [00:12<00:48, 27.50it/s][A
 14%|█▍        | 220/1562 [00:12<00:48, 27.49it/s][A
 14%|█▍        | 223/1562 [00:12<00:48, 27.57it/s][A
 14%|█▍        | 226/1562 [00:12<00:48, 27.58it/s][A
 15%|█▍        | 229/1562 [00:13<00:48, 27.56it/s][A
 15%|█▍        | 232/1562 [00:13<00:48, 27.52it/s][A
 15%|█▌        | 235/1562 [00:13<00:48, 27.47it/s][A
 15%|█▌        | 238/1562 [00:13<00:48, 27.45it/s][A
 15%|█▌        | 241/1562 [00:13<00:48, 27.40it/s][A
 16%|█▌        | 244/1562 [00:13<00:48, 27.35it/s][A
 16%|█▌        | 247/1562 [00:13<00:48, 27.24it/s][A
 16%|█▌        | 250/1562 [00:13<00:48, 27.28it/s][A
 16%|█▌        | 253/1562 [00:13<00:47, 27.34it/s][A
 16%|█▋        | 256/1562 [00:14<00:47, 27.45it/s][A
 17%|█▋        | 259/1562 [00:14<00:47, 27.47it/s][A
 17%|█▋        | 262/1562 [00:14<00:47, 27.54it/s][A
 17%|█▋        | 265/1562 [00:14<00:47, 27.49it/s][A
 17%|█▋        | 268/1562 [00:14<00:47, 27.53it/s][A
 17%|█▋        | 271/1562 [00:14<00:46, 27.64it/s][A
 18%|█▊        | 274/1562 [00:14<00:46, 27.65it/s][A
 18%|█▊        | 277/1562 [00:14<00:46, 27.67it/s][A
 18%|█▊        | 280/1562 [00:14<00:46, 27.71it/s][A
 18%|█▊        | 283/1562 [00:15<00:46, 27.70it/s][A
 18%|█▊        | 286/1562 [00:15<00:46, 27.71it/s][A
 19%|█▊        | 289/1562 [00:15<00:45, 27.68it/s][A
 19%|█▊        | 292/1562 [00:15<00:45, 27.69it/s][A
 19%|█▉        | 295/1562 [00:15<00:45, 27.70it/s][A
 19%|█▉        | 298/1562 [00:15<00:45, 27.72it/s][A
 19%|█▉        | 301/1562 [00:15<00:45, 27.69it/s][A
 19%|█▉        | 304/1562 [00:15<00:45, 27.71it/s][A
 20%|█▉        | 307/1562 [00:15<00:45, 27.73it/s][A
 20%|█▉        | 310/1562 [00:16<00:45, 27.54it/s][A
 20%|██        | 313/1562 [00:16<00:45, 27.40it/s][A
 20%|██        | 316/1562 [00:16<00:45, 27.28it/s][A
 20%|██        | 319/1562 [00:16<00:45, 27.17it/s][A
 21%|██        | 322/1562 [00:16<00:45, 27.30it/s][A
 21%|██        | 325/1562 [00:16<00:45, 27.32it/s][A
 21%|██        | 328/1562 [00:16<00:45, 27.40it/s][A
 21%|██        | 331/1562 [00:16<00:44, 27.44it/s][A
 21%|██▏       | 334/1562 [00:16<00:44, 27.61it/s][A
 22%|██▏       | 337/1562 [00:17<00:44, 27.33it/s][A
 22%|██▏       | 340/1562 [00:17<00:44, 27.51it/s][A
 22%|██▏       | 343/1562 [00:17<00:44, 27.60it/s][A
 22%|██▏       | 346/1562 [00:17<00:44, 27.61it/s][A
 22%|██▏       | 349/1562 [00:17<00:43, 27.66it/s][A
 23%|██▎       | 352/1562 [00:17<00:43, 27.75it/s][A
 23%|██▎       | 355/1562 [00:17<00:43, 27.73it/s][A
 23%|██▎       | 358/1562 [00:17<00:43, 27.72it/s][A
 23%|██▎       | 361/1562 [00:17<00:43, 27.68it/s][A
 23%|██▎       | 364/1562 [00:18<00:43, 27.75it/s][A
 23%|██▎       | 367/1562 [00:18<00:43, 27.67it/s][A
 24%|██▎       | 370/1562 [00:18<00:42, 27.73it/s][A
 24%|██▍       | 373/1562 [00:18<00:42, 27.78it/s][A
 24%|██▍       | 376/1562 [00:18<00:42, 27.79it/s][A
 24%|██▍       | 379/1562 [00:18<00:42, 27.74it/s][A
 24%|██▍       | 382/1562 [00:18<00:42, 27.70it/s][A
 25%|██▍       | 385/1562 [00:18<00:42, 27.77it/s][A
 25%|██▍       | 388/1562 [00:18<00:42, 27.73it/s][A
 25%|██▌       | 391/1562 [00:18<00:42, 27.72it/s][A
 25%|██▌       | 394/1562 [00:19<00:42, 27.71it/s][A
 25%|██▌       | 397/1562 [00:19<00:41, 27.74it/s][A
 26%|██▌       | 400/1562 [00:19<00:41, 27.70it/s][A
 26%|██▌       | 403/1562 [00:19<00:41, 27.74it/s][A
 26%|██▌       | 406/1562 [00:19<00:41, 27.76it/s][A
 26%|██▌       | 409/1562 [00:19<00:41, 27.71it/s][A
 26%|██▋       | 412/1562 [00:19<00:41, 27.70it/s][A
 27%|██▋       | 415/1562 [00:19<00:41, 27.75it/s][A
 27%|██▋       | 418/1562 [00:19<00:41, 27.76it/s][A
 27%|██▋       | 421/1562 [00:20<00:41, 27.78it/s][A
 27%|██▋       | 424/1562 [00:20<00:40, 27.76it/s][A
 27%|██▋       | 427/1562 [00:20<00:40, 27.74it/s][A
 28%|██▊       | 430/1562 [00:20<00:40, 27.75it/s][A
 28%|██▊       | 433/1562 [00:20<00:40, 27.72it/s][A
 28%|██▊       | 436/1562 [00:20<00:40, 27.79it/s][A
 28%|██▊       | 439/1562 [00:20<00:40, 27.79it/s][A
 28%|██▊       | 442/1562 [00:20<00:40, 27.78it/s][A
 28%|██▊       | 445/1562 [00:20<00:40, 27.74it/s][A
 29%|██▊       | 448/1562 [00:21<00:40, 27.80it/s][A
 29%|██▉       | 451/1562 [00:21<00:40, 27.75it/s][A
 29%|██▉       | 454/1562 [00:21<00:40, 27.70it/s][A
 29%|██▉       | 457/1562 [00:21<00:39, 27.66it/s][A
 29%|██▉       | 460/1562 [00:21<00:39, 27.65it/s][A
 30%|██▉       | 463/1562 [00:21<00:39, 27.74it/s][A
 30%|██▉       | 466/1562 [00:21<00:39, 27.81it/s][A
 30%|███       | 469/1562 [00:21<00:39, 27.84it/s][A
 30%|███       | 472/1562 [00:21<00:39, 27.84it/s][A
 30%|███       | 475/1562 [00:22<00:39, 27.83it/s][A
 31%|███       | 478/1562 [00:22<00:38, 27.88it/s][A
 31%|███       | 481/1562 [00:22<00:38, 27.88it/s][A
 31%|███       | 484/1562 [00:22<00:38, 27.82it/s][A
 31%|███       | 487/1562 [00:22<00:38, 27.79it/s][A
 31%|███▏      | 490/1562 [00:22<00:38, 27.73it/s][A
 32%|███▏      | 493/1562 [00:22<00:38, 27.81it/s][A
 32%|███▏      | 496/1562 [00:22<00:38, 27.78it/s][A
 32%|███▏      | 499/1562 [00:22<00:38, 27.83it/s][A
 32%|███▏      | 502/1562 [00:22<00:38, 27.86it/s][A
 32%|███▏      | 505/1562 [00:23<00:38, 27.79it/s][A
 33%|███▎      | 508/1562 [00:23<00:37, 27.83it/s][A
 33%|███▎      | 511/1562 [00:23<00:37, 27.76it/s][A
 33%|███▎      | 514/1562 [00:23<00:37, 27.74it/s][A
 33%|███▎      | 517/1562 [00:23<00:37, 27.74it/s][A
 33%|███▎      | 520/1562 [00:23<00:37, 27.68it/s][A
 33%|███▎      | 523/1562 [00:23<00:37, 27.70it/s][A
 34%|███▎      | 526/1562 [00:23<00:37, 27.74it/s][A
 34%|███▍      | 529/1562 [00:23<00:37, 27.74it/s][A
 34%|███▍      | 532/1562 [00:24<00:37, 27.70it/s][A
 34%|███▍      | 535/1562 [00:24<00:37, 27.54it/s][A
 34%|███▍      | 538/1562 [00:24<00:37, 27.61it/s][A
 35%|███▍      | 541/1562 [00:24<00:36, 27.63it/s][A
 35%|███▍      | 544/1562 [00:24<00:36, 27.61it/s][A
 35%|███▌      | 547/1562 [00:24<00:36, 27.69it/s][A
 35%|███▌      | 550/1562 [00:24<00:36, 27.69it/s][A
 35%|███▌      | 553/1562 [00:24<00:36, 27.75it/s][A
 36%|███▌      | 556/1562 [00:24<00:36, 27.71it/s][A
 36%|███▌      | 559/1562 [00:25<00:36, 27.70it/s][A
 36%|███▌      | 562/1562 [00:25<00:36, 27.76it/s][A
 36%|███▌      | 565/1562 [00:25<00:35, 27.77it/s][A
 36%|███▋      | 568/1562 [00:25<00:35, 27.66it/s][A
 37%|███▋      | 571/1562 [00:25<00:35, 27.67it/s][A
 37%|███▋      | 574/1562 [00:25<00:35, 27.63it/s][A
 37%|███▋      | 577/1562 [00:25<00:35, 27.61it/s][A
 37%|███▋      | 580/1562 [00:25<00:35, 27.64it/s][A
 37%|███▋      | 583/1562 [00:25<00:35, 27.64it/s][A
 38%|███▊      | 586/1562 [00:26<00:35, 27.61it/s][A
 38%|███▊      | 589/1562 [00:26<00:35, 27.64it/s][A
 38%|███▊      | 592/1562 [00:26<00:35, 27.54it/s][A
 38%|███▊      | 595/1562 [00:26<00:35, 27.51it/s][A
 38%|███▊      | 598/1562 [00:26<00:34, 27.60it/s][A
 38%|███▊      | 601/1562 [00:26<00:34, 27.63it/s][A
 39%|███▊      | 604/1562 [00:26<00:34, 27.66it/s][A
 39%|███▉      | 607/1562 [00:26<00:34, 27.67it/s][A
 39%|███▉      | 610/1562 [00:26<00:34, 27.70it/s][A
 39%|███▉      | 613/1562 [00:26<00:34, 27.71it/s][A
 39%|███▉      | 616/1562 [00:27<00:34, 27.67it/s][A
 40%|███▉      | 619/1562 [00:27<00:34, 27.65it/s][A
 40%|███▉      | 622/1562 [00:27<00:34, 27.63it/s][A
 40%|████      | 625/1562 [00:27<00:33, 27.63it/s][A
 40%|████      | 628/1562 [00:27<00:33, 27.62it/s][A
 40%|████      | 631/1562 [00:27<00:33, 27.65it/s][A
 41%|████      | 634/1562 [00:27<00:33, 27.65it/s][A
 41%|████      | 637/1562 [00:27<00:33, 27.71it/s][A
 41%|████      | 640/1562 [00:27<00:33, 27.58it/s][A
 41%|████      | 643/1562 [00:28<00:33, 27.62it/s][A
 41%|████▏     | 646/1562 [00:28<00:33, 27.60it/s][A
 42%|████▏     | 649/1562 [00:28<00:33, 27.61it/s][A
 42%|████▏     | 652/1562 [00:28<00:32, 27.61it/s][A
 42%|████▏     | 655/1562 [00:28<00:32, 27.58it/s][A
 42%|████▏     | 658/1562 [00:28<00:32, 27.57it/s][A
 42%|████▏     | 661/1562 [00:28<00:32, 27.60it/s][A
 43%|████▎     | 664/1562 [00:28<00:32, 27.68it/s][A
 43%|████▎     | 667/1562 [00:28<00:32, 27.68it/s][A
 43%|████▎     | 670/1562 [00:29<00:32, 27.71it/s][A
 43%|████▎     | 673/1562 [00:29<00:32, 27.72it/s][A
 43%|████▎     | 676/1562 [00:29<00:32, 27.68it/s][A
 43%|████▎     | 679/1562 [00:29<00:31, 27.72it/s][A
 44%|████▎     | 682/1562 [00:29<00:31, 27.75it/s][A
 44%|████▍     | 685/1562 [00:29<00:31, 27.73it/s][A
 44%|████▍     | 688/1562 [00:29<00:31, 27.64it/s][A
 44%|████▍     | 691/1562 [00:29<00:31, 27.66it/s][A
 44%|████▍     | 694/1562 [00:29<00:31, 27.71it/s][A
 45%|████▍     | 697/1562 [00:30<00:31, 27.68it/s][A
 45%|████▍     | 700/1562 [00:30<00:31, 27.66it/s][A
 45%|████▌     | 703/1562 [00:30<00:31, 27.71it/s][A
 45%|████▌     | 706/1562 [00:30<00:30, 27.78it/s][A
 45%|████▌     | 709/1562 [00:30<00:30, 27.72it/s][A
 46%|████▌     | 712/1562 [00:30<00:30, 27.60it/s][A
 46%|████▌     | 715/1562 [00:30<00:30, 27.59it/s][A
 46%|████▌     | 718/1562 [00:30<00:30, 27.63it/s][A
 46%|████▌     | 721/1562 [00:30<00:30, 27.59it/s][A
 46%|████▋     | 724/1562 [00:30<00:30, 27.61it/s][A
 47%|████▋     | 727/1562 [00:31<00:30, 27.64it/s][A
 47%|████▋     | 730/1562 [00:31<00:30, 27.67it/s][A
 47%|████▋     | 733/1562 [00:31<00:29, 27.71it/s][A
 47%|████▋     | 736/1562 [00:31<00:29, 27.69it/s][A
 47%|████▋     | 739/1562 [00:31<00:29, 27.69it/s][A
 48%|████▊     | 742/1562 [00:31<00:29, 27.76it/s][A
 48%|████▊     | 745/1562 [00:31<00:29, 27.79it/s][A
 48%|████▊     | 748/1562 [00:31<00:29, 27.74it/s][A
 48%|████▊     | 751/1562 [00:31<00:29, 27.69it/s][A
 48%|████▊     | 754/1562 [00:32<00:29, 27.68it/s][A
 48%|████▊     | 757/1562 [00:32<00:29, 27.75it/s][A
 49%|████▊     | 760/1562 [00:32<00:29, 27.56it/s][A
 49%|████▉     | 763/1562 [00:32<00:28, 27.58it/s][A
 49%|████▉     | 766/1562 [00:32<00:28, 27.55it/s][A
 49%|████▉     | 769/1562 [00:32<00:28, 27.62it/s][A
 49%|████▉     | 772/1562 [00:32<00:28, 27.64it/s][A
 50%|████▉     | 775/1562 [00:32<00:28, 27.57it/s][A
 50%|████▉     | 778/1562 [00:32<00:28, 27.48it/s][A
 50%|█████     | 781/1562 [00:33<00:28, 27.55it/s][A
 50%|█████     | 784/1562 [00:33<00:28, 27.57it/s][A
 50%|█████     | 787/1562 [00:33<00:28, 27.58it/s][A
 51%|█████     | 790/1562 [00:33<00:27, 27.65it/s][A
 51%|█████     | 793/1562 [00:33<00:27, 27.71it/s][A
 51%|█████     | 796/1562 [00:33<00:27, 27.71it/s][A
 51%|█████     | 799/1562 [00:33<00:27, 27.68it/s][A
 51%|█████▏    | 802/1562 [00:33<00:27, 27.71it/s][A
 52%|█████▏    | 805/1562 [00:33<00:27, 27.72it/s][A
 52%|█████▏    | 808/1562 [00:34<00:27, 27.76it/s][A
 52%|█████▏    | 811/1562 [00:34<00:27, 27.75it/s][A
 52%|█████▏    | 814/1562 [00:34<00:26, 27.74it/s][A
 52%|█████▏    | 817/1562 [00:34<00:26, 27.69it/s][A
 52%|█████▏    | 820/1562 [00:34<00:26, 27.72it/s][A
 53%|█████▎    | 823/1562 [00:34<00:26, 27.74it/s][A
 53%|█████▎    | 826/1562 [00:34<00:26, 27.69it/s][A
 53%|█████▎    | 829/1562 [00:34<00:26, 27.67it/s][A
 53%|█████▎    | 832/1562 [00:34<00:26, 27.73it/s][A
 53%|█████▎    | 835/1562 [00:35<00:26, 27.76it/s][A
 54%|█████▎    | 838/1562 [00:35<00:26, 27.67it/s][A
 54%|█████▍    | 841/1562 [00:35<00:26, 27.65it/s][A
 54%|█████▍    | 844/1562 [00:35<00:25, 27.67it/s][A
 54%|█████▍    | 847/1562 [00:35<00:25, 27.57it/s][A
 54%|█████▍    | 850/1562 [00:35<00:25, 27.59it/s][A
 55%|█████▍    | 853/1562 [00:35<00:25, 27.65it/s][A
 55%|█████▍    | 856/1562 [00:35<00:25, 27.69it/s][A
 55%|█████▍    | 859/1562 [00:35<00:25, 27.67it/s][A
 55%|█████▌    | 862/1562 [00:35<00:25, 27.70it/s][A
 55%|█████▌    | 865/1562 [00:36<00:25, 27.68it/s][A
 56%|█████▌    | 868/1562 [00:36<00:25, 27.68it/s][A
 56%|█████▌    | 871/1562 [00:36<00:24, 27.73it/s][A
 56%|█████▌    | 874/1562 [00:36<00:24, 27.71it/s][A
 56%|█████▌    | 877/1562 [00:36<00:24, 27.74it/s][A
 56%|█████▋    | 880/1562 [00:36<00:24, 27.66it/s][A
 57%|█████▋    | 883/1562 [00:36<00:24, 27.66it/s][A
 57%|█████▋    | 886/1562 [00:36<00:24, 27.69it/s][A
 57%|█████▋    | 889/1562 [00:36<00:24, 27.49it/s][A
 57%|█████▋    | 892/1562 [00:37<00:24, 27.46it/s][A
 57%|█████▋    | 895/1562 [00:37<00:24, 27.53it/s][A
 57%|█████▋    | 898/1562 [00:37<00:24, 27.60it/s][A
 58%|█████▊    | 901/1562 [00:37<00:23, 27.60it/s][A
 58%|█████▊    | 904/1562 [00:37<00:23, 27.51it/s][A
 58%|█████▊    | 907/1562 [00:37<00:23, 27.41it/s][A
 58%|█████▊    | 910/1562 [00:37<00:23, 27.43it/s][A
 58%|█████▊    | 913/1562 [00:37<00:23, 27.36it/s][A
 59%|█████▊    | 916/1562 [00:37<00:23, 27.49it/s][A
 59%|█████▉    | 919/1562 [00:38<00:23, 27.52it/s][A
 59%|█████▉    | 922/1562 [00:38<00:23, 27.55it/s][A
 59%|█████▉    | 925/1562 [00:38<00:23, 27.57it/s][A
 59%|█████▉    | 928/1562 [00:38<00:23, 27.55it/s][A
 60%|█████▉    | 931/1562 [00:38<00:22, 27.51it/s][A
 60%|█████▉    | 934/1562 [00:38<00:22, 27.55it/s][A
 60%|█████▉    | 937/1562 [00:38<00:22, 27.60it/s][A
 60%|██████    | 940/1562 [00:38<00:22, 27.55it/s][A
 60%|██████    | 943/1562 [00:38<00:22, 27.55it/s][A
 61%|██████    | 946/1562 [00:39<00:22, 27.58it/s][A
 61%|██████    | 949/1562 [00:39<00:22, 27.64it/s][A
 61%|██████    | 952/1562 [00:39<00:22, 27.60it/s][A
 61%|██████    | 955/1562 [00:39<00:22, 27.55it/s][A
 61%|██████▏   | 958/1562 [00:39<00:21, 27.50it/s][A
 62%|██████▏   | 961/1562 [00:39<00:21, 27.56it/s][A
 62%|██████▏   | 964/1562 [00:39<00:21, 27.57it/s][A
 62%|██████▏   | 967/1562 [00:39<00:21, 27.59it/s][A
 62%|██████▏   | 970/1562 [00:39<00:21, 27.52it/s][A
 62%|██████▏   | 973/1562 [00:40<00:21, 27.57it/s][A
 62%|██████▏   | 976/1562 [00:40<00:21, 27.58it/s][A
 63%|██████▎   | 979/1562 [00:40<00:21, 27.59it/s][A
 63%|██████▎   | 982/1562 [00:40<00:20, 27.67it/s][A
 63%|██████▎   | 985/1562 [00:40<00:20, 27.65it/s][A
 63%|██████▎   | 988/1562 [00:40<00:20, 27.64it/s][A
 63%|██████▎   | 991/1562 [00:40<00:20, 27.65it/s][A
 64%|██████▎   | 994/1562 [00:40<00:20, 27.70it/s][A
 64%|██████▍   | 997/1562 [00:40<00:20, 27.70it/s][A
 64%|██████▍   | 1000/1562 [00:40<00:20, 27.74it/s][A
 64%|██████▍   | 1003/1562 [00:41<00:20, 27.75it/s][A
 64%|██████▍   | 1006/1562 [00:41<00:20, 27.72it/s][A
 65%|██████▍   | 1009/1562 [00:41<00:19, 27.72it/s][A
 65%|██████▍   | 1012/1562 [00:41<00:19, 27.76it/s][A
 65%|██████▍   | 1015/1562 [00:41<00:19, 27.68it/s][A
 65%|██████▌   | 1018/1562 [00:41<00:19, 27.70it/s][A
 65%|██████▌   | 1021/1562 [00:41<00:19, 27.71it/s][A
 66%|██████▌   | 1024/1562 [00:41<00:19, 27.65it/s][A
 66%|██████▌   | 1027/1562 [00:41<00:19, 27.61it/s][A
 66%|██████▌   | 1030/1562 [00:42<00:19, 27.64it/s][A
 66%|██████▌   | 1033/1562 [00:42<00:19, 27.59it/s][A
 66%|██████▋   | 1036/1562 [00:42<00:19, 27.65it/s][A
 67%|██████▋   | 1039/1562 [00:42<00:18, 27.58it/s][A
 67%|██████▋   | 1042/1562 [00:42<00:18, 27.60it/s][A
 67%|██████▋   | 1045/1562 [00:42<00:18, 27.67it/s][A
 67%|██████▋   | 1048/1562 [00:42<00:18, 27.66it/s][A
 67%|██████▋   | 1051/1562 [00:42<00:18, 27.65it/s][A
 67%|██████▋   | 1054/1562 [00:42<00:18, 27.66it/s][A
 68%|██████▊   | 1057/1562 [00:43<00:18, 27.68it/s][A
 68%|██████▊   | 1060/1562 [00:43<00:18, 27.64it/s][A
 68%|██████▊   | 1063/1562 [00:43<00:18, 27.71it/s][A
 68%|██████▊   | 1066/1562 [00:43<00:17, 27.66it/s][A
 68%|██████▊   | 1069/1562 [00:43<00:17, 27.66it/s][A
 69%|██████▊   | 1072/1562 [00:43<00:17, 27.61it/s][A
 69%|██████▉   | 1075/1562 [00:43<00:17, 27.68it/s][A
 69%|██████▉   | 1078/1562 [00:43<00:17, 27.70it/s][A
 69%|██████▉   | 1081/1562 [00:43<00:17, 27.62it/s][A
 69%|██████▉   | 1084/1562 [00:44<00:17, 27.62it/s][A
 70%|██████▉   | 1087/1562 [00:44<00:17, 27.57it/s][A
 70%|██████▉   | 1090/1562 [00:44<00:17, 27.53it/s][A
 70%|██████▉   | 1093/1562 [00:44<00:16, 27.59it/s][A
 70%|███████   | 1096/1562 [00:44<00:16, 27.55it/s][A
 70%|███████   | 1099/1562 [00:44<00:16, 27.56it/s][A
 71%|███████   | 1102/1562 [00:44<00:16, 27.60it/s][A
 71%|███████   | 1105/1562 [00:44<00:16, 27.54it/s][A
 71%|███████   | 1108/1562 [00:44<00:16, 27.63it/s][A
 71%|███████   | 1111/1562 [00:45<00:16, 27.67it/s][A
 71%|███████▏  | 1114/1562 [00:45<00:16, 27.63it/s][A
 72%|███████▏  | 1117/1562 [00:45<00:16, 27.42it/s][A
 72%|███████▏  | 1120/1562 [00:45<00:16, 27.46it/s][A
 72%|███████▏  | 1123/1562 [00:45<00:15, 27.52it/s][A
 72%|███████▏  | 1126/1562 [00:45<00:15, 27.57it/s][A
 72%|███████▏  | 1129/1562 [00:45<00:15, 27.64it/s][A
 72%|███████▏  | 1132/1562 [00:45<00:15, 27.64it/s][A
 73%|███████▎  | 1135/1562 [00:45<00:15, 27.61it/s][A
 73%|███████▎  | 1138/1562 [00:45<00:15, 27.67it/s][A
 73%|███████▎  | 1141/1562 [00:46<00:15, 27.64it/s][A
 73%|███████▎  | 1144/1562 [00:46<00:15, 27.67it/s][A
 73%|███████▎  | 1147/1562 [00:46<00:15, 27.64it/s][A
 74%|███████▎  | 1150/1562 [00:46<00:14, 27.62it/s][A
 74%|███████▍  | 1153/1562 [00:46<00:14, 27.59it/s][A
 74%|███████▍  | 1156/1562 [00:46<00:14, 27.62it/s][A
 74%|███████▍  | 1159/1562 [00:46<00:14, 27.61it/s][A
 74%|███████▍  | 1162/1562 [00:46<00:14, 27.59it/s][A
 75%|███████▍  | 1165/1562 [00:46<00:14, 27.68it/s][A
 75%|███████▍  | 1168/1562 [00:47<00:14, 27.66it/s][A
 75%|███████▍  | 1171/1562 [00:47<00:14, 27.67it/s][A
 75%|███████▌  | 1174/1562 [00:47<00:14, 27.61it/s][A
 75%|███████▌  | 1177/1562 [00:47<00:13, 27.64it/s][A
 76%|███████▌  | 1180/1562 [00:47<00:13, 27.67it/s][A
 76%|███████▌  | 1183/1562 [00:47<00:13, 27.67it/s][A
 76%|███████▌  | 1186/1562 [00:47<00:13, 27.69it/s][A
 76%|███████▌  | 1189/1562 [00:47<00:13, 27.70it/s][A
 76%|███████▋  | 1192/1562 [00:47<00:13, 27.70it/s][A
 77%|███████▋  | 1195/1562 [00:48<00:13, 27.65it/s][A
 77%|███████▋  | 1198/1562 [00:48<00:13, 27.71it/s][A
 77%|███████▋  | 1201/1562 [00:48<00:13, 27.70it/s][A
 77%|███████▋  | 1204/1562 [00:48<00:12, 27.68it/s][A
 77%|███████▋  | 1207/1562 [00:48<00:12, 27.69it/s][A
 77%|███████▋  | 1210/1562 [00:48<00:12, 27.67it/s][A
 78%|███████▊  | 1213/1562 [00:48<00:12, 27.69it/s][A
 78%|███████▊  | 1216/1562 [00:48<00:12, 27.66it/s][A
 78%|███████▊  | 1219/1562 [00:48<00:12, 27.65it/s][A
 78%|███████▊  | 1222/1562 [00:49<00:12, 27.64it/s][A
 78%|███████▊  | 1225/1562 [00:49<00:12, 27.59it/s][A
 79%|███████▊  | 1228/1562 [00:49<00:12, 27.67it/s][A
 79%|███████▉  | 1231/1562 [00:49<00:11, 27.67it/s][A
 79%|███████▉  | 1234/1562 [00:49<00:11, 27.74it/s][A
 79%|███████▉  | 1237/1562 [00:49<00:11, 27.73it/s][A
 79%|███████▉  | 1240/1562 [00:49<00:11, 27.76it/s][A
 80%|███████▉  | 1243/1562 [00:49<00:11, 27.72it/s][A
 80%|███████▉  | 1246/1562 [00:49<00:11, 27.73it/s][A
 80%|███████▉  | 1249/1562 [00:49<00:11, 27.69it/s][A
 80%|████████  | 1252/1562 [00:50<00:11, 27.70it/s][A
 80%|████████  | 1255/1562 [00:50<00:11, 27.67it/s][A
 81%|████████  | 1258/1562 [00:50<00:10, 27.66it/s][A
 81%|████████  | 1261/1562 [00:50<00:10, 27.72it/s][A
 81%|████████  | 1264/1562 [00:50<00:10, 27.66it/s][A
 81%|████████  | 1267/1562 [00:50<00:10, 27.70it/s][A
 81%|████████▏ | 1270/1562 [00:50<00:10, 27.75it/s][A
 81%|████████▏ | 1273/1562 [00:50<00:10, 27.71it/s][A
 82%|████████▏ | 1276/1562 [00:50<00:10, 27.71it/s][A
 82%|████████▏ | 1279/1562 [00:51<00:10, 27.70it/s][A
 82%|████████▏ | 1282/1562 [00:51<00:10, 27.69it/s][A
 82%|████████▏ | 1285/1562 [00:51<00:09, 27.73it/s][A
 82%|████████▏ | 1288/1562 [00:51<00:09, 27.74it/s][A
 83%|████████▎ | 1291/1562 [00:51<00:09, 27.78it/s][A
 83%|████████▎ | 1294/1562 [00:51<00:09, 27.74it/s][A
 83%|████████▎ | 1297/1562 [00:51<00:09, 27.77it/s][A
 83%|████████▎ | 1300/1562 [00:51<00:09, 27.78it/s][A
 83%|████████▎ | 1303/1562 [00:51<00:09, 27.78it/s][A
 84%|████████▎ | 1306/1562 [00:52<00:09, 27.76it/s][A
 84%|████████▍ | 1309/1562 [00:52<00:09, 27.69it/s][A
 84%|████████▍ | 1312/1562 [00:52<00:09, 27.69it/s][A
 84%|████████▍ | 1315/1562 [00:52<00:08, 27.73it/s][A
 84%|████████▍ | 1318/1562 [00:52<00:08, 27.72it/s][A
 85%|████████▍ | 1321/1562 [00:52<00:08, 27.67it/s][A
 85%|████████▍ | 1324/1562 [00:52<00:08, 27.72it/s][A
 85%|████████▍ | 1327/1562 [00:52<00:08, 27.76it/s][A
 85%|████████▌ | 1330/1562 [00:52<00:08, 27.77it/s][A
 85%|████████▌ | 1333/1562 [00:53<00:08, 27.77it/s][A
 86%|████████▌ | 1336/1562 [00:53<00:08, 27.77it/s][A
 86%|████████▌ | 1339/1562 [00:53<00:08, 27.73it/s][A
 86%|████████▌ | 1342/1562 [00:53<00:07, 27.75it/s][A
 86%|████████▌ | 1345/1562 [00:53<00:07, 27.74it/s][A
 86%|████████▋ | 1348/1562 [00:53<00:07, 27.72it/s][A
 86%|████████▋ | 1351/1562 [00:53<00:07, 27.76it/s][A
 87%|████████▋ | 1354/1562 [00:53<00:07, 27.83it/s][A
 87%|████████▋ | 1357/1562 [00:53<00:07, 27.82it/s][A
 87%|████████▋ | 1360/1562 [00:53<00:07, 27.85it/s][A
 87%|████████▋ | 1363/1562 [00:54<00:07, 27.85it/s][A
 87%|████████▋ | 1366/1562 [00:54<00:07, 27.87it/s][A
 88%|████████▊ | 1369/1562 [00:54<00:06, 27.83it/s][A
 88%|████████▊ | 1372/1562 [00:54<00:06, 27.76it/s][A
 88%|████████▊ | 1375/1562 [00:54<00:06, 27.71it/s][A
 88%|████████▊ | 1378/1562 [00:54<00:06, 27.72it/s][A
 88%|████████▊ | 1381/1562 [00:54<00:06, 27.71it/s][A
 89%|████████▊ | 1384/1562 [00:54<00:06, 27.72it/s][A
 89%|████████▉ | 1387/1562 [00:54<00:06, 27.70it/s][A
 89%|████████▉ | 1390/1562 [00:55<00:06, 27.78it/s][A
 89%|████████▉ | 1393/1562 [00:55<00:06, 27.71it/s][A
 89%|████████▉ | 1396/1562 [00:55<00:05, 27.74it/s][A
 90%|████████▉ | 1399/1562 [00:55<00:05, 27.75it/s][A
 90%|████████▉ | 1402/1562 [00:55<00:05, 27.76it/s][A
 90%|████████▉ | 1405/1562 [00:55<00:05, 27.80it/s][A
 90%|█████████ | 1408/1562 [00:55<00:05, 27.74it/s][A
 90%|█████████ | 1411/1562 [00:55<00:05, 27.77it/s][A
 91%|█████████ | 1414/1562 [00:55<00:05, 27.84it/s][A
 91%|█████████ | 1417/1562 [00:56<00:05, 27.67it/s][A
 91%|█████████ | 1420/1562 [00:56<00:05, 27.72it/s][A
 91%|█████████ | 1423/1562 [00:56<00:05, 27.75it/s][A
 91%|█████████▏| 1426/1562 [00:56<00:04, 27.81it/s][A
 91%|█████████▏| 1429/1562 [00:56<00:04, 27.83it/s][A
 92%|█████████▏| 1432/1562 [00:56<00:04, 27.83it/s][A
 92%|█████████▏| 1435/1562 [00:56<00:04, 27.82it/s][A
 92%|█████████▏| 1438/1562 [00:56<00:04, 27.96it/s][A
 92%|█████████▏| 1441/1562 [00:56<00:04, 28.02it/s][A
 92%|█████████▏| 1444/1562 [00:57<00:04, 28.09it/s][A
 93%|█████████▎| 1447/1562 [00:57<00:04, 28.12it/s][A
 93%|█████████▎| 1450/1562 [00:57<00:03, 28.17it/s][A
 93%|█████████▎| 1453/1562 [00:57<00:03, 28.17it/s][A
 93%|█████████▎| 1456/1562 [00:57<00:03, 28.19it/s][A
 93%|█████████▎| 1459/1562 [00:57<00:03, 28.21it/s][A
 94%|█████████▎| 1462/1562 [00:57<00:03, 28.22it/s][A
 94%|█████████▍| 1465/1562 [00:57<00:03, 28.23it/s][A
 94%|█████████▍| 1468/1562 [00:57<00:03, 28.22it/s][A
 94%|█████████▍| 1471/1562 [00:57<00:03, 28.23it/s][A
 94%|█████████▍| 1474/1562 [00:58<00:03, 28.24it/s][A
 95%|█████████▍| 1477/1562 [00:58<00:03, 28.24it/s][A
 95%|█████████▍| 1480/1562 [00:58<00:02, 28.17it/s][A
 95%|█████████▍| 1483/1562 [00:58<00:02, 28.14it/s][A
 95%|█████████▌| 1486/1562 [00:58<00:02, 28.18it/s][A
 95%|█████████▌| 1489/1562 [00:58<00:02, 28.20it/s][A
 96%|█████████▌| 1492/1562 [00:58<00:02, 28.22it/s][A
 96%|█████████▌| 1495/1562 [00:58<00:02, 28.23it/s][A
 96%|█████████▌| 1498/1562 [00:58<00:02, 28.24it/s][A
 96%|█████████▌| 1501/1562 [00:59<00:02, 28.24it/s][A
 96%|█████████▋| 1504/1562 [00:59<00:02, 28.24it/s][A
 96%|█████████▋| 1507/1562 [00:59<00:01, 28.23it/s][A
 97%|█████████▋| 1510/1562 [00:59<00:01, 28.24it/s][A
 97%|█████████▋| 1513/1562 [00:59<00:01, 28.24it/s][A
 97%|█████████▋| 1516/1562 [00:59<00:01, 28.24it/s][A
 97%|█████████▋| 1519/1562 [00:59<00:01, 28.24it/s][A
 97%|█████████▋| 1522/1562 [00:59<00:01, 28.26it/s][A
 98%|█████████▊| 1525/1562 [00:59<00:01, 28.25it/s][A
 98%|█████████▊| 1528/1562 [00:59<00:01, 28.25it/s][A
 98%|█████████▊| 1531/1562 [01:00<00:01, 28.22it/s][A
 98%|█████████▊| 1534/1562 [01:00<00:00, 28.24it/s][A
 98%|█████████▊| 1537/1562 [01:00<00:00, 28.23it/s][A
 99%|█████████▊| 1540/1562 [01:00<00:00, 28.23it/s][A
 99%|█████████▉| 1543/1562 [01:00<00:00, 28.24it/s][A
 99%|█████████▉| 1546/1562 [01:00<00:00, 28.24it/s][A
 99%|█████████▉| 1549/1562 [01:00<00:00, 28.23it/s][A
 99%|█████████▉| 1552/1562 [01:00<00:00, 28.20it/s][A
100%|█████████▉| 1555/1562 [01:00<00:00, 28.23it/s][A
100%|█████████▉| 1558/1562 [01:01<00:00, 28.26it/s][A
100%|█████████▉| 1561/1562 [01:01<00:00, 28.26it/s][A100%|██████████| 1562/1562 [01:02<00:00, 25.13it/s]
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/logger_connector/result.py:431: It is recommended to use `self.log('fid_ema_T10_Tlatent10', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
