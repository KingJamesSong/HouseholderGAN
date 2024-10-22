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
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 1 processes
----------------------------------------------------------------------------------------------------

LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:424: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=3` in the `DataLoader` to improve performance.
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torch/utils/data/dataloader.py:617: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(

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

copy images:   0%|          | 1/781 [00:02<29:33,  2.27s/it][A
copy images:   0%|          | 2/781 [00:02<16:14,  1.25s/it][A
copy images:   0%|          | 3/781 [00:03<11:52,  1.09it/s][A
copy images:   1%|          | 4/781 [00:04<11:02,  1.17it/s][A
copy images:   1%|          | 5/781 [00:04<09:30,  1.36it/s][A
copy images:   1%|          | 6/781 [00:05<09:11,  1.40it/s][A
copy images:   1%|          | 7/781 [00:05<08:19,  1.55it/s][A
copy images:   1%|          | 8/781 [00:06<07:52,  1.64it/s][A
copy images:   1%|          | 9/781 [00:06<07:41,  1.67it/s][A
copy images:   1%|▏         | 10/781 [00:07<07:31,  1.71it/s][A
copy images:   1%|▏         | 11/781 [00:07<07:10,  1.79it/s][A
copy images:   2%|▏         | 12/781 [00:08<06:58,  1.84it/s][A
copy images:   2%|▏         | 13/781 [00:09<07:29,  1.71it/s][A
copy images:   2%|▏         | 14/781 [00:09<07:10,  1.78it/s][A
copy images:   2%|▏         | 15/781 [00:10<06:58,  1.83it/s][A
copy images:   2%|▏         | 16/781 [00:10<06:49,  1.87it/s][A
copy images:   2%|▏         | 17/781 [00:11<06:48,  1.87it/s][A
copy images:   2%|▏         | 18/781 [00:11<06:44,  1.89it/s][A
copy images:   2%|▏         | 19/781 [00:12<06:38,  1.91it/s][A
copy images:   3%|▎         | 20/781 [00:12<06:39,  1.91it/s][A
copy images:   3%|▎         | 21/781 [00:13<06:38,  1.91it/s][A
copy images:   3%|▎         | 22/781 [00:13<06:36,  1.91it/s][A
copy images:   3%|▎         | 23/781 [00:14<06:31,  1.93it/s][A
copy images:   3%|▎         | 24/781 [00:14<06:31,  1.93it/s][A
copy images:   3%|▎         | 25/781 [00:15<07:37,  1.65it/s][A
copy images:   3%|▎         | 26/781 [00:16<07:15,  1.73it/s][A
copy images:   3%|▎         | 27/781 [00:16<07:11,  1.75it/s][A
copy images:   4%|▎         | 28/781 [00:17<06:56,  1.81it/s][A
copy images:   4%|▎         | 29/781 [00:17<06:47,  1.84it/s][A
copy images:   4%|▍         | 30/781 [00:18<06:47,  1.84it/s][A
copy images:   4%|▍         | 31/781 [00:18<06:57,  1.80it/s][A
copy images:   4%|▍         | 32/781 [00:20<09:29,  1.31it/s][A
copy images:   4%|▍         | 33/781 [00:20<09:16,  1.34it/s][A
copy images:   4%|▍         | 34/781 [00:21<09:11,  1.35it/s][A
copy images:   4%|▍         | 35/781 [00:22<08:26,  1.47it/s][A
copy images:   5%|▍         | 36/781 [00:22<07:51,  1.58it/s][A
copy images:   5%|▍         | 37/781 [00:24<10:57,  1.13it/s][A
copy images:   5%|▍         | 38/781 [00:24<10:05,  1.23it/s][A
copy images:   5%|▍         | 39/781 [00:25<09:05,  1.36it/s][A
copy images:   5%|▌         | 40/781 [00:26<10:14,  1.21it/s][A
copy images:   5%|▌         | 41/781 [00:27<10:44,  1.15it/s][A
copy images:   5%|▌         | 42/781 [00:27<09:59,  1.23it/s][A
copy images:   6%|▌         | 43/781 [00:28<08:53,  1.38it/s][A
copy images:   6%|▌         | 44/781 [00:29<11:16,  1.09it/s][A
copy images:   6%|▌         | 45/781 [00:30<09:49,  1.25it/s][A
copy images:   6%|▌         | 46/781 [00:30<08:44,  1.40it/s][A
copy images:   6%|▌         | 47/781 [00:31<07:56,  1.54it/s][A
copy images:   6%|▌         | 48/781 [00:31<07:26,  1.64it/s][A
copy images:   6%|▋         | 49/781 [00:32<07:04,  1.72it/s][A
copy images:   6%|▋         | 50/781 [00:32<06:55,  1.76it/s][A
copy images:   7%|▋         | 51/781 [00:33<06:48,  1.79it/s][A
copy images:   7%|▋         | 52/781 [00:33<06:35,  1.84it/s][A
copy images:   7%|▋         | 53/781 [00:34<06:29,  1.87it/s][A
copy images:   7%|▋         | 54/781 [00:35<06:37,  1.83it/s][A
copy images:   7%|▋         | 55/781 [00:35<06:48,  1.78it/s][A
copy images:   7%|▋         | 56/781 [00:36<07:15,  1.67it/s][A
copy images:   7%|▋         | 57/781 [00:37<07:25,  1.63it/s][A
copy images:   7%|▋         | 58/781 [00:37<07:04,  1.70it/s][A
copy images:   8%|▊         | 59/781 [00:38<06:48,  1.77it/s][A
copy images:   8%|▊         | 60/781 [00:38<06:34,  1.83it/s][A
copy images:   8%|▊         | 61/781 [00:39<06:25,  1.87it/s][A
copy images:   8%|▊         | 62/781 [00:39<06:24,  1.87it/s][A
copy images:   8%|▊         | 63/781 [00:40<06:18,  1.90it/s][A
copy images:   8%|▊         | 64/781 [00:41<07:44,  1.54it/s][A
copy images:   8%|▊         | 65/781 [00:41<07:21,  1.62it/s][A
copy images:   8%|▊         | 66/781 [00:42<07:27,  1.60it/s][A
copy images:   9%|▊         | 67/781 [00:42<07:03,  1.69it/s][A
copy images:   9%|▊         | 68/781 [00:43<06:46,  1.76it/s][A
copy images:   9%|▉         | 69/781 [00:43<06:35,  1.80it/s][A
copy images:   9%|▉         | 70/781 [00:44<06:27,  1.83it/s][A
copy images:   9%|▉         | 71/781 [00:44<06:23,  1.85it/s][A
copy images:   9%|▉         | 72/781 [00:45<06:21,  1.86it/s][A
copy images:   9%|▉         | 73/781 [00:45<06:12,  1.90it/s][A
copy images:   9%|▉         | 74/781 [00:46<06:57,  1.69it/s][A
copy images:  10%|▉         | 75/781 [00:47<06:42,  1.76it/s][A
copy images:  10%|▉         | 76/781 [00:47<06:30,  1.80it/s][A
copy images:  10%|▉         | 77/781 [00:48<06:22,  1.84it/s][A
copy images:  10%|▉         | 78/781 [00:48<06:17,  1.86it/s][A
copy images:  10%|█         | 79/781 [00:49<06:34,  1.78it/s][A
copy images:  10%|█         | 80/781 [00:49<06:23,  1.83it/s][A
copy images:  10%|█         | 81/781 [00:50<06:19,  1.84it/s][A
copy images:  10%|█         | 82/781 [00:50<06:22,  1.83it/s][A
copy images:  11%|█         | 83/781 [00:51<06:13,  1.87it/s][A
copy images:  11%|█         | 84/781 [00:51<06:21,  1.83it/s][A
copy images:  11%|█         | 85/781 [00:52<06:24,  1.81it/s][A
copy images:  11%|█         | 86/781 [00:53<06:30,  1.78it/s][A
copy images:  11%|█         | 87/781 [00:53<06:27,  1.79it/s][A
copy images:  11%|█▏        | 88/781 [00:54<06:38,  1.74it/s][A
copy images:  11%|█▏        | 89/781 [00:54<06:27,  1.79it/s][A
copy images:  12%|█▏        | 90/781 [00:55<06:18,  1.82it/s][A
copy images:  12%|█▏        | 91/781 [00:55<06:37,  1.74it/s][A
copy images:  12%|█▏        | 92/781 [00:56<06:43,  1.71it/s][A
copy images:  12%|█▏        | 93/781 [00:57<06:29,  1.76it/s][A
copy images:  12%|█▏        | 94/781 [00:57<06:17,  1.82it/s][A
copy images:  12%|█▏        | 95/781 [00:58<06:12,  1.84it/s][A
copy images:  12%|█▏        | 96/781 [00:58<06:04,  1.88it/s][A
copy images:  12%|█▏        | 97/781 [00:59<07:40,  1.49it/s][A
copy images:  13%|█▎        | 98/781 [01:00<07:09,  1.59it/s][A
copy images:  13%|█▎        | 99/781 [01:00<06:50,  1.66it/s][A
copy images:  13%|█▎        | 100/781 [01:01<06:30,  1.75it/s][A
copy images:  13%|█▎        | 101/781 [01:01<06:17,  1.80it/s][A
copy images:  13%|█▎        | 102/781 [01:02<06:08,  1.84it/s][A
copy images:  13%|█▎        | 103/781 [01:02<06:06,  1.85it/s][A
copy images:  13%|█▎        | 104/781 [01:03<05:57,  1.89it/s][A
copy images:  13%|█▎        | 105/781 [01:03<05:51,  1.93it/s][A
copy images:  14%|█▎        | 106/781 [01:04<05:51,  1.92it/s][A
copy images:  14%|█▎        | 107/781 [01:04<06:05,  1.85it/s][A
copy images:  14%|█▍        | 108/781 [01:05<05:57,  1.89it/s][A
copy images:  14%|█▍        | 109/781 [01:05<05:55,  1.89it/s][A
copy images:  14%|█▍        | 110/781 [01:06<05:51,  1.91it/s][A
copy images:  14%|█▍        | 111/781 [01:07<06:00,  1.86it/s][A
copy images:  14%|█▍        | 112/781 [01:07<05:53,  1.89it/s][A
copy images:  14%|█▍        | 113/781 [01:08<06:03,  1.84it/s][A
copy images:  15%|█▍        | 114/781 [01:08<05:56,  1.87it/s][A
copy images:  15%|█▍        | 115/781 [01:09<06:05,  1.82it/s][A
copy images:  15%|█▍        | 116/781 [01:09<05:58,  1.85it/s][A
copy images:  15%|█▍        | 117/781 [01:10<05:50,  1.90it/s][A
copy images:  15%|█▌        | 118/781 [01:10<06:05,  1.81it/s][A
copy images:  15%|█▌        | 119/781 [01:11<05:56,  1.86it/s][A
copy images:  15%|█▌        | 120/781 [01:11<05:52,  1.88it/s][A
copy images:  15%|█▌        | 121/781 [01:12<05:45,  1.91it/s][A
copy images:  16%|█▌        | 122/781 [01:12<05:46,  1.90it/s][A
copy images:  16%|█▌        | 123/781 [01:13<06:08,  1.79it/s][A
copy images:  16%|█▌        | 124/781 [01:14<05:58,  1.83it/s][A
copy images:  16%|█▌        | 125/781 [01:14<05:52,  1.86it/s][A
copy images:  16%|█▌        | 126/781 [01:15<05:59,  1.82it/s][A
copy images:  16%|█▋        | 127/781 [01:15<05:51,  1.86it/s][A
copy images:  16%|█▋        | 128/781 [01:16<06:17,  1.73it/s][A
copy images:  17%|█▋        | 129/781 [01:16<06:09,  1.77it/s][A
copy images:  17%|█▋        | 130/781 [01:17<05:59,  1.81it/s][A
copy images:  17%|█▋        | 131/781 [01:17<05:46,  1.87it/s][A
copy images:  17%|█▋        | 132/781 [01:18<05:50,  1.85it/s][A
copy images:  17%|█▋        | 133/781 [01:18<05:48,  1.86it/s][A
copy images:  17%|█▋        | 134/781 [01:19<05:51,  1.84it/s][A
copy images:  17%|█▋        | 135/781 [01:20<05:51,  1.84it/s][A
copy images:  17%|█▋        | 136/781 [01:20<06:01,  1.79it/s][A
copy images:  18%|█▊        | 137/781 [01:21<05:53,  1.82it/s][A
copy images:  18%|█▊        | 138/781 [01:21<05:55,  1.81it/s][A
copy images:  18%|█▊        | 139/781 [01:22<05:48,  1.84it/s][A
copy images:  18%|█▊        | 140/781 [01:22<05:47,  1.85it/s][A
copy images:  18%|█▊        | 141/781 [01:23<06:38,  1.61it/s][A
copy images:  18%|█▊        | 142/781 [01:24<06:14,  1.71it/s][A
copy images:  18%|█▊        | 143/781 [01:24<05:57,  1.79it/s][A
copy images:  18%|█▊        | 144/781 [01:25<05:49,  1.82it/s][A
copy images:  19%|█▊        | 145/781 [01:25<06:36,  1.61it/s][A
copy images:  19%|█▊        | 146/781 [01:26<06:12,  1.70it/s][A
copy images:  19%|█▉        | 147/781 [01:27<06:12,  1.70it/s][A
copy images:  19%|█▉        | 148/781 [01:27<05:58,  1.77it/s][A
copy images:  19%|█▉        | 149/781 [01:28<06:24,  1.64it/s][A
copy images:  19%|█▉        | 150/781 [01:28<06:05,  1.72it/s][A
copy images:  19%|█▉        | 151/781 [01:29<06:06,  1.72it/s][A
copy images:  19%|█▉        | 152/781 [01:29<05:56,  1.77it/s][A
copy images:  20%|█▉        | 153/781 [01:30<05:54,  1.77it/s][A
copy images:  20%|█▉        | 154/781 [01:30<05:45,  1.81it/s][A
copy images:  20%|█▉        | 155/781 [01:31<05:56,  1.76it/s][A
copy images:  20%|█▉        | 156/781 [01:32<05:43,  1.82it/s][A
copy images:  20%|██        | 157/781 [01:32<06:06,  1.70it/s][A
copy images:  20%|██        | 158/781 [01:33<05:55,  1.75it/s][A
copy images:  20%|██        | 159/781 [01:33<05:59,  1.73it/s][A
copy images:  20%|██        | 160/781 [01:34<05:58,  1.73it/s][A
copy images:  21%|██        | 161/781 [01:34<05:45,  1.79it/s][A
copy images:  21%|██        | 162/781 [01:35<05:40,  1.82it/s][A
copy images:  21%|██        | 163/781 [01:36<05:38,  1.83it/s][A
copy images:  21%|██        | 164/781 [01:36<05:35,  1.84it/s][A
copy images:  21%|██        | 165/781 [01:37<05:26,  1.88it/s][A
copy images:  21%|██▏       | 166/781 [01:37<05:22,  1.90it/s][A
copy images:  21%|██▏       | 167/781 [01:38<05:24,  1.90it/s][A
copy images:  22%|██▏       | 168/781 [01:38<05:20,  1.91it/s][A
copy images:  22%|██▏       | 169/781 [01:39<06:33,  1.55it/s][A
copy images:  22%|██▏       | 170/781 [01:40<06:08,  1.66it/s][A
copy images:  22%|██▏       | 171/781 [01:40<05:50,  1.74it/s][A
copy images:  22%|██▏       | 172/781 [01:41<05:46,  1.76it/s][A
copy images:  22%|██▏       | 173/781 [01:41<05:33,  1.82it/s][A
copy images:  22%|██▏       | 174/781 [01:42<05:27,  1.85it/s][A
copy images:  22%|██▏       | 175/781 [01:42<05:24,  1.87it/s][A
copy images:  23%|██▎       | 176/781 [01:43<05:26,  1.85it/s][A
copy images:  23%|██▎       | 177/781 [01:43<05:55,  1.70it/s][A
copy images:  23%|██▎       | 178/781 [01:44<05:41,  1.77it/s][A
copy images:  23%|██▎       | 179/781 [01:44<05:29,  1.83it/s][A
copy images:  23%|██▎       | 180/781 [01:45<06:38,  1.51it/s][A
copy images:  23%|██▎       | 181/781 [01:46<06:10,  1.62it/s][A
copy images:  23%|██▎       | 182/781 [01:46<05:52,  1.70it/s][A
copy images:  23%|██▎       | 183/781 [01:47<05:41,  1.75it/s][A
copy images:  24%|██▎       | 184/781 [01:48<05:43,  1.74it/s][A
copy images:  24%|██▎       | 185/781 [01:48<05:55,  1.68it/s][A
copy images:  24%|██▍       | 186/781 [01:49<05:39,  1.75it/s][A
copy images:  24%|██▍       | 187/781 [01:49<05:32,  1.79it/s][A
copy images:  24%|██▍       | 188/781 [01:50<05:23,  1.83it/s][A
copy images:  24%|██▍       | 189/781 [01:50<05:29,  1.80it/s][A
copy images:  24%|██▍       | 190/781 [01:51<05:22,  1.83it/s][A
copy images:  24%|██▍       | 191/781 [01:51<05:22,  1.83it/s][A
copy images:  25%|██▍       | 192/781 [01:52<05:17,  1.86it/s][A
copy images:  25%|██▍       | 193/781 [01:52<05:11,  1.89it/s][A
copy images:  25%|██▍       | 194/781 [01:53<05:07,  1.91it/s][A
copy images:  25%|██▍       | 195/781 [01:53<05:03,  1.93it/s][A
copy images:  25%|██▌       | 196/781 [01:54<05:02,  1.93it/s][A
copy images:  25%|██▌       | 197/781 [01:54<05:02,  1.93it/s][A
copy images:  25%|██▌       | 198/781 [01:55<05:09,  1.88it/s][A
copy images:  25%|██▌       | 199/781 [01:56<06:05,  1.59it/s][A
copy images:  26%|██▌       | 200/781 [01:56<05:55,  1.63it/s][A
copy images:  26%|██▌       | 201/781 [01:57<05:41,  1.70it/s][A
copy images:  26%|██▌       | 202/781 [01:58<05:28,  1.76it/s][A
copy images:  26%|██▌       | 203/781 [01:58<05:29,  1.75it/s][A
copy images:  26%|██▌       | 204/781 [01:59<05:20,  1.80it/s][A
copy images:  26%|██▌       | 205/781 [01:59<05:23,  1.78it/s][A
copy images:  26%|██▋       | 206/781 [02:00<05:12,  1.84it/s][A
copy images:  27%|██▋       | 207/781 [02:00<05:06,  1.87it/s][A
copy images:  27%|██▋       | 208/781 [02:01<05:25,  1.76it/s][A
copy images:  27%|██▋       | 209/781 [02:01<05:16,  1.81it/s][A
copy images:  27%|██▋       | 210/781 [02:02<05:44,  1.66it/s][A
copy images:  27%|██▋       | 211/781 [02:03<05:34,  1.70it/s][A
copy images:  27%|██▋       | 212/781 [02:03<05:24,  1.75it/s][A
copy images:  27%|██▋       | 213/781 [02:04<05:15,  1.80it/s][A
copy images:  27%|██▋       | 214/781 [02:04<05:07,  1.84it/s][A
copy images:  28%|██▊       | 215/781 [02:05<05:03,  1.87it/s][A
copy images:  28%|██▊       | 216/781 [02:05<05:02,  1.87it/s][A
copy images:  28%|██▊       | 217/781 [02:06<04:57,  1.90it/s][A
copy images:  28%|██▊       | 218/781 [02:06<04:52,  1.93it/s][A
copy images:  28%|██▊       | 219/781 [02:07<04:52,  1.92it/s][A
copy images:  28%|██▊       | 220/781 [02:07<04:48,  1.94it/s][A
copy images:  28%|██▊       | 221/781 [02:08<04:54,  1.90it/s][A
copy images:  28%|██▊       | 222/781 [02:08<04:51,  1.92it/s][A
copy images:  29%|██▊       | 223/781 [02:09<04:49,  1.93it/s][A
copy images:  29%|██▊       | 224/781 [02:09<04:46,  1.94it/s][A
copy images:  29%|██▉       | 225/781 [02:10<04:46,  1.94it/s][A
copy images:  29%|██▉       | 226/781 [02:10<04:46,  1.94it/s][A
copy images:  29%|██▉       | 227/781 [02:11<05:04,  1.82it/s][A
copy images:  29%|██▉       | 228/781 [02:12<04:57,  1.86it/s][A
copy images:  29%|██▉       | 229/781 [02:12<04:55,  1.87it/s][A
copy images:  29%|██▉       | 230/781 [02:13<04:54,  1.87it/s][A
copy images:  30%|██▉       | 231/781 [02:13<04:48,  1.91it/s][A
copy images:  30%|██▉       | 232/781 [02:14<04:52,  1.88it/s][A
copy images:  30%|██▉       | 233/781 [02:14<04:50,  1.89it/s][A
copy images:  30%|██▉       | 234/781 [02:15<04:50,  1.89it/s][A
copy images:  30%|███       | 235/781 [02:15<04:46,  1.91it/s][A
copy images:  30%|███       | 236/781 [02:16<04:42,  1.93it/s][A
copy images:  30%|███       | 237/781 [02:16<04:41,  1.93it/s][A
copy images:  30%|███       | 238/781 [02:17<04:40,  1.94it/s][A
copy images:  31%|███       | 239/781 [02:17<04:43,  1.91it/s][A
copy images:  31%|███       | 240/781 [02:18<04:39,  1.94it/s][A
copy images:  31%|███       | 241/781 [02:18<04:36,  1.95it/s][A
copy images:  31%|███       | 242/781 [02:19<04:34,  1.96it/s][A
copy images:  31%|███       | 243/781 [02:19<04:35,  1.95it/s][A
copy images:  31%|███       | 244/781 [02:20<04:46,  1.87it/s][A
copy images:  31%|███▏      | 245/781 [02:21<05:02,  1.77it/s][A
copy images:  31%|███▏      | 246/781 [02:21<04:57,  1.80it/s][A
copy images:  32%|███▏      | 247/781 [02:22<04:58,  1.79it/s][A
copy images:  32%|███▏      | 248/781 [02:22<04:50,  1.83it/s][A
copy images:  32%|███▏      | 249/781 [02:23<04:46,  1.86it/s][A
copy images:  32%|███▏      | 250/781 [02:23<04:41,  1.89it/s][A
copy images:  32%|███▏      | 251/781 [02:24<04:45,  1.86it/s][A
copy images:  32%|███▏      | 252/781 [02:24<04:38,  1.90it/s][A
copy images:  32%|███▏      | 253/781 [02:25<04:42,  1.87it/s][A
copy images:  33%|███▎      | 254/781 [02:25<04:35,  1.91it/s][A
copy images:  33%|███▎      | 255/781 [02:26<04:53,  1.79it/s][A
copy images:  33%|███▎      | 256/781 [02:26<04:44,  1.85it/s][A
copy images:  33%|███▎      | 257/781 [02:27<04:38,  1.88it/s][A
copy images:  33%|███▎      | 258/781 [02:27<04:34,  1.90it/s][A
copy images:  33%|███▎      | 259/781 [02:28<05:04,  1.71it/s][A
copy images:  33%|███▎      | 260/781 [02:29<04:54,  1.77it/s][A
copy images:  33%|███▎      | 261/781 [02:29<05:04,  1.71it/s][A
copy images:  34%|███▎      | 262/781 [02:30<04:53,  1.77it/s][A
copy images:  34%|███▎      | 263/781 [02:30<04:43,  1.83it/s][A
copy images:  34%|███▍      | 264/781 [02:31<04:50,  1.78it/s][A