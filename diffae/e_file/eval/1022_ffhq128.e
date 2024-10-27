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
copy images:  34%|███▍      | 265/781 [02:32<04:54,  1.75it/s][A
copy images:  34%|███▍      | 266/781 [02:32<04:45,  1.80it/s][A
copy images:  34%|███▍      | 267/781 [02:33<04:37,  1.85it/s][A
copy images:  34%|███▍      | 268/781 [02:33<04:37,  1.85it/s][A
copy images:  34%|███▍      | 269/781 [02:34<04:31,  1.89it/s][A
copy images:  35%|███▍      | 270/781 [02:34<04:47,  1.78it/s][A
copy images:  35%|███▍      | 271/781 [02:35<04:41,  1.81it/s][A
copy images:  35%|███▍      | 272/781 [02:35<04:54,  1.73it/s][A
copy images:  35%|███▍      | 273/781 [02:36<04:46,  1.77it/s][A
copy images:  35%|███▌      | 274/781 [02:37<04:55,  1.72it/s][A
copy images:  35%|███▌      | 275/781 [02:37<04:43,  1.78it/s][A
copy images:  35%|███▌      | 276/781 [02:38<05:39,  1.49it/s][A
copy images:  35%|███▌      | 277/781 [02:39<05:26,  1.54it/s][A
copy images:  36%|███▌      | 278/781 [02:39<05:06,  1.64it/s][A
copy images:  36%|███▌      | 279/781 [02:40<04:58,  1.68it/s][A
copy images:  36%|███▌      | 280/781 [02:40<04:45,  1.75it/s][A
copy images:  36%|███▌      | 281/781 [02:41<04:37,  1.80it/s][A
copy images:  36%|███▌      | 282/781 [02:41<04:29,  1.85it/s][A
copy images:  36%|███▌      | 283/781 [02:42<04:24,  1.88it/s][A
copy images:  36%|███▋      | 284/781 [02:42<04:23,  1.88it/s][A
copy images:  36%|███▋      | 285/781 [02:43<04:19,  1.91it/s][A
copy images:  37%|███▋      | 286/781 [02:43<04:16,  1.93it/s][A
copy images:  37%|███▋      | 287/781 [02:44<04:13,  1.95it/s][A
copy images:  37%|███▋      | 288/781 [02:44<04:20,  1.89it/s][A
copy images:  37%|███▋      | 289/781 [02:45<04:15,  1.92it/s][A
copy images:  37%|███▋      | 290/781 [02:45<04:14,  1.93it/s][A
copy images:  37%|███▋      | 291/781 [02:46<04:12,  1.94it/s][A
copy images:  37%|███▋      | 292/781 [02:46<04:11,  1.95it/s][A
copy images:  38%|███▊      | 293/781 [02:48<05:52,  1.38it/s][A
copy images:  38%|███▊      | 294/781 [02:48<05:26,  1.49it/s][A
copy images:  38%|███▊      | 295/781 [02:49<05:02,  1.61it/s][A
copy images:  38%|███▊      | 296/781 [02:49<04:58,  1.62it/s][A
copy images:  38%|███▊      | 297/781 [02:50<04:41,  1.72it/s][A
copy images:  38%|███▊      | 298/781 [02:50<04:30,  1.79it/s][A
copy images:  38%|███▊      | 299/781 [02:51<04:31,  1.77it/s][A
copy images:  38%|███▊      | 300/781 [02:51<04:44,  1.69it/s][A
copy images:  39%|███▊      | 301/781 [02:52<04:34,  1.75it/s][A
copy images:  39%|███▊      | 302/781 [02:52<04:22,  1.83it/s][A
copy images:  39%|███▉      | 303/781 [02:53<04:15,  1.87it/s][A
copy images:  39%|███▉      | 304/781 [02:54<04:22,  1.82it/s][A
copy images:  39%|███▉      | 305/781 [02:54<04:19,  1.83it/s][A
copy images:  39%|███▉      | 306/781 [02:55<04:14,  1.87it/s][A
copy images:  39%|███▉      | 307/781 [02:55<04:10,  1.89it/s][A
copy images:  39%|███▉      | 308/781 [02:56<04:06,  1.92it/s][A
copy images:  40%|███▉      | 309/781 [02:56<04:10,  1.88it/s][A
copy images:  40%|███▉      | 310/781 [02:57<04:18,  1.82it/s][A
copy images:  40%|███▉      | 311/781 [02:57<04:12,  1.86it/s][A
copy images:  40%|███▉      | 312/781 [02:58<04:10,  1.87it/s][A
copy images:  40%|████      | 313/781 [02:58<04:06,  1.90it/s][A
copy images:  40%|████      | 314/781 [02:59<04:03,  1.92it/s][A
copy images:  40%|████      | 315/781 [02:59<04:02,  1.92it/s][A
copy images:  40%|████      | 316/781 [03:00<04:00,  1.93it/s][A
copy images:  41%|████      | 317/781 [03:00<03:58,  1.95it/s][A
copy images:  41%|████      | 318/781 [03:01<03:57,  1.95it/s][A
copy images:  41%|████      | 319/781 [03:02<04:15,  1.81it/s][A
copy images:  41%|████      | 320/781 [03:02<04:30,  1.70it/s][A
copy images:  41%|████      | 321/781 [03:03<05:19,  1.44it/s][A
copy images:  41%|████      | 322/781 [03:04<04:52,  1.57it/s][A
copy images:  41%|████▏     | 323/781 [03:04<04:39,  1.64it/s][A
copy images:  41%|████▏     | 324/781 [03:05<04:23,  1.73it/s][A
copy images:  42%|████▏     | 325/781 [03:05<04:13,  1.80it/s][A
copy images:  42%|████▏     | 326/781 [03:06<04:06,  1.85it/s][A
copy images:  42%|████▏     | 327/781 [03:06<04:34,  1.66it/s][A
copy images:  42%|████▏     | 328/781 [03:07<04:20,  1.74it/s][A
copy images:  42%|████▏     | 329/781 [03:08<04:16,  1.76it/s][A
copy images:  42%|████▏     | 330/781 [03:08<04:05,  1.83it/s][A
copy images:  42%|████▏     | 331/781 [03:09<04:11,  1.79it/s][A
copy images:  43%|████▎     | 332/781 [03:09<04:04,  1.84it/s][A
copy images:  43%|████▎     | 333/781 [03:10<03:57,  1.89it/s][A
copy images:  43%|████▎     | 334/781 [03:10<04:02,  1.84it/s][A
copy images:  43%|████▎     | 335/781 [03:11<03:59,  1.86it/s][A
copy images:  43%|████▎     | 336/781 [03:11<03:53,  1.91it/s][A
copy images:  43%|████▎     | 337/781 [03:12<03:58,  1.86it/s][A
copy images:  43%|████▎     | 338/781 [03:12<03:54,  1.89it/s][A
copy images:  43%|████▎     | 339/781 [03:13<03:50,  1.91it/s][A
copy images:  44%|████▎     | 340/781 [03:13<03:46,  1.94it/s][A
copy images:  44%|████▎     | 341/781 [03:14<03:43,  1.97it/s][A
copy images:  44%|████▍     | 342/781 [03:14<03:45,  1.94it/s][A
copy images:  44%|████▍     | 343/781 [03:15<03:43,  1.96it/s][A
copy images:  44%|████▍     | 344/781 [03:15<03:42,  1.96it/s][A
copy images:  44%|████▍     | 345/781 [03:16<03:41,  1.97it/s][A
copy images:  44%|████▍     | 346/781 [03:16<03:57,  1.83it/s][A
copy images:  44%|████▍     | 347/781 [03:17<03:53,  1.86it/s][A
copy images:  45%|████▍     | 348/781 [03:17<03:50,  1.88it/s][A
copy images:  45%|████▍     | 349/781 [03:18<03:46,  1.91it/s][A
copy images:  45%|████▍     | 350/781 [03:19<03:47,  1.90it/s][A
copy images:  45%|████▍     | 351/781 [03:19<04:23,  1.63it/s][A
copy images:  45%|████▌     | 352/781 [03:20<04:08,  1.72it/s][A
copy images:  45%|████▌     | 353/781 [03:20<04:04,  1.75it/s][A
copy images:  45%|████▌     | 354/781 [03:21<03:56,  1.81it/s][A
copy images:  45%|████▌     | 355/781 [03:21<03:51,  1.84it/s][A
copy images:  46%|████▌     | 356/781 [03:22<03:47,  1.87it/s][A
copy images:  46%|████▌     | 357/781 [03:22<03:43,  1.90it/s][A
copy images:  46%|████▌     | 358/781 [03:23<04:37,  1.53it/s][A
copy images:  46%|████▌     | 359/781 [03:24<04:23,  1.60it/s][A
copy images:  46%|████▌     | 360/781 [03:24<04:08,  1.69it/s][A
copy images:  46%|████▌     | 361/781 [03:25<03:58,  1.76it/s][A
copy images:  46%|████▋     | 362/781 [03:26<04:18,  1.62it/s][A
copy images:  46%|████▋     | 363/781 [03:26<04:03,  1.72it/s][A
copy images:  47%|████▋     | 364/781 [03:27<03:53,  1.78it/s][A
copy images:  47%|████▋     | 365/781 [03:27<03:46,  1.84it/s][A
copy images:  47%|████▋     | 366/781 [03:28<03:42,  1.87it/s][A
copy images:  47%|████▋     | 367/781 [03:28<03:40,  1.88it/s][A
copy images:  47%|████▋     | 368/781 [03:29<03:38,  1.89it/s][A
copy images:  47%|████▋     | 369/781 [03:29<03:34,  1.92it/s][A
copy images:  47%|████▋     | 370/781 [03:30<03:31,  1.94it/s][A
copy images:  48%|████▊     | 371/781 [03:30<03:31,  1.94it/s][A
copy images:  48%|████▊     | 372/781 [03:31<03:54,  1.75it/s][A
copy images:  48%|████▊     | 373/781 [03:32<03:48,  1.78it/s][A
copy images:  48%|████▊     | 374/781 [03:32<03:41,  1.83it/s][A
copy images:  48%|████▊     | 375/781 [03:33<03:37,  1.87it/s][A
copy images:  48%|████▊     | 376/781 [03:33<03:35,  1.88it/s][A
copy images:  48%|████▊     | 377/781 [03:34<03:33,  1.90it/s][A
copy images:  48%|████▊     | 378/781 [03:34<03:30,  1.92it/s][A
copy images:  49%|████▊     | 379/781 [03:35<03:32,  1.90it/s][A
copy images:  49%|████▊     | 380/781 [03:35<03:30,  1.90it/s][A
copy images:  49%|████▉     | 381/781 [03:36<03:45,  1.78it/s][A
copy images:  49%|████▉     | 382/781 [03:36<03:40,  1.81it/s][A
copy images:  49%|████▉     | 383/781 [03:37<04:26,  1.49it/s][A
copy images:  49%|████▉     | 384/781 [03:38<04:13,  1.57it/s][A
copy images:  49%|████▉     | 385/781 [03:38<03:56,  1.67it/s][A
copy images:  49%|████▉     | 386/781 [03:39<03:48,  1.73it/s][A
copy images:  50%|████▉     | 387/781 [03:40<03:56,  1.66it/s][A
copy images:  50%|████▉     | 388/781 [03:40<03:47,  1.72it/s][A
copy images:  50%|████▉     | 389/781 [03:41<03:38,  1.80it/s][A
copy images:  50%|████▉     | 390/781 [03:41<03:41,  1.76it/s][A
copy images:  50%|█████     | 391/781 [03:42<03:52,  1.67it/s][A
copy images:  50%|█████     | 392/781 [03:42<03:46,  1.71it/s][A
copy images:  50%|█████     | 393/781 [03:43<03:35,  1.80it/s][A
copy images:  50%|█████     | 394/781 [03:43<03:30,  1.84it/s][A
copy images:  51%|█████     | 395/781 [03:44<03:27,  1.86it/s][A
copy images:  51%|█████     | 396/781 [03:45<03:43,  1.72it/s][A
copy images:  51%|█████     | 397/781 [03:45<03:33,  1.79it/s][A
copy images:  51%|█████     | 398/781 [03:46<03:47,  1.69it/s][A
copy images:  51%|█████     | 399/781 [03:46<03:39,  1.74it/s][A
copy images:  51%|█████     | 400/781 [03:47<04:28,  1.42it/s][A
copy images:  51%|█████▏    | 401/781 [03:48<04:10,  1.52it/s][A
copy images:  51%|█████▏    | 402/781 [03:48<03:59,  1.58it/s][A
copy images:  52%|█████▏    | 403/781 [03:49<03:59,  1.58it/s][A
copy images:  52%|█████▏    | 404/781 [03:50<04:04,  1.54it/s][A
copy images:  52%|█████▏    | 405/781 [03:50<03:48,  1.64it/s][A
copy images:  52%|█████▏    | 406/781 [03:51<03:39,  1.71it/s][A
copy images:  52%|█████▏    | 407/781 [03:51<03:44,  1.67it/s][A
copy images:  52%|█████▏    | 408/781 [03:52<03:32,  1.75it/s][A
copy images:  52%|█████▏    | 409/781 [03:52<03:25,  1.81it/s][A
copy images:  52%|█████▏    | 410/781 [03:53<03:21,  1.84it/s][A
copy images:  53%|█████▎    | 411/781 [03:54<03:33,  1.73it/s][A
copy images:  53%|█████▎    | 412/781 [03:54<03:25,  1.80it/s][A
copy images:  53%|█████▎    | 413/781 [03:55<03:18,  1.85it/s][A
copy images:  53%|█████▎    | 414/781 [03:55<03:13,  1.89it/s][A
copy images:  53%|█████▎    | 415/781 [03:56<03:46,  1.62it/s][A
copy images:  53%|█████▎    | 416/781 [03:56<03:32,  1.71it/s][A
copy images:  53%|█████▎    | 417/781 [03:57<03:23,  1.78it/s][A
copy images:  54%|█████▎    | 418/781 [03:58<03:17,  1.83it/s][A
copy images:  54%|█████▎    | 419/781 [03:58<03:29,  1.72it/s][A
copy images:  54%|█████▍    | 420/781 [03:59<03:28,  1.73it/s][A
copy images:  54%|█████▍    | 421/781 [03:59<03:20,  1.80it/s][A
copy images:  54%|█████▍    | 422/781 [04:00<03:31,  1.70it/s][A
copy images:  54%|█████▍    | 423/781 [04:00<03:21,  1.77it/s][A
copy images:  54%|█████▍    | 424/781 [04:01<03:16,  1.82it/s][A
copy images:  54%|█████▍    | 425/781 [04:01<03:11,  1.86it/s][A
copy images:  55%|█████▍    | 426/781 [04:02<03:06,  1.91it/s][A
copy images:  55%|█████▍    | 427/781 [04:02<03:03,  1.93it/s][A
copy images:  55%|█████▍    | 428/781 [04:03<03:16,  1.79it/s][A
copy images:  55%|█████▍    | 429/781 [04:04<03:09,  1.86it/s][A
copy images:  55%|█████▌    | 430/781 [04:04<03:06,  1.88it/s][A
copy images:  55%|█████▌    | 431/781 [04:05<03:06,  1.88it/s][A
copy images:  55%|█████▌    | 432/781 [04:05<03:02,  1.92it/s][A
copy images:  55%|█████▌    | 433/781 [04:06<02:59,  1.94it/s][A
copy images:  56%|█████▌    | 434/781 [04:06<02:57,  1.96it/s][A
copy images:  56%|█████▌    | 435/781 [04:07<02:57,  1.95it/s][A
copy images:  56%|█████▌    | 436/781 [04:08<03:42,  1.55it/s][A
copy images:  56%|█████▌    | 437/781 [04:08<03:36,  1.59it/s][A
copy images:  56%|█████▌    | 438/781 [04:09<03:23,  1.68it/s][A
copy images:  56%|█████▌    | 439/781 [04:09<03:14,  1.76it/s][A
copy images:  56%|█████▋    | 440/781 [04:10<03:07,  1.82it/s][A
copy images:  56%|█████▋    | 441/781 [04:10<03:02,  1.86it/s][A
copy images:  57%|█████▋    | 442/781 [04:11<03:06,  1.82it/s][A
copy images:  57%|█████▋    | 443/781 [04:11<03:00,  1.87it/s][A
copy images:  57%|█████▋    | 444/781 [04:12<02:59,  1.87it/s][A
copy images:  57%|█████▋    | 445/781 [04:12<02:57,  1.89it/s][A
copy images:  57%|█████▋    | 446/781 [04:14<04:09,  1.35it/s][A
copy images:  57%|█████▋    | 447/781 [04:14<03:45,  1.48it/s][A
copy images:  57%|█████▋    | 448/781 [04:15<03:44,  1.48it/s][A
copy images:  57%|█████▋    | 449/781 [04:15<03:30,  1.58it/s][A
copy images:  58%|█████▊    | 450/781 [04:16<03:16,  1.68it/s][A
copy images:  58%|█████▊    | 451/781 [04:16<03:09,  1.74it/s][A
copy images:  58%|█████▊    | 452/781 [04:17<03:26,  1.60it/s][A
copy images:  58%|█████▊    | 453/781 [04:18<03:17,  1.66it/s][A
copy images:  58%|█████▊    | 454/781 [04:18<03:37,  1.50it/s][A
copy images:  58%|█████▊    | 455/781 [04:19<03:26,  1.58it/s][A
copy images:  58%|█████▊    | 456/781 [04:20<03:13,  1.68it/s][A
copy images:  59%|█████▊    | 457/781 [04:20<03:27,  1.56it/s][A
copy images:  59%|█████▊    | 458/781 [04:21<04:11,  1.29it/s][A
copy images:  59%|█████▉    | 459/781 [04:22<03:43,  1.44it/s][A
copy images:  59%|█████▉    | 460/781 [04:22<03:26,  1.55it/s][A
copy images:  59%|█████▉    | 461/781 [04:23<03:15,  1.64it/s][A
copy images:  59%|█████▉    | 462/781 [04:23<03:11,  1.67it/s][A
copy images:  59%|█████▉    | 463/781 [04:24<03:01,  1.75it/s][A
copy images:  59%|█████▉    | 464/781 [04:25<02:54,  1.82it/s][A
copy images:  60%|█████▉    | 465/781 [04:26<03:38,  1.44it/s][A
copy images:  60%|█████▉    | 466/781 [04:26<03:26,  1.52it/s][A
copy images:  60%|█████▉    | 467/781 [04:27<03:15,  1.61it/s][A
copy images:  60%|█████▉    | 468/781 [04:27<03:03,  1.70it/s][A
copy images:  60%|██████    | 469/781 [04:28<02:57,  1.76it/s][A
copy images:  60%|██████    | 470/781 [04:28<02:53,  1.79it/s][A
copy images:  60%|██████    | 471/781 [04:29<03:19,  1.55it/s][A
copy images:  60%|██████    | 472/781 [04:30<03:06,  1.66it/s][A
copy images:  61%|██████    | 473/781 [04:30<02:57,  1.74it/s][A
copy images:  61%|██████    | 474/781 [04:31<03:16,  1.56it/s][A
copy images:  61%|██████    | 475/781 [04:31<03:08,  1.63it/s][A
copy images:  61%|██████    | 476/781 [04:32<02:57,  1.72it/s][A
copy images:  61%|██████    | 477/781 [04:32<02:49,  1.80it/s][A
copy images:  61%|██████    | 478/781 [04:33<03:09,  1.60it/s][A
copy images:  61%|██████▏   | 479/781 [04:34<02:57,  1.70it/s][A
copy images:  61%|██████▏   | 480/781 [04:34<02:48,  1.79it/s][A
copy images:  62%|██████▏   | 481/781 [04:35<02:42,  1.85it/s][A
copy images:  62%|██████▏   | 482/781 [04:35<02:39,  1.88it/s][A
copy images:  62%|██████▏   | 483/781 [04:36<02:39,  1.87it/s][A
copy images:  62%|██████▏   | 484/781 [04:36<02:48,  1.76it/s][A
copy images:  62%|██████▏   | 485/781 [04:37<02:58,  1.65it/s][A
copy images:  62%|██████▏   | 486/781 [04:38<02:51,  1.72it/s][A
copy images:  62%|██████▏   | 487/781 [04:38<02:49,  1.74it/s][A
copy images:  62%|██████▏   | 488/781 [04:39<02:42,  1.80it/s][A
copy images:  63%|██████▎   | 489/781 [04:39<02:49,  1.72it/s][A
copy images:  63%|██████▎   | 490/781 [04:40<02:44,  1.77it/s][A
copy images:  63%|██████▎   | 491/781 [04:40<02:38,  1.83it/s][A
copy images:  63%|██████▎   | 492/781 [04:41<02:46,  1.74it/s][A
copy images:  63%|██████▎   | 493/781 [04:42<02:52,  1.67it/s][A
copy images:  63%|██████▎   | 494/781 [04:42<02:49,  1.69it/s][A
copy images:  63%|██████▎   | 495/781 [04:43<02:44,  1.74it/s][A
copy images:  64%|██████▎   | 496/781 [04:43<02:47,  1.70it/s][A
copy images:  64%|██████▎   | 497/781 [04:44<02:42,  1.74it/s][A
copy images:  64%|██████▍   | 498/781 [04:44<02:36,  1.81it/s][A
copy images:  64%|██████▍   | 499/781 [04:45<02:38,  1.78it/s][A
copy images:  64%|██████▍   | 500/781 [04:46<02:34,  1.82it/s][A
copy images:  64%|██████▍   | 501/781 [04:46<02:44,  1.70it/s][A
copy images:  64%|██████▍   | 502/781 [04:47<02:39,  1.75it/s][A
copy images:  64%|██████▍   | 503/781 [04:47<02:33,  1.81it/s][A
copy images:  65%|██████▍   | 504/781 [04:48<02:28,  1.86it/s][A
copy images:  65%|██████▍   | 505/781 [04:48<02:27,  1.88it/s][A
copy images:  65%|██████▍   | 506/781 [04:49<02:24,  1.90it/s][A
copy images:  65%|██████▍   | 507/781 [04:49<02:24,  1.90it/s][A
copy images:  65%|██████▌   | 508/781 [04:50<02:38,  1.73it/s][A
copy images:  65%|██████▌   | 509/781 [04:51<02:32,  1.78it/s][A
copy images:  65%|██████▌   | 510/781 [04:51<02:28,  1.83it/s][A
copy images:  65%|██████▌   | 511/781 [04:52<02:26,  1.84it/s][A
copy images:  66%|██████▌   | 512/781 [04:52<02:22,  1.89it/s][A
copy images:  66%|██████▌   | 513/781 [04:53<02:51,  1.56it/s][A
copy images:  66%|██████▌   | 514/781 [04:54<02:49,  1.58it/s][A
copy images:  66%|██████▌   | 515/781 [04:54<02:37,  1.69it/s][A
copy images:  66%|██████▌   | 516/781 [04:55<02:48,  1.57it/s][A
copy images:  66%|██████▌   | 517/781 [04:55<02:39,  1.65it/s][A
copy images:  66%|██████▋   | 518/781 [04:56<02:30,  1.74it/s][A
copy images:  66%|██████▋   | 519/781 [04:56<02:35,  1.68it/s][A
copy images:  67%|██████▋   | 520/781 [04:57<02:42,  1.61it/s][A
copy images:  67%|██████▋   | 521/781 [04:58<02:33,  1.69it/s][A
copy images:  67%|██████▋   | 522/781 [04:58<02:30,  1.73it/s][A
copy images:  67%|██████▋   | 523/781 [04:59<02:24,  1.79it/s][A
copy images:  67%|██████▋   | 524/781 [04:59<02:20,  1.83it/s][A
copy images:  67%|██████▋   | 525/781 [05:00<02:17,  1.87it/s][A
copy images:  67%|██████▋   | 526/781 [05:00<02:14,  1.90it/s][A
copy images:  67%|██████▋   | 527/781 [05:01<02:12,  1.92it/s][A
copy images:  68%|██████▊   | 528/781 [05:01<02:11,  1.93it/s][A
copy images:  68%|██████▊   | 529/781 [05:02<02:10,  1.94it/s][A
copy images:  68%|██████▊   | 530/781 [05:02<02:09,  1.93it/s][A
copy images:  68%|██████▊   | 531/781 [05:03<02:08,  1.94it/s][A
copy images:  68%|██████▊   | 532/781 [05:03<02:08,  1.94it/s][A
copy images:  68%|██████▊   | 533/781 [05:04<02:10,  1.91it/s][A
copy images:  68%|██████▊   | 534/781 [05:04<02:08,  1.92it/s][A
copy images:  69%|██████▊   | 535/781 [05:05<02:06,  1.94it/s][A
copy images:  69%|██████▊   | 536/781 [05:06<02:24,  1.70it/s][A
copy images:  69%|██████▉   | 537/781 [05:06<02:18,  1.76it/s][A
copy images:  69%|██████▉   | 538/781 [05:07<02:14,  1.81it/s][A
copy images:  69%|██████▉   | 539/781 [05:07<02:18,  1.75it/s][A
copy images:  69%|██████▉   | 540/781 [05:08<02:11,  1.83it/s][A
copy images:  69%|██████▉   | 541/781 [05:08<02:16,  1.76it/s][A
copy images:  69%|██████▉   | 542/781 [05:09<02:11,  1.82it/s][A
copy images:  70%|██████▉   | 543/781 [05:09<02:07,  1.87it/s][A
copy images:  70%|██████▉   | 544/781 [05:10<02:20,  1.68it/s][A
copy images:  70%|██████▉   | 545/781 [05:11<02:15,  1.74it/s][A
copy images:  70%|██████▉   | 546/781 [05:11<02:09,  1.81it/s][A
copy images:  70%|███████   | 547/781 [05:12<02:06,  1.85it/s][A
copy images:  70%|███████   | 548/781 [05:12<02:21,  1.65it/s][A
copy images:  70%|███████   | 549/781 [05:13<02:14,  1.73it/s][A
copy images:  70%|███████   | 550/781 [05:14<02:39,  1.45it/s][A
copy images:  71%|███████   | 551/781 [05:15<02:37,  1.46it/s][A
copy images:  71%|███████   | 552/781 [05:15<02:26,  1.57it/s][A
copy images:  71%|███████   | 553/781 [05:16<02:16,  1.67it/s][A
copy images:  71%|███████   | 554/781 [05:16<02:29,  1.52it/s][A
copy images:  71%|███████   | 555/781 [05:17<02:18,  1.63it/s][A
copy images:  71%|███████   | 556/781 [05:18<02:20,  1.60it/s][A
copy images:  71%|███████▏  | 557/781 [05:18<02:12,  1.69it/s][A
copy images:  71%|███████▏  | 558/781 [05:19<02:07,  1.75it/s][A
copy images:  72%|███████▏  | 559/781 [05:19<02:02,  1.81it/s][A
copy images:  72%|███████▏  | 560/781 [05:20<01:59,  1.85it/s][A
copy images:  72%|███████▏  | 561/781 [05:21<02:18,  1.59it/s][A
copy images:  72%|███████▏  | 562/781 [05:21<02:10,  1.68it/s][A
copy images:  72%|███████▏  | 563/781 [05:22<02:05,  1.74it/s][A
copy images:  72%|███████▏  | 564/781 [05:22<02:00,  1.81it/s][A
copy images:  72%|███████▏  | 565/781 [05:23<02:09,  1.67it/s][A
copy images:  72%|███████▏  | 566/781 [05:23<02:03,  1.75it/s][A
copy images:  73%|███████▎  | 567/781 [05:24<01:58,  1.80it/s][A
copy images:  73%|███████▎  | 568/781 [05:24<01:54,  1.87it/s][A
copy images:  73%|███████▎  | 569/781 [05:25<01:51,  1.91it/s][A
copy images:  73%|███████▎  | 570/781 [05:25<01:53,  1.86it/s][A
copy images:  73%|███████▎  | 571/781 [05:26<01:51,  1.88it/s][A
copy images:  73%|███████▎  | 572/781 [05:27<02:00,  1.74it/s][A
copy images:  73%|███████▎  | 573/781 [05:27<02:02,  1.69it/s][A
copy images:  73%|███████▎  | 574/781 [05:28<01:57,  1.76it/s][A
copy images:  74%|███████▎  | 575/781 [05:28<01:53,  1.81it/s][A
copy images:  74%|███████▍  | 576/781 [05:29<01:55,  1.77it/s][A
copy images:  74%|███████▍  | 577/781 [05:29<01:52,  1.81it/s][A
copy images:  74%|███████▍  | 578/781 [05:30<01:49,  1.86it/s][A
copy images:  74%|███████▍  | 579/781 [05:30<01:47,  1.89it/s][A
copy images:  74%|███████▍  | 580/781 [05:32<02:27,  1.36it/s][A
copy images:  74%|███████▍  | 581/781 [05:32<02:25,  1.38it/s][A
copy images:  75%|███████▍  | 582/781 [05:33<02:10,  1.52it/s][A
copy images:  75%|███████▍  | 583/781 [05:33<02:11,  1.50it/s][A
copy images:  75%|███████▍  | 584/781 [05:34<02:05,  1.57it/s][A
copy images:  75%|███████▍  | 585/781 [05:35<01:58,  1.65it/s][A
copy images:  75%|███████▌  | 586/781 [05:35<01:54,  1.71it/s][A
copy images:  75%|███████▌  | 587/781 [05:36<01:49,  1.77it/s][A
copy images:  75%|███████▌  | 588/781 [05:36<01:48,  1.77it/s][A
copy images:  75%|███████▌  | 589/781 [05:37<01:46,  1.81it/s][A
copy images:  76%|███████▌  | 590/781 [05:37<01:46,  1.79it/s][A
copy images:  76%|███████▌  | 591/781 [05:38<01:43,  1.84it/s][A
copy images:  76%|███████▌  | 592/781 [05:39<01:56,  1.62it/s][A
copy images:  76%|███████▌  | 593/781 [05:39<01:56,  1.62it/s][A
copy images:  76%|███████▌  | 594/781 [05:40<01:49,  1.71it/s][A
copy images:  76%|███████▌  | 595/781 [05:40<01:43,  1.79it/s][A
copy images:  76%|███████▋  | 596/781 [05:41<02:14,  1.38it/s][A
copy images:  76%|███████▋  | 597/781 [05:42<02:01,  1.51it/s][A
copy images:  77%|███████▋  | 598/781 [05:42<01:54,  1.59it/s][A
copy images:  77%|███████▋  | 599/781 [05:43<01:47,  1.69it/s][A
copy images:  77%|███████▋  | 600/781 [05:44<02:13,  1.36it/s][A
copy images:  77%|███████▋  | 601/781 [05:45<02:03,  1.46it/s][A
copy images:  77%|███████▋  | 602/781 [05:45<01:56,  1.53it/s][A
copy images:  77%|███████▋  | 603/781 [05:46<02:03,  1.44it/s][A
copy images:  77%|███████▋  | 604/781 [05:46<01:53,  1.56it/s][A
copy images:  77%|███████▋  | 605/781 [05:47<01:45,  1.67it/s][A
copy images:  78%|███████▊  | 606/781 [05:47<01:40,  1.75it/s][A
copy images:  78%|███████▊  | 607/781 [05:48<01:35,  1.82it/s][A
copy images:  78%|███████▊  | 608/781 [05:48<01:33,  1.85it/s][A
copy images:  78%|███████▊  | 609/781 [05:49<01:43,  1.67it/s][A
copy images:  78%|███████▊  | 610/781 [05:50<01:37,  1.75it/s][A
copy images:  78%|███████▊  | 611/781 [05:50<01:34,  1.80it/s][A
copy images:  78%|███████▊  | 612/781 [05:51<01:31,  1.85it/s][A
copy images:  78%|███████▊  | 613/781 [05:51<01:29,  1.88it/s][A
copy images:  79%|███████▊  | 614/781 [05:52<01:27,  1.91it/s][A
copy images:  79%|███████▊  | 615/781 [05:53<01:42,  1.62it/s][A
copy images:  79%|███████▉  | 616/781 [05:53<01:47,  1.54it/s][A
copy images:  79%|███████▉  | 617/781 [05:54<01:38,  1.67it/s][A
copy images:  79%|███████▉  | 618/781 [05:54<01:33,  1.75it/s][A
copy images:  79%|███████▉  | 619/781 [05:55<01:29,  1.81it/s][A
copy images:  79%|███████▉  | 620/781 [05:55<01:29,  1.80it/s][A
copy images:  80%|███████▉  | 621/781 [05:56<01:26,  1.85it/s][A
copy images:  80%|███████▉  | 622/781 [05:56<01:24,  1.88it/s][A
copy images:  80%|███████▉  | 623/781 [05:57<01:26,  1.83it/s][A
copy images:  80%|███████▉  | 624/781 [05:58<01:38,  1.59it/s][A
copy images:  80%|████████  | 625/781 [05:58<01:36,  1.62it/s][A
copy images:  80%|████████  | 626/781 [05:59<01:31,  1.70it/s][A
copy images:  80%|████████  | 627/781 [05:59<01:30,  1.70it/s][A
copy images:  80%|████████  | 628/781 [06:00<01:27,  1.76it/s][A
copy images:  81%|████████  | 629/781 [06:00<01:24,  1.81it/s][A
copy images:  81%|████████  | 630/781 [06:01<01:28,  1.71it/s][A
copy images:  81%|████████  | 631/781 [06:02<01:26,  1.73it/s][A
copy images:  81%|████████  | 632/781 [06:02<01:22,  1.80it/s][A
copy images:  81%|████████  | 633/781 [06:03<01:20,  1.83it/s][A
copy images:  81%|████████  | 634/781 [06:03<01:19,  1.84it/s][A
copy images:  81%|████████▏ | 635/781 [06:04<01:19,  1.84it/s][A
copy images:  81%|████████▏ | 636/781 [06:04<01:18,  1.86it/s][A
copy images:  82%|████████▏ | 637/781 [06:05<01:16,  1.89it/s][A
copy images:  82%|████████▏ | 638/781 [06:05<01:16,  1.88it/s][A
copy images:  82%|████████▏ | 639/781 [06:06<01:14,  1.90it/s][A
copy images:  82%|████████▏ | 640/781 [06:06<01:14,  1.89it/s][A
copy images:  82%|████████▏ | 641/781 [06:07<01:13,  1.90it/s][A
copy images:  82%|████████▏ | 642/781 [06:07<01:12,  1.91it/s][A
copy images:  82%|████████▏ | 643/781 [06:08<01:13,  1.87it/s][A
copy images:  82%|████████▏ | 644/781 [06:09<01:14,  1.84it/s][A
copy images:  83%|████████▎ | 645/781 [06:09<01:11,  1.90it/s][A
copy images:  83%|████████▎ | 646/781 [06:10<01:10,  1.90it/s][A
copy images:  83%|████████▎ | 647/781 [06:10<01:09,  1.93it/s][A
copy images:  83%|████████▎ | 648/781 [06:11<01:08,  1.94it/s][A
copy images:  83%|████████▎ | 649/781 [06:11<01:13,  1.80it/s][A
copy images:  83%|████████▎ | 650/781 [06:12<01:12,  1.80it/s][A
copy images:  83%|████████▎ | 651/781 [06:12<01:10,  1.85it/s][A
copy images:  83%|████████▎ | 652/781 [06:13<01:10,  1.84it/s][A
copy images:  84%|████████▎ | 653/781 [06:13<01:09,  1.85it/s][A
copy images:  84%|████████▎ | 654/781 [06:14<01:16,  1.66it/s][A
copy images:  84%|████████▍ | 655/781 [06:15<01:12,  1.74it/s][A
copy images:  84%|████████▍ | 656/781 [06:15<01:09,  1.81it/s][A
copy images:  84%|████████▍ | 657/781 [06:16<01:16,  1.62it/s][A
copy images:  84%|████████▍ | 658/781 [06:16<01:11,  1.72it/s][A
copy images:  84%|████████▍ | 659/781 [06:17<01:09,  1.75it/s][A
copy images:  85%|████████▍ | 660/781 [06:17<01:06,  1.82it/s][A
copy images:  85%|████████▍ | 661/781 [06:18<01:09,  1.73it/s][A
copy images:  85%|████████▍ | 662/781 [06:19<01:09,  1.72it/s][A
copy images:  85%|████████▍ | 663/781 [06:20<01:24,  1.39it/s][A
copy images:  85%|████████▌ | 664/781 [06:20<01:19,  1.46it/s][A
copy images:  85%|████████▌ | 665/781 [06:21<01:12,  1.59it/s][A
copy images:  85%|████████▌ | 666/781 [06:21<01:08,  1.68it/s][A
copy images:  85%|████████▌ | 667/781 [06:22<01:06,  1.72it/s][A
copy images:  86%|████████▌ | 668/781 [06:22<01:05,  1.73it/s][A
copy images:  86%|████████▌ | 669/781 [06:23<01:11,  1.56it/s][A
copy images:  86%|████████▌ | 670/781 [06:24<01:14,  1.49it/s][A
copy images:  86%|████████▌ | 671/781 [06:25<01:08,  1.61it/s][A
copy images:  86%|████████▌ | 672/781 [06:25<01:04,  1.70it/s][A
copy images:  86%|████████▌ | 673/781 [06:26<01:01,  1.75it/s][A
copy images:  86%|████████▋ | 674/781 [06:26<00:59,  1.81it/s][A
copy images:  86%|████████▋ | 675/781 [06:27<00:57,  1.83it/s][A
copy images:  87%|████████▋ | 676/781 [06:27<00:56,  1.86it/s][A
copy images:  87%|████████▋ | 677/781 [06:28<00:54,  1.89it/s][A
copy images:  87%|████████▋ | 678/781 [06:28<01:00,  1.69it/s][A
copy images:  87%|████████▋ | 679/781 [06:29<00:58,  1.76it/s][A
copy images:  87%|████████▋ | 680/781 [06:30<00:59,  1.69it/s][A
copy images:  87%|████████▋ | 681/781 [06:30<00:57,  1.75it/s][A
copy images:  87%|████████▋ | 682/781 [06:31<00:54,  1.82it/s][A
copy images:  87%|████████▋ | 683/781 [06:31<01:05,  1.51it/s][A
copy images:  88%|████████▊ | 684/781 [06:32<01:03,  1.53it/s][A
copy images:  88%|████████▊ | 685/781 [06:33<01:23,  1.16it/s][A
copy images:  88%|████████▊ | 686/781 [06:34<01:12,  1.31it/s][A
copy images:  88%|████████▊ | 687/781 [06:35<01:06,  1.42it/s][A
copy images:  88%|████████▊ | 688/781 [06:35<01:01,  1.51it/s][A
copy images:  88%|████████▊ | 689/781 [06:36<01:06,  1.39it/s][A
copy images:  88%|████████▊ | 690/781 [06:37<01:04,  1.41it/s][A
copy images:  88%|████████▊ | 691/781 [06:37<00:58,  1.54it/s][A
copy images:  89%|████████▊ | 692/781 [06:38<00:54,  1.65it/s][A
copy images:  89%|████████▊ | 693/781 [06:39<01:16,  1.15it/s][A
copy images:  89%|████████▉ | 694/781 [06:40<01:06,  1.31it/s][A
copy images:  89%|████████▉ | 695/781 [06:40<00:59,  1.46it/s][A
copy images:  89%|████████▉ | 696/781 [06:42<01:21,  1.05it/s][A
copy images:  89%|████████▉ | 697/781 [06:42<01:09,  1.21it/s][A
copy images:  89%|████████▉ | 698/781 [06:43<01:00,  1.38it/s][A
copy images:  90%|████████▉ | 699/781 [06:43<00:57,  1.43it/s][A
copy images:  90%|████████▉ | 700/781 [06:44<00:51,  1.57it/s][A
copy images:  90%|████████▉ | 701/781 [06:44<00:49,  1.63it/s][A
copy images:  90%|████████▉ | 702/781 [06:45<00:46,  1.71it/s][A
copy images:  90%|█████████ | 703/781 [06:46<00:43,  1.78it/s][A
copy images:  90%|█████████ | 704/781 [06:46<00:49,  1.55it/s][A
copy images:  90%|█████████ | 705/781 [06:47<00:45,  1.66it/s][A
copy images:  90%|█████████ | 706/781 [06:47<00:43,  1.73it/s][A
copy images:  91%|█████████ | 707/781 [06:48<00:41,  1.79it/s][A
copy images:  91%|█████████ | 708/781 [06:48<00:39,  1.85it/s][A
copy images:  91%|█████████ | 709/781 [06:49<00:38,  1.88it/s][A
copy images:  91%|█████████ | 710/781 [06:50<00:42,  1.67it/s][A
copy images:  91%|█████████ | 711/781 [06:50<00:40,  1.73it/s][A
copy images:  91%|█████████ | 712/781 [06:51<00:38,  1.79it/s][A
copy images:  91%|█████████▏| 713/781 [06:51<00:38,  1.75it/s][A
copy images:  91%|█████████▏| 714/781 [06:52<00:37,  1.79it/s][A
copy images:  92%|█████████▏| 715/781 [06:53<00:40,  1.64it/s][A
copy images:  92%|█████████▏| 716/781 [06:53<00:38,  1.70it/s][A
copy images:  92%|█████████▏| 717/781 [06:54<00:36,  1.77it/s][A
copy images:  92%|█████████▏| 718/781 [06:55<00:42,  1.47it/s][A
copy images:  92%|█████████▏| 719/781 [06:55<00:38,  1.59it/s][A
copy images:  92%|█████████▏| 720/781 [06:56<00:36,  1.67it/s][A
copy images:  92%|█████████▏| 721/781 [06:56<00:36,  1.63it/s][A
copy images:  92%|█████████▏| 722/781 [06:57<00:35,  1.69it/s][A
copy images:  93%|█████████▎| 723/781 [06:57<00:36,  1.58it/s][A
copy images:  93%|█████████▎| 724/781 [06:58<00:34,  1.64it/s][A
copy images:  93%|█████████▎| 725/781 [06:59<00:35,  1.60it/s][A
copy images:  93%|█████████▎| 726/781 [06:59<00:33,  1.64it/s][A
copy images:  93%|█████████▎| 727/781 [07:00<00:31,  1.74it/s][A
copy images:  93%|█████████▎| 728/781 [07:00<00:29,  1.79it/s][A
copy images:  93%|█████████▎| 729/781 [07:01<00:28,  1.83it/s][A
copy images:  93%|█████████▎| 730/781 [07:01<00:27,  1.86it/s][A
copy images:  94%|█████████▎| 731/781 [07:02<00:27,  1.85it/s][A
copy images:  94%|█████████▎| 732/781 [07:05<01:10,  1.45s/it][A
copy images:  94%|█████████▍| 733/781 [07:07<01:12,  1.51s/it][A
copy images:  94%|█████████▍| 734/781 [07:08<00:56,  1.20s/it][A
copy images:  94%|█████████▍| 735/781 [07:08<00:45,  1.01it/s][A
copy images:  94%|█████████▍| 736/781 [07:09<00:39,  1.13it/s][A
copy images:  94%|█████████▍| 737/781 [07:09<00:33,  1.29it/s][A
copy images:  94%|█████████▍| 738/781 [07:10<00:30,  1.42it/s][A
copy images:  95%|█████████▍| 739/781 [07:10<00:27,  1.54it/s][A
copy images:  95%|█████████▍| 740/781 [07:11<00:24,  1.65it/s][A
copy images:  95%|█████████▍| 741/781 [07:11<00:23,  1.73it/s][A
copy images:  95%|█████████▌| 742/781 [07:12<00:21,  1.80it/s][A
copy images:  95%|█████████▌| 743/781 [07:12<00:20,  1.86it/s][A
copy images:  95%|█████████▌| 744/781 [07:13<00:19,  1.91it/s][A
copy images:  95%|█████████▌| 745/781 [07:13<00:19,  1.87it/s][A
copy images:  96%|█████████▌| 746/781 [07:14<00:19,  1.83it/s][A
copy images:  96%|█████████▌| 747/781 [07:14<00:18,  1.86it/s][A
copy images:  96%|█████████▌| 748/781 [07:15<00:17,  1.86it/s][A
copy images:  96%|█████████▌| 749/781 [07:16<00:17,  1.87it/s][A
copy images:  96%|█████████▌| 750/781 [07:16<00:16,  1.84it/s][A
copy images:  96%|█████████▌| 751/781 [07:17<00:16,  1.81it/s][A
copy images:  96%|█████████▋| 752/781 [07:17<00:16,  1.80it/s][A
copy images:  96%|█████████▋| 753/781 [07:18<00:15,  1.83it/s][A
copy images:  97%|█████████▋| 754/781 [07:18<00:14,  1.88it/s][A
copy images:  97%|█████████▋| 755/781 [07:19<00:13,  1.90it/s][A
copy images:  97%|█████████▋| 756/781 [07:19<00:13,  1.84it/s][A
copy images:  97%|█████████▋| 757/781 [07:20<00:12,  1.88it/s][A
copy images:  97%|█████████▋| 758/781 [07:20<00:12,  1.84it/s][A
copy images:  97%|█████████▋| 759/781 [07:21<00:11,  1.87it/s][A
copy images:  97%|█████████▋| 760/781 [07:21<00:11,  1.90it/s][A
copy images:  97%|█████████▋| 761/781 [07:22<00:10,  1.92it/s][A
copy images:  98%|█████████▊| 762/781 [07:22<00:09,  1.94it/s][A
copy images:  98%|█████████▊| 763/781 [07:23<00:09,  1.95it/s][A
copy images:  98%|█████████▊| 764/781 [07:23<00:08,  1.96it/s][A
copy images:  98%|█████████▊| 765/781 [07:24<00:08,  1.86it/s][A
copy images:  98%|█████████▊| 766/781 [07:24<00:07,  2.07it/s][A
copy images:  98%|█████████▊| 767/781 [07:25<00:07,  1.83it/s][A
copy images:  98%|█████████▊| 768/781 [07:26<00:08,  1.54it/s][A
copy images:  98%|█████████▊| 769/781 [07:26<00:06,  1.74it/s][A
copy images:  99%|█████████▊| 770/781 [07:27<00:06,  1.69it/s][A
copy images:  99%|█████████▊| 771/781 [07:27<00:05,  1.92it/s][A
copy images:  99%|█████████▉| 772/781 [07:28<00:04,  2.13it/s][A
copy images:  99%|█████████▉| 773/781 [07:28<00:03,  2.29it/s][A
copy images:  99%|█████████▉| 774/781 [07:29<00:03,  2.10it/s][A
copy images:  99%|█████████▉| 775/781 [07:29<00:02,  2.26it/s][A
copy images:  99%|█████████▉| 776/781 [07:30<00:02,  1.72it/s][A
copy images:  99%|█████████▉| 777/781 [07:30<00:02,  1.91it/s][A
copy images: 100%|█████████▉| 778/781 [07:31<00:01,  1.81it/s][A
copy images: 100%|█████████▉| 779/781 [07:33<00:01,  1.10it/s][A
copy images: 100%|█████████▉| 780/781 [07:33<00:00,  1.29it/s][A
copy images: 100%|██████████| 781/781 [07:34<00:00,  1.53it/s][Acopy images: 100%|██████████| 781/781 [07:34<00:00,  1.72it/s]

generating images:   0%|          | 0/781 [00:00<?, ?it/s][A/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
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

generating images:   0%|          | 1/781 [00:08<1:52:14,  8.63s/it][A
generating images:   0%|          | 2/781 [00:16<1:44:16,  8.03s/it][A
generating images:   0%|          | 3/781 [00:23<1:38:37,  7.61s/it][A
generating images:   1%|          | 4/781 [00:30<1:35:22,  7.36s/it][A
generating images:   1%|          | 5/781 [00:37<1:33:15,  7.21s/it][A
generating images:   1%|          | 6/781 [00:44<1:31:52,  7.11s/it][A
generating images:   1%|          | 7/781 [00:51<1:31:23,  7.08s/it][A
generating images:   1%|          | 8/781 [00:58<1:30:44,  7.04s/it][A
generating images:   1%|          | 9/781 [01:05<1:30:55,  7.07s/it][A
generating images:   1%|▏         | 10/781 [01:12<1:30:18,  7.03s/it][A
generating images:   1%|▏         | 11/781 [01:19<1:30:49,  7.08s/it][A
generating images:   2%|▏         | 12/781 [01:27<1:33:23,  7.29s/it][A
generating images:   2%|▏         | 13/781 [01:34<1:35:08,  7.43s/it][A
generating images:   2%|▏         | 14/781 [01:42<1:33:35,  7.32s/it][A
generating images:   2%|▏         | 15/781 [01:49<1:32:41,  7.26s/it][A
generating images:   2%|▏         | 16/781 [01:56<1:32:12,  7.23s/it][A
generating images:   2%|▏         | 17/781 [02:03<1:32:31,  7.27s/it][A
generating images:   2%|▏         | 18/781 [02:10<1:31:56,  7.23s/it][A
generating images:   2%|▏         | 19/781 [02:18<1:32:53,  7.31s/it][A
generating images:   3%|▎         | 20/781 [02:25<1:31:44,  7.23s/it][A
generating images:   3%|▎         | 21/781 [02:32<1:30:46,  7.17s/it][A
generating images:   3%|▎         | 22/781 [02:39<1:29:42,  7.09s/it][A
generating images:   3%|▎         | 23/781 [02:46<1:29:14,  7.06s/it][A
generating images:   3%|▎         | 24/781 [02:53<1:30:55,  7.21s/it][A
generating images:   3%|▎         | 25/781 [03:00<1:29:48,  7.13s/it][A
generating images:   3%|▎         | 26/781 [03:07<1:29:16,  7.09s/it][A
generating images:   3%|▎         | 27/781 [03:15<1:29:55,  7.16s/it][A
generating images:   4%|▎         | 28/781 [03:22<1:29:26,  7.13s/it][A
generating images:   4%|▎         | 29/781 [03:31<1:37:17,  7.76s/it][A
generating images:   4%|▍         | 30/781 [03:38<1:34:23,  7.54s/it][A
generating images:   4%|▍         | 31/781 [03:45<1:33:40,  7.49s/it][A
generating images:   4%|▍         | 32/781 [03:52<1:32:22,  7.40s/it][A
generating images:   4%|▍         | 33/781 [04:00<1:32:02,  7.38s/it][A
generating images:   4%|▍         | 34/781 [04:08<1:36:33,  7.76s/it][A
generating images:   4%|▍         | 35/781 [04:16<1:34:20,  7.59s/it][A
generating images:   5%|▍         | 36/781 [04:23<1:31:51,  7.40s/it][A
generating images:   5%|▍         | 37/781 [04:31<1:36:17,  7.77s/it][A
generating images:   5%|▍         | 38/781 [04:38<1:33:16,  7.53s/it][A
generating images:   5%|▍         | 39/781 [04:45<1:31:00,  7.36s/it][A
generating images:   5%|▌         | 40/781 [04:53<1:33:33,  7.58s/it][A
generating images:   5%|▌         | 41/781 [05:01<1:35:08,  7.71s/it][A
generating images:   5%|▌         | 42/781 [05:09<1:36:10,  7.81s/it][A
generating images:   6%|▌         | 43/781 [05:16<1:33:11,  7.58s/it][A
generating images:   6%|▌         | 44/781 [05:23<1:30:56,  7.40s/it][A
generating images:   6%|▌         | 45/781 [05:31<1:32:57,  7.58s/it][A
generating images:   6%|▌         | 46/781 [05:38<1:30:55,  7.42s/it][A
generating images:   6%|▌         | 47/781 [05:45<1:29:31,  7.32s/it][A
generating images:   6%|▌         | 48/781 [05:53<1:28:38,  7.26s/it][A
generating images:   6%|▋         | 49/781 [06:00<1:28:21,  7.24s/it][A
generating images:   6%|▋         | 50/781 [06:08<1:31:54,  7.54s/it][A
generating images:   7%|▋         | 51/781 [06:15<1:30:17,  7.42s/it][A
generating images:   7%|▋         | 52/781 [06:23<1:32:32,  7.62s/it][A
generating images:   7%|▋         | 53/781 [06:30<1:30:00,  7.42s/it][A
generating images:   7%|▋         | 54/781 [06:38<1:30:37,  7.48s/it][A
generating images:   7%|▋         | 55/781 [06:45<1:28:34,  7.32s/it][A
generating images:   7%|▋         | 56/781 [06:52<1:27:05,  7.21s/it][A
generating images:   7%|▋         | 57/781 [06:59<1:26:02,  7.13s/it][A
generating images:   7%|▋         | 58/781 [07:06<1:25:07,  7.06s/it][A
generating images:   8%|▊         | 59/781 [07:13<1:24:54,  7.06s/it][A
generating images:   8%|▊         | 60/781 [07:20<1:24:47,  7.06s/it][A
generating images:   8%|▊         | 61/781 [07:27<1:24:10,  7.01s/it][A
generating images:   8%|▊         | 62/781 [07:34<1:23:42,  6.99s/it][A
generating images:   8%|▊         | 63/781 [07:40<1:23:21,  6.97s/it][A
generating images:   8%|▊         | 64/781 [07:47<1:23:04,  6.95s/it][A
generating images:   8%|▊         | 65/781 [07:54<1:22:49,  6.94s/it][A
generating images:   8%|▊         | 66/781 [08:01<1:22:42,  6.94s/it][A
generating images:   9%|▊         | 67/781 [08:08<1:23:30,  7.02s/it][A
generating images:   9%|▊         | 68/781 [08:15<1:23:24,  7.02s/it][A
generating images:   9%|▉         | 69/781 [08:23<1:25:01,  7.17s/it][A
generating images:   9%|▉         | 70/781 [08:30<1:24:11,  7.10s/it][A
generating images:   9%|▉         | 71/781 [08:37<1:23:46,  7.08s/it][A
generating images:   9%|▉         | 72/781 [08:45<1:27:33,  7.41s/it][A
generating images:   9%|▉         | 73/781 [08:52<1:25:45,  7.27s/it][A
generating images:   9%|▉         | 74/781 [08:59<1:24:59,  7.21s/it][A
generating images:  10%|▉         | 75/781 [09:07<1:28:42,  7.54s/it][A
generating images:  10%|▉         | 76/781 [09:15<1:27:09,  7.42s/it][A
generating images:  10%|▉         | 77/781 [09:22<1:27:14,  7.44s/it][A
generating images:  10%|▉         | 78/781 [09:30<1:27:54,  7.50s/it][A
generating images:  10%|█         | 79/781 [09:37<1:27:49,  7.51s/it][A
generating images:  10%|█         | 80/781 [09:46<1:31:06,  7.80s/it][A
generating images:  10%|█         | 81/781 [09:53<1:28:00,  7.54s/it][A
generating images:  10%|█         | 82/781 [10:00<1:25:41,  7.36s/it][A
generating images:  11%|█         | 83/781 [10:07<1:24:42,  7.28s/it][A
generating images:  11%|█         | 84/781 [10:16<1:30:16,  7.77s/it][A
generating images:  11%|█         | 85/781 [10:23<1:27:32,  7.55s/it][A
generating images:  11%|█         | 86/781 [10:30<1:25:36,  7.39s/it][A
generating images:  11%|█         | 87/781 [10:37<1:25:56,  7.43s/it][A
generating images:  11%|█▏        | 88/781 [10:44<1:24:14,  7.29s/it][A
generating images:  11%|█▏        | 89/781 [10:51<1:22:52,  7.19s/it][A
generating images:  12%|█▏        | 90/781 [10:58<1:21:50,  7.11s/it][A
generating images:  12%|█▏        | 91/781 [11:06<1:23:15,  7.24s/it][A
generating images:  12%|█▏        | 92/781 [11:13<1:24:04,  7.32s/it][A
generating images:  12%|█▏        | 93/781 [11:20<1:23:08,  7.25s/it][A
generating images:  12%|█▏        | 94/781 [11:27<1:21:57,  7.16s/it][A
generating images:  12%|█▏        | 95/781 [11:34<1:21:25,  7.12s/it][A
generating images:  12%|█▏        | 96/781 [11:42<1:22:51,  7.26s/it][A
generating images:  12%|█▏        | 97/781 [11:49<1:22:23,  7.23s/it][A
generating images:  13%|█▎        | 98/781 [11:56<1:21:15,  7.14s/it][A
generating images:  13%|█▎        | 99/781 [12:03<1:20:25,  7.08s/it][A
generating images:  13%|█▎        | 100/781 [12:10<1:20:03,  7.05s/it][A
generating images:  13%|█▎        | 101/781 [12:17<1:19:39,  7.03s/it][A
generating images:  13%|█▎        | 102/781 [12:24<1:19:09,  6.99s/it][A
generating images:  13%|█▎        | 103/781 [12:30<1:18:47,  6.97s/it][A
generating images:  13%|█▎        | 104/781 [12:37<1:18:32,  6.96s/it][A
generating images:  13%|█▎        | 105/781 [12:44<1:18:26,  6.96s/it][A
generating images:  14%|█▎        | 106/781 [12:51<1:18:14,  6.95s/it][A
generating images:  14%|█▎        | 107/781 [12:58<1:17:59,  6.94s/it][A
generating images:  14%|█▍        | 108/781 [13:05<1:17:50,  6.94s/it][A
generating images:  14%|█▍        | 109/781 [13:12<1:17:41,  6.94s/it][A
generating images:  14%|█▍        | 110/781 [13:19<1:17:32,  6.93s/it][A
generating images:  14%|█▍        | 111/781 [13:26<1:17:25,  6.93s/it][A
generating images:  14%|█▍        | 112/781 [13:33<1:17:16,  6.93s/it][A
generating images:  14%|█▍        | 113/781 [13:40<1:17:09,  6.93s/it][A
generating images:  15%|█▍        | 114/781 [13:47<1:17:02,  6.93s/it][A
generating images:  15%|█▍        | 115/781 [13:54<1:17:46,  7.01s/it][A
generating images:  15%|█▍        | 116/781 [14:02<1:22:40,  7.46s/it][A
generating images:  15%|█▍        | 117/781 [14:09<1:21:03,  7.32s/it][A
generating images:  15%|█▌        | 118/781 [14:17<1:20:21,  7.27s/it][A
generating images:  15%|█▌        | 119/781 [14:24<1:21:29,  7.39s/it][A
generating images:  15%|█▌        | 120/781 [14:32<1:20:56,  7.35s/it][A
generating images:  15%|█▌        | 121/781 [14:39<1:22:26,  7.49s/it][A
generating images:  16%|█▌        | 122/781 [14:46<1:20:23,  7.32s/it][A
generating images:  16%|█▌        | 123/781 [14:54<1:20:00,  7.29s/it][A
generating images:  16%|█▌        | 124/781 [15:01<1:21:36,  7.45s/it][A
generating images:  16%|█▌        | 125/781 [15:08<1:19:56,  7.31s/it][A
generating images:  16%|█▌        | 126/781 [15:15<1:18:53,  7.23s/it][A
generating images:  16%|█▋        | 127/781 [15:23<1:19:20,  7.28s/it][A
generating images:  16%|█▋        | 128/781 [15:30<1:18:18,  7.20s/it][A
generating images:  17%|█▋        | 129/781 [15:37<1:17:19,  7.12s/it][A
generating images:  17%|█▋        | 130/781 [15:44<1:17:01,  7.10s/it][A
generating images:  17%|█▋        | 131/781 [15:51<1:16:33,  7.07s/it][A
generating images:  17%|█▋        | 132/781 [15:58<1:16:06,  7.04s/it][A
generating images:  17%|█▋        | 133/781 [16:05<1:15:42,  7.01s/it][A
generating images:  17%|█▋        | 134/781 [16:12<1:15:18,  6.98s/it][A
generating images:  17%|█▋        | 135/781 [16:19<1:15:06,  6.98s/it][A
generating images:  17%|█▋        | 136/781 [16:26<1:15:13,  7.00s/it][A
generating images:  18%|█▊        | 137/781 [16:34<1:20:27,  7.50s/it][A
generating images:  18%|█▊        | 138/781 [16:41<1:18:59,  7.37s/it][A
generating images:  18%|█▊        | 139/781 [16:48<1:18:17,  7.32s/it][A
generating images:  18%|█▊        | 140/781 [16:56<1:18:18,  7.33s/it][A
generating images:  18%|█▊        | 141/781 [17:03<1:17:39,  7.28s/it][A
generating images:  18%|█▊        | 142/781 [17:10<1:16:55,  7.22s/it][A
generating images:  18%|█▊        | 143/781 [17:17<1:16:01,  7.15s/it][A
generating images:  18%|█▊        | 144/781 [17:24<1:15:11,  7.08s/it][A
generating images:  19%|█▊        | 145/781 [17:31<1:15:16,  7.10s/it][A
generating images:  19%|█▊        | 146/781 [17:39<1:16:08,  7.19s/it][A
generating images:  19%|█▉        | 147/781 [17:46<1:15:20,  7.13s/it][A
generating images:  19%|█▉        | 148/781 [17:53<1:15:18,  7.14s/it][A
generating images:  19%|█▉        | 149/781 [18:00<1:14:46,  7.10s/it][A
generating images:  19%|█▉        | 150/781 [18:07<1:15:51,  7.21s/it][A
generating images:  19%|█▉        | 151/781 [18:14<1:14:58,  7.14s/it][A
generating images:  19%|█▉        | 152/781 [18:21<1:14:33,  7.11s/it][A
generating images:  20%|█▉        | 153/781 [18:29<1:15:42,  7.23s/it][A
generating images:  20%|█▉        | 154/781 [18:36<1:14:53,  7.17s/it][A
generating images:  20%|█▉        | 155/781 [18:43<1:14:06,  7.10s/it][A
generating images:  20%|█▉        | 156/781 [18:50<1:14:15,  7.13s/it][A
generating images:  20%|██        | 157/781 [18:57<1:13:47,  7.10s/it][A
generating images:  20%|██        | 158/781 [19:04<1:14:16,  7.15s/it][A
generating images:  20%|██        | 159/781 [19:12<1:15:11,  7.25s/it][A
generating images:  20%|██        | 160/781 [19:19<1:15:13,  7.27s/it][A
generating images:  21%|██        | 161/781 [19:26<1:14:02,  7.17s/it][A
generating images:  21%|██        | 162/781 [19:33<1:13:49,  7.16s/it][A
generating images:  21%|██        | 163/781 [19:40<1:13:04,  7.09s/it][A
generating images:  21%|██        | 164/781 [19:47<1:12:28,  7.05s/it][A
generating images:  21%|██        | 165/781 [19:54<1:11:59,  7.01s/it][A
generating images:  21%|██▏       | 166/781 [20:01<1:13:42,  7.19s/it][A
generating images:  21%|██▏       | 167/781 [20:08<1:12:45,  7.11s/it][A
generating images:  22%|██▏       | 168/781 [20:15<1:12:03,  7.05s/it][A
generating images:  22%|██▏       | 169/781 [20:22<1:11:34,  7.02s/it][A
generating images:  22%|██▏       | 170/781 [20:29<1:11:27,  7.02s/it][A
generating images:  22%|██▏       | 171/781 [20:36<1:11:04,  6.99s/it][A
generating images:  22%|██▏       | 172/781 [20:43<1:11:50,  7.08s/it][A
generating images:  22%|██▏       | 173/781 [20:50<1:11:22,  7.04s/it][A
generating images:  22%|██▏       | 174/781 [20:58<1:13:33,  7.27s/it][A
generating images:  22%|██▏       | 175/781 [21:05<1:13:19,  7.26s/it][A
generating images:  23%|██▎       | 176/781 [21:13<1:13:43,  7.31s/it][A
generating images:  23%|██▎       | 177/781 [21:20<1:12:52,  7.24s/it][A
generating images:  23%|██▎       | 178/781 [21:28<1:15:05,  7.47s/it][A
generating images:  23%|██▎       | 179/781 [21:35<1:13:24,  7.32s/it][A
generating images:  23%|██▎       | 180/781 [21:42<1:12:19,  7.22s/it][A
generating images:  23%|██▎       | 181/781 [21:49<1:13:01,  7.30s/it][A
generating images:  23%|██▎       | 182/781 [21:56<1:11:54,  7.20s/it][A
generating images:  23%|██▎       | 183/781 [22:04<1:13:07,  7.34s/it][A
generating images:  24%|██▎       | 184/781 [22:11<1:11:56,  7.23s/it][A
generating images:  24%|██▎       | 185/781 [22:18<1:11:02,  7.15s/it][A
generating images:  24%|██▍       | 186/781 [22:25<1:10:15,  7.08s/it][A
generating images:  24%|██▍       | 187/781 [22:32<1:09:46,  7.05s/it][A
generating images:  24%|██▍       | 188/781 [22:39<1:09:18,  7.01s/it][A
generating images:  24%|██▍       | 189/781 [22:46<1:09:18,  7.02s/it][A
generating images:  24%|██▍       | 190/781 [22:53<1:09:10,  7.02s/it][A
generating images:  24%|██▍       | 191/781 [23:00<1:08:46,  6.99s/it][A
generating images:  25%|██▍       | 192/781 [23:07<1:08:28,  6.98s/it][A
generating images:  25%|██▍       | 193/781 [23:14<1:08:14,  6.96s/it][A
generating images:  25%|██▍       | 194/781 [23:21<1:08:01,  6.95s/it][A
generating images:  25%|██▍       | 195/781 [23:28<1:08:07,  6.97s/it][A
generating images:  25%|██▌       | 196/781 [23:36<1:11:40,  7.35s/it][A
generating images:  25%|██▌       | 197/781 [23:43<1:10:35,  7.25s/it][A
generating images:  25%|██▌       | 198/781 [23:50<1:09:57,  7.20s/it][A
generating images:  25%|██▌       | 199/781 [23:57<1:09:19,  7.15s/it][A
generating images:  26%|██▌       | 200/781 [24:04<1:08:34,  7.08s/it][A
generating images:  26%|██▌       | 201/781 [24:11<1:08:05,  7.04s/it][A
generating images:  26%|██▌       | 202/781 [24:18<1:07:37,  7.01s/it][A
generating images:  26%|██▌       | 203/781 [24:25<1:07:16,  6.98s/it][A
generating images:  26%|██▌       | 204/781 [24:32<1:07:03,  6.97s/it][A
generating images:  26%|██▌       | 205/781 [24:39<1:09:22,  7.23s/it][A
generating images:  26%|██▋       | 206/781 [24:47<1:08:41,  7.17s/it][A
generating images:  27%|██▋       | 207/781 [24:54<1:09:24,  7.26s/it][A
generating images:  27%|██▋       | 208/781 [25:02<1:12:15,  7.57s/it][A
generating images:  27%|██▋       | 209/781 [25:09<1:10:18,  7.38s/it][A
generating images:  27%|██▋       | 210/781 [25:17<1:11:57,  7.56s/it][A
generating images:  27%|██▋       | 211/781 [25:24<1:10:05,  7.38s/it][A
generating images:  27%|██▋       | 212/781 [25:31<1:09:17,  7.31s/it][A
generating images:  27%|██▋       | 213/781 [25:38<1:08:05,  7.19s/it][A
generating images:  27%|██▋       | 214/781 [25:46<1:09:26,  7.35s/it][A
generating images:  28%|██▊       | 215/781 [25:53<1:08:06,  7.22s/it][A
generating images:  28%|██▊       | 216/781 [26:00<1:07:08,  7.13s/it][A
generating images:  28%|██▊       | 217/781 [26:08<1:08:43,  7.31s/it][A
generating images:  28%|██▊       | 218/781 [26:16<1:13:00,  7.78s/it][A
generating images:  28%|██▊       | 219/781 [26:24<1:11:24,  7.62s/it][A
generating images:  28%|██▊       | 220/781 [26:31<1:09:35,  7.44s/it][A
generating images:  28%|██▊       | 221/781 [26:38<1:08:54,  7.38s/it][A
generating images:  28%|██▊       | 222/781 [26:45<1:07:40,  7.26s/it][A
generating images:  29%|██▊       | 223/781 [26:52<1:08:18,  7.35s/it][A
generating images:  29%|██▊       | 224/781 [27:01<1:10:27,  7.59s/it][A
generating images:  29%|██▉       | 225/781 [27:08<1:08:46,  7.42s/it][A
generating images:  29%|██▉       | 226/781 [27:15<1:09:18,  7.49s/it][A
generating images:  29%|██▉       | 227/781 [27:22<1:07:50,  7.35s/it][A
generating images:  29%|██▉       | 228/781 [27:29<1:07:13,  7.29s/it][A
generating images:  29%|██▉       | 229/781 [27:38<1:09:21,  7.54s/it][A
generating images:  29%|██▉       | 230/781 [27:45<1:07:49,  7.38s/it][A
generating images:  30%|██▉       | 231/781 [27:52<1:07:27,  7.36s/it][A
generating images:  30%|██▉       | 232/781 [27:59<1:06:07,  7.23s/it][A
generating images:  30%|██▉       | 233/781 [28:06<1:06:48,  7.32s/it][A
generating images:  30%|██▉       | 234/781 [28:13<1:05:36,  7.20s/it][A
generating images:  30%|███       | 235/781 [28:20<1:04:45,  7.12s/it][A
generating images:  30%|███       | 236/781 [28:27<1:04:49,  7.14s/it][A
generating images:  30%|███       | 237/781 [28:35<1:05:44,  7.25s/it][A
generating images:  30%|███       | 238/781 [28:42<1:05:01,  7.18s/it][A
generating images:  31%|███       | 239/781 [28:49<1:05:22,  7.24s/it][A
generating images:  31%|███       | 240/781 [28:57<1:05:18,  7.24s/it][A
generating images:  31%|███       | 241/781 [29:04<1:05:16,  7.25s/it][A
generating images:  31%|███       | 242/781 [29:11<1:04:25,  7.17s/it][A
generating images:  31%|███       | 243/781 [29:18<1:04:00,  7.14s/it][A
generating images:  31%|███       | 244/781 [29:25<1:04:50,  7.24s/it][A
generating images:  31%|███▏      | 245/781 [29:33<1:04:29,  7.22s/it][A
generating images:  31%|███▏      | 246/781 [29:40<1:03:49,  7.16s/it][A
generating images:  32%|███▏      | 247/781 [29:47<1:04:13,  7.22s/it][A
generating images:  32%|███▏      | 248/781 [29:56<1:08:14,  7.68s/it][A
generating images:  32%|███▏      | 249/781 [30:03<1:06:35,  7.51s/it][A
generating images:  32%|███▏      | 250/781 [30:10<1:05:33,  7.41s/it][A
generating images:  32%|███▏      | 251/781 [30:17<1:05:15,  7.39s/it][A
generating images:  32%|███▏      | 252/781 [30:24<1:04:32,  7.32s/it][A
generating images:  32%|███▏      | 253/781 [30:31<1:03:39,  7.23s/it][A
generating images:  33%|███▎      | 254/781 [30:38<1:02:46,  7.15s/it][A
generating images:  33%|███▎      | 255/781 [30:46<1:02:39,  7.15s/it][A
generating images:  33%|███▎      | 256/781 [30:53<1:02:56,  7.19s/it][A
generating images:  33%|███▎      | 257/781 [31:00<1:02:18,  7.14s/it][A
generating images:  33%|███▎      | 258/781 [31:07<1:02:04,  7.12s/it][A
generating images:  33%|███▎      | 259/781 [31:14<1:01:28,  7.07s/it][A
generating images:  33%|███▎      | 260/781 [31:21<1:01:43,  7.11s/it][A
generating images:  33%|███▎      | 261/781 [31:28<1:01:51,  7.14s/it][A
generating images:  34%|███▎      | 262/781 [31:35<1:01:16,  7.08s/it][A
generating images:  34%|███▎      | 263/781 [31:43<1:02:41,  7.26s/it][A
generating images:  34%|███▍      | 264/781 [31:50<1:01:58,  7.19s/it][A
generating images:  34%|███▍      | 265/781 [31:58<1:04:50,  7.54s/it][A
generating images:  34%|███▍      | 266/781 [32:06<1:05:32,  7.64s/it][A
generating images:  34%|███▍      | 267/781 [32:13<1:04:16,  7.50s/it][A
generating images:  34%|███▍      | 268/781 [32:20<1:02:53,  7.36s/it][A
generating images:  34%|███▍      | 269/781 [32:27<1:01:45,  7.24s/it][A
generating images:  35%|███▍      | 270/781 [32:34<1:00:57,  7.16s/it][A
generating images:  35%|███▍      | 271/781 [32:41<1:00:55,  7.17s/it][A
generating images:  35%|███▍      | 272/781 [32:48<1:00:10,  7.09s/it][A
generating images:  35%|███▍      | 273/781 [32:55<59:44,  7.06s/it]  [A
generating images:  35%|███▌      | 274/781 [33:03<1:02:18,  7.37s/it][A
generating images:  35%|███▌      | 275/781 [33:10<1:01:00,  7.24s/it][A
generating images:  35%|███▌      | 276/781 [33:17<1:00:11,  7.15s/it][A
generating images:  35%|███▌      | 277/781 [33:24<59:34,  7.09s/it]  [A
generating images:  36%|███▌      | 278/781 [33:31<59:21,  7.08s/it][A
generating images:  36%|███▌      | 279/781 [33:38<58:55,  7.04s/it][A
generating images:  36%|███▌      | 280/781 [33:45<58:32,  7.01s/it][A
generating images:  36%|███▌      | 281/781 [33:52<58:33,  7.03s/it][A
generating images:  36%|███▌      | 282/781 [34:00<58:59,  7.09s/it][A
generating images:  36%|███▌      | 283/781 [34:07<1:00:00,  7.23s/it][A
generating images:  36%|███▋      | 284/781 [34:15<1:01:31,  7.43s/it][A
generating images:  36%|███▋      | 285/781 [34:22<1:01:12,  7.40s/it][A
generating images:  37%|███▋      | 286/781 [34:30<1:01:00,  7.39s/it][A
generating images:  37%|███▋      | 287/781 [34:37<1:00:14,  7.32s/it][A
generating images:  37%|███▋      | 288/781 [34:44<1:00:48,  7.40s/it][A
generating images:  37%|███▋      | 289/781 [34:52<1:00:18,  7.35s/it][A
generating images:  37%|███▋      | 290/781 [34:59<59:39,  7.29s/it]  [A
generating images:  37%|███▋      | 291/781 [35:06<59:10,  7.24s/it][A
generating images:  37%|███▋      | 292/781 [35:13<58:28,  7.18s/it][A
generating images:  38%|███▊      | 293/781 [35:20<57:50,  7.11s/it][A
generating images:  38%|███▊      | 294/781 [35:27<57:58,  7.14s/it][A
generating images:  38%|███▊      | 295/781 [35:34<57:25,  7.09s/it][A
generating images:  38%|███▊      | 296/781 [35:41<57:10,  7.07s/it][A
generating images:  38%|███▊      | 297/781 [35:48<56:44,  7.03s/it][A
generating images:  38%|███▊      | 298/781 [35:56<57:55,  7.20s/it][A
generating images:  38%|███▊      | 299/781 [36:03<57:31,  7.16s/it][A
generating images:  38%|███▊      | 300/781 [36:11<59:15,  7.39s/it][A
generating images:  39%|███▊      | 301/781 [36:18<59:55,  7.49s/it][A
generating images:  39%|███▊      | 302/781 [36:26<59:38,  7.47s/it][A
generating images:  39%|███▉      | 303/781 [36:34<1:00:58,  7.65s/it][A
generating images:  39%|███▉      | 304/781 [36:41<59:28,  7.48s/it]  [A
generating images:  39%|███▉      | 305/781 [36:48<58:03,  7.32s/it][A
generating images:  39%|███▉      | 306/781 [36:55<57:38,  7.28s/it][A
generating images:  39%|███▉      | 307/781 [37:02<56:47,  7.19s/it][A
generating images:  39%|███▉      | 308/781 [37:09<56:09,  7.12s/it][A
generating images:  40%|███▉      | 309/781 [37:16<55:36,  7.07s/it][A
generating images:  40%|███▉      | 310/781 [37:23<56:02,  7.14s/it][A
generating images:  40%|███▉      | 311/781 [37:30<55:24,  7.07s/it][A
generating images:  40%|███▉      | 312/781 [37:37<55:12,  7.06s/it][A
generating images:  40%|████      | 313/781 [37:44<54:51,  7.03s/it][A
generating images:  40%|████      | 314/781 [37:51<54:54,  7.05s/it][A
generating images:  40%|████      | 315/781 [37:58<54:39,  7.04s/it][A
generating images:  40%|████      | 316/781 [38:06<55:19,  7.14s/it][A
generating images:  41%|████      | 317/781 [38:13<54:53,  7.10s/it][A
generating images:  41%|████      | 318/781 [38:20<54:50,  7.11s/it][A
generating images:  41%|████      | 319/781 [38:27<54:25,  7.07s/it][A
generating images:  41%|████      | 320/781 [38:34<54:06,  7.04s/it][A
generating images:  41%|████      | 321/781 [38:41<55:08,  7.19s/it][A
generating images:  41%|████      | 322/781 [38:48<54:40,  7.15s/it][A
generating images:  41%|████▏     | 323/781 [38:55<54:13,  7.10s/it][A
generating images:  41%|████▏     | 324/781 [39:03<54:10,  7.11s/it][A
generating images:  42%|████▏     | 325/781 [39:09<53:38,  7.06s/it][A
generating images:  42%|████▏     | 326/781 [39:16<53:18,  7.03s/it][A
generating images:  42%|████▏     | 327/781 [39:24<53:41,  7.10s/it][A
generating images:  42%|████▏     | 328/781 [39:31<53:11,  7.05s/it][A
generating images:  42%|████▏     | 329/781 [39:38<52:48,  7.01s/it][A
generating images:  42%|████▏     | 330/781 [39:44<52:29,  6.98s/it][A
generating images:  42%|████▏     | 331/781 [39:51<52:26,  6.99s/it][A
generating images:  43%|████▎     | 332/781 [39:59<54:21,  7.26s/it][A
generating images:  43%|████▎     | 333/781 [40:06<53:28,  7.16s/it][A
generating images:  43%|████▎     | 334/781 [40:13<52:49,  7.09s/it][A
generating images:  43%|████▎     | 335/781 [40:20<52:20,  7.04s/it][A
generating images:  43%|████▎     | 336/781 [40:27<51:57,  7.01s/it][A
generating images:  43%|████▎     | 337/781 [40:34<52:30,  7.09s/it][A
generating images:  43%|████▎     | 338/781 [40:42<52:38,  7.13s/it][A
generating images:  43%|████▎     | 339/781 [40:49<52:49,  7.17s/it][A
generating images:  44%|████▎     | 340/781 [40:56<52:12,  7.10s/it][A
generating images:  44%|████▎     | 341/781 [41:03<51:42,  7.05s/it][A
generating images:  44%|████▍     | 342/781 [41:10<52:16,  7.15s/it][A
generating images:  44%|████▍     | 343/781 [41:17<51:40,  7.08s/it][A
generating images:  44%|████▍     | 344/781 [41:24<51:13,  7.03s/it][A
generating images:  44%|████▍     | 345/781 [41:31<51:27,  7.08s/it][A
generating images:  44%|████▍     | 346/781 [41:38<51:12,  7.06s/it][A
generating images:  44%|████▍     | 347/781 [41:45<51:03,  7.06s/it][A
generating images:  45%|████▍     | 348/781 [41:52<51:00,  7.07s/it][A
generating images:  45%|████▍     | 349/781 [41:59<50:40,  7.04s/it][A
generating images:  45%|████▍     | 350/781 [42:06<50:51,  7.08s/it][A
generating images:  45%|████▍     | 351/781 [42:13<50:25,  7.04s/it][A
generating images:  45%|████▌     | 352/781 [42:21<50:55,  7.12s/it][A
generating images:  45%|████▌     | 353/781 [42:28<50:30,  7.08s/it][A
generating images:  45%|████▌     | 354/781 [42:35<50:42,  7.13s/it][A
generating images:  45%|████▌     | 355/781 [42:42<50:21,  7.09s/it][A
generating images:  46%|████▌     | 356/781 [42:49<50:04,  7.07s/it][A
generating images:  46%|████▌     | 357/781 [42:56<49:53,  7.06s/it][A
generating images:  46%|████▌     | 358/781 [43:03<50:26,  7.16s/it][A
generating images:  46%|████▌     | 359/781 [43:11<50:47,  7.22s/it][A
generating images:  46%|████▌     | 360/781 [43:18<50:02,  7.13s/it][A
generating images:  46%|████▌     | 361/781 [43:25<49:28,  7.07s/it][A
generating images:  46%|████▋     | 362/781 [43:31<49:02,  7.02s/it][A
generating images:  46%|████▋     | 363/781 [43:39<48:56,  7.03s/it][A
generating images:  47%|████▋     | 364/781 [43:45<48:39,  7.00s/it][A
generating images:  47%|████▋     | 365/781 [43:52<48:22,  6.98s/it][A
generating images:  47%|████▋     | 366/781 [43:59<48:08,  6.96s/it][A
generating images:  47%|████▋     | 367/781 [44:06<47:58,  6.95s/it][A
generating images:  47%|████▋     | 368/781 [44:13<47:48,  6.94s/it][A
generating images:  47%|████▋     | 369/781 [44:20<47:38,  6.94s/it][A
generating images:  47%|████▋     | 370/781 [44:27<47:30,  6.93s/it][A
generating images:  48%|████▊     | 371/781 [44:34<47:22,  6.93s/it][A
generating images:  48%|████▊     | 372/781 [44:41<47:14,  6.93s/it][A
generating images:  48%|████▊     | 373/781 [44:48<47:07,  6.93s/it][A
generating images:  48%|████▊     | 374/781 [44:55<47:00,  6.93s/it][A
generating images:  48%|████▊     | 375/781 [45:02<46:52,  6.93s/it][A
generating images:  48%|████▊     | 376/781 [45:09<46:46,  6.93s/it][A
generating images:  48%|████▊     | 377/781 [45:16<46:56,  6.97s/it][A
generating images:  48%|████▊     | 378/781 [45:23<47:16,  7.04s/it][A
generating images:  49%|████▊     | 379/781 [45:30<47:57,  7.16s/it][A
generating images:  49%|████▊     | 380/781 [45:38<48:36,  7.27s/it][A
generating images:  49%|████▉     | 381/781 [45:45<48:00,  7.20s/it][A
generating images:  49%|████▉     | 382/781 [45:52<48:27,  7.29s/it][A
generating images:  49%|████▉     | 383/781 [45:59<47:36,  7.18s/it][A
generating images:  49%|████▉     | 384/781 [46:06<47:20,  7.16s/it][A
generating images:  49%|████▉     | 385/781 [46:13<46:54,  7.11s/it][A
generating images:  49%|████▉     | 386/781 [46:20<46:26,  7.05s/it][A
generating images:  50%|████▉     | 387/781 [46:27<46:03,  7.01s/it][A
generating images:  50%|████▉     | 388/781 [46:34<45:46,  6.99s/it][A
generating images:  50%|████▉     | 389/781 [46:41<45:47,  7.01s/it][A
generating images:  50%|████▉     | 390/781 [46:49<47:50,  7.34s/it][A
generating images:  50%|█████     | 391/781 [46:56<47:07,  7.25s/it][A
generating images:  50%|█████     | 392/781 [47:04<47:01,  7.25s/it][A
generating images:  50%|█████     | 393/781 [47:11<46:15,  7.15s/it][A
generating images:  50%|█████     | 394/781 [47:18<46:06,  7.15s/it][A
generating images:  51%|█████     | 395/781 [47:25<46:09,  7.17s/it][A
generating images:  51%|█████     | 396/781 [47:32<45:53,  7.15s/it][A
generating images:  51%|█████     | 397/781 [47:39<45:35,  7.12s/it][A
generating images:  51%|█████     | 398/781 [47:46<45:29,  7.13s/it][A
generating images:  51%|█████     | 399/781 [47:54<45:40,  7.18s/it][A
generating images:  51%|█████     | 400/781 [48:01<45:20,  7.14s/it][A
generating images:  51%|█████▏    | 401/781 [48:07<44:49,  7.08s/it][A
generating images:  51%|█████▏    | 402/781 [48:15<44:35,  7.06s/it][A
generating images:  52%|█████▏    | 403/781 [48:21<44:12,  7.02s/it][A
generating images:  52%|█████▏    | 404/781 [48:28<43:56,  6.99s/it][A
generating images:  52%|█████▏    | 405/781 [48:35<43:42,  6.97s/it][A
generating images:  52%|█████▏    | 406/781 [48:43<44:52,  7.18s/it][A
generating images:  52%|█████▏    | 407/781 [48:50<44:16,  7.10s/it][A
generating images:  52%|█████▏    | 408/781 [48:57<43:49,  7.05s/it][A
generating images:  52%|█████▏    | 409/781 [49:04<43:28,  7.01s/it][A
generating images:  52%|█████▏    | 410/781 [49:11<43:17,  7.00s/it][A
generating images:  53%|█████▎    | 411/781 [49:18<43:02,  6.98s/it][A
generating images:  53%|█████▎    | 412/781 [49:25<42:59,  6.99s/it][A
generating images:  53%|█████▎    | 413/781 [49:32<42:55,  7.00s/it][A
generating images:  53%|█████▎    | 414/781 [49:39<42:52,  7.01s/it][A
generating images:  53%|█████▎    | 415/781 [49:46<42:47,  7.02s/it][A
generating images:  53%|█████▎    | 416/781 [49:53<42:40,  7.01s/it][A
generating images:  53%|█████▎    | 417/781 [50:00<42:24,  6.99s/it][A
generating images:  54%|█████▎    | 418/781 [50:07<42:16,  6.99s/it][A
generating images:  54%|█████▎    | 419/781 [50:14<42:55,  7.11s/it][A
generating images:  54%|█████▍    | 420/781 [50:21<43:05,  7.16s/it][A
generating images:  54%|█████▍    | 421/781 [50:29<43:27,  7.24s/it][A
generating images:  54%|█████▍    | 422/781 [50:36<43:37,  7.29s/it][A
generating images:  54%|█████▍    | 423/781 [50:44<43:35,  7.30s/it][A
generating images:  54%|█████▍    | 424/781 [50:51<42:54,  7.21s/it][A
generating images:  54%|█████▍    | 425/781 [50:58<42:32,  7.17s/it][A
generating images:  55%|█████▍    | 426/781 [51:05<42:20,  7.16s/it][A
generating images:  55%|█████▍    | 427/781 [51:12<41:54,  7.10s/it][A
generating images:  55%|█████▍    | 428/781 [51:19<41:33,  7.06s/it][A
generating images:  55%|█████▍    | 429/781 [51:26<41:12,  7.02s/it][A
generating images:  55%|█████▌    | 430/781 [51:33<40:54,  6.99s/it][A
generating images:  55%|█████▌    | 431/781 [51:39<40:45,  6.99s/it][A
generating images:  55%|█████▌    | 432/781 [51:47<41:11,  7.08s/it][A
generating images:  55%|█████▌    | 433/781 [51:54<40:52,  7.05s/it][A
generating images:  56%|█████▌    | 434/781 [52:01<40:32,  7.01s/it][A
generating images:  56%|█████▌    | 435/781 [52:08<41:18,  7.16s/it][A
generating images:  56%|█████▌    | 436/781 [52:15<40:48,  7.10s/it][A
generating images:  56%|█████▌    | 437/781 [52:22<40:24,  7.05s/it][A
generating images:  56%|█████▌    | 438/781 [52:29<40:34,  7.10s/it][A
generating images:  56%|█████▌    | 439/781 [52:36<40:09,  7.05s/it][A
generating images:  56%|█████▋    | 440/781 [52:43<39:48,  7.01s/it][A
generating images:  56%|█████▋    | 441/781 [52:50<39:33,  6.98s/it][A
generating images:  57%|█████▋    | 442/781 [52:57<39:20,  6.96s/it][A
generating images:  57%|█████▋    | 443/781 [53:04<39:10,  6.95s/it][A
generating images:  57%|█████▋    | 444/781 [53:11<39:08,  6.97s/it][A
generating images:  57%|█████▋    | 445/781 [53:18<39:02,  6.97s/it][A
generating images:  57%|█████▋    | 446/781 [53:25<39:05,  7.00s/it][A
generating images:  57%|█████▋    | 447/781 [53:32<38:50,  6.98s/it][A
generating images:  57%|█████▋    | 448/781 [53:39<38:48,  6.99s/it][A
generating images:  57%|█████▋    | 449/781 [53:46<38:34,  6.97s/it][A
generating images:  58%|█████▊    | 450/781 [53:53<38:23,  6.96s/it][A
generating images:  58%|█████▊    | 451/781 [54:00<38:13,  6.95s/it][A
generating images:  58%|█████▊    | 452/781 [54:07<38:05,  6.95s/it][A
generating images:  58%|█████▊    | 453/781 [54:15<40:04,  7.33s/it][A
generating images:  58%|█████▊    | 454/781 [54:22<39:16,  7.21s/it][A
generating images:  58%|█████▊    | 455/781 [54:29<38:40,  7.12s/it][A
generating images:  58%|█████▊    | 456/781 [54:36<38:13,  7.06s/it][A
generating images:  59%|█████▊    | 457/781 [54:43<37:53,  7.02s/it][A
generating images:  59%|█████▊    | 458/781 [54:49<37:40,  7.00s/it][A
generating images:  59%|█████▉    | 459/781 [54:56<37:30,  6.99s/it][A
generating images:  59%|█████▉    | 460/781 [55:03<37:29,  7.01s/it][A
generating images:  59%|█████▉    | 461/781 [55:11<37:31,  7.04s/it][A
generating images:  59%|█████▉    | 462/781 [55:18<37:56,  7.14s/it][A
generating images:  59%|█████▉    | 463/781 [55:26<39:13,  7.40s/it][A
generating images:  59%|█████▉    | 464/781 [55:33<38:19,  7.25s/it][A
generating images:  60%|█████▉    | 465/781 [55:40<37:41,  7.16s/it][A
generating images:  60%|█████▉    | 466/781 [55:47<37:11,  7.09s/it][A
generating images:  60%|█████▉    | 467/781 [55:54<37:03,  7.08s/it][A
generating images:  60%|█████▉    | 468/781 [56:01<36:42,  7.04s/it][A
generating images:  60%|██████    | 469/781 [56:08<37:00,  7.12s/it][A
generating images:  60%|██████    | 470/781 [56:16<37:39,  7.27s/it][A
generating images:  60%|██████    | 471/781 [56:23<38:10,  7.39s/it][A
generating images:  60%|██████    | 472/781 [56:31<37:43,  7.33s/it][A
generating images:  61%|██████    | 473/781 [56:38<37:14,  7.25s/it][A
generating images:  61%|██████    | 474/781 [56:45<36:40,  7.17s/it][A
generating images:  61%|██████    | 475/781 [56:51<36:10,  7.09s/it][A
generating images:  61%|██████    | 476/781 [56:58<35:48,  7.04s/it][A
generating images:  61%|██████    | 477/781 [57:06<35:52,  7.08s/it][A
generating images:  61%|██████    | 478/781 [57:14<37:08,  7.36s/it][A
generating images:  61%|██████▏   | 479/781 [57:20<36:21,  7.22s/it][A
generating images:  61%|██████▏   | 480/781 [57:27<35:46,  7.13s/it][A
generating images:  62%|██████▏   | 481/781 [57:34<35:20,  7.07s/it][A
generating images:  62%|██████▏   | 482/781 [57:41<35:00,  7.03s/it][A
generating images:  62%|██████▏   | 483/781 [57:48<34:44,  6.99s/it][A
generating images:  62%|██████▏   | 484/781 [57:55<34:34,  6.98s/it][A
generating images:  62%|██████▏   | 485/781 [58:02<34:22,  6.97s/it][A
generating images:  62%|██████▏   | 486/781 [58:09<34:12,  6.96s/it][A
generating images:  62%|██████▏   | 487/781 [58:16<34:03,  6.95s/it][A
generating images:  62%|██████▏   | 488/781 [58:23<33:54,  6.94s/it][A
generating images:  63%|██████▎   | 489/781 [58:30<33:46,  6.94s/it][A
generating images:  63%|██████▎   | 490/781 [58:37<33:38,  6.94s/it][A
generating images:  63%|██████▎   | 491/781 [58:44<33:30,  6.93s/it][A
generating images:  63%|██████▎   | 492/781 [58:51<33:24,  6.94s/it][A
generating images:  63%|██████▎   | 493/781 [58:58<33:18,  6.94s/it][A
generating images:  63%|██████▎   | 494/781 [59:04<33:11,  6.94s/it][A
generating images:  63%|██████▎   | 495/781 [59:11<33:03,  6.94s/it][A
generating images:  64%|██████▎   | 496/781 [59:18<32:57,  6.94s/it][A
generating images:  64%|██████▎   | 497/781 [59:25<32:49,  6.94s/it][A
generating images:  64%|██████▍   | 498/781 [59:32<32:41,  6.93s/it][A
generating images:  64%|██████▍   | 499/781 [59:39<32:34,  6.93s/it][A
generating images:  64%|██████▍   | 500/781 [59:46<32:27,  6.93s/it][A
generating images:  64%|██████▍   | 501/781 [59:53<32:21,  6.93s/it][A
generating images:  64%|██████▍   | 502/781 [1:00:00<32:13,  6.93s/it][A
generating images:  64%|██████▍   | 503/781 [1:00:07<32:06,  6.93s/it][A
generating images:  65%|██████▍   | 504/781 [1:00:14<32:03,  6.94s/it][A
generating images:  65%|██████▍   | 505/781 [1:00:21<31:55,  6.94s/it][A
generating images:  65%|██████▍   | 506/781 [1:00:28<31:49,  6.94s/it][A
generating images:  65%|██████▍   | 507/781 [1:00:35<31:47,  6.96s/it][A
generating images:  65%|██████▌   | 508/781 [1:00:42<31:52,  7.01s/it][A
generating images:  65%|██████▌   | 509/781 [1:00:49<32:16,  7.12s/it][A
generating images:  65%|██████▌   | 510/781 [1:00:57<32:48,  7.26s/it][A
generating images:  65%|██████▌   | 511/781 [1:01:04<32:21,  7.19s/it][A
generating images:  66%|██████▌   | 512/781 [1:01:11<32:01,  7.14s/it][A
generating images:  66%|██████▌   | 513/781 [1:01:18<31:54,  7.14s/it][A
generating images:  66%|██████▌   | 514/781 [1:01:25<31:30,  7.08s/it][A
generating images:  66%|██████▌   | 515/781 [1:01:32<31:52,  7.19s/it][A
generating images:  66%|██████▌   | 516/781 [1:01:39<31:25,  7.11s/it][A
generating images:  66%|██████▌   | 517/781 [1:01:46<31:03,  7.06s/it][A
generating images:  66%|██████▋   | 518/781 [1:01:53<30:52,  7.05s/it][A
generating images:  66%|██████▋   | 519/781 [1:02:00<30:54,  7.08s/it][A
generating images:  67%|██████▋   | 520/781 [1:02:07<30:41,  7.06s/it][A
generating images:  67%|██████▋   | 521/781 [1:02:14<30:31,  7.04s/it][A
generating images:  67%|██████▋   | 522/781 [1:02:21<30:20,  7.03s/it][A
generating images:  67%|██████▋   | 523/781 [1:02:28<30:14,  7.03s/it][A
generating images:  67%|██████▋   | 524/781 [1:02:35<30:06,  7.03s/it][A
generating images:  67%|██████▋   | 525/781 [1:02:43<30:12,  7.08s/it][A
generating images:  67%|██████▋   | 526/781 [1:02:50<29:57,  7.05s/it][A
generating images:  67%|██████▋   | 527/781 [1:02:57<29:45,  7.03s/it][A
generating images:  68%|██████▊   | 528/781 [1:03:04<29:30,  7.00s/it][A
generating images:  68%|██████▊   | 529/781 [1:03:11<29:56,  7.13s/it][A
generating images:  68%|██████▊   | 530/781 [1:03:18<29:42,  7.10s/it][A
generating images:  68%|██████▊   | 531/781 [1:03:25<29:31,  7.09s/it][A
generating images:  68%|██████▊   | 532/781 [1:03:32<29:13,  7.04s/it][A
generating images:  68%|██████▊   | 533/781 [1:03:39<29:34,  7.16s/it][A
generating images:  68%|██████▊   | 534/781 [1:03:46<29:12,  7.10s/it][A
generating images:  69%|██████▊   | 535/781 [1:03:53<28:53,  7.05s/it][A
generating images:  69%|██████▊   | 536/781 [1:04:00<28:37,  7.01s/it][A
generating images:  69%|██████▉   | 537/781 [1:04:07<28:26,  7.00s/it][A
generating images:  69%|██████▉   | 538/781 [1:04:14<28:15,  6.98s/it][A
generating images:  69%|██████▉   | 539/781 [1:04:21<28:12,  6.99s/it][A
generating images:  69%|██████▉   | 540/781 [1:04:28<28:03,  6.99s/it][A
generating images:  69%|██████▉   | 541/781 [1:04:35<27:55,  6.98s/it][A
generating images:  69%|██████▉   | 542/781 [1:04:43<28:25,  7.13s/it][A
generating images:  70%|██████▉   | 543/781 [1:04:50<28:14,  7.12s/it][A
generating images:  70%|██████▉   | 544/781 [1:04:57<27:56,  7.07s/it][A
generating images:  70%|██████▉   | 545/781 [1:05:04<27:44,  7.05s/it][A
generating images:  70%|██████▉   | 546/781 [1:05:11<27:35,  7.05s/it][A
generating images:  70%|███████   | 547/781 [1:05:18<27:27,  7.04s/it][A
generating images:  70%|███████   | 548/781 [1:05:25<27:13,  7.01s/it][A
generating images:  70%|███████   | 549/781 [1:05:32<27:02,  6.99s/it][A
generating images:  70%|███████   | 550/781 [1:05:39<26:50,  6.97s/it][A
generating images:  71%|███████   | 551/781 [1:05:46<26:43,  6.97s/it][A
generating images:  71%|███████   | 552/781 [1:05:53<26:36,  6.97s/it][A
generating images:  71%|███████   | 553/781 [1:05:59<26:26,  6.96s/it][A
generating images:  71%|███████   | 554/781 [1:06:06<26:17,  6.95s/it][A
generating images:  71%|███████   | 555/781 [1:06:13<26:09,  6.95s/it][A
generating images:  71%|███████   | 556/781 [1:06:20<26:01,  6.94s/it][A
generating images:  71%|███████▏  | 557/781 [1:06:27<25:56,  6.95s/it][A
generating images:  71%|███████▏  | 558/781 [1:06:34<25:51,  6.96s/it][A
generating images:  72%|███████▏  | 559/781 [1:06:41<25:49,  6.98s/it][A
generating images:  72%|███████▏  | 560/781 [1:06:48<25:48,  7.01s/it][A
generating images:  72%|███████▏  | 561/781 [1:06:55<25:40,  7.00s/it][A
generating images:  72%|███████▏  | 562/781 [1:07:02<25:29,  6.98s/it][A
generating images:  72%|███████▏  | 563/781 [1:07:09<25:29,  7.01s/it][A
generating images:  72%|███████▏  | 564/781 [1:07:16<25:22,  7.01s/it][A
generating images:  72%|███████▏  | 565/781 [1:07:23<25:18,  7.03s/it][A
generating images:  72%|███████▏  | 566/781 [1:07:30<25:04,  7.00s/it][A
generating images:  73%|███████▎  | 567/781 [1:07:37<24:56,  6.99s/it][A
generating images:  73%|███████▎  | 568/781 [1:07:44<24:58,  7.04s/it][A
generating images:  73%|███████▎  | 569/781 [1:07:52<25:06,  7.11s/it][A
generating images:  73%|███████▎  | 570/781 [1:07:59<24:59,  7.11s/it][A
generating images:  73%|███████▎  | 571/781 [1:08:06<24:43,  7.06s/it][A
generating images:  73%|███████▎  | 572/781 [1:08:13<24:27,  7.02s/it][A
generating images:  73%|███████▎  | 573/781 [1:08:20<24:14,  6.99s/it][A
generating images:  73%|███████▎  | 574/781 [1:08:27<24:17,  7.04s/it][A
generating images:  74%|███████▎  | 575/781 [1:08:34<24:11,  7.05s/it][A
generating images:  74%|███████▍  | 576/781 [1:08:41<24:03,  7.04s/it][A
generating images:  74%|███████▍  | 577/781 [1:08:48<23:49,  7.01s/it][A
generating images:  74%|███████▍  | 578/781 [1:08:55<23:43,  7.01s/it][A
generating images:  74%|███████▍  | 579/781 [1:09:02<23:33,  7.00s/it][A
generating images:  74%|███████▍  | 580/781 [1:09:09<23:23,  6.98s/it][A
generating images:  74%|███████▍  | 581/781 [1:09:16<23:50,  7.15s/it][A
generating images:  75%|███████▍  | 582/781 [1:09:23<23:31,  7.10s/it][A
generating images:  75%|███████▍  | 583/781 [1:09:30<23:18,  7.07s/it][A
generating images:  75%|███████▍  | 584/781 [1:09:37<23:03,  7.02s/it][A
generating images:  75%|███████▍  | 585/781 [1:09:44<22:52,  7.00s/it][A
generating images:  75%|███████▌  | 586/781 [1:09:51<23:04,  7.10s/it][A
generating images:  75%|███████▌  | 587/781 [1:09:58<22:46,  7.04s/it][A
generating images:  75%|███████▌  | 588/781 [1:10:05<22:33,  7.01s/it][A
generating images:  75%|███████▌  | 589/781 [1:10:12<22:33,  7.05s/it][A
generating images:  76%|███████▌  | 590/781 [1:10:19<22:23,  7.03s/it][A
generating images:  76%|███████▌  | 591/781 [1:10:26<22:10,  7.00s/it][A
generating images:  76%|███████▌  | 592/781 [1:10:34<22:22,  7.10s/it][A
generating images:  76%|███████▌  | 593/781 [1:10:41<22:10,  7.08s/it][A
generating images:  76%|███████▌  | 594/781 [1:10:48<22:19,  7.16s/it][A
generating images:  76%|███████▌  | 595/781 [1:10:56<22:45,  7.34s/it][A
generating images:  76%|███████▋  | 596/781 [1:11:03<22:28,  7.29s/it][A
generating images:  76%|███████▋  | 597/781 [1:11:10<22:00,  7.18s/it][A
generating images:  77%|███████▋  | 598/781 [1:11:17<21:41,  7.11s/it][A
generating images:  77%|███████▋  | 599/781 [1:11:24<21:23,  7.05s/it][A
generating images:  77%|███████▋  | 600/781 [1:11:31<21:10,  7.02s/it][A
generating images:  77%|███████▋  | 601/781 [1:11:38<20:57,  6.99s/it][A
generating images:  77%|███████▋  | 602/781 [1:11:45<20:47,  6.97s/it][A
generating images:  77%|███████▋  | 603/781 [1:11:52<20:42,  6.98s/it][A
generating images:  77%|███████▋  | 604/781 [1:11:58<20:33,  6.97s/it][A
generating images:  77%|███████▋  | 605/781 [1:12:05<20:24,  6.96s/it][A
generating images:  78%|███████▊  | 606/781 [1:12:12<20:16,  6.95s/it][A
generating images:  78%|███████▊  | 607/781 [1:12:19<20:08,  6.94s/it][A
generating images:  78%|███████▊  | 608/781 [1:12:26<20:00,  6.94s/it][A
generating images:  78%|███████▊  | 609/781 [1:12:33<19:52,  6.94s/it][A
generating images:  78%|███████▊  | 610/781 [1:12:40<19:48,  6.95s/it][A
generating images:  78%|███████▊  | 611/781 [1:12:47<19:41,  6.95s/it][A
generating images:  78%|███████▊  | 612/781 [1:12:54<19:45,  7.02s/it][A
generating images:  78%|███████▊  | 613/781 [1:13:01<19:34,  6.99s/it][A
generating images:  79%|███████▊  | 614/781 [1:13:08<19:23,  6.97s/it][A
generating images:  79%|███████▊  | 615/781 [1:13:15<19:14,  6.96s/it][A
generating images:  79%|███████▉  | 616/781 [1:13:22<19:06,  6.95s/it][A
generating images:  79%|███████▉  | 617/781 [1:13:29<18:58,  6.94s/it][A
generating images:  79%|███████▉  | 618/781 [1:13:36<18:50,  6.93s/it][A
generating images:  79%|███████▉  | 619/781 [1:13:43<18:43,  6.93s/it][A
generating images:  79%|███████▉  | 620/781 [1:13:50<18:42,  6.97s/it][A
generating images:  80%|███████▉  | 621/781 [1:13:57<18:33,  6.96s/it][A
generating images:  80%|███████▉  | 622/781 [1:14:04<18:25,  6.95s/it][A
generating images:  80%|███████▉  | 623/781 [1:14:11<18:17,  6.95s/it][A
generating images:  80%|███████▉  | 624/781 [1:14:17<18:09,  6.94s/it][A
generating images:  80%|████████  | 625/781 [1:14:24<18:02,  6.94s/it][A
generating images:  80%|████████  | 626/781 [1:14:31<17:55,  6.94s/it][A
generating images:  80%|████████  | 627/781 [1:14:38<17:47,  6.93s/it][A
generating images:  80%|████████  | 628/781 [1:14:45<17:41,  6.94s/it][A
generating images:  81%|████████  | 629/781 [1:14:52<17:37,  6.96s/it][A
generating images:  81%|████████  | 630/781 [1:14:59<17:42,  7.04s/it][A
generating images:  81%|████████  | 631/781 [1:15:06<17:32,  7.01s/it][A
generating images:  81%|████████  | 632/781 [1:15:14<17:37,  7.10s/it][A
generating images:  81%|████████  | 633/781 [1:15:21<17:30,  7.10s/it][A
generating images:  81%|████████  | 634/781 [1:15:28<17:22,  7.09s/it][A
generating images:  81%|████████▏ | 635/781 [1:15:35<17:11,  7.07s/it][A
generating images:  81%|████████▏ | 636/781 [1:15:42<17:09,  7.10s/it][A
generating images:  82%|████████▏ | 637/781 [1:15:49<17:10,  7.16s/it][A
generating images:  82%|████████▏ | 638/781 [1:15:56<16:57,  7.11s/it][A
generating images:  82%|████████▏ | 639/781 [1:16:03<16:42,  7.06s/it][A
generating images:  82%|████████▏ | 640/781 [1:16:10<16:32,  7.04s/it][A
generating images:  82%|████████▏ | 641/781 [1:16:17<16:23,  7.02s/it][A
generating images:  82%|████████▏ | 642/781 [1:16:24<16:12,  6.99s/it][A
generating images:  82%|████████▏ | 643/781 [1:16:31<16:07,  7.01s/it][A
generating images:  82%|████████▏ | 644/781 [1:16:38<15:57,  6.99s/it][A
generating images:  83%|████████▎ | 645/781 [1:16:45<15:52,  7.00s/it][A
generating images:  83%|████████▎ | 646/781 [1:16:52<15:44,  7.00s/it][A
generating images:  83%|████████▎ | 647/781 [1:16:59<15:35,  6.98s/it][A
generating images:  83%|████████▎ | 648/781 [1:17:06<15:28,  6.98s/it][A
generating images:  83%|████████▎ | 649/781 [1:17:13<15:19,  6.97s/it][A
generating images:  83%|████████▎ | 650/781 [1:17:20<15:11,  6.95s/it][A
generating images:  83%|████████▎ | 651/781 [1:17:27<15:09,  6.99s/it][A
generating images:  83%|████████▎ | 652/781 [1:17:34<15:00,  6.98s/it][A
generating images:  84%|████████▎ | 653/781 [1:17:41<14:51,  6.96s/it][A
generating images:  84%|████████▎ | 654/781 [1:17:48<14:42,  6.95s/it][A
generating images:  84%|████████▍ | 655/781 [1:17:55<14:37,  6.96s/it][A
generating images:  84%|████████▍ | 656/781 [1:18:02<14:32,  6.98s/it][A
generating images:  84%|████████▍ | 657/781 [1:18:09<14:25,  6.98s/it][A
generating images:  84%|████████▍ | 658/781 [1:18:16<14:24,  7.03s/it][A
generating images:  84%|████████▍ | 659/781 [1:18:23<14:19,  7.04s/it][A
generating images:  85%|████████▍ | 660/781 [1:18:30<14:24,  7.15s/it][A
generating images:  85%|████████▍ | 661/781 [1:18:37<14:12,  7.10s/it][A
generating images:  85%|████████▍ | 662/781 [1:18:44<13:58,  7.05s/it][A
generating images:  85%|████████▍ | 663/781 [1:18:51<13:48,  7.02s/it][A
generating images:  85%|████████▌ | 664/781 [1:18:58<13:41,  7.02s/it][A
generating images:  85%|████████▌ | 665/781 [1:19:05<13:31,  6.99s/it][A
generating images:  85%|████████▌ | 666/781 [1:19:12<13:26,  7.01s/it][A
generating images:  85%|████████▌ | 667/781 [1:19:19<13:16,  6.98s/it][A
generating images:  86%|████████▌ | 668/781 [1:19:26<13:07,  6.97s/it][A
generating images:  86%|████████▌ | 669/781 [1:19:33<13:01,  6.97s/it][A
generating images:  86%|████████▌ | 670/781 [1:19:40<12:56,  6.99s/it][A
generating images:  86%|████████▌ | 671/781 [1:19:47<12:48,  6.99s/it][A
generating images:  86%|████████▌ | 672/781 [1:19:54<12:39,  6.97s/it][A
generating images:  86%|████████▌ | 673/781 [1:20:01<12:33,  6.98s/it][A
generating images:  86%|████████▋ | 674/781 [1:20:08<12:24,  6.96s/it][A
generating images:  86%|████████▋ | 675/781 [1:20:15<12:19,  6.98s/it][A
generating images:  87%|████████▋ | 676/781 [1:20:22<12:15,  7.00s/it][A
generating images:  87%|████████▋ | 677/781 [1:20:30<12:22,  7.14s/it][A
generating images:  87%|████████▋ | 678/781 [1:20:37<12:09,  7.09s/it][A
generating images:  87%|████████▋ | 679/781 [1:20:44<11:58,  7.04s/it][A
generating images:  87%|████████▋ | 680/781 [1:20:51<11:55,  7.08s/it][A
generating images:  87%|████████▋ | 681/781 [1:20:58<11:44,  7.04s/it][A
generating images:  87%|████████▋ | 682/781 [1:21:05<11:33,  7.01s/it][A
generating images:  87%|████████▋ | 683/781 [1:21:12<11:24,  6.99s/it][A
generating images:  88%|████████▊ | 684/781 [1:21:18<11:16,  6.97s/it][A
generating images:  88%|████████▊ | 685/781 [1:21:25<11:08,  6.96s/it][A
generating images:  88%|████████▊ | 686/781 [1:21:32<11:01,  6.96s/it][A
generating images:  88%|████████▊ | 687/781 [1:21:39<10:53,  6.95s/it][A
generating images:  88%|████████▊ | 688/781 [1:21:46<10:45,  6.94s/it][A
generating images:  88%|████████▊ | 689/781 [1:21:53<10:38,  6.94s/it][A
generating images:  88%|████████▊ | 690/781 [1:22:00<10:31,  6.93s/it][A
generating images:  88%|████████▊ | 691/781 [1:22:07<10:23,  6.93s/it][A
generating images:  89%|████████▊ | 692/781 [1:22:14<10:17,  6.93s/it][A
generating images:  89%|████████▊ | 693/781 [1:22:21<10:10,  6.93s/it][A
generating images:  89%|████████▉ | 694/781 [1:22:28<10:02,  6.93s/it][A
generating images:  89%|████████▉ | 695/781 [1:22:35<09:55,  6.93s/it][A
generating images:  89%|████████▉ | 696/781 [1:22:42<09:49,  6.94s/it][A
generating images:  89%|████████▉ | 697/781 [1:22:49<09:48,  7.01s/it][A
generating images:  89%|████████▉ | 698/781 [1:22:57<09:59,  7.23s/it][A
generating images:  90%|████████▉ | 699/781 [1:23:03<09:45,  7.14s/it][A
generating images:  90%|████████▉ | 700/781 [1:23:10<09:33,  7.08s/it][A
generating images:  90%|████████▉ | 701/781 [1:23:17<09:22,  7.03s/it][A
generating images:  90%|████████▉ | 702/781 [1:23:24<09:13,  7.00s/it][A
generating images:  90%|█████████ | 703/781 [1:23:31<09:04,  6.98s/it][A
generating images:  90%|█████████ | 704/781 [1:23:38<08:56,  6.96s/it][A
generating images:  90%|█████████ | 705/781 [1:23:45<08:52,  7.01s/it][A
generating images:  90%|█████████ | 706/781 [1:23:52<08:46,  7.02s/it][A
generating images:  91%|█████████ | 707/781 [1:24:00<08:50,  7.16s/it][A
generating images:  91%|█████████ | 708/781 [1:24:07<08:44,  7.18s/it][A
generating images:  91%|█████████ | 709/781 [1:24:14<08:33,  7.14s/it][A
generating images:  91%|█████████ | 710/781 [1:24:21<08:29,  7.18s/it][A
generating images:  91%|█████████ | 711/781 [1:24:28<08:18,  7.12s/it][A
generating images:  91%|█████████ | 712/781 [1:24:35<08:08,  7.07s/it][A
generating images:  91%|█████████▏| 713/781 [1:24:42<07:58,  7.04s/it][A
generating images:  91%|█████████▏| 714/781 [1:24:50<08:02,  7.20s/it][A
generating images:  92%|█████████▏| 715/781 [1:24:57<07:50,  7.12s/it][A
generating images:  92%|█████████▏| 716/781 [1:25:04<07:42,  7.12s/it][A
generating images:  92%|█████████▏| 717/781 [1:25:11<07:33,  7.08s/it][A
generating images:  92%|█████████▏| 718/781 [1:25:18<07:25,  7.07s/it][A
generating images:  92%|█████████▏| 719/781 [1:25:25<07:18,  7.07s/it][A
generating images:  92%|█████████▏| 720/781 [1:25:32<07:08,  7.03s/it][A
generating images:  92%|█████████▏| 721/781 [1:25:39<07:00,  7.01s/it][A
generating images:  92%|█████████▏| 722/781 [1:25:46<06:52,  6.98s/it][A
generating images:  93%|█████████▎| 723/781 [1:25:53<06:45,  6.99s/it][A
generating images:  93%|█████████▎| 724/781 [1:26:00<06:40,  7.03s/it][A
generating images:  93%|█████████▎| 725/781 [1:26:07<06:31,  7.00s/it][A
generating images:  93%|█████████▎| 726/781 [1:26:14<06:23,  6.98s/it][A
generating images:  93%|█████████▎| 727/781 [1:26:21<06:15,  6.96s/it][A
generating images:  93%|█████████▎| 728/781 [1:26:28<06:08,  6.96s/it][A
generating images:  93%|█████████▎| 729/781 [1:26:35<06:03,  6.98s/it][A
generating images:  93%|█████████▎| 730/781 [1:26:42<05:56,  6.99s/it][A
generating images:  94%|█████████▎| 731/781 [1:26:49<05:49,  7.00s/it][A
generating images:  94%|█████████▎| 732/781 [1:26:56<05:47,  7.08s/it][A
generating images:  94%|█████████▍| 733/781 [1:27:03<05:39,  7.08s/it][A
generating images:  94%|█████████▍| 734/781 [1:27:10<05:30,  7.03s/it][A
generating images:  94%|█████████▍| 735/781 [1:27:17<05:22,  7.01s/it][A
generating images:  94%|█████████▍| 736/781 [1:27:24<05:14,  6.99s/it][A
generating images:  94%|█████████▍| 737/781 [1:27:31<05:06,  6.97s/it][A
generating images:  94%|█████████▍| 738/781 [1:27:38<04:59,  6.96s/it][A
generating images:  95%|█████████▍| 739/781 [1:27:45<04:51,  6.95s/it][A
generating images:  95%|█████████▍| 740/781 [1:27:52<04:46,  6.98s/it][A
generating images:  95%|█████████▍| 741/781 [1:27:59<04:38,  6.96s/it][A
generating images:  95%|█████████▌| 742/781 [1:28:06<04:31,  6.95s/it][A
generating images:  95%|█████████▌| 743/781 [1:28:12<04:23,  6.95s/it][A
generating images:  95%|█████████▌| 744/781 [1:28:19<04:16,  6.94s/it][A
generating images:  95%|█████████▌| 745/781 [1:28:26<04:09,  6.93s/it][A
generating images:  96%|█████████▌| 746/781 [1:28:33<04:03,  6.95s/it][A
generating images:  96%|█████████▌| 747/781 [1:28:40<03:56,  6.95s/it][A
generating images:  96%|█████████▌| 748/781 [1:28:48<03:52,  7.05s/it][A
generating images:  96%|█████████▌| 749/781 [1:28:56<03:56,  7.38s/it][A
generating images:  96%|█████████▌| 750/781 [1:29:03<03:45,  7.28s/it][A
generating images:  96%|█████████▌| 751/781 [1:29:10<03:35,  7.18s/it][A
generating images:  96%|█████████▋| 752/781 [1:29:17<03:26,  7.14s/it][A
generating images:  96%|█████████▋| 753/781 [1:29:24<03:21,  7.21s/it][A
generating images:  97%|█████████▋| 754/781 [1:29:31<03:15,  7.22s/it][A
generating images:  97%|█████████▋| 755/781 [1:29:38<03:05,  7.13s/it][A
generating images:  97%|█████████▋| 756/781 [1:29:46<02:59,  7.20s/it][A
generating images:  97%|█████████▋| 757/781 [1:29:53<02:50,  7.11s/it][A
generating images:  97%|█████████▋| 758/781 [1:29:59<02:42,  7.06s/it][A
generating images:  97%|█████████▋| 759/781 [1:30:08<02:42,  7.38s/it][A
generating images:  97%|█████████▋| 760/781 [1:30:15<02:32,  7.27s/it][A
generating images:  97%|█████████▋| 761/781 [1:30:22<02:24,  7.23s/it][A
generating images:  98%|█████████▊| 762/781 [1:30:29<02:16,  7.17s/it][A
generating images:  98%|█████████▊| 763/781 [1:30:36<02:07,  7.10s/it][A
generating images:  98%|█████████▊| 764/781 [1:30:43<02:03,  7.26s/it][A
generating images:  98%|█████████▊| 765/781 [1:30:50<01:55,  7.20s/it][A
generating images:  98%|█████████▊| 766/781 [1:30:58<01:48,  7.26s/it][A
generating images:  98%|█████████▊| 767/781 [1:31:05<01:40,  7.16s/it][A
generating images:  98%|█████████▊| 768/781 [1:31:12<01:33,  7.17s/it][A
generating images:  98%|█████████▊| 769/781 [1:31:19<01:25,  7.10s/it][A
generating images:  99%|█████████▊| 770/781 [1:31:26<01:17,  7.05s/it][A
generating images:  99%|█████████▊| 771/781 [1:31:33<01:10,  7.02s/it][A
generating images:  99%|█████████▉| 772/781 [1:31:40<01:03,  7.05s/it][A
generating images:  99%|█████████▉| 773/781 [1:31:47<00:57,  7.15s/it][A
generating images:  99%|█████████▉| 774/781 [1:31:54<00:49,  7.08s/it][A
generating images:  99%|█████████▉| 775/781 [1:32:01<00:42,  7.04s/it][A
generating images:  99%|█████████▉| 776/781 [1:32:08<00:35,  7.01s/it][A
generating images:  99%|█████████▉| 777/781 [1:32:15<00:27,  6.98s/it][A
generating images: 100%|█████████▉| 778/781 [1:32:22<00:20,  6.97s/it][A
generating images: 100%|█████████▉| 779/781 [1:32:29<00:13,  6.96s/it][A
generating images: 100%|█████████▉| 780/781 [1:32:36<00:06,  6.95s/it][A
generating images: 100%|██████████| 781/781 [1:32:43<00:00,  6.96s/it][Agenerating images: 100%|██████████| 781/781 [1:32:43<00:00,  7.12s/it]
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torch/utils/data/dataloader.py:617: UserWarning: This DataLoader will create 64 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(

  0%|          | 0/781 [00:00<?, ?it/s][A/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
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

  0%|          | 1/781 [00:06<1:26:19,  6.64s/it][A
  0%|          | 3/781 [00:06<23:10,  1.79s/it]  [A
  1%|          | 5/781 [00:06<11:41,  1.11it/s][A
  1%|          | 7/781 [00:07<07:05,  1.82it/s][A
  1%|          | 9/781 [00:07<04:45,  2.70it/s][A
  1%|▏         | 11/781 [00:07<03:23,  3.79it/s][A
  2%|▏         | 13/781 [00:07<02:32,  5.03it/s][A
  2%|▏         | 15/781 [00:07<01:59,  6.40it/s][A
  2%|▏         | 17/781 [00:07<01:38,  7.77it/s][A
  2%|▏         | 19/781 [00:07<01:23,  9.11it/s][A
  3%|▎         | 21/781 [00:08<01:14, 10.27it/s][A
  3%|▎         | 23/781 [00:08<01:06, 11.34it/s][A
  3%|▎         | 25/781 [00:08<01:01, 12.23it/s][A
  3%|▎         | 27/781 [00:08<00:58, 12.88it/s][A
  4%|▎         | 29/781 [00:08<00:56, 13.41it/s][A
  4%|▍         | 31/781 [00:08<00:54, 13.68it/s][A
  4%|▍         | 33/781 [00:08<00:53, 14.00it/s][A
  4%|▍         | 35/781 [00:09<00:52, 14.18it/s][A
  5%|▍         | 37/781 [00:09<00:54, 13.76it/s][A
  5%|▍         | 39/781 [00:09<00:52, 14.05it/s][A
  5%|▌         | 41/781 [00:09<00:52, 14.18it/s][A
  6%|▌         | 43/781 [00:09<00:53, 13.78it/s][A
  6%|▌         | 45/781 [00:09<00:51, 14.16it/s][A
  6%|▌         | 47/781 [00:09<00:53, 13.83it/s][A
  6%|▋         | 49/781 [00:10<00:51, 14.12it/s][A
  7%|▋         | 51/781 [00:10<00:52, 13.97it/s][A
  7%|▋         | 53/781 [00:10<00:51, 14.20it/s][A
  7%|▋         | 55/781 [00:10<00:52, 13.86it/s][A
  7%|▋         | 57/781 [00:10<00:53, 13.47it/s][A
  8%|▊         | 59/781 [00:10<00:54, 13.16it/s][A
  8%|▊         | 61/781 [00:10<00:52, 13.61it/s][A
  8%|▊         | 63/781 [00:11<00:51, 13.89it/s][A
  8%|▊         | 65/781 [00:11<00:51, 13.77it/s][A
  9%|▊         | 67/781 [00:11<00:50, 14.08it/s][A
  9%|▉         | 69/781 [00:11<00:49, 14.31it/s][A
  9%|▉         | 71/781 [00:11<00:49, 14.46it/s][A
  9%|▉         | 73/781 [00:11<00:48, 14.58it/s][A
 10%|▉         | 75/781 [00:11<00:48, 14.69it/s][A
 10%|▉         | 77/781 [00:11<00:47, 14.74it/s][A
 10%|█         | 79/781 [00:12<00:47, 14.79it/s][A
 10%|█         | 81/781 [00:12<00:47, 14.81it/s][A
 11%|█         | 83/781 [00:12<00:47, 14.85it/s][A
 11%|█         | 85/781 [00:12<00:46, 14.87it/s][A
 11%|█         | 87/781 [00:12<00:46, 14.89it/s][A
 11%|█▏        | 89/781 [00:12<00:46, 14.89it/s][A
 12%|█▏        | 91/781 [00:12<00:46, 14.91it/s][A
 12%|█▏        | 93/781 [00:13<00:46, 14.89it/s][A
 12%|█▏        | 95/781 [00:13<00:46, 14.89it/s][A
 12%|█▏        | 97/781 [00:13<00:47, 14.33it/s][A
 13%|█▎        | 99/781 [00:13<00:48, 14.03it/s][A
 13%|█▎        | 101/781 [00:13<00:48, 13.90it/s][A
 13%|█▎        | 103/781 [00:13<00:48, 14.04it/s][A
 13%|█▎        | 105/781 [00:13<00:48, 13.93it/s][A
 14%|█▎        | 107/781 [00:14<00:48, 13.82it/s][A
 14%|█▍        | 109/781 [00:14<00:47, 14.14it/s][A
 14%|█▍        | 111/781 [00:14<00:46, 14.35it/s][A
 14%|█▍        | 113/781 [00:14<00:47, 14.11it/s][A
 15%|█▍        | 115/781 [00:14<00:47, 13.92it/s][A
 15%|█▍        | 117/781 [00:14<00:46, 14.26it/s][A
 15%|█▌        | 119/781 [00:14<00:46, 14.36it/s][A
 15%|█▌        | 121/781 [00:15<00:45, 14.51it/s][A
 16%|█▌        | 123/781 [00:15<00:47, 13.87it/s][A
 16%|█▌        | 125/781 [00:15<00:46, 14.07it/s][A
 16%|█▋        | 127/781 [00:15<00:47, 13.86it/s][A
 17%|█▋        | 129/781 [00:15<00:46, 14.11it/s][A
 17%|█▋        | 131/781 [00:15<00:47, 13.80it/s][A
 17%|█▋        | 133/781 [00:15<00:47, 13.54it/s][A
 17%|█▋        | 135/781 [00:16<00:46, 13.83it/s][A
 18%|█▊        | 137/781 [00:16<00:47, 13.70it/s][A
 18%|█▊        | 139/781 [00:16<00:46, 13.87it/s][A
 18%|█▊        | 141/781 [00:16<00:45, 14.08it/s][A
 18%|█▊        | 143/781 [00:16<00:44, 14.25it/s][A
 19%|█▊        | 145/781 [00:16<00:44, 14.38it/s][A
 19%|█▉        | 147/781 [00:16<00:43, 14.48it/s][A
 19%|█▉        | 149/781 [00:17<00:43, 14.61it/s][A
 19%|█▉        | 151/781 [00:17<00:43, 14.65it/s][A
 20%|█▉        | 153/781 [00:17<00:42, 14.68it/s][A
 20%|█▉        | 155/781 [00:17<00:42, 14.70it/s][A
 20%|██        | 157/781 [00:17<00:42, 14.71it/s][A
 20%|██        | 159/781 [00:17<00:42, 14.71it/s][A
 21%|██        | 161/781 [00:17<00:43, 14.33it/s][A
 21%|██        | 163/781 [00:18<00:44, 14.01it/s][A
 21%|██        | 165/781 [00:18<00:43, 14.20it/s][A
 21%|██▏       | 167/781 [00:18<00:42, 14.33it/s][A
 22%|██▏       | 169/781 [00:18<00:42, 14.45it/s][A
 22%|██▏       | 171/781 [00:18<00:43, 14.13it/s][A
 22%|██▏       | 173/781 [00:18<00:42, 14.33it/s][A
 22%|██▏       | 175/781 [00:18<00:43, 14.05it/s][A
 23%|██▎       | 177/781 [00:19<00:43, 13.86it/s][A
 23%|██▎       | 179/781 [00:19<00:44, 13.66it/s][A
 23%|██▎       | 181/781 [00:19<00:42, 13.96it/s][A
 23%|██▎       | 183/781 [00:19<00:42, 14.19it/s][A
 24%|██▎       | 185/781 [00:19<00:42, 13.98it/s][A
 24%|██▍       | 187/781 [00:19<00:42, 13.92it/s][A
 24%|██▍       | 189/781 [00:19<00:41, 14.14it/s][A
 24%|██▍       | 191/781 [00:19<00:41, 14.30it/s][A
 25%|██▍       | 193/781 [00:20<00:40, 14.41it/s][A
 25%|██▍       | 195/781 [00:20<00:41, 14.07it/s][A
 25%|██▌       | 197/781 [00:20<00:42, 13.78it/s][A
 25%|██▌       | 199/781 [00:20<00:41, 14.06it/s][A
 26%|██▌       | 201/781 [00:20<00:40, 14.22it/s][A
 26%|██▌       | 203/781 [00:20<00:41, 13.93it/s][A
 26%|██▌       | 205/781 [00:20<00:40, 14.17it/s][A
 27%|██▋       | 207/781 [00:21<00:40, 14.35it/s][A
 27%|██▋       | 209/781 [00:21<00:39, 14.39it/s][A
 27%|██▋       | 211/781 [00:21<00:39, 14.54it/s][A
 27%|██▋       | 213/781 [00:21<00:38, 14.62it/s][A
 28%|██▊       | 215/781 [00:21<00:38, 14.65it/s][A
 28%|██▊       | 217/781 [00:21<00:40, 14.08it/s][A
 28%|██▊       | 219/781 [00:21<00:39, 14.32it/s][A
 28%|██▊       | 221/781 [00:22<00:38, 14.44it/s][A
 29%|██▊       | 223/781 [00:22<00:38, 14.52it/s][A
 29%|██▉       | 225/781 [00:22<00:38, 14.58it/s][A
 29%|██▉       | 227/781 [00:22<00:39, 14.16it/s][A
 29%|██▉       | 229/781 [00:22<00:38, 14.34it/s][A
 30%|██▉       | 231/781 [00:22<00:38, 14.44it/s][A
 30%|██▉       | 233/781 [00:22<00:39, 14.02it/s][A
 30%|███       | 235/781 [00:23<00:38, 14.22it/s][A
 30%|███       | 237/781 [00:23<00:37, 14.38it/s][A
 31%|███       | 239/781 [00:23<00:39, 13.72it/s][A
 31%|███       | 241/781 [00:23<00:39, 13.56it/s][A
 31%|███       | 243/781 [00:23<00:40, 13.39it/s][A
 31%|███▏      | 245/781 [00:23<00:38, 13.79it/s][A
 32%|███▏      | 247/781 [00:23<00:37, 14.06it/s][A
 32%|███▏      | 249/781 [00:24<00:38, 13.82it/s][A
 32%|███▏      | 251/781 [00:24<00:37, 14.09it/s][A
 32%|███▏      | 253/781 [00:24<00:36, 14.27it/s][A
 33%|███▎      | 255/781 [00:24<00:37, 13.87it/s][A
 33%|███▎      | 257/781 [00:24<00:37, 14.14it/s][A
 33%|███▎      | 259/781 [00:24<00:36, 14.16it/s][A
 33%|███▎      | 261/781 [00:24<00:36, 14.33it/s][A
 34%|███▎      | 263/781 [00:25<00:35, 14.39it/s][A
 34%|███▍      | 265/781 [00:25<00:35, 14.51it/s][A
 34%|███▍      | 267/781 [00:25<00:35, 14.45it/s][A
 34%|███▍      | 269/781 [00:25<00:35, 14.52it/s][A
 35%|███▍      | 271/781 [00:25<00:35, 14.55it/s][A
 35%|███▍      | 273/781 [00:25<00:35, 14.23it/s][A
 35%|███▌      | 275/781 [00:25<00:35, 14.37it/s][A
 35%|███▌      | 277/781 [00:26<00:35, 14.39it/s][A
 36%|███▌      | 279/781 [00:26<00:34, 14.49it/s][A
 36%|███▌      | 281/781 [00:26<00:35, 14.18it/s][A
 36%|███▌      | 283/781 [00:26<00:34, 14.33it/s][A
 36%|███▋      | 285/781 [00:26<00:34, 14.50it/s][A
 37%|███▋      | 287/781 [00:26<00:34, 14.53it/s][A
 37%|███▋      | 289/781 [00:26<00:33, 14.55it/s][A
 37%|███▋      | 291/781 [00:27<00:33, 14.59it/s][A
 38%|███▊      | 293/781 [00:27<00:33, 14.62it/s][A
 38%|███▊      | 295/781 [00:27<00:33, 14.63it/s][A
 38%|███▊      | 297/781 [00:27<00:32, 14.68it/s][A
 38%|███▊      | 299/781 [00:27<00:32, 14.73it/s][A
 39%|███▊      | 301/781 [00:27<00:33, 14.54it/s][A
 39%|███▉      | 303/781 [00:27<00:33, 14.22it/s][A
 39%|███▉      | 305/781 [00:27<00:34, 13.89it/s][A
 39%|███▉      | 307/781 [00:28<00:34, 13.72it/s][A
 40%|███▉      | 309/781 [00:28<00:33, 14.01it/s][A
 40%|███▉      | 311/781 [00:28<00:33, 14.23it/s][A
 40%|████      | 313/781 [00:28<00:33, 13.91it/s][A
 40%|████      | 315/781 [00:28<00:33, 13.71it/s][A
 41%|████      | 317/781 [00:28<00:33, 14.03it/s][A
 41%|████      | 319/781 [00:28<00:32, 14.22it/s][A
 41%|████      | 321/781 [00:29<00:32, 13.98it/s][A
 41%|████▏     | 323/781 [00:29<00:32, 14.19it/s][A
 42%|████▏     | 325/781 [00:29<00:32, 13.95it/s][A
 42%|████▏     | 327/781 [00:29<00:32, 14.18it/s][A
 42%|████▏     | 329/781 [00:29<00:31, 14.34it/s][A
 42%|████▏     | 331/781 [00:29<00:31, 14.43it/s][A
 43%|████▎     | 333/781 [00:29<00:30, 14.49it/s][A
 43%|████▎     | 335/781 [00:30<00:30, 14.54it/s][A
 43%|████▎     | 337/781 [00:30<00:31, 14.13it/s][A
 43%|████▎     | 339/781 [00:30<00:30, 14.36it/s][A
 44%|████▎     | 341/781 [00:30<00:30, 14.44it/s][A
 44%|████▍     | 343/781 [00:30<00:30, 14.51it/s][A
 44%|████▍     | 345/781 [00:30<00:30, 14.17it/s][A
 44%|████▍     | 347/781 [00:30<00:30, 14.33it/s][A
 45%|████▍     | 349/781 [00:31<00:29, 14.43it/s][A
 45%|████▍     | 351/781 [00:31<00:29, 14.56it/s][A
 45%|████▌     | 353/781 [00:31<00:30, 14.16it/s][A
 45%|████▌     | 355/781 [00:31<00:29, 14.29it/s][A
 46%|████▌     | 357/781 [00:31<00:30, 13.96it/s][A
 46%|████▌     | 359/781 [00:31<00:29, 14.18it/s][A
 46%|████▌     | 361/781 [00:31<00:29, 14.36it/s][A
 46%|████▋     | 363/781 [00:32<00:29, 14.26it/s][A
 47%|████▋     | 365/781 [00:32<00:28, 14.39it/s][A
 47%|████▋     | 367/781 [00:32<00:28, 14.52it/s][A
 47%|████▋     | 369/781 [00:32<00:29, 14.13it/s][A
 48%|████▊     | 371/781 [00:32<00:29, 13.80it/s][A
 48%|████▊     | 373/781 [00:32<00:28, 14.08it/s][A
 48%|████▊     | 375/781 [00:32<00:28, 14.00it/s][A
 48%|████▊     | 377/781 [00:33<00:29, 13.64it/s][A
 49%|████▊     | 379/781 [00:33<00:28, 13.98it/s][A
 49%|████▉     | 381/781 [00:33<00:28, 14.18it/s][A
 49%|████▉     | 383/781 [00:33<00:28, 13.97it/s][A
 49%|████▉     | 385/781 [00:33<00:27, 14.19it/s][A
 50%|████▉     | 387/781 [00:33<00:27, 14.30it/s][A
 50%|████▉     | 389/781 [00:33<00:28, 13.95it/s][A
 50%|█████     | 391/781 [00:34<00:27, 14.21it/s][A
 50%|█████     | 393/781 [00:34<00:27, 13.95it/s][A
 51%|█████     | 395/781 [00:34<00:27, 14.19it/s][A
 51%|█████     | 397/781 [00:34<00:27, 13.98it/s][A
 51%|█████     | 399/781 [00:34<00:26, 14.16it/s][A
 51%|█████▏    | 401/781 [00:34<00:26, 14.31it/s][A
 52%|█████▏    | 403/781 [00:34<00:26, 14.49it/s][A
 52%|█████▏    | 405/781 [00:35<00:25, 14.54it/s][A
 52%|█████▏    | 407/781 [00:35<00:26, 14.15it/s][A
 52%|█████▏    | 409/781 [00:35<00:26, 13.87it/s][A
 53%|█████▎    | 411/781 [00:35<00:26, 13.73it/s][A
 53%|█████▎    | 413/781 [00:35<00:26, 13.69it/s][A
 53%|█████▎    | 415/781 [00:35<00:26, 13.95it/s][A
 53%|█████▎    | 417/781 [00:35<00:26, 13.74it/s][A
 54%|█████▎    | 419/781 [00:36<00:26, 13.68it/s][A
 54%|█████▍    | 421/781 [00:36<00:26, 13.60it/s][A
 54%|█████▍    | 423/781 [00:36<00:25, 13.90it/s][A
 54%|█████▍    | 425/781 [00:36<00:25, 14.10it/s][A
 55%|█████▍    | 427/781 [00:36<00:24, 14.27it/s][A
 55%|█████▍    | 429/781 [00:36<00:24, 14.42it/s][A
 55%|█████▌    | 431/781 [00:36<00:24, 14.53it/s][A
 55%|█████▌    | 433/781 [00:37<00:24, 14.16it/s][A
 56%|█████▌    | 435/781 [00:37<00:24, 13.86it/s][A
 56%|█████▌    | 437/781 [00:37<00:24, 13.87it/s][A
 56%|█████▌    | 439/781 [00:37<00:24, 14.13it/s][A
 56%|█████▋    | 441/781 [00:37<00:24, 13.82it/s][A
 57%|█████▋    | 443/781 [00:37<00:24, 13.68it/s][A
 57%|█████▋    | 445/781 [00:37<00:24, 13.77it/s][A
 57%|█████▋    | 447/781 [00:38<00:23, 14.02it/s][A
 57%|█████▋    | 449/781 [00:38<00:23, 14.23it/s][A
 58%|█████▊    | 451/781 [00:38<00:22, 14.35it/s][A
 58%|█████▊    | 453/781 [00:38<00:23, 13.95it/s][A
 58%|█████▊    | 455/781 [00:38<00:23, 14.06it/s][A
 59%|█████▊    | 457/781 [00:38<00:23, 13.86it/s][A
 59%|█████▉    | 459/781 [00:38<00:22, 14.09it/s][A
 59%|█████▉    | 461/781 [00:39<00:22, 14.28it/s][A
 59%|█████▉    | 463/781 [00:39<00:22, 13.93it/s][A
 60%|█████▉    | 465/781 [00:39<00:23, 13.60it/s][A
 60%|█████▉    | 467/781 [00:39<00:23, 13.54it/s][A
 60%|██████    | 469/781 [00:39<00:22, 13.88it/s][A
 60%|██████    | 471/781 [00:39<00:22, 13.69it/s][A
 61%|██████    | 473/781 [00:39<00:22, 13.62it/s][A
 61%|██████    | 475/781 [00:40<00:21, 13.93it/s][A
 61%|██████    | 477/781 [00:40<00:21, 14.02it/s][A
 61%|██████▏   | 479/781 [00:40<00:21, 14.18it/s][A
 62%|██████▏   | 481/781 [00:40<00:21, 13.90it/s][A
 62%|██████▏   | 483/781 [00:40<00:21, 13.79it/s][A
 62%|██████▏   | 485/781 [00:40<00:21, 13.50it/s][A
 62%|██████▏   | 487/781 [00:40<00:21, 13.58it/s][A
 63%|██████▎   | 489/781 [00:41<00:21, 13.89it/s][A
 63%|██████▎   | 491/781 [00:41<00:20, 14.11it/s][A
 63%|██████▎   | 493/781 [00:41<00:20, 13.95it/s][A
 63%|██████▎   | 495/781 [00:41<00:20, 13.93it/s][A
 64%|██████▎   | 497/781 [00:41<00:20, 14.19it/s][A
 64%|██████▍   | 499/781 [00:41<00:20, 13.97it/s][A
 64%|██████▍   | 501/781 [00:41<00:20, 13.81it/s][A
 64%|██████▍   | 503/781 [00:42<00:20, 13.63it/s][A
 65%|██████▍   | 505/781 [00:42<00:20, 13.59it/s][A
 65%|██████▍   | 507/781 [00:42<00:19, 13.92it/s][A
 65%|██████▌   | 509/781 [00:42<00:20, 13.49it/s][A
 65%|██████▌   | 511/781 [00:42<00:19, 13.86it/s][A
 66%|██████▌   | 513/781 [00:42<00:19, 14.09it/s][A
 66%|██████▌   | 515/781 [00:42<00:19, 13.82it/s][A
 66%|██████▌   | 517/781 [00:43<00:19, 13.55it/s][A
 66%|██████▋   | 519/781 [00:43<00:18, 13.91it/s][A
 67%|██████▋   | 521/781 [00:43<00:19, 13.44it/s][A
 67%|██████▋   | 523/781 [00:43<00:19, 13.31it/s][A
 67%|██████▋   | 525/781 [00:43<00:18, 13.66it/s][A
 67%|██████▋   | 527/781 [00:43<00:18, 13.94it/s][A
 68%|██████▊   | 529/781 [00:43<00:18, 13.88it/s][A
 68%|██████▊   | 531/781 [00:44<00:17, 14.07it/s][A
 68%|██████▊   | 533/781 [00:44<00:17, 14.21it/s][A
 69%|██████▊   | 535/781 [00:44<00:17, 13.98it/s][A
 69%|██████▉   | 537/781 [00:44<00:17, 14.17it/s][A
 69%|██████▉   | 539/781 [00:44<00:16, 14.33it/s][A
 69%|██████▉   | 541/781 [00:44<00:16, 14.13it/s][A
 70%|██████▉   | 543/781 [00:44<00:16, 14.26it/s][A
 70%|██████▉   | 545/781 [00:45<00:16, 14.41it/s][A
 70%|███████   | 547/781 [00:45<00:16, 13.93it/s][A
 70%|███████   | 549/781 [00:45<00:16, 13.72it/s][A
 71%|███████   | 551/781 [00:45<00:16, 14.00it/s][A
 71%|███████   | 553/781 [00:45<00:16, 14.22it/s][A
 71%|███████   | 555/781 [00:45<00:15, 14.35it/s][A
 71%|███████▏  | 557/781 [00:45<00:16, 13.72it/s][A
 72%|███████▏  | 559/781 [00:46<00:16, 13.86it/s][A
 72%|███████▏  | 561/781 [00:46<00:15, 14.08it/s][A
 72%|███████▏  | 563/781 [00:46<00:15, 14.27it/s][A
 72%|███████▏  | 565/781 [00:46<00:14, 14.43it/s][A
 73%|███████▎  | 567/781 [00:46<00:14, 14.52it/s][A
 73%|███████▎  | 569/781 [00:46<00:14, 14.58it/s][A
 73%|███████▎  | 571/781 [00:46<00:14, 14.65it/s][A
 73%|███████▎  | 573/781 [00:47<00:14, 14.18it/s][A
 74%|███████▎  | 575/781 [00:47<00:14, 13.95it/s][A
 74%|███████▍  | 577/781 [00:47<00:14, 14.18it/s][A
 74%|███████▍  | 579/781 [00:47<00:14, 14.34it/s][A
 74%|███████▍  | 581/781 [00:47<00:14, 13.97it/s][A
 75%|███████▍  | 583/781 [00:47<00:13, 14.20it/s][A
 75%|███████▍  | 585/781 [00:47<00:14, 13.79it/s][A
 75%|███████▌  | 587/781 [00:48<00:14, 13.63it/s][A
 75%|███████▌  | 589/781 [00:48<00:13, 13.92it/s][A
 76%|███████▌  | 591/781 [00:48<00:13, 14.09it/s][A
 76%|███████▌  | 593/781 [00:48<00:13, 14.25it/s][A
 76%|███████▌  | 595/781 [00:48<00:12, 14.37it/s][A
 76%|███████▋  | 597/781 [00:48<00:12, 14.44it/s][A
 77%|███████▋  | 599/781 [00:48<00:12, 14.26it/s][A
 77%|███████▋  | 601/781 [00:49<00:12, 14.37it/s][A
 77%|███████▋  | 603/781 [00:49<00:12, 14.47it/s][A
 77%|███████▋  | 605/781 [00:49<00:12, 14.04it/s][A
 78%|███████▊  | 607/781 [00:49<00:12, 14.23it/s][A
 78%|███████▊  | 609/781 [00:49<00:12, 14.03it/s][A
 78%|███████▊  | 611/781 [00:49<00:12, 13.82it/s][A
 78%|███████▊  | 613/781 [00:49<00:12, 13.71it/s][A
 79%|███████▊  | 615/781 [00:50<00:11, 14.00it/s][A
 79%|███████▉  | 617/781 [00:50<00:11, 14.21it/s][A
 79%|███████▉  | 619/781 [00:50<00:11, 13.98it/s][A
 80%|███████▉  | 621/781 [00:50<00:11, 13.49it/s][A
 80%|███████▉  | 623/781 [00:50<00:11, 13.82it/s][A
 80%|████████  | 625/781 [00:50<00:11, 14.06it/s][A
 80%|████████  | 627/781 [00:50<00:10, 14.24it/s][A
 81%|████████  | 629/781 [00:51<00:10, 14.34it/s][A
 81%|████████  | 631/781 [00:51<00:10, 14.48it/s][A
 81%|████████  | 633/781 [00:51<00:10, 14.57it/s][A
 81%|████████▏ | 635/781 [00:51<00:10, 14.25it/s][A
 82%|████████▏ | 637/781 [00:51<00:10, 14.39it/s][A
 82%|████████▏ | 639/781 [00:51<00:10, 13.81it/s][A
 82%|████████▏ | 641/781 [00:51<00:09, 14.08it/s][A
 82%|████████▏ | 643/781 [00:52<00:10, 13.78it/s][A
 83%|████████▎ | 645/781 [00:52<00:09, 14.07it/s][A
 83%|████████▎ | 647/781 [00:52<00:09, 14.21it/s][A
 83%|████████▎ | 649/781 [00:52<00:09, 14.38it/s][A
 83%|████████▎ | 651/781 [00:52<00:09, 13.73it/s][A
 84%|████████▎ | 653/781 [00:52<00:09, 13.86it/s][A
 84%|████████▍ | 655/781 [00:52<00:08, 14.20it/s][A
 84%|████████▍ | 657/781 [00:53<00:08, 14.37it/s][A
 84%|████████▍ | 659/781 [00:53<00:08, 14.57it/s][A
 85%|████████▍ | 661/781 [00:53<00:08, 14.71it/s][A
 85%|████████▍ | 663/781 [00:53<00:08, 14.74it/s][A
 85%|████████▌ | 665/781 [00:53<00:07, 14.86it/s][A
 85%|████████▌ | 667/781 [00:53<00:07, 14.92it/s][A
 86%|████████▌ | 669/781 [00:53<00:07, 15.00it/s][A
 86%|████████▌ | 671/781 [00:53<00:07, 14.99it/s][A
 86%|████████▌ | 673/781 [00:54<00:07, 15.04it/s][A
 86%|████████▋ | 675/781 [00:54<00:07, 15.04it/s][A
 87%|████████▋ | 677/781 [00:54<00:06, 15.05it/s][A
 87%|████████▋ | 679/781 [00:54<00:06, 15.08it/s][A
 87%|████████▋ | 681/781 [00:54<00:06, 15.08it/s][A
 87%|████████▋ | 683/781 [00:54<00:06, 15.11it/s][A
 88%|████████▊ | 685/781 [00:54<00:06, 15.13it/s][A
 88%|████████▊ | 687/781 [00:55<00:06, 15.15it/s][A
 88%|████████▊ | 689/781 [00:55<00:06, 15.10it/s][A
 88%|████████▊ | 691/781 [00:55<00:05, 15.08it/s][A
 89%|████████▊ | 693/781 [00:55<00:05, 15.09it/s][A
 89%|████████▉ | 695/781 [00:55<00:05, 15.06it/s][A
 89%|████████▉ | 697/781 [00:55<00:05, 15.04it/s][A
 90%|████████▉ | 699/781 [00:55<00:05, 15.06it/s][A
 90%|████████▉ | 701/781 [00:55<00:05, 15.09it/s][A
 90%|█████████ | 703/781 [00:56<00:05, 15.11it/s][A
 90%|█████████ | 705/781 [00:56<00:05, 15.13it/s][A
 91%|█████████ | 707/781 [00:56<00:04, 15.11it/s][A
 91%|█████████ | 709/781 [00:56<00:04, 15.12it/s][A
 91%|█████████ | 711/781 [00:56<00:04, 15.09it/s][A
 91%|█████████▏| 713/781 [00:56<00:04, 15.12it/s][A
 92%|█████████▏| 715/781 [00:56<00:04, 15.14it/s][A
 92%|█████████▏| 717/781 [00:57<00:04, 15.11it/s][A
 92%|█████████▏| 719/781 [00:57<00:04, 15.12it/s][A
 92%|█████████▏| 721/781 [00:57<00:03, 15.08it/s][A
 93%|█████████▎| 723/781 [00:57<00:03, 15.09it/s][A
 93%|█████████▎| 725/781 [00:57<00:03, 15.07it/s][A
 93%|█████████▎| 727/781 [00:57<00:03, 15.11it/s][A
 93%|█████████▎| 729/781 [00:57<00:03, 15.07it/s][A
 94%|█████████▎| 731/781 [00:57<00:03, 15.09it/s][A
 94%|█████████▍| 733/781 [00:58<00:03, 15.10it/s][A
 94%|█████████▍| 735/781 [00:58<00:03, 15.08it/s][A
 94%|█████████▍| 737/781 [00:58<00:02, 15.10it/s][A
 95%|█████████▍| 739/781 [00:58<00:02, 15.12it/s][A
 95%|█████████▍| 741/781 [00:58<00:02, 15.13it/s][A
 95%|█████████▌| 743/781 [00:58<00:02, 15.15it/s][A
 95%|█████████▌| 745/781 [00:58<00:02, 15.15it/s][A
 96%|█████████▌| 747/781 [00:58<00:02, 15.10it/s][A
 96%|█████████▌| 749/781 [00:59<00:02, 15.14it/s][A
 96%|█████████▌| 751/781 [00:59<00:01, 15.15it/s][A
 96%|█████████▋| 753/781 [00:59<00:01, 15.14it/s][A
 97%|█████████▋| 755/781 [00:59<00:01, 15.15it/s][A
 97%|█████████▋| 757/781 [00:59<00:01, 15.16it/s][A
 97%|█████████▋| 759/781 [00:59<00:01, 15.11it/s][A
 97%|█████████▋| 761/781 [00:59<00:01, 15.12it/s][A
 98%|█████████▊| 763/781 [01:00<00:01, 15.11it/s][A
 98%|█████████▊| 765/781 [01:00<00:01, 15.10it/s][A
 98%|█████████▊| 767/781 [01:00<00:00, 15.12it/s][A
 98%|█████████▊| 769/781 [01:00<00:00, 15.13it/s][A
 99%|█████████▊| 771/781 [01:00<00:00, 15.11it/s][A
 99%|█████████▉| 773/781 [01:00<00:00, 15.11it/s][A
 99%|█████████▉| 775/781 [01:00<00:00, 15.08it/s][A
 99%|█████████▉| 777/781 [01:00<00:00, 15.11it/s][A
100%|█████████▉| 779/781 [01:01<00:00, 15.14it/s][A
100%|██████████| 781/781 [01:01<00:00, 15.09it/s][A100%|██████████| 781/781 [01:02<00:00, 12.41it/s]

  0%|          | 0/781 [00:00<?, ?it/s][A/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/torchvision/transforms/functional.py:136: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
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

  0%|          | 1/781 [00:06<1:26:18,  6.64s/it][A
  0%|          | 2/781 [00:07<38:25,  2.96s/it]  [A
  0%|          | 3/781 [00:07<21:46,  1.68s/it][A
  1%|          | 4/781 [00:07<13:50,  1.07s/it][A
  1%|          | 6/781 [00:07<07:03,  1.83it/s][A
  1%|          | 8/781 [00:07<04:22,  2.94it/s][A
  1%|▏         | 10/781 [00:07<03:02,  4.23it/s][A
  2%|▏         | 12/781 [00:07<02:16,  5.65it/s][A
  2%|▏         | 14/781 [00:08<01:47,  7.13it/s][A
  2%|▏         | 16/781 [00:08<01:29,  8.59it/s][A
  2%|▏         | 18/781 [00:08<01:17,  9.85it/s][A
  3%|▎         | 20/781 [00:08<01:09, 11.02it/s][A
  3%|▎         | 22/781 [00:08<01:04, 11.79it/s][A
  3%|▎         | 24/781 [00:08<01:00, 12.55it/s][A
  3%|▎         | 26/781 [00:08<00:56, 13.25it/s][A
  4%|▎         | 28/781 [00:08<00:55, 13.57it/s][A
  4%|▍         | 30/781 [00:09<00:53, 14.02it/s][A
  4%|▍         | 32/781 [00:09<00:52, 14.35it/s][A
  4%|▍         | 34/781 [00:09<00:51, 14.57it/s][A
  5%|▍         | 36/781 [00:09<00:50, 14.78it/s][A
  5%|▍         | 38/781 [00:09<00:50, 14.86it/s][A
  5%|▌         | 40/781 [00:09<00:49, 14.83it/s][A
  5%|▌         | 42/781 [00:09<00:49, 14.92it/s][A
  6%|▌         | 44/781 [00:10<00:49, 14.92it/s][A
  6%|▌         | 46/781 [00:10<00:48, 15.03it/s][A
  6%|▌         | 48/781 [00:10<00:48, 15.04it/s][A
  6%|▋         | 50/781 [00:10<00:48, 15.10it/s][A
  7%|▋         | 52/781 [00:10<00:48, 15.14it/s][A
  7%|▋         | 54/781 [00:10<00:48, 14.98it/s][A
  7%|▋         | 56/781 [00:10<00:48, 15.04it/s][A
  7%|▋         | 58/781 [00:10<00:48, 14.86it/s][A
  8%|▊         | 60/781 [00:11<00:48, 14.97it/s][A
  8%|▊         | 62/781 [00:11<00:49, 14.46it/s][A
  8%|▊         | 64/781 [00:11<00:49, 14.58it/s][A
  8%|▊         | 66/781 [00:11<00:48, 14.75it/s][A
  9%|▊         | 68/781 [00:11<00:48, 14.81it/s][A
  9%|▉         | 70/781 [00:11<00:49, 14.41it/s][A
  9%|▉         | 72/781 [00:11<00:48, 14.51it/s][A
  9%|▉         | 74/781 [00:12<00:48, 14.63it/s][A
 10%|▉         | 76/781 [00:12<00:47, 14.71it/s][A
 10%|▉         | 78/781 [00:12<00:47, 14.74it/s][A
 10%|█         | 80/781 [00:12<00:47, 14.77it/s][A
 10%|█         | 82/781 [00:12<00:47, 14.84it/s][A
 11%|█         | 84/781 [00:12<00:46, 14.93it/s][A
 11%|█         | 86/781 [00:12<00:46, 15.04it/s][A
 11%|█▏        | 88/781 [00:13<00:46, 15.02it/s][A
 12%|█▏        | 90/781 [00:13<00:45, 15.10it/s][A
 12%|█▏        | 92/781 [00:13<00:45, 15.10it/s][A
 12%|█▏        | 94/781 [00:13<00:45, 15.10it/s][A
 12%|█▏        | 96/781 [00:13<00:45, 15.10it/s][A
 13%|█▎        | 98/781 [00:13<00:45, 15.09it/s][A
 13%|█▎        | 100/781 [00:13<00:45, 15.05it/s][A
 13%|█▎        | 102/781 [00:13<00:45, 15.08it/s][A
 13%|█▎        | 104/781 [00:14<00:44, 15.11it/s][A
 14%|█▎        | 106/781 [00:14<00:44, 15.13it/s][A
 14%|█▍        | 108/781 [00:14<00:44, 15.12it/s][A
 14%|█▍        | 110/781 [00:14<00:44, 15.15it/s][A
 14%|█▍        | 112/781 [00:14<00:44, 15.11it/s][A
 15%|█▍        | 114/781 [00:14<00:44, 15.15it/s][A
 15%|█▍        | 116/781 [00:14<00:43, 15.14it/s][A
 15%|█▌        | 118/781 [00:14<00:44, 15.06it/s][A
 15%|█▌        | 120/781 [00:15<00:43, 15.07it/s][A
 16%|█▌        | 122/781 [00:15<00:43, 15.01it/s][A
 16%|█▌        | 124/781 [00:15<00:43, 14.98it/s][A
 16%|█▌        | 126/781 [00:15<00:43, 14.91it/s][A
 16%|█▋        | 128/781 [00:15<00:43, 14.93it/s][A
 17%|█▋        | 130/781 [00:15<00:43, 15.00it/s][A
 17%|█▋        | 132/781 [00:15<00:43, 14.79it/s][A
 17%|█▋        | 134/781 [00:16<00:45, 14.32it/s][A
 17%|█▋        | 136/781 [00:16<00:45, 14.20it/s][A
 18%|█▊        | 138/781 [00:16<00:45, 14.28it/s][A
 18%|█▊        | 140/781 [00:16<00:44, 14.38it/s][A
 18%|█▊        | 142/781 [00:16<00:44, 14.45it/s][A
 18%|█▊        | 144/781 [00:16<00:43, 14.49it/s][A
 19%|█▊        | 146/781 [00:16<00:43, 14.53it/s][A
 19%|█▉        | 148/781 [00:17<00:43, 14.63it/s][A
 19%|█▉        | 150/781 [00:17<00:42, 14.69it/s][A
 19%|█▉        | 152/781 [00:17<00:42, 14.67it/s][A
 20%|█▉        | 154/781 [00:17<00:42, 14.75it/s][A
 20%|█▉        | 156/781 [00:17<00:42, 14.77it/s][A
 20%|██        | 158/781 [00:17<00:42, 14.81it/s][A
 20%|██        | 160/781 [00:17<00:41, 14.82it/s][A
 21%|██        | 162/781 [00:17<00:41, 14.84it/s][A
 21%|██        | 164/781 [00:18<00:41, 14.78it/s][A
 21%|██▏       | 166/781 [00:18<00:41, 14.84it/s][A
 22%|██▏       | 168/781 [00:18<00:41, 14.86it/s][A
 22%|██▏       | 170/781 [00:18<00:41, 14.89it/s][A
 22%|██▏       | 172/781 [00:18<00:40, 14.87it/s][A
 22%|██▏       | 174/781 [00:18<00:40, 14.88it/s][A
 23%|██▎       | 176/781 [00:18<00:40, 14.89it/s][A
 23%|██▎       | 178/781 [00:19<00:40, 14.89it/s][A
 23%|██▎       | 180/781 [00:19<00:40, 14.85it/s][A
 23%|██▎       | 182/781 [00:19<00:40, 14.86it/s][A
 24%|██▎       | 184/781 [00:19<00:40, 14.87it/s][A
 24%|██▍       | 186/781 [00:19<00:40, 14.83it/s][A
 24%|██▍       | 188/781 [00:19<00:39, 14.85it/s][A
 24%|██▍       | 190/781 [00:19<00:39, 14.87it/s][A
 25%|██▍       | 192/781 [00:20<00:39, 14.86it/s][A
 25%|██▍       | 194/781 [00:20<00:39, 14.83it/s][A
 25%|██▌       | 196/781 [00:20<00:39, 14.86it/s][A
 25%|██▌       | 198/781 [00:20<00:39, 14.70it/s][A
 26%|██▌       | 200/781 [00:20<00:40, 14.43it/s][A
 26%|██▌       | 202/781 [00:20<00:41, 14.10it/s][A
 26%|██▌       | 204/781 [00:20<00:40, 14.23it/s][A
 26%|██▋       | 206/781 [00:20<00:40, 14.36it/s][A
 27%|██▋       | 208/781 [00:21<00:39, 14.46it/s][A
 27%|██▋       | 210/781 [00:21<00:39, 14.56it/s][A
 27%|██▋       | 212/781 [00:21<00:38, 14.67it/s][A
 27%|██▋       | 214/781 [00:21<00:38, 14.75it/s][A
 28%|██▊       | 216/781 [00:21<00:38, 14.74it/s][A
 28%|██▊       | 218/781 [00:21<00:38, 14.75it/s][A
 28%|██▊       | 220/781 [00:21<00:37, 14.77it/s][A
 28%|██▊       | 222/781 [00:22<00:37, 14.82it/s][A
 29%|██▊       | 224/781 [00:22<00:37, 14.86it/s][A
 29%|██▉       | 226/781 [00:22<00:37, 14.88it/s][A
 29%|██▉       | 228/781 [00:22<00:37, 14.81it/s][A
 29%|██▉       | 230/781 [00:22<00:37, 14.83it/s][A
 30%|██▉       | 232/781 [00:22<00:36, 14.86it/s][A
 30%|██▉       | 234/781 [00:22<00:36, 14.88it/s][A
 30%|███       | 236/781 [00:23<00:36, 14.87it/s][A
 30%|███       | 238/781 [00:23<00:36, 14.91it/s][A
 31%|███       | 240/781 [00:23<00:36, 14.93it/s][A
 31%|███       | 242/781 [00:23<00:36, 14.93it/s][A
 31%|███       | 244/781 [00:23<00:35, 14.93it/s][A
 31%|███▏      | 246/781 [00:23<00:35, 14.93it/s][A
 32%|███▏      | 248/781 [00:23<00:35, 14.94it/s][A
 32%|███▏      | 250/781 [00:23<00:35, 14.93it/s][A
 32%|███▏      | 252/781 [00:24<00:35, 14.93it/s][A
 33%|███▎      | 254/781 [00:24<00:35, 14.93it/s][A
 33%|███▎      | 256/781 [00:24<00:35, 14.93it/s][A
 33%|███▎      | 258/781 [00:24<00:35, 14.93it/s][A
 33%|███▎      | 260/781 [00:24<00:34, 14.90it/s][A
 34%|███▎      | 262/781 [00:24<00:35, 14.57it/s][A
 34%|███▍      | 264/781 [00:24<00:36, 14.11it/s][A
 34%|███▍      | 266/781 [00:25<00:36, 14.08it/s][A
 34%|███▍      | 268/781 [00:25<00:36, 14.23it/s][A
 35%|███▍      | 270/781 [00:25<00:35, 14.35it/s][A
 35%|███▍      | 272/781 [00:25<00:35, 14.49it/s][A
 35%|███▌      | 274/781 [00:25<00:34, 14.56it/s][A
 35%|███▌      | 276/781 [00:25<00:34, 14.68it/s][A
 36%|███▌      | 278/781 [00:25<00:34, 14.77it/s][A
 36%|███▌      | 280/781 [00:25<00:34, 14.72it/s][A
 36%|███▌      | 282/781 [00:26<00:33, 14.73it/s][A
 36%|███▋      | 284/781 [00:26<00:33, 14.76it/s][A
 37%|███▋      | 286/781 [00:26<00:33, 14.80it/s][A
 37%|███▋      | 288/781 [00:26<00:33, 14.82it/s][A
 37%|███▋      | 290/781 [00:26<00:33, 14.85it/s][A
 37%|███▋      | 292/781 [00:26<00:33, 14.76it/s][A
 38%|███▊      | 294/781 [00:26<00:32, 14.77it/s][A
 38%|███▊      | 296/781 [00:27<00:32, 14.79it/s][A
 38%|███▊      | 298/781 [00:27<00:32, 14.80it/s][A
 38%|███▊      | 300/781 [00:27<00:32, 14.79it/s][A
 39%|███▊      | 302/781 [00:27<00:32, 14.81it/s][A
 39%|███▉      | 304/781 [00:27<00:32, 14.80it/s][A
 39%|███▉      | 306/781 [00:27<00:32, 14.79it/s][A
 39%|███▉      | 308/781 [00:27<00:31, 14.79it/s][A
 40%|███▉      | 310/781 [00:28<00:31, 14.81it/s][A
 40%|███▉      | 312/781 [00:28<00:31, 14.83it/s][A
 40%|████      | 314/781 [00:28<00:31, 14.83it/s][A
 40%|████      | 316/781 [00:28<00:31, 14.81it/s][A
 41%|████      | 318/781 [00:28<00:31, 14.82it/s][A
 41%|████      | 320/781 [00:28<00:31, 14.86it/s][A
 41%|████      | 322/781 [00:28<00:30, 14.87it/s][A
 41%|████▏     | 324/781 [00:28<00:30, 14.84it/s][A
 42%|████▏     | 326/781 [00:29<00:31, 14.36it/s][A
 42%|████▏     | 328/781 [00:29<00:32, 14.03it/s][A
 42%|████▏     | 330/781 [00:29<00:31, 14.22it/s][A
 43%|████▎     | 332/781 [00:29<00:31, 14.33it/s][A
 43%|████▎     | 334/781 [00:29<00:30, 14.47it/s][A
 43%|████▎     | 336/781 [00:29<00:30, 14.56it/s][A
 43%|████▎     | 338/781 [00:29<00:30, 14.58it/s][A
 44%|████▎     | 340/781 [00:30<00:30, 14.62it/s][A
 44%|████▍     | 342/781 [00:30<00:29, 14.71it/s][A
 44%|████▍     | 344/781 [00:30<00:29, 14.80it/s][A
 44%|████▍     | 346/781 [00:30<00:29, 14.80it/s][A
 45%|████▍     | 348/781 [00:30<00:29, 14.85it/s][A
 45%|████▍     | 350/781 [00:30<00:28, 14.87it/s][A
 45%|████▌     | 352/781 [00:30<00:28, 14.89it/s][A
 45%|████▌     | 354/781 [00:31<00:28, 14.91it/s][A
 46%|████▌     | 356/781 [00:31<00:28, 14.82it/s][A
 46%|████▌     | 358/781 [00:31<00:28, 14.87it/s][A
 46%|████▌     | 360/781 [00:31<00:28, 14.90it/s][A
 46%|████▋     | 362/781 [00:31<00:28, 14.90it/s][A
 47%|████▋     | 364/781 [00:31<00:27, 14.91it/s][A
 47%|████▋     | 366/781 [00:31<00:27, 14.94it/s][A
 47%|████▋     | 368/781 [00:31<00:27, 14.95it/s][A
 47%|████▋     | 370/781 [00:32<00:27, 14.93it/s][A
 48%|████▊     | 372/781 [00:32<00:27, 14.93it/s][A
 48%|████▊     | 374/781 [00:32<00:27, 14.91it/s][A
 48%|████▊     | 376/781 [00:32<00:27, 14.91it/s][A
 48%|████▊     | 378/781 [00:32<00:27, 14.88it/s][A
 49%|████▊     | 380/781 [00:32<00:27, 14.83it/s][A
 49%|████▉     | 382/781 [00:32<00:26, 14.79it/s][A
 49%|████▉     | 384/781 [00:33<00:26, 14.84it/s][A
 49%|████▉     | 386/781 [00:33<00:26, 14.87it/s][A
 50%|████▉     | 388/781 [00:33<00:26, 14.87it/s][A
 50%|████▉     | 390/781 [00:33<00:26, 14.80it/s][A
 50%|█████     | 392/781 [00:33<00:26, 14.70it/s][A
 50%|█████     | 394/781 [00:33<00:26, 14.47it/s][A
 51%|█████     | 396/781 [00:33<00:26, 14.51it/s][A
 51%|█████     | 398/781 [00:33<00:26, 14.54it/s][A
 51%|█████     | 400/781 [00:34<00:26, 14.62it/s][A
 51%|█████▏    | 402/781 [00:34<00:25, 14.59it/s][A
 52%|█████▏    | 404/781 [00:34<00:25, 14.69it/s][A
 52%|█████▏    | 406/781 [00:34<00:25, 14.77it/s][A
 52%|█████▏    | 408/781 [00:34<00:25, 14.84it/s][A
 52%|█████▏    | 410/781 [00:34<00:25, 14.82it/s][A
 53%|█████▎    | 412/781 [00:34<00:24, 14.81it/s][A
 53%|█████▎    | 414/781 [00:35<00:24, 14.85it/s][A
 53%|█████▎    | 416/781 [00:35<00:24, 14.88it/s][A
 54%|█████▎    | 418/781 [00:35<00:24, 14.88it/s][A
 54%|█████▍    | 420/781 [00:35<00:24, 14.84it/s][A
 54%|█████▍    | 422/781 [00:35<00:24, 14.81it/s][A
 54%|█████▍    | 424/781 [00:35<00:24, 14.83it/s][A
 55%|█████▍    | 426/781 [00:35<00:23, 14.81it/s][A
 55%|█████▍    | 428/781 [00:36<00:23, 14.79it/s][A
 55%|█████▌    | 430/781 [00:36<00:23, 14.84it/s][A
 55%|█████▌    | 432/781 [00:36<00:23, 14.83it/s][A
 56%|█████▌    | 434/781 [00:36<00:23, 14.80it/s][A
 56%|█████▌    | 436/781 [00:36<00:23, 14.82it/s][A
 56%|█████▌    | 438/781 [00:36<00:23, 14.82it/s][A
 56%|█████▋    | 440/781 [00:36<00:23, 14.80it/s][A
 57%|█████▋    | 442/781 [00:36<00:22, 14.76it/s][A
 57%|█████▋    | 444/781 [00:37<00:22, 14.77it/s][A
 57%|█████▋    | 446/781 [00:37<00:22, 14.73it/s][A
 57%|█████▋    | 448/781 [00:37<00:22, 14.76it/s][A
 58%|█████▊    | 450/781 [00:37<00:22, 14.82it/s][A
 58%|█████▊    | 452/781 [00:37<00:22, 14.80it/s][A
 58%|█████▊    | 454/781 [00:37<00:22, 14.72it/s][A
 58%|█████▊    | 456/781 [00:37<00:22, 14.21it/s][A
 59%|█████▊    | 458/781 [00:38<00:22, 14.29it/s][A
 59%|█████▉    | 460/781 [00:38<00:22, 14.37it/s][A
 59%|█████▉    | 462/781 [00:38<00:21, 14.53it/s][A
 59%|█████▉    | 464/781 [00:38<00:21, 14.57it/s][A
 60%|█████▉    | 466/781 [00:38<00:21, 14.62it/s][A
 60%|█████▉    | 468/781 [00:38<00:21, 14.64it/s][A
 60%|██████    | 470/781 [00:38<00:21, 14.70it/s][A
 60%|██████    | 472/781 [00:39<00:21, 14.68it/s][A
 61%|██████    | 474/781 [00:39<00:20, 14.72it/s][A
 61%|██████    | 476/781 [00:39<00:20, 14.74it/s][A
 61%|██████    | 478/781 [00:39<00:20, 14.71it/s][A
 61%|██████▏   | 480/781 [00:39<00:20, 14.78it/s][A
 62%|██████▏   | 482/781 [00:39<00:20, 14.75it/s][A
 62%|██████▏   | 484/781 [00:39<00:20, 14.77it/s][A
 62%|██████▏   | 486/781 [00:39<00:19, 14.83it/s][A
 62%|██████▏   | 488/781 [00:40<00:19, 14.87it/s][A
 63%|██████▎   | 490/781 [00:40<00:19, 14.87it/s][A
 63%|██████▎   | 492/781 [00:40<00:19, 14.91it/s][A
 63%|██████▎   | 494/781 [00:40<00:19, 14.93it/s][A
 64%|██████▎   | 496/781 [00:40<00:19, 14.91it/s][A
 64%|██████▍   | 498/781 [00:40<00:19, 14.81it/s][A
 64%|██████▍   | 500/781 [00:40<00:19, 14.75it/s][A
 64%|██████▍   | 502/781 [00:41<00:18, 14.75it/s][A
 65%|██████▍   | 504/781 [00:41<00:18, 14.78it/s][A
 65%|██████▍   | 506/781 [00:41<00:18, 14.73it/s][A
 65%|██████▌   | 508/781 [00:41<00:18, 14.77it/s][A
 65%|██████▌   | 510/781 [00:41<00:18, 14.76it/s][A
 66%|██████▌   | 512/781 [00:41<00:18, 14.81it/s][A
 66%|██████▌   | 514/781 [00:41<00:18, 14.82it/s][A
 66%|██████▌   | 516/781 [00:41<00:17, 14.75it/s][A
 66%|██████▋   | 518/781 [00:42<00:18, 14.20it/s][A
 67%|██████▋   | 520/781 [00:42<00:18, 14.29it/s][A
 67%|██████▋   | 522/781 [00:42<00:18, 14.37it/s][A
 67%|██████▋   | 524/781 [00:42<00:17, 14.43it/s][A
 67%|██████▋   | 526/781 [00:42<00:17, 14.54it/s][A
 68%|██████▊   | 528/781 [00:42<00:17, 14.56it/s][A
 68%|██████▊   | 530/781 [00:42<00:17, 14.56it/s][A
 68%|██████▊   | 532/781 [00:43<00:16, 14.68it/s][A
 68%|██████▊   | 534/781 [00:43<00:16, 14.76it/s][A
 69%|██████▊   | 536/781 [00:43<00:16, 14.82it/s][A
 69%|██████▉   | 538/781 [00:43<00:16, 14.80it/s][A
 69%|██████▉   | 540/781 [00:43<00:16, 14.80it/s][A
 69%|██████▉   | 542/781 [00:43<00:16, 14.74it/s][A
 70%|██████▉   | 544/781 [00:43<00:16, 14.74it/s][A
 70%|██████▉   | 546/781 [00:44<00:15, 14.76it/s][A
 70%|███████   | 548/781 [00:44<00:15, 14.73it/s][A
 70%|███████   | 550/781 [00:44<00:15, 14.79it/s][A
 71%|███████   | 552/781 [00:44<00:15, 14.81it/s][A
 71%|███████   | 554/781 [00:44<00:15, 14.83it/s][A
 71%|███████   | 556/781 [00:44<00:15, 14.85it/s][A
 71%|███████▏  | 558/781 [00:44<00:15, 14.87it/s][A
 72%|███████▏  | 560/781 [00:44<00:14, 14.83it/s][A
 72%|███████▏  | 562/781 [00:45<00:15, 14.46it/s][A
 72%|███████▏  | 564/781 [00:45<00:14, 14.50it/s][A
 72%|███████▏  | 566/781 [00:45<00:14, 14.61it/s][A
 73%|███████▎  | 568/781 [00:45<00:14, 14.64it/s][A
 73%|███████▎  | 570/781 [00:45<00:14, 14.67it/s][A
 73%|███████▎  | 572/781 [00:45<00:14, 14.64it/s][A
 73%|███████▎  | 574/781 [00:45<00:14, 14.64it/s][A
 74%|███████▍  | 576/781 [00:46<00:13, 14.71it/s][A
 74%|███████▍  | 578/781 [00:46<00:13, 14.78it/s][A
 74%|███████▍  | 580/781 [00:46<00:13, 14.77it/s][A
 75%|███████▍  | 582/781 [00:46<00:13, 14.75it/s][A
 75%|███████▍  | 584/781 [00:46<00:13, 14.67it/s][A
 75%|███████▌  | 586/781 [00:46<00:13, 14.61it/s][A
 75%|███████▌  | 588/781 [00:46<00:13, 14.60it/s][A
 76%|███████▌  | 590/781 [00:47<00:13, 14.66it/s][A
 76%|███████▌  | 592/781 [00:47<00:12, 14.66it/s][A
 76%|███████▌  | 594/781 [00:47<00:12, 14.67it/s][A
 76%|███████▋  | 596/781 [00:47<00:12, 14.73it/s][A
 77%|███████▋  | 598/781 [00:47<00:12, 14.79it/s][A
 77%|███████▋  | 600/781 [00:47<00:12, 14.75it/s][A
 77%|███████▋  | 602/781 [00:47<00:12, 14.76it/s][A
 77%|███████▋  | 604/781 [00:47<00:11, 14.75it/s][A
 78%|███████▊  | 606/781 [00:48<00:11, 14.73it/s][A
 78%|███████▊  | 608/781 [00:48<00:11, 14.78it/s][A
 78%|███████▊  | 610/781 [00:48<00:11, 14.75it/s][A
 78%|███████▊  | 612/781 [00:48<00:11, 14.75it/s][A
 79%|███████▊  | 614/781 [00:48<00:11, 14.78it/s][A
 79%|███████▉  | 616/781 [00:48<00:11, 14.81it/s][A
 79%|███████▉  | 618/781 [00:48<00:10, 14.83it/s][A
 79%|███████▉  | 620/781 [00:49<00:10, 14.86it/s][A
 80%|███████▉  | 622/781 [00:49<00:10, 14.89it/s][A
 80%|███████▉  | 624/781 [00:49<00:10, 14.88it/s][A
 80%|████████  | 626/781 [00:49<00:10, 14.84it/s][A
 80%|████████  | 628/781 [00:49<00:10, 14.41it/s][A
 81%|████████  | 630/781 [00:49<00:10, 14.51it/s][A
 81%|████████  | 632/781 [00:49<00:10, 14.63it/s][A
 81%|████████  | 634/781 [00:50<00:10, 14.64it/s][A
 81%|████████▏ | 636/781 [00:50<00:09, 14.64it/s][A
 82%|████████▏ | 638/781 [00:50<00:09, 14.66it/s][A
 82%|████████▏ | 640/781 [00:50<00:09, 14.75it/s][A
 82%|████████▏ | 642/781 [00:50<00:09, 14.75it/s][A
 82%|████████▏ | 644/781 [00:50<00:09, 14.70it/s][A
 83%|████████▎ | 646/781 [00:50<00:09, 14.68it/s][A
 83%|████████▎ | 648/781 [00:50<00:09, 14.68it/s][A
 83%|████████▎ | 650/781 [00:51<00:08, 14.65it/s][A
 83%|████████▎ | 652/781 [00:51<00:08, 14.63it/s][A
 84%|████████▎ | 654/781 [00:51<00:08, 14.67it/s][A
 84%|████████▍ | 656/781 [00:51<00:08, 14.80it/s][A
 84%|████████▍ | 658/781 [00:51<00:08, 14.84it/s][A
 85%|████████▍ | 660/781 [00:51<00:08, 14.86it/s][A
 85%|████████▍ | 662/781 [00:51<00:07, 14.89it/s][A
 85%|████████▌ | 664/781 [00:52<00:07, 14.91it/s][A
 85%|████████▌ | 666/781 [00:52<00:07, 14.90it/s][A
 86%|████████▌ | 668/781 [00:52<00:07, 14.94it/s][A
 86%|████████▌ | 670/781 [00:52<00:07, 14.96it/s][A
 86%|████████▌ | 672/781 [00:52<00:07, 14.97it/s][A
 86%|████████▋ | 674/781 [00:52<00:07, 14.97it/s][A
 87%|████████▋ | 676/781 [00:52<00:07, 14.94it/s][A
 87%|████████▋ | 678/781 [00:52<00:06, 14.94it/s][A
 87%|████████▋ | 680/781 [00:53<00:06, 14.94it/s][A
 87%|████████▋ | 682/781 [00:53<00:06, 14.94it/s][A
 88%|████████▊ | 684/781 [00:53<00:06, 14.95it/s][A
 88%|████████▊ | 686/781 [00:53<00:06, 14.95it/s][A
 88%|████████▊ | 688/781 [00:53<00:06, 14.97it/s][A
 88%|████████▊ | 690/781 [00:53<00:06, 14.98it/s][A
 89%|████████▊ | 692/781 [00:53<00:05, 15.00it/s][A
 89%|████████▉ | 694/781 [00:54<00:05, 15.00it/s][A
 89%|████████▉ | 696/781 [00:54<00:05, 14.99it/s][A
 89%|████████▉ | 698/781 [00:54<00:05, 14.98it/s][A
 90%|████████▉ | 700/781 [00:54<00:05, 15.01it/s][A
 90%|████████▉ | 702/781 [00:54<00:05, 14.99it/s][A
 90%|█████████ | 704/781 [00:54<00:05, 14.98it/s][A
 90%|█████████ | 706/781 [00:54<00:05, 14.98it/s][A
 91%|█████████ | 708/781 [00:54<00:04, 14.97it/s][A
 91%|█████████ | 710/781 [00:55<00:04, 15.01it/s][A
 91%|█████████ | 712/781 [00:55<00:04, 15.03it/s][A
 91%|█████████▏| 714/781 [00:55<00:04, 15.01it/s][A
 92%|█████████▏| 716/781 [00:55<00:04, 14.99it/s][A
 92%|█████████▏| 718/781 [00:55<00:04, 14.99it/s][A
 92%|█████████▏| 720/781 [00:55<00:04, 15.03it/s][A
 92%|█████████▏| 722/781 [00:55<00:03, 15.01it/s][A
 93%|█████████▎| 724/781 [00:56<00:03, 15.00it/s][A
 93%|█████████▎| 726/781 [00:56<00:03, 14.99it/s][A
 93%|█████████▎| 728/781 [00:56<00:03, 15.01it/s][A
 93%|█████████▎| 730/781 [00:56<00:03, 14.98it/s][A
 94%|█████████▎| 732/781 [00:56<00:03, 14.98it/s][A
 94%|█████████▍| 734/781 [00:56<00:03, 14.96it/s][A
 94%|█████████▍| 736/781 [00:56<00:03, 14.96it/s][A
 94%|█████████▍| 738/781 [00:56<00:02, 14.97it/s][A
 95%|█████████▍| 740/781 [00:57<00:02, 14.98it/s][A
 95%|█████████▌| 742/781 [00:57<00:02, 14.98it/s][A
 95%|█████████▌| 744/781 [00:57<00:02, 14.96it/s][A
 96%|█████████▌| 746/781 [00:57<00:02, 14.97it/s][A
 96%|█████████▌| 748/781 [00:57<00:02, 14.98it/s][A
 96%|█████████▌| 750/781 [00:57<00:02, 14.96it/s][A
 96%|█████████▋| 752/781 [00:57<00:01, 14.96it/s][A
 97%|█████████▋| 754/781 [00:58<00:01, 14.97it/s][A
 97%|█████████▋| 756/781 [00:58<00:01, 14.96it/s][A
 97%|█████████▋| 758/781 [00:58<00:01, 14.96it/s][A
 97%|█████████▋| 760/781 [00:58<00:01, 14.96it/s][A
 98%|█████████▊| 762/781 [00:58<00:01, 14.96it/s][A
 98%|█████████▊| 764/781 [00:58<00:01, 14.98it/s][A
 98%|█████████▊| 766/781 [00:58<00:01, 14.96it/s][A
 98%|█████████▊| 768/781 [00:59<00:00, 14.96it/s][A
 99%|█████████▊| 770/781 [00:59<00:00, 14.97it/s][A
 99%|█████████▉| 772/781 [00:59<00:00, 15.01it/s][A
 99%|█████████▉| 774/781 [00:59<00:00, 15.01it/s][A
 99%|█████████▉| 776/781 [00:59<00:00, 15.02it/s][A
100%|█████████▉| 778/781 [00:59<00:00, 14.99it/s][A
100%|█████████▉| 780/781 [00:59<00:00, 14.98it/s][A100%|██████████| 781/781 [01:01<00:00, 12.67it/s]
/nfs/data_chaos/czhang/anaconda3/envs/householdergan/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/logger_connector/result.py:431: It is recommended to use `self.log('fid_ema_T10_Tlatent10', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
