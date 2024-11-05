import argparse

import torch
from torchvision import utils
import numpy as np
import os
from experiment import LitModel
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
from templates import *
from templates_latent import *
import cv2
from PIL import Image  
from torchvision import transforms

torch.random.manual_seed(15927)

nodes = 1


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument("--ckpt", type=str, required=True, help="diffae checkpoints")
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the diffae model"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=7, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--gpus", type=dict, default=[0, 1], help="gpus to run the model"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )
    parser.add_argument(
        "--factor",
        type=str,
        help="name of the closed form factorization result factor file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='output_dir',
        help="name of the closed form factorization result factor file",
    )

    parser.add_argument(
        "--diag_size",
        type=int,
        default=10,
        help="How many attributes to extract",
    )


    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print('args', args.factor)
    eigvec_dict = torch.load(args.factor)
    # Get only the EMA key
    ema_key = [key for key in eigvec_dict.keys() if key.startswith('ema_')][0]
    print(f"Using EMA key: {ema_key}")
    logdir = os.path.join(args.output_dir, 'logs')

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logdir,
                                             name=None,
                                             version='')

    gpus = args.gpus
    if len(gpus) == 1 and nodes == 1:
        accelerator = None
    else:
        accelerator = 'auto'

    conf = ffhq128_autoenc_130M()

    model = LitModel(conf)
    state = torch.load(args.ckpt, map_location='cpu')
    model_state_dict = state['state_dict']
    model.load_state_dict(model_state_dict, strict=True)

    # trainer = pl.Trainer(
    #     max_steps=conf.total_samples // conf.batch_size_effective,
    #     num_nodes=nodes,
    #     accelerator=accelerator,
    #     precision=16 if conf.fp16 else 32,
    #     callbacks=[
    #         LearningRateMonitor(),
    #     ],
    #     logger=tb_logger,
    #     accumulate_grad_batches=conf.accum_batches,
    #     strategy="ddp",
    # )

    img_size = conf.img_size
    noise = torch.randn(args.n_sample, 3, img_size, img_size).to(device='cuda:0')


    # dummy = torch.zeros_like(noise).to(device='cuda:0')

    # transform = transforms.Compose([ transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # image_path = 'imgs_align/sandy.png' 
    # img_align = Image.open(image_path).convert("RGB") 
    # image_tensor = transform(img_align).to(device='cuda:0')
    # image_tensor = image_tensor.unsqueeze(0)


    data = ImageDataset('imgs_align', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
    batch = data[0]['img'][None].to(device='cuda:0')

    model.ema_model.eval()
    model.ema_model.to(device='cuda:0')
    cond = model.encode(batch)
    xT = model.encode_stochastic(batch.to(device='cuda:0'), cond=cond, T=100)

    # img_align = cv2.imread('imgs_align/sandy.png')
    # noise_zero = torch.zeros_like(noise)

    with torch.no_grad():  # Disable gradient computation
        img = model.render(xT, cond=cond, T=100)
    grid0 = utils.save_image(img,
        os.path.join(args.output_dir, f"{args.out_prefix}_original_1105.png"),
        normalize=True,
        range=(0, 1),
        nrow=args.n_sample,
    )

    # print("shape of ema_key: ", eigvec_dict[ema_key].shape)
    # pdb.set_trace()

    # for index, key in enumerate(ema_key):
    index = 0
    key = ema_key
    print(f"Processing layer {index}...")

    cond_orig = cond.clone()
    with torch.no_grad():
        for j in range(args.diag_size):
            imglists = []
            cond = cond_orig.clone()
            for i in np.linspace(0, 10, 7):
                direction = eigvec_dict[key][:, j].unsqueeze(0).to(device='cuda:0') # (1, 512)
                direction = direction / direction.norm()
                
                # direction and cond are both (1, 512)
                # forward pass to get the image
                img1 = model.render(xT, cond=cond + i * direction, T=20)
                imglists.append(img1)

            imgs = torch.cat(imglists, dim=0)
            grid = utils.save_image(imgs,
                os.path.join(args.output_dir, f"{args.out_prefix}_layer-{index}-index-{j}__all.png"),
                normalize=True,
                range=(0, 1),
                nrow=args.n_sample,
            )
