import argparse

import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from experiment import LitModel
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
from templates import *
from templates_latent import *

import lpips


def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))


def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = t * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def lerp(a, b, t):
    return a + (b - a) * t


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Perceptual Path Length calculator")

    parser.add_argument(
        "--space", default="w", choices=["z", "w"], help="space that PPL calculated with"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch size for the models"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=5000,
        help="number of the samples for calculating PPL",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-4, help="epsilon for numerical stability"
    )
    parser.add_argument(
        "--crop", action="store_true", help="apply center crop to the images"
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument(
        "--sampling",
        default="end",
        choices=["end", "full"],
        help="set endpoint sampling method",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="diffae checkpoints")

    data_path = 'datasets/ffhq256.lmdb'

    args = parser.parse_args()

    latent_dim = 512
    ckpt = torch.load(args.ckpt)
    conf = ffhq128_autoenc_130M()

    model = LitModel(conf)
    state = torch.load(args.ckpt, map_location='cpu')
    model_state_dict = state['state_dict']
    model.load_state_dict(model_state_dict, strict=True)




    # percept = lpips.PerceptualLoss(
    #     model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    # )

    percept = lpips.LPIPS(net='vgg').to(device)

    distances = []

    n_batch = args.n_sample // args.batch
    resid = args.n_sample - (n_batch * args.batch)
    batch_sizes = [args.batch] * n_batch + [resid]

    data = FFHQlmdb(path=data_path,
                    image_size=args.size,
                    split='test')
    dataloader = torch.utils.data.DataLoader(data, 
                                            batch_size=args.batch, 
                                            num_workers=4, 
                                            drop_last=True,
                                            shuffle=True)
    

    # test LPIPS distance between simple images
    # test_image1 = torch.ones((1, 3, 256, 256)).to(device) * 0.5  
    # test_image2 = torch.ones((1, 3, 256, 256)).to(device) * 0.6  
    # test_image1 = test_image1 * 2 - 1  
    # test_image2 = test_image2 * 2 - 1  

   
    # test_dist = percept(test_image1, test_image2)
    # print("Test LPIPS distance between simple images:", test_dist)  # 0.0086


    with torch.no_grad():
        for batch_size in tqdm(batch_sizes):

            batch = next(iter(dataloader))

            model.ema_model.eval()
            model.ema_model.to(device='cuda:0')
            cond = model.encode(batch['img'].to(device))
            xT = model.encode_stochastic(batch['img'].to(device='cuda:0'), cond=cond, T=100)
            image = model.render(xT, cond=cond, T=100)

            # normalize image
            image = image * 2 - 1

            # print("Image range:", image.min().item(), image.max().item())

            raw_dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2)
            # print("Raw LPIPS distances:", raw_dist)

            eps = 1e-2  
            scaled_dist = raw_dist / (eps ** 2)
            # print("Scaled distances:", scaled_dist)

            # if args.crop:
            #     c = image.shape[2] // 8
            #     image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

            factor = image.shape[2] // 256
        
            if factor > 1:
                image = F.interpolate(
                    image, size=(256, 256), mode="bilinear", align_corners=False
                )

            # dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (
            #     args.eps ** 2
            # )
            distances.append(scaled_dist.to("cpu").numpy())

    distances = np.concatenate(distances, 0)

    if len(distances) == 0:
        print("Warning: distances array is empty")
        lo = 0  
    else:
        lo = np.percentile(distances, 1, interpolation="lower")
        hi = np.percentile(distances, 99, interpolation="higher")
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )

    print("ppl:", filtered_dist.mean())
