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
        "--batch", type=int, default=64, help="batch size for the models"
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

    args = parser.parse_args()

    latent_dim = 512
    ckpt = torch.load(args.ckpt)
    conf = ffhq128_autoenc_130M()

    model = LitModel(conf)
    state = torch.load(args.ckpt, map_location='cpu')
    model_state_dict = state['state_dict']
    model.load_state_dict(model_state_dict, strict=True)




    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    distances = []

    n_batch = args.n_sample // args.batch
    resid = args.n_sample - (n_batch * args.batch)
    batch_sizes = [args.batch] * n_batch + [resid]

    with torch.no_grad():
        for batch in tqdm(batch_sizes):
            # noise = g.make_noise()

            # inputs = torch.randn([batch * 2, latent_dim], device=device)
            # if args.sampling == "full":
            #     lerp_t = torch.rand(batch, device=device)
            # else:
            #     lerp_t = torch.zeros(batch, device=device)

            # if args.space == "w":
            #     latent = g.get_latent(inputs)
            #     latent_t0, latent_t1 = latent[::2], latent[1::2]
            #     latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None])
            #     latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + args.eps)
            #     latent_e = torch.stack([latent_e0, latent_e1], 1).view(*latent.shape)

            # image, _ = g([latent_e], input_is_latent=True, noise=noise)

            data = ImageDataset('imgs_align', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
            batch = data[0]['img'][None].to(device='cuda:0')

            model.ema_model.eval()
            model.ema_model.to(device='cuda:0')
            cond = model.encode(batch)
            xT = model.encode_stochastic(batch.to(device='cuda:0'), cond=cond, T=100)
            image = model.render(xT, cond=cond, T=100)


            if args.crop:
                c = image.shape[2] // 8
                image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

            factor = image.shape[2] // 256

            if factor > 1:
                image = F.interpolate(
                    image, size=(256, 256), mode="bilinear", align_corners=False
                )

            dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (
                args.eps ** 2
            )
            distances.append(dist.to("cpu").numpy())

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation="lower")
    hi = np.percentile(distances, 99, interpolation="higher")
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )

    print("ppl:", filtered_dist.mean())
