import argparse

import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import math
import lpips
from experiment import LitModel
from templates import *
from templates_latent import *


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
        help="number of the samples for calculating PIPL",
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

    parser.add_argument(
        "--factor",
        type=str,
        help="name of the closed form factorization result factor file",
    )

    args = parser.parse_args()

    latent_dim = 512
    #Load factor
    print('args', args.factor)
    eigvec_dict = torch.load(args.factor)
    ema_key = [key for key in eigvec_dict.keys() if key.startswith('ema_')]
    len_key = len(ema_key)
    # ema_value = eigvec_dict[ema_key]
    print(f"Using EMA key: {ema_key}")
    #Load checkpoint
    ckpt = torch.load(args.ckpt)

    conf = ffhq128_autoenc_130M()

    model = LitModel(conf)
    state = torch.load(args.ckpt, map_location='cpu')
    model_state_dict = state['state_dict']
    model.load_state_dict(model_state_dict, strict=True)

    percept = lpips.LPIPS(net='vgg').to(device)

    distances = []

    n_batch = args.n_sample // args.batch
    resid = args.n_sample - (n_batch * args.batch)
    batch_sizes = [args.batch] * n_batch + [resid]


    data_path = 'datasets/ffhq256.lmdb'
    data = FFHQlmdb(path=data_path,
                    image_size=args.size,
                    split='test')
    dataloader = torch.utils.data.DataLoader(data, 
                                            batch_size=args.batch, 
                                            num_workers=4, 
                                            drop_last=True,
                                            shuffle=True)

    with torch.no_grad():
        epoch = 0
        for batch_size in tqdm(batch_sizes):
            epoch += 1

            batch = next(iter(dataloader))

            model.ema_model.eval()
            model.ema_model.to(device='cuda:0')
            cond = model.encode(batch['img'].to(device))

            if args.sampling == "full":
                lerp_t = torch.rand(cond.shape[0]//2, device=device)
            else:
                lerp_t = torch.zeros(cond.shape[0]//2, device=device)

            latent_t0, latent_t1 = cond[::2], cond[1::2]
            latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None])
            #Random Eigenvector Direction
            key = np.random.randint(0, len_key)
            j = np.random.randint(0, 10)
            value_list = list(eigvec_dict.values())[key]
            direction = value_list[:, j].unsqueeze(0).to(cond.device)
            direction = direction / direction.norm() 
            latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None]) + args.eps * direction
            latent_e = torch.stack([latent_e0, latent_e1], 1).view(cond.shape)

            

            # Generate random noise for each interpolated point
            actual_pairs = batch['img'].size(0) // 2
            noise = torch.randn_like(batch['img'][:actual_pairs]).to(device)  # Half batch size
            xT_e0 = noise  # For first interpolation point
            xT_e1 = noise  # Use same noise for second point for fair comparison

            # Generate two sets of images from interpolated latents
            image_e0 = model.render(xT_e0, cond=latent_e0, T=100)  # First interpolation point
            image_e1 = model.render(xT_e1, cond=latent_e1, T=100)  # Second interpolation point (epsilon step)

            # normalize image
            image_e0 = image_e0 * 2 - 1
            image_e1 = image_e1 * 2 - 1
            raw_dist = percept(image_e0, image_e1).view(image_e0.shape[0])

            # eps = 1e-2  
            scaled_dist = raw_dist / (args.eps ** 2)

            factor = image_e0.shape[2] // 256

            if factor > 1:
                image_e0 = F.interpolate(
                    image_e0, size=(args.size, args.size), mode="bilinear", align_corners=False
                )
                image_e1 = F.interpolate(
                    image_e1, size=(args.size, args.size), mode="bilinear", align_corners=False
                )

            # print("scaled_dist:", scaled_dist)
            # pdb.set_trace()

            distances.append(scaled_dist.to("cpu").numpy())

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation="lower")
    hi = np.percentile(distances, 99, interpolation="higher")
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )

    print("finish ffhq multi mlp pipl!\n", filtered_dist.mean())
    print("pipl ffhq multi mlp eps 1e-1:", filtered_dist.mean())
