import argparse

import torch
from torchvision import utils
from model import Generator
import numpy as np
import os

torch.random.manual_seed(15927)

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
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=7, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
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
        "--ortho_id",
        type=int,
        default=-2,
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

    ckpt = torch.load(args.ckpt)
    g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier, ortho_id=args.ortho_id, diag_size=args.diag_size).to(args.device)

    g.load_state_dict(ckpt["g_ema"], strict=True)

    trunc = g.mean_latent(4096)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)

    img, _ = g(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True
    )

    for index, key in enumerate(eigvec_dict.keys()):

        for j in range(args.diag_size):
            imglists = []
            for i in np.linspace(-20, 20, 7):
                direction = i * eigvec_dict[key][:, j].unsqueeze(0).to(args.device)
                img1, _ = g.forward_test(
                    [latent],
                    direction,
                    index,
                    truncation=args.truncation,
                    truncation_latent=trunc,
                    input_is_latent=True,
                )
                imglists.append(img1)

            imgs = torch.cat(imglists, dim=0)
            grid = utils.save_image(imgs,
                os.path.join(args.output_dir, f"{args.out_prefix}_layer-{index}-index-{j}__all.png"),
                normalize=True,
                range=(-1, 1),
                nrow=args.n_sample,
            )
