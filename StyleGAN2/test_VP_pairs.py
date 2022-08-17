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
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
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
        default='output_vp_pairs',
        help="name of the closed form factorization result factor file",
    )

    parser.add_argument(
        "--is_ortho",
        action="store_true",
        help="name of the closed form factorization result factor file",
    )

    parser.add_argument(
        "--ortho_id",
        type=int,
        default=1,
        help="name of the closed form factorization result factor file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.output_dir_FID = args.output_dir + '_FID'
    if not os.path.exists(args.output_dir_FID):
        os.mkdir(args.output_dir_FID)

    print('args', args.factor)
    eigvec_dict = torch.load(args.factor)

    ckpt = torch.load(args.ckpt)
    g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier, ortho_id=args.ortho_id).to(args.device)

    g.load_state_dict(ckpt["g_ema"], strict=False)

    trunc = g.mean_latent(4096)
    print(eigvec_dict.keys())

    labels = []
    for i in range(10000):

        latent = torch.randn(1, 512, device=args.device)
        latent = g.get_latent(latent)

        key = np.random.randint(1,7) * 2
        j = np.random.randint(0, 10)

        delta_onehot = np.zeros((1, 6 * 10))
        delta_onehot[:, (key // 2 - 1) * 10 + j] = 1

        if i == 0:
            labels = delta_onehot
        else:
            labels = np.concatenate([labels, delta_onehot], axis=0)

        value_list = list(eigvec_dict.values())[key]

        direction = 5 * value_list[:, j].unsqueeze(0).to(args.device)
        img1, _ = g.forward_test(
            [latent],
            direction,
            key,
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )
        img0, _ = g.forward_test(
            [latent],
            torch.zeros_like(direction),
            key,
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )

        imgs = torch.cat([img0, img1], dim=3)

        utils.save_image(imgs,
            os.path.join(args.output_dir, 'pair_%06d.png' % (i)),
            normalize=True,
            range=(-1, 1),
            nrow=args.n_sample,
        )

        utils.save_image(img0,
            os.path.join(args.output_dir_FID, '%06d.png' % (i)),
            normalize=True,
            range=(-1, 1),
            nrow=args.n_sample,
        )

    np.save(os.path.join(args.output_dir, 'labels.npy'), labels)


