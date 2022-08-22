import argparse

import torch
from torchvision import utils
from model import Generator
import numpy as np
import os
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


loss_fn = lpips.PerceptualLoss(model="net-lin", net="vgg").cuda()

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

    g.load_state_dict(ckpt["g_ema"], strict=True)

    trunc = g.mean_latent(4096)
    print(eigvec_dict.keys())
    #Evaluate PPL
    #dist = []
    #for i in range(100):
    #    latent1 = torch.randn(100, 512, device=args.device)
    #    latent2 = torch.randn(100, 512, device=args.device)
    #    #Interpolation between two latents
    #    t = torch.rand([latent1.shape[0]], device=latent1.device)
    #    zt0 = slerp(latent1, latent2, t.unsqueeze(0))
    #    zt1 = slerp(latent1, latent2, t.unsqueeze(0) + 1e-4)
    #    img1, _ = g.forward_test([zt0],torch.zeros_like(latent1),0,truncation=args.truncation,truncation_latent=trunc,input_is_latent=True)
    #    img2, _ = g.forward_test([zt1],torch.zeros_like(latent1),0,truncation=args.truncation,truncation_latent=trunc,input_is_latent=True)
    #    with torch.no_grad():
    #        d = loss_fn.forward(img1, img2) / (1e-4 ** 2)
    #    dist.append(d)
    #dist = torch.cat(dist).cpu().numpy()
    #lo = np.percentile(dist, 1, interpolation='lower')
    #hi = np.percentile(dist, 99, interpolation='higher')
    #ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
    #print(float(ppl))

    #Evaluate FID and VP
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

        value_list = list(eigvec_dict.values())[key // 2 - 1]
        #value_list = eigvec_dict[key]

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


