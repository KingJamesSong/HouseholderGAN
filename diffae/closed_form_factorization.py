import argparse

import torch
import os
import pdb


def H(v):
    d = v.shape[0]
    I = torch.eye(d, d, device=v.device)
    v = v.reshape(d, 1)
    return I - 2 * v @ v.T

def Q(V):
    d = V.shape[0]
    norms = torch.norm(V.T, 2, dim=1)
    V = V / norms.view(1, d)
    M = torch.eye(d, d, device=V.device)
    for i in range(d):
        M = M @ H(V[:, i:i + 1])
    return M

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract factor/eigenvectors of latent spaces using closed form factorization"
    )

    parser.add_argument(
        "--out", type=str, default="factor.pt", help="name of the result factor file"
    )
    parser.add_argument("ckpt", type=str, help="name of the model checkpoint")

    parser.add_argument(
        "--is_ortho",
        action="store_true",
        help="name of the closed form factorization result factor file",
    )

    parser.add_argument(
        "--diag_size",
        type=int,
        default=10,
        help="size of idenity matrix to ",
    )



    args = parser.parse_args()

    if args.is_ortho:
        ckpt = torch.load(args.ckpt)
        # ckpt keys: ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'MixedPrecision', 'hparams_name', 'hyper_parameters']

        #weight_mat =  []
        U = None
        V = None

        eigvec_ = {}

        for k, v in ckpt['state_dict'].items():
            if '.U' in k:
                U = v

                print(k)
            if '.V' in k:
                V = v
                print(k)

            if U is not None and V is not None:
                d1 = U.shape[0]
                d2 = V.shape[0]

                S = torch.zeros(d1, d2).to(U)
                for i in range(args.diag_size):
                    S[i, i] = 1

                if d1 < d2:
                    weight = Q(U).mm(S).mm(Q(V))
                else:
                    weight = Q(U).mm(S).mm(Q(V))

                U = None
                V = None

                eigvec = torch.svd(weight).V.to("cpu")

                eigvec_[k] = eigvec

        torch.save(eigvec_, args.out)
    else:
        ckpt = torch.load(args.ckpt)
        modulate = {
            k: v
            for k, v in ckpt['state_dict'].items()
            # if "time_embed" in k and "style" in k and "weight" in k and "ema_model" in k
            if ("style_enc" in k or "style_dec" in k or "style_mid" in k) and "weight" in k and "ema_model" in k
        }

        # weight shape: [512, 512]

        # for k, v in modulate.items():
        #     print(f"Key: {k}, Shape: {v.shape}")
        # pdb.set_trace()

        weight_mat = []
        eigvec_ = {}
        for k, v in modulate.items():
            # weight_mat.append(v)
            eigvec = torch.svd(v).V.to("cpu")
            eigvec_[k] = eigvec
        
        print(eigvec_)

        torch.save(eigvec_, args.out)
