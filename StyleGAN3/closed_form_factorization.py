import argparse

import torch
import os
from training.model_ortho import Q
import legacy
import dnnlib
from torch_utils import misc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract factor/eigenvectors of latent spaces using closed form factorization"
    )

    parser.add_argument(
        "--out", type=str, default="factor.pt", help="name of the result factor file"
    )
    parser.add_argument("--resume_pkl", type=str, help="name of the model checkpoint")

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

        with dnnlib.util.open_url(args.resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)

        # weight_mat =  []
        U = None
        V = None

        eigvec_ = {}

        named_parameters = list(resume_data['G_ema'].named_parameters()) + list(resume_data['G_ema'].named_buffers())

        for k, v in named_parameters:

            if '.U' in k:
                U = v

            if '.V' in k:
                V = v

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
        with dnnlib.util.open_url(args.resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)

        named_parameters = list(resume_data['G_ema'].named_parameters()) + list(resume_data['G_ema'].named_buffers())

        modulate = {
            k: v
            for k, v in named_parameters
            if "synthesis" in k and "affine" in k and "input" not in k and "weight" in k
        }

        weight_mat = []
        eigvec_ = {}
        count = 0
        for k, v in modulate.items():

            if count % 4 == 0:
                # weight_mat.append(v)
                eigvec = torch.svd(v).V.to("cpu")
                eigvec_[k] = eigvec
            count += 1

        torch.save(eigvec_, args.out)

