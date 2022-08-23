import argparse

import torch
import os
from model_ortho import Q

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
        #weight_mat =  []
        U = None
        V = None

        eigvec_ = {}

        for k, v in ckpt["g_ema"].items():
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
            for k, v in ckpt["g_ema"].items()
            if "modulation" in k and "to_rgbs" not in k and "weight" in k
        }

        weight_mat = []
        eigvec_ = {}
        for k, v in modulate.items():
            # weight_mat.append(v)
            eigvec = torch.svd(v).V.to("cpu")
            eigvec_[k] = eigvec

        torch.save(eigvec_, args.out)
    #     W = weight_mat[4]
    #     eigvec = torch.svd(W).V.to("cpu")
    #
    # print('eigvec', eigvec.shape)

    # print(args.out)
    # if not os.path.exists(args.out):
    # os.mkdir(args.out)

    # print('1')
    # torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, args.out)
    # print('2')

