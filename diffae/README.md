# Householder Projector for Unsupervised Latent Semantics Discovery of GANs and Diffusion

## Abstract

Structured latent or conditioning representations are crucial for controllable visual generation, as they enable compact codes to induce interpretable and precise variations through intermediate features. Latent semantics discovery aims to identify human-understandable directions from such representations. A recent projector-based paradigm directly uses the principal directions of the projection matrix that maps latent or conditioning codes to feature parameters. However, the projection matrix in pretrained models is generally non-orthogonal, causing dominant directions to entangle multiple semantic attributes. A natural remedy is to introduce orthogonality into the projector, but directly enforcing full orthogonality over a high-dimensional projection matrix may scatter meaningful variations across too many directions, resulting in imperceptible or semantically meaningless traversals. To address this dilemma from a unified latent-to-feature conditioning perspective, we propose a Low-Rank Householder Orthogonal Projector (LR-HOP), which parameterizes the projection matrix with a flexible and general low-rank orthogonal matrix representation based on Householder transformations. The proposed design preserves orthogonality while explicitly controlling the number of effective semantic factors, leading to more balanced, interpretable, and controllable directions. To integrate the proposed module into pretrained visual backbones, we further introduce a nearest-orthogonal initialization followed by lightweight fine-tuning, which enables stable adaptation with marginal architectural modification. Extensive experiments on style-based generators, 3D-aware backbones, and diffusion autoencoders demonstrate that Householder Projector discovers more disentangled semantic directions and achieves more precise controllable variations across diverse generative architectures. Our code is publicly available at [GitHub](https://github.com/KingJamesSong/HouseholderGAN).

## Setup

Create the environment as described in the [repository README](../README.md#environment). Run the commands below from the `diffae/` directory. Replace bracketed placeholders with your own paths.

## Training on FFHQ

```bash
python run_ffhq128.py
```

## Test on FFHQ

```bash
python closed_form_factorization.py --out [factor_path] [checkpoint_path] --is_ortho
python apply_factor.py --output_dir [output_path] --ckpt [checkpoint_path] --factor [factor_path] --size 128
```

## Evaluation (PPL/PIPL)

```bash
python ppl.py --ckpt [checkpoint_path] --sampling full --eps 1e-1 --size 128
python pipl.py --ckpt [checkpoint_path] --factor [factor_path] --sampling full --eps 1e-1 --size 128
```

Set `is_ortho=True` in the model configuration before evaluation.

## Model Weights

| Dataset | Backbone | Resolution | Fine-tuned model | Pre-trained model |
| --- | --- | --- | --- | --- |
| FFHQ | DiffAE | 128x128 | [Download](https://drive.usercontent.google.com/download?id=1Fwc9hdgUWnYXnbhceUhZlYviUTYDBjdu&export=download&authuser=0) | [Download](https://drive.google.com/drive/folders/11pdjMQ6NS8GFFiGOq3fziNJxzXU1Mw3l) |

## Acknowledgement

The DiffAE backbone is based on [Diffusion Autoencoders: Toward a Meaningful and Decodable Representation](https://openaccess.thecvf.com/content/CVPR2022/html/Preechakul_Diffusion_Autoencoders_Toward_a_Meaningful_and_Decodable_Representation_CVPR_2022_paper.html).
