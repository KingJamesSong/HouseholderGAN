# LR-HOP 在 FFHQ128 / DiffAE 上的 PPL、PIPL 评估

本文记录现有代码实际采用的评估方式，以及已完成的 50,000-step、ε=10⁻⁴ 实验。论文中方法名称为 **LR-HOP**；代码中的 `--method hop` 和目录中的 `hop` 是同一方法的内部标识。

本评估使用 FFHQ 测试集中的 2,000 张图像，与 re-scoring analysis 使用的 20 张图像是不同实验。PPL 和 PIPL 都越小越好，衡量的是所定义扰动下的感知变化，不直接衡量属性准确性或解耦。

## 1. 代码和权重

服务器工作目录：

```text
/nfs/data_chaos/czhang/HouseholderGAN/diffae
```

主要实现：

| 文件 | 功能 |
|---|---|
| `evaluate_ffhq128_metrics.py` | 加载模型和数据；构建配对；提取方向；LPIPS 与截尾统计 |
| `generate_ffhq128_metric_samples.py` | 历史流程：顺序生成 FID 图像、PPL 对、PIPL 对 |
| `generate_ffhq128_path_samples.py` | 新版流程：只生成 PPL/PIPL 对，各指标开始前重置随机种子 |
| `score_ffhq128_saved_samples.py` | 对已保存的浮点图像对计算 LPIPS；包含 `score_pairs` |
| `score_ffhq128_path_samples.py` | 校验新版路径样本元数据，然后计算 PPL/PIPL |
| `model/unet_autoenc.py` | 将 PIPL 扰动注入选定区域的条件投影输入 |

已核对的 50k 权重：

```text
checkpoints/ffhq128_householder_random_single4090/checkpoints/step_050000.ckpt
```

模型基于 `ffhq128_autoenc_130M()` 构建，checkpoint 模式严格加载 `state_dict`；推理使用 `ema_model.eval()`，编码使用 `ema_model.encoder`，不使用在线训练参数。LR-HOP 启用区域条件投影；带 time/style MLP 的权重需要 `--use-time-style-mlp`，须与训练架构一致。

`--steps 50000` 是评分输出中的标签，不会切换权重或读取 checkpoint 内部 step。真正加载的权重由生成阶段的 `--checkpoint` 决定，必须核对 `metadata.json` 中的 `source_checkpoint`。

## 2. 评估配置

| 配置 | 本实验设置 |
|---|---|
| 数据 | FFHQ LMDB，`split='test'`，`do_augment=False` |
| 数据路径 | `/data/jichao/tcsvt/HouseholderGAN/diffae/datasets` |
| 样本顺序 | 固定 `test[0:2000]`，不 shuffle、不 drop last |
| 图像尺寸 | 128 × 128 |
| PPL / PIPL 图像对数 | 各 2,000 对，即各 4,000 张生成图像 |
| 随机种子 | 0；设置 Python、NumPy、PyTorch 和 CUDA 随机种子 |
| 生成 batch size | 16 对；PPL 编码时拼接 32 张输入图像 |
| DDIM 步数 | 100 |
| ε | 主实验显式指定 `1e-4` |
| PPL 采样 | `end`，插值位置固定为 t=0 |
| PIPL 区域 | `enc`、`mid`、`dec` |
| 每区域方向数 | 10，共 30 个区域–方向组合 |
| 感知网络 | `lpips.LPIPS(net='vgg').eval()` |
| 图像对保存 | CPU float32 `.pt`，不经过 PNG/JPEG 量化 |
| 统计 | 对 ε² 归一化后的距离做第 1/99 百分位截尾，报告均值 |

图像对中的两张图共用同一初始高斯噪声，避免把独立噪声变化混入路径变化。这里从随机噪声条件生成，不对每张真实图像执行 DDIM inversion。

ε 必须显式传入：不同脚本的默认值并不统一，新版路径脚本默认 `1e-2`，不能依赖默认值复现 `1e-4` 结果。

## 3. PPL：随机配对条件码之间的局部插值

### 3.1 配对

对前 2,000 张测试图，用 `np.random.default_rng(0)` 生成随机排列。排列中的每个索引与前一个索引配对，形成循环；每张图像恰好作为起点和终点各出现一次，且没有自身配对。

新版元数据保存完整 `ppl_partner_indices`，用于核对不同 checkpoint 的配对是否一致。

### 3.2 图像生成

设两张真实图像的编码为 z₁、z₂，n 为共享的初始高斯噪声，G 为 100 步条件 DDIM 生成器：

```text
z(t) = (1 − t) z₁ + t z₂
x₀ = G(n, z(t))
x₁ = G(n, z(t + ε))
```

使用 `torch.lerp` 线性插值，不使用球面插值，也不对 z₁、z₂做单位化。当前 `end` 配置固定 t=0，因此比较：

```text
x₀ = G(n, z₁)
x₁ = G(n, z₁ + ε(z₂ − z₁))
```

每对的 PPL 值为：

```text
d_PPL = LPIPS(x₀, x₁) / ε²
```

最终对 2,000 个值截尾并取均值。该协议评估编码条件空间中插值端点附近的平滑性，不能描述为全路径均匀采样的 PPL。

## 4. PIPL：沿区域方向的局部扰动

### 4.1 方向来源

分别从 EMA 模型的 `style_enc`、`style_mid`、`style_dec` 区域投影提取方向。Householder 投影的有效矩阵按代码重建为：

```text
W = H(U) S H(V)
```

在 CPU float32 上对 W 做 SVD。`torch.linalg.svd` 返回的第三项是 Vh，其前若干行是所使用的右奇异向量。每个向量具有单位 L2 范数。

有效秩阈值为 `max(singular_values) × 1e-6`，每区域最多取前 10 个有效方向。已完成的 50k 实验每区域均有 10 个方向。方向来自投影矩阵分解，不是 re-scoring 中按 Pose、Age、Gender 等标签挑选的方向。

### 4.2 扰动位置

设 z 为真实图像编码、r 为选定区域、v 为该区域方向：

```text
x_base = G(n, z)
x_edit = G_r(n, z; εv)
d_PIPL = LPIPS(x_base, x_edit) / ε²
```

其中 G_r 表示启用区域方向注入的生成过程。实际扰动加在模型内部的 `cond_emb` 上、选定 `style_r` 投影之前；另外两个区域使用未扰动条件。对于带 time/style MLP 的模型，不能把实现简单写成直接修改原始编码 z 后再送入所有区域。

基准图和编辑图使用相同初始噪声，保持 DDIM 步数及其他配置一致。使用单边差分：0 与 +ε，不使用 −ε / +ε 中心差分。

### 4.3 样本如何分配给方向

2,000 张图像按 batch size=16 分为 125 个 batch。每个 batch 使用一个区域–方向组合，依次循环：

```text
enc:0, ..., enc:9, mid:0, ..., mid:9, dec:0, ..., dec:9
```

**PIPL 总共是 2,000 对，不是 2,000 × 30 对，也不是同一张图评估所有方向。**

当前分配为：

- `enc:0` 至 `enc:4`：每组 80 对。
- 其余 25 组：每组 64 对。
- 合计：5 × 80 + 25 × 64 = 2,000 对。

总体 PIPL 将所有 2,000 个距离合并后截尾、取均值；它不是 30 个组均值的等权平均。`per_group` 另存每组单独截尾的统计结果。跨方法或 checkpoint 比较时必须保持 batch size、图像顺序和方向分配一致。

## 5. LPIPS 输入和统计

`model.render` 将扩散模型输出转换为图像空间。评分时按实现将保存的浮点图像乘 2 再减 1，送入 VGG-LPIPS；不额外执行人脸对齐、裁剪、缩放或 PNG 保存后重读，也不额外 clamp 评分图像。

先对每对 LPIPS 距离除以 ε²，再合并所有距离。计算第 1 和第 99 百分位 q₁、q₉₉，保留满足以下条件的值：

```text
q₁ ≤ dᵢ ≤ q₉₉
```

报告保留值的 `mean` 和 `std`；`std` 是 NumPy 默认总体标准差，不是标准误或置信区间。保存 `raw_count`、`count`、`trim_lower`、`trim_upper`。由于边界值包含在内且可能存在重复值，保留数量不一定恰好为 1,960。

ε=10⁻⁴ 时，ε²=10⁻⁸，数值差异会被放大。比较时要固定 ε、图像存储精度、LPIPS 版本和生成协议。随机种子固定不等于跨 GPU、软件版本的逐位一致。

## 6. 已完成的 50k / ε=10⁻⁴ 结果

历史样本及结果目录：

```text
/data/jichao/tcsvt/HouseholderGAN/diffae/output/ffhq128_hop_50k_metric_samples
```

下列数值已从该目录的 `metadata.json` 和 `metrics.json` 核对：

| 指标 | 原始均值 | 原始标准差 | 原始对数 | 截尾后对数 |
|---|---:|---:|---:|---:|
| PPL | 310.6407470703125 | 449.8682556152344 | 2,000 | 1,960 |
| PIPL | 21.142597198486328 | 39.47710037231445 | 2,000 | 1,980 |

结果文件记录原始分数，没有额外乘除 10。若论文表格明确采用 scaling factor=10，即显示原始值 / 10，则该实验显示为 **PPL=31.06、PIPL=2.11**。显示缩放不改变指标定义，也不能混用不同 ε、权重或架构的分数。

## 7. 复算历史保存样本

以下操作只对已保存的浮点图像对计算指标，不生成新样本、不提交 Slurm。需要 CUDA 环境及已有 LPIPS 权重：

```bash
cd /data/jichao/tcsvt/HouseholderGAN/diffae
source /data/jichao/miniforge3/etc/profile.d/conda.sh
conda activate diffae6000d
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python - <<'PY'
import json
from pathlib import Path
import lpips
import torch
from evaluate_ffhq128_metrics import seed_everything
from score_ffhq128_saved_samples import score_pairs

sample_dir = Path('output/ffhq128_hop_50k_metric_samples')
metadata = json.loads((sample_dir / 'metadata.json').read_text())
assert metadata['epsilon'] == 1e-4
assert metadata['samples'] == 2000
assert metadata['method'] == 'hop'
seed_everything(0)
device = torch.device('cuda:0')
perceptual_model = lpips.LPIPS(net='vgg').to(device).eval()
results = {
    'source_checkpoint': metadata['source_checkpoint'],
    'sample_metadata': metadata,
}
for metric in ('ppl', 'pipl'):
    results[metric] = score_pairs(
        sample_dir, metadata, metric, device, perceptual_model)
(sample_dir / 'path_metrics_recomputed.json').write_text(
    json.dumps(results, indent=2))
print({k: results[k]['mean'] for k in ('ppl', 'pipl')})
PY
```

`score_pairs` 会写出或更新 `ppl_distances.npy` 和 `pipl_distances.npy`；上述示例将汇总保存为新的 `path_metrics_recomputed.json`，不替换历史 `metrics.json`。

## 8. 新版仅生成路径样本的运行方式

以下采用明确的 ε=10⁻⁴ 和独立输出目录，先生成样本，再评分。生成目录必须不存在或为空；重复实验请更换目录。

```bash
cd /data/jichao/tcsvt/HouseholderGAN/diffae
source /data/jichao/miniforge3/etc/profile.d/conda.sh
conda activate diffae6000d
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_MODE=disabled
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python -u generate_ffhq128_path_samples.py \
  --mode checkpoint \
  --checkpoint checkpoints/ffhq128_householder_random_single4090/checkpoints/step_050000.ckpt \
  --data-path datasets \
  --output-dir output/lr_hop_50k_path_eps1e4_reproduction \
  --method hop --use-time-style-mlp \
  --seed 0 --device cuda:0 --num-workers 0 --batch-size 16 \
  --ddim-steps 100 --samples 2000 --epsilon 1e-4 \
  --ppl-sampling end --pipl-directions 10

python -u score_ffhq128_path_samples.py \
  --sample-dirs output/lr_hop_50k_path_eps1e4_reproduction \
  --steps 50000 --method hop --epsilon 1e-4 --device cuda:0 \
  --output-dir output/lr_hop_50k_path_eps1e4_reproduction_metrics
```

**新版与历史流程的随机数顺序不同。** 历史 `generate_ffhq128_metric_samples.py` 在设 seed 后依次生成 FID、PPL、PIPL，后续噪声沿用前一阶段消耗后的 RNG 状态；新版 `generate_ffhq128_path_samples.py` 在 PPL 和 PIPL 各自开始前重新设 seed。因此上述新版命令实现相同指标配置，但不能承诺精确复现第 6 节历史数值。需要复算历史数值时使用第 7 节已有样本；需要重新比较多个 checkpoint 时统一使用同一生成流程。

历史元数据缺少新版要求的 `mode`、`noise_seed_per_metric`、`ppl_partner_indices` 等字段，不能直接交给新版严格校验 CLI。不要补造字段使它通过验证。

## 9. 输出与核验

新版生成目录包含：

```text
metadata.json
ppl_pairs/0000.pt ... 0124.pt     # start、end
pipl_pairs/0000.pt ... 0124.pt    # base、edited、region、方向编号
```

评分后还包含：

```text
ppl_distances.npy                # 2,000 个 ε² 归一化后的原始距离
pipl_distances.npy
metrics.json
```

独立评分输出目录包含 `metrics_summary.json` 和 `metrics_table.md`。新版 CLI 可先加 `--validate-only` 校验元数据及文件结构，不加载模型执行感知评分。

比较结果前核对：实际 checkpoint、EMA 和模型架构、ε、固定图像及 PPL 配对、噪声 RNG 流程、DDIM 步数、batch size、PIPL 方向及组计数、LPIPS 骨干和截尾规则。PPL/PIPL 的 ε 是局部差分尺度，与语义遍历图中的 α 或 traversal 范围不是同一参数。
