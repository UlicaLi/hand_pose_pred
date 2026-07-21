# Exo RGB → Ego 2D Hand Pose

这是一个参照 EgoWorld Hand 分支思路实现的基线：**必选 exo RGB 图像**经过
ViT-B/16，再由 MLP 直接回归 ego 画面中的 `2 × 21 × 2` 个归一化关键点。
它不生成 ego 图像，也不预测 3D hand pose。

## 已实现的推荐方案

```text
exo RGB (796×448)
  └─ 保持宽高比 resize 到 384×216，padding 到 384×384
      └─ pretrained ViT-B/16 (CLS, 768-D)
          ├─ RGB MLP ────────────────────────────────┐
          └─ [可选 20-D camera vector] → camera MLP ─┤ residual logits
                                                     └─ sigmoid → [2,21,2]
```

- Backbone：torchvision `ViT-B/16`, 默认采用 384 分辨率的
  `IMAGENET1K_SWAG_E2E_V1` 预训练权重。
- RGB head：`LN(768) → Linear(512) → GELU → Dropout → Linear(84)`。
- Camera branch：exo/ego 内参、`inv(T_ego) @ T_exo` 的 `R6D+t`、三个可用性 mask，
  共 20 维；只作为 RGB logits 上的 residual。
- Camera head 最后一层零初始化，因此第二阶段起点严格等于 RGB 模型。
- 输出坐标位于 `[0,1]`，乘以 teacher NPZ 中的 ego image size（当前为 448×448）
  即得到像素坐标。
- teacher 的两个 hand slot 不是固定左右手，因此 loss 和 metric 都只枚举两种排列并取最优匹配。
- GT 只有同时满足 `hand_valid`、`keypoint_valid`、有限值和坐标位于 `[0,1]`
  才参与监督；无效点不会通过 clamp 伪装成有效 GT。
- teacher confidence 权重为
  `valid * clip((score - 0.2) / 0.8, 0, 1)`，初始版本不使用 bbox score。
- 数据增强只有颜色、噪声、轻微模糊和轻微擦除；没有 flip、rotation、crop 或透视变换。

manifest 中有部分帧没有 exo image，dataset 会自动过滤。当前可用单帧数量约为：

| split | manifest frames | 可用 exo + teacher frames |
|---|---:|---:|
| train | 122,157 | 112,647 |
| val | 15,183 | 13,995 |
| test | 15,408 | 14,022 |

## 环境

当前机器已有可用环境：

```bash
cd /home/limengfei/xingqunqi/xinyi_li/2d_hand_pose
PY=/home/limengfei/xingqunqi/xinyi_li/Exo2Ego-WanVACE-unified/.venv/bin/python
$PY -m pip install -r requirements.txt
```

首次启动 RGB 训练时，torchvision 会下载约 330 MB 的 ViT SWAG 权重。

## 推荐的两阶段训练

### Stage 1：RGB-only

RGB 分支始终独立受监督。默认冻结 backbone 4 个 epoch，随后整体微调：

```bash
CUDA_VISIBLE_DEVICES=0 $PY train.py --config configs/rgb.yaml
```

小样本连通性检查（不下载预训练权重）：

```bash
CUDA_VISIBLE_DEVICES=0 $PY train.py \
  --config configs/rgb.yaml \
  --no-pretrained \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --max-train-samples 16 \
  --max-val-samples 16 \
  --output-dir outputs/smoke_rgb
```

如显存不足，优先把 YAML 中 `batch_size` 调小，并增大 `accumulation_steps`，而不是改变输入几何。

### Stage 2：camera residual

用 Stage 1 最优 checkpoint 初始化。训练时默认采样：50% RGB-only、20% 仅内参、
30% 内参加相对外参。

```bash
CUDA_VISIBLE_DEVICES=0 $PY train.py \
  --config configs/camera.yaml \
  --init-checkpoint outputs/rgb/best.pt
```

恢复中断训练用 `--resume outputs/.../last.pt`；它会同时恢复 optimizer、scheduler 和 epoch。
两个阶段默认在 validation NME 连续 6 个 epoch 没有至少 `1e-4` 的改善时 early stop；
计数也会写入 checkpoint，恢复训练后不会丢失。

## 评估

```bash
$PY evaluate.py --checkpoint outputs/rgb/best.pt --split test
$PY evaluate.py --checkpoint outputs/camera/best.pt --split test
```

对 camera checkpoint 做严格 RGB-only ablation：

```bash
$PY evaluate.py --checkpoint outputs/camera/best.pt --split test --rgb-only
```

导出每帧原始预测（两个 slot 仍是无序的，不使用 GT 排列信息）：

```bash
$PY evaluate.py \
  --checkpoint outputs/camera/best.pt \
  --split test \
  --predictions outputs/camera/test_predictions.jsonl
```

评估输出包括 MPE（pixel）、对 ego 图像对角线归一化的 NME、PCK@0.05、
PCK@0.10、AUC@0.10，以及 wrist/palm/fingertip 分组误差；同时报告全部有效点、
teacher score ≥ 0.5 和 ≥ 0.7 三组结果。

## 单张 exo 图像推理

只输入 exo RGB（一定可用）：

```bash
$PY infer.py \
  --checkpoint outputs/rgb/best.pt \
  --exo-image /path/to/frame700.jpg \
  --output outputs/frame700_pose.json
```

输出 ego 2D 骨架可视化：

```bash
$PY infer.py \
  --checkpoint outputs/rgb/best.pt \
  --exo-image /data/limengfei/xingqunqi/Exo2Ego-dataset/takes/<take>/cam03/frame700.jpg \
  --ego-image /data/limengfei/xingqunqi/Exo2Ego-dataset/takes/<take>/aria/frame700.jpg \
  --output outputs/frame700_pose.json \
  --visualization outputs/frame700_pose.jpg
```

对于数据集标准目录，省略 `--ego-image` 时会自动查找 exo 相机同级目录中的
`<ego-camera>/<同名帧>`（默认 `aria/frame700.jpg`）；未找到时则在由
`--ego-width/--ego-height` 指定大小的空白画布上绘制。可视化中的青色和橙色分别表示
两个无序 hand slot，而不是固定的左/右手。

对 camera-stage 模型提供数据集相机参数：

```bash
$PY infer.py \
  --checkpoint outputs/camera/best.pt \
  --exo-image /data/limengfei/xingqunqi/Exo2Ego-dataset/takes/<take>/cam03/frame700.jpg \
  --pose-json /data/limengfei/xingqunqi/Exo2Ego-dataset/pose/<take>.json \
  --exo-camera cam03 \
  --ego-camera aria \
  --frame-idx 700
```

不传相机参数，或参数不可用时，模型自动退化为同一个 RGB head。输出的两个 hand slot
仍然是无序槽位；因为当前模型没有额外预测 hand presence，应用端不应把 slot 0/1
解释为固定左/右手。

## 目录

- `hand_pose/data.py`：manifest 展平、NPZ teacher、GT 过滤、letterbox、camera vector。
- `hand_pose/model.py`：ViT + RGB MLP + 可选 camera residual MLP。
- `hand_pose/losses.py`：confidence weighting 与双手两排列匹配。
- `hand_pose/metrics.py`：匹配后的 2D pose 指标。
- `train.py`：两阶段训练、warmup + cosine、AMP、checkpoint。
- `evaluate.py`：验证/测试、camera ablation、JSONL 预测导出。
- `infer.py`：单张 exo RGB 推理，可选 dataset pose JSON。
