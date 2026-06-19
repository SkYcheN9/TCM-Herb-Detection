# CBAM Diff 说明

本次只实现 CBAM 注意力模块与可选训练入口，不包含 BiFPN、Focal Loss、GhostConv 等其他改进。

## 修改内容

- `models/modules/cbam.py`
  - 新增 `ChannelAttention`、`SpatialAttention`、`CBAM`。
  - 新增 `register_ultralytics_modules()`，用于把自定义 `CBAM` 暴露给 Ultralytics YAML 解析器。
- `models/yolov8n_cbam.yaml`
  - 基于 YOLOv8n 结构新增 CBAM 版本。
  - 在 backbone 的 C2f 后插入 CBAM，保持输出特征尺寸不变。
- `configs/cbam.yaml`
  - 新增 CBAM 训练配置，设置 `enable_cbam: true`，默认输出到 `runs/cbam`。
- `configs/baseline.yaml`
  - 新增 `enable_cbam: false`，明确 Baseline 不启用 CBAM。
- `scripts/train_baseline.py`
  - 支持 `--config` 加载 YAML 配置。
  - 支持 `--enable-cbam true/false`。
  - 开启 CBAM 时先注册自定义模块，再交给 Ultralytics 构建模型。
  - 保留原 Baseline 默认行为：不传配置时仍使用 `yolov8n.pt`，输出到 `runs/baseline`。
- `tests/test_cbam.py`
  - 覆盖 CBAM 输入输出 shape 不变与反向传播。
  - 覆盖 Ultralytics 能从 `models/yolov8n_cbam.yaml` 构建包含 CBAM 的模型。
- `.gitignore`
  - 忽略 Ultralytics 自动生成的 `dataset/labels/*.cache`。

## 使用方式

Baseline：

```bash
.\.venv\Scripts\python.exe train.py --config configs/baseline.yaml
```

CBAM：

```bash
.\.venv\Scripts\python.exe train.py --config configs/cbam.yaml
```

## 兼容性说明

- CBAM 是 shape-preserving 模块，不改变 YOLOv8 检测头输入尺度。
- `enable_cbam` 默认为关闭；只有配置文件或命令行显式启用时才加载 CBAM YAML。
- Baseline 训练路径仍可直接使用 `train.py` 或 `scripts/train_baseline.py`。
