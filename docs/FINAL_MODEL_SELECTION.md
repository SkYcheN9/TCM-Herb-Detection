# 最终模型选型说明

最终消融阶段共训练 11 个模型，所有模型均使用同一份增强数据集，并采用 `yolov8n.pt` 进行公平预训练迁移。

| 排名 | 模型 | mAP50 | mAP50-95 | Precision | Recall | FPS | 结论 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Baseline+CBAM | 0.99227 | 0.80125 | 0.99425 | 0.99366 | 242.11 | 最高精度参考 |
| 2 | Baseline+CBAM+BiFPN | 0.99162 | 0.80076 | 0.99161 | 0.99051 | 302.73 | Web/桌面端默认部署 |
| 3 | Baseline+GhostConv | 0.98915 | 0.79822 | 0.98803 | 0.98988 | 306.11 | 树莓派端默认部署 |
| 4 | Baseline | 0.99118 | 0.79532 | 0.99332 | 0.99495 | 304.50 | 基线对照 |
| 5 | Baseline+DecoupledHead | 0.99133 | 0.79500 | 0.99246 | 0.99455 | 235.72 | 不作为部署模型 |
| 6 | Baseline+CBAM+BiFPN+GhostConv+DecoupledHead | 0.99026 | 0.79446 | 0.97997 | 0.98383 | 280.49 | 候选组合，不部署 |
| 7 | Baseline+BiFPN | 0.99251 | 0.79234 | 0.99197 | 0.99362 | 299.28 | 不作为部署模型 |
| 8 | Baseline+CBAM+BiFPN+GhostConv | 0.98853 | 0.79027 | 0.98294 | 0.98299 | 287.42 | 候选组合，不部署 |
| 9 | Baseline+CBAM+BiFPN+Focal | 0.99027 | 0.78909 | 0.98482 | 0.98622 | 198.90 | Focal 负向消融 |
| 10 | FullModel | 0.99012 | 0.78472 | 0.97765 | 0.97535 | 226.29 | 不作为部署模型 |
| 11 | Baseline+Focal | 0.98906 | 0.78438 | 0.98285 | 0.98031 | 235.14 | Focal 负向消融 |

## 部署选择

- Web 端和桌面端默认模型：`Baseline+CBAM+BiFPN`
- 最高精度参考模型：`Baseline+CBAM`
- Raspberry Pi 5 8GB 无算力棒模型：`Baseline+GhostConv`

`Baseline+CBAM+BiFPN` 的 mAP50-95 仅比最高精度的 `Baseline+CBAM` 低 0.00049，但 FPS 从 242.11 提升到 302.73，因此更适合作为 PC 本地端和网页端默认模型。

`Baseline+GhostConv` 在 11 个模型中速度最高，mAP50-95 仍达到 0.79822，因此更适合树莓派 5 无算力棒场景，后续应优先导出 ONNX/OpenVINO/NCNN 后在树莓派上实测。

Focal Loss 已完成损失函数优化实验，但 `Baseline+Focal`、`Baseline+CBAM+BiFPN+Focal` 和 `FullModel` 均降低 mAP50-95、Precision、Recall 或 FPS，因此论文中应将其作为负向消融结果分析，而不是作为最终部署模型。
