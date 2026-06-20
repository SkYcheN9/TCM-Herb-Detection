# 任务书验收对照清单

本文档根据 `docs/基于改进 YOLOv8 的中医药饮片智能检测与识别系统实践项目任务书.docx` 梳理当前实现状态。

## 数据集构建与增强

| 要求 | 状态 | 对应实现 |
| --- | --- | --- |
| LabelImg/YOLO 标准格式 | 已实现 | `scripts/check_dataset.py`、`src/tcm_slice_ai/dataset.py` |
| 固定 15 类类别顺序 | 已实现 | `dataset/classes.txt`、`dataset/data.yaml`、`src/tcm_slice_ai/constants.py` |
| 缺失标注、空标注、越界类别、bbox 格式检查 | 已实现 | `scripts/check_dataset.py` |
| 训练/验证集 7:3 或 8:2 划分 | 已实现 | `scripts/prepare_dataset.py`，当前 771/193，约 8:2 |
| 模糊、无效、重复图像清洗 | 已补充 | `scripts/clean_dataset.py` 检查可读性、分辨率、模糊度、精确重复、近似重复和标签问题 |
| Albumentations 离线增强 | 已补充 | `scripts/augment_dataset.py` |
| Mosaic、MixUp、HSV、随机裁剪 | 已补充 | `scripts/augment_dataset.py` 支持 Mosaic、MixUp、HueSaturationValue、RandomCropFromBorders |
| RandomBrightnessContrast、HorizontalFlip、CLAHE | 已补充 | `scripts/augment_dataset.py` |
| 正式增强数据集 | 已生成 | `dataset_augmented/data.yaml`，训练集 1942 张，验证集 193 张 |

## 系统功能要求

| 要求 | 状态 | 对应实现 |
| --- | --- | --- |
| 多目标密集检测 | 已实现基础能力 | YOLOv8 检测头、后端/桌面端检测服务均返回多实例结果 |
| bbox、类别、置信度输出 | 已实现 | `backend/schemas.py`、`backend/services/detector.py` |
| 自动计数 | 已实现 | 后端 `class_counts`/`total_count`，桌面端和 Web 端均展示统计 |
| 图片检测 | 已实现 | 后端 `/detect/image`、桌面端、Web 端 |
| 视频检测 | 已实现 | 后端 `/detect/video`、桌面端视频检测 |
| 摄像头检测 | 已实现 | 桌面端摄像头线程、Web 端 getUserMedia 抓拍检测 |
| 批量检测 | 已实现 | 后端 `/detect/batch`，桌面端批量入口已有服务层支持 |
| RTSP 检测 | 部分实现 | 底层 OpenCV/Ultralytics 可接流，但尚未提供独立 API 参数和 UI 入口 |
| CPU/GPU 兼容 | 已实现 | 训练和推理均支持 `auto/cpu/cuda` |
| 实时推理 FPS 展示 | 已实现 | 桌面端摄像头/视频 FPS、Web 端耗时、`benchmark.py` |

## 算法设计与创新改进

任务书要求“任选一项及以上”实质性改进，当前 5 个方向均已完成代码落地，不属于仅调参。

| 改进方向 | 状态 | 对应实现 |
| --- | --- | --- |
| YOLOv8 Baseline | 已实现 | `train.py`、`configs/baseline.yaml` |
| 注意力机制 CBAM | 已实现 | `models/modules/cbam.py`、`models/yolov8n_cbam.yaml` |
| BiFPN 特征融合 | 已实现 | `models/modules/bifpn.py`、`models/yolov8n_bifpn.yaml`、`models/yolov8n_cbam_bifpn.yaml` |
| Focal Loss | 已实现 | `models/losses/focal_loss.py`、`scripts/trainers.py` |
| GhostConv 轻量化骨干 | 已实现 | `models/yolov8n_ghost.yaml`、`configs/ghostconv.yaml`，替换部分 Backbone 下采样 Conv |
| CIoU Loss 改进 | 未实现 | 已选择 Focal Loss 满足损失函数优化方向 |
| Decoupled Head | 已实现 | `models/modules/decoupled_head.py`、`models/yolov8n_decoupled.yaml`、`configs/decoupled_head.yaml` |
| FullModel 五项改进组合 | 已实现 | `models/yolov8n_full.yaml`、`configs/full_model.yaml` |

## 消融实验与报告素材

| 要求 | 状态 | 对应实现 |
| --- | --- | --- |
| Baseline 消融项 | 已实现 | `scripts/ablation.py` |
| Baseline+CBAM | 已实现 | `scripts/ablation.py` |
| Baseline+CBAM+BiFPN | 已实现 | `scripts/ablation.py` |
| Baseline+CBAM+BiFPN+Focal | 已实现 | `scripts/ablation.py` |
| FullModel | 已实现 | `configs/full_model.yaml` |
| mAP50、mAP50-95、Precision、Recall、FPS | 已实现 | `reports/ablation/summary.csv` / `summary.xlsx` |
| CSV、Excel 导出 | 已实现 | `scripts/ablation.py` |
| PR、Loss、mAP 曲线 | 已实现 | `scripts/ablation.py` |
| 混淆矩阵 | 已实现 | Ultralytics `plots=True`，`evaluate.py` 与 `ablation.py` 会生成 |
| 类别统计图 | 部分实现 | 后端/前端/桌面端提供类别统计；独立离线统计图脚本尚未单独拆分 |

正式 5 组 CUDA 消融训练已完成，输出位于 `reports/ablation`。当前精度最高的是 Baseline：`mAP50=0.94706`、`mAP50-95=0.74452`；CBAM 与 CBAM+BiFPN 均保持 `mAP50 > 0.92`；Focal Loss 与 FullModel 在当前 100 epoch 和默认 Focal 参数下欠收敛，作为后续优化方向保留。

## 自动化脚本

| 脚本 | 状态 |
| --- | --- |
| `train.py` | 已实现 |
| `export.py` | 已实现，支持 `.pt`、`.onnx`、`.torchscript`、OpenVINO、NCNN |
| `benchmark.py` | 已补充，PC 侧 CPU/GPU FPS 测试 |
| `evaluate.py` | 已补充，验证并导出指标与图表 |
| `ablation.py` | 已实现 |
| `deploy_pi.py` | 已补充，树莓派部署打包 |

## 部署与界面

| 要求 | 状态 | 对应实现 |
| --- | --- | --- |
| FastAPI 后端 | 已实现 | `backend/main.py` |
| SQLite 开发数据库 | 已实现 | `backend/database.py` |
| History/statistics/detect 模块 | 已实现 | `backend/routers/` |
| PC 桌面端 PySide6 | 已实现 | `src/desktop/` |
| Dashboard/Detection/History/Settings/About | 已实现 | `src/desktop/app/views/` |
| Web: Next.js + TypeScript + TailwindCSS | 已实现 | `src/frontend/` |
| Web: 摄像头 getUserMedia | 已实现 | `src/frontend/app/detect/workbench.tsx` |
| Web: 响应式布局 | 已实现基础 | `src/frontend` 页面使用响应式 Tailwind 布局 |
| Raspberry Pi 摄像头检测 | 已实现 | `deployment/raspberry_pi/pi_camera_web.py` |
| Raspberry Pi ONNX/OpenVINO/NCNN 导出 | 已补充 | `export.py`、`deploy_pi.py` |

## 仍需现场或后续实验确认的事项

1. 复杂干扰样本现场验收效果需要教师提供新样本后验证。
2. FullModel 已完成完整训练，但当前结果未达到 `mAP50 >= 90%`、`mAP50-95 >= 70%`；需要继续调 Focal 参数、预训练策略或更长训练。
3. CPU、树莓派 FPS 目标需要在对应设备实测，`benchmark.py` 和 `benchmark_pi.py` 已提供测试入口。
4. RTSP 独立接口和 UI 入口尚未专门封装，若验收明确要求 RTSP，需要继续补 API 与界面。
