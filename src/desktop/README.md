# TCM-SliceAI Desktop Client

桌面客户端使用 PySide6 和 qfluentwidgets 构建，面向 PC 本地检测、历史记录、药材计数和结果导出。当前 11 组最终消融实验已经完成，桌面端默认使用 `Baseline+CBAM+BiFPN`。

## 启动

```powershell
.\.venv\Scripts\python.exe run_desktop.py
```

也可以使用模块入口：

```powershell
.\.venv\Scripts\python.exe -m src.desktop
```

## 当前模块

- Dashboard：工作站状态、指标卡片、模型能力概览
- Detection：USB Camera、Laptop Camera、图片、视频检测，显示 FPS、类别、数量和 GPU 状态
- History：检测记录、筛选与 Excel 导出
- Settings：运行设备、路径、界面和检测行为配置
- About：项目说明、固定类别和当前范围

## 依赖

```powershell
.\.venv\Scripts\python.exe -m pip install -r src\desktop\requirements-desktop.txt
```

当前 Windows 环境已验证 `PySide6==6.8.3` 能正常导入和创建主窗口。

## 摄像头检测

Detection 页面会自动探测常见摄像头索引，优先使用最终部署模型：

1. `final_results_full/reports/ablation/runs/baseline_cbam_bifpn/weights/best.pt`
2. `final_results_full/reports/ablation/runs/baseline_cbam/weights/best.pt`
3. `final_results_full/reports/ablation/runs/baseline_ghostconv/weights/best.pt`

如果这些文件不存在，会按内置候选路径回退到其他 `best.pt` 或本地 YOLO 权重。

实时检测支持：

- Laptop Camera
- USB Camera
- 图片检测
- 视频检测
- FPS 显示
- 药材计数
- 总数量
- CUDA/GPU 状态显示
- 结果保存到 `reports/desktop`
- 历史记录保存到 `src/desktop/data/history.db`
- Excel 导出到 `reports/desktop/exports`

## 最终部署口径

- 桌面端默认模型：`Baseline+CBAM+BiFPN`，mAP50-95 为 0.80076，兼顾精度和速度。
- 最高精度参考：`Baseline+CBAM`，mAP50-95 为 0.80125。
- 树莓派端模型：`Baseline+GhostConv`，mAP50-95 为 0.79822，更适合 CPU/OpenVINO 轻量部署。
- Focal Loss 与 FullModel 已完成实验，但作为负向消融结论保留，不作为部署模型。
