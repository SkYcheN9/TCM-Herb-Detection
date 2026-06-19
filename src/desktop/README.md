# TCM-SliceAI Desktop Client

桌面客户端使用 PySide6 和 qfluentwidgets 构建，当前阶段只实现目录结构和主界面，不修改训练代码与模型代码。

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

Detection 页面会自动探测常见摄像头索引，优先使用 `runs/baseline/weights/best.pt` 作为当前最佳模型。如果该文件不存在，会按内置候选路径回退到其他 `best.pt` 或本地 YOLO 权重。

实时检测支持：

- Laptop Camera
- USB Camera
- 图片检测
- 视频检测
- FPS 显示
- 类别计数
- 总数量
- CUDA/GPU 状态显示
- 结果保存到 `reports/desktop`
- 历史记录保存到 `src/desktop/data/history.db`
- Excel 导出到 `reports/desktop/exports`

## 后续接入点

后续检测功能建议在 `src/desktop/app/services/` 中封装 Detector 适配层，再由 `DetectionView` 调用，保持训练脚本和模型定义独立。
