# Raspberry Pi 5 部署方案

目标设备：Raspberry Pi 5 8GB  
目标性能：实时摄像头识别尽量接近或超过 10FPS  
推荐模型：`Baseline+GhostConv`，其 11 组消融中速度最高，mAP50-95 为 0.79822  
推荐运行格式：OpenVINO 优先，ONNX 备用，PyTorch 只做基线对比

## 1. 导出模型

在训练电脑的项目根目录执行：

```bash
python export.py
```

默认会优先导出最终轻量化模型：

```text
final_results_full/reports/ablation/runs/baseline_ghostconv/weights/best.pt
```

默认生成：

```text
best.pt
best.onnx
best_openvino/
export_manifest.json
```

默认导出尺寸为 `416`。这是给 Raspberry Pi 5 无算力棒部署的实时识别档位；如果精度优先可改为 `640`，如果 FPS 不足可改为 `320`：

```bash
python export.py --imgsz 320
```

复制到树莓派：

```bash
scp best.pt best.onnx -r best_openvino deployment/raspberry_pi benchmark_pi.py pi@raspberrypi.local:~/tcm-sliceai/
```

## 2. Benchmark

树莓派建议使用 Raspberry Pi OS 64-bit Bookworm，先更新系统并确认摄像头：

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0 libatlas-base-dev libopenblas0 \
  rpicam-apps python3-picamera2
rpicam-hello --list-cameras
```

创建环境：

```bash
cd ~/tcm-sliceai
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r deployment/raspberry_pi/requirements-pi.txt
```

执行三格式测速：

```bash
python benchmark_pi.py --imgsz 416 --warmup 10 --iterations 100
```

输出：

```text
reports/pi_benchmark.csv
```

验收标准：

```text
OpenVINO FPS >= 10
ONNX 可运行并记录 FPS
PyTorch 可运行并作为 CPU 基线
```

如果 OpenVINO 低于 10FPS：

```bash
python benchmark_pi.py --imgsz 320 --warmup 10 --iterations 100
```

## 3. 摄像头实时识别与局域网访问

推荐用 OpenVINO 启动：

```bash
source .venv/bin/activate
python deployment/raspberry_pi/pi_camera_web.py \
  --model best_openvino \
  --backend openvino \
  --imgsz 416 \
  --camera auto \
  --host 0.0.0.0 \
  --port 8000
```

访问地址：

```text
树莓派本机：http://127.0.0.1:8000
局域网设备：http://树莓派IP:8000
状态接口：http://树莓派IP:8000/api/status
视频流：http://树莓派IP:8000/stream
```

查看树莓派 IP：

```bash
hostname -I
```

USB 摄像头可指定：

```bash
python deployment/raspberry_pi/pi_camera_web.py --model best_openvino --backend openvino --camera 0
```

ONNX 备用：

```bash
python deployment/raspberry_pi/pi_camera_web.py --model best.onnx --backend onnx --imgsz 416
```

PyTorch 基线：

```bash
python deployment/raspberry_pi/pi_camera_web.py --model best.pt --backend pytorch --imgsz 416
```

## 性能策略

推荐档位：

| 场景 | 后端 | imgsz | 目标 |
| --- | --- | --- | --- |
| 验收实时识别 | OpenVINO | 416 | 接近或超过 10FPS |
| FPS 优先 | OpenVINO | 320 | 15FPS 目标 |
| 精度优先 | OpenVINO | 640 | 低 FPS，高召回 |
| 对比测试 | ONNX | 416 | 备用结果 |
| 基线测试 | PyTorch | 416 | CPU 基准 |

部署选型说明：

- 网页端/桌面端默认使用 `Baseline+CBAM+BiFPN`，mAP50-95 为 0.80076，速度 302.73 FPS。
- 最高精度模型为 `Baseline+CBAM`，mAP50-95 为 0.80125。
- 树莓派 5 8G 无算力棒默认使用 `Baseline+GhostConv`，mAP50-95 为 0.79822，速度 306.11 FPS，模型更适合 CPU/OpenVINO 轻量部署。

优化顺序：

1. 优先使用 `best_openvino/`。
2. 若 FPS 不足，将 `--imgsz 416` 降到 `--imgsz 320`。
3. 摄像头采集保持 `1280x720`，推理尺寸用 `imgsz` 控制。
4. 使用主动散热，避免 Pi 5 降频。
5. 保持 `--jpeg-quality 80`，局域网画面清晰且传输压力低。

## 开机自启

创建服务文件：

```bash
sudo nano /etc/systemd/system/tcm-sliceai.service
```

写入：

```ini
[Unit]
Description=TCM-SliceAI Raspberry Pi Detection
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/home/pi/tcm-sliceai
ExecStart=/home/pi/tcm-sliceai/.venv/bin/python /home/pi/tcm-sliceai/deployment/raspberry_pi/pi_camera_web.py --model best_openvino --backend openvino --imgsz 416 --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3
User=pi

[Install]
WantedBy=multi-user.target
```

启用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tcm-sliceai
sudo systemctl status tcm-sliceai
```

## 方案结论

交付路径为：

```text
训练电脑导出 best.pt / best.onnx / best_openvino
树莓派运行 benchmark_pi.py 对 PyTorch、ONNX、OpenVINO 三种格式测速
选择 OpenVINO 作为正式部署后端
pi_camera_web.py 接入树莓派摄像头并提供本地 Web 与局域网访问
```

验收时以 `reports/pi_benchmark.csv` 和 Web 页面右上角 FPS 为准，OpenVINO 达到 10FPS+ 即通过树莓派实时部署目标。
