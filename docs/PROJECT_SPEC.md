# 项目名称

TCM-SliceAI

中医药饮片智能检测与识别系统

英文名称：

Traditional Chinese Medicine Decoction Pieces Detection System Based on Improved YOLOv8

---

# 一、项目背景

本项目用于完成课程实践：

《基于改进YOLOv8的中医药饮片智能检测与识别系统》

系统需完成：

1. 中医药饮片检测
2. 分类识别
3. 自动计数
4. 实时推理
5. PC端部署
6. Web端部署
7. Raspberry Pi 5部署

最终满足：

CPU运行
GPU运行
边缘设备运行

三种环境兼容。

---

# 二、饮片类别

类别固定为15类。

类别顺序不可修改。

YOLO class.txt如下：

0 zexie
1 niuxi
2 gaoliangjiang
3 mudanpi
4 yuzhu
5 baizhi
6 baishao
7 dazao
8 danshen
9 gancao
10 baixianpi
11 baihe
12 sangzhi
13 jiegeng
14 banlangen

所有训练代码必须严格使用该顺序。

严禁自动重排类别。

---

# 三、数据集情况

当前数据规模：

约18×3×20张

≈1080张

预计存在部分缺失。

---

# 数据集目录

dataset/

    images/
        train/
        val/

    labels/
        train/
        val/

    data.yaml

---

# 数据增强

必须实现：

Albumentations

支持：

Mosaic
MixUp
RandomCrop
RandomBrightnessContrast
HueSaturationValue
HorizontalFlip
CLAHE

训练时可配置开启关闭。

---

# 四、模型设计

基线模型：

YOLOv8

框架：

Ultralytics

---

# 必须实现的改进

## 第一部分

CBAM

插入Backbone

支持配置开关

enable_cbam=true

---

## 第二部分

BiFPN

替换默认PAN-FPN

支持配置开关

enable_bifpn=true

---

## 第三部分

Focal Loss

支持：

Classification Loss

Gamma参数可配置

---

## 第四部分

GhostConv

替换部分Conv

实现轻量化模型

---

## 第五部分

Decoupled Head

实现分类头和回归头解耦

支持配置

---

# 五、消融实验

自动生成：

baseline

baseline+CBAM

baseline+CBAM+BiFPN

baseline+CBAM+BiFPN+Focal

Full Model

输出：

mAP50
mAP50-95
Precision
Recall
FPS

保存csv

保存excel

自动生成图表

---

# 六、训练系统

训练环境：

Python 3.10+

PyTorch 2.x

CUDA优先

支持CPU回退

自动检测：

torch.cuda.is_available()

---

# GPU策略

优先GPU训练

GPU不可用自动切CPU

---

# 导出格式

必须支持：

.pt

.onnx

.torchscript

openvino

ncnn

---

# 七、推理模块

实现统一接口

Detector

支持：

图片检测

视频检测

摄像头检测

RTSP检测

批量检测

---

输出：

bbox

class

confidence

count

---

# 八、PC桌面端

技术栈：

PySide6

禁止PyQt5

---

UI要求

现代化

科技感

非传统学生项目界面

参考：

Tesla Dashboard

Apple Vision Pro

现代SaaS风格

---

功能：

实时摄像头

图片检测

视频检测

批量检测

结果导出

历史记录

类别统计

FPS显示

GPU状态显示

---

模块：

Dashboard

Detection

History

Settings

About

---

支持深色模式

支持浅色模式

---

# 九、Web系统

前端：

Next.js

TypeScript

TailwindCSS

shadcn/ui

Framer Motion

---

禁止：

jQuery

Bootstrap

AdminLTE

旧式后台模板

---

设计风格：

Apple
Linear
Notion
OpenAI

融合风格

极简现代

---

功能：

首页

实时检测

图片上传

视频上传

历史记录

统计分析

模型管理

系统设置

---

支持：

PC

平板

手机

响应式布局

---

摄像头支持：

WebRTC

getUserMedia

---

# 十、后端

FastAPI

---

模块：

auth

detect

dataset

training

statistics

history

system

---

数据库：

SQLite开发

MySQL生产

ORM：

SQLAlchemy

---

# 十一、统计分析

自动生成：

类别分布

检测次数

识别成功率

高频类别

时间趋势

---

图表：

ECharts

---

# 十二、树莓派部署

目标设备：

Raspberry Pi 5

8GB

---

优化目标：

>=10FPS

---

部署格式：

ONNX

NCNN

OpenVINO

三种方案

---

实现：

摄像头检测

本地Web访问

局域网访问

---

# 十三、代码规范

必须采用：

src/

    backend/

    frontend/

    desktop/

    models/

    deployment/

    docs/

    scripts/

---

要求：

类型注解

中文注释

英文Docstring

PEP8

ESLint

Prettier

---

# 十四、自动化脚本

实现：

train.py

export.py

benchmark.py

evaluate.py

ablation.py

deploy_pi.py

---

# 十五、实验报告支持

自动生成：

训练曲线

Loss曲线

PR曲线

混淆矩阵

mAP曲线

类别统计图

---

导出：

PNG

PDF

Excel

---

# 十六、项目验收要求

必须满足：

1. 15类饮片识别

2. 多目标检测

3. 自动计数

4. 实时摄像头检测

5. CPU运行

6. GPU运行

7. Web运行

8. Raspberry Pi运行

9. 消融实验完整

10. 报告素材自动生成

---

# 十七、开发优先级

P0：

数据集
训练
改进YOLOv8

P1：

桌面端

P2：

Web端

P3：

树莓派部署

---

# 十八、预期性能目标

Full Model：

mAP50 ≥ 90%

mAP50-95 ≥ 70%

PC RTX显卡：

≥30 FPS

CPU：

≥8 FPS

树莓派：

≥10 FPS

---

生成完整项目代码。
生成目录结构。
生成所有核心代码。
生成README。
生成部署文档。
生成API文档。