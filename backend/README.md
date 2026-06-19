# TCM-SliceAI FastAPI Backend

后端独立放在 `backend/`，不修改训练代码。它提供三组核心接口：

- `detect`：图片上传检测、批量检测、类别列表
- `history`：检测记录查询、详情、删除、清空
- `statistics`：类别分布、检测次数、时间趋势、平均 FPS

## 安装依赖

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

## 启动服务

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger 文档：

```text
http://127.0.0.1:8000/docs
```

ReDoc 文档：

```text
http://127.0.0.1:8000/redoc
```

## 数据位置

- SQLite：`backend/data/backend.db`
- 上传图片：`backend/data/uploads/`
- 标注结果：`backend/data/outputs/`

模型会自动优先选择：

1. `runs/baseline/weights/best.pt`
2. `reports/ablation/runs/baseline/weights/best.pt`
3. `runs/detect/runs/baseline/weights/best.pt`
4. `yolo26n.pt`
5. `yolov8n.pt`

也可以在 `/detect/image` 的 `model_path` 表单字段中指定模型路径。

