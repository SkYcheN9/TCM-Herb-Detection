"""Raspberry Pi camera detection service with local and LAN web access."""

from __future__ import annotations

import argparse
import socket
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from pi_runtime import CLASS_NAMES, YoloDetector, annotate_frame


APP_TITLE = "TCM-SliceAI Raspberry Pi"


class CameraSource:
    """Camera source that supports Picamera2 first and OpenCV fallback."""

    def __init__(self, source: str, width: int, height: int, fps: int) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self._picam2 = None
        self._capture = None

    def open(self) -> None:
        """Open the configured camera source."""

        if self.source in {"auto", "picamera2"}:
            try:
                from picamera2 import Picamera2

                self._picam2 = Picamera2()
                config = self._picam2.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                    controls={"FrameRate": self.fps},
                )
                self._picam2.configure(config)
                self._picam2.start()
                time.sleep(0.5)
                return
            except Exception:
                if self.source == "picamera2":
                    raise

        index = 0 if self.source == "auto" else int(self.source)
        self._capture = cv2.VideoCapture(index)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open camera source: {self.source}")

    def read(self) -> np.ndarray | None:
        """Read one BGR frame."""

        if self._picam2 is not None:
            frame_rgb = self._picam2.capture_array()
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if self._capture is not None:
            ok, frame = self._capture.read()
            return frame if ok else None
        return None

    def close(self) -> None:
        """Close camera resources."""

        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2.close()
            self._picam2 = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class DetectionWorker:
    """Background camera inference loop."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.detector = YoloDetector(
            model_path=args.model,
            backend=args.backend,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            class_names=CLASS_NAMES,
        )
        self.camera = CameraSource(args.camera, args.width, args.height, args.camera_fps)
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None
        self.latest_jpeg: bytes | None = None
        self.stats: dict[str, object] = {
            "fps": 0.0,
            "backend": self.detector.backend,
            "model": str(args.model),
            "count": 0,
            "classes": {},
            "status": "starting",
        }

    def start(self) -> None:
        """Start camera capture and detection."""

        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the background loop."""

        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=3)
        self.camera.close()

    def _loop(self) -> None:
        self.camera.open()
        smooth_fps = 0.0
        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.02)
                continue

            result = self.detector.detect(frame)
            smooth_fps = result.fps if smooth_fps <= 0 else smooth_fps * 0.85 + result.fps * 0.15
            annotated = annotate_frame(frame, result.detections, smooth_fps)
            ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality])
            if not ok:
                continue

            counts = Counter(item.class_name for item in result.detections)
            with self.lock:
                self.latest_jpeg = encoded.tobytes()
                self.stats = {
                    "fps": round(smooth_fps, 2),
                    "backend": self.detector.backend,
                    "model": str(self.args.model),
                    "count": len(result.detections),
                    "classes": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
                    "status": "running",
                }

    def frame_stream(self) -> Iterator[bytes]:
        """Yield MJPEG stream chunks."""

        while True:
            with self.lock:
                frame = self.latest_jpeg
            if frame is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.001)

    def get_stats(self) -> dict[str, object]:
        """Return the latest detection statistics."""

        with self.lock:
            return dict(self.stats)


def parse_args() -> argparse.Namespace:
    """Parse service options."""

    parser = argparse.ArgumentParser(description="Run Raspberry Pi camera detection web service.")
    parser.add_argument("--model", default="best_openvino", help="best_openvino, best.onnx or best.pt")
    parser.add_argument("--backend", default="auto", choices=["auto", "openvino", "onnx", "pytorch"])
    parser.add_argument("--camera", default="auto", help="auto, picamera2, or OpenCV camera index such as 0")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def create_app(worker: DetectionWorker) -> Any:
    """Create the FastAPI application."""

    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse

    app = FastAPI(title=APP_TITLE)

    @app.on_event("startup")
    def _startup() -> None:
        worker.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        worker.stop()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{APP_TITLE}</title>
  <style>
    body {{ margin: 0; background: #0f172a; color: #e5e7eb; font-family: Arial, sans-serif; }}
    main {{ min-height: 100vh; display: grid; grid-template-rows: auto 1fr auto; }}
    header, footer {{ padding: 14px 18px; background: #111827; }}
    header {{ display: flex; justify-content: space-between; gap: 16px; align-items: center; }}
    h1 {{ margin: 0; font-size: 18px; }}
    #stats {{ color: #a7f3d0; font-size: 14px; }}
    .stage {{ display: grid; place-items: center; padding: 12px; }}
    img {{ max-width: 100%; max-height: calc(100vh - 120px); border-radius: 8px; background: #020617; }}
    footer {{ color: #94a3b8; font-size: 13px; }}
  </style>
</head>
<body>
  <main>
    <header><h1>TCM-SliceAI 实时识别</h1><div id="stats">连接中</div></header>
    <div class="stage"><img src="/stream" alt="camera stream" /></div>
    <footer>本机访问: http://127.0.0.1:{worker.args.port} · 局域网访问: http://{lan_ip()}:{worker.args.port}</footer>
  </main>
  <script>
    async function refresh() {{
      const response = await fetch('/api/status');
      const data = await response.json();
      document.getElementById('stats').textContent =
        `FPS ${{data.fps}} · 目标 ${{data.count}} · ${{data.backend}}`;
    }}
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""

    @app.get("/stream")
    def stream() -> StreamingResponse:
        return StreamingResponse(
            worker.frame_stream(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/status")
    def status() -> dict[str, object]:
        return worker.get_stats()

    return app


def lan_ip() -> str:
    """Best-effort LAN IP detection for display."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return "raspberrypi.local"
    finally:
        sock.close()


def main() -> int:
    """Start the web service."""

    args = parse_args()
    import uvicorn

    worker = DetectionWorker(args)
    app = create_app(worker)
    print(f"Local: http://127.0.0.1:{args.port}")
    print(f"LAN:   http://{lan_ip()}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
