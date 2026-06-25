import time

import cv2
from ultralytics import YOLO


MODEL_PATH = "/home/lm/herbal_project/best.pt"
WINDOW_NAME = "USB Minimal YOLO"


def draw_fps(frame, fps: float):
    text = f"FPS: {fps:.1f}"
    cv2.rectangle(frame, (10, 10), (130, 44), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def main():
    model = YOLO(MODEL_PATH, task="detect")

    print("正在初始化 USB 摄像头...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("摄像头打开失败，请检查 USB 摄像头是否插紧。")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("USB 摄像头推理已启动，按 q 退出。")

    smooth_fps = 0.0
    last_time = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("无法读取摄像头画面，退出。")
                break

            results = model.predict(
                source=frame,
                imgsz=512,
                conf=0.5,
                iou=0.45,
                verbose=False,
            )
            annotated_frame = results[0].plot()

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            smooth_fps = instant_fps if smooth_fps == 0.0 else smooth_fps * 0.9 + instant_fps * 0.1

            annotated_frame = draw_fps(annotated_frame, smooth_fps)
            cv2.imshow(WINDOW_NAME, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print("正在释放摄像头资源...")
        cap.release()
        cv2.destroyAllWindows()
        print("退出成功。")


if __name__ == "__main__":
    main()
