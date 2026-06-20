"use client";

import {
  ChangeEvent,
  DragEvent,
  ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import Image from "next/image";
import {
  AnimatePresence,
  motion,
  useMotionTemplate,
  useMotionValue,
  useSpring,
} from "framer-motion";
import {
  Aperture,
  Camera,
  CheckCircle2,
  Gauge,
  ImagePlus,
  Loader2,
  Play,
  Radio,
  RotateCcw,
  ScanLine,
  SlidersHorizontal,
  Sparkles,
  Square,
  UploadCloud,
  Video,
  X,
  Zap,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import type { ClassInfo, DetectApiResponse, DetectionApiItem } from "@/lib/api";
import { fallbackClasses } from "@/lib/api";
import { cn } from "@/lib/utils";

type Mode = "camera" | "upload";
type Device = "auto" | "cpu" | "cuda";
type PreviewSource = {
  url: string;
  name: string;
  width: number;
  height: number;
  file?: File;
};

const modeOptions = [
  { value: "camera" as const, label: "摄像头", icon: Video },
  { value: "upload" as const, label: "图片上传", icon: ImagePlus },
];

const deviceOptions: Device[] = ["auto", "cuda", "cpu"];

const classColorPalette = [
  "rgb(20 184 166)",
  "rgb(16 185 129)",
  "rgb(14 165 233)",
  "rgb(245 158 11)",
  "rgb(239 68 68)",
  "rgb(132 204 22)",
  "rgb(99 102 241)",
  "rgb(217 119 6)",
];

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function getClassLabel(name: string, classes: ClassInfo[]) {
  return classes.find((item) => item.name === name)?.chinese_name ?? name;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function confidenceTone(value: number) {
  if (value >= 0.85) {
    return "text-ok";
  }

  if (value >= 0.65) {
    return "text-signal";
  }

  return "text-muted-foreground";
}

export function DetectionWorkbench({ classes }: { classes: ClassInfo[] }) {
  const classList = classes.length > 0 ? classes : fallbackClasses;
  const [mode, setMode] = useState<Mode>("camera");
  const [preview, setPreview] = useState<PreviewSource | null>(null);
  const [detections, setDetections] = useState<DetectionApiItem[]>([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [message, setMessage] = useState("等待输入源");
  const [confidence, setConfidence] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [imageSize, setImageSize] = useState(640);
  const [device, setDevice] = useState<Device>("auto");
  const [autoLoop, setAutoLoop] = useState(false);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const loopTimerRef = useRef<number | null>(null);
  const isDetectingRef = useRef(false);

  const pointerX = useMotionValue(0);
  const pointerY = useMotionValue(0);
  const smoothX = useSpring(pointerX, { stiffness: 140, damping: 26 });
  const smoothY = useSpring(pointerY, { stiffness: 140, damping: 26 });
  const spotlight = useMotionTemplate`radial-gradient(420px circle at ${smoothX}px ${smoothY}px, rgba(20, 184, 166, 0.16), transparent 48%)`;

  const groupedDetections = useMemo(() => {
    return detections.reduce<Record<string, number>>((acc, item) => {
      acc[item.class] = (acc[item.class] ?? 0) + 1;
      return acc;
    }, {});
  }, [detections]);

  const topConfidence = useMemo(() => {
    return detections.reduce((max, item) => Math.max(max, item.confidence), 0);
  }, [detections]);

  const stopCamera = useCallback(() => {
    if (loopTimerRef.current) {
      window.clearInterval(loopTimerRef.current);
      loopTimerRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setIsCameraActive(false);
    setAutoLoop(false);
  }, []);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  useEffect(() => {
    return () => {
      if (preview?.file) {
        URL.revokeObjectURL(preview.url);
      }
    };
  }, [preview]);

  useEffect(() => {
    isDetectingRef.current = isDetecting;
  }, [isDetecting]);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    setMessage("正在打开摄像头");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "environment",
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setMode("camera");
      setIsCameraActive(true);
      setDetections([]);
      setPreview(null);
      setMessage("摄像头已就绪");
    } catch {
      setCameraError("无法访问摄像头，请检查浏览器权限或设备占用。");
      setMessage("摄像头不可用");
    }
  }, []);

  const captureFrameAsFile = useCallback(async () => {
    const video = videoRef.current;

    if (!video || video.readyState < 2) {
      throw new Error("摄像头画面尚未就绪");
    }

    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");

    if (!context) {
      throw new Error("无法创建画面采集上下文");
    }

    context.drawImage(video, 0, 0, width, height);

    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob(resolve, "image/jpeg", 0.92);
    });

    if (!blob) {
      throw new Error("摄像头抓拍失败");
    }

    return new File([blob], `camera-${Date.now()}.jpg`, {
      type: "image/jpeg",
    });
  }, []);

  const readImageSize = useCallback((url: string) => {
    return new Promise<{ width: number; height: number }>((resolve) => {
      const image = new window.Image();
      image.onload = () =>
        resolve({
          width: image.naturalWidth || 1,
          height: image.naturalHeight || 1,
        });
      image.onerror = () => resolve({ width: 1, height: 1 });
      image.src = url;
    });
  }, []);

  const setUploadedFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith("image/")) {
        setMessage("请选择图片文件");
        return;
      }

      if (preview?.file) {
        URL.revokeObjectURL(preview.url);
      }

      const url = URL.createObjectURL(file);
      const size = await readImageSize(url);
      setMode("upload");
      setPreview({
        url,
        name: file.name,
        width: size.width,
        height: size.height,
        file,
      });
      setDetections([]);
      setLatencyMs(null);
      setMessage("图片已载入，可开始检测");
    },
    [preview, readImageSize],
  );

  const detectFile = useCallback(
    async (file: File, nextPreview?: PreviewSource) => {
      setIsDetecting(true);
      setMessage("正在推理");
      const startedAt = performance.now();

      try {
        if (nextPreview) {
          setPreview(nextPreview);
        }

        const body = new FormData();
        body.append("file", file);
        body.append("confidence", String(confidence));
        body.append("iou", String(iou));
        body.append("image_size", String(imageSize));
        body.append("device", device);
        body.append("save_annotated", "true");

        const response = await fetch("/api/detect/image", {
          method: "POST",
          body,
        });

        const responseText = await response.text();
        const data = (
          responseText
            ? (JSON.parse(responseText) as DetectApiResponse & {
                detail?: unknown;
              })
            : { detail: "检测服务返回空响应" }
        ) as DetectApiResponse & { detail?: unknown };

        if (!response.ok) {
          throw new Error(
            typeof data.detail === "string" ? data.detail : "检测请求失败",
          );
        }

        setDetections(data.detections ?? []);
        setLatencyMs(performance.now() - startedAt);
        setMessage(
          data.count > 0 ? `识别到 ${data.count} 个目标` : "未识别到目标",
        );

      } catch (error) {
        setMessage(error instanceof Error ? error.message : "检测失败");
      } finally {
        setIsDetecting(false);
      }
    },
    [confidence, device, imageSize, iou],
  );

  const runUploadDetection = useCallback(() => {
    if (!preview?.file) {
      setMessage("请先上传图片");
      return;
    }

    void detectFile(preview.file);
  }, [detectFile, preview]);

  const runCameraDetection = useCallback(async () => {
    if (!isCameraActive) {
      setMessage("请先打开摄像头");
      return;
    }

    try {
      const file = await captureFrameAsFile();
      const url = URL.createObjectURL(file);
      const size = await readImageSize(url);

      await detectFile(file, {
        url,
        name: "摄像头抓拍",
        width: size.width,
        height: size.height,
        file,
      });
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "摄像头抓拍失败");
    }
  }, [
    captureFrameAsFile,
    detectFile,
    isCameraActive,
    readImageSize,
  ]);

  useEffect(() => {
    if (!autoLoop) {
      if (loopTimerRef.current) {
        window.clearInterval(loopTimerRef.current);
        loopTimerRef.current = null;
      }
      return;
    }

    loopTimerRef.current = window.setInterval(() => {
      if (!isDetectingRef.current) {
        void runCameraDetection();
      }
    }, 2600);

    return () => {
      if (loopTimerRef.current) {
        window.clearInterval(loopTimerRef.current);
        loopTimerRef.current = null;
      }
    };
  }, [autoLoop, runCameraDetection]);

  const handleDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDragging(false);
      const file = event.dataTransfer.files.item(0);

      if (file) {
        void setUploadedFile(file);
      }
    },
    [setUploadedFile],
  );

  const handleFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.item(0);

      if (file) {
        void setUploadedFile(file);
      }
    },
    [setUploadedFile],
  );

  const selectMode = useCallback(
    (nextMode: Mode) => {
      if (nextMode === "upload") {
        stopCamera();
      }

      setMode(nextMode);
      setMessage(nextMode === "camera" ? "准备摄像头输入" : "等待上传图片");
    },
    [stopCamera],
  );

  const clearResult = useCallback(() => {
    setDetections([]);
    setLatencyMs(null);
    setMessage("结果已清空");
  }, []);

  const previewWidth = preview?.width ?? 1568;
  const previewHeight = preview?.height ?? 1003;
  return (
    <div
      className="app-grid min-h-[calc(100vh-4rem)]"
      onPointerMove={(event) => {
        pointerX.set(event.clientX);
        pointerY.set(event.clientY);
      }}
    >
      <motion.div
        aria-hidden
        className="pointer-events-none fixed inset-0 opacity-80"
        style={{ background: spotlight }}
      />

      <div className="container relative z-10 py-6 lg:py-8">
        <motion.div
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 flex flex-col justify-between gap-4 lg:flex-row lg:items-end"
          initial={{ opacity: 0, y: 16 }}
          transition={{ duration: 0.45, ease: "easeOut" }}
        >
          <div>
            <Badge variant="outline" className="mb-3 gap-2 bg-background/75">
              <ScanLine className="size-3.5 text-ok" aria-hidden />
              Realtime Detection
            </Badge>
            <h1 className="max-w-3xl text-balance text-3xl font-semibold leading-tight sm:text-4xl">
              摄像头与图片上传检测
            </h1>
            <p className="mt-3 max-w-2xl text-pretty text-sm leading-6 text-muted-foreground sm:text-base">
              接入 FastAPI 推理接口，支持摄像头抓拍、自动循环检测、图片上传与目标框可视化。
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {modeOptions.map((item) => (
              <Button
                className={cn(
                  "gap-2 bg-background/78",
                  mode === item.value && "bg-foreground text-background",
                )}
                key={item.value}
                onClick={() => selectMode(item.value)}
                type="button"
                variant={mode === item.value ? "default" : "outline"}
              >
                <item.icon className="size-4" aria-hidden />
                {item.label}
              </Button>
            ))}
          </div>
        </motion.div>

        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
          <motion.section
            animate={{ opacity: 1, scale: 1 }}
            className="overflow-hidden rounded-lg border bg-card shadow-panel"
            initial={{ opacity: 0, scale: 0.985 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          >
            <div className="flex min-h-14 flex-wrap items-center justify-between gap-3 border-b bg-card/90 px-4 py-3">
              <div className="flex items-center gap-2">
                <span className="size-2 rounded-full bg-red-400" />
                <span className="size-2 rounded-full bg-amber-400" />
                <span className="size-2 rounded-full bg-ok" />
                <span className="ml-2 text-sm font-medium">视觉输入</span>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant={isDetecting ? "secondary" : "outline"}>
                  {isDetecting ? "推理中" : message}
                </Badge>
                <Badge variant="outline">
                  conf {formatPercent(confidence)} · iou {formatPercent(iou)}
                </Badge>
              </div>
            </div>

            <div className="grid gap-0 lg:grid-cols-[minmax(0,1fr)_260px]">
              <div className="relative min-h-[420px] bg-stone-100">
                <AnimatePresence mode="wait">
                  {mode === "camera" ? (
                    <motion.div
                      animate={{ opacity: 1 }}
                      className="absolute inset-0"
                      exit={{ opacity: 0 }}
                      initial={{ opacity: 0 }}
                      key="camera"
                    >
                      <video
                        autoPlay
                        className={cn(
                          "size-full object-cover",
                          !isCameraActive && "opacity-0",
                        )}
                        muted
                        playsInline
                        ref={videoRef}
                      />
                      {!isCameraActive ? (
                        <EmptySource
                          icon={Camera}
                          title="打开摄像头"
                          description="浏览器会请求摄像头权限，允许后即可抓拍检测。"
                          actionLabel="启动摄像头"
                          onAction={startCamera}
                        />
                      ) : null}
                    </motion.div>
                  ) : (
                    <motion.div
                      animate={{ opacity: 1 }}
                      className="absolute inset-0"
                      exit={{ opacity: 0 }}
                      initial={{ opacity: 0 }}
                      key="upload"
                    >
                      {preview ? (
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="relative inline-block max-h-full max-w-full">
                            <Image
                              alt={preview.name}
                              className="block h-auto max-h-full w-auto max-w-full object-contain"
                              height={preview.height}
                              sizes="(min-width: 1024px) 780px, 100vw"
                              src={preview.url}
                              unoptimized
                              width={preview.width}
                            />
                            <DetectionOverlay
                              classes={classList}
                              detections={detections}
                              height={previewHeight}
                              width={previewWidth}
                            />
                          </div>
                        </div>
                      ) : (
                        <div
                          className={cn(
                            "absolute inset-4 flex items-center justify-center rounded-lg border border-dashed bg-background/72 p-4 transition",
                            dragging && "border-primary bg-accent/70",
                          )}
                          onDragEnter={(event) => {
                            event.preventDefault();
                            setDragging(true);
                          }}
                          onDragLeave={() => setDragging(false)}
                          onDragOver={(event) => event.preventDefault()}
                          onDrop={handleDrop}
                        >
                          <EmptySource
                            icon={UploadCloud}
                            title="上传饮片图片"
                            description="支持 JPG、PNG、WEBP。上传后可立即提交推理。"
                            actionLabel="选择图片"
                            onAction={() => fileInputRef.current?.click()}
                          />
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>

                {mode === "camera" && preview ? (
                  <DetectionOverlay
                    classes={classList}
                    detections={detections}
                    height={previewHeight}
                    width={previewWidth}
                  />
                ) : null}

                <AnimatePresence>
                  {isDetecting ? (
                    <motion.div
                      animate={{ opacity: 1 }}
                      className="absolute inset-0 flex items-center justify-center bg-background/24 backdrop-blur-[1px]"
                      exit={{ opacity: 0 }}
                      initial={{ opacity: 0 }}
                    >
                      <motion.div
                        animate={{ y: [0, -4, 0] }}
                        className="rounded-lg border bg-card/90 px-4 py-3 shadow-panel"
                        transition={{ duration: 1.2, repeat: Infinity }}
                      >
                        <div className="flex items-center gap-3">
                          <Loader2 className="size-5 animate-spin text-primary" />
                          <div>
                            <p className="text-sm font-medium">模型正在推理</p>
                            <p className="text-xs text-muted-foreground">
                              YOLOv8 · CBAM · BiFPN · Focal Loss
                            </p>
                          </div>
                        </div>
                      </motion.div>
                    </motion.div>
                  ) : null}
                </AnimatePresence>

                {cameraError ? (
                  <div className="absolute bottom-4 left-4 right-4 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                    {cameraError}
                  </div>
                ) : null}
              </div>

              <aside className="border-t bg-card p-4 lg:border-l lg:border-t-0">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium">控制台</p>
                    <p className="text-xs text-muted-foreground">
                      输入源、阈值与设备
                    </p>
                  </div>
                  <Button
                    aria-label="清空结果"
                    onClick={clearResult}
                    size="icon"
                    type="button"
                    variant="outline"
                  >
                    <RotateCcw className="size-4" aria-hidden />
                  </Button>
                </div>

                <Separator className="my-4" />

                <div className="space-y-4">
                  <ControlRow
                    label="置信度"
                    value={formatPercent(confidence)}
                  >
                    <input
                      className="w-full accent-[hsl(var(--primary))]"
                      max="0.95"
                      min="0.05"
                      onChange={(event) =>
                        setConfidence(Number(event.target.value))
                      }
                      step="0.05"
                      type="range"
                      value={confidence}
                    />
                  </ControlRow>

                  <ControlRow label="IoU" value={formatPercent(iou)}>
                    <input
                      className="w-full accent-[hsl(var(--primary))]"
                      max="0.9"
                      min="0.1"
                      onChange={(event) => setIou(Number(event.target.value))}
                      step="0.05"
                      type="range"
                      value={iou}
                    />
                  </ControlRow>

                  <ControlRow label="图像尺寸" value={`${imageSize}`}>
                    <input
                      className="w-full accent-[hsl(var(--primary))]"
                      max="1280"
                      min="320"
                      onChange={(event) =>
                        setImageSize(Number(event.target.value))
                      }
                      step="32"
                      type="range"
                      value={imageSize}
                    />
                  </ControlRow>

                  <div>
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-sm font-medium">设备</span>
                      <SlidersHorizontal
                        className="size-4 text-muted-foreground"
                        aria-hidden
                      />
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      {deviceOptions.map((item) => (
                        <button
                          className={cn(
                            "h-9 rounded-md border bg-background text-xs font-medium transition",
                            device === item &&
                              "border-primary bg-primary text-primary-foreground",
                          )}
                          key={item}
                          onClick={() => setDevice(item)}
                          type="button"
                        >
                          {item.toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>

                  <Separator />

                  {mode === "camera" ? (
                    <div className="space-y-2">
                      <Button
                        className="w-full justify-between"
                        disabled={isDetecting}
                        onClick={
                          isCameraActive ? runCameraDetection : startCamera
                        }
                        type="button"
                      >
                        <Camera className="size-4" aria-hidden />
                        {isCameraActive ? "抓拍检测" : "启动摄像头"}
                        {isDetecting ? (
                          <Loader2 className="size-4 animate-spin" />
                        ) : (
                          <Zap className="size-4" aria-hidden />
                        )}
                      </Button>
                      <Button
                        className="w-full justify-between"
                        disabled={!isCameraActive}
                        onClick={() => setAutoLoop((value) => !value)}
                        type="button"
                        variant={autoLoop ? "secondary" : "outline"}
                      >
                        {autoLoop ? (
                          <Square className="size-4" aria-hidden />
                        ) : (
                          <Play className="size-4" aria-hidden />
                        )}
                        {autoLoop ? "停止自动检测" : "自动循环检测"}
                        <Radio
                          className={cn(
                            "size-4",
                            autoLoop && "text-ok",
                          )}
                          aria-hidden
                        />
                      </Button>
                      <Button
                        className="w-full"
                        disabled={!isCameraActive}
                        onClick={stopCamera}
                        type="button"
                        variant="outline"
                      >
                        <X className="size-4" aria-hidden />
                        关闭摄像头
                      </Button>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <Button
                        className="w-full justify-between"
                        onClick={() => fileInputRef.current?.click()}
                        type="button"
                        variant="outline"
                      >
                        <UploadCloud className="size-4" aria-hidden />
                        选择图片
                        <Aperture className="size-4" aria-hidden />
                      </Button>
                      <Button
                        className="w-full justify-between"
                        disabled={!preview?.file || isDetecting}
                        onClick={runUploadDetection}
                        type="button"
                      >
                        <ScanLine className="size-4" aria-hidden />
                        开始检测
                        {isDetecting ? (
                          <Loader2 className="size-4 animate-spin" />
                        ) : (
                          <Zap className="size-4" aria-hidden />
                        )}
                      </Button>
                    </div>
                  )}
                </div>
              </aside>
            </div>
          </motion.section>

          <aside className="grid gap-4">
            <ResultSummary
              count={detections.length}
              latencyMs={latencyMs}
              message={message}
              topConfidence={topConfidence}
            />

            <Card>
              <CardHeader>
                <CardTitle>目标列表</CardTitle>
                <CardDescription>检测类别、置信度与边界框坐标</CardDescription>
              </CardHeader>
              <CardContent>
                <AnimatePresence mode="popLayout">
                  {detections.length > 0 ? (
                    <motion.div className="space-y-3" layout>
                      {detections.map((item, index) => (
                        <motion.div
                          animate={{ opacity: 1, x: 0 }}
                          className="rounded-md border bg-background p-3"
                          exit={{ opacity: 0, x: 12 }}
                          initial={{ opacity: 0, x: 18 }}
                          key={`${item.class}-${index}-${item.confidence}`}
                          layout
                          transition={{ delay: index * 0.035 }}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <p className="truncate text-sm font-medium">
                                {getClassLabel(item.class, classList)}
                              </p>
                              <p className="text-xs text-muted-foreground">
                                {item.class}
                              </p>
                            </div>
                            <span
                              className={cn(
                                "text-sm font-semibold tabular-nums",
                                confidenceTone(item.confidence),
                              )}
                            >
                              {formatPercent(item.confidence)}
                            </span>
                          </div>
                          <div className="mt-3 grid grid-cols-4 gap-2 text-[11px] text-muted-foreground">
                            {(["x1", "y1", "x2", "y2"] as const).map((key) => (
                              <div
                                className="rounded-sm bg-muted px-2 py-1 tabular-nums"
                                key={key}
                              >
                                {key} {Math.round(item.bbox[key])}
                              </div>
                            ))}
                          </div>
                        </motion.div>
                      ))}
                    </motion.div>
                  ) : (
                    <motion.div
                      animate={{ opacity: 1 }}
                      className="flex min-h-44 flex-col items-center justify-center rounded-lg border border-dashed bg-muted/35 p-5 text-center"
                      exit={{ opacity: 0 }}
                      initial={{ opacity: 0 }}
                    >
                      <Sparkles className="mb-3 size-8 text-muted-foreground" />
                      <p className="text-sm font-medium">暂无检测结果</p>
                      <p className="mt-1 text-xs leading-5 text-muted-foreground">
                        启动摄像头或上传图片后，目标结果会显示在这里。
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>类别统计</CardTitle>
                <CardDescription>本次画面中的目标分布</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {Object.entries(groupedDetections).length > 0 ? (
                  Object.entries(groupedDetections).map(([name, count]) => (
                    <div className="space-y-2" key={name}>
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-medium">
                          {getClassLabel(name, classList)}
                        </span>
                        <span className="tabular-nums text-muted-foreground">
                          {count}
                        </span>
                      </div>
                      <Progress
                        value={clamp(
                          (count / Math.max(1, detections.length)) * 100,
                          8,
                          100,
                        )}
                      />
                    </div>
                  ))
                ) : (
                  <div className="rounded-md border bg-background px-3 py-3 text-sm text-muted-foreground">
                    等待检测样本
                  </div>
                )}
              </CardContent>
            </Card>
          </aside>
        </div>

        <input
          accept="image/*"
          className="hidden"
          onChange={handleFileChange}
          ref={fileInputRef}
          type="file"
        />
      </div>
    </div>
  );
}

function DetectionOverlay({
  classes,
  detections,
  height,
  width,
}: {
  classes: ClassInfo[];
  detections: DetectionApiItem[];
  height: number;
  width: number;
}) {
  return (
    <div className="pointer-events-none absolute inset-0">
      <AnimatePresence>
        {detections.map((item, index) => {
          const color = classColorPalette[index % classColorPalette.length];
          const left = clamp((item.bbox.x1 / width) * 100, 0, 100);
          const top = clamp((item.bbox.y1 / height) * 100, 0, 100);
          const boxWidth = clamp(
            ((item.bbox.x2 - item.bbox.x1) / width) * 100,
            0,
            100 - left,
          );
          const boxHeight = clamp(
            ((item.bbox.y2 - item.bbox.y1) / height) * 100,
            0,
            100 - top,
          );

          return (
            <motion.div
              animate={{ opacity: 1, scale: 1 }}
              className="absolute rounded-sm border-2 shadow-[0_0_0_1px_rgba(255,255,255,0.42)]"
              exit={{ opacity: 0, scale: 0.96 }}
              initial={{ opacity: 0, scale: 0.92 }}
              key={`${item.class}-${index}`}
              style={{
                borderColor: color,
                height: `${boxHeight}%`,
                left: `${left}%`,
                top: `${top}%`,
                width: `${boxWidth}%`,
              }}
              transition={{ delay: index * 0.035, type: "spring", bounce: 0.2 }}
            >
              <span
                className="absolute -top-7 left-0 max-w-44 truncate rounded-sm px-2 py-1 text-[11px] font-semibold text-white shadow-sm"
                style={{ backgroundColor: color }}
              >
                {getClassLabel(item.class, classes)} ·{" "}
                {formatPercent(item.confidence)}
              </span>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

function EmptySource({
  actionLabel,
  description,
  icon: Icon,
  onAction,
  title,
}: {
  actionLabel: string;
  description: string;
  icon: LucideIcon;
  onAction: () => void;
  title: string;
}) {
  return (
    <div className="absolute inset-0 flex items-center justify-center p-6">
      <div className="max-w-sm rounded-lg border bg-card/90 p-5 text-center shadow-panel backdrop-blur">
        <div className="mx-auto mb-4 flex size-12 items-center justify-center rounded-md bg-primary/10 text-primary">
          <Icon className="size-6" aria-hidden />
        </div>
        <h2 className="text-base font-semibold">{title}</h2>
        <p className="mt-2 text-sm leading-6 text-muted-foreground">
          {description}
        </p>
        <Button className="mt-5" onClick={onAction} type="button">
          {actionLabel}
        </Button>
      </div>
    </div>
  );
}

function ControlRow({
  children,
  label,
  value,
}: {
  children: ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="text-sm font-medium">{label}</span>
        <span className="text-xs tabular-nums text-muted-foreground">
          {value}
        </span>
      </div>
      {children}
    </div>
  );
}

function ResultSummary({
  count,
  latencyMs,
  message,
  topConfidence,
}: {
  count: number;
  latencyMs: number | null;
  message: string;
  topConfidence: number;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>检测结果</CardTitle>
        <CardDescription>{message}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-2">
          <SummaryTile
            icon={CheckCircle2}
            label="目标"
            value={String(count)}
          />
          <SummaryTile
            icon={Gauge}
            label="最高置信度"
            value={topConfidence ? formatPercent(topConfidence) : "--"}
          />
          <SummaryTile
            icon={Zap}
            label="耗时"
            value={latencyMs ? `${Math.round(latencyMs)}ms` : "--"}
          />
        </div>
      </CardContent>
    </Card>
  );
}

function SummaryTile({
  icon: Icon,
  label,
  value,
}: {
  icon: LucideIcon;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-md border bg-background p-3">
      <Icon className="size-4 text-muted-foreground" aria-hidden />
      <p className="mt-3 text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 truncate text-lg font-semibold tabular-nums">
        {value}
      </p>
    </div>
  );
}
