import Image from "next/image";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Box,
  Camera,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Cpu,
  Database,
  FileUp,
  Gauge,
  History,
  Layers3,
  LineChart,
  MonitorUp,
  Play,
  Radio,
  Settings2,
  Sparkles,
  Upload,
  Video,
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
import { getClasses, getStatistics } from "@/lib/api";
import { cn } from "@/lib/utils";

const navItems = [
  { label: "检测", href: "/detect" },
  { label: "历史", href: "/history" },
  { label: "统计", href: "/statistics" },
  { label: "模型", href: "/models" },
  { label: "设置", href: "/settings" },
];

const workModes = [
  {
    title: "实时摄像头",
    description: "WebRTC 输入，低延迟推理",
    icon: Camera,
    href: "/detect?mode=camera",
    tone: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  },
  {
    title: "图片上传",
    description: "单张样本检测与计数",
    icon: FileUp,
    href: "/detect?mode=image",
    tone: "bg-stone-100 text-stone-700 ring-stone-200",
  },
  {
    title: "视频分析",
    description: "逐帧识别与结果导出",
    icon: Video,
    href: "/detect?mode=video",
    tone: "bg-sky-50 text-sky-700 ring-sky-200",
  },
  {
    title: "批量检测",
    description: "目录级任务与报表生成",
    icon: Upload,
    href: "/detect?mode=batch",
    tone: "bg-amber-50 text-amber-700 ring-amber-200",
  },
];

const modelPipeline = [
  { label: "YOLOv8n", value: "预训练迁移", icon: Box },
  { label: "CBAM", value: "最高精度 80.12%", icon: Sparkles },
  { label: "BiFPN", value: "部署模型 80.08%", icon: Layers3 },
  { label: "GhostConv", value: "树莓派轻量端", icon: Cpu },
];

const finalModelCards = [
  { label: "网页/桌面端", value: "CBAM+BiFPN", detail: "mAP50-95 80.08% · 302.73 FPS" },
  { label: "最高精度", value: "CBAM", detail: "mAP50-95 80.12%" },
  { label: "树莓派端", value: "GhostConv", detail: "mAP50-95 79.82% · 306.11 FPS" },
];

const fixedChineseNames = new Map([
  ["zexie", "泽泻"],
  ["niuxi", "牛膝"],
  ["gaoliangjiang", "高良姜"],
  ["mudanpi", "牡丹皮"],
  ["yuzhu", "玉竹"],
  ["baizhi", "白芷"],
  ["baishao", "白芍"],
  ["dazao", "大枣"],
  ["danshen", "丹参"],
  ["gancao", "甘草"],
  ["baixianpi", "白鲜皮"],
  ["baihe", "百合"],
  ["sangzhi", "桑枝"],
  ["jiegeng", "桔梗"],
  ["banlangen", "板蓝根"],
]);

function formatNumber(value: number, digits = 0) {
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
}

function formatLatency(value: number) {
  if (!value) {
    return "待检测";
  }

  return `${formatNumber(value, 1)} ms`;
}

function formatLatestTime(value: string | null) {
  if (!value) {
    return "暂无记录";
  }

  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export default async function Home() {
  const [statistics, classes] = await Promise.all([
    getStatistics(),
    getClasses(),
  ]);

  const { summary, class_distribution: classDistribution } = statistics;
  const successRate =
    summary.total_records > 0
      ? Math.round((summary.successful_records / summary.total_records) * 100)
      : 0;
  const activeClasses = classDistribution.filter((item) => item.count > 0);
  const displayDistribution =
    activeClasses.length > 0 ? activeClasses : classDistribution.slice(0, 6);
  const maxClassCount = Math.max(
    1,
    ...displayDistribution.map((item) => item.count),
  );
  const trend = statistics.detection_trend.slice(-14);
  const maxTrendObjects = Math.max(1, ...trend.map((item) => item.object_count));
  const classPreview = classes.slice(0, 15).map((item) => ({
    ...item,
    chinese_name: fixedChineseNames.get(item.name) ?? item.chinese_name,
  }));
  const topClassName = summary.top_class
    ? fixedChineseNames.get(summary.top_class) ?? summary.top_class
    : "等待样本";

  return (
    <main className="min-h-screen overflow-hidden">
      <header className="sticky top-0 z-40 border-b border-border/70 bg-background/85 backdrop-blur-xl">
        <div className="container flex h-16 items-center justify-between gap-4">
          <Link href="/" className="flex items-center gap-3">
            <span className="flex size-9 items-center justify-center rounded-md bg-foreground text-background">
              <Activity className="size-4" aria-hidden="true" />
            </span>
            <span className="leading-tight">
              <span className="block text-sm font-semibold">TCM-SliceAI</span>
              <span className="block text-xs text-muted-foreground">
                中医药饮片智能检测
              </span>
            </span>
          </Link>

          <nav className="hidden items-center gap-1 md:flex">
            {navItems.map((item) => (
              <Link
                className="rounded-md px-3 py-2 text-sm text-muted-foreground transition hover:bg-secondary hover:text-foreground"
                href={item.href}
                key={item.href}
              >
                {item.label}
              </Link>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            <Badge variant="success" className="hidden gap-1 sm:inline-flex">
              <span className="size-1.5 rounded-full bg-ok" />
              API Ready
            </Badge>
            <Button asChild size="sm">
              <Link href="/detect">
                <Play className="size-4" aria-hidden="true" />
                开始检测
              </Link>
            </Button>
          </div>
        </div>
      </header>

      <section className="app-grid border-b border-border/70">
        <div className="container grid min-h-[calc(100vh-4rem)] gap-8 py-8 lg:grid-cols-[minmax(0,0.9fr)_minmax(560px,1.1fr)] lg:items-center lg:py-10">
          <div className="max-w-2xl">
            <Badge variant="outline" className="mb-5 gap-2 bg-background/75">
              <Radio className="size-3.5 text-ok" aria-hidden="true" />
              11 组消融实验已完成
            </Badge>
            <h1 className="max-w-2xl text-balance text-4xl font-semibold leading-[1.05] tracking-normal text-foreground sm:text-5xl lg:text-[3.45rem]">
              中医药饮片智能检测与识别系统
            </h1>
            <p className="mt-5 max-w-xl text-pretty text-base leading-7 text-muted-foreground sm:text-lg">
              项目训练、消融、迁移实验和多端部署链路已完成。网页端与桌面端默认采用 CBAM+BiFPN，树莓派端采用 GhostConv 轻量模型，并提供每味药材的实时统计计数。
            </p>

            <div className="mt-7 flex flex-col gap-3 sm:flex-row">
              <Button asChild size="lg" className="justify-between sm:min-w-40">
                <Link href="/detect">
                  <Camera className="size-4" aria-hidden="true" />
                  打开检测台
                  <ArrowRight className="size-4" aria-hidden="true" />
                </Link>
              </Button>
              <Button
                asChild
                size="lg"
                variant="outline"
                className="justify-between bg-background/75 sm:min-w-40"
              >
                <Link href="/history">
                  <History className="size-4" aria-hidden="true" />
                  查看历史
                </Link>
              </Button>
            </div>

            <div className="mt-9 grid max-w-xl grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricTile
                icon={Database}
                label="检测记录"
                value={formatNumber(summary.total_records)}
              />
              <MetricTile
                icon={Gauge}
                label="平均 FPS"
                value={summary.avg_fps ? formatNumber(summary.avg_fps, 1) : "0.0"}
              />
              <MetricTile
                icon={CheckCircle2}
                label="成功率"
                value={`${successRate}%`}
              />
              <MetricTile icon={Clock3} label="最近检测" value={formatLatestTime(summary.latest_detection_at)} />
            </div>
          </div>

          <div className="relative">
            <div className="relative overflow-hidden rounded-lg border border-border bg-card shadow-panel">
              <div className="flex items-center justify-between border-b border-border bg-card/95 px-4 py-3">
                <div className="flex items-center gap-2">
                  <span className="size-2 rounded-full bg-red-400" />
                  <span className="size-2 rounded-full bg-amber-400" />
                  <span className="size-2 rounded-full bg-ok" />
                </div>
                <div className="hidden items-center gap-2 text-xs text-muted-foreground sm:flex">
                  <Cpu className="size-3.5" aria-hidden="true" />
                  CUDA 优先 · CPU 回退
                </div>
              </div>

              <div className="grid gap-0 xl:grid-cols-[1fr_220px]">
                <div className="relative aspect-[16/11] min-h-[340px] overflow-hidden bg-stone-100 xl:aspect-auto">
                  <Image
                    src="/images/tcm-detection-preview.png"
                    alt="中药饮片检测预览"
                    fill
                    priority
                    unoptimized
                    sizes="(min-width: 1280px) 760px, 100vw"
                    className="object-cover"
                  />
                  <div className="absolute left-4 top-4 rounded-md border border-white/70 bg-white/85 px-3 py-2 text-xs shadow-sm backdrop-blur">
                    <span className="block font-medium text-foreground">
                      实时推理画面
                    </span>
                    <span className="text-muted-foreground">
                      bbox · class · confidence · count
                    </span>
                  </div>
                  <div className="absolute bottom-4 left-4 right-4 grid grid-cols-3 gap-2">
                    <OverlayStat label="目标数" value={formatNumber(summary.total_objects)} />
                    <OverlayStat label="Top" value={topClassName} />
                    <OverlayStat
                      label="Latency"
                      value={formatLatency(summary.avg_elapsed_ms)}
                    />
                  </div>
                </div>

                <aside className="border-t border-border bg-card p-4 xl:border-l xl:border-t-0">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium">模型流水线</p>
                      <p className="text-xs text-muted-foreground">
                        最终部署策略
                      </p>
                    </div>
                    <Badge variant="secondary">Final</Badge>
                  </div>
                  <Separator className="my-4" />
                  <div className="space-y-3">
                    {modelPipeline.map((item, index) => (
                      <div className="flex items-start gap-3" key={item.label}>
                        <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-md border bg-background">
                          <item.icon className="size-4" aria-hidden="true" />
                        </span>
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-medium">{item.label}</p>
                          <p className="truncate text-xs text-muted-foreground">
                            {item.value}
                          </p>
                        </div>
                        {index < modelPipeline.length - 1 ? (
                          <ChevronRight
                            className="mt-2 size-3.5 text-muted-foreground"
                            aria-hidden="true"
                          />
                        ) : null}
                      </div>
                    ))}
                  </div>
                  <Separator className="my-4" />
                  <div className="space-y-2">
                    {finalModelCards.map((item) => (
                      <div className="rounded-md border bg-background px-3 py-2" key={item.label}>
                        <div className="flex items-center justify-between gap-3">
                          <span className="text-xs text-muted-foreground">{item.label}</span>
                          <span className="text-sm font-medium">{item.value}</span>
                        </div>
                        <p className="mt-1 truncate text-xs text-muted-foreground">{item.detail}</p>
                      </div>
                    ))}
                  </div>
                </aside>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="border-b border-border/70 bg-background py-8">
        <div className="container">
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            {workModes.map((mode) => (
              <Link className="group block" href={mode.href} key={mode.title}>
                <Card className="h-full transition hover:-translate-y-0.5 hover:shadow-panel">
                  <CardHeader>
                    <div className="flex items-start justify-between gap-3">
                      <span
                        className={cn(
                          "flex size-10 items-center justify-center rounded-md ring-1",
                          mode.tone,
                        )}
                      >
                        <mode.icon className="size-5" aria-hidden="true" />
                      </span>
                      <ArrowRight
                        className="size-4 text-muted-foreground transition group-hover:translate-x-0.5 group-hover:text-foreground"
                        aria-hidden="true"
                      />
                    </div>
                    <CardTitle>{mode.title}</CardTitle>
                    <CardDescription>{mode.description}</CardDescription>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="bg-muted/35 py-10">
        <div className="container grid gap-4 lg:grid-cols-[1.05fr_0.95fr]">
          <Card>
            <CardHeader className="flex flex-row items-start justify-between gap-4 space-y-0">
              <div>
                <CardTitle>类别覆盖</CardTitle>
                <CardDescription>
                  固定 15 类顺序，与 YOLO classes.txt 保持一致
                </CardDescription>
              </div>
              <Badge variant="outline">{classPreview.length} 类</Badge>
            </CardHeader>
            <CardContent>
              <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                {classPreview.map((item) => (
                  <div
                    className="flex min-h-12 items-center justify-between rounded-md border bg-background px-3 py-2"
                    key={item.id}
                  >
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium">
                        {item.chinese_name}
                      </p>
                      <p className="truncate text-xs text-muted-foreground">
                        {item.name}
                      </p>
                    </div>
                    <span className="ml-2 text-xs tabular-nums text-muted-foreground">
                      {String(item.id).padStart(2, "0")}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4">
            <Card>
              <CardHeader className="flex flex-row items-start justify-between gap-4 space-y-0">
                <div>
                  <CardTitle>识别分布</CardTitle>
                  <CardDescription>来自检测记录的各药材累计计数</CardDescription>
                </div>
                <BarChart3 className="size-5 text-muted-foreground" aria-hidden="true" />
              </CardHeader>
              <CardContent className="space-y-4">
                {displayDistribution.map((item) => {
                  const label =
                    fixedChineseNames.get(item.class_name) ?? item.chinese_name;
                  const progress =
                    item.count > 0
                      ? Math.max(6, Math.round((item.count / maxClassCount) * 100))
                      : 0;

                  return (
                    <div className="space-y-2" key={item.class_id}>
                      <div className="flex items-center justify-between gap-3 text-sm">
                        <span className="font-medium">{label}</span>
                        <span className="tabular-nums text-muted-foreground">
                          {formatNumber(item.count)}
                        </span>
                      </div>
                      <Progress value={progress} />
                    </div>
                  );
                })}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-start justify-between gap-4 space-y-0">
                <div>
                  <CardTitle>近期趋势</CardTitle>
                  <CardDescription>最近 14 天检测对象数量</CardDescription>
                </div>
                <LineChart className="size-5 text-muted-foreground" aria-hidden="true" />
              </CardHeader>
              <CardContent>
                <div className="flex h-28 items-end gap-1.5">
                  {trend.length > 0
                    ? trend.map((item) => (
                        <div
                          className="flex min-w-0 flex-1 flex-col items-center gap-2"
                          key={item.date}
                        >
                          <div
                            className="w-full rounded-t-sm bg-primary/75"
                            style={{
                              height: `${Math.max(
                                8,
                                (item.object_count / maxTrendObjects) * 96,
                              )}px`,
                              opacity: item.object_count > 0 ? 1 : 0.18,
                            }}
                          />
                          <span className="hidden text-[10px] text-muted-foreground sm:block">
                            {item.date.slice(5)}
                          </span>
                        </div>
                      ))
                    : Array.from({ length: 14 }).map((_, index) => (
                        <div
                          className="flex flex-1 flex-col items-center gap-2"
                          key={index}
                        >
                          <div className="h-2 w-full rounded-t-sm bg-primary/15" />
                          <span className="hidden text-[10px] text-muted-foreground sm:block">
                            --
                          </span>
                        </div>
                      ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      <section className="border-t border-border/70 bg-background py-8">
        <div className="container flex flex-col justify-between gap-4 md:flex-row md:items-center">
          <div>
            <p className="text-sm font-medium">部署目标</p>
            <p className="mt-1 text-sm text-muted-foreground">
              网页端与桌面端优先精度和稳定性，树莓派 5 无算力棒端优先 OpenVINO/ONNX 与轻量模型实时性。
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline" className="gap-1.5">
              <MonitorUp className="size-3.5" aria-hidden="true" />
              Web
            </Badge>
            <Badge variant="outline" className="gap-1.5">
              <Cpu className="size-3.5" aria-hidden="true" />
              CUDA / CPU
            </Badge>
            <Badge variant="outline" className="gap-1.5">
              <Settings2 className="size-3.5" aria-hidden="true" />
              ONNX · NCNN · OpenVINO
            </Badge>
          </div>
        </div>
      </section>
    </main>
  );
}

function MetricTile({
  icon: Icon,
  label,
  value,
}: {
  icon: LucideIcon;
  label: string;
  value: string;
}) {
  return (
    <div className="min-h-24 rounded-lg border border-border bg-background/78 p-3 shadow-sm backdrop-blur">
      <Icon className="size-4 text-muted-foreground" aria-hidden />
      <p className="mt-4 text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 truncate text-lg font-semibold tabular-nums">{value}</p>
    </div>
  );
}

function OverlayStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-md border border-white/60 bg-white/82 px-3 py-2 shadow-sm backdrop-blur">
      <p className="text-[11px] uppercase text-muted-foreground">{label}</p>
      <p className="truncate text-sm font-semibold text-foreground">{value}</p>
    </div>
  );
}
