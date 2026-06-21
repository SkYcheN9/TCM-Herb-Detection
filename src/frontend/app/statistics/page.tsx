import Link from "next/link";
import {
  Activity,
  ArrowLeft,
  BarChart3,
  Clock3,
  Gauge,
  Target,
  Trophy,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { getStatistics } from "@/lib/api";

import { StatisticsCharts } from "./statistics-charts";

export const metadata = {
  title: "药材统计计数 | TCM-SliceAI",
};

function formatNumber(value: number, digits = 0) {
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
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

const classNames = new Map([
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

export default async function StatisticsPage() {
  const statistics = await getStatistics();
  const successRate =
    statistics.summary.total_records > 0
      ? (statistics.summary.successful_records /
          statistics.summary.total_records) *
        100
      : 0;
  const topClass = statistics.summary.top_class
    ? classNames.get(statistics.summary.top_class) ?? statistics.summary.top_class
    : "暂无";

  const metrics = [
    {
      label: "检测次数",
      value: formatNumber(statistics.summary.total_records),
      hint: `${formatNumber(statistics.summary.successful_records)} 次成功`,
      icon: BarChart3,
    },
    {
      label: "识别目标",
      value: formatNumber(statistics.summary.total_objects),
      hint: `最高频：${topClass}`,
      icon: Target,
    },
    {
      label: "成功率",
      value: `${formatNumber(successRate, 1)}%`,
      hint: `${formatNumber(statistics.summary.failed_records)} 次失败`,
      icon: Trophy,
    },
    {
      label: "平均 FPS",
      value: formatNumber(statistics.summary.avg_fps, 1),
      hint: `${formatNumber(statistics.summary.avg_elapsed_ms, 1)} ms`,
      icon: Gauge,
    },
  ];

  return (
    <main className="min-h-screen bg-background">
      <header className="sticky top-0 z-40 border-b border-border/70 bg-background/88 backdrop-blur-xl">
        <div className="container flex h-16 items-center justify-between gap-4">
          <Link href="/" className="flex items-center gap-3">
            <span className="flex size-9 items-center justify-center rounded-md bg-foreground text-background">
              <Activity className="size-4" aria-hidden />
            </span>
            <span className="leading-tight">
              <span className="block text-sm font-semibold">TCM-SliceAI</span>
              <span className="block text-xs text-muted-foreground">
                统计分析中心
              </span>
            </span>
          </Link>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="hidden gap-1.5 sm:inline-flex">
              <Clock3 className="size-3.5 text-ok" aria-hidden />
              {formatLatestTime(statistics.summary.latest_detection_at)}
            </Badge>
            <Button asChild size="sm" variant="outline">
              <Link href="/">
                <ArrowLeft className="size-4" aria-hidden />
                首页
              </Link>
            </Button>
          </div>
        </div>
      </header>

      <section className="app-grid border-b border-border/70">
        <div className="container py-8">
          <div className="mb-6 flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
            <div>
              <Badge variant="outline" className="mb-3 bg-background/75">
                ECharts Analytics
              </Badge>
              <h1 className="max-w-3xl text-balance text-3xl font-semibold leading-tight sm:text-4xl">
                药材统计计数
              </h1>
              <p className="mt-3 max-w-2xl text-pretty text-sm leading-6 text-muted-foreground sm:text-base">
                汇总检测次数、每味药材识别数量、类别占比与时间趋势，用于验收识别效果和后续盘点统计。
              </p>
            </div>
            <Button asChild>
              <Link href="/detect">继续检测</Link>
            </Button>
          </div>

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            {metrics.map((item) => (
              <Card className="bg-card/90" key={item.label}>
                <CardContent className="p-4">
                  <div className="flex items-start justify-between gap-3">
                    <span className="flex size-10 items-center justify-center rounded-md border bg-background">
                      <item.icon
                        className="size-5 text-muted-foreground"
                        aria-hidden
                      />
                    </span>
                    <Badge variant="secondary">{item.label}</Badge>
                  </div>
                  <p className="mt-5 text-3xl font-semibold tabular-nums">
                    {item.value}
                  </p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {item.hint}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      <StatisticsCharts statistics={statistics} />
    </main>
  );
}
