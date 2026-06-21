"use client";

import { useEffect, useMemo, useRef } from "react";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import { motion } from "framer-motion";
import { BarChart3, LineChart, PieChart, TimerReset } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { ClassDistributionItem, StatisticsResponse } from "@/lib/api";

const chineseNames = new Map([
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

const chartPalette = [
  "#13795b",
  "#14b8a6",
  "#0ea5e9",
  "#f59e0b",
  "#ef4444",
  "#84cc16",
  "#6366f1",
  "#d97706",
  "#64748b",
  "#22c55e",
];

function labelFor(item: ClassDistributionItem) {
  return chineseNames.get(item.class_name) ?? item.chinese_name;
}

function compactDate(value: string) {
  return value.slice(5);
}

function axisTextColor() {
  return "#6f6961";
}

const baseGrid = {
  left: 16,
  right: 16,
  top: 36,
  bottom: 12,
  containLabel: true,
};

export function StatisticsCharts({
  statistics,
}: {
  statistics: StatisticsResponse;
}) {
  const rankedClasses = useMemo(() => {
    return [...statistics.class_distribution].sort(
      (a, b) => b.count - a.count || a.class_id - b.class_id,
    );
  }, [statistics.class_distribution]);

  const activeClasses = rankedClasses.filter((item) => item.count > 0);
  const classChartData = activeClasses.length > 0 ? activeClasses : rankedClasses;
  const allClassRows = rankedClasses;
  const maxClassCount = Math.max(1, ...classChartData.map((item) => item.count));
  const trend = statistics.detection_trend;
  const latestTrend = trend.slice(-14);

  const classBarOption = useMemo<EChartsOption>(
    () => ({
      color: chartPalette,
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        valueFormatter: (value) => `${value} 个`,
      },
      grid: baseGrid,
      xAxis: {
        type: "value",
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: "rgba(120, 113, 108, 0.16)" } },
        axisLabel: { color: axisTextColor() },
      },
      yAxis: {
        type: "category",
        data: classChartData.map(labelFor).reverse(),
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: axisTextColor(), width: 80, overflow: "truncate" },
      },
      series: [
        {
          name: "目标数量",
          type: "bar",
          data: classChartData.map((item) => item.count).reverse(),
          barWidth: 12,
          itemStyle: {
            borderRadius: [0, 6, 6, 0],
          },
        },
      ],
    }),
    [classChartData],
  );

  const classPieOption = useMemo<EChartsOption>(
    () => ({
      color: chartPalette,
      tooltip: {
        trigger: "item",
        formatter: "{b}<br/>{c} 个 · {d}%",
      },
      legend: {
        bottom: 0,
        itemWidth: 8,
        itemHeight: 8,
        textStyle: { color: axisTextColor(), fontSize: 11 },
      },
      series: [
        {
          name: "类别占比",
          type: "pie",
          radius: ["52%", "74%"],
          center: ["50%", "44%"],
          avoidLabelOverlap: true,
          label: { show: false },
          emphasis: {
            label: {
              show: true,
              formatter: "{b}\n{d}%",
              color: "#1f1e1c",
              fontWeight: 600,
            },
          },
          data:
            activeClasses.length > 0
              ? activeClasses.map((item) => ({
                  name: labelFor(item),
                  value: item.count,
                }))
              : [{ name: "暂无数据", value: 1, itemStyle: { color: "#d8d2c8" } }],
        },
      ],
    }),
    [activeClasses],
  );

  const trendOption = useMemo<EChartsOption>(
    () => ({
      color: ["#13795b", "#0ea5e9"],
      tooltip: {
        trigger: "axis",
      },
      legend: {
        top: 0,
        right: 8,
        textStyle: { color: axisTextColor(), fontSize: 12 },
      },
      grid: {
        left: 16,
        right: 16,
        top: 42,
        bottom: 12,
        containLabel: true,
      },
      xAxis: {
        type: "category",
        boundaryGap: false,
        data: latestTrend.map((item) => compactDate(item.date)),
        axisLine: { lineStyle: { color: "rgba(120, 113, 108, 0.22)" } },
        axisTick: { show: false },
        axisLabel: { color: axisTextColor() },
      },
      yAxis: {
        type: "value",
        minInterval: 1,
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: "rgba(120, 113, 108, 0.16)" } },
        axisLabel: { color: axisTextColor() },
      },
      series: [
        {
          name: "检测次数",
          type: "line",
          smooth: true,
          data: latestTrend.map((item) => item.record_count),
          symbolSize: 7,
          areaStyle: {
            color: "rgba(19, 121, 91, 0.12)",
          },
          lineStyle: { width: 3 },
        },
        {
          name: "识别目标",
          type: "line",
          smooth: true,
          data: latestTrend.map((item) => item.object_count),
          symbolSize: 7,
          areaStyle: {
            color: "rgba(14, 165, 233, 0.10)",
          },
          lineStyle: { width: 3 },
        },
      ],
    }),
    [latestTrend],
  );

  const detectionCountOption = useMemo<EChartsOption>(
    () => ({
      color: ["#13795b", "#d8d2c8", "#ef4444"],
      tooltip: { trigger: "item", formatter: "{b}: {c}" },
      series: [
        {
          type: "pie",
          radius: ["56%", "78%"],
          center: ["50%", "50%"],
          label: {
            formatter: "{b}\n{c}",
            color: "#1f1e1c",
            fontWeight: 600,
          },
          data: [
            { name: "成功", value: statistics.summary.successful_records },
            { name: "失败", value: statistics.summary.failed_records },
            {
              name: "目标",
              value: Math.max(0, statistics.summary.total_objects),
            },
          ],
        },
      ],
    }),
    [statistics.summary],
  );

  return (
    <section className="bg-muted/35 py-8">
      <div className="container grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
        <div className="grid gap-4">
          <ChartCard
            description="展示 15 类饮片在历史检测中的累计识别数量"
            icon={BarChart3}
            title="药材计数总览"
          >
            <ReactECharts
              className="h-[360px]"
              option={classBarOption}
            />
          </ChartCard>

          <ChartCard
            description="最近 14 天检测次数与识别目标数量变化"
            icon={LineChart}
            title="趋势分析"
          >
            <ReactECharts
              className="h-[340px]"
              option={trendOption}
            />
          </ChartCard>
        </div>

        <div className="grid gap-4">
          <ChartCard
            description="成功、失败与识别目标数量概览"
            icon={TimerReset}
            title="检测次数"
          >
            <ReactECharts
              className="h-[280px]"
              option={detectionCountOption}
            />
          </ChartCard>

          <ChartCard
            description="有检测结果时按类别占比展示"
            icon={PieChart}
            title="类别占比"
          >
            <ReactECharts
              className="h-[280px]"
              option={classPieOption}
            />
          </ChartCard>

          <Card>
            <CardHeader>
              <CardTitle>全部药材计数</CardTitle>
              <CardDescription>按目标数量排序，覆盖固定 15 类饮片</CardDescription>
            </CardHeader>
            <CardContent className="max-h-[420px] space-y-4 overflow-y-auto pr-1">
              {allClassRows.map((item) => (
                <motion.div
                  animate={{ opacity: 1, x: 0 }}
                  className="space-y-2"
                  initial={{ opacity: 0, x: 10 }}
                  key={item.class_id}
                  transition={{ duration: 0.25 }}
                >
                  <div className="flex items-center justify-between gap-3 text-sm">
                    <span className="font-medium">{labelFor(item)}</span>
                    <span className="tabular-nums text-muted-foreground">
                      {item.count}
                    </span>
                  </div>
                  <Progress
                    value={
                      item.count > 0
                        ? Math.max(8, (item.count / maxClassCount) * 100)
                        : 0
                    }
                  />
                </motion.div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}

function ChartCard({
  children,
  description,
  icon: Icon,
  title,
}: {
  children: React.ReactNode;
  description: string;
  icon: typeof BarChart3;
  title: string;
}) {
  return (
    <motion.div
      animate={{ opacity: 1, y: 0 }}
      initial={{ opacity: 0, y: 14 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      <Card className="overflow-hidden">
        <CardHeader className="flex flex-row items-start justify-between gap-4 space-y-0">
          <div>
            <CardTitle>{title}</CardTitle>
            <CardDescription>{description}</CardDescription>
          </div>
          <span className="flex size-10 shrink-0 items-center justify-center rounded-md border bg-background text-muted-foreground">
            <Icon className="size-5" aria-hidden />
          </span>
        </CardHeader>
        <CardContent>{children}</CardContent>
      </Card>
    </motion.div>
  );
}

function ReactECharts({
  className,
  option,
}: {
  className: string;
  option: EChartsOption;
}) {
  const chartRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!chartRef.current) {
      return;
    }

    const chart = echarts.init(chartRef.current, undefined, {
      renderer: "canvas",
    });
    chart.setOption(option, true);

    const resize = () => chart.resize();
    window.addEventListener("resize", resize);

    return () => {
      window.removeEventListener("resize", resize);
      chart.dispose();
    };
  }, [option]);

  return <div className={className} ref={chartRef} />;
}
