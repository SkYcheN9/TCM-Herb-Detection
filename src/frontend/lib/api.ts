export type SummaryStatistics = {
  total_records: number;
  successful_records: number;
  failed_records: number;
  total_objects: number;
  avg_fps: number;
  avg_elapsed_ms: number;
  top_class: string | null;
  latest_detection_at: string | null;
};

export type ClassDistributionItem = {
  class_id: number;
  class_name: string;
  chinese_name: string;
  count: number;
  ratio: number;
};

export type TrendItem = {
  date: string;
  record_count: number;
  object_count: number;
};

export type StatisticsResponse = {
  summary: SummaryStatistics;
  class_distribution: ClassDistributionItem[];
  detection_trend: TrendItem[];
};

export type ClassInfo = {
  id: number;
  name: string;
  chinese_name: string;
};

export type BBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type DetectionApiItem = {
  bbox: BBox;
  class: string;
  confidence: number;
};

export type DetectApiResponse = {
  count: number;
  class_counts: Record<string, number>;
  chinese_class_counts: Record<string, number>;
  image_width?: number | null;
  image_height?: number | null;
  detections: DetectionApiItem[];
};

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export const fallbackClasses: ClassInfo[] = [
  { id: 0, name: "zexie", chinese_name: "泽泻" },
  { id: 1, name: "niuxi", chinese_name: "牛膝" },
  { id: 2, name: "gaoliangjiang", chinese_name: "高良姜" },
  { id: 3, name: "mudanpi", chinese_name: "牡丹皮" },
  { id: 4, name: "yuzhu", chinese_name: "玉竹" },
  { id: 5, name: "baizhi", chinese_name: "白芷" },
  { id: 6, name: "baishao", chinese_name: "白芍" },
  { id: 7, name: "dazao", chinese_name: "大枣" },
  { id: 8, name: "danshen", chinese_name: "丹参" },
  { id: 9, name: "gancao", chinese_name: "甘草" },
  { id: 10, name: "baixianpi", chinese_name: "白鲜皮" },
  { id: 11, name: "baihe", chinese_name: "百合" },
  { id: 12, name: "sangzhi", chinese_name: "桑枝" },
  { id: 13, name: "jiegeng", chinese_name: "桔梗" },
  { id: 14, name: "banlangen", chinese_name: "板蓝根" },
];

export const fallbackStatistics: StatisticsResponse = {
  summary: {
    total_records: 0,
    successful_records: 0,
    failed_records: 0,
    total_objects: 0,
    avg_fps: 0,
    avg_elapsed_ms: 0,
    top_class: null,
    latest_detection_at: null,
  },
  class_distribution: fallbackClasses.map((item) => ({
    class_id: item.id,
    class_name: item.name,
    chinese_name: item.chinese_name,
    count: 0,
    ratio: 0,
  })),
  detection_trend: [],
};

async function requestJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    next: { revalidate: 10 },
  });

  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function getStatistics(): Promise<StatisticsResponse> {
  try {
    return await requestJson<StatisticsResponse>("/statistics");
  } catch {
    return fallbackStatistics;
  }
}

export async function getClasses(): Promise<ClassInfo[]> {
  try {
    return await requestJson<ClassInfo[]>("/detect/classes");
  } catch {
    return fallbackClasses;
  }
}
