import { NextResponse } from "next/server";

import { API_BASE_URL } from "@/lib/api";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const formData = await request.formData();

  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}/detect/image`, {
      method: "POST",
      body: formData,
    });
  } catch {
    return NextResponse.json(
      {
        detail:
          "后端检测服务未启动或无法连接。请确认 FastAPI 正在 http://127.0.0.1:8000 运行。",
      },
      { status: 503 },
    );
  }

  const contentType = response.headers.get("content-type") ?? "";
  const responseText = await response.text();
  let payload: unknown = responseText
    ? { detail: responseText }
    : { detail: `后端返回空响应，状态码 ${response.status}` };

  if (contentType.includes("application/json") && responseText) {
    try {
      payload = JSON.parse(responseText);
    } catch {
      payload = {
        detail: "后端返回了无法解析的 JSON，请查看后端日志。",
      };
    }
  }

  return NextResponse.json(payload, { status: response.status });
}
