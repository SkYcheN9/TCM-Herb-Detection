import { NextResponse } from "next/server";

import { API_BASE_URL } from "@/lib/api";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const formData = await request.formData();
  const response = await fetch(`${API_BASE_URL}/detect/image`, {
    method: "POST",
    body: formData,
  });

  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : { detail: await response.text() };

  return NextResponse.json(payload, { status: response.status });
}
