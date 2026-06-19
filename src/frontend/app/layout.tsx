import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "TCM-SliceAI",
  description: "中医药饮片智能检测与识别系统",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
