import Link from "next/link";
import { Activity, ArrowLeft } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { getClasses } from "@/lib/api";

import { DetectionWorkbench } from "./workbench";

export const metadata = {
  title: "实时检测 | TCM-SliceAI",
};

export default async function DetectPage() {
  const classes = await getClasses();

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
                实时检测工作台
              </span>
            </span>
          </Link>
          <div className="flex items-center gap-2">
            <Badge variant="success" className="hidden gap-1 sm:inline-flex">
              <span className="size-1.5 rounded-full bg-ok" />
              Framer Motion
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

      <DetectionWorkbench classes={classes} />
    </main>
  );
}
