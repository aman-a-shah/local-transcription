import { NextRequest, NextResponse } from "next/server";
import { track } from "@vercel/analytics/server";
import { BUILDS, latestAssetUrl, resolveBuild } from "@/lib/releases";

export const runtime = "edge";

/**
 * GET /api/download?os=mac&arch=arm64
 * 302-redirects to the correct latest GitHub Release asset. Unknown combos fall
 * back to the download page so the visitor can choose.
 *
 * Every real download button funnels through here, so this is also where we
 * count downloads: a server-side "download" event fires before the redirect,
 * which catches every platform and isn't affected by client-side ad-blockers.
 * Tracking is best-effort — a failure here must never block the download.
 */
export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const key = resolveBuild(searchParams.get("os"), searchParams.get("arch"));
  if (!key) {
    return NextResponse.redirect(new URL("/download", req.url), 302);
  }
  const build = BUILDS[key];
  await track("download", {
    build: key,
    os: build.platform,
    arch: build.arch,
  }).catch(() => {});
  return NextResponse.redirect(latestAssetUrl(build.asset), 302);
}
