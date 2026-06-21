"use client";

import { useEffect } from "react";
import { track } from "@vercel/analytics";

/**
 * Approximate "how long visitors stay" — Vercel Web Analytics doesn't report
 * visit duration natively, so we measure time from page load until the tab is
 * hidden/closed and send a single bucketed custom event ("visit_duration").
 *
 * Duration is bucketed (not raw seconds) to keep the custom-event dimension
 * low-cardinality and to avoid recording anything that could identify a visitor.
 * Lives in the root layout, which doesn't remount on client-side navigation, so
 * this captures whole-session time on the site rather than per-page time.
 */
export function EngagementTracker() {
  useEffect(() => {
    const start = Date.now();
    let sent = false;

    const send = () => {
      if (sent) return;
      sent = true;
      const seconds = Math.round((Date.now() - start) / 1000);
      const bucket =
        seconds < 10 ? "0-10s" :
        seconds < 30 ? "10-30s" :
        seconds < 60 ? "30-60s" :
        seconds < 180 ? "1-3m" :
        seconds < 600 ? "3-10m" : "10m+";
      track("visit_duration", { bucket });
    };

    // `pagehide` is the most reliable "leaving" signal across browsers; the
    // visibilitychange handler catches tab switches / mobile backgrounding.
    const onVisibility = () => {
      if (document.visibilityState === "hidden") send();
    };
    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("pagehide", send);

    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("pagehide", send);
    };
  }, []);

  return null;
}
