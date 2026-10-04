# -*- coding: utf-8 -*-
"""验证 ⑧ 数据质量页：断点检验模块 + D7 修正后的体检表。"""
import time

from playwright.sync_api import sync_playwright

errs = []
t0 = time.time()
with sync_playwright() as pw:
    br = pw.chromium.launch(headless=True, args=["--no-sandbox"])
    pg = br.new_page(viewport={"width": 1600, "height": 1000})
    pg.on("console", lambda m: errs.append(("console", m.text)) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(("pageerror", str(e))))
    pg.goto("http://localhost:8509", wait_until="commit", timeout=90_000)
    for _ in range(120):
        try:
            if pg.get_by_text("综合判断", exact=False).count() > 0:
                break
        except Exception:
            pass
        pg.wait_for_timeout(5000)
    print(f"[1] 首页就绪 {time.time()-t0:.0f}s", flush=True)
    pg.get_by_text("⑧ 数据质量与溯源", exact=False).first.click(timeout=20_000)
    pg.wait_for_timeout(12_000)
    print(f"[2] ⑧已点击 errors={len(errs)}", flush=True)
    pg.screenshot(path="output/shots/07_⑧_v3.png")
    try:
        pg.get_by_text("口径断点敏感性检验", exact=False).first.scroll_into_view_if_needed(timeout=10_000)
        pg.wait_for_timeout(1500)
        pg.screenshot(path="output/shots/07_⑧_break_v3.png")
        print("[3] 断点检验块已定位", flush=True)
    except Exception as e:
        print("[3] 未找到:", type(e).__name__, flush=True)
    br.close()
bad = [t for _, t in errs if any(b in t for b in ("removeChild", "insertBefore", "NotFoundError"))]
print("console错误:", len(errs), " React DOM错误:", len(bad), flush=True)
for _, t in errs[:6]:
    print("  ·", t[:160], flush=True)
