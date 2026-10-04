# -*- coding: utf-8 -*-
"""最终浏览器验证：① 页标签完整性 + ⑥ 页无 NameError + console 干净。"""
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright


def log(*a):
    print(*a)
    sys.stdout.flush()


errs = []
with sync_playwright() as pw:
    br = pw.chromium.launch(headless=True, args=["--no-sandbox"])
    pg = br.new_page(viewport={"width": 1500, "height": 950})
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8509", wait_until="domcontentloaded", timeout=60000)
    for _ in range(60):
        try:
            if pg.get_by_text("综合判断", exact=False).count() > 0:
                break
        except Exception:
            pass
        pg.wait_for_timeout(4000)
    time.sleep(3)
    txt = pg.inner_text("body")
    log("嘻:", txt.count("嘻"), "| 风:", txt.count("风"),
          "| Wind:", txt.count("Wind"), "| HHI:", txt.count("HHI"))
    log("残留**:", len(re.findall(r"\*\*", txt)))
    log("期数表述:", re.findall(r"样本仅 \d+ 期|可建模区间 \d+ 期", txt))
    pg.screenshot(path="output/shots/final_01.png")

    pg.get_by_text("⑥ 轮动策略与因子裁决", exact=False).first.click(timeout=20000)
    time.sleep(20)
    log("⑥ NameError框:", pg.get_by_text("NameError", exact=False).count())
    log("⑥ console错误:", len(errs))
    for e in errs[:5]:
        log("  ·", e[:150])
    pg.screenshot(path="output/shots/final_06.png")
    br.close()
log("DONE")
os._exit(0)
