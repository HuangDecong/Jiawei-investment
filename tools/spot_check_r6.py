# -*- coding: utf-8 -*-
"""本轮交付抽查：③ 全历史堆叠图 / ④ 10 只 ETF / ⑤ 因子手册。"""
import os
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
    for _ in range(70):
        try:
            if pg.get_by_text("综合判断", exact=False).count() > 0:
                break
        except Exception:
            pass
        pg.wait_for_timeout(4000)
    time.sleep(3)

    # ③ 行业占比变迁
    pg.get_by_text("③ 行业占比变迁", exact=False).first.click(timeout=20000)
    time.sleep(12)
    t3 = pg.inner_text("body")
    log("③ 2019Q1 出现:", "2019Q1" in t3, "| 偏股混合口径注记:",
        "偏股混合型单类型逐期重抓" in t3)
    pg.screenshot(path="output/shots/r6_03.png")

    # ④ ETF 资金流
    pg.get_by_text("④ ETF 资金流", exact=False).first.click(timeout=20000)
    time.sleep(10)
    t4 = pg.inner_text("body")
    log("④ 512690 出现:", "512690" in t4, "| 512720 出现:", "512720" in t4,
        "| 军工:", "军工" in t4)
    pg.screenshot(path="output/shots/r6_04.png")

    # ⑤ 因子研究
    pg.get_by_text("⑤ 因子研究", exact=False).first.click(timeout=20000)
    time.sleep(10)
    t5 = pg.inner_text("body")
    log("⑤ 因子手册:", "因子手册" in t5, "| F12 卡:", "配置系数（超配倍数）" in t5,
        "| 检验标准总则:", "检验标准总则" in t5, "| 检验表迁入:", "表 1　RankIC 汇总" in t5)
    pg.screenshot(path="output/shots/r6_05.png")

    log("console错误:", len(errs))
    for e in errs[:5]:
        log("  ·", e[:150])
    log("DONE")
    sys.stdout.flush()
    os._exit(0)
