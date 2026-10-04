# -*- coding: utf-8 -*-
"""逐页截图，用于视觉走查（页面简洁性与美观度评估）。"""
from __future__ import annotations

import os
import sys

from playwright.sync_api import sync_playwright

URL = "http://localhost:8509"
PAGES = ["① 总览仪表盘", "② 拥挤度总览", "③ 行业占比变迁", "④ ETF 资金流",
         "⑤ 因子研究", "⑥ 轮动策略与因子裁决", "⑦ 稳健性检验",
         "⑧ 数据质量与溯源", "⑨ 方法说明与数据来源"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "output", "shots")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    pw = sync_playwright().start()
    br = None
    try:
        br = pw.chromium.launch(headless=True,
                                args=["--no-sandbox", "--disable-gpu",
                                      "--disable-dev-shm-usage"])
        pg = br.new_page(viewport={"width": 1600, "height": 1000})
        pg.goto(URL, wait_until="commit", timeout=90_000)
        pg.wait_for_timeout(7000)
        pg.screenshot(path=os.path.join(OUT, "00_dashboard.png"))
        print("✓ 00_dashboard.png")
        for i, name in enumerate(PAGES[1:], 1):
            if only and only not in name:
                continue
            try:
                pg.get_by_text(name, exact=False).first.click(timeout=12_000)
                pg.wait_for_timeout(5500)
                fn = f"{i:02d}_{name.split()[0]}.png"
                pg.screenshot(path=os.path.join(OUT, fn))
                print(f"✓ {fn}")
            except Exception as e:
                print(f"✗ {name}: {type(e).__name__}")
        return 0
    finally:
        try:
            if br is not None:
                br.close()
        except Exception:
            pass
        try:
            pw.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
