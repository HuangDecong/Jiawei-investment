# -*- coding: utf-8 -*-
"""逐页分段截图（新版 6 页），用于视觉走查。

用法：
    streamlit run app.py --server.port 8509 --server.headless true
    python tools/shot6.py            # 全部页面
    python tools/shot6.py ①          # 只截第 1 页
"""
from __future__ import annotations

import os
import sys
import time

from playwright.sync_api import sync_playwright

URL = "http://localhost:8509"
PAGES = ["① 结论总览", "② 本期解读", "③ 单只基金", "④ 行业拥挤度",
         "⑤ 因子检验", "⑥ 方法与口径"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "output", "shots6")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    pw = sync_playwright().start()
    br = None
    try:
        import glob
        exe = (glob.glob(os.path.expanduser(
            "~/AppData/Local/ms-playwright/chromium-*/chrome-win64/chrome.exe"))
            or [r"C://Program Files (x86)//Microsoft//Edge//Application//msedge.exe"])
        br = pw.chromium.launch(headless=True, executable_path=exe[0],
                                args=["--no-sandbox", "--disable-gpu",
                                      "--disable-dev-shm-usage"])
        pg = br.new_page(viewport={"width": 1680, "height": 1000})
        pg.goto(URL, wait_until="commit", timeout=120_000)
        pg.wait_for_timeout(9000)
        for i, name in enumerate(PAGES, 1):
            if only and not name.startswith(only):
                continue
            try:
                pg.get_by_text(name, exact=False).first.click(timeout=15_000)
            except Exception as e:
                print(f"! 点击失败 {name}: {type(e).__name__}")
                continue
            pg.wait_for_timeout(7000)
            # 分段截：每个视口一张，最多 4 段
            for k in range(4):
                # Streamlit 的滚动容器不是 window，而是内部 section；
                # 用鼠标滚轮事件最稳（window.scrollTo 对内部容器无效）
                pg.mouse.move(900, 500)
                for _ in range(6 if k else 0):
                    pg.mouse.wheel(0, 500)
                    pg.wait_for_timeout(120)
                pg.wait_for_timeout(800)
                fn = os.path.join(OUT, f"{i:02d}_{k}.png")
                pg.screenshot(path=fn)
                print("✓", os.path.basename(fn))
            pg.mouse.move(900, 500)
            for _ in range(40):
                pg.mouse.wheel(0, -600)
            pg.wait_for_timeout(600)
    finally:
        if br is not None:
            br.close()
        pw.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
