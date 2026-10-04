# -*- coding: utf-8 -*-
"""预热 streamlit cache：打开首页并等待重算完成（页面出现渲染内容）。"""
from __future__ import annotations

import sys
import time

from playwright.sync_api import sync_playwright

URL = "http://localhost:8509"
MARK = "综合判断"          # 首页面板①的标题文字
MAX_WAIT = int(sys.argv[1]) * 1000 if len(sys.argv) > 1 else 900_000


def main() -> int:
    t0 = time.time()
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        pg = br.new_page(viewport={"width": 1400, "height": 900})
        pg.goto(URL, wait_until="commit", timeout=90_000)
        while time.time() - t0 < MAX_WAIT / 1000:
            try:
                if pg.get_by_text(MARK, exact=False).count() > 0:
                    pg.wait_for_timeout(3000)
                    print(f"预热完成，用时 {time.time()-t0:.0f}s")
                    br.close()
                    return 0
            except Exception:
                pass
            pg.wait_for_timeout(5000)
        print(f"超时（{MAX_WAIT/1000:.0f}s）仍未出现标记")
        br.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
