# -*- coding: utf-8 -*-
"""
probe_console.py —— 用无头浏览器复现前端 React 报错并**定位到具体页面**

背景
----
用户报告三种 React DOM 错误：
    1. NotFoundError: 未能在"节点"上执行"removeChild"：被移除的节点不是该节点的子节点
    2. NotFoundError: 未能在"节点"上执行"insertBefore"：新节点要插入的节点不是该节点的子节点

这两类错误在 Streamlit 中几乎总是同一个根因：`st.markdown(..., unsafe_allow_html=True)`
里存在**畸形 HTML**（标签未闭合 / 块级元素被塞进行内元素 / 表格被包在段落里）。
浏览器会自动"纠正"这种 HTML（把 <table> 从 <p> 里提出来等），
于是真实 DOM 与 React 的虚拟 DOM 不再一致；切换页面触发重渲染时即抛 removeChild/insertBefore。

本脚本逐个页面访问并收集 console error，给出「哪个页面、哪条消息」的映射，
使修复可验证 —— 而不是靠猜测。
"""
from __future__ import annotations

import json
import sys

from playwright.sync_api import sync_playwright

URL = "http://localhost:8509"
PAGES = ["① 总览仪表盘", "② 拥挤度总览", "③ 行业占比变迁", "④ ETF 资金流",
         "⑤ 因子研究", "⑥ 轮动策略与因子裁决", "⑦ 稳健性检验",
         "⑧ 数据质量与溯源", "⑨ 方法说明与数据来源"]

BAD = ("removeChild", "insertBefore", "NotFoundError", "not a child",
       "The node to be removed", "The node before which")


def main() -> int:
    find = "--find" in sys.argv       # 只找页面，不逐页切换
    pw = sync_playwright().start()
    br = None
    try:
        br = pw.chromium.launch(headless=True,
                                args=["--no-sandbox", "--disable-gpu",
                                      "--disable-dev-shm-usage"])
        pg = br.new_page(viewport={"width": 1680, "height": 1050})
        errs = []
        pg.on("console", lambda m: errs.append((m.type, m.text))
              if m.type in ("error",) else None)
        pg.on("pageerror", lambda e: errs.append(("pageerror", str(e))))

        pg.goto(URL, wait_until="commit", timeout=90_000)
        pg.wait_for_timeout(6000)

        def snap(tag):
            hits = [t for k, t in errs if any(b.lower() in t.lower() for b in BAD)]
            return tag, len(errs), len(hits), (hits[:1] or [""])[0][:150]

        rows = [snap("首页(①)")]
        if not find:
            for name in PAGES[1:]:
                try:
                    pg.get_by_text(name, exact=False).first.click(timeout=12_000)
                    pg.wait_for_timeout(5000)
                    rows.append(snap(name))
                except Exception as e:                        # pragma: no cover
                    rows.append((name, -1, -1, f"点击失败 {type(e).__name__}"))

        print(f"{'页面':<26}{'console总数':>10}{'React DOM 错误':>14}  首条")
        print("-" * 108)
        for name, tot, hits, first in rows:
            flag = "  <<< 有问题" if hits > 0 else ""
            print(f"{name:<26}{tot:>10}{hits:>14}  {first}{flag}")
        total = sum(h for _, _, h, _ in rows if h > 0)
        print("-" * 108)
        print(f"React DOM 错误合计 = {total}")
        seen = set()
        for k, t in errs:
            if any(b.lower() in t.lower() for b in BAD):
                key = t[:220]
                if key not in seen:
                    seen.add(key)
                    print(f"  · [{k}] {t[:260]}")
        return 0
    finally:
        # 浏览器 teardown 偶发 "Connection closed while reading from the driver"，
        # 这是 Playwright 驱动的已知收尾噪声，不应让脚本带异常退出、更不应吞掉结果。
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
