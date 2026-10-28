# -*- coding: utf-8 -*-
"""AppTest 逐页异常检测：页面清单从 app.py 现读，避免改了导航就漏测。

用法：python tools/apptest_all_pages.py
输出：每页 exception 数、图表/表格数量、总异常数（非 0 即视为回归）。
"""
import os
import re
import sys

from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")


def pages() -> list[str]:
    """从 app.py 的 PAGES 定义里读出页面清单。

    硬编码页面名会随导航调整而失效——曾因页面改名让测试直接抛
    ValueError: '① 总览' is not in list，看起来像业务报错，
    其实是测试自己过期了。
    """
    src = open(APP, encoding="utf-8").read()
    m = re.search(r"PAGES\s*=\s*\[(.*?)\]", src, re.S)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


def main() -> int:
    ps = pages()
    if not ps:
        print("未能从 app.py 解析 PAGES")
        return 2
    print(f"共 {len(ps)} 页：{'、'.join(ps)}")
    total = 0
    for p in ps:
        at = AppTest.from_file(APP, default_timeout=2400)
        at.run()
        at.radio(key="page").set_value(p)
        at.run()
        n = len(at.exception)
        total += n
        charts = len(at.get("plotly_chart"))
        dfs = len(at.dataframe)
        mets = len(at.metric)
        flag = "OK " if n == 0 else "ERR"
        print(f"  {flag} {p}　异常 {n}　图表 {charts}　表格 {dfs}　指标卡 {mets}")
        for e in at.exception:
            print("      ERR:", str(e.value)[:300].replace("\n", " | "))
        if n == 0 and charts + dfs == 0:
            print("      警告：该页没有任何图表或表格，内容可能未渲染")
    print("TOTAL exceptions =", total)
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
