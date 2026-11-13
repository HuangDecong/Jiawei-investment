# -*- coding: utf-8 -*-
"""部署路径测试：**没有数据库、没有抓取依赖**时，六页是否仍然可用。

为什么必须单独测
----------------
本地跑得通 ≠ 线上跑得通。线上（Streamlit Community Cloud）只拿到 git 里的文件：
  · `data/db/*.db` 被 .gitignore 排除 → 前三页必须走 `data/derived/*.csv.gz` 快照；
  · 依赖装的是 requirements.txt，2026-11-11 就因为少了 lxml / akshare，
    ④ 季报页整页显示「接口未取到」，读者以为是基金没有数据。
这两个条件本地永远碰不到（数据库在、包也全），所以要显式模拟。

模拟方式
--------
· `FCD_FORCE_FALLBACK=1`：canon 强制走压缩快照（这个开关本来就是为此留的）；
· 把 `qreport.DB_PATH` 指到一个空库：季报页没有任何缓存，只能现场抓或明确报错；
· `--no-deps`：把依赖探测结果钉成「lxml、akshare 都缺」，验证提示文案与
  「不再逐个接口空转」的行为。

用法
----
    python tools/test_deploy_paths.py             # 无库 + 缺依赖（线上最坏情况）
    python tools/test_deploy_paths.py --with-deps # 无库 + 依赖齐全（会真的联网抓一只基金）

退出码 0＝全部通过；非 0＝有页面异常或关键提示缺失。
"""
from __future__ import annotations

import argparse
import os
import re
import sys

from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")
sys.path.insert(0, os.path.join(ROOT, "src"))


def pages() -> list[str]:
    src = open(APP, encoding="utf-8").read()
    m = re.search(r"PAGES\s*=\s*\[(.*?)\]", src, re.S)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


def page_text(at) -> str:
    """把页面上的文字拼起来，用于断言关键提示。

    注意必须包含 `html` 元素：本项目的正文走 `st.html`（见 app.py 的 _h），
    不在 `at.markdown` 里。只查 markdown 会得出「页面什么都没写」的假结论。
    """
    out = []
    for kind in ("html", "markdown", "caption", "text", "info", "warning", "error"):
        try:
            for el in at.get(kind):
                out.append(str(getattr(el, "value", "") or ""))
        except Exception:  # noqa: BLE001
            pass
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-deps", action="store_true",
                    help="不模拟缺依赖（会真的联网抓一只基金，较慢）")
    args = ap.parse_args()

    os.environ["FCD_FORCE_FALLBACK"] = "1"     # canon 走 csv.gz
    os.environ["FCD_OFFLINE_TEST"] = "1"
    import qreport as Q                        # noqa: E402

    tmp_db = os.path.join(ROOT, "output", "_deploy_sim_q.db")
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    os.environ["FCD_NO_RELOAD"] = "1"          # 否则热重载会把 _DEP_PROBE 重置
    Q.DB_PATH = tmp_db                         # 空库：模拟线上没有 data/db
    Q.ensure_db()
    if not args.with_deps:
        Q._DEP_PROBE = ["lxml", "akshare"]     # 模拟 requirements 漏装

    mode = "依赖齐全（会联网）" if args.with_deps else "缺依赖（最坏情况）"
    print(f"部署路径模拟：无数据库 · {mode}")
    bad = 0
    for p in pages():
        at = AppTest.from_file(APP, default_timeout=2400)
        at.run()
        at.radio(key="page").set_value(p)
        at.run()
        n = len(at.exception)
        charts = len(at.get("plotly_chart"))
        dfs = len(at.dataframe)
        flag = "OK " if n == 0 else "ERR"
        print(f"  {flag} {p}　异常 {n}　图表 {charts}　表格 {dfs}")
        for e in at.exception:
            print("      ERR:", str(e.value)[:300].replace("\n", " | "))
        if n:
            bad += 1
        if p.startswith("④") and not args.with_deps:
            txt = page_text(at)
            if "缺少依赖" not in txt:
                print("      MISS: ④ 页未出现「缺少依赖」提示 —— 读者会误判成基金无数据")
                bad += 1
            elif "可能是新成立或已清盘" in txt:
                print("      MISS: ④ 页仍把抓取失败写成「可能已清盘」")
                bad += 1
    print("TOTAL 失败页 =", bad)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
