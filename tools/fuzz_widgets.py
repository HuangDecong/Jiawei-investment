# -*- coding: utf-8 -*-
"""控件遍历测试：把每个下拉/单选/滑块的每个选项都点一遍，确认页面不报错。

为什么需要：页面能渲染 ≠ 每个选项都能渲染。典型翻车点是「某个行业只有
十几只基金持有」「某期数据不足」「某只基金某期没披露」这类边缘样本——
默认值永远碰不到，用户一点就白屏。

用法：
    python tools/fuzz_widgets.py            # 全量（约 6–10 分钟）
    python tools/fuzz_widgets.py --fast     # 只试首尾选项
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

FAILS: list[str] = []
N_RUN = 0


def run(at: AppTest, tag: str) -> None:
    global N_RUN
    N_RUN += 1
    at.run()
    if at.exception:
        msg = str(at.exception[0].value)[:200]
        FAILS.append(f"{tag} → {msg}")
        print(f"  ✗ {tag}: {msg}")
        try:                      # 出错后回到干净状态，避免连锁误报
            at.run()
        except Exception:
            pass


def pick(seq: list, fast: bool) -> list:
    if not fast or len(seq) <= 3:
        return seq
    return [seq[0], seq[len(seq) // 2], seq[-1]]


def fuzz_crowding(at: AppTest, fast: bool) -> None:
    at.radio(key="page").set_value("③ 行业拥挤度")
    run(at, "③ 进入页面")
    focus = [w for w in at.selectbox if w.key == "csi_focus"][0]
    opts = list(focus.options)
    for o in pick(opts, fast):
        focus = [w for w in at.selectbox if w.key == "csi_focus"][0]
        focus.set_value(o)
        run(at, f"③ 聚焦={o[:14]}")
    # 窗口 × 阈值 × 分位带
    for wlab in list([w for w in at.selectbox if w.key == "csi_win"][0].options):
        [w for w in at.selectbox if w.key == "csi_win"][0].set_value(wlab)
        run(at, f"③ 窗口={wlab}")
    for thr in (60, 95):
        [w for w in at.slider if w.key == "csi_thr"][0].set_value(thr)
        run(at, f"③ 阈值={thr}")
    [w for w in at.checkbox if w.key == "csi_band"][0].set_value(False)
    run(at, "③ 关闭分位带")
    # 双分位页的行业选择与表格列
    sel3 = [w for w in at.selectbox if w.key == "sel_ind3"]
    if sel3:
        for o in pick(list(sel3[0].options), fast):
            [w for w in at.selectbox if w.key == "sel_ind3"][0].set_value(o)
            run(at, f"③ 双分位={o[:10]}")
    bar = [w for w in at.selectbox if w.key == "csi_bar_q"]
    if bar:
        for o in pick(list(bar[0].options), fast):
            [w for w in at.selectbox if w.key == "csi_bar_q"][0].set_value(o)
            run(at, f"③ 排名期={o}")
    for o in list([w for w in at.radio if w.key == "map_tbl_cols"][0].options):
        [w for w in at.radio if w.key == "map_tbl_cols"][0].set_value(o)
        run(at, f"③ 表格={o[:8]}")


def fuzz_fund(at: AppTest, fast: bool) -> None:
    """④ 单只基金：一个选择器、两个视角。

    合并后页面里有两套内容（季报全解 / 持仓结构），控件 key 也不同，
    所以先切视角，再把当前视角下的控件点一遍。
    """
    at.radio(key="page").set_value("④ 单只基金")
    run(at, "④ 进入页面")
    hub = [w for w in at.selectbox if w.key == "hub_fund"]
    if not hub:
        return
    for o in pick(list(hub[0].options), True):
        [w for w in at.selectbox if w.key == "hub_fund"][0].set_value(o)
        run(at, f"④ 基金={o[:16]}")
    views = [w for w in at.radio if w.key == "hub_view"][0]
    for v in list(views.options):
        [w for w in at.radio if w.key == "hub_view"][0].set_value(v)
        run(at, f"④ 视角={v[:10]}")
        if v.startswith("持仓结构"):
            for key in ("fund_basis", "mkt_metric", "mkt_struct_mode",
                        "fund_struct_mode"):
                ws = [w for w in at.radio if w.key == key]
                if not ws:
                    continue
                for o in list(ws[0].options):
                    [w for w in at.radio if w.key == key][0].set_value(o)
                    run(at, f"④ {key}={o[:8]}")
            mi = [w for w in at.selectbox if w.key == "mkt_ind"]
            if mi:
                for o in pick(list(mi[0].options), fast):
                    [w for w in at.selectbox if w.key == "mkt_ind"][0].set_value(o)
                    run(at, f"④ 全市场行业={o[:8]}")
            fq = [w for w in at.selectbox if w.key == "fund_q_pick"]
            if fq:
                for o in pick(list(fq[0].options), fast):
                    [w for w in at.selectbox if w.key == "fund_q_pick"][0].set_value(o)
                    run(at, f"④ 结构期次={o}")
        else:
            qr = [w for w in at.selectbox if w.key == "qr_q"]
            if qr:
                for o in pick(list(qr[0].options), fast):
                    [w for w in at.selectbox if w.key == "qr_q"][0].set_value(o)
                    run(at, f"④ 季报季度={o}")


def fuzz_misc(at: AppTest) -> None:
    at.radio(key="page").set_value("② 本期解读（全市场）")
    run(at, "② 进入页面")
    for o in list([w for w in at.radio if w.key == "stock_basis"][0].options):
        [w for w in at.radio if w.key == "stock_basis"][0].set_value(o)
        run(at, f"② 个股口径={o[:10]}")
    for pg in ("① 结论总览", "⑤ 因子检验", "⑥ 方法与口径"):
        at.radio(key="page").set_value(pg)
        run(at, f"{pg} 进入页面")


def fuzz_qreport(at: AppTest, fast: bool) -> None:
    """④ 单只基金 · 季报全解：季度选择器现在覆盖全历史（40 期），只抽首/中/末。"""
    at.radio(key="page").set_value("④ 单只基金")
    run(at, "④ 进入季报页")
    for w in [w for w in at.radio if w.key == "hub_view"]:
        if not str(w.value).startswith("季报全解"):
            w.set_value([o for o in w.options if str(o).startswith("季报全解")][0])
            run(at, "④ 切到季报全解")
    qr = [w for w in at.selectbox if w.key == "qr_q"]
    if not qr:
        return
    for q in pick(list(qr[0].options), fast):
        [w for w in at.selectbox if w.key == "qr_q"][0].set_value(q)
        run(at, f"④ 季度={q}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=1200)
    at.run()
    fuzz_crowding(at, args.fast)
    fuzz_fund(at, args.fast)
    fuzz_qreport(at, args.fast)
    fuzz_misc(at)
    print(f"\n共 {N_RUN} 次渲染，失败 {len(FAILS)} 项，用时 {time.time() - t0:.0f}s")
    for f in FAILS:
        print(" -", f)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
