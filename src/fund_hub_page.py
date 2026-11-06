# -*- coding: utf-8 -*-
"""④ 单只基金：一只基金的两个视角，一个选择器。

为什么合并
----------
原来「单只基金」和「季报全解」是两页：同一只基金要选两次、两份表头、
两份前十大，读者要在两个页面之间对照同一个数。但它们问的其实是同一件事的
两个层次：

    · 季报全解：季报**原文披露了什么**（资产、份额、前十、买卖、债券、财务）；
    · 持仓结构：用持仓**反推的结构**（申万 31 行、与同行比、历史漂移）。

一个看「原文」，一个看「结构」，口径不同、不能互相替代，但也没必要分两页。
合并后只保留一个选择器，切换视角不换基金，页数从 7 页减到 6 页。

与「② 季报解读」的分工
----------------------
② 是全市场横截面（4,900 只基金汇总，回答「机构整体在做什么」）；
本页是单只基金纵截面（一只基金十几期，回答「这只产品在做什么」）。
两者样本范围不同，不重复。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import canon as cn
import fund_page
import qreport_page
import ui


def _funds() -> pd.DataFrame:
    p = cn.profiles()
    if p is None or p.empty:
        return pd.DataFrame(columns=["code", "name", "scale_yi"])
    d = p.copy()
    d["code"] = d["code"].astype(str).str.zfill(6)
    if "scale_yi" not in d.columns:
        d["scale_yi"] = np.nan
    return d.sort_values("scale_yi", ascending=False, na_position="last")


def _views() -> dict[str, str]:
    return {
        "季报全解：这一季季报到底说了什么":
            "资产配置 / 行业分类 / 份额 / 前十名 / 累计买卖 / 债券 / 财务指标，"
            "全部为季报原文披露值，缺数据就留空，不用替代口径。",
        "持仓结构：它在赌什么、变没变":
            "用前十大或全持仓反推申万 31 行结构，和同行比、看历史漂移、"
            "算拥挤暴露——回答「这只基金的结构性押注」。",
    }


def render(h) -> None:
    funds = _funds()
    if funds.empty:
        h('<div class="cg-note">基金档案尚未就绪，先运行 '
          '<span class="mono">python tools/fetch_fund_profile.py</span>。</div>')
        return

    c1, c2 = st.columns([0.46, 0.54])
    with c1:
        opts = ui.fund_options(funds)
        pick = st.selectbox("选基金（可输入代码或名称搜索，按规模排序）",
                            opts, index=0, key="hub_fund")
        code = pick.split("　")[0].strip().zfill(6)
    with c2:
        view = st.radio("看哪一面", list(_views()), horizontal=True,
                        key="hub_view",
                        help="两个视角用同一只基金、不同口径：\n"
                             "季报全解＝季报原文披露值；持仓结构＝前十大/全持仓反推的申万行业结构。")
        st.caption(_views()[view])

    if view.startswith("季报全解"):
        qreport_page.render(h, code=code, show_picker=False)
    else:
        fund_page.render(h, code=code, show_picker=False)
