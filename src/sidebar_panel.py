# -*- coding: utf-8 -*-
"""左侧边栏：指标速查（可搜索）+ 口径速查 + 页面导航提示。

为什么放在侧边栏
----------------
指标说明如果放在正文里，会把页面撑成教科书；放在代码注释里，用户看不到。
侧边栏是「随时可查、不占正文」的位置：读者看到任何一个数，往左一瞥就能查到
它的公式、阈值、标准来源与常见误用。
"""
from __future__ import annotations

import streamlit as st

import config as C
import glossary as g


def _body(page: str = "") -> None:
    """侧边栏与弹窗共用的主体：搜索 + 指标卡 + 口径速查。"""
    kw = st.text_input("搜索指标", value="", key=f"gl_kw_{_ctx()}",
                       placeholder="如：拥挤 / 广度 / GK / 分位")
    keys = g.search(kw)
    if not keys:
        st.caption("没有匹配的指标，试试「分位」「广度」「主动」。")
        keys = list(g.ITEMS)

    def _lab(k):
        v = g.ITEMS[k]
        return f"{v['cat']} · {v['name']}"

    pick = st.selectbox("选择指标", keys, format_func=_lab, key=f"gl_pick_{_ctx()}")
    st.markdown(g.card_html(pick, C.COLOR), unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-size:10.5px;line-height:1.7;color:{C.COLOR["text_dim"]};'
        f'margin-top:6px">'
        f'数据：东方财富基金定期报告（主口径）　行业：申万一级 31 个<br>'
        f'数字全部由 <span style="font-family:monospace">src/canon.py</span> 统一提供。</div>',
        unsafe_allow_html=True)


def _ctx() -> str:
    """同一页面里侧边栏与弹窗有两个容器，widget key 必须不同，否则冲突。"""
    return "dlg" if st.session_state.get("_gl_in_dialog") else "sb"


@st.dialog("指标速查 · 含义 / 公式 / 阈值 / 来源 / 投资含义", width="large")
def glossary_dialog() -> None:
    """顶栏「指标速查」按钮弹出的可搜索词典。"""
    st.session_state["_gl_in_dialog"] = True
    try:
        _body()
    finally:
        st.session_state["_gl_in_dialog"] = False
    if st.button("关闭", key="gl_close"):
        st.session_state["gl_show"] = False
        st.rerun()


def render(page: str = "") -> None:
    with st.sidebar:
        st.markdown(
            f'<div style="font-size:15px;font-weight:700;margin:2px 0 2px">指标速查</div>'
            f'<div style="font-size:10.5px;color:{C.COLOR["text_dim"]};line-height:1.6;'
            f'margin-bottom:6px">26 个指标的含义、原理、公式、计算过程、阈值、'
            f'标准来源、经济内容、投资含义与常见误用。可直接输入关键词搜索。</div>',
            unsafe_allow_html=True)
        _body(page)
        st.markdown("---")
        st.markdown(
            f'<div style="font-size:12px;font-weight:700;margin-bottom:3px">三套口径</div>'
            f'<div style="font-size:10.5px;line-height:1.75;color:#37485C">'
            f'<b>A 全持仓</b>（半年报）：占净值比，可做主动/被动分解，最准但一年两次。<br>'
            f'<b>B 前十大</b>（季报）：每只基金取前 10 只再归一化，31 期都有，'
            f'用来算广度与环比。<br>'
            f'<b>C 个股</b>：同披露类型两期对比，看谁被建仓/移出。<br>'
            f'<span style="color:{C.COLOR["threshold"]}">三套不能相加</span>——'
            f'它们的分母与披露范围都不同。</div>',
            unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            f'<div style="font-size:10.5px;line-height:1.7;color:{C.COLOR["text_dim"]}">'
            f'当前页：<b>{page or "—"}</b><br>'
            f'侧边栏可以点左上角的「«」收起；收起后用顶栏的'
            f'<b>「📖 指标速查」</b>随时弹出同一份词典。</div>',
            unsafe_allow_html=True)
