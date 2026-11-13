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
import version_info as vinfo


def _body(page: str = "") -> None:
    """侧边栏与弹窗共用的主体：搜索 + 指标卡 + 口径速查。"""
    _ctxkey = _ctx()
    kw = st.text_input("搜索指标", value="", key=f"gl_kw_{_ctxkey}",
                       placeholder="如：拥挤 / 广度 / GK / 分位")
    # 搜索词一变，就把选择重置到「第一条命中」。
    # 不重置的话，selectbox 会保留上一次的选项（比如搜「价格分位」还停在 CSI），
    # 看起来像「搜了但没搜到」。
    if st.session_state.get(f"_gl_lastkw_{_ctxkey}") != kw:
        st.session_state[f"_gl_lastkw_{_ctxkey}"] = kw
        st.session_state.pop(f"gl_pick_{_ctxkey}", None)
    keys = g.search(kw)
    if not keys:
        st.caption("没有匹配的指标，试试「分位」「广度」「主动」。")
        keys = list(g.ITEMS)

    def _lab(k):
        v = g.ITEMS[k]
        return f"{v['cat']} · {v['name']}"

    pick = st.selectbox("选择指标", keys, format_func=_lab, key=f"gl_pick_{_ctxkey}")
    # 词典卡平均 500–900px 高，直接铺开会把「三套口径」「版本行」顶到侧栏外面——
    # 读者以为没有版本信息。给卡片一个固定高度、内部滚动，底部信息就永远可见。
    # 侧栏窄、弹窗宽：同一个容器在弹窗里给更高，避免刚打开就要滚。
    _h = 520 if st.session_state.get("_gl_in_dialog") else 300
    with st.container(height=_h, border=True):
        st.markdown(g.card_html(pick, C.COLOR, compact=not st.session_state.get("_gl_in_dialog")),
                    unsafe_allow_html=True)


    # 光靠搜索不够：读者不知道「有哪些词可以查」。
    # 给一份按分类排的全量清单，扫一眼就知道覆盖到哪。
    with st.expander(f"全部 {len(g.ITEMS)} 个指标一览（按分类）", expanded=False):
        # 弹窗够宽，用三列表格；侧边栏太窄，只列名称
        if st.session_state.get("_gl_in_dialog"):
            st.markdown(g.compact_table(C.COLOR), unsafe_allow_html=True)
        else:
            st.markdown(g.name_list_html(C.COLOR), unsafe_allow_html=True)


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
            f'margin-bottom:6px">{len(g.ITEMS)} 个指标的含义、原理、公式、计算过程、阈值、'
            f'标准来源、经济内容、投资含义与常见误用。可直接输入关键词搜索。</div>',
            unsafe_allow_html=True)
        _body(page)
        st.markdown("---")
        # 「三套口径」的完整对照表在 ⑥ 方法页；侧栏只留一句指路，
        # 不再把同一张表的两段说明抄一遍（那是侧栏被撑长的主因之一）。
        st.markdown(
            f'<div style="font-size:10.5px;line-height:1.7;color:{C.COLOR["text_dim"]}">'
            f'<b>三套口径不能相加</b>（全持仓 / 前十大 / 个股只数）：'
            f'分母与披露范围不同，混算没有含义。完整对照见 '
            f'<b>⑥ 方法与口径 · 二、三套口径</b>。</div>',
            unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            f'<div style="font-size:10.5px;line-height:1.7;color:{C.COLOR["text_dim"]}">'
            f'当前页 <b>{page or "—"}</b>　·　侧栏可点「«」收起，'
            f'收起后用顶栏「📖 指标速查」打开同一份词典。</div>',
            unsafe_allow_html=True)
        # 版本行固定在侧栏最底部：**判断线上跑的是哪一版，只看这一行**。
        # 内容来自 version.json（构建时写入），运行时不调 git。
        _vi = vinfo.info()
        st.markdown(
            f'<div style="font-size:10px;line-height:1.6;color:#8A97A5;'
            f'border-top:1px dashed {C.COLOR["grid"]};padding-top:6px;margin-top:8px">'
            f'<b style="color:#5A6B7D">版本 {vinfo.sha7()}</b>'
            f'　·　{str(_vi.get("app_version", ""))}'
            f'<br>{str(_vi.get("build_time", ""))[:16]}'
            f'<br>{vinfo.deps_text()}'
            f'<br>口径表 {_vi.get("caliber_version") or "—"}</div>',
            unsafe_allow_html=True)
