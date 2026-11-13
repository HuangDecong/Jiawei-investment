# -*- coding: utf-8 -*-
"""统一的页面视觉组件：结论板、分析块、指标条。

为什么单独抽一层
----------------
之前每页各写一套 HTML，字号从 9px 到 12.5px 混用，读者反馈「看不清、不像一个网站」。
这里把三种最常用的结构固定下来，全站只调一处：

    concl_board   结论清单（带序号徽标，每条一张卡）
    analysis      图下方的三段式分析：事实 → 经济含义 → 投资含义
    stat_strip    一行 KPI 数字条（轻量，不抢结论的注意力）

统一之后还有一个好处：所有页面的信息层级一致，读者不用重新适应。
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config as C

# 正文与标题的基准字号：全站只在这里改
FS_BODY = 13.5
FS_SMALL = 12.5
FS_TITLE = 15


def concl_board(items: list[str], accent: str | None = None,
                numbered: bool = True) -> str:
    """结论清单：每条一张卡，左侧序号徽标，正文 13.5px。

    items 里的字符串可以带 <b> 等内联标签；不要传整段 <div>。
    """
    accent = accent or C.COLOR["price"]
    out = []
    for i, txt in enumerate(items, 1):
        badge = (
            f'<div style="flex:0 0 24px;height:24px;border-radius:12px;background:{accent};'
            f'color:#fff;font-size:12px;font-weight:700;display:flex;align-items:center;'
            f'justify-content:center;margin-top:2px">{i}</div>' if numbered else "")
        out.append(
            f'<div style="display:flex;gap:10px;background:#FFFFFF;'
            f'border:1px solid {C.COLOR["line"]};border-radius:9px;'
            f'padding:10px 14px;margin-bottom:7px">'
            f'{badge}'
            f'<div style="flex:1;font-size:{FS_BODY}px;line-height:1.95;color:#26374A">'
            f'{txt}</div></div>')
    return "".join(out)


def analysis(h, fact: str, econ: str = "", invest: str = "",
             tag1: str = "数据", tag2: str = "含义", tag3: str = "动作") -> None:
    """图下方的读数卡：**左栏数据、右栏含义与动作**。

    为什么从「竖着叠三段」改成两栏
    ------------------------------
    三段竖排时，每张图下面要占 300–400px；六张图就是 2000px 的纯文字，
    读者滚到第三张就忘了第一张在说什么。改成两栏后高度由
    max(数据, 含义+动作) 决定，实测省掉约四成纵向空间，
    而且「看到了什么 / 所以怎么做」左右并置，比上下更容易对照。

    文字本身不改：事实仍是可复算的数字，经济含义仍给机制，
    动作仍是条件句——只是不再各占一行。
    """
    if not fact and not econ and not invest:
        return
    left = (f'<div style="flex:1 1 44%;min-width:330px;padding:11px 15px 13px;'
            f'background:#FFFFFF">'
            f'<div style="font-size:{FS_SMALL}px;font-weight:700;color:{C.COLOR["price"]};'
            f'margin-bottom:3px">{tag1}</div>'
            f'<div style="font-size:{FS_BODY}px;line-height:1.9;color:#26374A">{fact}</div>'
            f'</div>') if fact else ""
    # 没有「数据」栏时（例如行业页的 KPI 已经把数字说完），
    # 含义与动作左右并排，而不是上下叠——否则整段文字会横跨 150 字宽，最难读。
    _stack = not fact
    if econ:
        right_rows = (
            f'<div style="padding:11px 15px 10px;flex:1 1 {"48%" if _stack else "auto"}">'
            f'<div style="font-size:{FS_SMALL}px;font-weight:700;color:{C.COLOR["accent"]};'
            f'margin-bottom:3px">{tag2}</div>'
            f'<div style="font-size:{FS_BODY}px;line-height:1.9;color:#26374A">{econ}</div>'
            f'</div>')
    else:
        right_rows = ""
    if invest:
        right_rows += (
            f'<div style="padding:9px 15px 12px;background:#FFF9F9;'
            f'{"border-left:1px dashed " + C.COLOR["grid"] if _stack else "border-top:1px dashed " + C.COLOR["grid"]};'
            f'flex:1 1 {"46%" if _stack else "auto"}">'
            f'<div style="font-size:{FS_SMALL}px;font-weight:700;'
            f'color:{C.COLOR["threshold"]};margin-bottom:3px">{tag3}</div>'
            f'<div style="font-size:{FS_BODY}px;line-height:1.9;color:#26374A">{invest}</div>'
            f'</div>')
    right = (f'<div style="flex:1 1 56%;min-width:340px;display:flex;flex-wrap:wrap;'
             f'{"border-left:1px solid " + C.COLOR["grid"] if not _stack else ""};'
             f'background:{C.COLOR["panel"] if not _stack else "#FFFFFF"}">'
             f'{right_rows}</div>') if right_rows else ""
    h(f'<div style="display:flex;flex-wrap:wrap;border:1px solid {C.COLOR["line"]};'
      f'border-radius:10px;overflow:hidden;margin:7px 0 20px;'
      f'box-shadow:0 1px 2px rgba(16,24,40,0.03)">{left}{right}</div>')


def data_insufficient(n, need, why: str = "", unit: str = "期") -> str:
    """统一的「样本不足」说明块：**全站只此一种写法**。

    要求（2026-11-12）：不许给看似正常的 IC 值，也不许留白；必须写清
    「样本 n=?、要求多少、为什么不做检验」，读者才知道这是边界、不是故障。
    """
    head = f"样本 n={n}，不足 {need} {unit}，不做检验"
    return (f'<div class="cg-note">{head}'
            + (f"：{why}" if why else "") + "</div>")


def stat_strip(items: list[tuple[str, str, str]], accent: str | None = None) -> str:
    """一行 KPI：标题 / 数值 / 注释。数值用主色，注释灰。"""
    accent = accent or C.COLOR["text"]
    cells = "".join(
        f'<div style="flex:1 1 0;min-width:96px;background:#FFFFFF;'
        f'border:1px solid {C.COLOR["line"]};border-radius:9px;padding:9px 12px">'
        f'<div style="font-size:{FS_SMALL}px;color:{C.COLOR["text_dim"]};'
        f'white-space:nowrap">{k}</div>'
        f'<div style="font-size:21px;font-weight:700;line-height:1.3;'
        f'color:{accent};font-variant-numeric:tabular-nums">{v}</div>'
        f'<div style="font-size:{FS_SMALL}px;color:{C.COLOR["text_dim"]};'
        f'line-height:1.55">{s}</div></div>' for k, v, s in items)
    return (f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:2px 0 6px">'
            f'{cells}</div>')


def section(title: str, sub: str = "") -> str:
    """小节标题：主标题 + 可选的一句说明（灰、同一行）。"""
    sub_html = (f'<span style="font-weight:400;font-size:{FS_SMALL}px;'
                f'color:{C.COLOR["text_dim"]};margin-left:8px">{sub}</span>'
                if sub else "")
    return (f'<div style="font-size:{FS_TITLE}px;font-weight:700;'
            f'color:{C.COLOR["text"]};margin:16px 0 7px;padding-left:9px;'
            f'border-left:4px solid {C.COLOR["price"]}">{title}{sub_html}</div>')


def source_footer(h, page: str, term: str = "") -> None:
    """页脚统一的「来源 + 公式」区。

    每个页面都必须有：谁提供数据、哪一天取的、原始披露在哪（可点）、
    用什么代码算的；再给一个按钮，直接打开该页主指标的公式与计算过程。
    `term` 是 glossary 的键（如 csi / gk_active），打开弹窗时定位到它。
    """
    import glossary as g  # noqa: PLC0415
    import sources  # noqa: PLC0415

    h(sources.source_line(page))
    # 按钮不拉满：页脚是附属信息，不该出现一个横跨半屏的按钮。
    if st.button("📖 公式与计算过程", key=f"src_gl_{page}",
                 help="打开该页主指标的定义、公式、计算步骤、阈值与来源"):
        if term and term in g.ITEMS:
            st.session_state["_gl_force_term"] = term
        st.session_state["gl_show"] = True
        st.session_state["_gl_page"] = st.session_state.get("page")
        st.rerun()
    st.caption("定义 / 公式 / 计算步骤 / 阈值 / 标准来源 / 常见误用，逐项可查；"
               "所有词条与页面上的数字同源（src/canon.py）。")


def safe_str(v, default: str = "—") -> str:
    """把 NaN / None / 空串统一成一个可展示的占位符。

    档案里有几百只基金缺名称、经理、基准（抓取时接口失败）。直接 f-string
    会显示成「nan」，在页面上和「数据缺失」分不清，审计也会判成 bug。
    """
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none") else default


def fund_options(df: pd.DataFrame, with_scale: bool = True) -> list[str]:
    """基金下拉选项：「代码　名称　规模」。

    为什么不用 pandas 的字符串加法：档案里有 368 只基金的名称是 NaN
    （抓取时接口失败），`Series + "　"` 会把这些行原样保留成 NaN，
    下拉框里就出现一个值为 `nan` 的选项；用户选中它，页面在
    `pick.split("　")` 处直接抛 AttributeError —— 白屏。

    这里逐行构造并给缺失值兜底，保证选项**一定是字符串**。
    """
    out = []
    for r in df.itertuples(index=False):
        nm = r.name if isinstance(r.name, str) and r.name.strip() else "—"
        code = str(getattr(r, "code", "")).zfill(6)
        if with_scale:
            s = getattr(r, "scale_yi", float("nan"))
            sc = f"{s:.0f}亿" if pd.notna(s) else "—"
            out.append(f"{code}　{nm}　{sc}")
        else:
            out.append(f"{code}　{nm}")
    return out
