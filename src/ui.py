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
             tag1: str = "事实", tag2: str = "经济含义",
             tag3: str = "投资含义") -> None:
    """图下方的三段式分析块。

    为什么不再叫「读图」
    --------------------
    「读图：横轴是…、纵轴是…」只教会读者看坐标轴，没有提供任何判断。
    这里改成三段实质内容：数据本身说了什么 → 背后的经济机制 →
    对仓位/买卖的具体含义。读者要的是后两段。
    """
    rows = []
    if fact:
        rows.append((tag1, fact, C.COLOR["price"], "#FFFFFF"))
    if econ:
        rows.append((tag2, econ, C.COLOR["accent"], C.COLOR["panel"]))
    if invest:
        rows.append((tag3, invest, C.COLOR["threshold"], "#FFF9F9"))
    body = ""
    for tag, txt, col, bg in rows:
        body += (
            f'<div style="display:flex;gap:10px;padding:9px 14px;background:{bg};'
            f'border-top:1px solid {C.COLOR["grid"]}">'
            f'<div style="flex:0 0 62px;font-size:{FS_SMALL}px;font-weight:700;'
            f'color:{col};line-height:1.85">{tag}</div>'
            f'<div style="flex:1;font-size:{FS_BODY}px;line-height:1.95;'
            f'color:#26374A">{txt}</div></div>')
    h(f'<div style="border:1px solid {C.COLOR["line"]};border-radius:9px;'
      f'overflow:hidden;margin:6px 0 18px">{body}</div>')


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
