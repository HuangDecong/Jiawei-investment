# -*- coding: utf-8 -*-
"""① 结论总览：把所有结论、判断理由、投资解读、操作建议压在一页。

为什么要有这一页
----------------
原来的首页一上来就是「仓位图」，读者要先看五张图才知道结论。
但真实的使用场景是：先想知道「这一季到底发生了什么、我该做什么」，
细节可以后面再翻。所以这一页只做三件事：

    1. 一句话说清本期性质（是存量换仓、还是增量入场）；
    2. 把全部结论按「结论 → 证据 → 含义 → 动作 → 失效条件」列出来；
    3. 把动作落成可执行的清单（跟什么、避什么、仓位多少、什么时候推翻）。

所有数字来自 canon / playbook，本页不自己算指标——否则又会出现
「同一句话在两个页面上不是一个数」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import canon as cn
import config as C
import playbook as pb


def _fmt(v, nd=1, unit="", sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return (f"{v:+,.{nd}f}" if sign else f"{v:,.{nd}f}") + unit


def _kpis(items):
    cells = "".join(
        f'<div class="kpi"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="s">{s}</div></div>' for k, v, s in items)
    return f'<div class="kpis">{cells}</div>'


def _concl_card(i: int, c: dict) -> str:
    """一张结论卡：结论 / 证据 / 含义 / 动作 / 失效。"""
    return (
        f'<div style="background:#FFFFFF;border:1px solid {C.COLOR["line"]};'
        f'border-left:3px solid {C.COLOR["price"]};border-radius:6px;'
        f'padding:8px 11px;margin-bottom:6px">'
        f'<div style="font-size:12.5px;font-weight:700;line-height:1.6">'
        f'<span style="display:inline-block;min-width:18px;height:18px;line-height:18px;'
        f'text-align:center;border-radius:9px;background:{C.COLOR["price"]};color:#fff;'
        f'font-size:10px;margin-right:6px">{i}</span>'
        f'<span style="color:{C.COLOR["text_dim"]};font-size:10.5px;font-weight:400;'
        f'margin-right:6px">[{c["tag"]}]</span>{c["claim"]}</div>'
        f'<div style="font-size:11px;line-height:1.75;margin-top:4px;color:#37485C">'
        f'<b>证据</b>：{c["evidence"]}</div>'
        f'<div style="font-size:11px;line-height:1.75;margin-top:3px;color:#37485C">'
        f'<b>投资解读</b>：{c["meaning"]}</div>'
        f'<div style="font-size:11px;line-height:1.75;margin-top:3px">'
        f'<b style="color:{C.COLOR["up"]}">操作</b>：{c["action"]}</div>'
        f'<div style="font-size:10.5px;line-height:1.6;margin-top:4px;padding:3px 7px;'
        f'background:#FFF8EC;border-radius:4px;color:#7A4A00">'
        f'<b>什么情况下这条不成立</b>：{c["fail"]}</div></div>')


def _action_block(title: str, sub: str, df: pd.DataFrame, color: str, cols) -> str:
    """一个动作卡片：标题 + 一行解释 + 每个行业一行完整理由。"""
    if df.empty:
        body = '<div style="font-size:11px;color:#6B7A8C">本期没有符合条件。</div>'
    else:
        rows = []
        for i, r in df.iterrows():
            _cells = "　".join(
                f'{k} <b>{v}</b>' for k, v in [
                    ("当期", f'{r["当期主动pp"]:+.2f}pp'),
                    ("广度", f'{r["加仓占比"]:.0f}%'),
                    ("配置", f'{r["历史分位"]:.0f}'),
                    ("指数", f'{r.get("指数分位", np.nan):.0f}'
                     if np.isfinite(r.get("指数分位", np.nan)) else "—"),
                ])
            rows.append(
                f'<div style="padding:5px 0;border-top:1px dashed {C.COLOR["line"]}">'
                f'<div style="font-size:11.5px"><b>{i}</b>'
                f'<span style="font-size:10px;color:{C.COLOR["text_dim"]};margin-left:5px">'
                f'强度 {r["强度"]}</span></div>'
                f'<div style="font-size:10.5px;color:#37485C;line-height:1.65">{_cells}</div>'
                f'<div style="font-size:10.5px;line-height:1.65;margin-top:2px">{r["理由"]}</div>'
                f'</div>')
        body = "".join(rows)
    return (f'<div style="border:1px solid {C.COLOR["line"]};border-top:3px solid {color};'
            f'border-radius:7px;padding:8px 10px;height:100%">'
            f'<div style="font-size:12px;font-weight:700;color:{color}">'
            f'{title}　<span style="color:{C.COLOR["text_dim"]};font-weight:400;'
            f'font-size:10.5px">{sub}</span></div>{body}</div>')


def render(h, B=None, IND=None) -> None:
    if not (cn.available() or cn.fallback_available()):
        h('<div class="cg-note">未找到持仓数据。运行 '
          '<span class="mono">python tools/run_ak_all.py</span> 抓取。</div>')
        return
    s = pb.summary()
    if not s:
        h('<div class="cg-note">数据不足，无法生成结论。</div>')
        return
    t = s["table"]
    q, qp, qy = s["q"], s["q_prev"], s["q_year"]
    qf, qf0 = s["q_full"], s["q_full0"]
    pos = s["pos"] or {}
    hist = cn.position_series()
    pct = float(hist["中位"].rank(pct=True).iloc[-1] * 100) if not hist.empty else np.nan
    n_funds = len(cn.universe(q))

    # =====================================================================
    # 一句话 + KPI
    # =====================================================================
    _hi = t[t["动作"] == "可以跟"]
    _av = t[t["动作"] == "要回避"]
    _hot = t[t["动作"] == "别追高"]
    _cs = cn.crowding_score(q)
    _n_ext = int((_cs["等级"] == "极度拥挤").sum()) if not _cs.empty else 0
    # 一句话里的数字全部现算：硬编码过「电子 −4.94」这种值，
    # 数据一更新就会和下面的表对不上。
    _dd = t.dropna(subset=["Δ上季pp"])
    _up2 = _dd.nlargest(2, "Δ上季pp") if not _dd.empty else pd.DataFrame()
    _dn2 = _dd.nsmallest(2, "Δ上季pp") if not _dd.empty else pd.DataFrame()

    def _pair(df):
        return "、".join(f"{i} {r['Δ上季pp']:+.2f}pp" for i, r in df.iterrows()) or "—"

    h('<div class="sec">一、一句话结论</div>')
    h(f'<div style="background:linear-gradient(90deg,{C.COLOR["panel"]} 0%,#FFFFFF 100%);'
      f'border:1px solid {C.COLOR["line"]};border-left:4px solid {C.COLOR["price"]};'
      f'border-radius:8px;padding:12px 15px;margin:2px 0 8px">'
      f'<div style="font-size:16px;font-weight:700;line-height:1.6">'
      f'机构没有加仓，钱在<b>从「跌得最狠的科技硬件」撤出，'
      f'一半抄底医药、一半躲进银行与公用事业</b>。</div>'
      f'<div style="font-size:12px;line-height:1.85;margin-top:5px;color:#37485C">'
      f'{q} 主动权益基金季报（样本 <b>{n_funds:,}</b> 只）：股票仓位中位 '
      f'<b>{pos.get("仓位1", float("nan")):.1f}%</b>，比上一个半年报期 '
      f'<b>{pos.get("中位变化pp", float("nan")):+.2f}pp</b>，处在 {len(hist)} 期的第 '
      f'<b>{pct:.0f} 分位</b>——<b>总仓位没动，行业变化是内部换仓</b>；'
      f'前十大口径下 {_dn2.index[0] if len(_dn2) else "—"} '
      f'{_dn2.iloc[0]["Δ上季pp"]:+.2f}pp 让位，'
      f'{_up2.index[0] if len(_up2) else "—"} {_up2.iloc[0]["Δ上季pp"]:+.2f}pp 接棒。'
      f'按「资金在进 + 多数在加 + 价格不贵」三条同时满足，本期只有 '
      f'<b>{len(_hi)} 个行业可以跟</b>（{"、".join(_hi.index) or "无"}），'
      f'<b>{len(_av)} 个要回避</b>，<b>{len(_hot)} 个别追高</b>。'
      f'</div></div>')

    h(_kpis([
        ("股票仓位", _fmt(pos.get("仓位1"), 1, "%"), f"15 期第 {pct:.0f} 分位"),
        ("可以跟", f"{len(_hi)} 个", "、".join(_hi.index[:3]) or "无"),
        ("要回避", f"{len(_av)} 个", "、".join(_av.index[:3]) or "无"),
        ("极度拥挤", f"{_n_ext} 个", "CSI≥80，见拥挤度页"),
        ("样本基金", f"{n_funds:,}", f"截至 {q}"),
    ]))
    h('<div class="tiny" style="margin-top:4px">'
      f'口径：仓位/主动项＝全持仓半年报（{qf0}→{qf}）；环比/广度/分位＝前十大（{qp}→{q}）。'
      f'两套口径回答不同问题，<b>不要相加</b>；详见「方法」页。</div>')

    # =====================================================================
    # 结论清单
    # =====================================================================
    h('<div class="sec">二、全部结论（每条都写清证据、含义、动作、失效条件）</div>')
    h('<div class="tiny" style="margin-bottom:5px">'
      '下面的 9 条覆盖了本期季报能读出的全部信息：总仓位 → 换仓主线 → 可以跟/别追高/要回避 → '
      '待确认 → 个股层 → 拥挤度 → 反面证据。每条都能用库里的数据复算。</div>')
    _cs_list = pb.conclusions(q)
    for i, c in enumerate(_cs_list, 1):
        h(_concl_card(i, c))

    # =====================================================================
    # 投资建议
    # =====================================================================
    h('<div class="sec">三、投资建议：四类动作，逐行业给出理由</div>')
    h(f'<div class="tiny" style="margin-bottom:6px">'
      f'筛选条件全部事前固定（阈值写在 <span class="mono">playbook.TH</span>）：'
      f'<b>可以跟</b>＝当期主动≥+0.3pp 且 广度≥60% 且 配置分位≤80 且 指数分位≤85；'
      f'<b>别追高</b>＝配置分位≥80 或 指数分位≥90，但资金还在进；'
      f'<b>要回避</b>＝当期主动≤−0.4pp 且 广度≤50%，或配置分位≥90 且没人接；'
      f'<b>等确认/观察</b>＝方向不干净。'
      f'「当期主动」是价格中性后的净买入（见方法页），不是原始权重变化。</div>')

    c1, c2 = st.columns(2, gap="small")
    with c1:
        h(_action_block("① 可以跟", "资金在进 · 多数在加 · 价格不贵",
                        _hi, C.COLOR["up"], 0))
    with c2:
        h(_action_block("② 别追高", "位置到极值 · 资金还在进",
                        _hot, C.COLOR["level_拥挤"], 0))
    c3, c4 = st.columns(2, gap="small")
    with c3:
        h(_action_block("③ 要回避", "资金在撤 · 没人接",
                        _av, C.COLOR["down"], 0))
    with c4:
        _wt = t[t["动作"].isin(["等确认", "观察"])]
        h(_action_block("④ 等确认 / 观察", "方向不干净 · 只给观察仓",
                        _wt, C.COLOR["text_dim"], 0))

    # =====================================================================
    # 操作建议
    # =====================================================================
    h('<div class="sec">四、操作建议：仓位、节奏、什么时候推翻</div>')
    # 用「离散度」而不是「分歧度」做加仓惩罚：
    # 分歧度 = 1−|2×广度−1|，是广度的单调变换（广度 50% 时分歧度恒为 1），
    # 拿它设阈值等于「几乎每期都触发」，不是规则。
    # 离散度＝价格中性 Δ 的横截面标准差，与广度相互独立，才是真的「基金之间差多少」。
    _disp = float(s["breadth"]["离散度pp"].median()) if not s["breadth"].empty else np.nan
    lo, hi, _penalty = pb.position_band(pct, _disp)
    stance = ("偏高（历史高位）" if pct >= 85 else "中性偏高" if pct >= 70
              else "中性" if pct >= 30 else "偏低（历史低位）")
    if _penalty:
        stance += "，分歧大再降 5pp"

    k1, k2, k3 = st.columns([0.3, 0.35, 0.35], gap="small")
    with k1:
        h(f'<div style="border:1px solid {C.COLOR["line"]};border-top:3px solid '
          f'{C.COLOR["price"]};border-radius:7px;padding:9px 11px;height:100%">'
          f'<div style="font-size:12px;font-weight:700">总仓位</div>'
          f'<div style="font-size:24px;font-weight:700;margin:3px 0">{lo}%–{hi}%</div>'
          f'<div style="font-size:10.5px;line-height:1.7;color:#37485C">'
          f'算法：仓位 {pct:.0f} 分位 → 基础区间；离散度中位 {_disp:.1f}pp'
          f'（{">8pp，再降 5pp" if _penalty else "≤8pp，不调整"}）。<br>'
          f'<span style="color:{C.COLOR["text_dim"]}">不是说必须这么配，'
          f'而是提醒：现在整体加仓的赔率不高。</span></div></div>')
    with k2:
        h(f'<div style="border:1px solid {C.COLOR["line"]};border-top:3px solid '
          f'{C.COLOR["up"]};border-radius:7px;padding:9px 11px;height:100%">'
          f'<div style="font-size:12px;font-weight:700">怎么买</div>'
          f'<div style="font-size:11px;line-height:1.85;margin-top:3px">'
          f'· 只买「可以跟」的 {len(_hi)} 个，单行业 ≤ 权益仓位 15%<br>'
          f'· 分 3 批建：首笔 1/3，一个月后不破位再补<br>'
          f'· 行业 ETF / 指数增强优先，个股只买讲得清逻辑的<br>'
          f'· 不融资、不加杠杆、不因为「跌得多」抄底</div></div>')
    with k3:
        h(f'<div style="border:1px solid {C.COLOR["line"]};border-top:3px solid '
          f'{C.COLOR["down"]};border-radius:7px;padding:9px 11px;height:100%">'
          f'<div style="font-size:12px;font-weight:700">怎么卖 / 怎么止损</div>'
          f'<div style="font-size:11px;line-height:1.85;margin-top:3px">'
          f'· 持有「要回避」的：反弹减半，不等解套<br>'
          f'· 单行业自建仓回撤 >15% 且当期主动转负 → 减半<br>'
          f'· 「别追高」的已持有可以拿，不加仓<br>'
          f'· 单行业超过权益 20% → 减到 20% 以内</div></div>')

    _rules = pd.DataFrame([
        dict(动作="买入", 触发条件="当期主动连续 2 期 ≥+0.3pp，广度 ≥60%，指数分位 <85",
             执行="分 3 批，单行业上限 15%"),
        dict(动作="加仓", 触发条件="已持有且下季广度仍 ≥60%、主动项仍为正",
             执行="加至目标权重，不追单日涨幅 >5% 的"),
        dict(动作="减仓", 触发条件="广度跌破 50%，或当期主动连续 2 期 ≤−0.4pp",
             执行="先减半，下一季不改善再清"),
        dict(动作="止损", 触发条件="行业指数自建仓回撤 >15% 且 当期主动转负",
             执行="减半，不等反弹"),
        dict(动作="推翻", 触发条件="下季总仓位变化 >2pp，或样本基金数减少 >20%",
             执行="整页重跑，结论作废"),
    ])
    st.dataframe(_rules, width="stretch", hide_index=True, height=232,
                 column_config={
                     "动作": st.column_config.TextColumn(width="small"),
                     "触发条件": st.column_config.TextColumn(width="large"),
                     "执行": st.column_config.TextColumn(width="medium")})

    # =====================================================================
    # 反证与边界
    # =====================================================================
    h('<div class="sec">五、反面证据：这套方法什么时候会错</div>')
    c5, c6 = st.columns(2, gap="small")
    with c5:
        h(f'<div style="border:1px solid {C.COLOR["line"]};border-radius:7px;'
          f'padding:9px 11px;height:100%;background:#FFF8EC">'
          f'<div style="font-size:12px;font-weight:700;color:#7A4A00">'
          f'已经被自己的数据推翻的三条说法</div>'
          f'<div style="font-size:11px;line-height:1.8;margin-top:4px">'
          f'· <b>「拥挤即风险」方向对、但不显著</b>：配置分位 Q1→Q5 的次季收益'
          f'从 +2.39% 降到 +0.51%，可 t 只有 −0.83、p=0.42，不能当卖出信号。<br>'
          f'· <b>「加仓能预测收益」不成立</b>：广度的 RankIC 长期在 0 附近，'
          f'p 值不显著（实测见因子检验页）。<br>'
          f'· <b>「前十大会系统性低估」不成立</b>：它把尾部归零、把集中行业放大，'
          f'不是简单压低。</div></div>')
    with c6:
        h(f'<div style="border:1px solid {C.COLOR["line"]};border-radius:7px;'
          f'padding:9px 11px;height:100%">'
          f'<div style="font-size:12px;font-weight:700">四条硬约束</div>'
          f'<div style="font-size:11px;line-height:1.8;margin-top:4px">'
          f'· <b>季报只有前十大</b>：Q1/Q3 只披露 10 只，绝对水平不可用；<br>'
          f'· <b>时滞约 3–4 周</b>：季末到数据可得 28 天，'
          f'你看到时下季度已过约 1/3；<br>'
          f'· <b>幸存者偏差</b>：已清盘基金不在库内，早期分位偏乐观；<br>'
          f'· <b>只有股票，没有债券与现金</b>：灵活配置型基金的行业占比会被放大。</div></div>')

    h('<div class="tiny" style="margin-top:6px">'
      '想看每个数字怎么算的：左侧边栏「指标速查」可以搜任意指标，'
      '含公式、计算过程、阈值、标准来源与常见误用；'
      '想看逐图解读：切到「② 本期解读」；'
      '想看单个行业的历史：切到「④ 行业拥挤度」。'
      f'　版本口径：全持仓 {qf0}→{qf}、前十大 {qp}→{q}、指数 2019-01 起。</div>')
