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

字号与版式
----------
这一页是「读」的页面，不是「查」的页面，所以正文用 13.5–14px、
标题 15px、结论主句 20px。之前 10.5–12.5px 的密度适合投屏演示，
不适合在笔记本上逐条看——读者的反馈是「字太小、看着累」。

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

FS_BODY = 13.5
FS_SMALL = 12.5


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
    """一张结论卡：结论 / 证据 / 含义 / 动作 / 失效。

    版式：左侧序号 + 右侧四段。每段前面用一个小色块标签，
    比「证据：」「投资解读：」这种纯文字前缀更容易扫读。
    """
    def row(label, text, color, bg="#FFFFFF", weight="400"):
        return (
            f'<div style="display:flex;gap:9px;padding:6px 0 0">'
            f'<div style="flex:0 0 62px;font-size:{FS_SMALL}px;font-weight:700;'
            f'color:{color};line-height:1.85">{label}</div>'
            f'<div style="flex:1;font-size:{FS_BODY}px;line-height:1.95;'
            f'color:#26374A;font-weight:{weight}">{text}</div></div>')

    return (
        f'<div style="background:#FFFFFF;border:1px solid {C.COLOR["line"]};'
        f'border-left:4px solid {C.COLOR["price"]};border-radius:9px;'
        f'padding:11px 15px 13px;margin-bottom:9px">'
        f'<div style="display:flex;gap:9px;align-items:flex-start">'
        f'<div style="flex:0 0 25px;height:25px;border-radius:13px;'
        f'background:{C.COLOR["price"]};color:#fff;font-size:13px;font-weight:700;'
        f'display:flex;align-items:center;justify-content:center">{i}</div>'
        f'<div style="flex:1">'
        f'<div style="font-size:15px;font-weight:700;line-height:1.7;color:#1B2A3A">'
        f'{c["claim"]}'
        f'<span style="font-weight:400;font-size:{FS_SMALL}px;'
        f'color:{C.COLOR["text_dim"]};margin-left:7px">［{c["tag"]}］</span></div>'
        f'{row("证据", c["evidence"], C.COLOR["price"])}'
        f'{row("经济含义", c["meaning"], C.COLOR["accent"])}'
        f'{row("操作", c["action"], C.COLOR["up"])}'
        f'<div style="margin-top:7px;padding:5px 9px;background:#FFF8EC;'
        f'border-radius:5px;font-size:{FS_SMALL}px;line-height:1.8;color:#7A4A00">'
        f'<b>什么时候这条不成立</b>：{c["fail"]}</div>'
        f'</div></div></div>')


def _action_block(title: str, sub: str, df: pd.DataFrame, color: str, cols=0) -> str:
    """一个动作卡片：标题 + 一行解释 + 每个行业一行完整理由。"""
    if df.empty:
        body = (f'<div style="font-size:{FS_SMALL}px;color:#6B7A8C;'
                f'padding:6px 0">本期没有符合条件。</div>')
    else:
        rows = []
        for i, r in df.iterrows():
            _cells = "　".join(
                f'{k} <b>{v}</b>' for k, v in [
                    ("当期", f'{r["当期主动pp"]:+.2f}pp'),
                    ("广度", f'{r["加仓占比"]:.0f}%'),
                    ("配置", f'{r["历史分位"]:.0f}'),
                    ("价格", f'{r.get("指数分位", np.nan):.0f}'
                     if np.isfinite(r.get("指数分位", np.nan)) else "—"),
                ])
            rows.append(
                f'<div style="padding:7px 0;border-top:1px dashed {C.COLOR["line"]}">'
                f'<div style="font-size:14px;font-weight:700">{i}'
                f'<span style="font-size:{FS_SMALL}px;color:{C.COLOR["text_dim"]};'
                f'font-weight:400;margin-left:6px">{r["强度"]}</span></div>'
                f'<div style="font-size:{FS_SMALL}px;color:#37485C;line-height:1.8;'
                f'margin-top:2px">{_cells}</div>'
                f'<div style="font-size:{FS_BODY}px;line-height:1.85;margin-top:3px;'
                f'color:#26374A">{r["理由"]}</div>'
                f'</div>')
        body = "".join(rows)
    return (f'<div style="border:1px solid {C.COLOR["line"]};border-top:4px solid {color};'
            f'border-radius:9px;padding:10px 13px;height:100%">'
            f'<div style="font-size:15px;font-weight:700;color:{color}">'
            f'{title}　<span style="color:{C.COLOR["text_dim"]};font-weight:400;'
            f'font-size:{FS_SMALL}px">{sub}</span></div>{body}</div>')


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
    # 用「当期主动 pp」（价格中性后的净买卖）而不是环比：标题说的是「钱」，
    # 环比含价格漂移，会把「跌出来的占比下降」也说成资金撤出。
    _flow = ("当期主动pp" if "当期主动pp" in t.columns
             and t["当期主动pp"].notna().any() else "Δ上季pp")
    _dd = t.dropna(subset=[_flow])
    _up2 = _dd.nlargest(2, _flow) if not _dd.empty else pd.DataFrame()
    _dn2 = _dd.nsmallest(2, _flow) if not _dd.empty else pd.DataFrame()

    # 标题也必须现算：硬编码过「科技硬件→医药/银行」，数据一换就是错的。
    _dpos = float(pos.get("中位变化pp", np.nan)) if pos else np.nan
    _pos_txt = ("机构没有加仓" if not np.isfinite(_dpos) or abs(_dpos) < 0.5
                else f"机构在{'加' if _dpos > 0 else '减'}仓（{_dpos:+.1f}pp）")
    if len(_dn2) >= 1 and len(_up2) >= 1:
        # 「钱流进去」和「现在能不能买」是两件事：如果流入的行业正好是「别追高」，
        # 必须在同一句里说清楚，否则读者会以为系统自相矛盾。
        _hot_in = [i for i in _up2.index if i in set(_hot.index)]
        _hot_note = (f'（其中{"、".join(_hot_in)}已在高位，只算资金流、不是买点）'
                     if _hot_in else '')
        _head = (f'{_pos_txt}：钱在从 <b>{"、".join(_dn2.index)}</b> 撤出，'
                 f'加向 <b>{"、".join(_up2.index)}</b>{_hot_note}'
                 + (f'；可以跟 {len(_hi)} 个、要回避 {len(_av)} 个。' if len(_hi) or len(_av) else '。'))
    else:
        _head = f'{_pos_txt}：本期行业间没有形成一致的方向。'

    h('<div class="sec">一、一句话结论</div>')
    h(f'<div style="background:linear-gradient(90deg,{C.COLOR["panel"]} 0%,#FFFFFF 100%);'
      f'border:1px solid {C.COLOR["line"]};border-left:5px solid {C.COLOR["price"]};'
      f'border-radius:10px;padding:15px 20px;margin:2px 0 10px">'
      f'<div style="font-size:20px;font-weight:700;line-height:1.6;color:#16283C">'
      f'{_head}</div>'
      f'<div style="font-size:14px;line-height:2.0;margin-top:8px;color:#2C3E50">'
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
        ("股票仓位", _fmt(pos.get("仓位1"), 1, "%"), f"{len(hist)} 期第 {pct:.0f} 分位"),
        ("可以跟", f"{len(_hi)} 个", "、".join(_hi.index[:3]) or "无"),
        ("要回避", f"{len(_av)} 个", "、".join(_av.index[:3]) or "无"),
        ("极度拥挤", f"{_n_ext} 个", "CSI≥80 且样本充足"),
        ("样本基金", f"{n_funds:,}", f"截至 {q}"),
    ]))
    h('<div class="tiny" style="margin:6px 0 2px">'
      f'口径：仓位/主动项＝全持仓半年报（{qf0}→{qf}）；环比/广度/分位＝前十大'
      f'（{qp}→{q}），同比＝对上年同季（{qy}）。'
      f'两套口径回答不同问题，<b>不要相加</b>；详见「⑥ 方法与口径」。</div>')

    # =====================================================================
    # 结论清单
    # =====================================================================
    h('<div class="sec">二、全部结论：每条都写清证据、经济含义、操作与失效条件</div>')
    h(f'<div class="cg-note" style="font-size:12.5px">'
      f'下面这 9 条覆盖本期季报的九个方面：总仓位 → 换仓主线 → '
      f'可以跟/别追高/要回避 → 待确认 → 个股层 → 拥挤度 → 反面证据。'
      f'每条都能用库里的数据复算。</div>')
    _cs_list = pb.conclusions(q)
    for i, c in enumerate(_cs_list, 1):
        h(_concl_card(i, c))

    # =====================================================================
    # 投资建议
    # =====================================================================
    h('<div class="sec">三、投资建议：四类动作，逐行业给出理由</div>')
    h(f'<div class="cg-note" style="font-size:12.5px">'
      f'筛选条件全部事前固定（阈值写在 <span class="mono">playbook.TH</span>）：'
      f'<b>可以跟</b>＝当期主动≥+0.3pp 且 广度≥60% 且 配置分位≤80 且 价格分位≤85；'
      f'<b>别追高</b>＝配置分位≥80 或 价格分位≥90，但资金还在进；'
      f'<b>要回避</b>＝当期主动≤−0.4pp 且 广度≤50%，或配置分位≥90 且没人接；'
      f'<b>等确认/观察</b>＝方向不干净。'
      f'「当期主动」是价格中性后的净买入（见方法页），不是原始权重变化。'
      f'31 个行业全部有交代：四类动作之外，其余归为「中性」。</div>')

    c1, c2 = st.columns(2, gap="small")
    with c1:
        h(_action_block("① 可以跟", "资金在进 · 多数在加 · 价格不贵",
                        _hi, C.COLOR["up"]))
    with c2:
        h(_action_block("② 别追高", "位置到极值 · 资金还在进",
                        _hot, C.COLOR["level_拥挤"]))
    c3, c4 = st.columns(2, gap="small")
    with c3:
        h(_action_block("③ 要回避", "资金在撤 · 没人接",
                        _av, C.COLOR["down"]))
    with c4:
        _wt = t[t["动作"].isin(["等确认", "观察"])]
        h(_action_block("④ 等确认 / 观察", "方向不干净 · 只给观察仓",
                        _wt, C.COLOR["text_dim"]))

    # 31 个行业必须全部有交代：四类动作之外还有「中性」，
    # 不写出来读者会以为「没列到的 13 个是漏了」。
    _neu = t[t["动作"] == "中性"]
    if len(_neu):
        _small_n = _neu[_neu.get("可靠", pd.Series(index=_neu.index)).eq("小样本")]
        _txt_n = (f'其余 <b>{len(_neu)}</b> 个行业未触发任何动作（动作表里统一记为「中性」）：'
                  f'{"、".join(_neu.index)}。中性＝不满足上面四类动作的任一条件，'
                  f'维持不动、不新增也不减仓。')
        if len(_small_n):
            _txt_n += (f'其中 {len(_small_n)} 个是<b>小样本、不评级</b>'
                       f'（{"、".join(_small_n.index)}）：持有基金 &lt;150 只或'
                       f'前十配置 &lt;0.5%，分位与 CSI 噪音太大；'
                       f'它们只是被归入「不动作」，不代表真的中性。')
        h(f'<div class="tiny" style="margin:4px 0 10px">{_txt_n}</div>')

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

    k1, k2, k3 = st.columns([0.30, 0.35, 0.35], gap="small")
    with k1:
        h(f'<div style="border:1px solid {C.COLOR["line"]};border-top:4px solid '
          f'{C.COLOR["price"]};border-radius:9px;padding:11px 14px;height:100%">'
          f'<div style="font-size:14px;font-weight:700">权益总仓位</div>'
          f'<div style="font-size:30px;font-weight:700;margin:4px 0;color:#16283C">'
          f'{lo}%–{hi}%</div>'
          f'<div style="font-size:{FS_BODY}px;line-height:1.9;color:#2C3E50">'
          f'仓位 {pct:.0f} 分位 → 基础区间；离散度中位 {_disp:.1f}pp'
          f'（{">8pp，再降 5pp" if _penalty else "≤8pp，不调整"}）。<br>'
          f'<span style="color:{C.COLOR["text_dim"]}">不是说必须这么配，'
          f'而是提醒：现在整体加仓的赔率不高。</span></div></div>')
    with k2:
        h(f'<div style="border:1px solid {C.COLOR["line"]};border-top:4px solid '
          f'{C.COLOR["up"]};border-radius:9px;padding:11px 14px;height:100%">'
          f'<div style="font-size:14px;font-weight:700">怎么买</div>'
          f'<div style="font-size:{FS_BODY}px;line-height:1.95;margin-top:4px">'
          f'· 只买「可以跟」的 {len(_hi)} 个，单行业 ≤ 权益仓位 15%<br>'
          f'· 分 3 批建：首笔 1/3，一个月后不破位再补<br>'
          f'· 行业 ETF / 指数增强优先，个股只买讲得清逻辑的<br>'
          f'· 不融资、不加杠杆、不因为「跌得多」抄底</div></div>')
    with k3:
        h(f'<div style="border:1px solid {C.COLOR["line"]};border-top:4px solid '
          f'{C.COLOR["down"]};border-radius:9px;padding:11px 14px;height:100%">'
          f'<div style="font-size:14px;font-weight:700">怎么卖 / 怎么止损</div>'
          f'<div style="font-size:{FS_BODY}px;line-height:1.95;margin-top:4px">'
          f'· 持有「要回避」的：反弹减半，不等解套<br>'
          f'· 自建仓回撤 >15% 且当期主动转负 → 减半<br>'
          f'· 「别追高」的已持有可以拿，不加仓<br>'
          f'· 单行业超过权益 20% → 减到 20% 以内</div></div>')

    _rules = pd.DataFrame([
        dict(动作="买入", 触发条件="当期主动连续 2 期 ≥+0.3pp，广度 ≥60%，价格分位 <85",
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
        h(f'<div style="border:1px solid {C.COLOR["line"]};border-radius:9px;'
          f'padding:11px 14px;height:100%;background:#FFF8EC">'
          f'<div style="font-size:14px;font-weight:700;color:#7A4A00">'
          f'已经被自己的数据推翻的三条说法</div>'
          f'<div style="font-size:{FS_BODY}px;line-height:1.95;margin-top:5px">'
          f'· <b>「拥挤即风险」方向对、但不显著</b>：{pb.crowding_evidence()}<br>'
          f'· <b>「加仓能预测收益」不成立</b>：广度的 RankIC 长期在 0 附近，'
          f'p 值不显著（实测见「⑤ 因子检验」）。<br>'
          f'· <b>「前十大会系统性低估」不成立</b>：它把尾部归零、把集中行业放大，'
          f'不是简单压低。</div></div>')
    with c6:
        h(f'<div style="border:1px solid {C.COLOR["line"]};border-radius:9px;'
          f'padding:11px 14px;height:100%">'
          f'<div style="font-size:14px;font-weight:700">四条硬约束</div>'
          f'<div style="font-size:{FS_BODY}px;line-height:1.95;margin-top:5px">'
          f'· <b>季报只有前十大</b>：Q1/Q3 只披露 10 只，绝对水平不可用；<br>'
          f'· <b>时滞约 3–4 周</b>：季末到数据可得 28 天，'
          f'你看到时下季度已过约 1/3；<br>'
          f'· <b>幸存者偏差</b>：已清盘基金不在库内，早期分位偏乐观；<br>'
          f'· <b>只有股票，没有债券与现金</b>：灵活配置型基金的行业占比会被放大。</div></div>')

    h('<div class="tiny" style="margin-top:8px">'
      '想看每个数字怎么算的：左侧边栏「指标速查」可以搜任意指标，'
      '含公式、计算过程、阈值、标准来源与常见误用；'
      '想看逐图证据：切到「② 本期解读（全市场）」；'
      '想看单个行业的历史：切到「③ 行业拥挤度」。'
      f'　版本口径：全持仓 {qf0}→{qf}、前十大 {qp}→{q}、指数 2015-01 起。</div>')
