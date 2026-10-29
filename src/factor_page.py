# -*- coding: utf-8 -*-
"""⑤ 因子检验：季报因子的标准流程——定义、计算、IC、显著性、分层、结论。

页面原则
--------
1. **先给结论**：9 个因子里有几个能用、几个不能用、为什么；
2. **每一步都可复核**：公式、期数、样本、阈值全部写出；
3. **不藏坏结果**：被证伪的因子照样展示，并写清「所以前面的判断要怎么改」。

数据来自 data/derived/factor_test.json（由 tools/run_factor_test.py 生成）：
现算要 60–80 秒，页面读 JSON 不到 10ms。页面上有「重算」按钮可临时现算。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C
import report_charts as rc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, "data", "derived", "factor_test.json")

_VERDICT_COLOR = {"强": C.COLOR["up"], "可用": C.COLOR["up"], "仅辅助": "#E8710A",
                  "反向": C.COLOR["accent"], "无效": C.COLOR["text_dim"],
                  "样本不足": "#C3CBD4"}


def _load() -> dict:
    if not os.path.exists(JSON_PATH):
        return {}
    try:
        with open(JSON_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _recompute() -> dict:
    """现算一次并写回 JSON（页面按钮触发，约 60–80 秒）。"""
    import canon as cn
    import factor_test as ft
    tab = ft.run_all()
    out = _load() or {}
    out["generated"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    out["factors"] = []
    for k, meta in ft.FACTORS.items():
        r = ft.test_factor(k)
        lab, why = ft.verdict(r)
        d = dict(key=k, name=meta["name"], direction=meta["direction"],
                 freq=meta["freq"], formula=meta["formula"], expect=meta["expect"],
                 verdict=lab, why=why, ic_mean=r.get("ic_mean"), ic_ir=r.get("ic_ir"),
                 t=r.get("t"), p=r.get("p"), win=r.get("win"), n=r.get("n"),
                 ls_mean=r.get("ls_mean"), ls_win=r.get("ls_win"),
                 ann=r.get("ann"), mono=r.get("mono"))
        if r.get("ok"):
            d["ic_series"] = {str(i): round(float(v), 4) for i, v in r["ic"].items()}
            d["layers"] = {str(i): round(float(v), 3) for i, v in r["layers"].items()}
        out["factors"].append(d)
    try:
        with open(JSON_PATH, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
    except Exception:
        pass
    return out


def _kpis(items):
    cells = "".join(
        f'<div class="kpi"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="s">{s}</div></div>' for k, v, s in items)
    return f'<div class="kpis">{cells}</div>'


def _ic_fig(d: dict) -> go.Figure:
    """IC 柱 + 累计 IC 线：看方向是否稳定，还是靠一两期撑起来。"""
    s = pd.Series({k: v for k, v in d.get("ic_series", {}).items() if v is not None})
    if s.empty:
        return rc._base(go.Figure(), "无 IC 序列")
    s.index = list(s.index)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(s.index), y=s.values, name="单期 IC",
                         marker_color=[C.COLOR["up"] if v > 0 else C.COLOR["down"]
                                       for v in s.values], opacity=0.75,
                         hovertemplate="%{x}　IC %{y:+.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=list(s.index), y=s.cumsum().values, name="累计 IC",
                             mode="lines+markers", line=dict(color=C.COLOR["price"], width=2),
                             yaxis="y2",
                             hovertemplate="%{x}　累计 %{y:+.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C.COLOR["line"], width=1))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                  title="累计 IC", tickfont=dict(size=9)),
                      yaxis=dict(title="单期 IC", gridcolor=C.COLOR["grid"]),
                      xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                      legend=dict(orientation="h", y=1.12, font=dict(size=9)))
    fig = rc._base(fig, "", 330, legend=True)
    fig.update_layout(margin=dict(l=48, r=48, t=34, b=30))
    return fig


def _layer_fig(d: dict) -> go.Figure:
    """五分层平均次期收益：看单调性。"""
    L = d.get("layers") or {}
    if not L:
        return rc._base(go.Figure(), "无分层数据")
    xs = [f"Q{int(k)+1}" for k in sorted(L, key=lambda x: int(x))]
    ys = [L[k] for k in sorted(L, key=lambda x: int(x))]
    fig = go.Figure(go.Bar(x=xs, y=ys,
                           marker_color=[C.COLOR["down"] if v < 0 else C.COLOR["up"]
                                         for v in ys],
                           text=[f"{v:+.2f}%" for v in ys], textposition="outside",
                           textfont=dict(size=9.5),
                           hovertemplate="%{x}　次期平均 %{y:+.2f}%<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C.COLOR["line"], width=1))
    fig.update_yaxes(title_text="次期平均收益 %")
    fig.update_xaxes(title_text="按因子值五等分（Q1 最低 → Q5 最高）")
    fig = rc._base(fig, "", 300)
    fig.update_layout(margin=dict(l=48, r=16, t=22, b=36))
    return fig


def render(h, B=None, IND=None) -> None:
    d = _load()
    if not d:
        h('<div class="cg-note">尚未生成因子检验结果。运行 '
          '<span class="mono">python tools/run_factor_test.py</span> 生成 '
          '<span class="mono">data/derived/factor_test.json</span>。</div>')
        if st.button("现在现算（约 60–80 秒）", key="ft_calc0"):
            with st.spinner("正在检验 9 个因子…"):
                d = _recompute()
            st.rerun()
        return

    F = d.get("factors", [])
    if not F:
        h('<div class="cg-note">结果文件为空，请重跑 tools/run_factor_test.py。</div>')
        return

    _ok = [f for f in F if f["verdict"] in ("强", "可用")]
    _rev = [f for f in F if f["verdict"] == "反向"]
    _no = [f for f in F if f["verdict"] in ("无效", "样本不足")]
    _aux = [f for f in F if f["verdict"] == "仅辅助"]

    # =====================================================================
    # 本页结论
    # =====================================================================
    h('<div class="sec">本页结论：9 个季报因子，能用的有几个</div>')
    _best = max(F, key=lambda x: abs(x.get("ic_mean") or 0))
    h(f'<div style="background:{C.COLOR["panel"]};border-left:3px solid '
      f'{C.COLOR["threshold"]};border-radius:0 6px 6px 0;padding:10px 13px;'
      f'font-size:12.5px;line-height:1.95">'
      f'把 9 个因子按标准流程（扩张窗口分位、Spearman IC、Newey-West t、五分层）'
      f'在 {d["sample"]["q0"]}–{d["sample"]["q1"]} 的 {len(F)} 组面板上跑一遍，结论是：'
      f'<b>没有一个是「正向可用」的</b>——{len(_ok)} 个可用、{len(_aux)} 个仅辅助、'
      f'{len(_rev)} 个反向、{len(_no)} 个无效。<br>'
      f'唯一接近显著的是 <b>{_best["name"]}</b>（IC {_best["ic_mean"]:+.3f}、'
      f't {_best["t"]:+.2f}、p {_best["p"]:.3f}），但方向是<b>负</b>的：'
      f'加仓越一致的行业，次季反而越弱。<br>'
      f'<b>所以这套系统的定位必须说清楚</b>：它是「读懂季报、识别拥挤、分配仓位」的工具，'
      f'<b>不是收益预测模型</b>。前面页面上的「可以跟」＝机构在做 + 赔率不差，'
      f'不等于「会涨」。</div>')

    h(_kpis([
        ("检验因子", f"{len(F)} 个", "9 个季报指标"),
        ("正向可用", f"{len(_ok)} 个", "IC≥0.05 且 t≥1.96"),
        ("反向显著", f"{len(_rev)} 个", "方向与定义相反"),
        ("无效", f"{len(_no)} 个", "t<1.64"),
        ("最短样本", f'{min(f["n"] for f in F)} 期', "半年频因子受限"),
    ]))
    h(f'<div class="tiny">结果生成于 {d.get("generated", "—")}，'
      f'样本 {d["sample"]["quarters"]} 期 × {d["sample"]["industries"]} 行业，'
      f'最新期基金数 {d["sample"].get("n_funds", "—")}。'
      f'阈值：{"、".join(f"{k}={v}" for k, v in d.get("thresholds", {}).items())}。</div>')

    # =====================================================================
    # 标准流程
    # =====================================================================
    h('<div class="sec">一、标准流程：五步，每步都可复核</div>')
    _steps = [
        ("① 定义", "说清因子在每个（期, 行业）上取什么值、方向是「越大越看多」还是「越大越看空」。"
                   "方向必须事前声明，不能等看到结果再改。"),
        ("② 计算", "只用当期及之前的信息。分位数用<b>扩张窗口</b>（每期只排到当期），"
                   "不能用全样本排名——那会让 2019 年的分位用上 2026 年的数据，IC 被系统性高估。"),
        ("③ 单期 IC", "每期在 31 个行业上算 Spearman(因子值, 次期收益)，得到一条 IC 时序。"
                      "用秩相关是因为收益有极端值，秩相关对异常值稳健。"),
        ("④ 汇总检验", "IC 均值、IC_IR＝均值/标准差、Newey-West t（修正自相关）、"
                       "双侧 p 值、胜率。四个数一起看，只看均值会被一两期大赚骗过去。"),
        ("⑤ 分层", "按因子值五等分，看次期收益是否单调。IC 说「排序相关」，"
                   "分层说「关系形状」——只有单调的因子才能做打分加权。"),
    ]
    h('<div style="display:flex;gap:6px;flex-wrap:wrap">' + "".join(
        f'<div style="flex:1 1 180px;min-width:170px;background:#FFFFFF;'
        f'border:1px solid {C.COLOR["line"]};border-top:3px solid {C.COLOR["price"]};'
        f'border-radius:6px;padding:7px 9px">'
        f'<div style="font-size:11.5px;font-weight:700;margin-bottom:3px">{a}</div>'
        f'<div style="font-size:10.5px;line-height:1.7;color:#37485C">{b}</div></div>'
        for a, b in _steps) + '</div>')

    # =====================================================================
    # 因子定义表
    # =====================================================================
    h('<div class="sec">二、因子定义：衡量什么、怎么算、预期方向</div>')
    _rows = "".join(
        f'<tr><td style="padding:4px 6px;border-top:1px solid {C.COLOR["grid"]};'
        f'font-size:11px;font-weight:600;white-space:nowrap">{f["name"]}</td>'
        f'<td style="padding:4px 6px;border-top:1px solid {C.COLOR["grid"]};'
        f'font-size:10.5px">{f["formula"]}</td>'
        f'<td style="padding:4px 6px;border-top:1px solid {C.COLOR["grid"]};'
        f'font-size:10.5px;white-space:nowrap">'
        f'{"越大越看多" if f["direction"] > 0 else "越大越看空"}　{f["freq"]}</td>'
        f'<td style="padding:4px 6px;border-top:1px solid {C.COLOR["grid"]};'
        f'font-size:10.5px;color:{C.COLOR["text_dim"]}">{f["expect"]}</td></tr>'
        for f in F)
    h(f'<div style="max-height:330px;overflow:auto;border:1px solid {C.COLOR["line"]};'
      f'border-radius:6px"><table style="width:100%;border-collapse:collapse">'
      f'<thead style="position:sticky;top:0;background:{C.COLOR["panel"]};z-index:2">'
      f'<tr><th style="padding:5px 6px;text-align:left;font-size:11px">因子</th>'
      f'<th style="padding:5px 6px;text-align:left;font-size:11px">计算公式</th>'
      f'<th style="padding:5px 6px;text-align:left;font-size:11px">方向 / 频率</th>'
      f'<th style="padding:5px 6px;text-align:left;font-size:11px">为什么预期有效</th>'
      f'</tr></thead><tbody>{_rows}</tbody></table></div>')

    # =====================================================================
    # 汇总表
    # =====================================================================
    h('<div class="sec">三、检验结果汇总：9 个因子排一排</div>')
    _t = pd.DataFrame([{
        "因子": f["name"],
        "方向": "正" if f["direction"] > 0 else "负",
        "IC": f["ic_mean"], "IC_IR": f["ic_ir"], "t 值": f["t"], "p 值": f["p"],
        "胜率%": f["win"], "期数": f["n"],
        "多空%": f["ls_mean"], "单调性": f["mono"],
        "结论": f["verdict"],
    } for f in F]).sort_values("IC", key=lambda s: s.abs(), ascending=False)
    st.dataframe(
        _t, width="stretch", hide_index=True, height=340,
        column_config={
            "IC": st.column_config.NumberColumn(format="%+.3f"),
            "IC_IR": st.column_config.NumberColumn(format="%+.2f"),
            "t 值": st.column_config.NumberColumn(format="%+.2f"),
            "p 值": st.column_config.NumberColumn(format="%.3f"),
            "胜率%": st.column_config.NumberColumn(format="%.0f"),
            "多空%": st.column_config.NumberColumn(format="%+.2f",
                                                 help="Q5 − Q1 的次期平均收益差"),
            "单调性": st.column_config.NumberColumn(format="%+.2f",
                                                 help="五层收益与层序号的秩相关，±1 为完全单调"),
        })
    h(f'<div class="tiny">IC 为 29 期左右的均值；t 用 Newey-West（滞后 4 期）修正自相关；'
      f'p 为双侧。判定阈值事前固定：|IC|&lt;0.03 或 |t|&lt;1.64 记「无效」；'
      f'IC 为负且显著记「反向」；|IC|≥0.05 且 IC_IR≥0.5 且 |t|≥1.96 记「可用」。'
      f'「半年主动净额」只有 14 期样本，统计功效很弱，结论仅供参考。</div>')

    # =====================================================================
    # 逐因子
    # =====================================================================
    h('<div class="sec">四、逐因子看：IC 稳不稳、分层单不单调</div>')
    _opts = [f'{f["name"]}（{f["verdict"]}）' for f in F]
    _pick = st.selectbox("选择因子", _opts, key="ft_pick")
    f = F[_opts.index(_pick)]
    _c = _VERDICT_COLOR.get(f["verdict"], C.COLOR["text_dim"])
    h(f'<div style="border:1px solid {C.COLOR["line"]};border-left:3px solid {_c};'
      f'border-radius:0 6px 6px 0;padding:8px 11px;margin-bottom:6px">'
      f'<div style="font-size:12px;font-weight:700">{f["name"]}　'
      f'<span style="color:{_c}">{f["verdict"]}</span>　'
      f'<span style="font-size:10.5px;font-weight:400;color:{C.COLOR["text_dim"]}">'
      f'IC {f["ic_mean"]:+.3f}　IC_IR {f["ic_ir"]:+.2f}　t {f["t"]:+.2f}　'
      f'p {f["p"]:.3f}　胜率 {f["win"]:.0f}%　{f["n"]} 期</span></div>'
      f'<div style="font-size:11px;line-height:1.75;margin-top:3px">{f["why"]}</div></div>')

    c1, c2 = st.columns([0.62, 0.38], gap="small")
    with c1:
        st.plotly_chart(_ic_fig(f), width="stretch", config={"displayModeBar": False})
    with c2:
        st.plotly_chart(_layer_fig(f), width="stretch", config={"displayModeBar": False})

    _s = pd.Series({k: v for k, v in f.get("ic_series", {}).items() if v is not None})
    _pos = int((_s > 0).sum())
    _neg = int((_s < 0).sum())
    _maxq = _s.abs().idxmax() if len(_s) else "—"
    h(f'<div class="cg-note"><b>怎么读</b>：左图蓝柱是单期 IC（正＝因子值高的行业次期涨得多），'
      f'橙线是累计 IC——如果累计线一路向下，说明它长期反向；'
      f'如果只有一两期贡献，说明不稳定。右图是五分层：'
      f'<b>严格递增才是好因子</b>。<br>'
      f'本期该因子：{_pos} 期为正、{_neg} 期为负，绝对值最大的一期是 '
      f'<b>{_maxq}</b>（{_s.get(_maxq, float("nan")):+.3f}）。'
      + (f' 分层从 Q1 {f["layers"]["0"]:+.2f}% 到 Q5 {f["layers"]["4"]:+.2f}%，'
         f'单调性 {f["mono"]:+.2f}。' if f.get("layers") else '')
      + '</div>')

    # =====================================================================
    # 结论怎么影响前面的判断
    # =====================================================================
    h('<div class="sec">五、这些结果怎么反过来影响「结论总览」</div>')
    h('<div class="cg-note" style="line-height:1.95">'
      '· <b>「可以跟」不是「会涨」</b>：规则筛的是「机构在做 + 多数在加 + 价格不贵」，'
      '这是<b>赔率与风险的组合</b>，不是收益预测。因子检验已经证明，'
      '这些指标对次季收益没有稳定预测力。<br>'
      '· <b>广度只能用来判断共识是否成立</b>，不能用来追涨。'
      '它的 IC 是负的（−0.04，10% 显著），意味着「大家一窝蜂加」之后往往跑输——'
      '这恰好支持「在拥挤处减仓、在冷门处观察」的做法，而不是「跟着加」。<br>'
      '· <b>配置分位、价格分位、CSI 都不显著</b>：分层方向与「拥挤即风险」一致（配置分位 Q1 +2.39% → Q5 +0.51%），但 t 值只有 −0.83、p=0.42，且不单调，所以只能当风险提示，不能当卖出信号。'
      '页面保留拥挤度，是为了<b>控制仓位大小</b>（挤的地方少放钱），'
      '不是因为它能预测下跌。<br>'
      '· <b>真正有用的信息不是「预测」，而是「约束」</b>：'
      '知道机构在撤（没人接）就不要抄底，知道仓位已在历史高位就不要满仓——'
      '这些不提高收益，但能显著降低踩踏的概率。</div>')

    h('<div class="sec">六、局限：为什么结论可能偏保守</div>')
    h('<div class="cg-note" style="line-height:1.9">'
      f'· <b>样本短</b>：{d["sample"]["quarters"]} 期、31 个行业，'
      f'横截面最多 31 个观测，统计功效有限；半年频因子只有 14 期，几乎测不出东西。<br>'
      '· <b>行业收益用申万一级指数</b>：基金实际持有的是个股，行业指数与「基金能买到的收益」'
      '之间有跟踪误差；等权行业组合也不是可投资组合。<br>'
      '· <b>没有交易成本与容量约束</b>：即使 IC 为正，扣掉换手成本后也可能不赚钱。<br>'
      '· <b>多重检验</b>：9 个因子同时测，出现一个 p&lt;0.1 的结果很正常，'
      '不能当成「发现」；要样本外再验证一季。<br>'
      '· <b>前视已尽量排除但仍不完美</b>：分位用扩张窗口、收益用季末到季末，'
      '但基金持仓的披露时滞（约 3–4 周）没有在 IC 里扣除——'
      '真实可交易窗口比测试窗口短。</div>')

    if st.button("重算全部因子（约 60–80 秒）", key="ft_calc"):
        with st.spinner("正在重算…"):
            _recompute()
        st.rerun()
