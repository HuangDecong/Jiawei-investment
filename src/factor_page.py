# -*- coding: utf-8 -*-
"""⑤ 因子检验：每个季报因子「衡量什么、怎么算、什么情况下失灵」。

页面原则
--------
读者要的不是「怎么看 IC 柱状图」，而是：

    这个因子对应现实里的什么行为 → 为什么值得看 → 怎么算（可复算）
    → 阈值多少、阈值哪来的 → 什么情况下失效 → 有什么风险 → 检验结果如何

所以本页以「因子说明书」为主体，检验数字放在每个因子的卡片里，
不再单独用大段文字讲统计流程。

数据来自 data/derived/factor_test.json（tools/run_factor_test.py 生成）：
现算要 60–80 秒，读 JSON 不到 10ms。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import canon as cn
import config as C
import report_charts as rc
import ui
import caliber as CAL
from factor_doc import DOC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, "data", "derived", "factor_test.json")

_VERDICT_COLOR = {"强": C.COLOR["up"], "可用": C.COLOR["up"], "仅辅助": "#E8710A",
                  "反向": C.COLOR["accent"], "无效": C.COLOR["text_dim"],
                  "样本不足": "#C3CBD4"}

# 速查表用的一句话（详细解释放在下方的因子说明书里）。
# 表格里的长文本列在 1080px 宽度下必然被截断，与其让读者去点开，
# 不如给一句能读完的定义。
_BRIEF = {
    "breadth": "多少基金在加（共识度）",
    "active_q": "剥离价格后的净买入",
    "delta_q": "一季的配置结构移动",
    "delta_y": "一年的配置意图",
    "cfg_pct": "押注在自身历史的位置",
    "idx_pct": "价格相对自身历史贵不贵",
    "csi": "配置＋价格＋资金的综合拥挤",
    "dispersion": "基金之间的分歧有多大",
    "active_h": "半年报口径的真加仓",
}


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
    import factor_test as ft
    ft.run_all()
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
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(s.index), y=s.values, name="单期 IC",
                         marker_color=[C.COLOR["up"] if v > 0 else C.COLOR["down"]
                                       for v in s.values], opacity=0.8,
                         hovertemplate="%{x}　IC %{y:+.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=list(s.index), y=s.cumsum().values, name="累计 IC",
                             mode="lines+markers", line=dict(color=C.COLOR["price"], width=2),
                             yaxis="y2",
                             hovertemplate="%{x}　累计 %{y:+.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C.COLOR["line"], width=1))
    _tk = [q for q in s.index if str(q).endswith("Q1")] + [s.index[-1]]
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                  title="累计 IC", tickfont=dict(size=10)),
                      yaxis=dict(title="单期 IC", gridcolor=C.COLOR["grid"],
                                 tickfont=dict(size=10)),
                      xaxis=dict(tickmode="array", tickvals=_tk, ticktext=_tk,
                                 tickangle=-45, tickfont=dict(size=10)),
                      legend=dict(orientation="h", y=1.14, font=dict(size=10)))
    fig = rc._base(fig, "", 320, legend=True)
    fig.update_layout(margin=dict(l=48, r=48, t=34, b=34))
    return fig


def _layer_fig(d: dict) -> go.Figure:
    """五分层平均次期收益：看单调性。"""
    L = d.get("layers") or {}
    if not L:
        return rc._base(go.Figure(), "无分层数据")
    ks = sorted(L, key=lambda x: int(x))
    xs = [f"Q{int(k)+1}" for k in ks]
    ys = [L[k] for k in ks]
    fig = go.Figure(go.Bar(x=xs, y=ys,
                           marker_color=[C.COLOR["down"] if v < 0 else C.COLOR["up"]
                                         for v in ys],
                           text=[f"{v:+.2f}%" for v in ys], textposition="outside",
                           textfont=dict(size=10),
                           hovertemplate="%{x}　次期平均 %{y:+.2f}%<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C.COLOR["line"], width=1))
    fig.update_yaxes(title_text="次期平均收益 %")
    fig.update_xaxes(title_text="按因子值五等分（Q1 低 → Q5 高）")
    fig = rc._base(fig, "", 300)
    fig.update_layout(margin=dict(l=48, r=16, t=22, b=42))
    return fig


def _field(label: str, text: str, color: str) -> str:
    return (f'<div style="display:flex;gap:10px;padding:7px 0;'
            f'border-top:1px solid {C.COLOR["grid"]}">'
            f'<div style="flex:0 0 84px;font-size:12.5px;font-weight:700;'
            f'color:{color};line-height:1.85">{label}</div>'
            f'<div style="flex:1;font-size:13.5px;line-height:1.95;color:#26374A">'
            f'{text}</div></div>')


def render(h, B=None, IND=None) -> None:
    d = _load()
    if not d:
        h('<div class="cg-note">尚未生成因子检验结果。运行 '
          '<span class="mono">python tools/run_factor_test.py</span> 生成。</div>')
        if st.button("现在现算（约 60–80 秒）", key="ft_calc0"):
            with st.spinner("正在检验 9 个因子…"):
                d = _recompute()
            st.rerun()
        return

    F = d.get("factors", [])
    _bd = next((f for f in F if f.get("key") == "breadth"), {})
    _bd_ic = float(_bd.get("ic_mean", float("nan")))
    _bd_t = float(_bd.get("t", float("nan")))
    _bd_n = _bd.get("n", "—")
    if not F:
        h('<div class="cg-note">结果文件为空，请重跑 tools/run_factor_test.py。</div>')
        return

    _ok = [f for f in F if f["verdict"] in ("强", "可用")]
    _rev = [f for f in F if f["verdict"] == "反向"]
    _no = [f for f in F if f["verdict"] in ("无效", "样本不足")]
    _aux = [f for f in F if f["verdict"] == "仅辅助"]
    _best = max(F, key=lambda x: abs(x.get("ic_mean") or 0))

    # =====================================================================
    # 结论
    # =====================================================================
    h('<div class="sec">一、结论：这些因子能干什么、不能干什么</div>')
    h(ui.concl_board([
        f'<b>9 个季报因子，没有一个能预测次季收益。</b>'
        f'按标准流程（扩张窗口分位、Spearman IC、Newey-West t、五分层）在 '
        f'{d["sample"]["q0"]}–{d["sample"]["q1"]} 上检验：'
        f'{len(_ok)} 个可用、{len(_aux)} 个仅辅助、{len(_rev)} 个反向显著、'
        f'{len(_no)} 个无效。',
        # 方向必须跟着数据走：偶尔一次重算会把 IC 由负转正，
        # 写死「方向是负的」就会与同一页的卡片、分层收益自相矛盾。
        (f'<b>唯一接近显著的是「{_best["name"]}」，方向为'
         + ("正" if _best["ic_mean"] >= 0 else "负") + '</b>'
         f'（IC {_best["ic_mean"]:+.3f}、t {_best["t"]:+.2f}、p {_best["p"]:.3f}）：'
         + (f'加仓越一致的行业，次季<b>略强</b>——这与「共识形成＝边际买家减少」的'
            f'假设相反；但 t 未过 1.96、p={_best["p"]:.2f}，不能当买入信号。'
            if _best["ic_mean"] >= 0 else
            f'加仓越一致的行业，次季反而更弱——这是「共识形成」的典型现象。')),
        f'<b>所以这套系统的定位是「读季报」，不是「选行业」。</b>'
        f'因子在这里的作用有三个：描述机构在做什么（广度/主动净额）、'
        f'度量风险积累到什么程度（CSI/分位）、约束仓位（拥挤度惩罚）。'
        f'它们不产生买入信号。',
    ]))
    h(_kpis([
        ("检验因子", f"{len(F)} 个", "覆盖季报可算的全部指标"),
        ("正向可用", f"{len(_ok)} 个", "IC≥0.05 且 t≥1.96"),
        ("反向显著", f"{len(_rev)} 个", "方向与定义相反"),
        ("无效", f"{len(_no)} 个", "|t|<1.64"),
        ("样本", f'{d["sample"]["quarters"]} 期', f'{d["sample"]["industries"]} 行业 × '
                                                f'{d["sample"].get("n_funds", "—")} 只基金'),
    ]))

    # =====================================================================
    # 速查表
    # =====================================================================
    h('<div class="sec">二、九个因子一览：各自衡量什么、结论如何</div>')
    _rows = []
    for f in F:
        _rows.append({
            "因子": f["name"],
            "衡量什么": _BRIEF.get(f["key"], f["name"]),
            "IC": f.get("ic_mean"), "t": f.get("t"), "p": f.get("p"),
            "结论": f["verdict"],
        })
    _t = pd.DataFrame(_rows).sort_values("IC", key=lambda s: s.abs(), ascending=False)
    st.dataframe(
        _t, width="stretch", hide_index=True, height=372,
        column_config={
            "因子": st.column_config.TextColumn(width="small"),
            "衡量什么": st.column_config.TextColumn(width="medium"),
            "IC": st.column_config.NumberColumn(format="%+.3f", width="small"),
            "t": st.column_config.NumberColumn(format="%+.2f", width="small"),
            "p": st.column_config.NumberColumn(format="%.3f", width="small"),
            "结论": st.column_config.TextColumn(width="small"),
        })
    h('<div class="tiny">IC＝每期 31 个行业上「因子值 vs 次期收益」的 Spearman 秩相关均值；'
      't 用 Newey-West（滞后 4 期）修正自相关；p 为双侧。'
      f'判定阈值事前固定：|IC|&lt;0.03 或 |t|&lt;1.64 记「无效」；'
      f'IC 为负且显著记「反向」；|IC|≥0.05、IC_IR≥0.5、|t|≥1.96 记「可用」。</div>')

    # =====================================================================
    # 逐因子说明书
    # =====================================================================
    h('<div class="sec">三、九个因子：逐个展开，看完整说明书</div>')
    h('<div class="tiny" style="margin:-2px 0 8px">'
      '九张卡片默认折叠，标题行只留「名称 · 结论」——IC/t 在上表已列，不重复；'
      '点开任意一张，可看到经济内容、投资含义、公式、步骤、阈值、失效条件与风险。'
      '第一张（最显著的因子）默认展开。</div>')
    # 按 |IC| 降序：最值得看的排最前，与上方速查表顺序一致
    F_sorted = sorted(F, key=lambda x: abs(x.get("ic_mean") or 0), reverse=True)
    for _i, f in enumerate(F_sorted, 1):
        doc = DOC.get(f["key"], {})
        _c = _VERDICT_COLOR.get(f["verdict"], C.COLOR["text_dim"])
        # 标题只留「序号 · 名称 · 结论」：IC / t / p / 期数在上方速查表里已有一行，
        # 这里再抄一遍，折叠状态下看起来就是同一张表重复列了九次。
        _lab = f'{_i}. {f["name"]}　·　{f["verdict"]}' 
        with st.expander(_lab, expanded=(_i == 1)):
            _steps = doc.get("steps") or []
            _steps_html = "<br>".join(f"{k}. {v}" for k, v in enumerate(_steps, 1)) or "—"
            _formula_html = ('<span style="font-family:ui-monospace,Consolas,monospace;'
                             'font-size:12.5px">' + str(f.get("formula", "—")) + "</span>")
            h(f'<div style="border:1px solid {C.COLOR["line"]};border-left:4px solid {_c};'
              f'border-radius:9px;padding:10px 14px;background:#FFFFFF">'
              f'<div style="font-size:12px;color:{C.COLOR["text_dim"]}">'
              f'IC {f["ic_mean"]:+.3f}（t {f["t"]:+.2f}，{f["n"]} 期）'
              f'　·　胜率 {f["win"]:.0f}%　·　单调性 '
              f'{(f["mono"] if f.get("mono") is not None else float("nan")):+.2f}'
              f'<span style="margin-left:8px">（判据见上方速查表）</span></div>' 
              + _field("经济内容", doc.get("econ", "—"), C.COLOR["price"])
              + _field("投资含义", doc.get("invest", "—"), C.COLOR["accent"])
              + _field("为什么这样算", doc.get("why", "—"), C.COLOR["text_dim"])
              + _field("公式", _formula_html, C.COLOR["price"])
              + _field("计算步骤", _steps_html, C.COLOR["text_dim"])
              + _field("判定阈值", doc.get("standard", "—"), C.COLOR["threshold"])
              + _field("阈值来源", doc.get("src", "—"), C.COLOR["text_dim"])
              + _field("失效条件", doc.get("fail", "—"), C.COLOR["level_拥挤"])
              + _field("使用风险", doc.get("risk", "—"), C.COLOR["threshold"])
              + '</div>')

            _c1, _c2 = st.columns([0.62, 0.38], gap="small")
            with _c1:
                st.plotly_chart(_ic_fig(f), width="stretch",
                                config={"displayModeBar": False})
            with _c2:
                st.plotly_chart(_layer_fig(f), width="stretch",
                                config={"displayModeBar": False})

            _s = pd.Series({k: v for k, v in f.get("ic_series", {}).items()
                            if v is not None})
            if len(_s):
                _pos, _neg = int((_s > 0).sum()), int((_s < 0).sum())
                _maxq = _s.abs().idxmax()
                h(f'<div class="tiny">左图：每根柱是一期的 IC，橙线是累计 IC——'
                  f'累计线一路向下说明长期反向，只靠一两期撑起说明不稳定；'
                  f'右图：按因子值五等分，Q1→Q5 严格递增才是好因子。<br>'
                  f'该因子 {_pos} 期为正、{_neg} 期为负，绝对值最大的一期是 '
                  f'<b>{_maxq}</b>（{_s.get(_maxq, float("nan")):+.3f}）。'
                  + (f' 分层从 Q1 {f["layers"]["0"]:+.2f}% 到 Q5 {f["layers"]["4"]:+.2f}%，'
                     f'单调性 {f["mono"]:+.2f}。' if f.get("layers") else '') + '</div>')
            else:
                _nn = int(f.get("n") or 0)
                h(ui.data_insufficient(
                    _nn, CAL.MIN_IC_N,
                    "该因子只有最近几期有值（历史口径断层），"
                    "IC 与分层回测没有统计意义；保留原始数值供核对"))

    # =====================================================================
    # 为什么不用旧因子
    # =====================================================================
    with st.expander("附：为什么换掉早期那套因子（F1 / F2 / F11）", expanded=False):
        # 期数不能写死：每过一个季度就会变，写死会在下一季变成错数。
        _n_csi = len(cn.csi_history())
        h('<div class="cg-note" style="font-size:12.5px">'
          '早期用的是 <b>F1 超配比例</b>（基金配置 − 基准权重）、<b>F2 超配 Z</b>、'
          '<b>F3 超配分位</b>、<b>F11 综合拥挤度</b>（多分量加权）、<b>F15 拥挤背离</b>，'
          '来自 Wind 持仓 + 基准权重 + 成交额占比，43 期。换掉的四个原因：<br>'
          '① <b>两套口径打架</b>：超配要用沪深300/中证800 的行业权重，而页面其他数字来自'
          '东财持仓，同一行业会出现两个超配率。新因子全部来自 canon 一套数据。<br>'
          '② <b>不可复现</b>：F11 是拟合加权的复合因子，权重没有样本外验证，'
          '外人算不出同一个数；新因子每个只有一个公式。<br>'
          '③ <b>含价格漂移</b>：旧超配比例没剥离涨跌，行业涨了没人买也会显示超配上升。<br>'
          '④ <b>依赖 Wind 授权</b>：接口会截断、换机器就跑不了；新因子只用东财公开披露。<br>'
          '<b>代价</b>：算不了「相对沪深300 的超配」（只能用同行平均作基准）；'
          f'期数口径也随之改变（旧：Wind 43 期；新：东财持仓，CSI 当前可算 {_n_csi} 期）；'
          '没有换手率/成交额类「交易拥挤」因子——'
          '这是取舍，不是单向升级。</div>')

    # =====================================================================
    # 对前面结论的影响
    # =====================================================================
    h('<div class="sec">四、这些结果怎么反过来约束前面的判断</div>')
    h(ui.concl_board([
        '<b>「可以跟」不是「会涨」</b>：规则筛的是「机构在做 + 多数在加 + 价格不贵」，'
        '这是赔率与风险的组合。因子检验已证明这些指标对次季收益没有稳定预测力。',
        ('<b>广度只能用来判断共识是否成立</b>：它的 IC 为 '
         f'{_bd_ic:+.3f}（t={_bd_t:+.2f}、n={_bd_n}），'
         + ('方向为负但未过显著线——' if (not np.isfinite(_bd_t) or abs(_bd_t) < 1.96)
            else '方向为负且显著——')
         + '意味着「大家一窝蜂加」之后并不占优。这支持「在拥挤处减仓、在冷门处观察」，'
         '而不是「跟着加」。'),
        '<b>拥挤度只能用来约束仓位</b>：CSI 与配置分位的 t 值都不显著，'
        '但方向与「拥挤即风险」一致。所以 ≥80 时压仓位上限，而不是清仓。',
        '<b>真正有用的是「约束」而不是「预测」</b>：知道机构在撤就别抄底、'
        '知道仓位已在历史高位就别满仓——这些不提高收益，但能显著降低踩踏概率。',
    ], numbered=False))

    with st.expander("附：稳健性与局限（数字不好看，但必须写）", expanded=False):
        try:
            import playbook as _pb
            _sens = _pb.sensitivity()
            st.dataframe(_sens[["情景", "可以跟", "要回避", "与基准重合", "重合率"]],
                         width="stretch", hide_index=True, height=210,
                         column_config={"重合率": st.column_config.ProgressColumn(
                             format="%.0f", min_value=0, max_value=100)})
            _min = float(_sens["重合率"].min())
            h(f'<div class="tiny">把「可以跟」「要回避」的门槛各扰动 ±20%，看名单换不换人：'
              f'最低重合率 {_min:.0f}%。回避名单在放宽方向完全稳定、收紧方向不稳；'
              f'可以跟名单在严格方向会清空——说明这几家的共识只是刚过 60% 这条线，'
              f'仓位要按「试探」而不是「重仓」给。</div>')
        except Exception as _e:
            h(f'<div class="tiny">稳健性表暂不可用：{type(_e).__name__}</div>')
        h(f'<div class="cg-note" style="font-size:12.5px">'
          f'· <b>样本短</b>：{d["sample"]["quarters"]} 期、31 个行业，横截面最多 31 个观测，'
          f'统计功效有限；半年频因子只有 14 期，几乎测不出东西。<br>'
          f'· <b>收益用申万一级指数</b>：基金实际持有的是个股，指数与「基金能买到的收益」'
          f'之间有跟踪误差；等权行业组合也不是可投资组合。<br>'
          f'· <b>没有交易成本与容量约束</b>：即使 IC 为正，扣掉换手成本后也可能不赚钱。<br>'
          f'· <b>多重检验</b>：9 个因子同时测，出现一个 p&lt;0.1 的结果很正常，'
          f'不能当成「发现」。<br>'
          f'· <b>披露时滞没扣</b>：IC 用季末到季末的收益，但持仓要 3–4 周后才可得，'
          f'真实可交易窗口比测试窗口短。</div>')

    if st.button("重算全部因子（约 60–80 秒）", key="ft_calc"):
        with st.spinner("正在重算…"):
            _recompute()
        st.rerun()

    ui.source_footer(h, "factor", "rankic")
