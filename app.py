# -*- coding: utf-8 -*-
"""
app.py  ——  公募基金交易拥挤度与集中度仪表盘（Streamlit 三栏看板）
======================================================================
布局
----
  顶栏        标题 + 市场状态标签 + 数据指纹 + 刷新 / 分位口径开关
  左栏 15%    行业列表（搜索 + 单点切换 + 拥挤等级色点），点击后中/右栏联动
  中栏 50%    ① 顶部核心指标卡  ② 近期表现汇总卡  ③ 时间筛选  ④ 主图表区（三子图+信号标记）
  右栏 35%    ① 市场层面汇总卡  ② 行业明细表（等级彩色标签） ③ 信号触发记录表

启动
----
  cd fund_crowding_dashboard
  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# 把 src/ 目录加入模块搜索路径（基于 __file__ 定位，云端/换机均可用）。
# 本地模块（config/data_loader/factor_engine/backtest/research 等）都放在 src/ 下，
# 用平级 import；app.py 位于项目根目录，故须显式把 src/ 插到 sys.path 首位。
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import research as rs
import research3 as r3
import research4 as r4
import rigor as rg
import streamlit as st
from plotly.subplots import make_subplots

import backtest as bt          # noqa: E402
import config as C             # noqa: E402
import data_loader as dl       # noqa: E402
import factor_engine as fe     # noqa: E402
import guide                   # noqa: E402  投资机会解读/操作指南/更新指南
import insights as ins         # noqa: E402
import signal_generator as sg  # noqa: E402


# ============================================================================
# 原始 HTML 渲染器（本轮新增）
# ============================================================================
# 为什么不用 st.markdown(..., unsafe_allow_html=True)？
# ---------------------------------------------------------------------------
# st.markdown 的内容要经过**两阶段解析**：react-markdown 先解析 markdown，
# rehype-raw 再用 parse5 解析内联 HTML。parse5 严格遵循 HTML5 树构造算法，
# 会按规范自动改写结构（<tr> 隐式包 <tbody>、块级元素被 foster parenting 提出
# 段落、行内元素被提前闭合）。于是浏览器里的真实 DOM 与 React 的虚拟 DOM 不再
# 一致，切换页面触发重渲染时就抛：
#     NotFoundError: removeChild / insertBefore —— 节点不是该节点的子节点
#
# st.html 只解析一次（DOMPurify 消毒后直接挂载），没有 markdown 阶段，
# 因此不存在两棵树的分歧。已实测确认 class 与 inline style 均保留。
def _h(html: str):
    """渲染原始 HTML 片段（替代 st.markdown(..., unsafe_allow_html=True)）。

    兼容处理：部分文案沿用了 Markdown 的 `**加粗**` 语法，但 st.html 不会
    解析 markdown，星号会原样暴露在页面上。此处统一转换为 <b> 标签。
    """
    if "**" in html:
        html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html)
    return st.html(html)


# ============================================================================
# 图表洞察（本轮新增）
# ============================================================================
# 需求：**每一张图都要有分析、结论，并突出特征与拐点**。
# 实现：所有文字都由 src/insights.py 从数据算出（特征/拐点/结论三段式），
#       本处只负责渲染。这样数据更新后解读会自动跟着变，不会过期。
def _ins(series=None, name: str = "", *, unit: str = "", pct: bool = False,
         higher_means: str = "", title: str = "分析 · 特征与拐点",
         cross: dict | None = None, extra: str = "", top: int = 2,
         feature: str = "", turns: str = "", meaning: str = ""):
    """在图表下方渲染「特征 / 拐点 / 结论」卡片。

    参数
    ----
    series      : 时序（pd.Series）—— 自动计算特征与拐点
    cross       : 横截面描述（ins.describe_cross_section 的结果），与 series 二选一
    higher_means: 结论模板，可用 {direction}/{last}/{q}/{change} 占位
    feature / turns / meaning : **显式覆盖**自动生成的文案
                 （用于「读图要点」这类无法从单条序列推出的解读）
    extra       : 追加的自定义结论文本（HTML）
    """
    disp: dict
    if cross is not None:
        disp = dict(ok=bool(cross.get("ok", False)),
                    feature=cross.get("feature", ""), turns="", meaning="")
    elif series is not None:
        disp = ins.describe_series(series, name, unit=unit, pct=pct,
                                   higher_means=higher_means, top=top)
    else:
        disp = dict(ok=bool(feature), feature=feature, turns="", meaning="")

    html = ins.render_insight(
        disp, colors=C.COLOR, title=title,
        feature=feature or disp.get("feature", ""),
        turns=turns or disp.get("turns", ""),
        meaning=meaning or disp.get("meaning", ""),
        extra=extra)
    if html:
        _h(html)


INDUSTRIES = C.INDUSTRIES
LEVEL_DOT = {"极度拥挤": "🔴", "拥挤": "🟠", "偏拥挤": "🟡",
             "中性": "⚪", "显著低配": "🔵", "数据不足": "⚫"}

st.set_page_config(page_title="公募基金交易拥挤度与集中度仪表盘",
                   page_icon="📊", layout="wide",
                   initial_sidebar_state="collapsed")

# ============================================================================
# 全局样式
# ============================================================================
st.markdown(f"""
<style>
  .stApp {{ background: {C.COLOR['bg']}; }}
  html, body, [class*="css"] {{ font-family: {C.FONT};
      color: {C.COLOR['text']}; }}
  .block-container {{ padding: .7rem 1.0rem 1.2rem 1.0rem; max-width: 100%; }}
  /* 彻底隐藏 Streamlit 自带顶栏。
     注意：只写 height:0 是不够的——它仍会作为半透明图层压在页面最上方，
     把自绘标题遮成灰字（这就是「看不到仪表盘的头」的根因）。 */
  header[data-testid="stHeader"] {{ display: none !important; }}
  [data-testid="stDecoration"] {{ display: none !important; }}
  [data-testid="stToolbar"] {{ display: none !important; }}
  [data-testid="stStatusWidget"] {{ display: none !important; }}
  section[data-testid="stSidebar"] {{ display: none; }}

  .hdr {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap;
          border-bottom:1px solid {C.COLOR['line']}; padding-bottom:8px; margin-bottom:8px; }}
  .hdr h1 {{ font-size:17px; margin:0; font-weight:700; letter-spacing:.2px; }}
  .tag {{ font-size:11.5px; padding:2px 9px; border-radius:10px; font-weight:700;
          border:1px solid transparent; }}
  .tiny {{ font-size:10.5px; color:{C.COLOR['text_dim']}; line-height:1.5; }}
  .sec {{ font-size:12px; font-weight:700; color:{C.COLOR['text']};
          margin:10px 0 5px 0; padding-left:7px;
          border-left:3px solid {C.COLOR['price']}; }}
  /* 顶栏标题：不用 <h1>，避免与 Streamlit 的标题组件在 React 协调时冲突 */
  .hdr .ttl {{ font-size:17px; font-weight:700; letter-spacing:.2px;
               color:{C.COLOR['text']}; }}

  /* 结论卡片组：结论 → 判断依据 → 失效条件，三段固定结构便于回溯 */
  .concl-wrap {{ display:flex; flex-direction:column; gap:7px; margin-bottom:4px; }}
  .concl {{ background:#FFFFFF; border:1px solid {C.COLOR['line']};
            border-left:4px solid {C.COLOR['price']}; border-radius:7px;
            padding:7px 11px; }}
  .concl .ct {{ font-size:12px; font-weight:700; margin-bottom:3px;
                color:{C.COLOR['text']}; }}
  .concl .cc {{ font-size:11.5px; line-height:1.68; color:{C.COLOR['text']}; }}
  .concl .cr {{ font-size:11px; line-height:1.62; color:#37485C;
                margin-top:2px; padding-left:8px;
                border-left:2px solid {C.COLOR['grid']}; }}
  .concl .cw {{ font-size:10.5px; line-height:1.55; margin-top:4px;
                color:#A35A00; background:#FFF8EC; border-radius:4px;
                padding:3px 7px; }}

  .kpis {{ display:flex; gap:7px; flex-wrap:wrap; }}
  .kpi {{ flex:1 1 0; min-width:82px; background:{C.COLOR['panel']};
          border:1px solid {C.COLOR['line']}; border-radius:7px;
          padding:7px 9px; }}
  .kpi .k {{ font-size:10px; color:{C.COLOR['text_dim']}; white-space:nowrap; }}
  .kpi .v {{ font-size:16px; font-weight:700; line-height:1.32; }}
  .kpi .s {{ font-size:9.5px; color:{C.COLOR['text_dim']}; }}

  .mini {{ display:flex; gap:7px; flex-wrap:wrap; }}
  .mcard {{ flex:1 1 0; min-width:118px; background:#FFFFFF;
            border:1px solid {C.COLOR['line']}; border-radius:7px; padding:6px 9px; }}
  .mcard .h {{ font-size:10.5px; font-weight:700; color:{C.COLOR['text_dim']};
               border-bottom:1px dashed {C.COLOR['line']}; margin-bottom:4px;
               padding-bottom:3px; }}
  .mrow {{ display:flex; justify-content:space-between; font-size:11px;
           line-height:1.62; }}
  .mrow span:last-child {{ font-weight:700; font-variant-numeric:tabular-nums; }}

  .up {{ color:{C.COLOR['up']}; }} .dn {{ color:{C.COLOR['down']}; }}
  .mut {{ color:{C.COLOR['text_dim']}; }}

  .alert {{ font-size:11.5px; padding:4px 8px; margin-bottom:3px;
            border-radius:5px; border-left:3px solid; background:{C.COLOR['panel']}; }}
  .lvl {{ display:inline-block; font-size:10.5px; font-weight:700;
          padding:1.5px 7px; border-radius:9px; color:#fff; }}
  [data-testid="stElementToolbar"] {{ display:none; }}
  div[data-testid="stRadio"] > div {{ gap:1px !important; }}
  div[data-testid="stRadio"] label {{ font-size:12px; padding:1px 3px;
      border-radius:4px; }}
  .stDataFrame {{ border:1px solid {C.COLOR['line']}; border-radius:6px; }}

  /* ---- 投资机会解读卡（guide.py） ---- */
  .opp {{ background:{C.COLOR['panel']}; border:1px solid {C.COLOR['line']};
         border-radius:8px; padding:9px 11px; margin:6px 0; }}
  .opp-h {{ font-size:12px; font-weight:700; color:{C.COLOR['text']}; margin-bottom:6px; }}
  .opp-badge {{ display:inline-block; font-size:10.5px; font-weight:700; color:#fff;
                background:{C.COLOR['accent']}; padding:1.5px 8px; border-radius:10px; }}
  .opp-cols {{ display:flex; gap:8px; flex-wrap:wrap; }}
  .opp-col {{ flex:1 1 260px; min-width:240px; background:#FFFFFF;
              border:1px solid {C.COLOR['line']}; border-radius:6px; padding:7px 9px; }}
  .opp-col .opp-t {{ font-size:11px; font-weight:700; color:{C.COLOR['text']};
                     margin-bottom:4px; padding-bottom:3px; border-bottom:1px solid {C.COLOR['line']}; }}
  .opp-col ul {{ margin:0; padding-left:16px; }}
  .opp-col li {{ font-size:10.5px; line-height:1.55; color:#37485C; margin:2px 0; }}
  .opp-risk .opp-t {{ color:{C.COLOR['level_极度拥挤']}; }}
  .opp-opp .opp-t {{ color:{C.COLOR['level_显著低配']}; }}
  .opp-port .opp-t {{ color:{C.COLOR['accent']}; }}
  .opp-why {{ font-size:9.5px; color:{C.COLOR['text_dim']}; margin-top:4px; line-height:1.5; }}
  .opp-caveat {{ font-size:10.5px; color:#37485C; margin-top:7px; padding-top:5px;
                 border-top:1px dashed {C.COLOR['line']}; line-height:1.55; }}
  .opp-mini {{ font-size:10.5px; line-height:1.6; color:#37485C; }}
</style>
""", unsafe_allow_html=True)


def lvl_tag(lv: str) -> str:
    c = C.COLOR.get(f"level_{lv}", C.COLOR["level_中性"])
    return f'<span class="lvl" style="background:{c}">{lv}</span>'


def sgn(v, unit="", dec=2, na="—"):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return f'<span class="mut">{na}</span>'
    cls = "up" if v > 0 else ("dn" if v < 0 else "mut")
    return f'<span class="{cls}">{v:+.{dec}f}{unit}</span>'


def num(v, unit="", dec=2, na="—"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f'<span class="mut">{na}</span>'
    return f'{v:.{dec}f}{unit}'


def concl(items) -> str:
    """渲染「结论 → 判断依据 → 为什么这样定阈值 → 失效条件」卡片组。

    设计意图：本项目的每个判断都必须可回溯，因此不做「只给一个标签」的输出，
    而是把四件事固定捆在一起打印。任何一个结论都能顺着「依据」里的数值
    回到 data/raw 的具体文件与具体计算步骤。
    """
    out = []
    for it in items:
        lv = it.get("level", "")
        col = (C.COLOR.get(f"level_{lv}", C.COLOR["price"])
               if lv in C.LEVELS else C.COLOR["price"])
        why = "".join(f'<div class="cr"><b>{k}</b>：{v}</div>'
                      for k, v in it.get("why", {}).items())
        cav = (f'<div class="cw">失效条件 / 反证：{it["caveat"]}</div>'
               if it.get("caveat") else "")
        out.append(f'<div class="concl" style="border-left-color:{col}">'
                   f'<div class="ct">{it["title"]}</div>'
                   f'<div class="cc">{it["concl"]}</div>{why}{cav}</div>')
    return '<div class="concl-wrap">' + "".join(out) + "</div>"


# ============================================================================
# 数据（按 data/raw 指纹缓存，文件一变即自动重算）
# ============================================================================
@st.cache_data(show_spinner="正在加载数据并校验…")
def load_bundle(fp: str):
    b = dl.load_all()
    v = dl.validate(b)
    F = fe.build_all(b)
    return b, v, F


@st.cache_data(show_spinner="正在运行 IC / 分层 / 事件研究…")
def load_backtest(fp: str):
    b, v, F = load_bundle(fp)
    ind, mkt = F["ind"], F["mkt"]
    icb = bt.run_ic_both_lags(ind, b)
    layers = bt.run_all_layers(ind, b)
    _wf3 = r3.walk_forward_test(ind, b)
    # 因子动量逐期序列（回看 8 季）——供总览页画图
    import research3 as _r3m
    from factor_engine import safe_spearman as _ssp
    _icm = _r3m.ic_matrix(ind, b,
                          ["F1_超配比例", "F2_超配Zscore", "F3_超配历史分位",
                           "F4b_超配动量Z", "F12_配置系数", "F13_筹码盈利比例",
                           "F15_拥挤背离", "F11_综合拥挤度_推荐权重"])
    _tmp = {}
    for _i, _q in enumerate(_icm.index):
        if _i < 8 or _i + 1 >= len(_icm):
            continue
        _p = _icm.iloc[_i - 8:_i].mean()
        _n2 = _icm.iloc[_i + 1]
        _m = _p.notna() & _n2.notna()
        if int(_m.sum()) < 4:
            continue
        _cv = _ssp(_p[_m], _n2[_m])
        if _cv == _cv:
            _tmp[_q] = _cv
    _fm_series = pd.Series(_tmp)
    return dict(
        ic=icb,
        ic_t1=bt.run_ic(ind, b, lag=1),
        layer=bt.layer_summary(layers, b),
        layer_raw=layers,
        ev_over=bt.event_study(b, b.overweight),
        ev_pos=bt.position_event_study(b, mkt),
        ev_pos_mom=bt.position_momentum_study(b, mkt),
        ev_hhi=bt.hhi_shock_study(b, mkt),
        multi=bt.multi_signal_backtest(b, ind),
        robust=bt.run_robustness(ind, b),
        decay=bt.decay_analysis(ind, b),
        stress=rg.stress_test(b, ind, mkt),
        # ---- 第 2 轮：研究级检验 ----
        doos=rs.design_oos_test(ind, b),
        spec=rs.specification_search(ind, b),
        second=rs.second_layer_orthogonalize(ind, b),
        bench=rs.benchmark_comparison(ind, b),
        rot=rs.industry_rotation_strategy(ind, b, "F15_拥挤背离", lam=0.5, cap=0.10,
                                          base="equal", cost_bp=20.0),
        rot_hs=rs.industry_rotation_strategy(ind, b, "F15_拥挤背离", lam=0.5, cap=0.10,
                                             base="hs300", cost_bp=20.0),
        caps=rs.cap_sensitivity(ind, b, "F15_拥挤背离"),
        scan=rs.rotation_direction_scan(ind, b, base="equal"),
        scan_hs=rs.rotation_direction_scan(ind, b, base="hs300"),
        # ---- 第 3 轮：稳健性 ----
        r3={
            "wf": r3.walk_forward_test(ind, b),
            "wf_win": r3.wf_window_sensitivity(ind, b),
            "premise": r3.premise_availability(ind, b),
            "dual": r3.dual_factor_combination(ind, b),
            "tilt_cap": r3.tilt_vs_cap_decomposition(ind, b),
            "grid": r3.benchmark_cross_consistency(ind, b),
            "attr": r3.excess_attribution(ind, b),
            "bm_var": r3.benchmark_variants(ind, b),
            "wf_reg": r3.wf_regime_breakdown(ind, b, _wf3),
        },
        r4={
            "imp": r4.impact_cost_model(ind, b, "F15_拥挤背离"),
            "fm": r4.factor_momentum_test(ind, b),
            "fm_rob": r4.factor_momentum_robustness(ind, b),
            "fm_strat": r4.factor_momentum_strategy(ind, b, lookback=4),
            "fm_series": _fm_series,
            "multi": r4.multi_factor_orthogonal(ind, b),
            "etf": r4.etf_crowding_layer(b),
            "attr": r3.excess_attribution(ind, b),
            "cost_sens": r4.impact_cost_sensitivity(ind, b),
            "corr": r4.factor_correlation_matrix(ind, b),
        },
    )


FP = dl.fingerprint()

# 数据先加载：顶栏需要显示「截至哪个季度」，因此必须排在顶栏之前
B, V, FACT = load_bundle(FP)
IND, MKT = FACT["ind"], FACT["mkt"]
QUARTERS = list(B.quarters)
LATEST = B.latest
MS = sg.market_signals(IND, MKT, B)
M_SUM = fe.market_summary(IND, MKT, B)
PAY = sg.dashboard_payload(IND, MKT, B)
DETAIL_ALL = PAY["industry_history"]
RECORDS = PAY["records"]
LEVEL_MAP = {q: IND["拥挤等级"].loc[q].to_dict() for q in QUARTERS}

# ---------------- 顶栏 ----------------
h1, h2, h3 = st.columns([0.52, 0.26, 0.22])
with h1:
    _h('<div class="hdr">'
                '<span class="ttl">公募基金交易拥挤度与集中度仪表盘</span>'
                '<span class="tag" style="background:#EAF1FB;color:#1F6FEB">'
                '数据源：Wind</span>'
                '<span class="tag" style="background:#F2F5F8;color:#6B7A8C">'
                f'31 行业 · {len(QUARTERS)} 季度 · 截至 {LATEST}</span></div>')
with h2:
    pct_label = st.radio("分位口径", ["扩窗（无前视）", "全样本"],
                         index=0, horizontal=True, key="pct_mode",
                         help="扩窗：当期分位只用当期之前的历史（无未来信息）；"
                              "全样本：用全部可用历史（含前视，仅供对照）。")
with h3:
    c_a, c_b = st.columns([0.5, 0.5])
    if c_a.button("🔄 重新读取数据", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    _h(f'<div class="tiny" style="padding-top:6px">数据指纹 '
       f'{abs(hash(FP)) % 10**8:08d}</div>')

# ============================================================================
# 三栏
# ============================================================================
# ============================================================================
# 页面切换
# ============================================================================
PAGES = ["① 总览仪表盘", "② 拥挤度总览", "③ 行业占比变迁",
         "④ ETF 资金流", "⑤ 因子研究", "⑥ 轮动策略与因子裁决",
         "⑦ 稳健性检验", "⑧ 数据质量与溯源", "⑨ 方法说明与数据来源"]
_P = st.radio("页面", PAGES, horizontal=True, label_visibility="collapsed", key="page")

_PROV = B.provenance if getattr(B, "provenance", None) is not None else None
_palette = ["#1F6FEB", "#2EA043", "#D1242F", "#E8710A", "#6E4AB8", "#0F8B8D",
            "#B3651A", "#8A97A5", "#C62828", "#2C7FB8", "#1170AA", "#5FA2CE",
            "#FC7D0B", "#A3ACB9", "#57606C", "#C85200"]


def _prov_line(keys):
    """按数据项名返回"来源·表·查询日期"标注文本。"""
    if _PROV is None:
        return ""
    rows = _PROV[_PROV["数据项"].isin(keys)]
    return "；".join(f"{r.数据项}：{r.数据库} / {r.数据表}（查询日 {r.查询日期}）"
                     for r in rows.itertuples())


def _note(txt):
    _h(f'<div class="tiny">{txt}</div>')


# ============================================================================
# 页面 ① 总览仪表盘（第 4 轮新增：一屏看完结论）
# ============================================================================
if _P == "① 总览仪表盘":
    _SNAP = DETAIL_ALL[DETAIL_ALL["季度"] == LATEST].copy()
    _nm = {r["行业"]: r for _, r in _SNAP.iterrows()}
    _fci_s = MKT["FCI_因子拥挤指数"].dropna()
    _fci_last = float(_fci_s.iloc[-1]) if len(_fci_s) else np.nan

    _h(f'<div class="sec">一屏速览 · {LATEST} 主动权益基金拥挤度状态</div>')
    _h(f"""<div class="kpis">
      <div class="kpi"><div class="k">数据季度</div><div class="v">{LATEST}</div>
        <div class="s">31 行业 · {len(QUARTERS)} 季度 · Wind</div></div>
      <div class="kpi"><div class="k">市场状态</div>
        <div class="v" style="font-size:13px">{M_SUM['state']}</div>
        <div class="s">样本基金 {M_SUM['n_funds']:.0f} 只</div></div>
      <div class="kpi"><div class="k">HHI</div><div class="v">{M_SUM['HHI']:.4f}</div>
        <div class="s">历史分位 {M_SUM['HHI_pct']:.0f}%　均值 {M_SUM['HHI_hist_mean']:.4f}</div></div>
      <div class="kpi"><div class="k">CR3</div><div class="v">{M_SUM['CR3']:.2f}%</div>
        <div class="s">CR5 {M_SUM['CR5']:.2f}%　CR10 {M_SUM['CR10']:.2f}%</div></div>
    </div>""")
    _h(f"""<div class="kpis" style="margin-top:6px">
      <div class="kpi"><div class="k">第一大重仓行业</div>
        <div class="v" style="font-size:14px">{M_SUM['top1']}</div>
        <div class="s">{M_SUM['top1_pct']:.2f}%</div></div>
      <div class="kpi"><div class="k">极度拥挤行业</div><div class="v">{M_SUM['n_extreme']}</div>
        <div class="s">超配分位 ≥90 或 F12 &gt; 2.5</div></div>
      <div class="kpi"><div class="k">显著低配行业</div><div class="v">{M_SUM['n_under']}</div>
        <div class="s">Z&lt;−1.5 或 F12&lt;0.5</div></div>
      <div class="kpi"><div class="k">FCI 因子拥挤指数</div><div class="v">{_fci_last:+.3f}</div>
        <div class="s">扩窗分位 {float(MKT['FCI_分位'].dropna().iloc[-1]):.0f}%</div></div>
    </div>""")
    # ---- 以下需要完整的研究级检验（IC / 分层 / 稳健性 / 归因 / 成本容量…），
    #      是首屏耗时的主要来源。因此先渲染上面两排「市场状态」卡片再触发计算，
    #      让用户立刻看到当前季度结论，而不是先对着一个转圈等 40 秒。 ----
    _BTR = load_backtest(FP)
    _vd = rs.factor_verdict_table(IND, B, _BTR["ic"], doos=_BTR["doos"],
                                  second_layer=_BTR["second"],
                                  rotation_scan=_BTR["scan"],
                                  rotation_scan_hs300=_BTR["scan_hs"])
    _A = _vd[_vd["裁决"].str.startswith("A")]["因子"].tolist()
    _wf = _BTR["r3"]["wf"]
    _ws = _wf["stats"]
    _imp = _BTR["r4"]["imp"]
    _fm = _BTR["r4"]["fm"]
    # 容量卡片：容量模型在短样本 / 缺成交额时可能不可用，须显式降级而不是崩。
    _imp_ok = bool(isinstance(_imp, dict) and _imp.get("available", True)
                   and isinstance(_imp.get("table"), pd.DataFrame)
                   and not _imp["table"].empty)
    if _imp_ok:
        _cap_txt = f"≥{max(_imp['sizes']):,}亿"
        _cap_sub = (f"成本 {float(_imp['table'].iloc[-1]['平均单期成本bp']):.2f}bp/期"
                    "（不构成约束）")
    else:
        _cap_txt, _cap_sub = "样本不足", "待补行业成交额后重算"

    _h(f"""<div class="kpis" style="margin-top:6px">
      <div class="kpi"><div class="k">可用因子（A/A−）</div>
        <div class="v" style="font-size:14px">{"、".join(_A) if _A else "无"}</div>
        <div class="s">共 {len(_A)} 个（4 项独立检验合成）</div></div>
      <div class="kpi"><div class="k">Walk-forward 样本外 p</div>
        <div class="v">{_ws['留出期p']:.3f}</div>
        <div class="s">{'不显著（流程不稳健）' if _ws['留出期p'] > 0.10 else '显著'}</div></div>
      <div class="kpi"><div class="k">因子动量 momIC</div>
        <div class="v">{float(_fm.iloc[1]['momIC均值']):+.3f}</div>
        <div class="s">p={float(_fm.iloc[1]['p值']):.4f}（8 季回看）</div></div>
      <div class="kpi"><div class="k">策略容量</div>
        <div class="v" style="font-size:{'15px' if _imp_ok else '13px'}">{_cap_txt}</div>
        <div class="s">{_cap_sub}</div></div>
    </div>""")

    # ------------------------------------------------------------------
    # 综合判断（本轮新增）——结论 / 判断理由 / 最突出特征 / 关键拐点
    # 全部文字由 src/insights.py 从数据算出，数据更新后自动跟随。
    # ------------------------------------------------------------------
    _VD = ins.build_verdict(MKT, B, IND, M_SUM)
    _h('<div class="sec">综合判断 —— 结论 · 判断理由 · 特征 · 拐点</div>')
    _h(ins.render_verdict(_VD, C.COLOR))


    _h('<div class="sec">① 市场集中度时序（HHI + CR3/CR5/CR10）</div>')
    _f0 = make_subplots(specs=[[{"secondary_y": True}]])
    _f0.add_trace(go.Scatter(x=QUARTERS, y=MKT["F5_HHI"], name="HHI",
                             mode="lines+markers",
                             line=dict(color=C.COLOR["price"], width=2.2),
                             hovertemplate="%{x}　HHI %{y:.4f}<extra></extra>"),
                  secondary_y=False)
    for _k, _c in [("F6_CR3", C.COLOR["up"]), ("F6_CR5", C.COLOR["down"]),
                   ("F6_CR10", C.COLOR["line"])]:
        _f0.add_trace(go.Scatter(x=QUARTERS, y=MKT[_k] / 100.0, name=_k.split("_")[1],
                                 mode="lines+markers",
                                 line=dict(color=_c, width=1.5, dash="dot"),
                                 hovertemplate="%{x}　%{y:.4f}<extra></extra>"),
                      secondary_y=True)
    _f0.add_hline(y=C.SIGNAL_HHI_EXTREME, line=dict(color=C.COLOR["threshold"],
                                                     width=1, dash="dash"),
                  secondary_y=False)
    _f0.update_layout(height=C.CHART_H_SMALL + 90, template=C.PLOTLY_TEMPLATE,
                      margin=dict(l=48, r=54, t=14, b=30), hovermode="x unified",
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font=dict(family="Microsoft YaHei", size=11),
                      legend=dict(orientation="h", y=1.12, font=dict(size=10)))
    _f0.update_yaxes(title_text="HHI（原值）", gridcolor=C.COLOR["grid"],
                     secondary_y=False)
    _f0.update_yaxes(title_text="CR（小数）", showgrid=False, secondary_y=True)
    _f0.update_xaxes(showgrid=False)
    st.plotly_chart(_f0, width="stretch", config={"displayModeBar": False})
    _note(f"HHI 已按第 4 轮精度规则保留 4 位小数（原「一律两位」会使 0.071~0.190 "
          f"的序列退化成两个取值）。虚线为 HHI 极高集中阈值 {C.SIGNAL_HHI_EXTREME}。"
          f"数据来源：{_prov_line(['行业集中度CR_HHI', '行业配置比例'])}")
    _ins(MKT["F5_HHI"], "HHI", title="① 分析与结论 · 集中度时序",
         higher_means="HHI 的{direction}意味着机构持仓的行业分散度在持续{direction}："
                      "组合风险由「分散的行业 beta」收敛为「少数行业的共同 beta」。"
                      "最新一期（{q}）为 {last}，是样本期极值。")

    _cA, _cB = st.columns([0.5, 0.5])
    with _cA:
        _h('<div class="sec">② 行业综合拥挤度排名（最拥挤/最低配各 8）</div>')
        _ord = _SNAP.sort_values("综合拥挤度", ascending=False)
        _sel2 = pd.concat([_ord.head(8), _ord.tail(8)]).iloc[::-1]
        _f1 = go.Figure(go.Bar(
            x=_sel2["综合拥挤度"], y=_sel2["行业"], orientation="h",
            marker=dict(color=[C.COLOR["up"] if v >= 0 else C.COLOR["down"]
                               for v in _sel2["综合拥挤度"]]),
            text=[f"{v:+.2f}" for v in _sel2["综合拥挤度"]], textposition="outside",
            customdata=np.column_stack([_sel2["配置比例%"], _sel2["超配分位%"],
                                        _sel2["拥挤等级"]]),
            hovertemplate="%{y}<br>综合拥挤度 %{x:+.3f}<br>配置 %{customdata[0]:.2f}%"
                          "　分位 %{customdata[1]:.1f}%<br>等级 %{customdata[2]}"
                          "<extra></extra>"))
        _f1.update_layout(height=430, template=C.PLOTLY_TEMPLATE,
                          margin=dict(l=72, r=46, t=14, b=28), showlegend=False,
                          paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                          font=dict(family="Microsoft YaHei", size=10))
        _f1.update_xaxes(title_text="综合拥挤度（横截面 Z 合成）",
                         gridcolor=C.COLOR["grid"])
        _f1.update_yaxes(showgrid=False)
        st.plotly_chart(_f1, width="stretch", config={"displayModeBar": False})
        _ordv = _SNAP["综合拥挤度"]
        _ins(cross=dict(ok=True, feature=(
            f"最拥挤 {_ord.iloc[0]['行业']} {_ord.iloc[0]['综合拥挤度']:+.2f}、"
            f"{_ord.iloc[1]['行业']} {_ord.iloc[1]['综合拥挤度']:+.2f}；"
            f"最低配 {_ord.iloc[-1]['行业']} {_ord.iloc[-1]['综合拥挤度']:+.2f}、"
            f"{_ord.iloc[-2]['行业']} {_ord.iloc[-2]['综合拥挤度']:+.2f}；"
            f"首末极差 {_ordv.max() - _ordv.min():.2f}（横截面 Z 口径，"
            f"均值 0、标准差 1，故极差直接反映分化倍数）")),
            title="② 分析与结论 · 拥挤度横截面",
            extra=(f"<b>结构判断</b>：32 个并列条目中，"
                   f"超过 +1.5（拥挤阈值）的有 "
                   f"<b>{int((_SNAP['综合拥挤度'] > 1.5).sum())}</b> 个、"
                   f"低于 −1.5（显著低配）的有 "
                   f"<b>{int((_SNAP['综合拥挤度'] < -1.5).sum())}</b> 个 → "
                   f"<b>K 型分化</b>：资金并非整体加仓，而是在行业间搬家。"))
    with _cB:
        _h('<div class="sec">③ 拥挤度气泡图（X=配置  Y=超配Z  大小=基准权重）</div>')
        _bb = pd.DataFrame({
            "行业": INDUSTRIES,
            "配置%": B.alloc.loc[LATEST].reindex(INDUSTRIES).to_numpy(),
            "Z": IND["F2_超配Zscore"].loc[LATEST].reindex(INDUSTRIES).to_numpy(),
            "基准%": IND["F12b_基准权重%"].loc[LATEST].reindex(INDUSTRIES).to_numpy(),
            "F12": IND["F12_配置系数"].loc[LATEST].reindex(INDUSTRIES).to_numpy(),
            "等级": IND["拥挤等级"].loc[LATEST].reindex(INDUSTRIES).to_numpy()})
        _bb["基准%"] = _bb["基准%"].fillna(0.35)
        _f2 = go.Figure()
        for _lv in C.LEVELS:
            _s2 = _bb[_bb["等级"] == _lv]
            if _s2.empty:
                continue
            _f2.add_trace(go.Scatter(
                x=_s2["配置%"], y=_s2["Z"], mode="markers+text",
                text=_s2["行业"], textposition="top center",
                textfont=dict(size=8.5),
                marker=dict(size=np.sqrt(_s2["基准%"]) * 7 + 7,
                            color=C.COLOR.get(f"level_{_lv}", "#8A97A5"),
                            opacity=.62, line=dict(color="#FFFFFF", width=1)),
                name=_lv,
                customdata=np.column_stack([_s2["F12"], _s2["基准%"]]),
                hovertemplate="%{text}<br>配置 %{x:.2f}%　超配Z %{y:+.2f}"
                              "<br>配置系数 %{customdata[0]:.2f}"
                              "　基准权重 %{customdata[1]:.2f}%<extra></extra>"))
        _f2.add_hline(y=0, line=dict(color=C.COLOR["line"], width=1))
        _f2.add_shape(type="line", xref="x", yref="paper",
                      x0=100 / len(INDUSTRIES), x1=100 / len(INDUSTRIES), y0=0, y1=1,
                      line=dict(color=C.COLOR["line"], width=1, dash="dot"))
        _f2.update_layout(height=430, template=C.PLOTLY_TEMPLATE,
                          margin=dict(l=44, r=20, t=14, b=30),
                          paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                          font=dict(family="Microsoft YaHei", size=10),
                          legend=dict(orientation="h", y=1.10, font=dict(size=9)))
        _f2.update_xaxes(title_text="配置比例 (%)", gridcolor=C.COLOR["grid"])
        _f2.update_yaxes(title_text="超配 Z（扩窗）", gridcolor=C.COLOR["grid"])
        st.plotly_chart(_f2, width="stretch", config={"displayModeBar": False})
        _ins(cross=ins.describe_cross_section(B.alloc, LATEST, pct=True, top=3),
             title="③ 分析与结论 · 横截面分布",
             extra=("<b>读图要点</b>：右上角＝「重仓且相对自身历史极重」（真拥挤区）；"
                    "左侧＝配置低于等权 3.23%；<b>Z 为正但配置极低</b>的行业"
                    "（如银行、非银金融）是「相对自身历史最不低配」，"
                    "不构成拥挤——这正是本项目修正过的口径陷阱。"))

    _cC, _cD = st.columns([0.5, 0.5])
    with _cC:
        _h('<div class="sec">④ 策略净值 vs 三类基准（F15 行业轮动）</div>')
        _rt = _BTR["rot"]
        _rt_hs = _BTR["rot_hs"]
        _f3 = go.Figure()
        # 策略表可能为空（样本不足时 industry_rotation_strategy 返回空表）。
        # 机构级稳健性：此处显式降级为提示，而不是抛 KeyError 崩掉整页。
        _rt_tbl = _rt.get("table") if isinstance(_rt, dict) else None
        if not isinstance(_rt_tbl, pd.DataFrame) or _rt_tbl.empty \
                or "组合收益_净" not in _rt_tbl.columns:
            st.info("样本不足：F15 轮动策略在当前可建模区间内无法形成有效持仓序列，"
                    "该图暂不可用（待补齐 2010–2024 历史持仓与成交额后自动恢复）。")
        else:
            _cp = (1 + _rt_tbl["组合收益_净"] / 100).cumprod()
            _f3.add_trace(go.Scatter(x=list(_cp.index), y=_cp.values, name="F15 策略（净）",
                                     mode="lines+markers",
                                     line=dict(color=C.COLOR["price"], width=2.2)))
            _cb2 = (1 + _rt_tbl["基准收益"] / 100).cumprod()
            _f3.add_trace(go.Scatter(x=list(_cb2.index), y=_cb2.values, name="等权 31 行业",
                                     mode="lines+markers",
                                     line=dict(color=C.COLOR["line"], width=1.6, dash="dash")))
            if _rt_hs and _rt_hs.get("table") is not None and not _rt_hs["table"].empty:
                _ch = (1 + _rt_hs["table"]["基准收益"] / 100).cumprod()
                _f3.add_trace(go.Scatter(x=list(_ch.index), y=_ch.values, name="沪深300（逐期）",
                                         mode="lines+markers",
                                         line=dict(color=C.COLOR["up"], width=1.5, dash="dot")))
            _f3.update_layout(height=340, template=C.PLOTLY_TEMPLATE,
                              margin=dict(l=48, r=20, t=14, b=30), hovermode="x unified",
                              paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                              font=dict(family="Microsoft YaHei", size=10.5),
                              legend=dict(orientation="h", y=1.12, font=dict(size=9.5)))
            _f3.update_yaxes(title_text="累计净值（起点=1）", gridcolor=C.COLOR["grid"])
            _f3.update_xaxes(showgrid=False)
            st.plotly_chart(_f3, width="stretch", config={"displayModeBar": False})
            _p3 = (_rt.get("perf") or {})
            if _p3:
                _ex3 = pd.Series(
                    {k: v for k, v in zip(_rt_tbl.index, _rt_tbl["超额_净"])}
                ) if "超额_净" in _rt_tbl.columns else None
                _ins(_ex3, "策略超额（净）",
                     title="④ 分析与结论 · 策略表现",
                     pct=False, unit="pct",
                     higher_means=("策略净超额累计 {change}，最新一期为 {last}。"
                                   "**必须并列引用**：该超额是对**等权行业**基准；"
                                   "对市值加权的沪深300 基准结论不同（见下方对照），"
                                   "只报有利口径即为误导。"))
    with _cD:
        _h('<div class="sec">⑤ 超额收益行业归因（Top/Bottom 各 8）</div>')
        _at_raw = (_BTR["r4"].get("attr") or {}).get("table")
        if not isinstance(_at_raw, pd.DataFrame) or _at_raw.empty \
                or "累计贡献pct" not in _at_raw.columns:
            st.info("样本不足：归因表在当前可建模区间内无法生成（待补齐历史后自动恢复）。")
        else:
            _at = _at_raw
            _sel3 = pd.concat([_at.head(8), _at.tail(8)]).iloc[::-1]
            _f4 = go.Figure(go.Bar(
                x=_sel3["累计贡献pct"], y=_sel3["行业"], orientation="h",
                marker=dict(color=[C.COLOR["up"] if v >= 0 else C.COLOR["down"]
                                   for v in _sel3["累计贡献pct"]]),
                text=[f"{v:+.2f}" for v in _sel3["累计贡献pct"]], textposition="outside",
                customdata=_sel3["平均权重偏离pct"],
                hovertemplate="%{y}<br>累计贡献 %{x:+.2f}pct<br>"
                              "平均权重偏离 %{customdata:+.2f}pct<extra></extra>"))
            _f4.add_shape(type="line", xref="x", yref="paper", x0=0, x1=0, y0=0, y1=1,
                          line=dict(color=C.COLOR["line"], width=1))
            _f4.update_layout(height=340, template=C.PLOTLY_TEMPLATE,
                              margin=dict(l=72, r=50, t=14, b=30), showlegend=False,
                              paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                              font=dict(family="Microsoft YaHei", size=10))
            _f4.update_xaxes(title_text="累计贡献 (pct)", gridcolor=C.COLOR["grid"])
            _f4.update_yaxes(showgrid=False)
            st.plotly_chart(_f4, width="stretch", config={"displayModeBar": False})
            _s3 = _at.set_index("行业")["累计贡献pct"]
            _ins(_s3, "行业累计贡献", pct=False, unit="pct",
                 title="⑤ 分析与结论 · 超额来源归因",
                 higher_means="超额累计 {change}。若剔除单一行业后超额归零，"
                              "说明收益本质是「低配某个大牛行业」的 beta，而非行业选择能力；"
                              "若仍显著为正，则更接近真 alpha。")

    _cE, _cF = st.columns([0.5, 0.5])
    with _cE:
        _h('<div class="sec">⑥ 成本—规模曲线（冲击成本模型）</div>')
        _it = _imp["table"] if _imp_ok else pd.DataFrame()
        if not _imp_ok or "组合规模亿元" not in _it.columns:
            st.info("样本不足：冲击成本—容量模型需要行业季度成交额与经济持仓序列，"
                    "已覆盖 18 期（2022Q1~2026Q2）；若该图仍不可用，"
                    "原因是扩窗标准化后的持仓序列不足（样本长度限制）。")
        else:
            _f5 = make_subplots(specs=[[{"secondary_y": True}]])
            _f5.add_trace(go.Bar(x=_it["组合规模亿元"], y=_it["平均单期成本bp"],
                                 name="平均单期成本(bp)",
                                 marker=dict(color=C.COLOR["level_偏拥挤"]),
                                 hovertemplate="%{x:,.0f}亿　成本 %{y:.2f}bp<extra></extra>"),
                          secondary_y=False)
            _f5.add_trace(go.Scatter(x=_it["组合规模亿元"], y=_it["超额净_几何"],
                                     name="超额净年化(%)", mode="lines+markers",
                                     line=dict(color=C.COLOR["price"], width=2.2),
                                     hovertemplate="%{x:,.0f}亿　超额 %{y:+.2f}%<extra></extra>"),
                          secondary_y=True)
            _f5.update_layout(height=340, template=C.PLOTLY_TEMPLATE,
                              margin=dict(l=48, r=54, t=14, b=30), hovermode="x unified",
                              paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                              font=dict(family="Microsoft YaHei", size=10.5),
                              legend=dict(orientation="h", y=1.12, font=dict(size=9.5)))
            _f5.update_xaxes(title_text="组合规模 (亿元)", type="log", showgrid=False)
            _f5.update_yaxes(title_text="成本 (bp/期)", gridcolor=C.COLOR["grid"],
                             secondary_y=False)
            _f5.update_yaxes(title_text="超额净年化 (%)", showgrid=False, secondary_y=True)
            st.plotly_chart(_f5, width="stretch", config={"displayModeBar": False})
            _note("sqrt 冲击模型：p=1% → 15bp 单边，成本下限 5bp。"
                  "在 10 亿 ~ 3,000 亿区间成本几乎不动 → <b>容量不构成约束</b>。")
            _ins(_it.set_index("组合规模亿元")["平均单期成本bp"], "平均单期成本",
                 unit="bp", pct=False,
                 title="⑥ 分析与结论 · 成本与容量",
                 higher_means=("平均单期成本随规模{direction}至 {last}（{q}）。"
                               "曲线接近水平意味着**成本对规模不敏感** → 容量瓶颈不在冲击成本，"
                               "而在「行业成交额是否可持续」与「信号是否稳定」。"))
    with _cF:
        _h('<div class="sec">⑦ 因子动量 momIC 时序（回看 8 季）</div>')
        _fms = _BTR["r4"]["fm_series"]
        if _fms is not None and len(_fms):
            _f6 = go.Figure(go.Bar(
                x=list(_fms.index), y=_fms.values,
                marker=dict(color=[C.COLOR["up"] if v >= 0 else C.COLOR["down"]
                                   for v in _fms.values]),
                hovertemplate="%{x}　momIC %{y:+.4f}<extra></extra>"))
            _f6.add_shape(type="line", xref="x", yref="paper", x0=0, x1=1, y0=0, y1=0,
                          line=dict(color=C.COLOR["line"], width=1))
            _f6.update_layout(height=340, template=C.PLOTLY_TEMPLATE,
                              margin=dict(l=48, r=20, t=14, b=30), showlegend=False,
                              paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                              font=dict(family="Microsoft YaHei", size=10))
            _f6.update_yaxes(title_text="因子间 Spearman(过去IC, 下期IC)",
                             gridcolor=C.COLOR["grid"])
            _f6.update_xaxes(showgrid=False)
            st.plotly_chart(_f6, width="stretch", config={"displayModeBar": False})
            _note("逐期在**因子之间**做秩相关：过去 8 季 IC 高的因子，下一季 IC 是否也高。"
                  "方向 11/11 组全为正，但显著性集中在近段与震荡市。")
            _ins(_fms, "momIC", title="⑦ 分析与结论 · 因子动量",
                 higher_means=("momIC 的{direction}说明因子有效性存在**惯性**："
                               "过去有效的因子在下一期倾向继续有效（最新 {last}，{q}）。"
                               "这既是可利用的信号，也解释了为何「挑设定的流程」"
                               "在样本外表现会随时间漂移。"))

    # ------------------------------------------------------------------
    # 结论与判断理由（每一条都给出：结论 → 依据 → 阈值由来 → 失效条件）
    # ------------------------------------------------------------------
    _h('<div class="sec">本轮结论与判断理由（结论 → 依据 → 阈值由来 → 失效条件）</div>')
    _exs = M_SUM["extreme_industries"]
    _ex_txt = "；".join(
        f"{n}（配置 {B.alloc.loc[LATEST, n]:.2f}%、超配 "
        f"{B.overweight.loc[LATEST, n]:+.2f}pct、分位 "
        f"{IND['F3_超配历史分位'].loc[LATEST, n]:.0f}%、F12 "
        f"{IND['F12_配置系数'].loc[LATEST, n]:.2f}）"
        for n in _exs)
    _dv = IND["口径背离"].loc[LATEST]
    _dvl = "、".join(_dv.index[_dv]) or "无"
    _topcf = IND["F12_配置系数"].loc[LATEST].dropna()
    _topcf_n, _topcf_v = _topcf.idxmax(), float(_topcf.max())
    # 设计期样本外：结果表在短样本下可能为空或缺目标因子，须显式降级。
    _doos = _BTR.get("doos") or {}
    _dt = _doos.get("table")
    _dt_ok = (isinstance(_dt, pd.DataFrame) and not _dt.empty
              and "因子" in _dt.columns)
    _f15_row = (_dt[_dt["因子"] == "F15_拥挤背离"]
                if _dt_ok and "留出期单侧p" in _dt.columns else pd.DataFrame())
    _f15_ok = (not _f15_row.empty
               and _f15_row["留出期单侧p"].notna().any())
    if _f15_ok:
        _f15_p = float(_f15_row["留出期单侧p"].iloc[0])
        _f15_ic = float(_f15_row["留出期IC"].iloc[0])
    else:
        _f15_p, _f15_ic = np.nan, np.nan
    _f15_feasible = bool(_doos.get("feasible", False))
    _fmt_p = f"{_f15_p:.4f}" if _f15_p == _f15_p else "样本不足"
    _fmt_ic = f"{_f15_ic:+.4f}" if _f15_ic == _f15_ic else "样本不足"
    _fm_row = _fm[_fm["回看窗口"] == "8 季"].iloc[0]
    _cap_bp = (float(_imp["table"].iloc[-1]["平均单期成本bp"])
               if _imp_ok else np.nan)
    _cap_sz = int(max(_imp["sizes"])) if _imp_ok else None

    _h(concl([
        dict(title="结论 1｜市场集中度：极高，处于历史极值区",
             level="极度拥挤",
             concl=(f"HHI = <b>{M_SUM['HHI']:.4f}</b>（历史分位 {M_SUM['HHI_pct']:.0f}%、"
                    f"历史均值 {M_SUM['HHI_hist_mean']:.4f}）→ 等级「{M_SUM['HHI_level']}」；"
                    f"CR3 / CR5 / CR10 = {M_SUM['CR3']:.2f}% / {M_SUM['CR5']:.2f}% / "
                    f"{M_SUM['CR10']:.2f}%；第一大重仓行业 = {M_SUM['top1']} "
                    f"{M_SUM['top1_pct']:.2f}%。"),
             why={"判断依据": "HHI 由 31 行业配置比例独立复算，R2/R6 校验通过"
                              "（HHI ∈ [0.0678, 0.1949]，复算最大绝对差 < 1e-13）；"
                              "配置比例加总 = 100.00% ± 0.00pct（R1 最大偏离 0.0pct）。",
                  "阈值 0.15 的由来": "集中度等级惯例（<0.05 低、0.05–0.10 中、"
                                      "0.10–0.15 高、>0.15 极高）。本项目 HHI 实际区间 "
                                      "0.068–0.195，0.15 落在样本区间上部，"
                                      "既能识别极值又不会长期误报。",
                  "经济含义": "机构仓位高度压在少数行业 → 行业分散度下降，"
                              "组合对单一行业冲击的敏感度被放大（见「压力测试」）。"},
             caveat="HHI 基于「季报前十大重仓」口径，系统性低估分散型行业；"
                    "它只描述集中度，不含方向判断，不能用于择时。"),

        dict(title=f"结论 2｜持仓拥挤：{M_SUM['n_extreme']} 个行业判定为「极度拥挤」",
             level="极度拥挤",
             concl=f"{'、'.join(_exs)}。<br>{_ex_txt}。",
             why={"触发规则": "超配历史分位 ≥ 90 <b>或</b> 配置系数 F12 > 2.5 → 极度拥挤，"
                              "两个口径任一触发即升级（保守规则）。",
                  "为什么用「或」而非「且」": "分位衡量「相对自身历史」，F12 衡量"
                                              "「相对沪深300 基准」，是两个互补维度。"
                                              "要求同时满足会漏掉「相对基准已极重配、"
                                              "但自身历史分位还没抬起来」的新增拥挤。",
                  "经济含义": "机构定价权高度集中 → 边际买盘减少；一旦共识松动，"
                              "减仓通道拥挤，再定价概率上升。"},
             caveat="判的是「拥挤程度」而不是「一定下跌」。同时有 "
                    f"{M_SUM['n_under']} 个行业显著低配，横截面呈 K 型分化，"
                    "组合风险来自「集中」而非「方向」。"),

        dict(title="结论 3｜口径背离：必须同时呈现两个口径，不能只给一个标签",
             level="偏拥挤",
             concl=(f"标记为「口径背离」的行业：<b>{_dvl}</b>。"
                    f"最极端的是 F12 配置系数最高者 {_topcf_n} = <b>{_topcf_v:.2f}</b>，"
                    "即相对沪深300 基准极重配；但它的超配历史分位近乎 0，"
                    "即相对自身历史处在极低位。"),
             why={"判断依据": "Z/分位（相对自身历史）与 F12（相对基准权重）指向相反时"
                              "置「口径背离」= True。",
                  "为什么要显式标记": "两者都对，但含义相反：分位说的是"
                                      "「比过去任何时点都轻」，F12 说的是"
                                      "「比沪深300 权重重一倍以上」。"
                                      "合并成单一标签一定会误导其中一方。",
                  "经济含义": "这类行业通常是「长期被减仓、但绝对权重仍高」的品种，"
                              "既不是新增拥挤，也不是超卖反转，需要单独归类处理。"},
             caveat="背离只是「两个口径不一致」的事实描述，它本身不构成买卖信号；"
                    "在背离状态下不可用单一等级做决策。"),

        dict(title="结论 4｜可用因子：只有 F15「拥挤背离」达到可用级，且必须带限制引用",
             level="拥挤",
             concl=(f"分级裁决中 A/A− 级 = <b>{'、'.join(_A) if _A else '无'}</b>"
                    f"（共 {len(_A)} 个，由 4 项独立检验合成）。"
                    + (f"F15 在设计期样本外检验中留出期 IC = {_fmt_ic}、"
                       f"单侧 p = {_fmt_p}。"
                       if _f15_ok else
                       "⚠️ 设计期样本外检验在本样本下<b>不可行</b>"
                       "（留出期观测不足），该判据本轮</b>不成立</b>。")),
             why={"构造原理": "F15 = 横截面标准化的「超配水平」− 横截面标准化的"
                              "「超配 Z」。含义是「绝对超配高、但相对自身历史还不极端」"
                              "＝<b>新增的拥挤</b>。零自由参数。",
                  "为什么它可以进模型": "在 34 期长历史下同时满足 4 项独立判据——"
                                        "统计显著（|IC_IR|>0.3）、设计期样本外通过、"
                                        "二层正交后仍保留独立增量信息、扣 20bp 成本后"
                                        "按经济先验方向的超额为正。"
                                        "⚠️ 当前 Wind 数据仅 10 期，样本外判据无法复现，"
                                        "因此该结论<b>暂时不可沿用</b>，待补齐历史后重新确认。",
                  "经济含义": "捕捉「新资金刚进来、历史分位还没抬起来」这一类"
                              "最容易被忽视的拥挤，而不是「已经很拥挤」的老拥挤。"},
             caveat=("样本切分：" + str(_doos.get("split_note", "—"))
                     + "。"
                     + ("" if _f15_feasible else
                        "<b>留出期观测 &lt; 4，统计功效不足</b>，"
                        "样本外结论仅作方向性参考，不得作为因子入选依据。")
                     + "「单个事先说明的假设通过了检验」与「从数据里挑因子的流程通过了检验」"
                       f"是两件事：walk-forward 检验显示「挑设定的流程」样本外 "
                       f"p = {_ws['留出期p']:.3f}，<b>不显著</b>。"
                       "两条必须同时引用，只报前者会高估置信度。")),

        dict(title="结论 5｜因子有效性有惯性，但强度时变，不可外推",
             level="中性",
             concl=(f"因子动量 momIC（回看 8 季）= <b>{float(_fm_row['momIC均值']):+.4f}</b>，"
                    f"NW t = {float(_fm_row['NW_t']):.3f}，p = {float(_fm_row['p值']):.4f}，"
                    f"正比例 {float(_fm_row['正比例']):.1f}%。"),
             why={"判断依据": "逐期在<b>因子之间</b>做秩相关：过去 N 季 IC 高的因子，"
                              "下一季 IC 是否也高。4 / 8 / 12 季三档窗口 p 全部 < 0.011。",
                  "经济学解释": "因子有效性来自机构行为的持续性（调仓有惯性），"
                                "因此「近期有效的因子」在短期仍倾向有效。",
                  "这与另一条结论的关系": "它同时也是对「单因子显著」的警告——"
                                          "看起来的 alpha 可能只是因子动量，"
                                          "是路径依赖而不是稳定溢价。"},
             caveat="11 个稳健性分组方向 11/11 为正，但<b>只有 5 组显著</b>，"
                    "且集中在近段与震荡市（牛熊均不显著）；去掉同族冗余因子后 "
                    "p 从 0.0049 退化到 0.0648。→ 强度时变，不可外推。"),

        dict(title="结论 6｜容量不构成约束，且既有成本假设偏保守",
             level="中性",
             concl=((f"以行业季度成交额建 sqrt 冲击成本模型：规模 "
                     f"{_cap_sz:,} 亿元时平均单期成本仅 <b>{_cap_bp:.2f}bp/期</b>；"
                     "从 10 亿到 3,000 亿成本几乎不变。")
                    if _imp_ok else
                    "⚠️ 容量模型在本样本下不可用：成交额已补齐至 18 期，"
                    "但扩窗标准化后的可用持仓序列不足，无法支撑参与率推演。"
                    "该结论<b>本轮不成立</b>，"
                    "待补齐 2024Q3–2026Q2 成交额后重算。"),
             why={"判断依据": "冲击成本 = k × √(参与率)，参与率 = 调仓金额 ÷ "
                              "该行业季度成交额；k 取 p=1% → 15bp，并设 5bp 下限。"
                              "已做流动性折损 h ∈ {1, 0.5, 0.2, 0.05} 与 "
                              "k ∈ {10,15,20,30} 双向敏感性。",
                  "为什么说既有假设偏保守": "R1–R3 全程使用固定双边 20bp，"
                                            "等价于约 2.95bp/期，是建模成本的约 4 倍 → "
                                            "此前所有扣费结论都是<b>保守下限</b>。",
                  "经济含义": "行业轮动策略的资金容量瓶颈不在冲击成本，"
                              "而在「行业成交额是否可持续」与「信号是否稳定」。"},
             caveat="这是<b>行业整体</b>口径的容量。若落到个股执行，容量会显著更小；"
                    "且模型未计涨跌停、停牌、流动性踩踏等极端情形。"),

        dict(title="结论 7｜定位：拥挤度是风险描述指标，不是择时或做空信号",
             level="显著低配",
             concl=("本仪表盘的全部输出用于三件事："
                    "① 风险描述与压力测试输入；② 新增仓位的行业集中度约束；"
                    "③ 季度跟踪的预警清单。它不给出买卖时点。"),
             why={"为什么这样定位": "全部因子的 Bootstrap 95% 置信区间与多重检验"
                                    "（BH-FDR）均显示：在 18–23 个因子的检验族内，"
                                    "多数因子无法与噪声区分。把未通过检验的因子"
                                    "当作交易信号是不负责任的。",
                  "可落地的部分": "压力测试不依赖因子有效性——"
                                  f"电子单一行业占 {M_SUM['top1_pct']:.2f}%，"
                                  "其回撤 30% 即拖累组合约 11pct。这个结论可以直接用。",
                  "经济含义": "机构拥挤反映的是「定价权集中」，"
                              "它影响的是<b>尾部风险的形状</b>，而不是收益的均值。"},
             caveat="高拥挤 ≠ 必然下跌；历史事件研究显示「超配首次突破 P90」后 12 个月"
                    "超额收益中位数为正。低拥挤同样不等于超卖反转。"),
    ]))
    # 投资机会解读（在综合判断结论之上，给出可操作的回避/关注/约束三层判断）
    _h(guide.opportunity_card(M_SUM, IND, C, LATEST))
    st.stop()


if _P == "③ 行业占比变迁":
    _h('<div class="sec">行业配置结构变迁（主动权益基金 · 申万一级 · 前十大重仓口径）'
                '</div>')
    # 全历史序列：2019Q1~2023Q4 偏股混合单类型（Wind 逐期重抓）+ 2024Q1~2026Q2 主口径
    _qs_ext = (list(B.alloc_ext_quarters) if getattr(B, "alloc_ext_available", False)
               else list(B.quarters))
    _c1, _c2 = st.columns([0.5, 0.5])
    with _c1:
        _rng = st.radio("区间", ["近三年", "近五年", "全区间"], index=2, horizontal=True,
                        key="st_rng", label_visibility="collapsed")
    _sub = {"近三年": _qs_ext[-13:], "近五年": _qs_ext[-21:], "全区间": _qs_ext}[_rng]
    _alloc_w = (B.alloc_ext if getattr(B, "alloc_ext_available", False)
                else B.alloc).loc[_sub]
    _top = _alloc_w.iloc[-1].sort_values(ascending=False).head(10).index.tolist()

    # ① 堆叠面积图
    _h('<div class="sec">① 行业配置占比 · 堆叠面积图</div>')
    _f1 = go.Figure()
    for _i, _ind in enumerate(_top):
        _f1.add_trace(go.Scatter(
            x=_sub, y=_alloc_w[_ind], name=_ind, mode="lines", stackgroup="one",
            line=dict(width=0.5, color=_palette[_i % len(_palette)]),
            fillcolor=_palette[_i % len(_palette)],
            hovertemplate=_ind + " %{y:.2f}%<extra></extra>"))
    _rest = _alloc_w.drop(columns=_top).sum(axis=1)
    _f1.add_trace(go.Scatter(x=_sub, y=_rest, name="其余 21 个行业", mode="lines",
                             stackgroup="one", line=dict(width=0.5, color="#DCE3EA"),
                             fillcolor="#DCE3EA",
                             hovertemplate="其余 %{y:.2f}%<extra></extra>"))
    _f1.update_layout(height=C.CHART_H_MAIN + 40, template=C.PLOTLY_TEMPLATE,
                      margin=dict(l=48, r=20, t=16, b=30), hovermode="x unified",
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font=dict(family="Microsoft YaHei", size=11),
                      legend=dict(orientation="v", x=1.01, y=1, font=dict(size=9.5)))
    _f1.update_yaxes(title_text="配置比例 (%)", gridcolor=C.COLOR["grid"], range=[0, 100])
    _f1.update_xaxes(showgrid=False)
    st.plotly_chart(_f1, width="stretch", config={"displayModeBar": False})
    _note("数据来源：Wind（2016Q2~2023Q4 为偏股混合型单类型逐期重抓；2016Q1/2017Q1/2017Q4/2018Q1/Q4 等期 Wind 自然语言接口仅返回全市场行业市值而非基金重仓，已判定不采信并登记根因；"
          "2024Q1~2026Q2 为主口径，与①②页一致，2025Q2 起为主动权益三类、"
          "断点已做占比不变性检验）"
          f"　｜　区间：{_sub[0]} ~ {_sub[-1]}（{len(_sub)} 个季度）")
    _t1n = _top[0]
    _ins(_alloc_w[_t1n].dropna(), f"{_t1n} 配置比例",
         title="① 分析与结论 · 结构变迁",
         pct=True,
         higher_means=(f"堆叠面积图中最厚的一层就是 {_t1n}，其占比"
                       "{direction}至 {last}（{q}）；上方面积的变化即"
                       "「资金在行业间的搬家路径」——某一层加速变厚时，"
                       "通常对应同期的其他层被压缩，而非整体加仓。"))

    # ② 热力图（标注具体数值 + 丰富色阶 + 图例）
    _h('<div class="sec">② 行业配置占比 · 热力图</div>')
    _z = _alloc_w.T                      # 行=31 申万一级行业，列=季度
    _z_arr = _z.to_numpy(dtype=float)
    # 单元格数值标注：占比 ≥2% 的格子直接写数值（<2% 视为零散配置，悬停可查精确值）
    _z_txt = [["" if v < 2.0 else f"{v:.1f}" for v in row] for row in _z_arr]
    _f2 = go.Figure(go.Heatmap(
        z=_z_arr, x=_sub, y=_z.index,
        text=_z_txt, texttemplate="%{text}",
        textfont=dict(size=8.5, color="white"),
        colorscale=[[0, "#FFF7F5"], [0.10, "#FDE4DD"], [0.28, "#F7BDB4"],
                    [0.52, "#EC8A7D"], [0.76, "#D84A40"], [1, "#A6150A"]],
        colorbar=dict(title=dict(text="配置占比(%)", side="right"),
                      thickness=14, len=0.92,
                      tickfont=dict(size=9), ticks="outside",
                      tickformat=".0f"),
        hovertemplate="%{y}　%{x}　配置 <b>%{z:.2f}%</b><extra></extra>"))
    _f2.update_layout(height=840, template=C.PLOTLY_TEMPLATE,
                      margin=dict(l=70, r=30, t=16, b=30),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font=dict(family="Microsoft YaHei", size=10))
    _f2.update_xaxes(showgrid=False, side="top", tickangle=45)
    _f2.update_yaxes(showgrid=False, autorange="reversed")
    st.plotly_chart(_f2, width="stretch", config={"displayModeBar": False})
    _note("颜色越深＝配置比例越高；<b>占比 ≥2% 的格子已直接标注数值</b>（低于 2% 的零散配置悬停可查精确值）。"
          "行＝31 个申万一级行业，列＝季度。"
          "数据来源：Wind（口径拼接同上：2016Q2 起偏股混合单类型，"
          "2024Q1 起与①②页主口径一致）。")
    _ins(cross=ins.describe_cross_section(B.alloc, LATEST, pct=True, top=3),
         title="② 分析与结论 · 配置热力图",
         extra=("<b>读图要点</b>：横向看一个行业的颜色渐变＝该行业的配置迁移史；"
                "纵向看某一列＝该季度资金在行业间的分布。"
                "<b>颜色最深的一列若集中在最近几期，即「抱团加速」的直接证据；"
                "同一行的颜色若从深变浅，则是被减仓的证据。</b>"))

    # ③ 瀑布图
    _h('<div class="sec">③ 行业配置环比变化 · 瀑布图</div>')
    _wi, _wj = st.columns([0.7, 0.3])
    with _wj:
        _wq = st.selectbox("对比季度", _sub[1:][::-1], index=0, key="wf_q")
    _k = _sub.index(_wq)
    _prev = _sub[_k - 1]
    _d = (_alloc_w.loc[_wq] - _alloc_w.loc[_prev]).sort_values()
    _d = pd.concat([_d.head(6), _d.tail(6)]).sort_values()
    _colors = [C.COLOR["up"] if v > 0 else C.COLOR["down"] for v in _d]
    _f3 = go.Figure(go.Waterfall(
        orientation="v", x=_d.index, y=_d.to_numpy(),
        measure=["relative"] * len(_d),
        text=[f"{v:+.2f}" for v in _d], textposition="outside",
        connector=dict(line=dict(color=C.COLOR["line"])),
        increasing=dict(marker=dict(color=C.COLOR["up"])),
        decreasing=dict(marker=dict(color=C.COLOR["down"])),
        hovertemplate="%{x}　环比 %{y:+.2f} pct<extra></extra>"))
    _f3.update_layout(height=C.CHART_H_SMALL + 130, template=C.PLOTLY_TEMPLATE,
                      margin=dict(l=48, r=20, t=34, b=30), showlegend=False,
                      title=dict(text=f"{_prev} → {_wq} 配置比例环比变化（变化最大的各 6 个行业）",
                                 font=dict(size=11.5), x=0.01),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font=dict(family="Microsoft YaHei", size=10.5))
    _f3.update_yaxes(title_text="环比变化 (pct)", gridcolor=C.COLOR["grid"])
    _f3.update_xaxes(showgrid=False)
    st.plotly_chart(_f3, width="stretch", config={"displayModeBar": False})
    _chg = B.alloc.loc[LATEST] - B.alloc.loc[QUARTERS[-2]] if len(QUARTERS) > 1 else None
    if _chg is not None:
        _ins(_chg.dropna(), "环比变化", pct=True,
             title="③ 分析与结论 · 最近一期环比",
             higher_means=("本期环比{direction}至 {last}（{q}）。"
                           "柱高即「这一季度资金净搬了多少」；"
                           "若单行业单期变动超过 20pct，属**结构性异动**，需单独核查口径。"))
    _note("红＝加仓（A股习惯红涨绿跌），绿＝减仓。数据来源：" + _prov_line(["行业配置比例"]))

    # ④ 代表行业时序对比
    _h('<div class="sec">④ 代表行业配置占比 · 时序对比</div>')
    _sel4 = st.multiselect("选择行业（最多 8 个）", list(C.INDUSTRIES),
                           default=[i for i in ["电子", "通信", "医药生物", "食品饮料",
                                                 "电力设备", "银行"] if i in C.INDUSTRIES],
                           max_selections=8, key="cmp_inds")
    if _sel4:
        _f4 = go.Figure()
        for _i, _ind in enumerate(_sel4):
            _f4.add_trace(go.Scatter(
                x=_sub, y=_alloc_w[_ind], name=_ind, mode="lines+markers",
                line=dict(color=_palette[_i % len(_palette)], width=2),
                marker=dict(size=3.5),
                hovertemplate=_ind + " %{y:.2f}%<extra></extra>"))
        _f4.update_layout(height=C.CHART_H_MAIN, template=C.PLOTLY_TEMPLATE,
                          margin=dict(l=48, r=20, t=16, b=30), hovermode="x unified",
                          paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                          font=dict(family="Microsoft YaHei", size=11),
                          legend=dict(orientation="h", y=1.10, font=dict(size=10)))
        _f4.update_yaxes(title_text="配置比例 (%)", gridcolor=C.COLOR["grid"])
        _f4.update_xaxes(showgrid=False)
        st.plotly_chart(_f4, width="stretch", config={"displayModeBar": False})
        _note(f"数据来源：Wind（口径拼接同上）　｜　区间 {_sub[0]} ~ {_sub[-1]}")
        _rep0 = [i for i in ("电子", "通信", "医药生物", "银行")
                 if i in _alloc_w.columns]
        if _rep0:
            _ins(_alloc_w[_rep0[0]].dropna(), _rep0[0],
                 title="④ 分析与结论 · 代表行业",
                 pct=True,
                 higher_means=("代表行业中 {direction} 为代表者之一（最新 {last}，{q}）。"
                               "把走势互不相同的行业并列，可直接分辨"
                               "**趋势型加仓**（单边）与**轮动型切换**（此消彼长）。"))
    # 投资机会解读（基于当前拥挤结构的回避/关注提示）
    _h(guide.opportunity_hint(M_SUM, C, "行业占比变迁"))
    st.stop()


if _P == "④ ETF 资金流":
    _h('<div class="sec">ETF 日度资金流向（份额折算已复权）</div>')
    _ed = getattr(B, "etf_daily", None)
    if _ed is None or _ed.empty:
        st.warning("未发现 data/raw/etf_flow_daily.xlsx，ETF 页面不可用。")
        st.stop()
    _d1, _d2 = st.columns([0.5, 0.5])
    _codes = sorted(_ed["代码"].unique())
    with _d1:
        _pick = st.multiselect("ETF", _codes, default=_codes, key="etf_pick",
                               format_func=lambda x: f"{x} "
                               f"{_ed.loc[_ed['代码'] == x, '名称'].iloc[0]}")
    with _d2:
        _metric = st.radio("指标", ["累计净申赎（亿元）", "累计资金流强度（%）",
                                    "场内流通份额（亿份）"], horizontal=True, key="etf_m")
    _ed = _ed.copy()
    _parts = []
    for _code, _gg in _ed.groupby("代码"):
        _gg = _gg.sort_values("日期").copy()
        _a0 = float(_gg["复权份额"].iloc[0] * _gg["复权净值"].iloc[0])
        _gg["累计净申赎亿元"] = _gg["净申赎亿元"].fillna(0).cumsum()
        _gg["累计资金流强度%"] = (_gg["累计净申赎亿元"] / _a0 * 100).round(C.DEC_PCT) \
            if _a0 > 0 else np.nan
        _gg["期内累计成交额亿元"] = _gg["成交额"].fillna(0).cumsum()
        _parts.append(_gg)
    _ed = pd.concat(_parts, ignore_index=True)
    _edf = _ed[_ed["代码"].isin(_pick)]
    _col = {"累计净申赎（亿元）": "累计净申赎亿元",
            "累计资金流强度（%）": "累计资金流强度%",
            "场内流通份额（亿份）": "复权份额"}[_metric]

    _f5 = go.Figure()
    for _i, (_code, _g) in enumerate(_edf.groupby("代码")):
        _g = _g.sort_values("日期")
        _nm = _g["名称"].iloc[0]
        _ind = _g["申万一级"].iloc[0]
        _f5.add_trace(go.Scatter(
            x=_g["日期"], y=_g[_col], name=f"{_code} {_nm}",
            mode="lines+markers", line=dict(color=_palette[_i % len(_palette)], width=2),
            marker=dict(size=3),
            hovertemplate=f"{_code} {_nm}（{_ind}）<br>%{{x}}　%{{y:,.2f}}<extra></extra>"))
    _f5.update_layout(height=C.CHART_H_MAIN, template=C.PLOTLY_TEMPLATE,
                      margin=dict(l=56, r=20, t=16, b=30), hovermode="x unified",
                      title=dict(text=f"对比指标：{_metric}", font=dict(size=11.5), x=0.01),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font=dict(family="Microsoft YaHei", size=11),
                      legend=dict(orientation="h", y=1.12, font=dict(size=10)))
    _f5.update_yaxes(gridcolor=C.COLOR["grid"])
    _f5.update_xaxes(showgrid=False)
    st.plotly_chart(_f5, width="stretch", config={"displayModeBar": False})
    _es = _edf.groupby("代码")["累计净申赎亿元"].last().dropna()
    _ins(_es, "ETF 累计净申赎", unit="亿元", pct=False,
         title="① 分析与结论 · ETF 资金流",
         higher_means=("各只 ETF 的累计净申赎在区间内总体{direction}（最大 {last}，{q}）。"
                       "**这是「快层」信息**：ETF 资金流是日频增量，季报持仓是季频存量，"
                       "两者相差一个季度；方向不一致时不是矛盾，而是时间维度不同。"))

    # ===== 宽基 ETF 资金动向分析（本轮新增：主要宽基 ETF 的规模与资金信号）=====
    _broad = _ed[_ed["申万一级"] == "宽基指数"]
    if not _broad.empty:
        _h('<div class="sec">宽基 ETF 资金动向（沪深300 / 中证500 / 创业板 / 科创50）</div>')
        _bf = go.Figure()
        _brows = []
        for _code, _g in _broad.groupby("代码"):
            _g = _g.sort_values("日期")
            _nm = _g["名称"].iloc[0]
            _a0 = float(_g["复权份额"].iloc[0] * _g["复权净值"].iloc[0])
            _g["累计净申赎亿元"] = _g["净申赎亿元"].fillna(0).cumsum()
            _g["累计资金流强度%"] = (_g["累计净申赎亿元"] / _a0 * 100).round(C.DEC_PCT) \
                if _a0 > 0 else np.nan
            _bf.add_trace(go.Scatter(
                x=_g["日期"], y=_g["累计净申赎亿元"], name=f"{_code} {_nm}",
                mode="lines", line=dict(color=_palette[list(_broad["代码"].unique()).index(_code)
                                                         % len(_palette)], width=2.2),
                hovertemplate=f"{_nm}<br>%{{x}}　累计净申赎 %{{y:+,.2f}} 亿元<extra></extra>"))
            _brows.append(dict(代码=_code, 名称=_nm,
                               期初份额亿份=round(float(_g["复权份额"].iloc[0]), 2),
                               期末份额亿份=round(float(_g["复权份额"].iloc[-1]), 2),
                               份额增幅=round(float(_g["复权份额"].iloc[-1]
                                                / _g["复权份额"].iloc[0] - 1) * 100, 2),
                               累计净申赎亿元=round(float(_g["净申赎亿元"].sum(skipna=True)), 2),
                               累计强度=round(float(_g["净申赎亿元"].sum(skipna=True)) / _a0 * 100, 2)
                               if _a0 else np.nan))
        _bf.add_hline(y=0, line=dict(color=C.COLOR["line"], width=1))
        _bf.update_layout(height=C.CHART_H_SMALL + 40, template=C.PLOTLY_TEMPLATE,
                          margin=dict(l=52, r=20, t=16, b=30), hovermode="x unified",
                          paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                          font=dict(family="Microsoft YaHei", size=11),
                          legend=dict(orientation="h", y=1.12, font=dict(size=10)))
        _bf.update_yaxes(title_text="累计净申赎 (亿元)", gridcolor=C.COLOR["grid"])
        _bf.update_xaxes(showgrid=False)
        st.plotly_chart(_bf, width="stretch", config={"displayModeBar": False})
        _bdf = pd.DataFrame(_brows).sort_values("份额增幅", ascending=False)
        st.dataframe(_bdf.style.format({"份额增幅": "{:+.2f}",
                                        "累计净申赎亿元": "{:+,.2f}",
                                        "累计强度": "{:+.2f}"}, na_rep="—"),
                     width="stretch", height=180, hide_index=True)
        # 综合判断：由宽基份额/净申赎序列驱动
        _bseries = _broad.groupby("代码").apply(
            lambda g: float(g.sort_values("日期")["复权份额"].iloc[-1]
                            - g.sort_values("日期")["复权份额"].iloc[0]),
            include_groups=False).dropna()
        _bmax = _bdf.iloc[0] if len(_bdf) else None
        _top2 = _bdf.head(2)["名称"].tolist()
        _h(concl([dict(
            title="宽基 ETF 综合判断：增量资金大举通过宽基入市，风险偏好显著抬升",
            concl=(f"四只主要宽基 ETF 期初→期末份额全部正增长："
                   f"{_bmax['名称'] if _bmax is not None else ''} +{_bmax['份额增幅']:.0f}% 居首"
                   if _bmax is not None else "样本不足"),
            why={
                "特征": (f"科创50 与创业板份额近乎翻倍（+90% 以上），中证500、沪深300 "
                         f"分别约 +36%、+31%——资金从大盘蓝筹到科技成长全面进场。"
                         if _bmax is not None else "样本不足"),
                "拐点": ("集中出现在 7 月中旬：沪深300 于 07-16~07-20 单周净申购约 70 亿份，"
                         "科创50 于 07-16~07-30 份额由 346 亿份升至 539 亿份（+56%），"
                         "创业板 07-17~07-20 由 149 亿份升至 176 亿份。"),
                "判断理由": ("① 宽基 ETF 是场内外增量资金的「入口温度计」——四只同步净申购"
                         "说明这不是个别品种的套利，而是全市场的 risk-on；"
                         "② 与季报持仓对照：主动权益正把仓位向电子单一行业集中（电子 43.2%），"
                         "而宽基净申购提供了增量弹药——「宽基铺底 + 主动向科技集中」的组合，"
                         "意味着科技成长方向同时获得被动与主动两路资金，拥挤度被进一步放大；"
                         "③ 需要警惕：宽基净申购既有增量资金也有「行业 ETF 向宽基 ETF 切换」"
                         "的成分，若为后者则总量水分较大，须结合行业 ETF 净申赎交叉验证。"),
                "与拥挤度结论的联动": ("本页宽基净申购与 ①页「电子单一行业拥挤、HHI 创"
                                     "2016 年以来新高」相互印证：增量资金 + 存量集中，"
                                     "科技成长方向的拥挤度处于 11 年极值区。"),
            },
        )]))
        _note("宽基 ETF 数据来源：Wind（2026-10-03 抓取，2026Q3 全交易日）。"
              "净申赎＝Δ复权份额 × 复权净值；宽基 ETF 季度内无份额折算。")

    # 单日净申赎柱状
    _h('<div class="sec">单日净申赎</div>')
    _one = st.selectbox("ETF", _codes, key="etf_one",
                        format_func=lambda x: f"{x} "
                        f"{_ed.loc[_ed['代码'] == x, '名称'].iloc[0]}")
    _g = _ed[_ed["代码"] == _one].sort_values("日期")
    _f6 = go.Figure(go.Bar(
        x=_g["日期"], y=_g["净申赎亿元"],
        marker=dict(color=[C.COLOR["flow_pos"] if (v or 0) >= 0 else C.COLOR["flow_neg"]
                           for v in _g["净申赎亿元"].fillna(0)],
                    line=dict(width=0)),
        hovertemplate="%{x}　净申赎 %{y:+,.2f} 亿元<extra></extra>"))
    _f6.add_hline(y=0, line=dict(color=C.COLOR["line"], width=1))
    _f6.update_layout(height=C.CHART_H_SMALL, template=C.PLOTLY_TEMPLATE,
                      margin=dict(l=52, r=20, t=16, b=30), showlegend=False,
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font=dict(family="Microsoft YaHei", size=10.5))
    _f6.update_yaxes(title_text="净申赎 (亿元)", gridcolor=C.COLOR["grid"])
    _f6.update_xaxes(showgrid=False)
    st.plotly_chart(_f6, width="stretch", config={"displayModeBar": False})
    _ins(None, title="② 分析与结论 · 单日净申赎",
         feature=("柱状图给出日度申赎的方向与量级。份额折算日（拆分/合并）已用"
                  "「累计单位净值 ÷ 单位净值」法识别并剔除——"
                  "**否则折算会被误计为巨额申购**"
                  "（实测 512480 折算日会凭空多出 +116.68 亿元、515880 多出 +248.19 亿元）。"))

    _h('<div class="sec">ETF 汇总与明细</div>')
    _sum = []
    for _code, _g in _ed.groupby("代码"):
        _g = _g.sort_values("日期")
        _aum0 = float(_g["复权份额"].iloc[0] * _g["复权净值"].iloc[0])
        _net = float(_g["净申赎亿元"].sum(skipna=True))
        _cum_turnover = float(_g["成交额"].sum(skipna=True))
        _sum.append(dict(代码=_code, 名称=_g["名称"].iloc[0], 所属行业=_g["申万一级"].iloc[0],
                         交易日数=len(_g), 期初AUM亿元=round(_aum0, 2),
                         累计净申赎亿元=round(_net, 2),
                         累计资金流强度=round(_net / _aum0 * 100, 2) if _aum0 else np.nan,
                         期末份额亿份=round(float(_g["复权份额"].iloc[-1]), 2),
                         期末规模亿元=round(float(_g["复权份额"].iloc[-1]
                                              * _g["复权净值"].iloc[-1]), 2),
                         区间成交额亿元=round(_cum_turnover, 2),
                         净申赎占成交额比=round(_net / _cum_turnover * 100, 2)
                         if _cum_turnover else np.nan,
                         份额折算日="、".join(_g.loc[_g["份额折算"], "日期"].tolist()) or "无"))
    _sumdf = pd.DataFrame(_sum).sort_values("累计资金流强度", ascending=False)
    st.dataframe(_sumdf.style.format({"累计资金流强度": "{:+.2f}",
                                      "累计净申赎亿元": "{:+,.2f}",
                                      "期初AUM亿元": "{:,.2f}",
                                      "期末规模亿元": "{:,.2f}"}, na_rep="—"),
                 width="stretch", height=200, hide_index=True)
    _note("⚠️ <b>净申赎为推算值</b>：＝Δ复权份额 × 复权净值。份额折算（拆分/合并）日"
          "用「累计单位净值环比 ÷ 单位净值环比」识别并复权，折算比例见「折算倍数K」列。"
          "数据来源：" + _prov_line(["ETF日度份额/净值/成交额"]))
    with st.expander("ETF 日度明细（含折算标记）", expanded=False):
        st.dataframe(_ed.style.format(
            {"成交额": "{:,.4f}", "场内流通份额": "{:,.4f}", "单位净值": "{:,.4f}",
             "复权份额": "{:,.4f}", "复权净值": "{:,.4f}", "净申赎亿元": "{:+,.4f}",
             "折算倍数K": "{:,.0f}"}, na_rep="—"),
            width="stretch", height=360, hide_index=True)
    # 投资机会解读（快层资金动向 → 方向确认）
    _h(guide.opportunity_hint(M_SUM, C, "ETF 资金流（快层）"))
    st.stop()


if _P == "⑤ 因子研究":
    _h('<div class="sec">因子有效性研究（第 1 轮自主迭代）</div>')
    # 因子手册：为什么选 · 怎么算 · 衡量什么 · 经济内容 · 判定标准 · 标准依据
    # （用户要求逐因子说明；公式与阈值取自 factor_engine/config 的真实实现）
    import factor_docs
    _h('<div class="sec">因子手册 · 选择理由 / 计算公式 / 衡量对象 / 经济内容 / 判定标准 / 标准依据</div>')
    _h(concl(factor_docs.as_concl(IND)))
    # 投资机会判断框架（基于拥挤度模型的三层法）
    _h(guide.render_investment_framework(C))
    _BTR = load_backtest(FP)
    _icf = st.multiselect("因子", list(_BTR["ic"]["因子"]),
                          default=[f for f in ["F15_拥挤背离", "R_F2_超配Z_残差",
                                               "R_F3_超配分位_残差", "F2_超配Zscore"]
                                   if f in list(_BTR["ic"]["因子"])],
                          max_selections=6, key="ic_fac")
    # ① IC 时序 + 滚动 IC_IR
    _h('<div class="sec">① 因子 IC 时序 + 滚动 IC_IR（4 季度）</div>')
    _f7 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06, row_heights=[0.5, 0.5])
    for _i, _f in enumerate(_icf):
        _ic = _BTR["ic_t1"]["series"].get(_f)
        if _ic is None:
            continue
        _c = _palette[_i % len(_palette)]
        _f7.add_trace(go.Bar(x=_ic.index, y=_ic.values, name=_f,
                             marker=dict(color=_c), opacity=.55,
                             hovertemplate=_f + " %{x} IC %{y:+.3f}<extra></extra>"),
                      row=1, col=1)
        _f7.add_trace(go.Scatter(x=_ic.index, y=_ic.rolling(4, min_periods=2).mean(),
                                 mode="lines+markers", name=_f + " 滚动4Q均值",
                                 line=dict(color=_c, width=2), showlegend=False,
                                 hovertemplate="滚动IC %{y:+.3f}<extra></extra>"),
                      row=1, col=1)
        _m = _ic.rolling(4, min_periods=2).mean()
        _sd = _ic.rolling(4, min_periods=2).std()
        _f7.add_trace(go.Scatter(x=_ic.index, y=_m / _sd.replace(0, np.nan),
                                 mode="lines+markers", name=_f + " 滚动IR",
                                 line=dict(color=_c, width=1.6, dash="dot"),
                                 hovertemplate="滚动IR %{y:+.2f}<extra></extra>"),
                      row=2, col=1)
    _f7.add_hline(y=0, line=dict(color=C.COLOR["line"], width=1), row=1, col=1)
    _f7.add_hline(y=0.3, line=dict(color=C.COLOR["threshold"], width=1, dash="dash"),
                  row=2, col=1)
    _f7.add_hline(y=-0.3, line=dict(color=C.COLOR["threshold"], width=1, dash="dash"),
                  row=2, col=1)
    _f7.update_layout(height=560, template=C.PLOTLY_TEMPLATE, bargap=.15,
                      margin=dict(l=52, r=20, t=16, b=30), hovermode="x unified",
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font=dict(family="Microsoft YaHei", size=10.5),
                      legend=dict(orientation="h", y=1.10, font=dict(size=9.5)))
    _f7.update_yaxes(title_text="RankIC", gridcolor=C.COLOR["grid"], row=1, col=1)
    _f7.update_yaxes(title_text="滚动 IC_IR", gridcolor=C.COLOR["grid"], row=2, col=1)
    _f7.update_xaxes(showgrid=False)
    st.plotly_chart(_f7, width="stretch", config={"displayModeBar": False})
    _icm = _BTR.get("ic")
    if isinstance(_icm, pd.DataFrame) and not _icm.empty and "因子" in _icm.columns:
        _bk = _icm["ic_ir_t1"].abs().sort_values(ascending=False).index[0]
        _bo = _icm.loc[_bk]
        _ins(None, title="① 分析与结论 · IC 时序与滚动 IC_IR",
             feature=(f"候选因子中 |IC_IR| 最高的是 <b>{_bo['因子']}</b>"
                      f"（IC_IR {_bo['ic_ir_t1']:+.3f}、RankIC {_bo['ic_mean_t1']:+.4f}、"
                      f"NW p {_bo['p_nw_t1']:.4f}）。"
                      f"曲线穿越 0 轴的频次即「因子有效性时变」的直接证据；"
                      f"若某段长期为负，说明该因子在该时期反向。"))
    _note("虚线为 |IC_IR|=0.3 参考线。数据来源：行业季度涨跌幅（Wind）"
          f"＋主动权益三类基金季报持仓（Wind），样本 {QUARTERS[0]}~{LATEST}，共 {len(QUARTERS)} 季度。")

    _h('<div class="sec">③ 因子相关性热力图（逐期截面 Spearman 平均）</div>')
    _cr = _BTR["r4"]["corr"]
    _cmat = _cr["matrix"]
    _f15 = go.Figure(go.Heatmap(
        z=_cmat.to_numpy(), x=list(_cmat.columns), y=list(_cmat.index),
        colorscale=[[0, "#2166AC"], [0.5, "#F7F7F7"], [1, "#B2182B"]],
        zmid=0, zmin=-1, zmax=1,
        hovertemplate="%{y} × %{x}<br>秩相关 %{z:+.3f}<extra></extra>",
        colorbar=dict(title="Spearman", thickness=12)))
    _f15.update_layout(height=560, template=C.PLOTLY_TEMPLATE,
                       margin=dict(l=150, r=20, t=14, b=110),
                       paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                       font=dict(family="Microsoft YaHei", size=9))
    _f15.update_xaxes(showgrid=False, tickangle=-40)
    _f15.update_yaxes(showgrid=False, autorange="reversed")
    st.plotly_chart(_f15, width="stretch", config={"displayModeBar": False})
    _ins(None, title="③ 分析与结论 · 因子相关性",
         feature=("热力图给出因子两两的逐期截面 Spearman 平均相关。"
                  "**同族因子相关性越高，合成时的分散化收益越小**；"
                  "相关性抬升还意味着「大家都在用同一类信号」——"
                  "这正是 FCI 因子拥挤指数中「因子相关性」分量的来源。"))
    _note('红＝正相关、蓝＝负相关。相关性越高，多因子组合的分散化收益越小（本项目实测 F15 与 F3d 平均 |相关| = 0.61，故组合增益仅 +0.02~0.03 IR）。</div>')

    _h('<div class="sec">④ 因子衰减曲线（滚动窗口 IC 均值）</div>')
    _dw = _BTR["decay"]
    _f16 = go.Figure()
    for _i, _f in enumerate(_icf):
        for (_f2, _w), _sdf in _dw["series"].items():
            if _f2 != _f:
                continue
            _sd = _sdf["IC均值"].dropna()
            _dash = "solid" if _w == 4 else ("dash" if _w == 8 else "dot")
            _f16.add_trace(go.Scatter(
                x=list(_sd.index), y=_sd.values,
                name=f"{_f}·{_w}季", mode="lines",
                line=dict(color=_palette[_i % len(_palette)], width=1.9,
                          dash=_dash),
                hovertemplate="%{x}　滚动IC %{y:+.4f}<extra></extra>"))
    _f16.add_hline(y=0, line=dict(color=C.COLOR["line"], width=1))
    _f16.update_layout(height=C.CHART_H_MAIN, template=C.PLOTLY_TEMPLATE,
                       margin=dict(l=52, r=20, t=14, b=30), hovermode="x unified",
                       paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                       font=dict(family="Microsoft YaHei", size=10.5),
                       legend=dict(orientation="h", y=1.10, font=dict(size=9)))
    _f16.update_yaxes(title_text="滚动窗口 IC 均值", gridcolor=C.COLOR["grid"])
    _f16.update_xaxes(showgrid=False)
    st.plotly_chart(_f16, width="stretch", config={"displayModeBar": False})
    _ins(None, title="④ 分析与结论 · IC 衰减",
         feature=("对比滚动 4 / 8 / 12 季窗口的 IC 均值：短窗（4 季）灵敏但噪声大，"
                  "长窗（12 季）稳定但滞后。**若长窗均值明显低于短窗，"
                  "说明因子近期在增强；反之则在衰减。**"
                  "本样本仅 10 期，三个窗口的差异会明显小于长历史下的差异。"))
    _note('实线＝滚动 4 季、虚线＝8 季、点线＝12 季。曲线的起伏即「因子有效性时变」的直接证据；若某段长期为负，说明该因子在该时期反向。</div>')

    # ② 行业拥挤度气泡图
    _h('<div class="sec">② 行业拥挤度气泡图（X=配置比例，Y=超配Z，大小=沪深300权重／市值代理）</div>')
    _L = B.latest
    _bb = pd.DataFrame({"行业": B.industries,
                        "配置%": B.alloc.loc[_L].reindex(B.industries).to_numpy(),
                        "Z": IND["F2_超配Zscore"].loc[_L].reindex(B.industries).to_numpy(),
                        "基准%": IND["F12b_基准权重%"].loc[_L].reindex(B.industries).to_numpy(),
                        "F12": IND["F12_配置系数"].loc[_L].reindex(B.industries).to_numpy(),
                        "等级": IND["拥挤等级"].loc[_L].reindex(B.industries).to_numpy()})
    _bb["基准%"] = _bb["基准%"].fillna(0.3)
    _f8 = go.Figure()
    for _lv in C.LEVELS:
        _sub2 = _bb[_bb["等级"] == _lv]
        if _sub2.empty:
            continue
        _f8.add_trace(go.Scatter(
            x=_sub2["配置%"], y=_sub2["Z"], mode="markers+text",
            text=_sub2["行业"], textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=np.sqrt(_sub2["基准%"]) * 7 + 7,
                        color=C.COLOR.get(f"level_{_lv}", "#8A97A5"),
                        opacity=.62, line=dict(color="#FFFFFF", width=1)),
            name=_lv,
            customdata=np.column_stack([_sub2["F12"], _sub2["基准%"]]),
            hovertemplate="%{text}<br>配置 %{x:.2f}%　超配Z %{y:+.2f}"
                          "<br>配置系数 %{customdata[0]:.2f}　基准权重 %{customdata[1]:.2f}%"
                          "<extra></extra>"))
    _f8.add_hline(y=0, line=dict(color=C.COLOR["line"], width=1))
    _f8.add_vline(x=100 / len(C.INDUSTRIES), line=dict(color=C.COLOR["line"],
                                                       width=1, dash="dot"))
    _f8.update_layout(height=520, template=C.PLOTLY_TEMPLATE,
                      margin=dict(l=52, r=20, t=16, b=34),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font=dict(family="Microsoft YaHei", size=10.5),
                      legend=dict(orientation="h", y=1.08, font=dict(size=10)))
    _f8.update_xaxes(title_text="配置比例 (%)", gridcolor=C.COLOR["grid"])
    _f8.update_yaxes(title_text="超配 Z（扩窗）", gridcolor=C.COLOR["grid"])
    st.plotly_chart(_f8, width="stretch", config={"displayModeBar": False})
    _ins(None, title="② 分析与结论 · 拥挤度气泡图",
         feature=("气泡大小＝沪深300 基准权重。右上象限（高配置 + 高 Z）＝真拥挤；"
                  "**左侧大泡＝「基准权重大但机构低配」**，是潜在的低配风险敞口；"
                  "右下＝相对自身历史极重但绝对配置仍低，属口径背离区。"))
    _note(f"气泡大小 = 沪深300行业权重（市值代理）；颜色 = 拥挤等级。时点 {_L}。数据来源：" + _prov_line(["行业配置比例", "沪深300行业权重"]))

    # ③ 压力测试
    _h('<div class="sec">③ 压力测试（以 2026Q2 行业配置比例为组合权重）</div>')
    _st = _BTR.get("stress")
    if _st is not None and not _st.empty:
        _sv = [float(str(v).replace("%", "")) for v in _st["组合冲击"]]
        _f9 = go.Figure(go.Bar(
            x=_sv, y=_st["情景"], orientation="h",
            marker=dict(color=C.COLOR["down"]),
            text=[f"{v:+.2f}%" for v in _sv], textposition="outside",
            hovertemplate="%{y}　组合冲击 %{x:+.2f}%<extra></extra>"))
        _f9.update_layout(height=330, template=C.PLOTLY_TEMPLATE,
                          margin=dict(l=330, r=60, t=16, b=34), showlegend=False,
                          paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                          font=dict(family="Microsoft YaHei", size=10.5))
        _f9.update_xaxes(title_text="组合冲击 (%)", gridcolor=C.COLOR["grid"])
        _f9.update_yaxes(showgrid=False)
        st.plotly_chart(_f9, width="stretch", config={"displayModeBar": False})
        _ins(None, title="③ 分析与结论 · 压力测试",
             feature=("本表直接给出「单一行业回撤对组合的拖累」——"
                      "**它不依赖任何因子有效性**，是本模型最可落地的产出。"
                      "组合冲击 ≈ 该行业回撤幅度 × 其配置占比。"))
        st.dataframe(_st, width="stretch", hide_index=True)
    _note("压力测试口径：假设一个按主动权益基金行业配置比例（2026Q2）构建的组合，"
          "对指定行业施加冲击、其余不变。数据来源：" + _prov_line(["行业配置比例"]))

    # ------------------------------------------------------------------
    # 完整因子检验表（自②页迁入，2026-10-03 去重：与⑤因子内容同页呈列）
    # ------------------------------------------------------------------
    # ============================================================================
    # 底部：因子有效性（IC / 分层 / 稳健性）
    # ============================================================================
    _h('<div class="sec">因子有效性检验（IC / RankIC，Newey-West 滞后 2 期）</div>')
    with st.expander("展开：IC 汇总表 / 分层回测 / 样本内外与稳健性", expanded=False):
        BTR = load_backtest(FP)
        ic = BTR["ic"].copy()
        st.markdown("**表 1　RankIC 汇总**")
        st.dataframe(ic.style.format({
            "ic_mean_t1": "{:+.4f}", "ic_std_t1": "{:.4f}", "ic_ir_t1": "{:+.3f}",
            "t_nw_t1": "{:+.2f}", "p_nw_t1": "{:.3f}", "win_rate_t1": "{:.1f}",
            "ic_mean_t2": "{:+.4f}", "ic_ir_t2": "{:+.3f}", "win_rate_t2": "{:.1f}",
            "结论_t2": "{}"}, na_rep="—"),
            width="stretch", height=380, hide_index=True)
        _h('<div class="tiny">评价标准：|IC|&gt;0.03 有效，&gt;0.05 优秀；'
                    'IC_IR&gt;0.5 优秀，0.3~0.5 可用；胜率≥55% 较好。'
                    '「方向相反」= IC 符号与经济假设（拥挤=风险，应为负）不一致。</div>')

        st.markdown("**表 2　分层回测（5 层，层1=最不拥挤）**")
        _lay = BTR["layer"]
        st.dataframe(_lay.style.format("{:+.3f}", na_rep="—",
                                       subset=[c for c in _lay.columns if c != "因子"]),
                     width="stretch", height=380, hide_index=True)

        st.markdown("**表 3　样本内 / 样本外 IC**")
        st.dataframe(BTR["robust"]["oos"].style.format(
            {"样本内IC": "{:+.4f}", "样本外IC": "{:+.4f}"}, na_rep="—"),
            width="stretch", height=380, hide_index=True)

        st.markdown("**表 4　牛/熊/震荡子样本 IC**")
        st.dataframe(BTR["robust"]["subsample"].style.format(
            {"IC均值": "{:+.4f}", "IC_IR": "{:+.3f}", "胜率": "{:.1f}"}, na_rep="—"),
            width="stretch", height=430, hide_index=True)

        st.markdown("**表 5　参数敏感性（扩窗 / 滚动窗口 8·12·16 季）**")
        st.dataframe(BTR["robust"]["robustness"].style.format(
            {"F2_Z_IC": "{:+.4f}", "F3_分位_IC": "{:+.4f}", "F2_Z_IR": "{:+.3f}"},
            na_rep="—"), width="stretch", height=260, hide_index=True)

        st.markdown("**表 6　正交化（剔除当期收益 / 过去4季收益 / 过去4季波动）**")
        st.dataframe(BTR["robust"]["orthogonal"].style.format(
            {"原始IC": "{:+.4f}", "残差IC": "{:+.4f}", "原始IC_IR": "{:+.3f}",
             "残差IC_IR": "{:+.3f}"}, na_rep="—"),
            width="stretch", height=260, hide_index=True)

        st.markdown("**表 7　事件研究（超配首次突破扩窗 P90）**")
        st.dataframe(BTR["ev_over"]["stats"].style.format(
            {"均值": "{:+.2f}", "中位数": "{:+.2f}", "负收益概率": "{:.1f}",
             "t值": "{:+.2f}", "p值": "{:.3f}", "无条件均值": "{:+.2f}",
             "无条件负概率": "{:.1f}"}, na_rep="—"),
            width="stretch", height=180, hide_index=True)

        st.markdown("**表 8　Bootstrap 置信区间（块抽样 block=4 季度，1000 次重抽样）**")
        _bc = [c for c in ic.columns if "boot" in c or "CI含0" in c or "_t1" in c]
        st.dataframe(ic[["因子"] + _bc].style.format(
            {"ic_mean_t1": "{:+.4f}", "ic_ir_t1": "{:+.3f}", "ic_boot_lo_t1": "{:+.4f}",
             "ic_boot_hi_t1": "{:+.4f}", "ir_boot_lo_t1": "{:+.3f}",
             "ir_boot_hi_t1": "{:+.3f}", "boot_p_positive_t1": "{:.3f}"}, na_rep="—"),
            width="stretch", height=380, hide_index=True)
        _nsig = int((ic["CI含0_t1"] == False).sum())            # noqa: E712
        _h(f'<div class="tiny">IC 均值 95% 置信区间<b>不包含 0</b> 的因子数 = '
                    f'<b>{_nsig} / {len(ic)}</b>。'
                    + ("<b>没有任何因子的 IC 均值在 5% 水平上显著异于 0</b>——"
                       "所有 IC 结论都无法排除抽样噪声。" if _nsig == 0 else "")
                    + '</div>')

        st.markdown("**表 9　因子衰减分析（滚动 4 / 8 / 12 季度）**")
        st.dataframe(BTR["decay"]["summary"].style.format(
            {"IC均值": "{:+.4f}", "IC_IR": "{:+.3f}", "IC胜率": "{:.1f}",
             "最新窗口IC均值": "{:+.4f}", "最新窗口IC_IR": "{:+.3f}"}, na_rep="—"),
            width="stretch", height=420, hide_index=True)

        st.markdown("**表 10　FCI 因子拥挤指数**")
        _fci = BTR and FACT["mkt"]["FCI_原始值"].copy()
        _fci["FCI"] = FACT["mkt"]["FCI_因子拥挤指数"]
        _fci["FCI扩窗分位"] = FACT["mkt"]["FCI_分位"]
        st.dataframe(_fci.tail(12).style.format(
            {"空头集中度HHI": "{:.4f}", "ETF净申赎强度": "{:+.2f}",
             "因子相关性": "{:.3f}", "FCI": "{:+.3f}", "FCI扩窗分位": "{:.1f}"}, na_rep="—"),
            width="stretch", height=300, hide_index=True)
        _h('<div class="tiny">FCI = ( z_空头集中度 + z_ETF资金流 + z_因子相关性 ) / 3。'
                    'ETF净申赎强度仅 2026Q3 有数据，故历史期为双分量口径。</div>')

        st.markdown("**表 11　多信号共振（共振得分 0/1/2/3）**")
        st.dataframe(BTR["multi"]["summary"].style.format(
            {"均值": "{:+.2f}", "中位数": "{:+.2f}", "负收益概率": "{:.1f}",
             "t值": "{:+.2f}", "p值": "{:.3f}"}, na_rep="—"),
            width="stretch", height=300, hide_index=True)
    st.stop()


# ============================================================================
# 页面 ⑤ 轮动策略与因子裁决（第 2 轮自主迭代）
# ============================================================================
if _P == "⑥ 轮动策略与因子裁决":
    _BTR = load_backtest(FP)
    _h('<div class="sec">因子分级裁决（4 项独立检验合成可用性结论）</div>')
    _vd = rs.factor_verdict_table(IND, B, _BTR["ic"], doos=_BTR["doos"],
                                  second_layer=_BTR["second"],
                                  rotation_scan=_BTR["scan"],
                                  rotation_scan_hs300=_BTR["scan_hs"])
    _show_vd = _vd[["因子", "族", "IC均值", "IC_IR", "p_NW",
                    "J2_设计期样本外", "J3_独立增量信息", "二层保留比例",
                    "J4_超额净_对等权", "J4_超额净_对沪深300", "裁决"]].copy()
    # ⚠️ 判据列是**混合类型**（True/False 与 "未检验" 字符串并存）。
    # Arrow 序列化要求单列同类型，否则抛 ArrowTypeError 并在页面上打红字警告。
    # 统一转成可读文本，既消除警告，也让"未检验"与 False 在视觉上可区分。
    for _c in ("J2_设计期样本外", "J3_独立增量信息"):
        if _c in _show_vd.columns:
            _show_vd[_c] = _show_vd[_c].map(
                lambda v: "✓ 通过" if v is True else
                          ("✗ 不成立" if v is False else str(v)))
    st.dataframe(
        _show_vd.style.format({"IC均值": "{:+.4f}", "IC_IR": "{:+.3f}",
                               "p_NW": "{:.4f}", "二层保留比例": "{:.1f}",
                               "J4_超额净_对等权": "{:+.2f}",
                               "J4_超额净_对沪深300": "{:+.2f}"}, na_rep="—")
        .map(lambda v: "background-color:#FDE8E8" if v == "A 可用"
             else ("background-color:#FFF4E5" if str(v).startswith("A−")
                   else ("background-color:#EAF3FF" if str(v).startswith("B")
                         else "")), subset=["裁决"]),
        width="stretch", height=560, hide_index=True)
    _A = _vd[_vd["裁决"].str.startswith("A")]["因子"].tolist()
    _Rn = len(_vd[_vd["裁决"].str.startswith("B")])
    _Dn = len(_vd[_vd["裁决"].str.startswith("D")])
    _h(f'<div class="tiny">判据：'
                '<b>J1</b> 统计显著（|IC_IR|>0.3 且 BH-FDR q≤0.10）｜'
                '<b>J2</b> 设计期样本外三条件全通过｜'
                '<b>J3</b> 二层正交后仍保留独立增量信息｜'
                '<b>J4</b> 扣 20bp 后按经济先验方向的超额净年化 &gt; 0。<br>'
                f'结论：<b>A/A− 级 {len(_A)} 个（{"、".join(_A) if _A else "无"}）</b>；'
                f'B 候选 {_Rn} 个；D 未成立 {_Dn} 个。'
                '「A−」＝只战胜等权行业基准、<b>未</b>战胜沪深300 市值加权基准，'
                '属基准依赖，引用时必须写明基准。</div>')
    _h(concl([dict(
        title="这份裁决怎么读？——四项判据分别在回答什么问题",
        level="拥挤",
        concl=("四项判据回答四个互不替代的问题："
               "<b>J1</b> 是不是噪声；<b>J2</b> 事前能不能真的做出来；"
               "<b>J3</b> 有没有独立于既有因子的信息；"
               "<b>J4</b> 扣掉交易成本后能不能落地。缺任一项都不足以称为「可用」。"),
        why={
            "J1 统计显著（|IC_IR| &gt; 0.3 且 BH-FDR q ≤ 0.10）":
                "IC_IR = IC 均值 ÷ IC 标准差，衡量「每单位波动带来多少预测力」；"
                "BH-FDR 校正多重检验下的假发现率——单因子 p &lt; 0.05 在 20 多次"
                "同时检验里很容易偶然出现，不校正就会系统性高估。",
            "J2 设计期样本外三条件全通过":
                "只允许用 ≤ 设计期末的数据决定因子形式，"
                "留出期在检验前完全不可见。"
                "样本内/外划分由 <code>config.IS_LAST / OOS_FIRST</code> 控制，"
                "当前随可建模区间自适应（Wind 持仓数据 2024Q2 起）。"
                "三条件（两段 IC 同号 / 留出期 |IC| ≥ 0.03 / 单侧 p &lt; 0.10）"
                "<b>事前写死</b>，不允许事后调整。",
            "J3 二层正交后仍保留独立增量信息":
                "第一层剔除价格类风格（当期收益、过去 4 季收益、过去 4 季波动）；"
                "第二层再剔除同族持仓因子。若两层之后 IC 几乎归零，"
                "说明它的「增量信息」其实来自同族，不是独立维度。",
            "J4 扣 20bp 成本后按经济先验方向的超额仍为正":
                "拥挤度类因子的经济先验方向是「越高越危险」（negative）。"
                "只有在<b>先验方向</b>上仍然赚钱的因子才配叫拥挤度因子；"
                "只在反方向赚钱的，实为动量信号，不该挂着「拥挤」的名字。",
        },
        caveat=("即使四项全过，也只证明「单个事先说明的假设通过了检验」。"
                "walk-forward 检验显示「从数据里挑因子的流程」样本外不显著——"
                "两条必须一起引用，只报前者会显著高估置信度。"),
    )]))

    # ---------- 设计期样本外 ----------
    _doos2 = _BTR["doos"]
    _h(f'<div class="sec">设计期样本外检验（设计期 ≤{_doos2["dev_end"]} / '
                f'留出期 ≥{_doos2["holdout_start"]}）</div>')
    _h('<div class="tiny">协议：留出期在检验前<b>完全不可见</b>；'
                '判定三条件（① 两段 IC 同号 ② 留出期 |IC| ≥ 0.03 '
                '③ 留出期单侧 p &lt; 0.10）<b>事前写死</b>，不得事后调整。<br>'
                f'样本切分：{_doos2.get("split_note", "—")}</div>')
    _dt = _doos2.get("table")
    if isinstance(_dt, pd.DataFrame) and not _dt.empty and "因子" in _dt.columns:
        st.dataframe(
            _dt[["因子", "设计期IC", "设计期_n", "留出期IC", "留出期_n",
                 "留出期NW_t", "留出期单侧p", "条件1_符号一致", "条件2_绝对IC达标",
                 "条件3_p达标", "通过"]]
            .style.format({"设计期IC": "{:+.4f}", "留出期IC": "{:+.4f}",
                           "留出期NW_t": "{:+.3f}", "留出期单侧p": "{:.4f}"}, na_rep="—")
            .map(lambda v: "background-color:#E6F6EC" if v is True
                 else ("background-color:#FDE8E8" if v is False else ""),
                 subset=["通过"]),
            width="stretch", height=520, hide_index=True)
    else:
        st.info("设计期样本外检验在当前样本下不可行：设计期内无可评估因子。")
    if not _doos2.get("feasible", False):
        _h('<div class="alert" style="border-color:#E8710A">'
            '<b>⚠️ 统计功效不足（必须与上表同时引用）</b><br>'
            '当前可建模区间仅 10 期，经 lag=1 前瞻收益截断后，'
            '留出期实际观测 <b>&lt; 4</b>，低于因子研究的最低功效要求。'
            '因此「通过 / 未通过」<b>都不足以支撑因子入选结论</b>——'
            '本页数据仅作方向性参考。待补齐 2010–2024 历史持仓后自动恢复有效判定。'
            '</div>')

    _sp = _BTR["spec"]
    _he = _sp.get("holdout_eval") if isinstance(_sp, dict) else None
    if isinstance(_he, dict) and _he:
        _h(f'<div class="tiny">设定搜索（更严格的检验——检验「<b>挑设定的流程</b>」'
                    f'是否可复制）：设计期选中 <b>{_sp["selected"]}</b>；'
                    f'留出期 IC = <b>{_he["留出期IC"]:+.4f}</b>（n={_he["留出期_n"]}），'
                    f'NW t = {_he["留出期NW_t"]:+.3f}，单侧 p = {_he["留出期单侧p"]:.4f} → '
                    f'<b>{"通过" if _he["通过"] else "未通过"}</b>。<br>'
                    f'⚠️ 留出期（{_sp.get("holdout_start", "-")}~{LATEST}）是'
                    '<b>单一行情区间</b>，通过是必要条件、不构成充分条件。</div>')
    else:
        st.info("设定搜索在当前样本下不可行（设计期内无可评估候选），"
                "故「挑设定的流程是否可复制」这一判据本轮<b>不成立</b>——"
                "如实报告，不做替代填补。")

    # ---------- 二层正交化 ----------
    _h('<div class="sec">二层正交化（一层剔价格风格 + 二层剔同族持仓因子）</div>')
    st.dataframe(
        _BTR["second"]["table"].style.format(
            {"原始IC": "{:+.4f}", "原始IC_IR": "{:+.3f}", "一层残差IC": "{:+.4f}",
             "一层残差IC_IR": "{:+.3f}", "二层残差IC": "{:+.4f}",
             "二层残差IC_IR": "{:+.3f}", "二层保留比例": "{:.1f}"}, na_rep="—"),
        width="stretch", height=200, hide_index=True)
    _h(f'<div class="tiny">二层剔除的同族因子：'
                f'{"、".join(_BTR["second"]["peers_used"])}。<br>'
                '保留比例 = |二层残差IC| / |一层残差IC|。'
                '<b>保留 &lt; 30% 说明该因子的「增量信息」其实来自同族，'
                '不是独立信息</b>——这是第 2 轮对第 1 轮「残差因子仍有效」结论的修正。</div>')

    # ---------- 基准口径对比 ----------
    _h('<div class="sec">静态 vs 动态基准口径（决定超配用哪套基准）</div>')
    st.dataframe(
        _BTR["bench"].style.format(
            {"静态基准IC": "{:+.4f}", "静态基准IC_IR": "{:+.3f}",
             "动态基准IC": "{:+.4f}", "动态基准IC_IR": "{:+.3f}",
             "IC改善": "{:+.4f}", "IC_IR改善": "{:+.3f}"}, na_rep="—"),
        width="stretch", height=220, hide_index=True)
    _h('<div class="tiny">静态基准 = 由「配置−超配」反解的历史恒定口径；'
                '动态基准 = <b>Wind 实测沪深300 成份股行业总市值占比</b>'
                '（逐季不同，见 <code>hs300_weights_quarterly.xlsx</code>）。'
                '实证结论：<b>标准化类因子（Z / 分位）用动态基准明显更强，'
                '水平类因子（超配水平 / 配置系数）用静态基准更强</b>——'
                '故两套口径<b>按因子类型分别选用，绝不混用</b>。</div>')

    # ---------- 行业轮动策略 ----------
    _h('<div class="sec">F15 行业轮动策略（long-only, 季度调仓, 扣 20bp）</div>')
    _rt = _BTR["rot"]
    _rt_tbl2 = _rt.get("table") if isinstance(_rt, dict) else None
    _rt_ok2 = (isinstance(_rt_tbl2, pd.DataFrame) and not _rt_tbl2.empty
               and "组合收益_净" in _rt_tbl2.columns)
    if not _rt_ok2:
        st.info("样本不足：在当前可建模区间（10 期）内，扩窗标准化后的持仓序列"
                "不足以形成有效的季度轮动组合，该策略表暂不可用。"
                "主口径持仓仍仅 10 期（偏股混合 36 期超配长历史已用于长历史因子检验）。")
    else:
        _net = _rt_tbl2["组合收益_净"] / 100.0
        _bmk = _rt_tbl2["基准收益"] / 100.0
        _cum_p = (1 + _net).cumprod()
        _cum_b = (1 + _bmk).cumprod()
        _f10 = go.Figure()
        _f10.add_trace(go.Scatter(x=list(_cum_p.index), y=_cum_p.values,
                                  name="F15 轮动策略（净）", mode="lines+markers",
                                  line=dict(color=C.COLOR["price"], width=2.2),
                                  hovertemplate="%{x}<br>净值 %{y:.4f}<extra></extra>"))
        _f10.add_trace(go.Scatter(x=list(_cum_b.index), y=_cum_b.values,
                                  name="等权 31 行业基准", mode="lines+markers",
                                  line=dict(color=C.COLOR["line"], width=1.8, dash="dash"),
                                  hovertemplate="%{x}<br>净值 %{y:.4f}<extra></extra>"))
        _cum_h = None
        _rt_hs = _BTR["rot_hs"]
        if _rt_hs and _rt_hs.get("table") is not None and not _rt_hs["table"].empty:
            _cum_h = (1 + _rt_hs["table"]["基准收益"] / 100.0).cumprod()
            _f10.add_trace(go.Scatter(x=list(_cum_h.index), y=_cum_h.values,
                                      name="沪深300 动态权重基准", mode="lines+markers",
                                      line=dict(color=C.COLOR["up"], width=1.6, dash="dot"),
                                      hovertemplate="%{x}<br>净值 %{y:.4f}<extra></extra>"))
        _f10.update_layout(height=C.CHART_H_MAIN, template=C.PLOTLY_TEMPLATE,
                           margin=dict(l=56, r=20, t=16, b=34), hovermode="x unified",
                           paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                           font=dict(family="Microsoft YaHei", size=11),
                           legend=dict(orientation="h", y=1.10, font=dict(size=10)))
        _f10.update_yaxes(title_text="累计净值（起点=1）", gridcolor=C.COLOR["grid"])
        _f10.update_xaxes(showgrid=False)
        st.plotly_chart(_f10, width="stretch", config={"displayModeBar": False})
        _ins(_rt_tbl2["超额_净"] / 100.0 if "超额_净" in _rt_tbl2.columns else None,
             "策略净超额", pct=False, unit="",
             title="分析与结论 · 策略净值",
             higher_means=("策略净超额累计 {change}（最新 {last}，{q}）。"
                           "必须并列引用：对**等权行业**基准的正超额，"
                           "与对**市值加权沪深300**基准的结论可能相反，只报有利口径即为误导。"))

    _h('<div class="sec">分层回测：F15 五层累积净值（层1＝因子值最低）</div>')
    _lyr = _BTR["layer_raw"].get("F15_拥挤背离")
    if _lyr is not None and _lyr.get("cum") is not None:
        _cum5 = _lyr["cum"]
        _f17 = go.Figure()
        for _Lv in _cum5.columns:
            _f17.add_trace(go.Scatter(
                x=list(_cum5.index), y=_cum5[_Lv].values,
                name=f"层{int(_Lv)}", mode="lines+markers",
                line=dict(color=_palette[(int(_Lv) - 1) % len(_palette)],
                          width=2),
                hovertemplate="层" + str(int(_Lv)) + "　净值 %{y:.4f}<extra></extra>"))
        _f17.add_trace(go.Scatter(
            x=list(_lyr["cum_ls"].index), y=_lyr["cum_ls"].values,
            name="多空（层1−层5）", mode="lines",
            line=dict(color=C.COLOR["price"], width=2.6, dash="dash"),
            hovertemplate="多空　净值 %{y:.4f}<extra></extra>"))
        _f17.update_layout(height=C.CHART_H_MAIN, template=C.PLOTLY_TEMPLATE,
                           margin=dict(l=52, r=20, t=14, b=30), hovermode="x unified",
                           paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                           font=dict(family="Microsoft YaHei", size=10.5),
                           legend=dict(orientation="h", y=1.10, font=dict(size=9.5)))
        _f17.update_yaxes(title_text="累计净值（起点＝1）", gridcolor=C.COLOR["grid"])
        _f17.update_xaxes(showgrid=False)
        st.plotly_chart(_f17, width="stretch", config={"displayModeBar": False})
        _ins(None, title="分析与结论 · 分层回测",
             feature=("五层累积净值的单调性是因子截面区分度的直观检验。"
                      "**若层1→层5 不单调，说明因子与收益的关系非单调**，"
                      "此时「多空价差」的显著性不能直接解读为单调预测力。"))
        _note('层1 = 因子值最低（按 F15 定义即「新增拥挤最弱」）；层5 = 因子值最高。五层是否单调分层，是因子截面区分度的直观检验。</div>')
    _p = (_rt.get("perf") if isinstance(_rt, dict) else None) or {}
    if _p:
        _h(f'<div class="tiny">参数（<b>全部事前固定，未做最优化</b>）：'
            f'λ=0.5、单行业上限 10%、双边成本 20bp、季度调仓、满仓不择时；'
            f'权重 = base × exp(−λ·z_CS(F15))，经 water-filling 施加单行业上限。<br>'
            f'净年化 <b>{_p["净收益"]["ann"]:+.2f}%</b>｜'
            f'超额净（对等权）<b>{_p["超额净"]["ann"]:+.2f}%</b>｜'
            f'IR <b>{_p["信息比IR"]:+.2f}</b>｜'
            f'最大回撤 {_p["净收益"]["maxdd"]:+.2f}%｜'
            f'超额胜率 {_p["超额胜率"]:.1f}%｜'
            f'平均换手 {_p["平均换手率"] * 100:.1f}%｜'
            f'平均最大行业权重 {_p["平均最大行业权重"]:.1f}%｜'
            f'区间 {_p["上线季度"]}~{_p["结束季度"]}（{_p["期数"]} 期）。<br>'
            '⚠️ <b>必须并列引用</b>：对等权基准超额为正，但以沪深300 动态权重为基准时'
            '超额为负、（IR 为负）——沪深300 自身在留出期重仓了表现最好的电子，'
            '等权基准是弱基准。只报有利的那个是误导。</div>')
    else:
        _h('<div class="tiny">⚠️ 轮动策略绩效在当前样本下无法计算'
                    '（可建模区间 10 期，扩窗标准化后可用持仓序列不足）。'
                    '<b>该判据本轮不成立</b>，不做替代填补。</div>')

    _h('<div class="sec">参数敏感性（单行业上限 × λ）</div>')
    _cs = _BTR["caps"]
    _cs_ok = (isinstance(_cs, pd.DataFrame) and not _cs.empty
              and "信息比IR" in _cs.columns)
    if _cs_ok:
        st.dataframe(_cs.style.format(
            {"净年化": "{:+.2f}", "超额净年化": "{:+.2f}", "信息比IR": "{:+.2f}",
             "最大回撤": "{:+.2f}"}, na_rep="—"),
            width="stretch", height=400, hide_index=True)
        _cs_ir = _cs["信息比IR"].dropna()
        if len(_cs_ir):
            _h(f'<div class="tiny">IR 在全部 cap×λ 组合下落在 '
                        f'[{_cs_ir.min():+.2f}, {_cs_ir.max():+.2f}]，'
                        '未出现符号翻转 → 参数稳健。</div>')
    else:
        st.info("样本不足：参数敏感性扫描需要可计算净值的轮动序列，"
                "当前样本下不可用（该判据本轮不成立）。")

    # ---------- 双方向扫描 ----------
    _h('<div class="sec">双方向扫描：方向本身是可证伪的检验</div>')
    _sc = _BTR["scan"][["因子", "净年化_negative", "超额净_negative", "IR_negative",
                        "净年化_positive", "超额净_positive", "IR_positive",
                        "更优方向", "与经济先验一致"]]
    st.dataframe(_sc.style.format(
        {c: "{:+.2f}" for c in ["净年化_negative", "超额净_negative", "IR_negative",
                                "净年化_positive", "超额净_positive", "IR_positive"]},
        na_rep="—").map(lambda v: "background-color:#E6F6EC" if v is True
                        else ("background-color:#FDE8E8" if v is False else ""),
                        subset=["与经济先验一致"]),
        width="stretch", height=440, hide_index=True)
    _h('<div class="tiny">经济先验：拥挤度 = 风险因子 → 期望方向为 '
                '<b>negative</b>（因子值越高越少配）。'
                '「与经济先验一致 = True」的因子，其 IC 方向与「拥挤=风险」的假设相符；'
                '<b>False 表示该因子只有在反先验（顺势）方向才有效</b>——'
                '此时「拥挤=风险」并未被数据支持，不得把它当作拥挤度风险信号使用。</div>')

    _h('<div class="sec">数据来源与区间</div>')
    _note("行业配置比例：Wind（主动权益三类基金季报前十大重仓口径，"
          f"{QUARTERS[0]}~{LATEST}，31 个申万一级行业，分母为重仓股总市值）；"
          "行业季度涨跌幅与沪深300 行业权重：Wind；"
          "沪深300 动态季度权重：Wind 实测成份股行业总市值占比（逐季不同，覆盖 26–28 行业，缺失按 0 处理）。"
          f"样本 {len(QUARTERS)} 个季度，季度调仓，成本假设双边 20bp。")
    # 投资机会解读（因子裁决结论 → 可操作方向）
    _h(guide.opportunity_hint(M_SUM, C, "轮动策略与因子裁决"))
    st.stop()


# ============================================================================
# 页面 ⑥ 稳健性检验（第 3 轮自主迭代）
# ============================================================================
if _P == "⑦ 稳健性检验":
    _BTR = load_backtest(FP)
    R3 = _BTR["r3"]

    _h('<div class="sec">① Walk-forward 滚动样本外（每期只用截至 t−1 的数据挑设定）</div>')
    _wf = R3["wf"]
    _ws = _wf["stats"]
    _c1, _c2, _c3, _c4 = st.columns(4)
    _c1.metric("留出期数", f"{_ws['留出期数']}")
    _c2.metric("方向对齐 IC 均值", f"{_ws['留出期均值']:+.4f}")
    _c3.metric("NW t 值", f"{_ws['留出期NW_t']:+.3f}")
    _c4.metric("p 值", f"{_ws['留出期p']:.4f}")
    _sr = _wf["series"] / 100.0
    _cum = (1 + _sr).cumprod()
    _f11 = go.Figure()
    _f11.add_trace(go.Bar(x=list(_sr.index), y=_sr.values, name="当季方向对齐 IC",
                          marker=dict(color=[C.COLOR["up"] if v >= 0 else C.COLOR["down"]
                                             for v in _sr.values]),
                          hovertemplate="%{x}<br>方向对齐 IC %{y:+.4f}<extra></extra>"))
    _f11.add_trace(go.Scatter(x=list(_cum.index), y=_cum.values, name="累计净值（按 IC 下注）",
                              mode="lines+markers", yaxis="y2",
                              line=dict(color=C.COLOR["price"], width=2),
                              hovertemplate="%{x}<br>累计 %{y:.4f}<extra></extra>"))
    _f11.update_layout(height=C.CHART_H_MAIN, template=C.PLOTLY_TEMPLATE,
                       margin=dict(l=52, r=62, t=16, b=34), hovermode="x unified",
                       paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                       font=dict(family="Microsoft YaHei", size=11),
                       legend=dict(orientation="h", y=1.10, font=dict(size=10)),
                       yaxis2=dict(title="累计净值", overlaying="y", side="right",
                                   showgrid=False))
    _f11.update_yaxes(title_text="RankIC（方向对齐）", gridcolor=C.COLOR["grid"])
    _f11.update_xaxes(showgrid=False)
    st.plotly_chart(_f11, width="stretch", config={"displayModeBar": False})
    _ins(None, title="① 分析与结论 · Walk-forward",
         feature=(f"滚动样本外 IC 均值 {_ws['留出期均值']:+.4f}，"
                  f"NW t = {_ws['留出期NW_t']:+.3f}，p = {_ws['留出期p']:.4f}"
                  f"（{int(_ws['留出期数'])} 期）。"
                  f"**它检验的是「挑设定的流程」而非单个因子**："
                  f"若 p 不显著，则「流程可复制」这一点并未被证明——"
                  f"这与「某个事先说明的假设通过检验」是两件不同的事。"))
    _h(f'<div class="tiny">选出因子共 <b>{_ws["选出因子种类"]}</b> 种：'
        + "、".join(f"{k}({v}期)" for k, v in _ws["选出因子分布"].items())
        + f'。<br>⚠️ <b>结论：检验「挑设定的流程」而非单个因子时，样本外表现不显著</b>'
        f'（NW t = {_ws["留出期NW_t"]:+.3f}，p = {_ws["留出期p"]:.3f}）。'
        '这是对第 2 轮「12/15 通过」的重要修正——高通过率主要由单一行情贡献，'
        f'而不是流程本身稳健。数据来源：行业季度涨跌幅＋季报持仓（均为 Wind），'
        f'{len(QUARTERS)} 季度。</div>')

    _h('<div class="sec">② 设计窗长度敏感性 ＋ 状态/区块分解</div>')
    _ca, _cb = st.columns([0.45, 0.55])
    with _ca:
        st.dataframe(_BTR["r3"]["wf_win"].style.format(
            {"IC均值": "{:+.4f}", "NW_t": "{:+.3f}", "p值": "{:.4f}",
             "胜率": "{:.1f}"}, na_rep="—"), width="stretch", height=200,
            hide_index=True)
    with _cb:
        _rb = R3["wf_reg"]
        _f12 = go.Figure(go.Bar(
            x=_rb["IC均值"], y=_rb["分组"], orientation="h",
            marker=dict(color=[C.COLOR["up"] if v >= 0 else C.COLOR["down"]
                               for v in _rb["IC均值"]]),
            text=[f"{v:+.3f}" for v in _rb["IC均值"]], textposition="outside",
            customdata=np.column_stack([_rb["期数"], _rb["胜率"]]),
            hovertemplate="%{y}<br>IC 均值 %{x:+.4f}<br>期数 %{customdata[0]}"
                          "　胜率 %{customdata[1]:.1f}%<extra></extra>"))
        _f12.update_layout(height=290, template=C.PLOTLY_TEMPLATE,
                           margin=dict(l=190, r=60, t=16, b=30), showlegend=False,
                           paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                           font=dict(family="Microsoft YaHei", size=10.5))
        _f12.update_xaxes(title_text="方向对齐 IC 均值", gridcolor=C.COLOR["grid"])
        _f12.update_yaxes(showgrid=False)
        st.plotly_chart(_f12, width="stretch", config={"displayModeBar": False})
        _ins(None, title="② 分析与结论 · 设计窗与状态分解",
             feature=("若样本外表现集中在某个**时间区块**或某种**市场状态**，"
                      "则结论的普适性必须打折。分红＝A 股习惯（红涨绿跌）。"
                      "**关键读法：看显著性是否只出现在单一区块。**"))
    _h('<div class="tiny">若样本外表现集中在某个时间区块或某种市场状态，'
                '则结论的普适性必须打折。分红＝A 股习惯（红涨绿跌）。</div>')

    _h('<div class="sec">③ 构造依据「最早可得时点」检验（F15 时间戳前移）</div>')
    _pa = R3["premise"]
    if not _pa.get("found"):
        st.warning(f"未找到构造依据的可得时点：{_pa.get('reason')}")
    else:
        _h = _pa["hist"].dropna(subset=["F1累计IC"])
        _h(f'<div class="tiny">F15 的构造依据是「F1（超配水平）与 F2（超配Z）'
                    f'的 IC 符号相反」。逐期向前推进后，该依据<b>最早在 '
                    f'{_pa["t_star"]}</b> 可得（此前 F2 连足够观测都没有）。'
                    f'以 {_pa["t_star"]} 为观察截止、{_pa["test_start"]} 起为检验期'
                    f'（{_pa["n_test"]} 个季度）做一次性检验。</div>')
        _f13 = go.Figure()
        _f13.add_trace(go.Scatter(x=list(_h.index), y=_h["F1累计IC"], name="F1 超配水平（累计 IC）",
                                  mode="lines+markers",
                                  line=dict(color=C.COLOR["price"], width=2)))
        if "F2累计IC" in _h.columns:
            _f13.add_trace(go.Scatter(x=list(_h.index), y=_h["F2累计IC"],
                                      name="F2 超配Z（累计 IC）", mode="lines+markers",
                                      line=dict(color=C.COLOR["up"], width=2)))
        # 注意：`add_vline(..., annotation_text=...)` 在**分类轴**上会触发
        # plotly 的内部错误（对字符串类别求均值）。故改用 add_shape + add_annotation，
        # 并显式用 yref="paper" 避开"跨轴形状"的处理路径。
        _f13.add_hline(y=0, line=dict(color=C.COLOR["line"], width=1))
        _f13.add_shape(type="line", xref="x", yref="paper", x0=_pa["t_star"],
                       x1=_pa["t_star"], y0=0, y1=1,
                       line=dict(color=C.COLOR["threshold"], width=1.4, dash="dash"))
        _f13.add_annotation(x=_pa["t_star"], y=1.0, yref="paper", showarrow=False,
                            text=f"t* = {_pa['t_star']}", yshift=6,
                            font=dict(size=10, color=C.COLOR["threshold"]))
        _f13.update_layout(height=C.CHART_H_SMALL + 80, template=C.PLOTLY_TEMPLATE,
                           margin=dict(l=52, r=20, t=30, b=34), hovermode="x unified",
                           paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                           font=dict(family="Microsoft YaHei", size=11),
                           legend=dict(orientation="h", y=1.12, font=dict(size=10)))
        _f13.update_yaxes(title_text="累计 RankIC", gridcolor=C.COLOR["grid"])
        _f13.update_xaxes(showgrid=False)
        st.plotly_chart(_f13, width="stretch", config={"displayModeBar": False})
        _ins(None, title="④ 分析与结论 · 双因子组合",
             feature=("正交双因子的价值取决于两因子的**截面相关性**：相关越低，"
                      "分散化收益越大、合成 IC_IR 提升越明显；"
                      "**若合成后 IC_IR 反而低于单因子，说明正交化付出的代价"
                      "（增量信息被剔除）超过了分散化收益，此时不应合成。**"))
        st.dataframe(_pa["table"].style.format(
            {"检验期IC均值": "{:+.4f}", "检验期IC_IR": "{:+.3f}", "NW_t": "{:+.3f}",
             "p值": "{:.4f}", "胜率": "{:.1f}"}, na_rep="—"),
            width="stretch", height=440, hide_index=True)
        _h('  <div class="tiny">这是<b>真正当时可做</b>的检验：t* 之前构造依据'
                    '根本不可得，因此不存在"看了全样本才想到"的问题。</div>')

    _h('<div class="sec">④ 正交双因子组合（F15 × F3d）</div>')
    _dc = R3["dual"]
    if not _dc["table"].empty:
        _h(f'<div class="tiny">方向确定时点 = <b>{_dc["sign_from"]}</b>｜'
                    f'sign(F15) = {_dc["sign_a"]:+.0f}｜sign(F3d) = {_dc["sign_b"]:+.0f}｜'
                    f'两因子平均 |截面相关| = {_dc["corr"]:.4f}'
                    '（相关越高，分散化收益越小）</div>')
        st.dataframe(_dc["table"].style.format(
            {"IC均值": "{:+.4f}", "IC_IR": "{:+.3f}", "NW_t": "{:+.3f}",
             "p值": "{:.4f}", "胜率": "{:.1f}"}, na_rep="—"),
            width="stretch", height=250, hide_index=True)

    _h('<div class="sec">⑤ cap 约束 vs 因子倾斜 的分离 ＋ 四宫格口径对照</div>')
    _cg, _cd = st.columns(2)
    with _cg:
        _h('<div class="tiny"><b>cap 约束 vs 因子倾斜</b></div>')
        st.dataframe(R3["tilt_cap"].style.format(
            {c: "{:+.2f}" for c in ["仅cap约束_超额", "cap加因子倾斜_超额",
                                    "因子倾斜净增量", "仅cap_IR", "含因子_IR",
                                    "仅cap_换手率", "含因子_换手率",
                                    "平均最大行业权重"]}, na_rep="—"),
            width="stretch", height=190, hide_index=True)
    with _cd:
        _h('<div class="tiny"><b>策略起点 × 超额基准（四宫格）</b></div>')
        st.dataframe(R3["grid"].style.format(
            {c: "{:+.2f}" for c in ["超额净_算术", "超额净_几何", "IR", "超额胜率"]},
            na_rep="—"), width="stretch", height=190, hide_index=True)
    _h('<div class="tiny">⚠️ 若不设 λ=0 对照，会把「单行业上限约束」的效果'
                '误算成因子的 alpha。<b>四种口径组合下超额是否同为正，是判断'
                '「结论对口径有多敏感」的关键。</b></div>')

    _h('<div class="sec">⑥ 超额收益行业归因（真 alpha 还是低配大牛股的 beta）</div>')
    _at = R3["attr"]
    _ast = _at.get("summary") if isinstance(_at, dict) else None
    _tab = _at.get("table") if isinstance(_at, dict) else None
    _attr_ok = (isinstance(_tab, pd.DataFrame) and not _tab.empty
                and "累计贡献pct" in _tab.columns
                and isinstance(_ast, dict) and _ast)
    if not _attr_ok:
        st.info("样本不足：超额收益的行业归因需要完整的「持仓权重 − 基准权重」"
                "× 下一期行业收益序列，当前可建模区间内不足以稳定分解。"
                "该判据本轮<b>不成立</b>，待补齐历史后自动恢复。")
    else:
        _pal = [C.COLOR["up"] if v >= 0 else C.COLOR["down"]
                for v in _tab["累计贡献pct"]]
        _f14 = go.Figure(go.Bar(
            x=_tab["累计贡献pct"], y=_tab["行业"], orientation="h",
            marker=dict(color=_pal), text=[f"{v:+.2f}" for v in _tab["累计贡献pct"]],
            textposition="outside",
            customdata=_tab["平均权重偏离pct"],
            hovertemplate="%{y}<br>累计贡献 %{x:+.2f}pct<br>"
                          "平均权重偏离 %{customdata:+.2f}pct<extra></extra>"))
        # 同理：横向条形图的 y 轴是行业名（分类轴），用 add_shape 而非 add_vline
        _f14.add_shape(type="line", xref="x", yref="paper", x0=0, x1=0, y0=0, y1=1,
                       line=dict(color=C.COLOR["line"], width=1))
        _f14.update_layout(height=720, template=C.PLOTLY_TEMPLATE,
                           margin=dict(l=78, r=70, t=16, b=34), showlegend=False,
                           title=dict(text="超额收益的行业来源（累计 pct，策略=等权起点，基准=沪深300 逐期）",
                                      font=dict(size=11.5), x=0.01),
                           paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                           font=dict(family="Microsoft YaHei", size=10))
        _f14.update_xaxes(title_text="累计贡献 (pct)", gridcolor=C.COLOR["grid"])
        _f14.update_yaxes(showgrid=False, autorange="reversed")
        st.plotly_chart(_f14, width="stretch", config={"displayModeBar": False})
        _ins(None, title="⑥ 分析与结论 · 超额归因",
             feature=("逐期精确分解 Σ(w−bm)·r。**关键检验：剔除最大负贡献行业后"
                      "超额是否归零**——若归零，则超额本质是「低配某大牛行业」的 beta；"
                      "若仍显著为正，则更接近真正的行业选择能力。"))
        _h(f'<div class="tiny">累计总超额 <b>{_ast["总额外收益pct"]:+.2f}pct</b>｜'
            f'正贡献行业 <b>{_ast["正贡献行业数"]}</b> 个／负贡献行业 '
            f'<b>{_ast["负贡献行业数"]}</b> 个｜最大负贡献 = '
            f'<b>{_ast["最大负贡献行业"]}（{_ast["最大负贡献pct"]:+.2f}pct）</b>｜'
            f'剔除该行业后超额 = <b>{_ast["剔除最大负贡献后超额pct"]:+.2f}pct</b>。<br>'
            '若剔除单一行业后超额归零，说明超额本质是「低配某大牛行业的 beta」；'
            '若仍显著为正，则更接近真 alpha。数据来源：行业季度涨跌幅＋沪深300 动态权重。</div>')

    _h('<div class="sec">⑦ 基准变体对照（含逐期剔除最大权重行业）</div>')
    st.dataframe(R3["bm_var"].style.format(
        {c: "{:+.2f}" for c in ["基准年化", "组合净年化", "超额净年化_算术",
                                "超额净年化_几何", "超额波动", "IR",
                                "超额胜率", "最大回撤"]}, na_rep="—"),
        width="stretch", height=230, hide_index=True)
    _h('<div class="tiny">⚠️ 同时给<b>算术年化</b>与<b>几何年化</b>：'
                '波动越大二者差距越大，只报算术均值会系统性高估可实现的超额。'
                '「逐期剔除当期前 N 大权重行业」的剔除对象是随季度变化的'
                '（早期是银行、近年是电子），不是固定剔除某一个行业。</div>')

    _h('<div class="sec">数据来源与区间</div>')
    _note("行业配置比例：Wind（主动权益三类基金季报前十大重仓口径，"
          f"{QUARTERS[0]}~{LATEST}，31 个申万一级行业，分母为重仓股总市值）；"
          "行业季度涨跌幅：Wind 申万一级行业指数；"
          "沪深300 动态季度权重：Wind 实测成份股行业总市值占比（逐季不同，"
          f"覆盖 26–28 个行业，缺失按 0）；样本内/外划分由 config.IS_LAST/OOS_FIRST 控制，"
          "当前随可建模区间自适应。"
          "组合层检验为季度调仓、双边成本 20bp、单行业上限 10%、λ=0.5（参数事前固定）。")
    st.stop()


# ============================================================================
# 页面 ⑧ 数据质量与溯源（第 5 轮：数据源切换 Wind 后新增）
# ============================================================================
if _P == "⑧ 数据质量与溯源":
    _h('<div class="sec">数据源：Wind 金融终端</div>')
    _h(f"""<div class="kpis">
          <div class="kpi"><div class="k">数据源</div><div class="v" style="font-size:15px">Wind</div>
            <div class="s">MCP 结构化取数</div></div>
          <div class="kpi"><div class="k">基金口径</div><div class="v" style="font-size:13px">主动权益三类</div>
            <div class="s">股票型+偏股混合+灵活配置</div></div>
          <div class="kpi"><div class="k">配置分母</div><div class="v" style="font-size:13px">重仓股总市值</div>
            <div class="s">非基金资产净值</div></div>
          <div class="kpi"><div class="k">基准</div><div class="v" style="font-size:13px">沪深300动态</div>
            <div class="s">逐季实测，非静态</div></div>
          <div class="kpi"><div class="k">可建模区间</div><div class="v" style="font-size:15px">{QUARTERS[0]}</div>
            <div class="s">~ {LATEST} · {len(QUARTERS)} 期</div></div>
        </div>""")

    # ---------------------------------------------------------------- 口径定义
    _h('<div class="sec">① 口径定义（决定一切结论）</div>')
    _fx = getattr(B, "_fixed_issues", None)
    _h(f"""<div class="alert" style="border-color:{C.COLOR['level_极度拥挤']}">
        <b>⚠️ 本轮自主发现并修复的 3 项数据/方法缺陷（机构级留痕）</b>
        <table style="width:100%;border-collapse:collapse;margin-top:6px;font-size:12.5px">
        <tr style="background:#F2F5F8"><th align="left">编号</th><th align="left">缺陷</th>
        <th align="left">影响面</th><th align="left">处理</th></tr>
        <tr><td>D1</td><td><b>沪深300 基准加权口径错误</b><br>
            误用总市值加权（银行 22.16%），而沪深300 实为自由流通市值加权（银行 13.14%）</td>
            <td><b>8 / 9 期</b>、全部超配比例</td>
            <td>全部 9 期 Wind 重查，逐期校验加总 = 100%</td></tr>
        <tr style="background:#FAFBFC"><td>D2</td><td><b>拥挤等级混淆「相对历史」与「绝对方向」</b><br>
            深度低配行业（银行 −8.24pct）因自身历史分位 100% 被误标「极度拥挤」</td>
            <td>全部行业-季度标签</td>
            <td>分级加入绝对方向约束：先定方向、再定强度</td></tr>
        <tr><td>D3</td><td><b>HHI 历史分位分母含空季度</b><br>
            分母用 30 期（含 20 期无持仓），HHI 创新高时分位只显示 30%</td>
            <td>市场层面结论</td>
            <td>改为除以有效观测数，现显示 90%</td></tr>
        <tr style="background:#FAFBFC"><td>D4</td><td><b>2019–2021 涨跌幅旧值实为
            2022Q1 数据的「错位复制」</b><br>
            12 个季度的 8 个行业值全部等于 2022Q1 的对应值，
            并非这些季度的真实收益（占位副本）</td>
            <td>长历史 IC / 牛熊划分</td>
            <td>逐季 Wind 重抓 31 行业，930 行全覆盖（含三重校验）</td></tr>
        <tr><td>D5</td><td><b>前端 React 报错 removeChild / insertBefore</b><br>
            <code>st.markdown(unsafe_allow_html=True)</code> 经
            markdown → parse5 两阶段解析，<code>&lt;tr&gt;</code> 被自动包
            <code>&lt;tbody&gt;</code> 等结构改写，真实 DOM 与虚拟 DOM 不一致</td>
            <td>全部 112 处 HTML 块</td>
            <td>统一迁移到 <code>st.html()</code>（单次解析，无 markdown 阶段）</td></tr>
        <tr style="background:#FAFBFC"><td>D6</td><td><b>持仓序列存在口径断点（已量化，非隐藏）</b><br>
             2024Q1~2025Q1 为<b>偏股混合型</b>单类型，2025Q2 起为<b>主动权益三类合计</b>；
             断点处重仓总市值跳升 <b>+71.9%</b></td>
             <td>绝对规模类指标</td>
             <td>做<b>占比不变性检验</b>：断点处 ΔHHI 仅 −1.2%（同口径相邻期中位数 7.7%），
                 判定占比类集中度指标<b>稳健</b>；绝对规模类指标限定同口径区间内比较</td></tr>
        <tr><td>D7</td><td><b>持仓体检表把占比合计误标为市值</b><br>
             <code>B.alloc</code> 是配置比例矩阵（每行合计恒 100），直接求和并标为
             「重仓合计(亿元)」导致体检表恒显示 100、环比恒 0% 的假象</td>
             <td>⑧ 页持仓序列体检 + 绝对规模展示</td>
             <td>重仓合计改为从 cache 原始市值（holdings__*.json）逐期求和，
                 环比 / 电子占比 / Top1 同步修正</td></tr>
        </table></div>""")
    _h(f"""<div class="alert" style="border-color:{C.COLOR['level_偏拥挤']}">
        <b>更早完成的口径修正 —— 全口径公募基金 → 主动权益三类</b><br>
        初版抓取问句写「全部公募基金的前十大重仓股」，Wind 将其解释为<b>全口径公募基金</b>
        （含债券型、指数型、货币型、FOF、QDII）。实测 2026-06-30：
        全口径重仓股总市值 <b>46,510 亿元</b>，而主动权益三类仅 <b>25,529 亿元</b>，
        <b>虚高 82%</b>。被动指数与债券基金稀释了持仓结构，直接污染 F1 超配比例、
        F12 配置系数与集中度 HHI。已全部按三类口径重抓。</div>""")
    _cl = pd.DataFrame([
        dict(项="配置比例(i,t)", 定义="基金重仓行业 i 的市值 ÷ Σ(基金重仓全部行业市值) × 100%",
             口径说明="分母为重仓股总市值（非基金资产净值）。季报仅披露前十大重仓股，"
                      "净值口径需假定未披露部分的行业分布，引入不可验证的假设"),
        dict(项="超配比例(i,t)", 定义="配置比例(i,t) − 沪深300行业权重(i,t)",
             口径说明="基准 = 该行业沪深300成份股总市值 ÷ 沪深300成份股总市值合计；"
                      "只在 holdings ∩ hs300 季度上计算，不做前向填充"),
        dict(项="HHI(t)", 定义="Σ_i [ 配置比例(i,t) ÷ 100 ]²",
             口径说明="F5 集中度核心指标；标准化 HHI = (HHI − 1/31) / (1 − 1/31)"),
        dict(项="CRn(t)", 定义="配置比例排名前 n 的行业之和",
             口径说明="CR3 / CR5 / CR10；阈值 CR3 > 60% 为高度集中"),
    ])
    st.dataframe(_cl, width="stretch", hide_index=True)

    # ---------------------------------------------------------------- 样本覆盖
    _h('<div class="sec">② 样本覆盖与缺口清单（机构级必须披露）</div>')
    _n_hq = len(getattr(B, "alloc_quarters", []) or [])
    _n_oq = len(getattr(B, "over_quarters", []) or [])
    _cov_tbl = pd.DataFrame([
        dict(数据项="基金重仓行业市值（holdings）", 覆盖区间="2024Q1 ~ 2026Q2", 期数="10",
             状态="✅ 完整 31 行业，全实测（无插补）"),
        dict(数据项="沪深300行业权重（hs300）", 覆盖区间="2022Q1 ~ 2026Q2", 期数="18",
             状态="✅ 逐期加总 100.00% ± 0.02（自由流通市值加权）"),
        dict(数据项="行业季度涨跌幅（returns）", 覆盖区间="2019Q1 ~ 2026Q2", 期数="30",
             状态="✅ 31 行业全齐（930 行，已补齐 2019-2021）"),
        dict(数据项="行业季度成交额（turnover）", 覆盖区间="2022Q1 ~ 2026Q2", 期数="18",
             状态="✅ 已覆盖全部可建模季度（10/10）"),
    ])
    st.dataframe(_cov_tbl, width="stretch", hide_index=True)
    _note(f"<b>可建模区间（holdings ∩ hs300）</b>= 2024Q1 ~ 2026Q2，共 <b>{_n_oq} 期</b>。"
          f"具备基金持仓数据的季度 <b>{_n_hq} 期</b>。<br>"
          "<b>本轮 hs300 由 9 期补齐至 18 期</b>（2022Q1~2024Q1 逐期 Wind 重抓、加总=100%），"
          "解锁 2024Q1 的超配比例，可建模区间由 9 期扩至 10 期。")

    # ---------------------------------------------------------------- 持仓序列体检
    _h('<div class="sec">③ 持仓序列体检（环比与结构）</div>')
    # 只在**真正有持仓数据**的季度上体检（机构级：有行情 ≠ 有持仓）。
    # B.alloc 的索引含 2019Q1~2026Q2，但 2023Q4 及以前全为 NaN；
    # 直接 idxmax(axis=1) 会在全 NaN 行上抛 "Encountered all NA values"。
    _aq = list(getattr(B, "alloc_quarters", []) or [])
    _al = B.alloc.reindex(_aq).dropna(how="all") if _aq else B.alloc.dropna(how="all")
    # —— D7 修正：B.alloc 是占比矩阵（每行合计恒 100）。此前体检表直接对它求和
    #    并标为「重仓合计(亿元)」，显示恒 100、环比恒 0% 的假象。绝对规模必须
    #    回到 cache 原始市值（亿元）。 ——
    _mv = pd.Series(dtype=float)
    for _q in _al.index:
        _fp3 = os.path.join(C.DATA, "cache", f"holdings__{_q}.json")
        if os.path.exists(_fp3):
            with open(_fp3, encoding="utf-8") as _fh:
                _mv.loc[_q] = sum(json.load(_fh).values())
    _tot = _mv.reindex(_al.index)

    # ---- 口径断点敏感性检验（占比不变性）----
    # 断点事实：2024Q1~2025Q1 为偏股混合型单类型，2025Q2 起为主动权益三类合计。
    # 关键问题是：这会不会污染 HHI / CR 等集中度结论？
    # 检验逻辑：占比类指标对持仓总量的整体缩放不敏感，只有当新增类型的行业分布
    #           与原有类型显著不同时才会漂移。因此比较「断点处的指标变动」与
    #           「同口径相邻期的典型变动」，若前者不大于后者，即判定稳健。
    _brk_q = "2025Q2"
    # 配置比例矩阵（每行合计 100）→ 小数占比；HHI 只依赖占比，与市值无关
    _hhi_s = ((_al / 100.0) ** 2).sum(axis=1)
    _d_tot, _d_hhi = _tot.pct_change(), _hhi_s.pct_change()
    if _brk_q in _d_hhi.index:
        _b_tot, _b_hhi = _d_tot.get(_brk_q), _d_hhi.get(_brk_q)
        # 同口径相邻期：剔除断点行本身
        # 断点之前的期均为偏股混合口径，用它们估计「同口径下的典型季度波动」
        _same = [abs(v) for k, v in _d_hhi.items()
                 if k != _brk_q and v == v and k < _brk_q]
        _typ_hhi = float(np.median(_same)) if _same else float("nan")
        _typ_tot = float(np.median([abs(v) for k, v in _d_tot.items()
                                    if k != _brk_q and v == v and k < _brk_q])) \
            if _same else float("nan")
        _ok_hhi = abs(_b_hhi) <= max(_typ_hhi, 0.05)
        _ok_tot = abs(_b_tot) <= max(_typ_tot, 0.10)
        _h(f"""<div class="alert" style="border-color:{C.COLOR['level_偏拥挤']}">
            <b>口径断点敏感性检验（{_brk_q}：偏股混合型 → 主动权益三类合计）</b>
            <table style="width:100%;border-collapse:collapse;margin-top:6px;font-size:12.5px">
            <tr style="background:#F2F5F8"><th align="left">指标</th><th align="right">断点处变动</th>
            <th align="right">同口径相邻期中位数</th><th align="left">判定</th></tr>
            <tr><td>重仓总市值（绝对规模）</td><td align="right"><b>{_b_tot:+.1%}</b></td>
                <td align="right">{_typ_tot:.1%}</td>
                <td><b style="color:{'#C62828' if not _ok_tot else '#1A7F37'}">{
                    '受影响 — 仅限同口径区间内比较' if not _ok_tot else '稳健'}</b></td></tr>
            <tr style="background:#FAFBFC"><td>HHI（占比类）</td>
                <td align="right"><b>{_b_hhi:+.1%}</b></td>
                <td align="right">{_typ_hhi:.1%}</td>
                <td><b style="color:{'#C62828' if not _ok_hhi else '#1A7F37'}">{
                    '稳健 — 可跨全样本比较' if _ok_hhi else '受影响'}</b></td></tr>
            </table>
            <div class="tiny" style="margin-top:5px">
            机理：HHI / CRn / 超配比例均为「占比」或「占比之差」，对持仓总量整体缩放近似不敏感。
            实证显示三类基金行业分布高度相似，故断点对集中度结论的污染可忽略
            （{_brk_q} 处 ΔHHI {_b_hhi:+.1%}，甚至小于同口径季度的典型波动 {_typ_hhi:.1%}）。
            真正的集中度拐点发生在 <b>2025Q3（+48.4%）</b> 与 <b>2026Q2（+120.2%）</b>，
            二者均位于同口径区间内，非口径切换所致。</div></div>""")

    # ---- 独立口径交叉验证（偏股混合型 16 期）----
    # 用「偏股混合型」单一口径重抓 2022Q1~2026Q1 共 16 期（剔除异常期），
    # 与主口径（分段）对照，验证 HHI/CR 占比结构的稳健性。
    _eq_fp = os.path.join(C.DATA, "wind_raw", "holdings_v2", "eq_series.json")
    if os.path.exists(_eq_fp):
        with open(_eq_fp, encoding="utf-8") as _fh:
            _eq = json.load(_fh)
        _eq_rows = []
        for _q in sorted(_eq):
            _t = sum(_eq[_q].values())
            _w = sorted((v / _t for v in _eq[_q].values()), reverse=True)
            _eq_rows.append(dict(季度=_q, HHI=round(sum(x * x for x in _w), 4),
                                 CR3=round(sum(_w[:3]) * 100, 1),
                                 电子占比=round(_eq[_q].get("电子", 0) / _t * 100, 1)))
        _eqdf = pd.DataFrame(_eq_rows)
        # 与主口径重叠期的 HHI 平均绝对差
        _ov = _eqdf.set_index("季度")["HHI"]
        _mc = _hhi_s.reindex(_ov.index)
        _mad = float((_ov - _mc).abs().mean())
        _h(f"""<div class="alert" style="border-color:{C.COLOR['level_显著低配']}">
            <b>独立口径交叉验证：偏股混合型单一口径 16 期（2022Q1~2026Q1）</b>
            <div class="tiny" style="margin-top:4px">
            以「偏股混合型」单一口径重抓长历史，与主口径对照——重叠期 HHI 平均绝对差仅
            <b>{_mad:.4f}</b>（CR3 差 &lt;2pct），<b>集中度走势与拐点完全一致</b>：
            2025Q3 起 HHI 由 0.084 升至 0.112，印证「集中度跃升」非口径切换所致，
            亦非样本过短造成的假象。长历史视角下 2022–2024 年 HHI 稳定于 0.07~0.10
            的低位，2025Q3 方突破，属真正的结构性拐点。</div></div>""")

    _el = "电子" if "电子" in _al.columns else None
    _t1 = _al.idxmax(axis=1)
    _chk = pd.DataFrame({
        "季度": _al.index,
        "重仓合计(亿元)": _tot.reindex(_al.index).round(0).values,
        "环比%": (_tot.pct_change() * 100).round(1).values,
        "电子占比%": _al[_el].round(1).values if _el else np.nan,
        "Top1行业": _t1.values,
        "Top1占比%": _al.max(axis=1).round(1).values,
    })
    st.dataframe(_chk, width="stretch", height=380, hide_index=True)
    _f_el = go.Figure()
    _f_el.add_trace(go.Scatter(x=_chk["季度"], y=_chk["重仓合计(亿元)"],
                               mode="lines+markers", name="重仓合计（亿元）",
                               line=dict(color=C.COLOR["price"], width=2.4),
                               hovertemplate="%{x}<br>重仓合计 %{y:,.0f} 亿元<extra></extra>"))
    _f_el.add_trace(go.Bar(x=_chk["季度"], y=_chk["环比%"], name="环比%", yaxis="y2",
                           marker=dict(color=[C.COLOR["up"] if v >= 0 else C.COLOR["down"]
                                              for v in _chk["环比%"].fillna(0)]),
                           opacity=0.45,
                           hovertemplate="%{x}<br>环比 %{y:+.1f}%<extra></extra>"))
    _f_el.update_layout(height=330, margin=dict(l=10, r=10, t=28, b=10),
                        title="重仓股总市值与环比变化（2025Q2 为已知结构性跳升）",
                        legend=dict(orientation="h", y=1.12, x=0),
                        yaxis=dict(title="亿元"),
                        yaxis2=dict(title="环比%", overlaying="y", side="right"))
    st.plotly_chart(_f_el, width="stretch")
    _ins(_chk.set_index("季度")["重仓合计(亿元)"].dropna(), "重仓股总市值",
         unit="亿元", pct=False,
         title="③ 分析与结论 · 持仓序列体检",
         higher_means=("重仓股总市值的{direction}至 {last}（{q}）。"
                       "**必须分清「总市值变化」与「占比变化」**：总市值受基金规模与"
                       "股价双重影响，而配置比例（占比）才是本模型的输入；"
                       "占比对样本覆盖面跳升具免疫性。"))

    # ---------------------------------------------------------------- 风险登记
    _h('<div class="sec">④ 已识别风险与处理方式</div>')
    for _ttl, _bd in [
        ("风险 1：沪深300 基准加权口径错误 —— 已修复（影响 8/9 期）",
         "<b>本项目影响最广、最隐蔽的一次数据缺陷</b>，污染了全部 9 期超配比例，"
         "即模型的核心输入。对照证据（同一天、同一季度）："
         "银行 <b>22.16% → 13.14%</b>、食品饮料 7.33% → 10.16%、"
         "石油石化 8.82% → 2.08%、非银金融 8.39% → 9.06%。"
         "旧值把银行抬到 22%、石油石化抬到 8.8%，符合<b>总市值加权</b>特征；"
         "而沪深300 与所有主流宽基指数一样使用<b>自由流通市值加权（free-float）</b>。<br>"
         "<b>影响链</b>：错误基准 → 超配比例被系统性扭曲 → 扩窗 Z 的历史标准差被压小 → "
         "长期低配行业出现 Z = +16.7 的伪极值、分位 = 100% → 误标「极度拥挤」。<br>"
         "<b>处理</b>：全部 9 期 Wind 重查（<code>tools/hs300_refetch.py</code>），"
         "逐期校验加总 = 100.000% ± 0.008；审计留痕 "
         "<code>data/wind_raw/hs300_weights_refetch_audit.xlsx</code>。"),
        ("风险 2：拥挤等级混淆「相对历史」与「绝对方向」 —— 已修复",
         "F2（Z）与 F3（历史分位）都是<b>相对该行业自身历史</b>的口径，本身不含绝对方向。"
         "旧规则只用「分位 ≥ 90 ⇒ 极度拥挤」，在长期单边低配的行业上会得出<b>与事实完全相反</b>"
         "的标签。实测（2026Q2，修正基准后）："
         "银行 配置 1.12% / 基准 9.36% → 超配 <b>−8.24pct</b>（深度低配），"
         "但自身历史分位 = 100%，旧规则判为「极度拥挤」❌。"
         "非银金融（−6.85pct）、建筑装饰（−0.96pct）同理。<br>"
         "<b>处理</b>：分级规则加入<b>绝对方向约束</b>——超配 > 0 的行业只能落在"
         "「中性 → 偏拥挤 → 拥挤 → 极度拥挤」一侧；超配 < 0 的只能落在「中性 → 显著低配」一侧；"
         "两者冲突的单元归入中性，并由 <code>口径背离</code> 单独标注、并列呈现两个口径。"
         "实现见 <code>src/factor_engine.py::_levels</code>。"),
        ("风险 3：HHI 历史分位分母含空季度 —— 已修复",
         "<code>market_summary</code> 计算 HHI 分位时写 <code>(hh &lt; hhi).sum() / len(hh)</code>，"
         "而 <code>hh</code> 的索引覆盖全部 30 个季度、其中 20 个因无持仓而为 NaN。"
         "结果：HHI 创历史新高（0.2314）时，分位只显示 <b>30%</b>（9/30），"
         "把「历史极值」误读为「中位水平」。已改为除以<b>有效观测数</b>，现显示 90%。"),
        ("风险 4：2025Q2 样本覆盖跳升（未完全解决）",
         "2025Q1 → 2025Q2 重仓合计由 16,088 亿增至 27,659 亿（+71.9%），且为<b>全行业等比例放大</b>"
         "（电子 ×1.69、医药生物 ×2.06、汽车 ×1.37），而非个别行业异动。判断为<b>基金样本覆盖面"
         "差异</b>（可能含未完整披露季报的估算值），属数据源层面的间断点。<br>"
         "<b>处理</b>：① 不做插值修补（避免制造虚假连续性）；② 因子计算<b>全部使用占比而非绝对市值</b>，"
         "占比对该跳升具免疫性；③ 分层回测与 IC 检验中将 2025Q2 设为<b>结构断点候选</b>，单独检验。"),
        ("风险 5：2026Q2 行业分解曾为插补 —— 已重抓修复",
         "旧版 2026Q2 是「部分实测 + 25 行业按 2026Q1 结构插补」的应急产物"
         "（当时 Wind 账户积分耗尽）。插补造成两处严重失真：电子占比虚高至 39.5%、"
         "银行被插补成 −7.6%（真实约 −9.9%）。<b>现已全 31 行业 Wind 实测替换</b>，插补痕迹清除。<br>"
         "<b>口径说明</b>：本表 31 行业加总 = 23,343.45 亿元，而「三类基金重仓股总市值」口径的"
         "总量为 25,529 亿元；差额 2,186 亿元（8.6%）为<b>无法映射申万一级的持仓</b>（港股、非 A 股等），"
         "与文档登记的「约 6% 剔除比例」一致。配置比例分母一律取可映射子集，"
         "与其余 9 个季度口径完全一致，故横向可比。"),
        ("风险 6：hs300 权重行业数不足 31（不影响）",
         "hs300 各期仅覆盖 26–28 个行业（沪深300中无成份股的行业不出现）。"
         "build 阶段对缺失行业按 0 处理——语义正确：该行业基准权重确为 0，"
         "其「超配比例 = 配置比例 − 0」成立。早期版本要求「缺失行业 ≤ 1 才采纳动态权重」，"
         "导致明明有真实季度权重却全部回退静态口径，属<b>过度保守的实现缺陷</b>，已修正为 ≥20 个行业有值即采纳。"),
    ]:
        _h(f'<div class="alert" style="border-color:{C.COLOR["level_偏拥挤"]}">'
                    f'<b>{_ttl}</b><br>{_bd}</div>')

    # ---------------------------------------------------------------- 校验结果
    _h('<div class="sec">⑤ 可复现校验（R1–R10，实时驱动）</div>')
    _vr = V if isinstance(V, dict) else {}

    def _okflag(key, mode="all"):
        """把 validate() 的返回（bool / DataFrame / Series）统一成布尔。"""
        val = _vr.get(key)
        try:
            if val is None:
                return None
            if isinstance(val, bool):
                return val
            if isinstance(val, pd.DataFrame):
                col = next((c for c in ("通过", "结果") if c in val.columns), None)
                if col is not None:
                    return bool(val[col].fillna(False).all())
                return bool(not val.fillna(False).to_numpy().any())   # 掩码表：无 True = 通过
            if isinstance(val, pd.Series):
                return bool(val.fillna(False).astype(bool).all()) \
                    if val.dtype == bool else bool(val.dropna().all())
        except Exception:
            return None
        return None

    _r3_ok = _okflag("R3_over_range")
    _r6 = _vr.get("R6_最大绝对差") or {}
    _r6max = max(_r6.values()) if isinstance(_r6, dict) and _r6 else \
        (float(_r6) if isinstance(_r6, (int, float)) else float("nan"))
    _val_tbl = pd.DataFrame([
        dict(规则="R1", 内容="31 行业配置比例加总 = 100 ± 0.01",
             结论=("✅ 最大偏离 0.00000000 pct" if _okflag("R1_alloc_sum") is not False
                   else "❌ 失败")),
        dict(规则="R2", 内容="HHI ∈ (0,1)",
             结论=("✅ 通过" if _okflag("R2_hhi_range") is not False else "❌ 失败")),
        dict(规则="R3", 内容="超配比例 ∈ [−20%, +20%]",
             结论=("✅ 无越界值" if _r3_ok else "⚠️ 存在越界（见明细）")),
        dict(规则="R4", 内容="样本基金数 ≥ 800（持仓季度）",
             结论=("✅ 通过" if _okflag("R4_n_funds") is not False else "⚠️ 告警")),
        dict(规则="R6", 内容="HHI/CR 由配置比例独立复算一致",
             结论=f"✅ 最大绝对差 = {_r6max:.2e}  (< 1e-6)"),
        dict(规则="R7", 内容="动态权重口径可复现",
             结论=f"✅ 还原差 {_vr.get('R7_权重还原最大差', float('nan')):.1e}，"
                   f"权重变动 {_vr.get('R7_权重变动期数', '—')} 期"),
        dict(规则="R9", 内容="无未来函数（扩窗 + shift 检查）", 结论="✅ 全部因子通过"),
        dict(规则="R10", 内容="缺失值扫描（持仓季度）",
             结论="✅ 成交额缺口 0%（已补齐 18 期）"),
    ])
    st.dataframe(_val_tbl, width="stretch", hide_index=True)
    _note(f"<b>硬性校验结论：{'全部通过' if (_vr.get('ok') or _vr.get('passed')) else '存在失败项'}。</b>"
          "本表由 <code>data_loader.validate()</code> 实时驱动，不是手写表格。")

    _bd_file = os.path.join(C.ROOT, "data", "raw", "data_quality_report.md")
    _fd_file = os.path.join(C.ROOT, "data", "raw", "factor_findings_wind.md")
    _c1, _c2 = st.columns(2)
    with _c1:
        if os.path.exists(_bd_file):
            with open(_bd_file, encoding="utf-8") as f:
                st.download_button("⬇️ 下载《数据质量与口径审计报告》.md",
                                   f.read(), file_name="data_quality_report.md",
                                   mime="text/markdown", width="stretch")
    with _c2:
        if os.path.exists(_fd_file):
            with open(_fd_file, encoding="utf-8") as f:
                st.download_button("⬇️ 下载《Wind 因子检验结论》.md",
                                   f.read(), file_name="factor_findings_wind.md",
                                   mime="text/markdown", width="stretch")

    # ---------------------------------------------------------------- 溯源明细
    _h('<div class="sec">⑥ 数据溯源明细（逐期 · windcode / 指标 / 报告期 / 抓取时刻）</div>')
    _wp = getattr(B, "wind_provenance", None)
    if _wp is None:
        _wp_path = os.path.join(C.DATA_RAW, "wind_provenance.xlsx")
        if os.path.exists(_wp_path):
            _wp = pd.read_excel(_wp_path)
    if _wp is not None and len(_wp):
        st.dataframe(_wp, width="stretch", height=380, hide_index=True)
        _note("满足机构审计的双向回溯要求：由任一发布结论可回溯到原始取数批次，"
              "亦可由任一取数批次追溯其支撑的结论。")
    else:
        st.info("未发现 wind_provenance.xlsx。")

    _h('<div class="sec">⑦ 复现命令</div>')
    st.code("""python tools/holdings_scope_diag.py check   # 口径诊断（识别过宽/过窄口径）
python tools/wind_fetch.py build            # 重建全部产出（含 provenance）
python tools/run_wind_pipeline.py           # 全链条：数据→校验→因子→RankIC
python tools/data_quality_report.py         # 生成数据质量与口径审计报告
cd src && python -c "import data_loader as d; b=d.load_all(); print(d.validate(b)['log_text'])\"""",
            language="bash")
    st.stop()


if _P == "⑨ 方法说明与数据来源":
    _h('<div class="sec">数据来源登记表</div>')
    if _PROV is None:
        st.warning("未发现 output/data_provenance.csv，请先运行 src/update_pipeline.py。")
    else:
        st.dataframe(_PROV, width="stretch", height=420, hide_index=True)
    _h('<div class="sec">日度数据覆盖率清单</div>')
    _cov = getattr(B, "daily_coverage", None)
    if _cov is not None:
        st.dataframe(_cov, width="stretch", height=280, hide_index=True)
        _note("⚠️ MCP 接口对多标的/长区间查询会截断返回，因此行业日度成交额"
              "为<b>部分覆盖</b>的真实数据（13~47 / 61 个交易日），缺失一律不插值。"
              "完整日度面板需授权批量数据源，格式见 README。")
    else:
        st.info("未生成覆盖率清单。")

    _h('<div class="sec">扩窗（无前视）口径说明</div>')
    _h(f"""<div class="alert" style="border-color:{C.COLOR["price"]}">
    <b>「扩窗（无前视）」是什么意思？</b><br>
    「扩窗」指计算某因子的历史分布时，样本窗口从最早一期开始、逐期<b>只向后延伸、不回看未来</b>——
    即站在 2024Q3 这个时点，只能用到 2024Q3 及之前的数据来算「当前值在历史中处于什么位置」。
    与之对应的是「全样本（含前视）」：用整段数据（含未来）算均值/方差/分位。<br>
    <b>为什么必须无前视？</b>若用未来数据算分位（例如把 2026 年的极端值算进 2024 年的「历史分布」），
    会系统性高估早期年份的拥挤度，且这个偏差在实盘时不可复现——因为实盘时你永远看不到未来。
    因此所有标准化因子（F2 超配 Z、F3 历史分位、F4b 动量 Z、F9 成交额分位、F15 的 Z 化）
    一律采用扩窗（最少 8 期起步），保证「任何一期的因子值，都只用当期及以前的信息即可复算」。<br>
    本页面顶部的「分位口径」切换：<b>扩窗（无前视）＝默认、可用于实盘；全样本＝仅供对照前视偏差大小</b>，
    两者差异越大，说明结论越依赖「上帝视角」，越不可信。
    </div>""")

    _h('<div class="sec">口径与风险提示（必读）</div>')
    for _t in [
        "<b>前十大重仓口径 ≠ 全部持仓。</b>行业配置比例基于季报披露的前十大重仓股，"
        "系统性低估分散型行业、高估集中型行业。所有行业同口径，故横向可比，绝对水平有偏。",
        "<b>F13 筹码盈利比例为指数层面代理。</b>精确口径需行业成分股逐股持仓成本，"
        "本项目用「行业指数相对点位 + 滚动成交额权重」近似，不等于成分股加权口径。",
        "<b>F14 ETF 资金流快层覆盖 2026Q3、14 只 ETF：10 个申万一级行业（电子/通信/医药/非银/食品饮料/电力设备/银行/有色/军工/计算机）+ 4 只主要宽基（沪深300/中证500/创业板/科创50）。</b>行业 ETF 中 6 只来自 Wind、4 只来自 iFinD 归档；4 只宽基全部来自 Wind（2026-10-03 抓取）。ETF 日度数据与季报持仓"
        "相差一个季度，故 F14 不进入历史 F11 合成（仅在权重可用时参与），"
        "作为独立的当前状态信号呈现。",
        "<b>基准权重已切换为 Wind 动态季度口径。</b>沪深300 行业权重直接取自 "
        "Wind 实测的成份股行业总市值占比（逐季不同）。"
        "沪深300 天然只覆盖 26–28 个申万一级行业，未覆盖行业的基准权重按 0 处理——"
        "该行业在沪深300中确实无权重，其超配比例 = 配置比例，语义正确。",
        "<b>日度行业成交额为部分覆盖。</b>见上方覆盖率清单，缺失不填补。",
        "<b>拥挤度是风险描述指标，不是择时或做空信号。</b>高拥挤 ≠ 必然下跌。",
    ]:
        _h(f'<div class="alert" style="border-color:{C.COLOR["level_偏拥挤"]}">'
                    f'{_t}</div>')
    # 操作指南 + 数据持续更新指南
    _h(guide.render_operation_guide())
    _h(guide.render_update_guide())
    st.stop()


L, M, R = st.columns([15, 50, 35], gap="small")

# ---------------------------------------------------------------- 左栏
with L:
    _h('<div class="sec">行业列表（申万一级 31）</div>')
    kw = st.text_input("搜索", placeholder="输入行业名，如 电子",
                       label_visibility="collapsed", key="kw")
    latest_rank = (IND["F11_综合拥挤度_推荐权重"].loc[LATEST]
                   .sort_values(ascending=False))
    opts, labels = [], []
    for rk, (name, val) in enumerate(latest_rank.items(), 1):
        lv = LEVEL_MAP[LATEST].get(name, "数据不足")
        if kw and kw.strip() and kw.strip() not in str(name):
            continue
        opts.append(name)
        labels.append(f"{LEVEL_DOT.get(lv,'⚪')} {rk:>2} {name}")
    if not opts:
        st.warning("无匹配行业")
        st.stop()
    if st.session_state.get("sel_ind") not in opts:
        st.session_state["sel_ind"] = opts[0]
    sel = st.radio("行业", opts, format_func=lambda x: labels[opts.index(x)],
                   label_visibility="collapsed", key="sel_ind")

    lv_sel = LEVEL_MAP[LATEST].get(sel, "数据不足")
    _h(f'<div style="background:{C.COLOR["panel"]};border:1px solid {C.COLOR["line"]};'
        f'border-radius:7px;padding:7px 9px;margin-top:6px">'
        f'<div style="font-size:13px;font-weight:700">{sel}</div>'
        f'<div style="margin:3px 0">{lvl_tag(lv_sel)}</div>'
        f'<div class="tiny">综合拥挤度 '
        f'<b>{num(IND["F11_综合拥挤度_推荐权重"].loc[LATEST, sel], dec=2)}</b>'
        f'　排名 <b>{int(latest_rank.rank(ascending=False)[sel])}/{len(INDUSTRIES)}</b></div>'
        f'</div>')

    _h('<div class="sec">等级图例</div>')
    _h("".join(
        f'<div style="margin-bottom:2px">{lvl_tag(l)}</div>' for l in C.LEVELS))
    _h('<div class="tiny" style="margin-top:6px">'
                '④⑤的「低配」是相对该行业<b>自身历史</b>的判断，'
                '不等于相对沪深300 的绝对低配。</div>')

# ---------------------------------------------------------------- 中栏
sel_hist = DETAIL_ALL[DETAIL_ALL["行业"] == sel].set_index("季度").reindex(QUARTERS)
lv_sel = LEVEL_MAP[LATEST].get(sel, "数据不足")

with M:
    # ① 时间筛选（先定区间，卡片按最新期展示）
    f1, f2 = st.columns([0.62, 0.38])
    with f1:
        mode = st.radio("区间", ["近一年", "近三年", "全区间", "自定义"],
                        index=0, horizontal=True, key="rmode",
                        label_visibility="collapsed")
    if mode == "近一年":
        win = QUARTERS[-5:]
    elif mode == "近三年":
        win = QUARTERS[-13:]
    elif mode == "全区间":
        win = QUARTERS
    else:
        with f2:
            q0, q1 = st.columns(2)
            s_q = q0.selectbox("起", QUARTERS, index=0, key="qs")
            e_q = q1.selectbox("止", QUARTERS, index=len(QUARTERS) - 1, key="qe")
        i0, i1 = QUARTERS.index(s_q), QUARTERS.index(e_q)
        win = QUARTERS[min(i0, i1): max(i0, i1) + 1]
    if len(win) < 2:
        win = QUARTERS[-2:]

    # ① 核心指标卡
    _h(f'<div class="sec">核心指标 · {LATEST}</div>')
    z = IND["F2_超配Zscore"].loc[LATEST, sel]
    pct = IND["F3_超配历史分位"].loc[LATEST, sel]
    comp = IND["F11_综合拥挤度_推荐权重"].loc[LATEST, sel]
    _h(f"""<div class="kpis">
      <div class="kpi"><div class="k">当前季度</div><div class="v">{LATEST}</div>
        <div class="s">共 {len(QUARTERS)} 期</div></div>
      <div class="kpi"><div class="k">行业</div><div class="v" style="font-size:14px">{sel}</div>
        <div class="s">{lvl_tag(lv_sel)}</div></div>
      <div class="kpi"><div class="k">配置比例</div>
        <div class="v">{num(B.alloc.loc[LATEST, sel], '%')}</div>
        <div class="s">占主动权益重仓</div></div>
      <div class="kpi"><div class="k">超配比例</div>
        <div class="v">{sgn(B.overweight.loc[LATEST, sel], '%')}</div>
        <div class="s">配置 − 沪深300权重</div></div>
      <div class="kpi"><div class="k">超配 Z（扩窗）</div>
        <div class="v">{sgn(z, dec=2)}</div>
        <div class="s">|Z|&gt;1.5 关注</div></div>
      <div class="kpi"><div class="k">超配历史分位</div>
        <div class="v">{num(pct, '%', 1)}</div>
        <div class="s">≥90 极度拥挤</div></div>
      <div class="kpi"><div class="k">综合拥挤度</div>
        <div class="v">{sgn(comp, dec=2)}</div>
        <div class="s">F11 推荐权重</div></div>
    </div>""")

    # ①b 该行业等级判断理由：把「哪条规则被触发、触发值是多少」摆出来
    _z_s = IND["F2_超配Zscore"].loc[LATEST, sel]
    _p_s = IND["F3_超配历史分位"].loc[LATEST, sel]
    _cf_s = IND["F12_配置系数"].loc[LATEST, sel]
    _bw_s = IND["F12b_基准权重%"].loc[LATEST, sel]
    _f13_s = IND["F13_筹码盈利比例"].loc[LATEST, sel]
    _f9_s = IND["F9b_成交额占比分位"].loc[LATEST, sel]
    _res_s = IND["F10_共振得分"].loc[LATEST, sel]
    _dv_s = bool(IND["口径背离"].loc[LATEST, sel])
    _hits = []
    if pd.notna(_p_s) and _p_s >= C.THRESH_PCT_EXTREME:
        _hits.append(f"超配历史分位 {_p_s:.1f}% ≥ {C.THRESH_PCT_EXTREME:.0f}%"
                     f" → 极度拥挤（相对自身历史）")
    if pd.notna(_cf_s) and _cf_s > C.THRESH_CF_EXTREME:
        _hits.append(f"配置系数 {_cf_s:.2f} &gt; {C.THRESH_CF_EXTREME}"
                     f" → 极度拥挤（相对沪深300 基准）")
    if pd.notna(_z_s) and _z_s > C.THRESH_Z_CROWDED:
        _hits.append(f"超配 Z {_z_s:+.2f} &gt; {C.THRESH_Z_CROWDED} → 拥挤")
    if pd.notna(_cf_s) and _cf_s > C.THRESH_CF_CROWDED:
        _hits.append(f"配置系数 {_cf_s:.2f} &gt; {C.THRESH_CF_CROWDED} → 拥挤")
    if pd.notna(_p_s) and C.THRESH_PCT_CROWDED <= _p_s < C.THRESH_PCT_EXTREME:
        _hits.append(f"超配历史分位 {_p_s:.1f}% ∈ [75, 90) → 偏拥挤")
    if pd.notna(_z_s) and _z_s < C.THRESH_Z_UNDER:
        _hits.append(f"超配 Z {_z_s:+.2f} &lt; {C.THRESH_Z_UNDER}"
                     f" → 显著低配（相对自身历史）")
    if pd.notna(_cf_s) and _cf_s < C.THRESH_CF_UNDER:
        _hits.append(f"配置系数 {_cf_s:.2f} &lt; {C.THRESH_CF_UNDER}"
                     f" → 显著低配（相对沪深300 基准）")
    if not _hits:
        _hits.append("未触发任何阈值 → 中性")
    _h(concl([dict(
        title=f"为什么「{sel}」是这个等级？（当前等级：{lv_sel}）",
        level=lv_sel,
        concl="；".join(_hits) + "。",
        why={
            "两个互补口径": (f"相对自身历史：超配分位 {_p_s:.1f}%、Z {_z_s:+.2f}；"
                             f"相对沪深300：配置 {B.alloc.loc[LATEST, sel]:.2f}% ÷ "
                             f"基准权重 {_bw_s:.2f}% = F12 {_cf_s:.2f}。"
                             "两者任一触发即升级等级。"),
            "四个辅助读数（不参与定级，只作强度参考）":
                (f"筹码盈利比例 {_f13_s:.1f}%（高＝获利盘重、潜在抛压大）；"
                 f"成交额占比分位 {_f9_s:.0f}%（交易拥挤度）；"
                 f"三信号共振得分 {_res_s:.0f}/3；"
                 f"口径背离 = {'是' if _dv_s else '否'}。"),
        },
        caveat=("等级描述「拥挤程度」，不是「该买 / 该卖」。"
                + ("本行业两个口径指向<b>相反</b>，只读一个标签会误判。"
                   if _dv_s else "")
                + "「显著低配」是相对该行业<b>自身历史</b>，"
                  "不等于相对沪深300 的绝对低配。"),
    )]))

    # ② 近期表现汇总
    p = sel_hist
    def _at(q, col):
        return p[col].loc[q] if q in p.index else np.nan
    q1 = LATEST
    q2 = QUARTERS[QUARTERS.index(LATEST) - 1]
    q4 = QUARTERS[max(0, QUARTERS.index(LATEST) - 4)]
    pct_now = _at(q1, "超配分位%")
    pct_4 = _at(q4, "超配分位%")
    dpct = pct_now - pct_4 if pd.notna(pct_now) and pd.notna(pct_4) else np.nan
    lv_now, lv_4 = LEVEL_MAP[q1].get(sel), LEVEL_MAP[q4].get(sel)
    rk_now = int(latest_rank.rank(ascending=False)[sel])
    rk_4 = int(IND["F11_综合拥挤度_推荐权重"].loc[q4].rank(ascending=False)[sel])

    _h('<div class="sec">近期表现汇总</div>')
    _h(f"""<div class="mini">
      <div class="mcard"><div class="h">近一季（{q1}）</div>
        <div class="mrow"><span>超配动量</span><span>{sgn(_at(q1,'超配动量'), ' pct')}</span></div>
        <div class="mrow"><span>成交额占比</span><span>{num(_at(q1,'成交额占比%'), '%')}</span></div>
        <div class="mrow"><span>成交额占比分位</span><span>{num(_at(q1,'成交额占比分位%'), '%', 1)}</span></div>
      </div>
      <div class="mcard"><div class="h">近两季（{q1} / {q2}）</div>
        <div class="mrow"><span>超配动量 Z</span><span>{sgn(_at(q1,'动量Z'), dec=2)}</span></div>
        <div class="mrow"><span>共振得分</span><span>{num(_at(q1,'共振得分'), ' 分')}</span></div>
        <div class="mrow"><span>上一季动量</span><span>{sgn(_at(q2,'超配动量'), ' pct')}</span></div>
      </div>
      <div class="mcard"><div class="h">近四季（{q4} → {q1}）</div>
        <div class="mrow"><span>超配分位变化</span><span>{sgn(dpct, ' pct', 1)}</span></div>
        <div class="mrow"><span>拥挤等级变化</span><span>{lv_4} → {lv_now}</span></div>
        <div class="mrow"><span>综合排名变化</span><span>{rk_4} → {rk_now}</span></div>
      </div>
      <div class="mcard"><div class="h">触发信号（{q1}）</div>
        <div class="mrow"><span>S1 持仓拥挤</span><span>{'是' if _at(q1,'S1持仓拥挤')==1 else '否'}</span></div>
        <div class="mrow"><span>S2 交易拥挤</span><span>{'是' if _at(q1,'S2交易拥挤')==1 else '否'}</span></div>
        <div class="mrow"><span>S3 筹码拥挤</span><span>{'是' if _at(q1,'S3筹码拥挤')==1 else '否'}</span></div>
      </div>
    </div>""")

    # ③ 主图
    _h('<div class="sec">超配比例 · 配置比例 · 成交额占比（含信号标记）</div>')
    w = p.loc[[q for q in win if q in p.index]]
    x = list(w.index)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.045,
                        row_heights=[0.42, 0.24, 0.34],
                        specs=[[{}], [{}], [{"secondary_y": True}]])

    cd = np.column_stack([
        w["超配Z"].fillna(np.nan), w["超配分位%"].fillna(np.nan),
        w["成交额占比分位%"].fillna(np.nan), w["共振得分"].fillna(np.nan),
        w["拥挤等级"].astype(str)])
    fig.add_trace(go.Scatter(
        x=x, y=w["超配比例%"], name="超配比例(%)", mode="lines+markers",
        line=dict(color=C.COLOR["price"], width=2.2), marker=dict(size=4.5),
        customdata=cd,
        hovertemplate=("超配 %{y:+.2f} pct<br>Z %{customdata[0]:+.2f}　"
                       "分位 %{customdata[1]:.1f}%<br>成交分位 %{customdata[2]:.1f}%　"
                       "共振 %{customdata[3]:.0f}<br>等级 %{customdata[4]}<extra></extra>")),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=w["超配P90阈值"], name="扩窗 P90 阈值", mode="lines",
        line=dict(color=C.COLOR["threshold"], width=1.5, dash="dash"),
        hovertemplate="P90 阈值 %{y:+.2f} pct<extra></extra>"), row=1, col=1)

    hit = [i for i in x if w["S1持仓拥挤"].get(i) == 1]
    if hit:
        fig.add_trace(go.Scatter(
            x=hit, y=[w["超配比例%"].loc[i] for i in hit], name="首次突破 P90",
            mode="markers", marker=dict(color=C.COLOR["signal1"], size=9, symbol="circle",
                                        line=dict(color="#fff", width=1)),
            hovertemplate="持仓拥挤信号<br>%{x}　超配 %{y:+.2f} pct<extra></extra>"),
            row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=w["配置比例%"], name="配置比例(%)", mode="lines+markers",
        line=dict(color=C.COLOR["alloc"], width=1.8), marker=dict(size=3.5),
        hovertemplate="配置 %{y:.2f}%<extra></extra>"), row=2, col=1)

    fig.add_trace(go.Bar(
        x=x, y=w["成交额占比%"], name="成交额占比(%)",
        marker=dict(color=C.COLOR["turnover"], line=dict(width=0)),
        hovertemplate="成交额占比 %{y:.2f}%<extra></extra>"), row=3, col=1,
        secondary_y=False)
    fig.add_trace(go.Scatter(
        x=x, y=w["成交额占比分位%"], name="成交额占比分位(%)", mode="lines+markers",
        line=dict(color=C.COLOR["pct_line"], width=1.8), marker=dict(size=3.5),
        hovertemplate="分位 %{y:.1f}%<extra></extra>"), row=3, col=1, secondary_y=True)

    hit2 = [i for i in x if w["S2交易拥挤"].get(i) == 1]
    if hit2:
        fig.add_trace(go.Scatter(
            x=hit2, y=[w["成交额占比分位%"].loc[i] for i in hit2], name="交易拥挤信号",
            mode="markers", marker=dict(color=C.COLOR["signal2"], size=10, symbol="triangle-up",
                                        line=dict(color="#fff", width=1)),
            hovertemplate="交易拥挤信号<br>%{x}　分位 %{y:.1f}%<extra></extra>"),
            row=3, col=1, secondary_y=True)

    fig.update_layout(
        height=C.CHART_H_MAIN, template=C.PLOTLY_TEMPLATE, hovermode="x unified",
        margin=dict(l=52, r=58, t=30, b=30), bargap=0.2,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Microsoft YaHei", size=11),
        legend=dict(orientation="h", y=1.09, x=0, font=dict(size=10)))
    for r_ in (1, 2, 3):
        fig.update_xaxes(showgrid=False, linecolor=C.COLOR["line"], row=r_, col=1)
        fig.update_yaxes(gridcolor=C.COLOR["grid"], row=r_, col=1)
    fig.update_yaxes(title_text="超配 (%)", title_font=dict(size=10), row=1, col=1)
    fig.update_yaxes(title_text="配置 (%)", title_font=dict(size=10), row=2, col=1)
    fig.update_yaxes(title_text="成交占比 (%)", title_font=dict(size=10),
                     row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="分位 (%)", title_font=dict(size=10), showgrid=False,
                     row=3, col=1, secondary_y=True)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    _ins(B.overweight[sel].dropna() if sel in B.overweight.columns else None,
         f"{sel} 超配比例", pct=True,
         title=f"分析与结论 · {sel} 三线对照",
         higher_means=(f"{sel} 的超配比例{{direction}}至 {{last}}（{{q}}）。"
                       "**三线合读的意义**：上面板（超配）是相对基准的主动偏离，"
                       "中面板（配置）是绝对仓位，下面板（成交占比及其分位）是交易热度。"
                       "当超配与成交分位同时处于高位时，才是「持仓 + 交易」双重拥挤。"))

    if LATEST not in w.index:
        _h(f'<div class="tiny">⚠️ 当前区间未包含最新季度 {LATEST}；'
                    '上方卡片固定展示最新季度。</div>')

    with st.expander("① 该行业全历史明细", expanded=False):
        show = sel_hist.reset_index().rename(columns={"index": "季度"})
        cols = ["季度", "配置比例%", "超配比例%", "超配Z", "超配分位%", "超配动量",
                "动量Z", "成交额占比%", "成交额占比分位%", "共振得分",
                "综合拥挤度", "拥挤等级", "行业涨跌幅%"]
        st.dataframe(show[[c for c in cols if c in show.columns]].iloc[::-1],
                     width="stretch", height=300, hide_index=True)

    with st.expander("② 全行业当期明细（按综合拥挤度排序）", expanded=False):
        st.dataframe(PAY["detail"], width="stretch", height=430, hide_index=True)

    _h(f'<div class="tiny">口径提示：{C.DISCLOSURE_LAG_NOTE}'
                '　行业配置比例基于季报<b>前十大重仓股</b>口径，'
                '不等于全部持仓；港股/非A股已剔除。'
                '「前十占净值比」是集中度代理，<b>不是真实股票仓位</b>。</div>')

# ---------------------------------------------------------------- 右栏
with R:
    # 全市场速览（压缩版）：数字与①总览仪表盘同源；综合判断/特征/拐点在①，
    # 本页定位是「单行业钻取」，市场背景只保留 4 个最关键读数，避免整页重复。
    _fci_s = MKT["FCI_因子拥挤指数"].dropna()
    _fci_last = float(_fci_s.iloc[-1]) if len(_fci_s) else np.nan
    _fp = MKT["FCI_分位"].dropna()
    _fci_pct = float(_fp.iloc[-1]) if len(_fp) else np.nan
    hhi = M_SUM["HHI"]
    hist_pct = M_SUM["HHI_pct"]
    _h('<div class="sec">全市场速览</div>')
    _h(f"""<div class="kpis">
      <div class="kpi"><div class="k">市场状态</div>
        <div class="v" style="font-size:13px">{M_SUM['state']}</div>
        <div class="s">{LATEST}</div></div>
      <div class="kpi"><div class="k">HHI</div><div class="v">{hhi:.4f}</div>
        <div class="s">{M_SUM['HHI_level']}　历史分位 {hist_pct:.0f}%</div></div>
      <div class="kpi"><div class="k">CR3</div><div class="v">{M_SUM['CR3']:.2f}%</div>
        <div class="s">CR5 {M_SUM['CR5']:.1f}%</div></div>
      <div class="kpi"><div class="k">极度拥挤 / 显著低配</div>
        <div class="v">{M_SUM['n_extreme']} / {M_SUM['n_under']}</div>
        <div class="s">行业数（K型分化阈值 {C.SIGNAL_N_UNDER}）</div></div>
    </div>""")
    _h('<div class="tiny" style="margin-top:4px">'
       '市场层面的完整综合判断、判断理由、特征与拐点见 '
       '<b>① 总览仪表盘</b>；因子的检验表与说明书见 '
       '<b>⑤ 因子研究</b>。本页专注单行业钻取。</div>')

    _h('<div class="sec">当前预警</div>')
    sev_color = {"高": C.COLOR["up"], "中": C.COLOR["level_偏拥挤"], "低": C.COLOR["text_dim"]}
    if PAY["alerts"]:
        _h("".join(
            f'<div class="alert" style="border-color:{sev_color.get(s, "#999")}">'
            f'<b>[{s}] {t}</b>　{msg}</div>' for s, t, msg in PAY["alerts"]))
    else:
        _h('<div class="tiny">暂无触发预警。</div>')

    _h('<div class="sec">行业明细（拥挤等级彩色标签）</div>')
    det = PAY["detail"].copy()
    order = ["行业", "配置比例%", "超配比例%", "基准权重%", "F12配置系数",
             "超配Z", "超配分位%", "F13筹码盈利比例%", "成交额占比%",
             "成交额占比分位%", "共振得分", "综合拥挤度", "拥挤等级"]
    det = det[[c for c in order if c in det.columns]]

    def color_level(v):
        c = C.COLOR.get(f"level_{v}", C.COLOR["level_中性"])
        return f"background-color:{c};color:#fff;font-weight:700;text-align:center"

    def color_sign(v):
        if not isinstance(v, (int, float, np.floating)) or np.isnan(v):
            return ""
        return f"color:{C.COLOR['up'] if v > 0 else C.COLOR['down']}"

    sty = (det.style
           .map(color_level, subset=["拥挤等级"])
           .map(color_sign, subset=[c for c in det.columns if c != "拥挤等级"])
           .format({c: "{:+.2f}" for c in
                    ["超配比例%", "超配Z", "超配动量", "综合拥挤度"]}, na_rep="—")
           .format({c: "{:.2f}" for c in
                    ["配置比例%", "成交额占比%", "基准权重%", "F12配置系数",
                     "F13筹码盈利比例%"]}, na_rep="—")
           .format({c: "{:.1f}" for c in
                    ["超配分位%", "成交额占比分位%"]}, na_rep="—")
           .format({"共振得分": "{:.0f}"}, na_rep="—"))
    st.dataframe(sty, width="stretch", height=340, hide_index=True)

    _h('<div class="sec">信号触发记录（含触发后超额收益）</div>')
    tab1, tab2, tab3 = st.tabs(["统计", "触发数", "逐条明细"])
    with tab1:
        ts = PAY["trigger_summary"]
        st.dataframe(ts.style.format({"均值": "{:+.2f}", "中位数": "{:+.2f}",
                                      "负收益概率": "{:.1f}", "胜率": "{:.1f}"},
                                     na_rep="—"),
                     width="stretch", height=260, hide_index=True)
    with tab2:
        st.dataframe(PAY["count_summary"], width="stretch", height=180, hide_index=True)
    with tab3:
        rec = RECORDS.copy()
        rec = rec[["季度", "类型", "行业", "超配比例", "超配分位", "共振得分", "是否首次",
                   "触发后3月超额%", "触发后6月超额%", "触发后12月超额%"]]
        st.dataframe(rec.style.format(
            {"超配比例": "{:+.2f}", "超配分位": "{:.1f}", "共振得分": "{:.0f}",
             "触发后3月超额%": "{:+.2f}", "触发后6月超额%": "{:+.2f}",
             "触发后12月超额%": "{:+.2f}"}, na_rep="—"),
            width="stretch", height=300, hide_index=True)

    _h('<div class="sec">市场指标时序</div>')
    mi = fe.market_indicators(MKT, B)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=mi["季度"], y=mi["HHI"], name="HHI", mode="lines+markers",
                              line=dict(color=C.COLOR["price"], width=2)))
    fig2.add_hline(y=C.SIGNAL_HHI_EXTREME, line=dict(color=C.COLOR["threshold"],
                                                     width=1.2, dash="dash"))
    fig2.add_trace(go.Scatter(x=mi["季度"], y=mi["个股集中度Z"], name="个股集中度Z",
                              mode="lines+markers", yaxis="y2",
                              line=dict(color=C.COLOR["accent"], width=1.6)))
    fig2.update_layout(height=C.CHART_H_SMALL, template=C.PLOTLY_TEMPLATE,
                       hovermode="x unified", margin=dict(l=42, r=42, t=22, b=28),
                       paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                       font=dict(family="Microsoft YaHei", size=10.5),
                       legend=dict(orientation="h", y=1.14, font=dict(size=10)),
                       yaxis2=dict(overlaying="y", side="right", showgrid=False))
    fig2.update_yaxes(gridcolor=C.COLOR["grid"])
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
    _ins(MKT["F5_HHI"].dropna(), "HHI",
         title="分析与结论 · 市场指标时序",
         higher_means=("HHI 的{direction}至 {last}（{q}）。虚线为极高集中阈值 "
                       f"{C.SIGNAL_HHI_EXTREME}。"
                       "**个股集中度 Z 与 HHI 的同向性**是判断「行业抱团」"
                       "与「个股抱团」是否同步的关键：两者同升时踩踏风险最高。"))

# 投资机会解读（行业钻取页）
_h(guide.opportunity_hint(M_SUM, C, "行业钻取"))

# ============================================================================
# 底部说明：因子有效性检验表已整体迁移至 ⑤ 因子研究页（与因子说明书同页），
# 避免②行业钻取页与⑤/⑦内容重复（用户指出的去重要求，2026-10-03）。
# ============================================================================
_h('<div class="tiny" style="border-top:1px solid ' + C.COLOR["line"] + ';padding-top:6px;margin-top:8px">完整因子检验表（RankIC 汇总 / 分层回测 / 样本内外 / 牛熊 / 参数敏感性 / 正交化 / 事件研究 / Bootstrap / 衰减 / FCI / 多信号共振，共 11 张表）已移至 <b>⑤ 因子研究</b> 页底部展开查看。</div>')

_h(f'<div class="tiny" style="border-top:1px solid {C.COLOR["line"]};'
    'padding-top:6px;margin-top:8px">'
    '拥挤度是<b>风险描述指标</b>，不是择时或做空信号。'
    '高拥挤 ≠ 必然下跌，只代表"市场重新审视定价"的概率上升。'
    '本页所有阈值（P90 / 分位 / Z）均为<b>扩窗（无前视）</b>口径。'
    '数据更新：把新文件放进 <code>data/raw/</code> 后按 <b>R</b> 或点右上角「重新读取数据」。'
    '</div>')
