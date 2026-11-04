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

import glob
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

# ---------------------------------------------------------------------------
# src/ 模块热重载（解决云端“入口更新了、模块还是旧的”问题）
# ---------------------------------------------------------------------------
# 现象：Streamlit 每次交互会重跑脚本，但**已 import 的模块留在 sys.modules**；
#       云端部署更新后只有 app.py 被重新执行，src/*.py 的改动不会生效，
#       曾导致 app.py 调用了新函数、而 data_loader 还是旧版 → AttributeError。
# 处理：用 src/*.py 的最大 mtime 与上次记录比较，一旦变新就把本项目模块从
#       sys.modules 里摘掉，让下面的 import 重新加载。代价仅 12 次 stat，可忽略。
# ---------------------------------------------------------------------------
_LOCAL_MODULES = ("config", "data_loader", "factor_engine", "backtest", "research",
                  "research3", "research4", "rigor", "signal_generator",
                  "insights", "guide", "factor_docs", "metric_doc", "attribution",
                  "change_analysis", "change_page", "institution",
                  "canon", "report_page", "report_charts", "insight_charts", "map_page",
                  "fund_page", "overview_page", "playbook", "glossary", "sidebar_panel",
                  "factor_page", "factor_test", "factor_doc", "method_page", "ui")


def _reload_local_modules_if_changed() -> None:
    import tempfile
    try:
        stamp = max(os.path.getmtime(os.path.join(_SRC_DIR, f))
                    for f in os.listdir(_SRC_DIR) if f.endswith(".py"))
    except Exception:
        return
    marker = os.path.join(tempfile.gettempdir(), ".fcd_src_stamp")
    try:
        with open(marker, encoding="utf-8") as fh:
            prev = float(fh.read().strip() or 0)
    except Exception:
        prev = 0.0
    if stamp > prev + 1e-6:
        for _m in _LOCAL_MODULES:
            sys.modules.pop(_m, None)
        try:
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(str(stamp))
        except Exception:
            pass


_reload_local_modules_if_changed()

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
import metric_doc as md        # noqa: E402  指标释义：定义/经济内容/投资含义/读法
import attribution as atr      # noqa: E402  配置变化归因（主动/被动）与压力测算
import change_analysis as ca   # noqa: E402  环比变化：Δ超配Z/四象限/IC/多空/成本
import change_page as cp       # noqa: E402  环比变化页渲染（结论先行）
import institution as inst     # noqa: E402  六度量/六验证/决策表/数据健康
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
                   initial_sidebar_state="expanded")

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
  /* 侧边栏：指标速查。之前为了版面把它整块隐藏，现在要用它承载词典，
     所以改成可见 + 白底细边框；宽度由 Streamlit 控制，不写死。 */
  section[data-testid="stSidebar"] {{
      background:#FFFFFF; border-right:1px solid {C.COLOR['line']}; }}
  section[data-testid="stSidebar"] > div {{ padding-top:.8rem; }}
  section[data-testid="stSidebar"] .block-container {{ padding:.4rem .7rem; }}
  section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
      font-size:11px; }}
  /* 收起按钮必须常显：顶栏被隐藏后，它容易被一起盖掉，
     读者就再也拉不出侧边栏了。 */
  [data-testid="stSidebarCollapseButton"] {{
      display:block !important; visibility:visible !important; opacity:1 !important; }}
  [data-testid="stSidebarCollapsedControl"] {{
      display:block !important; visibility:visible !important; z-index:1000;
      background:#FFFFFF; border:1px solid {C.COLOR['line']}; border-radius:0 6px 6px 0; }}

  .hdr {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap;
          border-bottom:1px solid {C.COLOR['line']}; padding-bottom:8px; margin-bottom:8px; }}
  .hdr h1 {{ font-size:19px; margin:0; font-weight:700; letter-spacing:.2px; }}
  .tag {{ font-size:11.5px; padding:2px 9px; border-radius:10px; font-weight:700;
          border:1px solid transparent; }}
  .tiny {{ font-size:12px; color:{C.COLOR['text_dim']}; line-height:1.75; }}
  .sec {{ font-size:15px; font-weight:700; color:{C.COLOR['text']};
          margin:16px 0 7px 0; padding-left:9px;
          border-left:4px solid {C.COLOR['price']}; }}
  /* 顶栏标题：不用 <h1>，避免与 Streamlit 的标题组件在 React 协调时冲突 */
  .hdr .ttl {{ font-size:19px; font-weight:700; letter-spacing:.2px;
               color:{C.COLOR['text']}; }}

  /* 结论卡片组：结论 → 判断依据 → 失效条件，三段固定结构便于回溯 */
  .concl-wrap {{ display:flex; flex-direction:column; gap:7px; margin-bottom:4px; }}
  .concl {{ background:#FFFFFF; border:1px solid {C.COLOR['line']};
            border-left:4px solid {C.COLOR['price']}; border-radius:7px;
            padding:7px 11px; }}
  .concl .ct {{ font-size:14px; font-weight:700; margin-bottom:4px;
                color:{C.COLOR['text']}; }}
  .concl .cc {{ font-size:13px; line-height:1.85; color:{C.COLOR['text']}; }}
  .concl .cr {{ font-size:12.5px; line-height:1.8; color:#37485C;
                margin-top:3px; padding-left:9px;
                border-left:2px solid {C.COLOR['grid']}; }}
  .concl .cw {{ font-size:12px; line-height:1.7; margin-top:5px;
                color:#A35A00; background:#FFF8EC; border-radius:4px;
                padding:4px 8px; }}

  .kpis {{ display:flex; gap:7px; flex-wrap:wrap; }}
  .kpi {{ flex:1 1 0; min-width:82px; background:{C.COLOR['panel']};
          border:1px solid {C.COLOR['line']}; border-radius:7px;
          padding:7px 9px; }}
  .kpi .k {{ font-size:11.5px; color:{C.COLOR['text_dim']}; white-space:nowrap; }}
  .kpi .v {{ font-size:20px; font-weight:700; line-height:1.32;
             font-variant-numeric:tabular-nums; }}
  .kpi .s {{ font-size:11px; color:{C.COLOR['text_dim']}; line-height:1.5; }}

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
  div[data-testid="stRadio"] label {{ font-size:14px; padding:2px 5px;
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
  /* 指标释义卡 / 拆解链 / 四象限（src/metric_doc.py） */
  {md.css(C.COLOR)}
  /* 环比变化页（src/change_page.py） */
  {cp.css(C.COLOR)}

  /* ---- 全站说明块统一（必须放在 cp.css 之后覆盖它的 10.5px）---- */
  .cg-note {{ font-size:13px; line-height:1.95; color:#2C3E50;
      background:{C.COLOR['panel']}; border-left:3px solid {C.COLOR['price']};
      border-radius:0 8px 8px 0; padding:10px 14px; margin:6px 0 10px; }}
  .cg-note b {{ color:{C.COLOR['text']}; }}
  /* 表格与选择器的字号也上调一档，避免「正文大、控件小」的割裂感 */
  [data-testid="stSelectbox"] label, [data-testid="stTextInput"] label,
  [data-testid="stCheckbox"] label, [data-testid="stRadio"] p {{
      font-size:13px !important; }}
  [data-testid="stMetricLabel"] p {{ font-size:12px !important; }}
  [data-testid="stMetricValue"] {{ font-size:22px !important; }}
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

# 构建版本：每次实质更新时手动递增，页头展示，用于确认「线上跑的是哪一版」
APP_VERSION = "2026.11.04-r41.0"


def _fp_short() -> str:
    """展示用数据指纹：文本文件按「换行归一化后的内容」哈希，二进制按大小。
    - 不用 mtime：本地与云端是两次 checkout，mtime 必然不同；
    - 文本先做 CRLF→LF 归一：Windows 检出为 CRLF、Linux 为 LF，不归一则会不一致。
    这样「同一批数据 ⇒ 同一个指纹」，本地与线上可直接对比。"""
    import hashlib
    items = []
    paths = sorted(glob.glob(os.path.join(C.DATA_RAW, "*")))
    paths += sorted(glob.glob(os.path.join(C.DATA, "cache", "*.json")))
    paths.append(os.path.join(C.DATA, "wind_raw", "holdings_v2", "eq_series.json"))
    for p in paths:
        if not os.path.isfile(p):
            continue
        name = os.path.basename(p)
        if p.lower().endswith((".json", ".csv", ".txt", ".md")):
            with open(p, "rb") as f:
                body = f.read().replace(b"\r\n", b"\n")
            items.append(f"{name}:{hashlib.md5(body).hexdigest()[:10]}")
        else:
            items.append(f"{name}:{os.path.getsize(p)}")
    return hashlib.md5("|".join(items).encode("utf-8")).hexdigest()[:8]

# 数据先加载：顶栏需要显示「截至哪个季度」，因此必须排在顶栏之前。
# 注：这一套是 Wind 口径，仅「⑥ 方法与口径」页的覆盖/体检段落还在用，
# 其他 5 页全部走 canon（东方财富持仓）。试过按页跳过，但那几段旧代码
# 分散在方法页中后部，条件化后会在多处 AttributeError，风险大于 4 秒收益，
# 因此保留无条件加载。
B, V, FACT = load_bundle(FP)
IND, MKT = FACT["ind"], FACT["mkt"]
QUARTERS = list(B.quarters)
LATEST = B.latest
@st.cache_data(show_spinner=False)
def _derived(fp: str):
    """市场信号 / 市场摘要 / 仪表盘 payload：缓存起来，避免每次点击都重算。

    这三项原来写在模块级，Streamlit 每次交互都会重跑一遍——点「指标速查」
    要等 4 秒，主要就耗在这里（它们与页面无关，但每次都要算）。
    """
    _B, _V, _F = load_bundle(fp)
    _ind, _mkt = _F["ind"], _F["mkt"]
    return (sg.market_signals(_ind, _mkt, _B),
            fe.market_summary(_ind, _mkt, _B),
            sg.dashboard_payload(_ind, _mkt, _B))


MS, M_SUM, PAY = _derived(FP)
DETAIL_ALL = PAY["industry_history"]
RECORDS = PAY["records"]
LEVEL_MAP = {q: IND["拥挤等级"].loc[q].to_dict() for q in QUARTERS}

def _hdr_tags() -> str:
    """顶栏标签：期次数、基金数、样本口径——全部现算，不写死。

    早期这里写死了「样本 43 季（主口径有效 3 季）」，数据扩到 31 期后就成了错的。
    """
    try:
        import canon as _cn
        qs = _cn.quarters()
        if not qs:
            raise RuntimeError
        n = len(qs)
        nf = len(_cn.universe(qs[-1]))
        return ('<span class="tag" style="background:#F2F5F8;color:#6B7A8C">'
                f'31 行业 · {n} 期 · {nf:,} 只基金 · 截至 {qs[-1]}</span>'
                '<span class="tag" style="background:#F2F5F8;color:#6B7A8C">'
                '全持仓（半年报）看水平 · 前十大（季报）看广度</span>')
    except Exception:
        return ('<span class="tag" style="background:#F2F5F8;color:#6B7A8C">'
                f'31 行业 · 样本 {len(QUARTERS)} 季 · 截至 {LATEST}</span>')


# ---------------- 顶栏 ----------------
h1, h2, h3 = st.columns([0.50, 0.22, 0.28])
with h1:
    _h('<div class="hdr">'
                '<span class="ttl">公募基金交易拥挤度与集中度仪表盘</span>'
                '<span class="tag" style="background:#EAF1FB;color:#1F6FEB">'
                '持仓：东方财富全量披露</span>'
                + _hdr_tags() + '</div>')
with h2:
    pct_label = st.radio("分位口径", ["扩窗（无前视）", "全样本"],
                         index=0, horizontal=True, key="pct_mode",
                         help="扩窗：当期分位只用当期之前的历史（无未来信息）；"
                              "全样本：用全部可用历史（含前视，仅供对照）。")
with h3:
    c_a, c_b, c_c = st.columns([0.42, 0.32, 0.26])
    # 指标速查弹窗：侧边栏收起后，这是随时查指标的入口
    if c_a.button("📖 指标速查", width="stretch", key="btn_glossary"):
        st.session_state["gl_show"] = True
        st.session_state["_gl_page"] = st.session_state.get("page")
    if c_b.button("🔄 刷新", width="stretch", key="btn_reload"):
        st.cache_data.clear()
        st.rerun()
    with c_c:
        _h(f'<div class="tiny" style="padding-top:2px;line-height:1.45;'
           f'white-space:nowrap">{APP_VERSION.split("-")[-1]}<br>'
           f'{_fp_short()}</div>')

# 弹窗要在页面路由之前调用：各页在渲染完会 st.stop()，放在后面就永远不执行。
# 切换页面时必须自动关掉——否则每 rerun 一次它都会重新弹出，
# 看起来像「自己冒出来的」。这里记下打开时所在的页，换页就关。
if st.session_state.get("gl_show"):
    _page_now = st.session_state.get("page")
    _page_open = st.session_state.get("_gl_page")
    if _page_open is None:
        st.session_state["_gl_page"] = _page_now
    elif _page_now != _page_open:
        st.session_state["gl_show"] = False
        st.session_state["_gl_page"] = None
if st.session_state.get("gl_show"):
    try:
        import sidebar_panel as _sp
        _sp.glossary_dialog()
    except Exception:
        st.session_state["gl_show"] = False

# ============================================================================
# 三栏
# ============================================================================
# ============================================================================
# 页面切换
# ============================================================================
# 页面顺序按「先结论、后证据、再工具」排，且**行业视角在个股视角之前**：
#   ① 结论总览（所有判断与动作）→ ② 本期解读（逐图证据）
#   → ③ 行业拥挤度（先看 31 个行业谁挤）→ ④ 单只基金（再看具体产品）
#   → ⑤ 因子检验（这套方法靠不靠谱）→ ⑥ 方法与口径（口径与来源）
# 行业在前是有原因的：选基金之前先知道哪个行业在什么位置，
# 否则会先被单只基金的历史业绩带跑。
PAGES = ["① 结论总览", "② 本期解读", "③ 行业拥挤度", "④ 单只基金",
         "⑤ 因子检验", "⑥ 方法与口径"]
# 深链：?p=2 直达第 2 页（便于分享与截图走查）；非法值静默忽略，不影响默认行为。
try:
    _qp = str(st.query_params.get("p", "")).strip()
except Exception:
    _qp = ""
if _qp.isdigit() and 1 <= int(_qp) <= len(PAGES):
    st.session_state.setdefault("page", PAGES[int(_qp) - 1])
_P = st.radio("页面", PAGES, horizontal=True, label_visibility="collapsed", key="page")

# 侧边栏指标速查：每页都有，随时可查任何一个数的定义与算法
try:
    import sidebar_panel
    sidebar_panel.render(_P)
except Exception as _e:      # 侧边栏失败不能拖垮正文
    with st.sidebar:
        st.caption(f"指标速查暂不可用：{type(_e).__name__}")

_PROV = B.provenance if getattr(B, "provenance", None) is not None else None
# 新 6 页不加载 Wind，_PROV 恒为 None，_prov_line() 会退化成「未接入」文案；
# 这几个页面本来也不用它。
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


def _quarter_ticks(qs, max_labels: int = 11) -> list:
    """季度轴刻度稀疏化。

    42 个季度全标注会挤成一团（截图里标签斜压、互相重叠）。规则：
      * 刻度数 ≤ max_labels：全标；
      * 超过：只标每年 Q1 + 末期，保证「年—季」可读且不重叠。
    """
    qs = list(qs)
    if not qs or len(qs) <= max_labels:
        return qs
    out = [q for q in qs if str(q).endswith("Q1")]
    if qs[-1] not in out:
        out.append(qs[-1])
    return out


def _set_quarter_axis(fig, qs, row=None, col=None, angle: int = -45, **kw):
    """把季度刻度稀疏化后写入 x 轴（可指定 subplot 的 row/col）。"""
    tk = _quarter_ticks(qs)
    opt = dict(tickmode="array", tickvals=tk, ticktext=tk, tickangle=angle)
    opt.update(kw)
    if row is None:
        fig.update_xaxes(**opt)
    else:
        fig.update_xaxes(row=row, col=col, **opt)


# ============================================================================
# 页面 ① 结论总览（所有判断、理由、投资解读、操作建议）
# ============================================================================
if _P == "① 结论总览":
    import overview_page
    overview_page.render(_h)
    st.stop()


# ============================================================================
# 页面 ② 本期解读（逐图证据；不重复①的结论）
# ============================================================================
if _P == "② 本期解读":
    import report_page
    report_page.render(_h)
    st.stop()


# ============================================================================
# 页面 ④ 单只基金
# ============================================================================
if _P == "④ 单只基金":
    import fund_page
    fund_page.render(_h)
    st.stop()


# ============================================================================
# 页面 ③ 行业拥挤度
# ============================================================================
if _P == "③ 行业拥挤度":
    import map_page
    map_page.render(_h)
    st.stop()


# ============================================================================
# 页面 ⑤ 因子检验（标准流程：定义→计算→IC→显著性→分层→结论）
# ============================================================================
if _P == "⑤ 因子检验":
    import factor_page
    factor_page.render(_h)
    st.stop()



if _P == "（已停用）旧版因子检验":
    # 因子与检验：定义一览（表格）+ 完整说明（可展开）+ IC/分层/稳健性
    import factor_docs
    import re as _re

    def _plain(s):
        s = _re.sub(r"<br\s*/?>", " ", str(s))
        s = _re.sub(r"</?(code|b|i|strong|em|u)>", "", s)
        return (s.replace("&gt;", ">").replace("&lt;", "<")
                 .replace("&amp;", "&").strip())

    _cards = factor_docs.as_concl(IND)
    _rows = []
    for _c in _cards:
        _why = _c.get("why", {})
        _rows.append({
            "因子": _plain(_c.get("title", "")),
            "衡量什么": _plain(_c.get("concl", "")),
            "计算公式": _plain(_why.get("计算公式", _why.get("RankIC", ""))),
            "判定标准": _plain(_why.get("判定标准",
                              _why.get("IC_IR", _why.get("显著性", "")))),
        })
    _h('<div class="sec">因子定义一览（公式与阈值取自代码真实实现）</div>')
    # 用 HTML 表格而非 st.dataframe：dataframe 在窄屏会把长文本截断，
    # 而这里每列都需要可读（衡量对象 / 判定标准），故显式换行并限制表体高度。
    _th = "padding:5px 8px;text-align:left;font-weight:700;font-size:12px;"
    _td = "padding:5px 8px;vertical-align:top;font-size:12px;line-height:1.5;"
    _tr = "".join(
        f'<tr style="border-top:1px solid {C.COLOR["line"]}">'
        f'<td style="{_td}width:20%;font-weight:600">{r["因子"]}</td>'
        f'<td style="{_td}width:40%">{r["衡量什么"]}</td>'
        f'<td style="{_td}width:40%">{r["判定标准"]}</td></tr>'
        for r in _rows)
    _h(f"""<div style="max-height:430px;overflow:auto;border:1px solid {C.COLOR['line']};
border-radius:6px;background:#FFFFFF">
      <table style="width:100%;border-collapse:collapse">
        <thead style="position:sticky;top:0;background:{C.COLOR['panel']};z-index:2">
          <tr><th style="{_th}">因子</th><th style="{_th}">衡量什么</th>
              <th style="{_th}">判定标准</th></tr></thead>
        <tbody>{_tr}</tbody></table></div>""")
    _note("完整版（含计算公式、为什么选这个因子、经济内容与标准依据）见下方展开区。")
    with st.expander("展开：每个因子的完整说明（计算公式 / 为什么选 / 经济内容 / 标准依据）",
                     expanded=False):
        _h(concl(_cards))
    _BTR = load_backtest(FP)

    # 统计量释义（本轮新增）：IC / IR / t / 胜率 / 分层 / 事件研究
    # 数值取参考因子的实测结果，随数据自动更新；切换因子只影响图上曲线，不影响口径说明。
    def _nx(v, fmt="{:.2f}", na="—"):
        return na if v is None or (isinstance(v, float) and pd.isna(v)) else fmt.format(v)

    # 注：_BTR["ic"] 是 t+1 / t+2 合并表（列名带 _t1/_t2 后缀），
    # 单一滞后口径的原始表在 _BTR["ic_t1"]["table"]。
    _ict = _BTR["ic_t1"]["table"].set_index("因子")
    _fref = next((f for f in ["F11_综合拥挤度_推荐权重", "F3_超配历史分位",
                              "F2_超配Zscore", "F1_超配比例"] if f in _ict.index),
                 _ict.index[0])
    _ir = _ict.loc[_fref]
    _lref = None
    if isinstance(_BTR["layer"], pd.DataFrame) and "因子" in _BTR["layer"].columns:
        _lm = _BTR["layer"].set_index("因子")
        if _fref in _lm.index:
            _lref = _lm.loc[_fref]
    _ev0 = None
    _evs = _BTR["ev_over"].get("stats")
    if isinstance(_evs, pd.DataFrame) and len(_evs):
        _ev0 = _evs.iloc[0]

    _h('<div class="sec">统计量怎么读 · 判断「这个因子是否真的有用」的六个证据</div>')
    _h(md.render_grid(md.STAT, {
        "rankic": (f"{_ir['ic_mean']:+.3f}", f"{int(_ir['n'])} 期均值"),
        "icir": (_nx(_ir["ic_ir"]), "均值 ÷ 标准差"),
        "t": (_nx(_ir["t_nw"]), f"p = {_nx(_ir['p_nw'], '{:.3f}')}"),
        "win": (f"{_ir['win_rate']:.0f}%", "同向期数占比"),
        "layer": (_nx(_lref["单调性Spearman"] if _lref is not None else None, "{:+.2f}"),
                  "5 层年化收益的秩相关"),
        "event": (f"{_ev0['均值']:+.2f}%" if _ev0 is not None else "—",
                  f"{int(_ev0['触发次数'])} 次触发 · 3 个月" if _ev0 is not None else ""),
    }, C.COLOR, cols=2))
    _note(f"本期值取自参考因子 <b>{_fref}</b>：RankIC {_ir['ic_mean']:+.3f}、"
          f"N={int(_ir['n'])}、IC_IR {_nx(_ir['ic_ir'])}、NW t {_nx(_ir['t_nw'])}。"
          "「事件研究」用的是超配突破 P90 的首次触发样本（全行业合并），不是单因子 IC，"
          "两者回答的是不同问题：IC 问排序能力，事件研究问触发后的实际盈亏。")

    _icf = st.multiselect("因子", list(_BTR["ic"]["因子"]),
                          default=[f for f in ["F15_拥挤背离", "R_F2_超配Z_残差",
                                               "R_F3_超配分位_残差", "F2_超配Zscore"]
                                   if f in list(_BTR["ic"]["因子"])],
                          max_selections=6, key="ic_fac")
    # ① IC 时序 + 滚动 IC_IR
    _h('<div class="sec">① 因子 IC 时序与滚动 IC_IR（4 季度）</div>')
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

    with st.expander("因子相关性与衰减（进阶）", expanded=False):

        _h('<div class="sec">因子相关性热力图（逐期截面 Spearman 平均）</div>')
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

        _h('<div class="sec">因子衰减曲线（滚动窗口 IC 均值）</div>')
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
    _h('<div class="sec">② 压力测试（以 2026Q2 行业配置比例为组合权重）</div>')
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
    _h('<div class="sec">③ 因子有效性检验（IC / RankIC，Newey-West 滞后 2 期）</div>')
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


# ============================================================================
# 页面 ⑤ 轮动策略与因子裁决（第 2 轮自主迭代）
# ============================================================================
if _P == "（已停用）旧版因子检验":
    _BTR = load_backtest(FP)
    _h('<div class="sec">④ 因子分级裁决（4 项独立检验合成可用性结论）</div>')
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
    with st.expander("样本外与正交化检验（进阶）", expanded=False):

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
    _h('<div class="sec">⑤ F15 行业轮动策略（long-only，季度调仓，扣 20bp）</div>')
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

    _h('<div class="sec">⑥ 分层回测：F15 五层累积净值（层1＝因子值最低）</div>')
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

    with st.expander("参数敏感性与方向检验（进阶）", expanded=False):

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


    # ============================================================================
    # 页面 ⑥ 稳健性检验（第 3 轮自主迭代）
    # ============================================================================
if _P == "（已停用）旧版因子检验":

    with st.expander("进阶稳健性检验（Walk-forward / 设计窗 / 构造依据 / 归因 / 基准变体 共 7 项）", expanded=False):

        _BTR = load_backtest(FP)
        R3 = _BTR["r3"]

        _h('<div class="sec">Walk-forward 滚动样本外（每期只用截至 t−1 的数据挑设定）</div>')
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

        _h('<div class="sec">设计窗长度敏感性 ＋ 状态/区块分解</div>')
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

        _h('<div class="sec">构造依据「最早可得时点」检验（F15 时间戳前移）</div>')
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

        _h('<div class="sec">正交双因子组合（F15 × F3d）</div>')
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

        _h('<div class="sec">cap 约束 vs 因子倾斜 的分离 ＋ 四宫格口径对照</div>')
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

        _h('<div class="sec">超额收益行业归因（真 alpha 还是低配大牛股的 beta）</div>')
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

        _h('<div class="sec">基准变体对照（含逐期剔除最大权重行业）</div>')
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
# ============================================================================
# 页面 ⑥ 方法与口径（精简版：口径 → 数据 → 关键指标 → 自检 → 局限）
# ============================================================================
if _P == "⑥ 方法与口径":
    import method_page
    method_page.render(_h)
    st.stop()


# ============================================================================
# 旧版单只基金（左右栏布局，已被 src/fund_page.py 取代，仅作回滚备份）
# ============================================================================
if _P == "（已停用）旧版单只基金":
    import fund_page
    fund_page.render(_h)
    st.stop()


L, M = st.columns([15, 85], gap="small")

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

    # ①a 为什么是这个数：把「超配 X pct」拆回分子、分母与差额（本轮新增）
    #    全部数字现算，避免「文字写死、数据已变」；四象限判定同一处给出。
    _alloc_0 = B.alloc.loc[LATEST, sel]
    _bench_0 = IND["F12b_基准权重%"].loc[LATEST, sel]
    _over_0 = B.overweight.loc[LATEST, sel]
    _f12_0 = IND["F12_配置系数"].loc[LATEST, sel]
    _f13_0 = IND["F13_筹码盈利比例"].loc[LATEST, sel]
    _turn_0 = IND["F9a_成交额占比%"].loc[LATEST, sel]
    _turnp_0 = IND["F9b_成交额占比分位"].loc[LATEST, sel]
    _rk_0 = int(latest_rank.rank(ascending=False)[sel])
    _top3_a = "、".join(f"{k} {v:.2f}%" for k, v in
                        B.alloc.loc[LATEST].dropna().sort_values(ascending=False).head(3).items())
    _top3_b = "、".join(f"{k} {v:.2f}%" for k, v in
                        IND["F12b_基准权重%"].loc[LATEST].dropna()
                        .sort_values(ascending=False).head(3).items())
    def _fv(x, tpl="{:.2f}%", na="—"):
        """数值安全格式化：缺值不显示为 nan。"""
        try:
            return na if x is None or pd.isna(x) else tpl.format(x)
        except Exception:
            return na

    _ctx0 = dict(
        sel=sel, q=LATEST, alloc=_alloc_0, bench=_bench_0, over=_over_0, f12=_f12_0,
        alloc_rank=int(B.alloc.loc[LATEST].rank(ascending=False)[sel]),
        bench_rank=int(IND["F12b_基准权重%"].loc[LATEST].rank(ascending=False)[sel]),
        n_ind=len(INDUSTRIES),
        n_funds=float(MKT["F8_样本基金数"].loc[LATEST]),
        n_stocks=float(MKT["F7_个股集中度"].loc[LATEST].get("个股总数", np.nan)),
        cr3=float(MKT["F6_CR3"].loc[LATEST]),
        top3_bench=_top3_b, top3_alloc=_top3_a, f13=_f13_0,
        turn=_turn_0, turn_pct=_turnp_0, pct=pct, z=z, bench_name="沪深300")

    _h('<div class="sec">为什么是这个数 · 超配拆解</div>')
    _h(md.decompose_overweight(_ctx0, C.COLOR))
    _h(md.render_regime(md.crowding_regime(pct, _turnp_0), C.COLOR))
    _note("四象限口径：横轴＝超配历史分位（机构仓位高度），纵轴＝成交额占比历史分位（交易热度）。"
          "同样是高拥挤，「产业趋势型」与「主题交易型」的持续性完全不同——"
          "前者由盈利上修驱动、可以维持数个季度，后者以成交萎缩结束。")


    # ①a2 配置变化归因：主动加减仓 vs 价格漂移（本轮新增）
    #     回答「占比上升是买出来的还是涨出来的」——两者的投资含义完全相反。
    _dec = atr.decompose(B.alloc, B.market["ret"])
    _sp = atr.split_one(_dec, LATEST, sel)
    _i_latest = QUARTERS.index(LATEST)
    _q0a = QUARTERS[_i_latest - 1] if _i_latest > 0 else None
    if _sp and _sp.get("ok"):
        _vd2 = atr.verdict(_sp)
        _ov_src = (B.over_ext[sel] if getattr(B, "over_ext_available", False)
                   and sel in B.over_ext.columns else B.overweight[sel])
        _ov_hist = _ov_src.dropna().iloc[-12:]
        _ov_tgt = float(_ov_hist.median()) if len(_ov_hist) else np.nan
        _turn_q = (float(B.market["turn"].loc[LATEST, sel])
                   if sel in B.market["turn"].columns else np.nan)
        _tot = (float(B.holdings_value.loc[LATEST])
                if LATEST in getattr(B, "holdings_value", pd.Series()).index else np.nan)
        _pr = atr.pressure(_over_0, _ov_tgt, _alloc_0, _tot, _turn_q)
        _r_q = (float(B.market["ret"].loc[LATEST, sel])
                if sel in B.market["ret"].columns else np.nan)
        _h('<div class="sec">配置变化从哪来 · 主动加减仓 vs 价格漂移</div>')
        _press = ""
        if np.isfinite(_pr["d_over"]) and _pr["d_over"] > 0:
            _press = (
                f'<div class="mcard"><div class="h">近 12 季超配中位数</div>'
                f'<div class="mrow"><span>目标水平</span><span>{_ov_tgt:.2f}%</span></div>'
                f'<div class="mrow"><span>当前</span><span>{_over_0:+.2f}%</span></div></div>'
                f'<div class="mcard"><div class="h">需回落的超配</div>'
                f'<div class="mrow"><span>幅度</span><span>{_pr["d_over"]:.2f} pct</span></div>'
                f'<div class="mrow"><span>占该行业持仓</span>'
                f'<span>{_pr["need_pct_industry"]:.0f}%</span></div></div>'
                f'<div class="mcard"><div class="h">折算规模（缓存口径）</div>'
                f'<div class="mrow"><span>金额</span><span>{_fv(_pr["need_yi"], "{:,.0f} 亿")}</span></div>'
                f'<div class="mrow"><span>≈ 成交额</span>'
                f'<span>{_fv(_pr["days_of_turnover"], "{:.1f} 个交易日")}</span></div></div>')
        else:
            _press = ('<div class="mcard"><div class="h">无需回归</div>'
                      '<div class="mrow"><span>当前超配</span>'
                      f'<span>{_over_0:+.2f}%</span></div>'
                      '<div class="mrow"><span>已低于近 12 季中位数</span>'
                      f'<span>{_over_0 - _ov_tgt:+.2f} pct</span></div></div>')
        _h(f"""<div class="dz">
          <div class="dz-h">{sel}：{_q0a} → {LATEST}，配置 {_sp['上季配置']:.2f}% → {_sp['本季配置']:.2f}%，
            <b style="color:{C.COLOR['up'] if _sp['变化'] >= 0 else C.COLOR['down']}">
            {_sp['变化']:+.2f} pct</b></div>
          <div class="ch-wrap">
            <div class="ch-cell"><div class="ch-t"><span class="ch-i">1</span>上季实际配置</div>
              <div class="ch-v">{_sp['上季配置']:.2f}%</div>
              <div class="ch-x">季报前十大重仓口径</div></div>
            <div class="ch-op">→</div>
            <div class="ch-cell"><div class="ch-t"><span class="ch-i">2</span>只涨不买（被动漂移）</div>
              <div class="ch-v">{_sp['被动漂移后']:.2f}%</div>
              <div class="ch-x">按行业指数当季 {_fv(_r_q, '{:+.2f}%')} 折算，
                贡献 <b>{_sp['被动贡献']:+.2f} pct</b></div></div>
            <div class="ch-op">→</div>
            <div class="ch-cell"><div class="ch-t"><span class="ch-i">3</span>本季实际配置</div>
              <div class="ch-v">{_sp['本季配置']:.2f}%</div>
              <div class="ch-x">与「只涨不买」的差额＝主动加减仓
                <b>{_sp['主动贡献']:+.2f} pct</b></div></div>
          </div>
          <div class="rg" style="background:#F8FAFC">
            <div class="rg-t">判定：<b>{_vd2['tag']}</b></div>
            <div class="rg-r">{_vd2['text']}</div>
          </div>
          <div class="dz-sub">压力测算 · 若超配回落到近 12 季中位数</div>
          <div class="mini">{_press}</div>
          <div class="dz-note">口径：被动漂移＝上季配置 ×（1＋行业指数涨跌幅）后按 31 行业归一，
            差额为主动加减仓（Σ主动 ≈ 0）。行业指数收益是个股组合的近似，
            偏差主要落在「主动」项；金额按缓存重仓市值折算，D14 已注明该口径偏低约 40%，
            故金额与天数只作量级参考，比例结论不受影响。</div>
        </div>""")

    # 指标释义：12 项卡片收进折叠区，紧跟「为什么是这个数」之后——
    # 首屏不被文字挤压，需要查定义时一次点击即可，不必滑到页尾。
    with st.expander("指标释义 · 12 项：定义 / 经济含义 / 投资含义 / 读法 / 误读",
                     expanded=False):
        _h(md.render_grid(md.INDUSTRY, {
            "alloc": (_fv(_alloc_0), f"占全部重仓市值　排名 {_ctx0['alloc_rank']}/{len(INDUSTRIES)}"),
            "bench": (_fv(_bench_0), f"指数权重　排名 {_ctx0['bench_rank']}/{len(INDUSTRIES)}"),
            "over": (_fv(_over_0, "{:+.2f} pct"), f"{_fv(_alloc_0)} − {_fv(_bench_0)}",
                     "up" if pd.notna(_over_0) and _over_0 >= 0 else "dn"),
            "Z": (_fv(z, "{:+.2f}"), "扩窗 8 期起，无前视"),
            "pct": (_fv(pct, "{:.1f}%"), "0~100，越高越极端"),
            "F12": (_fv(_f12_0, "{:.2f}×"), f"{_fv(_alloc_0)} ÷ {_fv(_bench_0)}"),
            "F13": (_fv(_f13_0, "{:.1f}%"), "浮盈盘占比"),
            "turn": (_fv(_turn_0), f"历史分位 {_fv(_turnp_0, '{:.0f}%')}"),
            "F4": (_fv(IND["F4_超配动量"].loc[LATEST, sel], "{:+.2f} pct"),
                   f"上季超配 {_fv(B.overweight.loc[QUARTERS[-2], sel] if len(QUARTERS) > 1 else None, '{:+.2f}')}"),
            "F10": (_fv(IND["F10_共振得分"].loc[LATEST, sel], "{:.0f}/3"),
                    "S1 {} / S2 {} / S3 {}".format(
                        *[int(v) if pd.notna(v) else 0 for v in (
                            IND["F10_S1_持仓拥挤"].loc[LATEST, sel],
                            IND["F10_S2_交易拥挤"].loc[LATEST, sel],
                            IND["F10_S3_筹码拥挤"].loc[LATEST, sel])])),
            "F11": (_fv(comp, "{:+.2f}"), f"全行业排名 {_rk_0}/{len(INDUSTRIES)}"),
            "level": (lvl_tag(lv_sel), "分位 / 配置系数双阈值取最高级"),
        }, C.COLOR, cols=2))

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

    # 投资建议：紧跟行业结论（等级判断理由）之后，不再压到页尾
    _h(guide.opportunity_hint(M_SUM, C, "行业钻取"))

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
    # 主口径扩窗需 8 期历史，2025Q2 及更早的等级为「数据不足」；
    # 同比/四季对比改用长历史等级（F11e，37 期）兜底，避免展示无意义的「数据不足」。
    _lve = IND.get("F11e_等级_长历史")
    if lv_4 in (None, "数据不足") and _lve is not None and q4 in _lve.index:
        _v = _lve.loc[q4, sel]
        lv_4 = _v if isinstance(_v, str) and _v else lv_4
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
      <div class="mcard"><div class="h">近四季（{q4} → {q1}；等级用长历史口径）</div>
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
    # 去掉「三块面板全为空」的季度：全区间下 2016~2021 无持仓也无成交额，
    # 若不裁掉会在左侧留下大片空白（截图里最刺眼的问题之一）。
    _plot_cols = [c for c in ["超配比例%", "配置比例%", "成交额占比%"]
                  if c in w.columns]
    if _plot_cols:
        _keep = w[_plot_cols].notna().any(axis=1)
        if _keep.any():
            w = w.loc[_keep]
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
        _set_quarter_axis(fig, x, row=r_, col=1, showgrid=False,
                          linecolor=C.COLOR["line"])
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
with M:
    # 全市场速览 / 综合判断已整合到 ① 总览，本页不再重复（2026-10-18 精简）。

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
# 投资建议：紧跟行业结论之后（用户要求「投资建议放在结论后面」）

# ============================================================================
_h('<div class="sec">拥挤度变化榜 · 谁在变挤 / 谁在变松</div>')
_ch1, _ch2 = st.columns([0.45, 0.55])
with _ch1:
    _chg_mode = st.radio("变化口径", ["季度环比", "年度同比"], horizontal=True,
                         key="chg_mode", label_visibility="collapsed")
with _ch2:
    _chg_n = st.slider("显示条数", 3, 10, 5, key="chg_n",
                       label_visibility="collapsed")

# 长历史口径优先：F11e（偏股混合单类型，2016Q2~2026Q2 共 36 期），
# 仅在有值时使用；缺失时回退主口径（主动权益三类，10 期）。
_f11_long = IND.get("F11e_综合拥挤度_长历史")
_use_long = (_f11_long is not None and _f11_long.notna().any().any())
_f11_all = _f11_long if _use_long else IND["F11_综合拥挤度_推荐权重"]
_f11_name = "F11e 长历史综合拥挤度（偏股混合口径）" if _use_long else "F11 综合拥挤度（三类口径）"
_q_now = LATEST
if _chg_mode == "季度环比":
    _q_base = QUARTERS[-2] if len(QUARTERS) >= 2 else LATEST
    _chg_label = f"{_q_now} vs {_q_base}（环比）"
else:
    _q_base = QUARTERS[-5] if len(QUARTERS) >= 5 else QUARTERS[0]
    _chg_label = f"{_q_now} vs {_q_base}（同比，去年同季）"
_delta = (_f11_all.loc[_q_now] - _f11_all.loc[_q_base]).dropna().sort_values()
_delta = _delta[_delta.abs() > 1e-9]
if _delta.empty:
    st.info("当前样本不足以计算变化（需至少两期）。")
else:
    _top_inc = _delta.tail(_chg_n)      # 变挤
    _top_dec = _delta.head(_chg_n)      # 变松
    _sel_chg = pd.concat([_top_dec, _top_inc]).sort_values()
    _colors_chg = [C.COLOR["down"] if v < 0 else C.COLOR["up"]
                   for v in _sel_chg.values]
    _f_chg = go.Figure(go.Bar(
        x=_sel_chg.values, y=_sel_chg.index, orientation="h",
        marker=dict(color=_colors_chg),
        text=[f"{v:+.2f}" for v in _sel_chg.values], textposition="outside",
        hovertemplate="%{y}　变化 %{x:+.3f}<extra></extra>"))
    _f_chg.update_layout(height=max(280, 34 * len(_sel_chg) + 70),
                         template=C.PLOTLY_TEMPLATE,
                         margin=dict(l=70, r=40, t=18, b=30),
                         paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                         font=dict(family="Microsoft YaHei", size=10.5),
                         xaxis_title="综合拥挤度变化（Z）")
    _f_chg.update_xaxes(gridcolor=C.COLOR["grid"], zeroline=True,
                        zerolinecolor=C.COLOR["line"])
    _f_chg.update_yaxes(showgrid=False)
    st.plotly_chart(_f_chg, width="stretch", config={"displayModeBar": False})
    _inc_names = "、".join(_top_inc.index[::-1][:3])
    _dec_names = "、".join(_top_dec.index[:3])
    _note(f"口径：{_chg_label}。红＝拥挤度上升（变挤），绿＝下降（变松）；"
          f"仅显示变化最大的各 {_chg_n} 个行业。"
          f"变挤前三：<b>{_inc_names}</b>；变松前三：<b>{_dec_names}</b>。"
          f"变化值基于{_f11_name}（截面 Z），跨期可比。")

# ============================================================================
# 拥挤度热力图（31 行业 × 季度，红色系，标注数值）
# ============================================================================
_h('<div class="sec">拥挤度热力图 · 31 行业 × 季度</div>')
# 只保留有值的季度（长历史口径 36 期；缺失期不画空白）
_valid_q = [q for q in QUARTERS
            if q in _f11_all.index and _f11_all.loc[q].notna().any()]
_hm = (_f11_all.loc[_valid_q].T if _valid_q else _f11_all.T)
# 行按最新一期拥挤度降序排列：最拥挤的行业置顶，纵向一眼看出梯队
if _hm.shape[1]:
    _last_v = _hm.iloc[:, -1]
    _hm = _hm.reindex(_last_v.sort_values(ascending=False).index)
_hm_arr = _hm.to_numpy(dtype=float)
_hm_txt = [["" if (v != v or abs(v) < 1.0) else f"{v:.1f}" for v in row]
           for row in _hm_arr]
_hm_fs = 8 if len(_valid_q) <= 14 else (7 if len(_valid_q) <= 24 else 6.5)
_f_hm = go.Figure(go.Heatmap(
    z=_hm_arr, x=list(_hm.columns), y=list(_hm.index),
    text=_hm_txt, texttemplate="%{text}",
    textfont=dict(size=_hm_fs, color="#40201A"),
    colorscale=[[0, "#FFFFFF"], [0.15, "#FCE3DE"], [0.35, "#F7BEB3"],
                [0.55, "#EE8B78"], [0.78, "#D8513C"], [1, "#A61B0B"]],
    zmid=0, zmin=-2.0, zmax=2.5, xgap=0.6, ygap=0.6,
    colorbar=dict(title=dict(text="拥挤度Z", side="right"),
                  thickness=13, len=0.92, tickfont=dict(size=9),
                  ticks="outside", tickformat=".1f"),
    hovertemplate="%{y}　%{x}　拥挤度 <b>%{z:.2f}</b><extra></extra>"))
_f_hm.update_layout(height=880, template=C.PLOTLY_TEMPLATE,
                    margin=dict(l=70, r=34, t=26, b=30),
                    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                    font=dict(family="Microsoft YaHei", size=10))
_f_hm.update_xaxes(showgrid=False, side="top", tickangle=45)
_f_hm.update_yaxes(showgrid=False, autorange="reversed")
st.plotly_chart(_f_hm, width="stretch", config={"displayModeBar": False})
_note(f"颜色越深＝拥挤度越高（红色系）；<b>|Z| ≥ 1.0 的格子已标注数值</b>，"
      f"其余悬停可查精确值。行＝31 个申万一级行业，列＝季度（{len(_valid_q)} 期："
      f"{_valid_q[0]} ~ {_valid_q[-1]}）。口径：{_f11_name}，"
      "数据来源 Wind 前十大重仓股。<b>深红＝该行业相对自身历史处于极端拥挤</b>，"
      "白色＝接近自身历史中位。")


# ============================================================================
# 底部说明：因子有效性检验表已整体迁移至 ④ 因子与检验页。
# ============================================================================
_h('<div class="tiny" style="border-top:1px solid ' + C.COLOR["line"] + ';padding-top:6px;margin-top:8px">完整因子检验表（RankIC 汇总 / 分层回测 / 样本内外 / 牛熊 / 参数敏感性 / 正交化 / 事件研究 / Bootstrap / 衰减 / FCI / 多信号共振）已移至 <b>④ 因子与检验</b> 页查看。</div>')

_h(f'<div class="tiny" style="border-top:1px solid {C.COLOR["line"]};'
    'padding-top:6px;margin-top:8px">'
    '拥挤度是<b>风险描述指标</b>，不是择时或做空信号。'
    '高拥挤 ≠ 必然下跌，只代表"市场重新审视定价"的概率上升。'
    '本页所有阈值（P90 / 分位 / Z）均为<b>扩窗（无前视）</b>口径。'
    '数据更新：把新文件放进 <code>data/raw/</code> 后按 <b>R</b> 或点右上角「重新读取数据」。'
    '</div>')
