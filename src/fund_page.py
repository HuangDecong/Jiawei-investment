# -*- coding: utf-8 -*-
"""② 单只基金透视：这只基金到底在赌什么、结构有没有漂、和同行差在哪。

页面顺序按「先看结构、再看偏离、最后看个股」：
    1. 行业结构漂移（全部 31 个行业，全样本期）——第一眼就要看到它押注的变化；
    2. 与同行的偏离——哪些是它自己的判断，哪些是全员都在做；
    3. 拥挤暴露——它的持仓有多少压在已经很挤的行业上（组合有多脆弱）；
    4. 前十大——季报能看到的最后一块拼图。

口径
----
· 结构有两条序列，页面上可切换：
    前十大（全样本期，归一化 100%）——季频、最长，看「股票里怎么分」；
    全持仓（全部半年报期，占净值比）——半年频、最准，看「净值里放了多少」；
· 对比基准是同期样本基金的**平均值**，不是沪深 300——这里问「和同行比」，
  不是「和市场比」；后者需要指数权重，本库没有。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import canon as cn
import config as C
import report_charts as rc
import ui


def _base(fig, title, h=420, legend=False):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=C.COLOR["text"]), x=0.005),
        height=h, margin=dict(l=6, r=12, t=46, b=10),
        paper_bgcolor=C.COLOR["bg"], plot_bgcolor=C.COLOR["bg"],
        font=dict(size=11, color=C.COLOR["text"]), showlegend=legend,
        legend=dict(orientation="h", y=-0.14, x=0, font=dict(size=9)),
        hoverlabel=dict(font_size=11),
    )
    fig.update_xaxes(gridcolor=C.COLOR["grid"], zerolinecolor=C.COLOR["line"])
    fig.update_yaxes(gridcolor=C.COLOR["grid"], zerolinecolor=C.COLOR["line"])
    return fig


def _read(h, fact: str, econ: str, invest: str = "") -> None:
    """图下方的三段式分析：事实 → 经济含义 → 投资含义。"""
    ui.analysis(h, fact, econ, invest)


@st.cache_data(show_spinner=False)
def _funds() -> pd.DataFrame:
    p = cn.profiles()
    if p.empty:
        return pd.DataFrame(columns=["code", "name", "scale_yi"])
    d = p.dropna(subset=["scale_yi"]).copy()
    d = d[d["scale_yi"] > 0]
    return d.sort_values("scale_yi", ascending=False)


@st.cache_data(show_spinner=False)
def _peer(q: str) -> pd.Series:
    """同行基准：样本基金的**平均**行业配置。

    这里不能用中位数：前十大口径下，多数行业有一半以上的基金不持有，
    中位数恒等于 0，算出来的「低配」全是 0，没有信息。
    平均值等价于「等权同行组合」，正是 Brinson 意义上的基准。
    """
    p = cn.panel_top10(q)
    if p.empty:
        return pd.Series(dtype=float)
    u = cn.universe(q)
    idx = p.index.intersection(list(u)) if u else p.index
    return p.loc[idx].mean()


@st.cache_data(show_spinner=False)
def _peer_full(q: str) -> pd.Series:
    p = cn.panel_full(q, normalize=False)
    if p.empty:
        return pd.Series(dtype=float)
    u = cn.universe(q)
    idx = p.index.intersection(list(u)) if u else p.index
    return p.loc[idx].mean()


@st.cache_data(show_spinner=False)
def _top_holdings(code: str, q: str) -> pd.DataFrame:
    d = cn.fund_holdings(code, q, limit=12)
    if not d.empty:
        d = d.copy()
        d["mv_yi"] = d["mv"] / 1e4
    return d


def render(h, code: str | None = None, show_picker: bool = True) -> None:
    funds = _funds()
    q = cn.latest()
    pf = cn.pair_full()
    if funds.empty or not pf:
        h('<div class="cg-note">基金档案尚未抓取完成，先运行 '
          '<span class="mono">python tools/fetch_fund_profile.py</span>。</div>')
        return
    qf = pf[1]

    # ---------------- 选择器 ----------------
    # 与「季报全解」合并成一页后，基金由上面的统一选择器传入，
    # 这里不再重复放一个选择框（两个框会各记一份状态，切换时容易对不上）。
    def _picker() -> str:
        _i = 0
        if code:
            _hit = [j for j, s in enumerate(opts) if s.startswith(str(code) + "　")]
            _i = _hit[0] if _hit else 0
        return st.selectbox("选择基金（可直接输入代码或名称搜索，按规模排序）",
                            opts, index=_i, key="fund_pick").split("　")[0].strip()

    opts = ui.fund_options(funds)
    if show_picker or not code:
        c1, c2 = st.columns([0.5, 0.5])
        with c1:
            code = _picker()
        with c2:
            _mode = st.radio(
                "结构口径",
                [f"前十大（{len(cn.quarters())} 期，季频）",
                 f"全持仓（{len(cn.full_quarters())} 期，半年频）"],
                horizontal=True, key="fund_basis",
                help="前十大：每期只取最重的 10 只再归一化，季频、最长，"
                     "看「股票里怎么分」；全持仓：半年报全部持仓占净值比，"
                     "最准但一年只有两次，看「净值里放了多少股票」。")
    else:
        code = str(code).zfill(6)
        _mode = st.radio(
            "结构口径",
            [f"前十大（{len(cn.quarters())} 期，季频）",
             f"全持仓（{len(cn.full_quarters())} 期，半年频）"],
            horizontal=True, key="fund_basis",
            help="前十大：每期只取最重的 10 只再归一化，季频、最长，"
                 "看「股票里怎么分」；全持仓：半年报全部持仓占净值比，"
                 "最准但一年只有两次，看「净值里放了多少股票」。")

    basis = "top10" if _mode.startswith("前十大") else "full"
    hist = cn.fund_structure(code, basis)
    if hist.empty:
        h(f'<div class="cg-note">该基金在库内没有{"前十大" if basis=="top10" else "全持仓"}'
          f'期的数据（{code}）。</div>')
        return

    prof = cn.profiles()
    row = prof[prof["code"] == code]
    nm = ui.safe_str(row["name"].iloc[0], code) if len(row) else code
    typ = ui.safe_str(row["ftype"].iloc[0]) if len(row) else "—"
    inc = row["inception"].iloc[0] if len(row) else pd.NaT
    scale = row["scale_yi"].iloc[0] if len(row) else np.nan
    mgr = ui.safe_str(row["managers"].iloc[0]) if len(row) else "—"
    bench = ui.safe_str(row["benchmark"].iloc[0]) if len(row) else "—"
    style = cn.benchmark_style(code)

    cur = hist.iloc[-1]
    q_cur = hist.index[-1]
    peer = _peer(q_cur) if basis == "top10" else _peer_full(q_cur)
    diff = (cur - peer).dropna().sort_values()
    hhi = float(((cur / 100) ** 2).sum())
    top3 = cur.nlargest(3)

    # ---------------- 档案 ----------------
    h(f'<div class="sec">{nm}（{code}）· {q_cur} '
      f'{"前十大口径" if basis == "top10" else "全持仓口径"}</div>')
    _age = (pd.Timestamp.today() - inc).days / 365.25 if pd.notna(inc) else np.nan
    k = st.columns(5)
    k[0].metric("类型", str(typ))
    k[1].metric("成立", f"{_age:.1f} 年" if np.isfinite(_age) else "—",
                help=str(inc)[:10] if pd.notna(inc) else "")
    k[2].metric("规模", f"{scale:.0f} 亿" if np.isfinite(scale) else "—")
    k[3].metric("经理", str(mgr).split()[0] if mgr and mgr != "—" else "—", help=str(mgr))
    k[4].metric("行业集中度 HHI", f"{hhi:.4f}",
                help="Σ(行业占比)²，越大越集中。前十大口径通常 >0.15，"
                     "全持仓口径 <0.10 算分散")

    # =====================================================================
    # 本页结论（先给判断，再给图）
    # =====================================================================
    _chg_all = (hist.iloc[-1] - hist.iloc[0]).dropna().sort_values() if len(hist) > 1 else pd.Series(dtype=float)
    _drift = float(_chg_all.abs().sum() / 2) if len(_chg_all) else np.nan   # 半周转量
    _tp = top3.sum()
    _dev = diff.tail(2).iloc[::-1] if len(diff) else pd.Series(dtype=float)
    _low = diff.head(2) if len(diff) else pd.Series(dtype=float)
    _q_tbl = cn.fund_period_table(code, q_cur)
    _q_chg = _q_tbl["环比pp"].dropna().sort_values() if not _q_tbl.empty else pd.Series(dtype=float)
    _q_peer = _q_tbl["同行环比pp"].dropna() if not _q_tbl.empty else pd.Series(dtype=float)

    h('<div class="sec">本页结论：这只基金在赌什么、变没变、和同行差在哪</div>')
    _bl = []
    _bl.append(
        f'<b>押注</b>（{q_cur}，{"前十大" if basis == "top10" else "全持仓"}口径）：'
        f'前三大行业 {"、".join(f"{i} {v:.1f}%" for i, v in top3.items())}，'
        f'合计 <b>{_tp:.1f}%</b>，HHI {hhi:.4f}。'
        + ("<b>高度集中</b>——它的涨跌基本由这三个行业决定，"
           "买它等于买这三个行业，不要再叠加同方向的行业基金。"
           if _tp > 45 else
           "<b>相对分散</b>——个股选择比行业判断更重要，"
           "行业方向对它影响有限。"))
    if len(_chg_all):
        _big = _chg_all.abs().nlargest(2)
        _bl.append(
            f'<b>变没变</b>：从 {hist.index[0]} 到 {hist.index[-1]}，'
            + "、".join(f"{i} {_chg_all[i]:+.1f}pp" for i in _big.index)
            + f' 变化最大，结构半周转量 <b>{_drift:.0f}pp</b>。'
            + ("<b>风格已经换过</b>——你当初买它的理由可能不再成立，"
               "要按现在的行业逻辑重新评估。" if _drift > 25 else
               "<b>风格基本稳定</b>——它更像一个「行业贝塔工具」，"
               "按行业逻辑用，不要指望它在行业反转时保护你。"))
    if len(_dev):
        _lowtxt = ("、".join(f"{i} {v:+.1f}pp" for i, v in _low.items())
                   if len(_low) and float(_low.max()) < -0.05 else "没有明显低配（不持有的行业同行也不配）")
        _bl.append(
            f'<b>和同行比</b>：最超配 {"、".join(f"{i} {v:+.1f}pp" for i, v in _dev.items())}；'
            f'最低配 {_lowtxt}。'
            "这些偏离才是它的「主动判断」——如果你已经重仓了它超配的方向，"
            "买它就是重复下注。")
    if len(_q_chg):
        _mine = _q_chg.abs().nlargest(1)
        _ind = _mine.index[0]
        _pv = float(_q_peer.get(_ind, np.nan))
        _mv = float(_q_chg[_ind])
        # 只看方向会把「同行 −4.9pp、它 −90pp」也叫「随大流」，必须带幅度判断
        if np.isfinite(_pv) and np.sign(_pv) == np.sign(_mv) and abs(_pv) > 0.05:
            if abs(_mv) > 2 * abs(_pv) + 1:
                _rel = (f'<b>方向与同行一致，但力度是同行平均的 '
                        f'{abs(_mv) / max(abs(_pv), 0.01):.0f} 倍</b>——'
                        f'这是它自己的大动作，不是随大流，值得单独判断。')
            else:
                _rel = "<b>与同行同向、幅度相近</b>，是随大流，不是独立判断。"
        else:
            _rel = "<b>与同行反向或同行几乎没动</b>，这是它自己的选择，值得单独判断。"
        _bl.append(
            f'<b>最新一季</b>（{_q_tbl.attrs["q_prev"]}→{_q_tbl.attrs["q"]}）：'
            f'它把 {_ind} 调了 {_mv:+.2f}pp，同行平均 {_pv:+.2f}pp——' + _rel)
    h(ui.concl_board(_bl))
    h(f'<div class="tiny">比较基准：<b>{bench}</b>　风格标签：<b>{style}</b>。'
      f'基准只用于判断「它该买什么」，本页的偏离度是与<b>同行平均</b>比，'
      f'不是与基准比（基准权重不在本库内）。</div>')

    # =====================================================================
    # ① 全市场：公募基金整体的行业结构（先看大盘子，再看这一只）
    # =====================================================================
    _nq = len(cn.quarters())
    h(f'<div class="sec">① 先看大盘子：公募基金整体的行业结构变化（{_nq} 期）</div>')
    h('<div class="tiny" style="margin-bottom:3px">'
      '口径：每期把所有样本基金的前十大归一化后等权平均，得到「一家典型的主动权益基金」'
      '的行业结构。这是<b>全市场</b>的钱怎么分，不是单只基金。</div>')
    Mkt = cn.level_matrix("top10")
    cmp_m = cn.compare_periods()
    if not Mkt.empty:
        _mk_mode = st.radio("显示方式", ["前 10 大行业 + 其他", "全部 31 个行业"],
                            horizontal=True, key="mkt_struct_mode",
                            help="31 个色块叠加时后 20 个只有 1–3pp，默认合并显示")
        _mk_last = Mkt.iloc[-1]
        _mk_chg_all = (Mkt.iloc[-1] - Mkt.iloc[0]).dropna()
        _mk_ord = _mk_last.sort_values(ascending=False).index.tolist()
        _mk_fig = rc.fig_ind_structure(
            Mkt, f"全市场行业结构（{Mkt.index[0]}–{Mkt.index[-1]}，前十大等权平均）",
            residual_label="其他（港股/未映射）", label_top=4,
            max_items=10 if _mk_mode.startswith("前 10") else None)
        st.plotly_chart(_mk_fig, width="stretch",
                        config={"displayModeBar": False})
        # 图例必须用「这张图实际用的颜色」：堆叠图按顺序取 12 色高对比调色板，
        # 图例若还按行业固定色画，就会出现「柱子是蓝、图例是红」。
        h(rc.ind_legend_html(_mk_ord, _mk_last, _mk_chg_all, cols=4,
                             colors=(_mk_fig.layout.meta or {}).get("ind_colors")))
        _mk_top = _mk_last.nlargest(4)
        _mk_chg = _mk_chg_all.sort_values()
        _read(h,
              f'最新一期全市场前四大行业：'
              + "、".join(f"{i} {v:.1f}%" for i, v in _mk_top.items())
              + f'，合计 {_mk_top.sum():.1f}%（单只基金可能完全不同）。'
              f'从 {Mkt.index[0]} 到现在，抬升最多的是 '
              + "、".join(f"{i} {v:+.1f}pp" for i, v in _mk_chg.tail(2).iloc[::-1].items())
              + f'，让位最多的是 '
              + "、".join(f"{i} {v:+.1f}pp" for i, v in _mk_chg.head(2).items()) + '。',
              f'全市场的行业结构，是几千只基金各自决策加总的结果，'
              f'所以它变化得很慢——一年下来前四大行业通常只换一到两个。'
              f'这种慢，恰恰说明它是<b>产业结构的映射</b>：A 股上市公司的行业构成、'
              f'各行业的盈利占比在变化，基金持仓只是跟着走。'
              f'当某一层的厚度在两年里翻倍，背后往往是这个行业的<b>市值占比与盈利占比</b>'
              f'同时上升，而不是单纯的资金推动。',
              f'这张图是「基准答案」：<b>先知道全市场在往哪挪，再判断一只基金是顺应还是逆行。</b>'
              f'如果一只基金的行业变化形状和它一致，说明它只是随大流，你买它拿到的是行业贝塔；'
              f'形状明显不同，才是基金经理的主动选择，这时才需要判断他的判断对不对。'
              f'还有一个用法：当某行业在全市场的占比创了新高，说明它已经成为「系统性重要」'
              f'的板块——它的波动会直接影响指数，也会影响其他基金的净值。')

    # ---------------- 指标条形图：31 个行业，悬停即见数值 ----------------
    if not cmp_m.empty:
        h('<div style="font-size:12px;font-weight:700;margin:8px 0 3px">'
          '31 个行业逐个看：选一个指标，条形长度就是数值（悬停显示精确值）</div>')
        _metric = st.radio(
            "指标", ["配置比例 %", "环比 pp", "同比 pp", "配置分位", "拥挤度 CSI"],
            horizontal=True, key="mkt_metric")
        _map = {"配置比例 %": ("配置_前十", "%.2f", "当前配置（前十大归一化，%）"),
                "环比 pp": ("Δ上季pp", "%+.2f", "对上一季的变化（pp，含价格）"),
                "同比 pp": ("Δ上年pp", "%+.2f", "对上年同季的变化（pp）"),
                "配置分位": ("历史分位", "%.0f", f"在自身 {_nq} 期中的排名"),
                "拥挤度 CSI": ("CSI", "%.0f", "(配置分位+价格分位+资金分位)/3")}
        _col, _fmt, _lab = _map[_metric]
        if _col in cmp_m.columns:
            _s = cmp_m[_col].dropna().sort_values()
            _c = [C.COLOR["up"] if v >= 0 else C.COLOR["down"] for v in _s] \
                if _metric.endswith("pp") else [C.COLOR["price"]] * len(_s)
            _fig = go.Figure(go.Bar(
                y=list(_s.index), x=_s.values, orientation="h", marker_color=_c,
                text=[_fmt % v for v in _s.values], textposition="outside",
                textfont=dict(size=9),
                hovertemplate="<b>%{y}</b><br>" + _lab + "：%{x:" +
                              ("+.2f" if _fmt == "%+.2f" else ".2f" if _fmt == "%.2f" else ".0f") +
                              "}<extra></extra>"))
            _fig.add_vline(x=0, line=dict(color=C.COLOR["line"], width=1))
            _fig.update_xaxes(title_text=_lab)
            _fig.update_layout(margin=dict(l=6, r=54, t=16, b=30))
            st.plotly_chart(_base(_fig, "", 620), width="stretch",
                            config={"displayModeBar": False})
            # 每换一个指标，解读跟着换——图换了、结论不换，等于没解读
            _hi3, _lo3 = _s.tail(3).iloc[::-1], _s.head(3)
            # 五元组：最高组说法 / 最低组说法 / 可读性说明 / 经济含义 / 投资含义
            _mk = {
                "配置比例 %": (
                    "押得最重", "押得最轻", "%",
                    "当前配置偏低不代表不好，只代表「全市场没在这儿放钱」；"
                    "要判断该不该买，还要看价格与资金",
                    f'全市场最重的是 {_hi3.index[0]}（{_hi3.iloc[0]:.1f}%），'
                    f'这就是「基准答案」——你手里的基金如果没配它，你其实是在赌它跑输。'
                    f'最轻的 {_lo3.index[0]} 只有 {_lo3.iloc[0]:.1f}%，说明机构整体不看好；'
                    f'这时买入该行业的基金，你赚的是「预期差」，必须先想清楚自己比市场多知道什么。'),
                "环比 pp": (
                    "加得最多", "减得最多", "pp",
                    "环比含价格漂移：跌得多的行业即使没人卖，占比也会掉，"
                    "必须配合「加仓占比」判断",
                    f'本期全市场加得最多的是 {_hi3.index[0]}（{_hi3.iloc[0]:+.2f}pp）。'
                    f'如果这个方向恰好也是你要买的基金的重仓方向，你是在顺着机构的钱进场，'
                    f'好处是有人抬轿、坏处是拥挤；减得最多的 {_lo3.index[0]}'
                    f'（{_lo3.iloc[0]:+.2f}pp）要分两种：主动撤离（加仓占比低）还是价格下跌（占比被动缩水），'
                    f'前者别接，后者可能是机会。'),
                "同比 pp": (
                    "这一年加得最多", "这一年减得最多", "pp",
                    "同比剔除季节性，看的是「这一年机构信了什么」，"
                    "不是「这个季度变了什么」",
                    f'过去一年机构加得最多的是 {_hi3.index[0]}（{_hi3.iloc[0]:+.2f}pp），'
                    f'这是「一年的共识」——共识到了这一步，超额收益通常已经被赚掉了，'
                    f'再追是把一年的涨幅让给别人。减得最多的 {_lo3.index[0]}'
                    f'（{_lo3.iloc[0]:+.2f}pp）如果基本面没坏，往往是下一轮预期差的来源。'),
                "配置分位": (
                    "历史最重", "历史最轻", "分位",
                    "分位只描述「相对自己的历史」，小行业分位高不等于危险，要配合横截面分位",
                    f'{_hi3.index[0]} 处在自己历史的 {_hi3.iloc[0]:.0f} 分位：'
                    f'这是「机构对它最重视的时刻」，但只有横截面分位也高，才是真的重仓拥挤；'
                    f'{_lo3.index[0]} 在 {_lo3.iloc[0]:.0f} 分位，是「多年没人理」的状态，'
                    f'便宜但不代表会涨，要等资金信号。'),
                "拥挤度 CSI": (
                    "最拥挤", "最冷门", "分",
                    "CSI 是水位不是水流：高水位只压缩仓位上限，不是卖出信号",
                    f'{_hi3.index[0]} 的 CSI 是 {_hi3.iloc[0]:.0f}，属于'
                    f'{"极度拥挤" if _hi3.iloc[0] >= 80 else "拥挤" if _hi3.iloc[0] >= 65 else "中性"}：'
                    f'这时买它，边际买家最少、卖出最集中，仓位要压、止损要紧，但不必清仓'
                    f'（本库因子检验显示高 CSI 不预测下跌）。{_lo3.index[0]} '
                    f'（{_lo3.iloc[0]:.0f} 分）反过来：便宜且没人挤，但要等「广度转正 + 主动净买入」'
                    f'再动手，否则可能一直冷下去。')}[_metric]
            _read(h,
                  f'当前指标「{_lab}」。最高：' + "、".join(f"{i} {v:.2f}" for i, v in _hi3.items())
                  + f'；最低：' + "、".join(f"{i} {v:.2f}" for i, v in _lo3.items()) + '。'
                  f'条形长度就是数值，鼠标悬停可看精确值。',
                  f'<b>{_mk[0]}的是 {_hi3.index[0]}</b>，{_mk[1]}的是 {_lo3.index[0]}。'
                  f'{_mk[3]}。切换上面的指标，这张图和这段解读会一起变——'
                  f'同一个行业在不同指标下的位置可能完全相反，这正是要看多个维度的原因。',
                  _mk[4])

        # ---------------- 单个行业：全市场 vs 这只基金 ----------------
        h('<div style="font-size:12px;font-weight:700;margin:8px 0 3px">'
          '也可以只看一个行业：全市场平均 vs 这只基金</div>')
        _ind_opts = list(cmp_m.index)
        # 默认选「这只基金最新一期最重的行业」，而不是列表第一个。
        # 排第一的往往是「农林牧渔」这类它根本不持有的行业，一进来就是一条
        # 贴着 0 的平线，看起来像数据缺失（读者截图反馈过）。
        _default = (_ind_opts[0] if not len(hist)
                    else (hist.iloc[-1].idxmax() if hist.iloc[-1].notna().any()
                          else _ind_opts[0]))
        # 换基金时重置到新基金最重的行业；同一只基金内保留用户选择。
        if (st.session_state.get("_mkt_fund") != code
                or st.session_state.get("mkt_ind") not in _ind_opts):
            st.session_state["_mkt_fund"] = code
            st.session_state["mkt_ind"] = _default
        _ind = st.selectbox("选择行业（可输入搜索）", _ind_opts, key="mkt_ind")
        _f = hist[_ind] if _ind in hist.columns else pd.Series(dtype=float)
        _m = Mkt[_ind] if _ind in Mkt.columns else pd.Series(dtype=float)
        _f_max = float(pd.to_numeric(_f, errors="coerce").max()) if len(_f) else np.nan
        _never = bool(np.isfinite(_f_max) and _f_max <= 0.001)
        _fig2 = go.Figure()
        _fig2.add_trace(go.Scatter(x=list(_m.index), y=_m.values, mode="lines+markers",
                                   name="全市场平均", line=dict(color=C.COLOR["text_dim"], width=2),
                                   marker=dict(size=4),
                                   hovertemplate="%{x}　全市场 %{y:.2f}%<extra></extra>"))
        if len(_f) and not _never:
            _fig2.add_trace(go.Scatter(x=list(_f.index), y=_f.values, mode="lines+markers",
                                       name=nm, line=dict(color=C.COLOR["price"], width=2),
                                       marker=dict(size=4),
                                       hovertemplate="%{x}　本基金 %{y:.2f}%<extra></extra>"))
        elif _never:
            # 不画一条贴 0 的平线（读者会误以为缺数据），改为在 0 处标一条虚线 + 说明
            _fig2.add_hline(y=0, line=dict(color=C.COLOR["price"], width=1.4, dash="dot"),
                            annotation_text=f"{nm}：{len(_f)} 期均为 0（从未持有）",
                            annotation_position="bottom left",
                            annotation_font=dict(size=9, color=C.COLOR["price"]))
        _fig2.update_yaxes(title_text="占前十大 %", tickfont=dict(size=11))
        # 31 个季度标签全标会糊成一片，只标每年 Q1 + 末期
        _tk2 = [x for x in _m.index if str(x).endswith("Q1")] + [list(_m.index)[-1]]
        _fig2.update_xaxes(tickmode="array", tickvals=_tk2, ticktext=_tk2,
                           tickangle=-45, tickfont=dict(size=10))
        st.plotly_chart(_base(_fig2, f"{_ind}：全市场 vs {nm}", 360, legend=True),
                        width="stretch", config={"displayModeBar": False})
        _mf, _mm = (float(_f.iloc[-1]) if len(_f) else np.nan), (float(_m.iloc[-1]) if len(_m) else np.nan)
        _hist_note = ""
        if np.isfinite(_mf) and np.isfinite(_mm):
            _hist_note = (" 历史上两条线" +
                          ("基本重合，说明它在这个行业上没有主动判断。"
                           if abs(_mf - _mm) < 1.5 else
                           "长期分离，说明这是它持续的超配/低配方向。"))
        if _never:
            h(f'<div class="cg-note" style="margin:-4px 0 12px">'
              f'<b>{_ind}：本基金 {len(_f)} 期前十大里从未出现该行业</b>，所以蓝线是 0——'
              f'这不是缺数据，是它确实没买。上面默认已选它最新一期最重的行业；'
              f'想看它的行业押注，请换一个有持仓的行业。全市场平均 '
              f'{_mm:.2f}%，说明这个行业并非没人要，只是这只基金不做。</div>')
        _read(h,
              f'灰线是全市场平均' + ('' if _never else '，蓝线是这只基金') +
              f'。最新一期：全市场 {_mm:.2f}%、本基金 {_mf:.2f}%，'
              f'偏离 <b>{_mf - _mm:+.2f}pp</b>。'
              + ('' if _never else _hist_note),
              f'<b>「全部」看结构，「单个」看偏离。</b>单看这只基金的曲线，无法判断它是'
              f'「主动超配」还是「全市场都这样」；叠上全市场平均，才知道哪些是它的判断。'
              f'这也是为什么本页先给全市场、再给单只基金。',
              f'如果这个行业正是你看好的方向：本基金<b>超配</b>（蓝线在上）它才帮得到你；'
              f'<b>低配</b>的话，你买它等于买了一个「打折的看好」，涨幅会被稀释；'
              f'<b>持平</b>则说明它在这个行业上没有观点，你想要的是行业贝塔，不如直接买行业 ETF。'
              f'反过来，如果这个行业是你不看好的：它超配越多，你越不该买——'
              f'你是在为一个自己不认同的赌注付管理费。')

        # ---------------- 全市场台账 ----------------
        with st.expander("展开：全市场 31 个行业的台账（本期 / 上期 / 上年同期 / 分位）",
                         expanded=False):
            _t = cmp_m.reset_index()[["行业", "配置_前十", "上季_前十", "上年同期_前十",
                                      "Δ上季pp", "Δ上年pp", "历史分位", "加仓占比", "判读"]]
            _t = _t.rename(columns={
                "配置_前十": f"本期%({cmp_m.attrs['q']})",
                "上季_前十": f"上期%({cmp_m.attrs['q_prev']})",
                "上年同期_前十": f"上年同期%({cmp_m.attrs['q_year']})",
                "Δ上季pp": "环比pp", "Δ上年pp": "同比pp",
                "历史分位": f"配置分位({_nq}期)", "加仓占比": f"加仓占比%({cmp_m.attrs['q']})"})
            st.dataframe(_t.sort_values(f"本期%({cmp_m.attrs['q']})", ascending=False),
                         width="stretch", hide_index=True, height=420,
                         column_config={
                             f"本期%({cmp_m.attrs['q']})": st.column_config.NumberColumn(format="%.2f"),
                             f"上期%({cmp_m.attrs['q_prev']})": st.column_config.NumberColumn(format="%.2f"),
                             f"上年同期%({cmp_m.attrs['q_year']})": st.column_config.NumberColumn(format="%.2f"),
                             "环比pp": st.column_config.NumberColumn(format="%+.2f"),
                             "同比pp": st.column_config.NumberColumn(format="%+.2f"),
                             f"配置分位({_nq}期)": st.column_config.ProgressColumn(
                                 format="%.0f", min_value=0, max_value=100),
                             f"加仓占比%({cmp_m.attrs['q']})": st.column_config.NumberColumn(
                                 format="%.0f", help="分母是「有实质变动的基金」"),
                         })

    # =====================================================================
    # ② 这只基金：全部行业的结构漂移
    # =====================================================================
    h('<div class="sec">② 再看这一只：它的 31 个行业结构漂移</div>')
    _c1, _c2 = st.columns([0.72, 0.28], gap="small")
    with _c1:
        _maxi = st.radio("显示方式", ["前 10 大行业 + 其他", "全部 31 个行业"],
                         horizontal=True, key="fund_struct_mode",
                         help="31 个色块叠在一起时，后面 20 个只有 1–3pp，看不清；"
                              "默认合并成「其他 N 个行业」")
    with _c2:
        _qsel = st.selectbox("看哪一季的全部行业", list(hist.index)[::-1],
                             index=0, key="fund_q_pick")
    _mi = 10 if _maxi.startswith("前 10") else None
    fig = rc.fig_ind_structure(
        hist, f"{nm}　行业结构（{hist.index[0]}–{hist.index[-1]}，"
              f"{'前十大内部占比' if basis == 'top10' else '占净值比'}）",
        residual_label=("其他（港股/未映射）" if basis == "top10"
                        else "现金/债券/港股及其他"),
        label_top=4, max_items=_mi)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # 图例即表格：色块 + 行业 + 最新占比 + 相对首期变化
    _first, _last = hist.iloc[0], hist.iloc[-1]
    _chg_all = (_last - _first).dropna()
    _ord = _last.sort_values(ascending=False).index.tolist()
    h(rc.ind_legend_html(_ord, _last, _chg_all, cols=4,
                         colors=(fig.layout.meta or {}).get("ind_colors")))
    h('<div class="tiny" style="margin:-4px 0 8px">图例右侧：'
      '<b>黑字</b>＝最新一期占比；<b>红/绿小字</b>＝相对 '
      f'{hist.index[0]} 的变化（pp）。按最新占比降序。</div>')

    # ---- 同季度横向：这一季 31 个行业各占多少，一次全看到 ----
    h('<div style="font-size:14px;font-weight:700;margin:12px 0 4px">'
      f'{_qsel}：这只基金全部行业的占比（一次列全，不用逐个悬停）</div>')
    _row = hist.loc[_qsel].dropna().sort_values(ascending=False)
    _row = _row[_row >= 0.05]
    _fig_bar = go.Figure(go.Bar(
        y=list(_row.index)[::-1], x=list(_row.values)[::-1], orientation="h",
        marker_color=[rc.ind_color(i) for i in list(_row.index)[::-1]],
        text=[f"{v:.1f}%" for v in list(_row.values)[::-1]], textposition="outside",
        textfont=dict(size=10.5),
        hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>"))
    _fig_bar.update_xaxes(title_text="占前十大 %" if basis == "top10" else "占净值 %",
                          range=[0, max(5, float(_row.max()) * 1.18)],
                          tickfont=dict(size=11))
    _fig_bar.update_yaxes(tickfont=dict(size=11))
    st.plotly_chart(_base(_fig_bar, "", max(320, 22 * len(_row))),
                    width="stretch", config={"displayModeBar": False})

    _up = (_last - _first).dropna().sort_values()
    _big_up, _big_dn = _up.tail(2).iloc[::-1], _up.head(2)
    _share_last = _last.nlargest(4)
    _read(h,
          f'最新一期（{hist.index[-1]}）前四大是 '
          + "、".join(f"{i} {v:.1f}%" for i, v in _share_last.items())
          + f'，合计 {_share_last.sum():.1f}%；从首期（{hist.index[0]}）到现在，'
          f'抬升最多的是 ' + "、".join(f"{i} {v:+.1f}pp" for i, v in _big_up.items())
          + f'，让位最多的是 ' + "、".join(f"{i} {v:+.1f}pp" for i, v in _big_dn.items())
          + f'。结构半周转量 {_drift:.0f}pp（'
          + ("换过风格" if _drift > 25 else "基本稳定") + '）。',
          f'基金的行业结构变化，只有两种来源：<b>主动调仓</b>与<b>价格漂移</b>。'
          f'主动调仓反映基金经理对景气的判断，会持续 2–4 个季度；价格漂移是行业涨跌'
          f'自动改变权重，一个季度 1–2pp 的变动大多来自它。分辨方法很简单：'
          f'看这一层是不是<b>连续三期同向变厚/变薄</b>——连续才是调仓，单季跳动多是价格。'
          f'结构半周转量 {_drift:.0f}pp 衡量的是「这些年它换了多少赛道」，'
          f'数值越大，说明你买它时的理由越可能已经不成立。',
          f'这只基金前四大占 {_share_last.sum():.0f}%，'
          + ("集中度偏高——它的涨跌基本由这四个行业决定，"
             "买它就等于同时押这四个方向，不要再叠加同方向的行业基金，否则是重复下注。"
             if _share_last.sum() > 55 else
             "集中度适中，行业之间有对冲，单一行业回调对净值的冲击相对小。")
          + (f'结构半周转量 {_drift:.0f}pp，属于'
             + ("「换过风格」——你当初买它的理由可能已不成立，要按它现在的行业逻辑重新评估。"
                if _drift > 25 else "「基本稳定」——它更像一个行业贝塔工具，按行业逻辑用即可。")
             if np.isfinite(_drift) else ''))

    # ---------------- 上期 / 本期 / 上年同期 台账 ----------------
    if not _q_tbl.empty:
        h('<div style="font-size:12px;font-weight:700;margin:8px 0 3px">'
          f'本期 vs 上期 vs 上年同期（{_q_tbl.attrs["q"]} / {_q_tbl.attrs["q_prev"]} / '
          f'{_q_tbl.attrs["q_year"]}）</div>')
        _show = _q_tbl.reset_index()
        _show = _show[_show[["本期", "上期", "上年同期"]].abs().max(axis=1) > 0.05]
        _show = _show.reindex(_show["本期"].sort_values(ascending=False).index)
        _show = _show.rename(columns={"本期": f"本期%({_q_tbl.attrs['q']})",
                                      "上期": f"上期%({_q_tbl.attrs['q_prev']})",
                                      "上年同期": f"上年同期%({_q_tbl.attrs['q_year']})",
                                      "环比pp": "环比pp", "同比pp": "同比pp",
                                      "同行本期": "同行平均%", "同行环比pp": "同行环比pp",
                                      "相对同行pp": "相对同行pp"})
        st.dataframe(_show.round(2), width="stretch", hide_index=True, height=330,
                     column_config={
                         f"本期%({_q_tbl.attrs['q']})": st.column_config.NumberColumn(format="%.2f"),
                         f"上期%({_q_tbl.attrs['q_prev']})": st.column_config.NumberColumn(format="%.2f"),
                         f"上年同期%({_q_tbl.attrs['q_year']})": st.column_config.NumberColumn(format="%.2f"),
                         "环比pp": st.column_config.NumberColumn(format="%+.2f"),
                         "同比pp": st.column_config.NumberColumn(format="%+.2f"),
                         "同行平均%": st.column_config.NumberColumn(format="%.2f"),
                         "同行环比pp": st.column_config.NumberColumn(format="%+.2f"),
                         "相对同行pp": st.column_config.NumberColumn(
                             format="%+.2f", help="本期 − 同行平均；正数＝超配"),
                     })
        h(f'<div class="tiny">口径：{"前十大归一化" if basis == "top10" else "全持仓占净值比"}。'
          f'「同比」是同一季度（{_q_tbl.attrs["q_year"]}）对本期，'
          f'消除季节性；「同行环比」是同期样本基金平均值的变化，'
          f'用来区分「它自己的选择」与「全行业都在做」。'
          f'两者相减＝相对同行的超额变化，才是它的主动判断。</div>')

    # =====================================================================
    # ② 与同行的偏离
    # =====================================================================
    h('<div class="sec">③ 它和同行差在哪：超配的行业、低配的行业</div>')
    # 先剔掉「几乎为 0」的偏离：不持有某行业时 diff=0，画出来是一排空条，
    # 既占位置又让人以为「没有偏离」，实际是「这只基金不持有」。
    _dv = diff[diff.abs() >= 0.05]
    if _dv.empty:
        _dv = diff
    d = _dv.reindex(_dv.abs().sort_values(ascending=False).index).head(12).iloc[::-1]
    fig2 = go.Figure(go.Bar(
        y=[f"{i}　{v:+.1f}pp" for i, v in zip(d.index, d)],
        x=d.values, orientation="h",
        marker_color=[C.COLOR["up"] if v > 0 else C.COLOR["down"] for v in d],
        customdata=np.stack([cur.reindex(d.index), peer.reindex(d.index)], axis=-1),
        hovertemplate=("%{y}<br>该基金 %{customdata[0]:.2f}%<br>"
                       "同行平均 %{customdata[1]:.2f}%<extra></extra>")))
    fig2.add_vline(x=0, line=dict(color=C.COLOR["line"], width=1))
    fig2.update_xaxes(title_text="该基金配置 − 同行平均（pp）")
    st.plotly_chart(_base(fig2, "相对同行的行业偏离（前 12 大偏离，单位 pp）", 420),
                    width="stretch", config={"displayModeBar": False})
    _read(h,
          f'横轴是「这只基金 − 同行平均」：右边＝比同行押得更重，左边＝更轻。'
          f'最超配 ' + "、".join(f"{i}（{v:+.1f}pp）" for i, v in _dev.items())
          + f'；最低配 ' + "、".join(f"{i}（{v:+.1f}pp）" for i, v in _low.items()) + '。',
          f'<b>偏离才是这只基金的「观点」</b>：它跑赢或跑输同行，几乎完全由这些偏离决定。'
          f'如果它超配的方向正好是你已经重仓的，买它就是重复下注；'
          f'如果你看好某个行业而它低配，那它帮不了你。'
          f'反过来，如果你不看好它最超配的行业，就不要买——'
          f'你买的是一个自己不认同的赌注。',
          f'把它最超配的两个方向拿去对照你自己的持仓：'
          + ("<b>已经重仓 → 不要买</b>，你的组合会从「分散」变成「押一个行业」，"
             "行业回调时两个产品一起跌。" if len(_dev) else "")
          + (f' 你的核心看好方向如果是 {_low.index[0]}，而它低配 {_low.iloc[0]:+.1f}pp，'
             f'那它无法表达你的观点，不如直接买该行业 ETF 或超配该行业的基金。'
             if len(_low) and float(_low.min()) < -0.5 else '')
          + ' 只有当「它超配的方向」与「你看好的方向」重叠、且这个方向当前不拥挤时，'
            '它才是合适的载体；重叠但拥挤（CSI≥80），就应该减半仓位而不是加。')

    # =====================================================================
    # ④ 拥挤暴露：它的持仓有多「堵」
    # =====================================================================
    h('<div class="sec">④ 它的持仓有多「堵」：拥挤暴露</div>')
    _ex = cn.crowding_exposure(code)
    if _ex.empty:
        h('<div class="cg-note">该基金没有可比的行业结构，无法计算拥挤暴露。</div>')
    else:
        _ex = _ex.dropna(subset=["本基金加权CSI"])
        _last = _ex.index[-1]
        _cw = float(_ex.loc[_last, "本基金加权CSI"])
        _mw = float(_ex.loc[_last, "全市场加权CSI"])
        _ch = float(_ex.loc[_last, "本基金高拥挤"])
        _mh = float(_ex.loc[_last, "全市场高拥挤"])
        _own = float((_ex["本基金加权CSI"] <= _cw).mean() * 100)
        _gap = _cw - _mw

        _fg = go.Figure()
        # 五档第一层只画两条关键线：80 以上是「会被一起卖」的区域，
        # 65 以上是「偏拥挤」。画满五档会盖住两条曲线。
        _fg.add_hrect(y0=80, y1=100, fillcolor="rgba(198,40,40,0.10)",
                      line_width=0, layer="below")
        _fg.add_hrect(y0=65, y1=80, fillcolor="rgba(232,113,10,0.08)",
                      line_width=0, layer="below")
        _fg.add_trace(go.Scatter(
            x=list(_ex.index), y=_ex["全市场加权CSI"], mode="lines",
            name="全市场（前十大加权）",
            line=dict(color="#7C8B9C", width=2.0, dash="dot"),
            hovertemplate="%{x}<br>全市场 %{y:.1f}<extra></extra>"))
        _fg.add_trace(go.Scatter(
            x=list(_ex.index), y=_ex["本基金加权CSI"], mode="lines+markers",
            name="本基金",
            line=dict(color=C.COLOR["price"], width=2.6), marker=dict(size=3.5),
            hovertemplate="%{x}<br>本基金 %{y:.1f}<extra></extra>"))
        _fg.add_annotation(x=_ex.index[-1], y=_cw, xanchor="left", xshift=4,
                           text=f"{_cw:.0f}", showarrow=False,
                           font=dict(size=10, color=C.COLOR["price"]))
        _fg.update_yaxes(title_text="持仓加权 CSI（0＝最冷，100＝最挤）",
                         range=[0, 104], tickfont=dict(size=10))
        # 每期都标：只有 39 期，横轴宽 1000px，逐个标得下
        _tk = list(_ex.index)
        _fg.update_xaxes(tickmode="array", tickvals=_tk, ticktext=_tk,
                         tickangle=-90, tickfont=dict(size=9, color="#40505F"))
        _f = _base(_fg, "", 430, legend=True)
        # 左：纵轴标题是竖排中文，l=6 会跟刻度数字叠在一起；右：末端数值标注在
        # 坐标区外，不留白会被裁成半个数字。
        _f.update_layout(margin=dict(l=64, r=40, t=18, b=76))
        st.plotly_chart(_f, width="stretch", config={"displayModeBar": False})

        h(ui.stat_strip([
            ("持仓加权 CSI", f"{_cw:.0f}", f"{_last}｜全市场 {_mw:.0f}"),
            ("高拥挤仓位（≥80）", f"{_ch:.0f}%", f"全市场 {_mh:.0f}%"),
            ("与全市场之差", f"{_gap:+.0f}", "正＝比同行更堵"),
            ("自身历史分位", f"{_own:.0f}", f"{len(_ex)} 期中第 "
                                            f"{int((_ex['本基金加权CSI'] <= _cw).sum())} 低"),
        ]))

        _peak_q = _ex["本基金加权CSI"].idxmax()
        _peak_v = float(_ex["本基金加权CSI"].max())
        _read(h,
              f'两条线都是「按前十大权重加权的 CSI」：本基金最新 <b>{_cw:.0f}</b> 分，'
              f'全市场 <b>{_mw:.0f}</b> 分（差 {_gap:+.0f}）。'
              f'按阈值口径，它 <b>{_ch:.0f}%</b> 的仓位压在 ≥80 分的行业上，'
              f'全市场是 {_mh:.0f}%。历史最高出现在 {_peak_q}（{_peak_v:.0f} 分）。'
              f'<br>两个口径要一起看：加权 CSI 连续、平滑，回答「整体有多贵」；'
              f'高拥挤仓位是阈值口径，回答「多少仓位会在转向时被一起卖」，'
              f'它会在行业跨过 80 分时整块跳变——所以别拿单期变化当趋势。',
              f'<b>拥挤暴露衡量的是「脆弱性」，不是「收益潜力」。</b>'
              f'同样看好电子，40% 仓位在电子 CSI=40 时是「便宜的重仓」，'
              f'在 CSI=95 时是「随时会被踩踏的重仓」——持仓没变，风险变了。'
              f'这也是为什么本页把它单列：行业结构告诉你「押了什么」，'
              f'拥挤暴露告诉你「这个押注现在有多容易被挤出去」。',
              f'<b>怎么用</b>：① 暴露高于全市场且加权 CSI ≥80 → 这只基金在替你在拥挤交易上加杠杆，'
              f'仓位打七折或换成暴露更低的同类；② 暴露低于全市场 → 它没在这次抱团里，'
              f'不必因为它「没跟上」而赎回；③ 看变化：暴露从 30% 冲到 90% 往往意味着'
              f'基金经理在追高，比单看行业结构更早发现风格漂移。'
              f'注意：本指标只覆盖前十大（季报口径），中小盘基金的前十大占比低，'
              f'读数会被低估。')

    # =====================================================================
    # ③ 前十大
    # =====================================================================
    # 优先用**最新一季**（2026Q3）的前十大，与上面的结构图、结论同期；
    # 季报只披露前十大，所以这一张表就是「季报能看到的全部持仓」。
    # 若该季无数据，再回退到最近的全持仓期（更全但更旧）。
    tp = _top_holdings(code, q)
    _hold_q = q
    _hold_tag = "季报前十大，即季报能看到的全部持仓"
    if tp.empty:
        tp = _top_holdings(code, qf)
        _hold_q = qf
        _hold_tag = "全持仓（半年报，全部持仓的前 12 大）"
    h(f'<div class="sec">⑤ 它的前十大到底是谁（{_hold_q}）</div>')
    if tp.empty:
        h('<div class="cg-note">该期无持仓明细。</div>')
        return
    tp = tp.rename(columns={"stock_code": "代码", "stock_name": "股票",
                            "industry": "行业", "weight_pct": "占净值%",
                            "shares": "持股数(万股)", "mv_yi": "持仓市值(亿)",
                            "is_hk": "港股"})
    # 东财明细里「持股数」本身就是万股、「持仓市值」是万元，再除 1e4 会全变 0
    tp["持股数(万股)"] = pd.to_numeric(tp["持股数(万股)"], errors="coerce").round(1)
    tp["港股"] = np.where(tp["港股"].fillna(0) == 1, "是", "")
    tp["行业"] = tp["行业"].fillna("未映射")
    st.dataframe(
        tp[["代码", "股票", "行业", "占净值%", "持股数(万股)", "持仓市值(亿)", "港股"]],
        width="stretch", hide_index=True, height=380,
        column_config={"占净值%": st.column_config.NumberColumn(format="%.2f"),
                       "持股数(万股)": st.column_config.NumberColumn(format="%,.1f"),
                       "持仓市值(亿)": st.column_config.NumberColumn(format="%.2f")})
    _sum = float(tp["占净值%"].sum())
    _top3s = float(tp["占净值%"].head(3).sum())
    _ind_cnt = tp["行业"].value_counts()
    h(f'<div class="cg-note">前十大合计占净值 <b>{_sum:.1f}%</b>，前 3 大合计 '
      f'<b>{_top3s:.1f}%</b>；行业分布上 '
      + "、".join(f"{i} {v} 只" for i, v in _ind_cnt.head(3).items())
      + f'。口径：{_hold_tag}。<br><b>所以呢</b>：季报只披露前十大，'
      f'你看到的永远只是「最重要的那部分」；'
      + ('这只基金前十大就占了净值的绝大多数，前十大几乎等于全部组合，'
         '可以直接把它当前十大来研究。' if _sum > 60 else
         f'只有 {_sum:.0f}%，说明还有 {100 - _sum:.0f}% 的仓位看不见——'
         f'这部分只能靠净值反推，不能靠季报。')
      + ('前三大就占 {:.1f}%，个股集中度很高，单只股票的黑天鹅会直接反映在净值上。'
         .format(_top3s) if _top3s > 30 else '前三大占比不高，个股风险相对分散。')
      + '</div>')
    h('<div class="tiny">数据：东方财富基金定期报告（半年报全持仓、季报前十大）。'
      '本页所有对比都是「与同行平均」，不是与指数比；'
      '要看它有没有跑赢基准，需要基准指数收益，本库尚未接入。</div>')
