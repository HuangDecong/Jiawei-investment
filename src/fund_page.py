# -*- coding: utf-8 -*-
"""② 单只基金透视：这只基金到底在赌什么、结构有没有漂、和同行差在哪。

页面顺序按「先看结构、再看偏离、最后看个股」：
    1. 行业结构漂移（全部 31 个行业，31 期）——第一眼就要看到它押注的变化；
    2. 与同行的偏离——哪些是它自己的判断，哪些是全员都在做；
    3. 前十大——季报能看到的最后一块拼图。

口径
----
· 结构有两条序列，页面上可切换：
    前十大（31 期，归一化 100%）——季频、最长，看「股票里怎么分」；
    全持仓（15 期，占净值比）——半年频、最准，看「净值里放了多少」；
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


def _read(h, read: str, concl: str) -> None:
    """统一的「读图 + 结论」块——每张图都必须有。"""
    h(f'<div style="background:{C.COLOR["panel"]};border-left:3px solid {C.COLOR["price"]};'
      f'padding:9px 12px;border-radius:0 6px 6px 0;margin:4px 0 14px">'
      f'<div style="font-size:12px;line-height:1.9"><b>读图</b>：{read}</div>'
      f'<div style="font-size:12.5px;line-height:1.9;margin-top:5px">'
      f'<b style="color:{C.COLOR["threshold"]}">结论</b>：{concl}</div></div>')


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


def render(h) -> None:
    funds = _funds()
    q = cn.latest()
    pf = cn.pair_full()
    if funds.empty or not pf:
        h('<div class="cg-note">基金档案尚未抓取完成，先运行 '
          '<span class="mono">python tools/fetch_fund_profile.py</span>。</div>')
        return
    qf = pf[1]

    # ---------------- 选择器 ----------------
    c1, c2 = st.columns([0.5, 0.5])
    with c1:
        opts = (funds["code"] + "　" + funds["name"].astype(str)
                + "　" + funds["scale_yi"].map(lambda x: f"{x:.1f}亿"))
        pick = st.selectbox("选择基金（可直接输入代码或名称搜索，按规模排序）",
                            opts.tolist(), index=0, key="fund_pick")
        code = pick.split("　")[0].strip()
    with c2:
        _mode = st.radio("结构口径", ["前十大（31 期，季频）", "全持仓（15 期，半年频）"],
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
    nm = row["name"].iloc[0] if len(row) else code
    typ = row["ftype"].iloc[0] if len(row) else "—"
    inc = row["inception"].iloc[0] if len(row) else pd.NaT
    scale = row["scale_yi"].iloc[0] if len(row) else np.nan
    mgr = row["managers"].iloc[0] if len(row) else "—"
    bench = row["benchmark"].iloc[0] if len(row) else "—"
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
    h('<div class="cg-note" style="line-height:1.9">'
      + "<br>".join(f"{i}. {x}" for i, x in enumerate(_bl, 1)) + "</div>")
    h(f'<div class="tiny">比较基准：<b>{bench}</b>　风格标签：<b>{style}</b>。'
      f'基准只用于判断「它该买什么」，本页的偏离度是与<b>同行平均</b>比，'
      f'不是与基准比（基准权重不在本库内）。</div>')

    # =====================================================================
    # ① 全市场：公募基金整体的行业结构（先看大盘子，再看这一只）
    # =====================================================================
    h('<div class="sec">① 先看大盘子：公募基金整体的行业结构变化（31 期）</div>')
    h('<div class="tiny" style="margin-bottom:3px">'
      '口径：每期把所有样本基金的前十大归一化后等权平均，得到「一家典型的主动权益基金」'
      '的行业结构。这是<b>全市场</b>的钱怎么分，不是单只基金。</div>')
    Mkt = cn.level_matrix("top10")
    cmp_m = cn.compare_periods()
    if not Mkt.empty:
        st.plotly_chart(
            rc.fig_ind_structure(
                Mkt, f"全市场行业结构（{Mkt.index[0]}–{Mkt.index[-1]}，前十大等权平均）",
                residual_label="其他（港股/未映射）", label_top=4),
            width="stretch", config={"displayModeBar": False})
        h(rc.ind_legend_html(list(Mkt.columns)))
        _mk_last = Mkt.iloc[-1]
        _mk_top = _mk_last.nlargest(4)
        _mk_chg = (Mkt.iloc[-1] - Mkt.iloc[0]).dropna().sort_values()
        _read(h,
              f'最新一期全市场前四大行业：'
              + "、".join(f"{i} {v:.1f}%" for i, v in _mk_top.items())
              + f'，合计 {_mk_top.sum():.1f}%（单只基金可能完全不同）。'
              f'从 {Mkt.index[0]} 到现在，抬升最多的是 '
              + "、".join(f"{i} {v:+.1f}pp" for i, v in _mk_chg.tail(2).iloc[::-1].items())
              + f'，让位最多的是 '
              + "、".join(f"{i} {v:+.1f}pp" for i, v in _mk_chg.head(2).items()) + '。',
              f'这张图是「基准答案」：<b>先知道全市场在往哪挪，再判断一只基金是顺应还是逆行。</b>'
              f'如果一只基金的行业变化和这张图的形状一致，它只是随大流；'
              f'形状明显不同，才是它的主动选择。')

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
                "配置分位": ("历史分位", "%.0f", "在自身 31 期中的排名"),
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
            _mk = {"配置比例 %": ("押得最重", "押得最轻", "%", "当前配置偏低不代表不好，只代表"
                                 "「全市场没在这儿放钱」；要判断该不该买，还要看价格与资金"),
                   "环比 pp": ("加得最多", "减得最多", "pp", "环比含价格漂移：跌得多的行业"
                              "即使没人卖，占比也会掉，必须配合「加仓占比」判断"),
                   "同比 pp": ("这一年加得最多", "这一年减得最多", "pp", "同比剔除季节性，"
                              "看的是「这一年机构信了什么」，不是「这个季度变了什么」"),
                   "配置分位": ("历史最重", "历史最轻", "分位", "分位只描述「相对自己的历史」，"
                               "小行业分位高不等于危险，要配合横截面分位"),
                   "拥挤度 CSI": ("最拥挤", "最冷门", "分", "CSI 是水位不是水流：高水位只压缩"
                                 "仓位上限，不是卖出信号")}[_metric]
            _read(h,
                  f'当前指标「{_lab}」。最高：' + "、".join(f"{i} {v:.2f}" for i, v in _hi3.items())
                  + f'；最低：' + "、".join(f"{i} {v:.2f}" for i, v in _lo3.items()) + '。'
                  f'条形长度就是数值，鼠标悬停可看精确值。',
                  f'<b>{_mk[0]}的是 {_hi3.index[0]}</b>，{_mk[1]}的是 {_lo3.index[0]}。'
                  f'{_mk[3]}。切换上面的指标，这张图和这段解读会一起变——'
                  f'同一个行业在不同指标下的位置可能完全相反，这正是要看多个维度的原因。')

        # ---------------- 单个行业：全市场 vs 这只基金 ----------------
        h('<div style="font-size:12px;font-weight:700;margin:8px 0 3px">'
          '也可以只看一个行业：全市场平均 vs 这只基金</div>')
        _ind = st.selectbox("选择行业（可输入搜索）", list(cmp_m.index), key="mkt_ind")
        _f = hist[_ind] if _ind in hist.columns else pd.Series(dtype=float)
        _m = Mkt[_ind] if _ind in Mkt.columns else pd.Series(dtype=float)
        _fig2 = go.Figure()
        _fig2.add_trace(go.Scatter(x=list(_m.index), y=_m.values, mode="lines+markers",
                                   name="全市场平均", line=dict(color=C.COLOR["text_dim"], width=2),
                                   marker=dict(size=4),
                                   hovertemplate="%{x}　全市场 %{y:.2f}%<extra></extra>"))
        if len(_f):
            _fig2.add_trace(go.Scatter(x=list(_f.index), y=_f.values, mode="lines+markers",
                                       name=nm, line=dict(color=C.COLOR["price"], width=2),
                                       marker=dict(size=4),
                                       hovertemplate="%{x}　本基金 %{y:.2f}%<extra></extra>"))
        _fig2.update_yaxes(title_text="占前十大 %")
        _fig2.update_xaxes(tickangle=-45)
        st.plotly_chart(_base(_fig2, f"{_ind}：全市场 vs {nm}", 340, legend=True),
                        width="stretch", config={"displayModeBar": False})
        _mf, _mm = (float(_f.iloc[-1]) if len(_f) else np.nan), (float(_m.iloc[-1]) if len(_m) else np.nan)
        _hist_note = ""
        if np.isfinite(_mf) and np.isfinite(_mm):
            _hist_note = (" 历史上两条线" +
                          ("基本重合，说明它在这个行业上没有主动判断。"
                           if abs(_mf - _mm) < 1.5 else
                           "长期分离，说明这是它持续的超配/低配方向。"))
        _read(h,
              f'灰线是全市场平均，蓝线是这只基金。最新一期：全市场 '
              f'{_mm:.2f}%、本基金 {_mf:.2f}%，偏离 <b>{_mf - _mm:+.2f}pp</b>。'
              + _hist_note,
              f'<b>「全部」看结构，「单个」看偏离。</b>单看这只基金的曲线，无法判断它是'
              f'「主动超配」还是「全市场都这样」；叠上全市场平均，才知道哪些是它的判断。'
              f'这也是为什么本页先给全市场、再给单只基金。')

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
                "历史分位": "配置分位(31期)", "加仓占比": f"加仓占比%({cmp_m.attrs['q']})"})
            st.dataframe(_t.sort_values(f"本期%({cmp_m.attrs['q']})", ascending=False),
                         width="stretch", hide_index=True, height=420,
                         column_config={
                             f"本期%({cmp_m.attrs['q']})": st.column_config.NumberColumn(format="%.2f"),
                             f"上期%({cmp_m.attrs['q_prev']})": st.column_config.NumberColumn(format="%.2f"),
                             f"上年同期%({cmp_m.attrs['q_year']})": st.column_config.NumberColumn(format="%.2f"),
                             "环比pp": st.column_config.NumberColumn(format="%+.2f"),
                             "同比pp": st.column_config.NumberColumn(format="%+.2f"),
                             "配置分位(31期)": st.column_config.ProgressColumn(
                                 format="%.0f", min_value=0, max_value=100),
                             f"加仓占比%({cmp_m.attrs['q']})": st.column_config.NumberColumn(
                                 format="%.0f", help="分母是「有实质变动的基金」"),
                         })

    # =====================================================================
    # ② 这只基金：全部行业的结构漂移
    # =====================================================================
    h('<div class="sec">② 再看这一只：它的 31 个行业结构漂移</div>')
    fig = rc.fig_ind_structure(
        hist, f"{nm}　行业结构（{hist.index[0]}–{hist.index[-1]}，"
              f"{'前十大内部占比' if basis == 'top10' else '占净值比'}）",
        residual_label=("其他（港股/未映射）" if basis == "top10"
                        else "现金/债券/港股及其他"),
        label_top=4)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    h(rc.ind_legend_html(list(hist.columns)))

    _first, _last = hist.iloc[0], hist.iloc[-1]
    _up = (_last - _first).dropna().sort_values()
    _big_up, _big_dn = _up.tail(2).iloc[::-1], _up.head(2)
    _share_last = _last.nlargest(4)
    _read(h,
          f'图里每一层是一个行业，层越厚＝在组合里占比越高；'
          f'按 {len(hist)} 期的均值从大到小排，所以下面的层是它的「基本盘」。'
          f'最新一期（{hist.index[-1]}）前四大是 '
          + "、".join(f"{i} {v:.1f}%" for i, v in _share_last.items())
          + f'，合计 {_share_last.sum():.1f}%。'
          f'从首期到现在，抬升最多的是 '
          + "、".join(f"{i} {v:+.1f}pp" for i, v in _big_up.items())
          + f'，让位最多的是 '
          + "、".join(f"{i} {v:+.1f}pp" for i, v in _big_dn.items()) + '。',
          f'<b>结构图看三件事</b>：① 有没有一层越来越厚——那是它在持续加注的方向；'
          f'② 有没有一层突然消失——那是清仓式换仓；③ 总面积（全持仓口径）有没有变——'
          f'那是仓位升降。<b>本期这只基金</b>：'
          + (f'结构半周转量 {_drift:.0f}pp，属于'
             + ("「换过风格」" if _drift > 25 else "「基本稳定」")
             + f'；' if np.isfinite(_drift) else '')
          + (f'前四大占 {_share_last.sum():.0f}%，集中度'
             + ("偏高，行业风险大于个股风险" if _share_last.sum() > 55 else "适中")
             + '。')
          + ' 单季 1–2pp 的波动多是价格漂移，连续 3 期同向才是真的调结构。')

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
          f'你买的是一个自己不认同的赌注。')

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
    h(f'<div class="sec">④ 它的前十大到底是谁（{_hold_q}）</div>')
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
