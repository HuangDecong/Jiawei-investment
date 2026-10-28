# -*- coding: utf-8 -*-
"""单只基金透视：这只基金到底在赌什么、和同行差在哪、结构有没有漂。

为什么需要这一页
----------------
前面几页都是「全市场视角」，回答「机构整体在做什么」。
但投资者真正持有的是**一只具体的基金**，需要回答三个问题：

1. 它的行业结构和同行差在哪（超配了什么、低配了什么）；
2. 它的结构有没有漂（同一只基金两年前的行业分布和现在一样吗）；
3. 它的前十大到底是谁（季报只披露前十大，这十只就是全部可见信息）。

口径
----
· 结构用**全持仓期**（半年报）的占净值比：季报只披露前十大，
  算出来的行业占比会把每个行业都压低，不能用来做「超配/低配」判断；
· 对比基准是**全市场样本基金的中位数**，不是沪深300——这里问的是
  「和同行比」，不是「和市场比」；后者需要指数权重，本库没有。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import canon as cn
import config as C


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


@st.cache_data(show_spinner=False)
def _funds() -> pd.DataFrame:
    """可选的基金清单：有档案、有规模，按规模排序。"""
    p = cn.profiles()
    if p.empty:
        return pd.DataFrame(columns=["code", "name", "scale_yi"])
    d = p.dropna(subset=["scale_yi"]).copy()
    d = d[d["scale_yi"] > 0]
    return d.sort_values("scale_yi", ascending=False)


@st.cache_data(show_spinner=False)
def _history(code: str) -> pd.DataFrame:
    """该基金在各全持仓期的行业占比（占净值比 %）。"""
    rows = {}
    for q in cn.full_quarters():
        p = cn.panel_full(q, normalize=False)
        if p.empty or code not in p.index:
            continue
        rows[q] = p.loc[code]
    return pd.DataFrame(rows).T.sort_index() if rows else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _peer(q: str) -> pd.Series:
    """同期同行的行业中位数。"""
    p = cn.panel_full(q, normalize=False)
    if p.empty:
        return pd.Series(dtype=float)
    u = cn.universe(q)
    idx = p.index.intersection(list(u)) if u else p.index
    return p.loc[idx].median()


@st.cache_data(show_spinner=False)
def _top_holdings(code: str, q: str) -> pd.DataFrame:
    with cn._con() as con:
        d = pd.read_sql(
            "SELECT stock_code, stock_name, industry, weight_pct, shares, mv, is_hk "
            "FROM fund_holdings WHERE code=? AND quarter=? "
            "ORDER BY weight_pct DESC LIMIT 12", con, params=(code, q))
    if not d.empty:
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
    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        opts = (funds["code"] + "　" + funds["name"].astype(str)
                + "　" + funds["scale_yi"].map(lambda x: f"{x:.1f}亿"))
        pick = st.selectbox("选择基金（按规模排序，可输入代码或名称搜索）",
                            opts.tolist(), index=0, key="fund_pick")
        code = pick.split("　")[0].strip()
    with c2:
        st.caption("口径：结构用全持仓期（半年报）的占净值比；"
                   "对比基准是同行的中位数，不是沪深300——这里问的是「和同行比」。")

    hist = _history(code)
    if hist.empty:
        h(f'<div class="cg-note">该基金在库内没有全持仓期数据（{code}）。</div>')
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
    peer = _peer(hist.index[-1])
    diff = (cur - peer).dropna().sort_values()
    hhi = float(((cur / 100) ** 2).sum())
    top3 = cur.nlargest(3)

    # ---------------- 档案 ----------------
    h(f'<div class="sec">{nm}（{code}）· {hist.index[-1]} 全持仓</div>')
    _age = (pd.Timestamp.today() - inc).days / 365.25 if pd.notna(inc) else np.nan
    k = st.columns(5)
    k[0].metric("类型", str(typ))
    k[1].metric("成立", f"{_age:.1f} 年" if np.isfinite(_age) else "—",
                help=str(inc)[:10] if pd.notna(inc) else "")
    k[2].metric("规模", f"{scale:.0f} 亿" if np.isfinite(scale) else "—")
    k[3].metric("经理", str(mgr).split()[0] if mgr and mgr != "—" else "—",
                help=str(mgr))
    k[4].metric("行业集中度 HHI", f"{hhi:.4f}",
                help="Σ(行业占比)²，越大越集中。0.05 以下算分散")

    h(f'<div class="cg-note">比较基准：<b>{bench}</b>　风格标签：<b>{style}</b>。'
      f'当前前三大行业 ' + "、".join(
          f'{i} {v:.1f}%' for i, v in top3.items())
      + f'，合计 <b>{top3.sum():.1f}%</b>。'
      f'<b>所以呢</b>：' +
      ("行业集中度偏高，这只基金的涨跌基本由前三个行业决定，"
       "买它等于买这三个行业，不要再叠加同方向的行业基金。"
       if top3.sum() > 45 else
       "行业相对分散，个股选择比行业判断更重要。")
      + '</div>')

    # ---------------- 1. 与同行的差异 ----------------
    h('<div class="sec">① 它和同行差在哪：超配的行业、低配的行业</div>')
    d = diff.reindex(diff.abs().sort_values(ascending=False).index).head(12).iloc[::-1]
    fig = go.Figure(go.Bar(
        y=[f"{i}　{v:+.1f}pp" for i, v in zip(d.index, d)],
        x=d.values, orientation="h",
        marker_color=[C.COLOR["up"] if v > 0 else C.COLOR["down"] for v in d],
        hovertemplate="%{y}<br>与同行中位差 %{x:+.2f}pp<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=C.COLOR["line"], width=1))
    fig.update_xaxes(title_text="该基金配置 − 同行中位（pp）")
    st.plotly_chart(_base(fig, "相对同行的行业偏离（前 12 大偏离）", 420),
                    width="stretch", config={"displayModeBar": False})
    _hi, _lo = diff.tail(2).iloc[::-1], diff.head(2)
    h(f'<div class="cg-note">最超配：'
      + "、".join(f'{i}（{v:+.1f}pp）' for i, v in _hi.items())
      + f'；最低配：' + "、".join(f'{i}（{v:+.1f}pp）' for i, v in _lo.items())
      + '。<br><b>所以呢</b>：这些偏离才是这只基金的「主动判断」。'
        '如果它超配的行业正好是你已经重仓的方向，买它就是重复下注；'
        '如果你看好某个行业而它低配，那它帮不了你。</div>')

    # ---------------- 2. 结构漂移 ----------------
    h('<div class="sec">② 它的结构有没有漂：同一只基金这些年押注的变化</div>')
    top_ind = cur.nlargest(6).index.tolist()
    M = hist[top_ind].copy()
    M["其他"] = hist.drop(columns=top_ind, errors="ignore").sum(axis=1)
    fig2 = go.Figure()
    palette = ["#1F6FEB", "#2EA043", "#D1242F", "#E8710A", "#6E4AB8", "#0F8B8D", "#C3CBD4"]
    for i, col in enumerate(M.columns):
        fig2.add_trace(go.Scatter(
            x=list(M.index), y=M[col].values, mode="lines", name=col,
            stackgroup="one", line=dict(width=0.6, color=palette[i % len(palette)]),
            fillcolor=palette[i % len(palette)],
            hovertemplate=f"{col}<br>%{{x}}：%{{y:.1f}}%<extra></extra>"))
    fig2.update_yaxes(title_text="占净值比 %")
    fig2.update_xaxes(tickangle=-45)
    st.plotly_chart(_base(fig2, "行业结构的漂移（前 6 大行业 + 其他）", 400, legend=True),
                    width="stretch", config={"displayModeBar": False})
    if len(hist) >= 3:
        _first, _last = hist.iloc[0], hist.iloc[-1]
        _chg = (_last - _first).dropna().sort_values()
        h(f'<div class="cg-note">从 {hist.index[0]} 到 {hist.index[-1]}，'
          f'加得最多的是 <b>{_chg.index[-1]}（{_chg.iloc[-1]:+.1f}pp）</b>、'
          f'减得最多的是 <b>{_chg.index[0]}（{_chg.iloc[0]:+.1f}pp）</b>。'
          f'<br><b>所以呢</b>：如果这两块变化超过 10pp，说明基金经理的风格已经变了，'
          f'你当初买它的理由可能不再成立；如果基本没动，它就是个「行业贝塔工具」，'
          f'按行业逻辑用，不要指望它在行业反转时保护你。</div>')

    # ---------------- 3. 前十大 ----------------
    h(f'<div class="sec">③ 它的前十大到底是谁（{qf} 全持仓）</div>')
    tp = _top_holdings(code, qf)
    if tp.empty:
        h('<div class="cg-note">该期无持仓明细。</div>')
        return
    tp = tp.rename(columns={"stock_code": "代码", "stock_name": "股票",
                            "industry": "行业", "weight_pct": "占净值%",
                            "shares": "持股数(万股)", "mv_yi": "持仓市值(亿)",
                            "is_hk": "港股"})
    # 东财明细里「持股数」本身就是万股、「持仓市值」是万元，
    # 再除一次 1e4 会全部显示成 0.0——这里只做四舍五入。
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
    h(f'<div class="cg-note">前十大合计占净值 <b>{_sum:.1f}%</b>，'
      f'前 3 大合计 <b>{float(tp["占净值%"].head(3).sum()):.1f}%</b>。'
      f'<br><b>所以呢</b>：季报只披露前十大，所以你看到的永远只是「最重要的那部分」。'
      f'前十大占比越高，季报信息量越大；如果只有 20%，说明这只基金很分散，'
      f'季报能告诉你的其实很少——这时候不用花时间研究它的持仓。</div>')
    h('<div class="tiny">数据：东方财富基金定期报告（半年报全持仓、季报前十大）。'
      '本页所有对比都是「与同行中位数」，不是与指数比；'
      '要看它有没有跑赢基准，需要基准指数收益，本库尚未接入。</div>')
