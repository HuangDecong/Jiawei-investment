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
    # ① 全部行业的结构漂移
    # =====================================================================
    h('<div class="sec">① 全部 31 个行业的结构漂移：它这些年把钱挪到了哪里</div>')
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
    h('<div class="sec">② 它和同行差在哪：超配的行业、低配的行业</div>')
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
    h(f'<div class="sec">③ 它的前十大到底是谁（{_hold_q}）</div>')
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
