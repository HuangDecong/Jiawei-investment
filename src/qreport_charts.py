# -*- coding: utf-8 -*-
"""季报页的图表：全部按「一眼能读出结论」设计，不做装饰性元素。

三条原则
--------
1. 一张图只说一件事：资产配置不说行业，行业不说个股；
2. 数值直接写在图里（条端/点旁），不要求读者去对照坐标轴；
3. 口径写在标题里：全持仓 / 前十名 / 半年报，读者不用翻文档。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as C
from report_charts import _base

_BLUE = "#1F6FEB"
_ORANGE = "#E8710A"
_GRAY = "#8A97A5"
_RED = "#C62828"
_GREEN = "#1A7F37"


def _thin(seq, max_labels: int = 12):
    """季度刻度稀疏化：超过 max_labels 时只保留每年 Q1 与末期。

    40 期全标会挤成一条黑带（读者反馈「标签叠在一起看不清」）；
    季报页的横轴按年读就够了，具体季度看悬浮提示。
    """
    seq = list(seq)
    if len(seq) <= max_labels:
        return seq
    out = [s for s in seq if str(s).endswith("Q1")]
    if seq and seq[-1] not in out:
        out.append(seq[-1])
    return out


def fig_asset_mix(a: pd.DataFrame, n: int | None = 8,
                  q: str | None = None) -> go.Figure:
    """股票/债券/现金占净值比（分组柱）。

    为什么分组而不是堆叠：三者之和经常不等于 100%（有应付赎回款、回购杠杆），
    堆叠会让人以为「加起来就是 100」，分组柱只比每一条自己的变化，不会误导。

    n=None 表示画全部历史（老基金能到 40 期）——这是「怎么才几个季度」
    的直接修复：季报资产配置本来就有全历史，没有理由只画 8 期。
    """
    if a is None or a.empty:
        return _base(go.Figure(), "暂无资产配置数据", 240)
    d = a[a["quarter"].notna()].sort_values("period")
    if n:
        d = d.tail(n)
    x = [str(v) for v in d["quarter"]]
    fig = go.Figure()
    _last = max(len(x) - 4, 0)
    for col, name, color in (("stock_pct", "股票", _BLUE),
                             ("bond_pct", "债券", _ORANGE),
                             ("cash_pct", "现金", _GRAY)):
        if col not in d.columns or d[col].notna().sum() == 0:
            continue
        _txt = [(f"{v:.0f}" if (np.isfinite(v) and (i >= _last or x[i] == q))
                 else "") for i, v in enumerate(d[col])]
        fig.add_trace(go.Bar(
            x=x, y=d[col], name=name, marker_color=color, marker_line_width=0,
            text=_txt, textposition="outside", textfont=dict(size=9),
            hovertemplate="%{x}<br>" + name + "占净值 %{y:.2f}%<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.22, bargroupgap=0.04)
    # 清算期的现金占比可以到 3000%+（净资产只剩几十万，现金含应收款）。
    # 硬压到 108% 会让柱子被截断却看不出来，所以在标题里注明「已截断」。
    _mx = float(np.nanmax([d[c].max() for c in ("stock_pct", "bond_pct", "cash_pct")
                           if c in d.columns])) if len(d) else np.nan
    _clipped = bool(np.isfinite(_mx) and _mx > 108)
    fig.update_yaxes(title_text="占基金净值 %", range=[0, 108],
                     tickfont=dict(size=9.5))
    _tk = _thin(x)
    fig.update_xaxes(tickmode="array", tickvals=_tk, ticktext=_tk,
                     tickfont=dict(size=9))
    _row = d[d["quarter"] == q].iloc[-1] if (q and (d["quarter"] == q).any()) else d.iloc[-1]
    _cap = (f"资产配置：{_row['quarter']} 股票 {_row.get('stock_pct', np.nan):.1f}%"
            f"｜现金 {_row.get('cash_pct', np.nan):.1f}%"
            f"（共 {len(d)} 期）")
    if _clipped:
        _cap += f"　纵轴截断到 108%（最高 {_mx:,.0f}%，清算期真实披露）"
    fig = _base(fig, _cap, 300, legend=True)
    fig.update_layout(margin=dict(l=52, r=14, t=52, b=30),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right",
                                  font=dict(size=9)))
    return fig


def fig_scale_bridge(nav0: float, eff_s: float, eff_n: float, cross: float,
                     nav1: float, np0: float, np1: float) -> go.Figure:
    """净资产变化瀑布：期初 → 份额效应 → 净值效应 → 交叉项 → 期末（亿元）。

    这是恒等式拆解，不是估算：净资产 = 份额 × 份额净值，三项相加严格等于差额。
    读者最容易被「规模涨了」误导，这张图把「靠申购堆出来的」和「靠净值涨出来的」
    分开画：前者带来建仓压力，后者只是账面浮盈。
    """
    labels = [f"期初净资产", "份额效应\n（申赎）", "净值效应\n（涨跌）",
              "交叉项", "期末净资产"]
    vals = [nav0, eff_s, eff_n, cross, nav1]
    fig = go.Figure(go.Waterfall(
        x=labels, measure=["absolute", "relative", "relative", "relative", "total"],
        y=vals,
        text=[f"{v:+.1f}" if i not in (0, 4) else f"{v:.1f}"
              for i, v in enumerate(vals)],
        textposition="outside", textfont=dict(size=10),
        connector=dict(line=dict(color="#C3CBD4", width=1)),
        increasing=dict(marker_color=_RED), decreasing=dict(marker_color=_GREEN),
        totals=dict(marker_color="#2F3B49"),
        hovertemplate="%{x}<br>%{y:+.2f} 亿元<extra></extra>"))
    _lo = min(0, min(vals)) * 1.2
    _hi = max(vals) * 1.25
    fig.update_yaxes(title_text="亿元", range=[_lo, _hi], tickfont=dict(size=9.5))
    fig.update_xaxes(tickfont=dict(size=9.5))
    fig = _base(fig, f"规模变化拆解：份额净值 {np0:.3f} → {np1:.3f} 元", 300)
    fig.update_layout(margin=dict(l=54, r=16, t=50, b=34))
    return fig


def fig_industry_area(t: pd.DataFrame) -> go.Figure:
    """行业配置的季度演变（折线，占净值 %）。

    看的是「押注挪没挪」，不是绝对值：某行业从 3% 涨到 20% 就是一次转向，
    哪怕股票仓位没变。只画最新一期最大的 6 个行业，其余归入「其他」。

    为什么用折线而不是堆叠面积：行业表是按年补抓的，中间会缺几年（2019–2024）。
    堆叠面积在缺口处会断成几座「孤岛」，看着像图坏了；折线在缺口处断开，
    读者一眼就知道「这段没抓」，不会误读成「配置归零」。
    """
    if t is None or t.empty:
        return _base(go.Figure(), "暂无行业配置序列", 240)
    cols = list(t.columns)
    _pal = [_BLUE, _ORANGE, "#2EA043", "#6E4AB8", "#0F8B8D", "#B3651A", _GRAY]
    fig = go.Figure()
    for i, c in enumerate(cols):
        fig.add_trace(go.Scatter(
            x=list(t.index), y=t[c], name=str(c), mode="lines+markers",
            connectgaps=False, line=dict(width=1.8, color=_pal[i % len(_pal)]),
            marker=dict(size=2.5),
            hovertemplate="%{x}<br>" + str(c) + " %{y:.2f}%<extra></extra>"))
    x = [str(v) for v in t.index]
    _tk = _thin(x)
    fig.update_xaxes(tickmode="array", tickvals=_tk, ticktext=_tk,
                     tickangle=-45, tickfont=dict(size=9))
    fig.update_yaxes(title_text="占基金净值 %", tickfont=dict(size=9))
    _gaps = int(t.isna().all(axis=1).sum())
    _cap = "行业配置的季度演变（A 股门类，季报原文）"
    if _gaps:
        _cap += f"　断点＝该期未抓取（{_gaps} 期）"
    fig = _base(fig, _cap, 360, legend=True)
    fig.update_layout(margin=dict(l=48, r=12, t=52, b=78),
                      legend=dict(orientation="h", y=-0.30, x=0,
                                  font=dict(size=8.5)))
    return fig


def fig_top10_diff(df: pd.DataFrame, top: int = 10) -> go.Figure:
    """前十大进出：按占净值变化排序，红＝新进/增持，绿＝退出/减持。

    条端只写数值，状态放在悬浮里：19 条横条上再写「新进前十 +9.86」，
    标签会互相压住（读者反馈过「字叠在一起」）。
    """
    if df is None or df.empty:
        return _base(go.Figure(), "暂无两期可比的持仓", 200)
    x = df.copy()
    x["_v"] = x["d_pct"].fillna(x["本期%"]).fillna(0)
    x = pd.concat([x.nsmallest(top, "_v"), x.nlargest(top, "_v")]).drop_duplicates()
    x = x.sort_values("_v")
    _map = {"新进前十": _RED, "升入前十": _RED, "增持": _RED, "不变": _GRAY,
            "减持": _GREEN, "退出前十": _GREEN, "清仓": _GREEN}
    cols = [_map.get(str(s), _GRAY) for s in x["状态"]]
    fig = go.Figure(go.Bar(
        x=x["_v"], y=x["名称"], orientation="h", marker_color=cols,
        marker_line_width=0,
        text=[f"{v:+.2f}" for v in x["_v"]],
        textposition="outside", textfont=dict(size=9),
        customdata=x[["本期%", "上期%", "d_sh", "状态"]].values,
        hovertemplate="<b>%{y}</b>　%{customdata[3]}<br>"
                      "占净值 %{customdata[1]:.2f}% → %{customdata[0]:.2f}%<br>"
                      "股数变化 %{customdata[2]:+.0%}<extra></extra>"))
    _m = float(x["_v"].abs().max()) * 1.45 if len(x) else 1
    fig.update_xaxes(range=[-_m, _m], title_text="占净值变化（pp）",
                     tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9.5))
    fig = _base(fig, "前十大换了谁（本期 − 上期，占净值 pp；红＝新进/升入/增持，绿＝退出/减持/清仓）",
                max(240, 26 * len(x) + 70))
    fig.update_layout(margin=dict(l=8, r=30, t=46, b=32))
    return fig


def fig_industry(x: pd.DataFrame, top: int = 12) -> go.Figure:
    """行业配置横向条（A股证监会门类 + 港股通合计一条）。

    入参是 `qreport.industry_view()` 整理好的表（sec_name / ratio_pct / prev_pct），
    两套分类已经分开：港股通汇总成一条，避免把「工业（恒生）」和「制造业（证监会）」
    并排比较。
    """
    if x is None or x.empty:
        return _base(go.Figure(), "暂无行业配置数据", 240)
    x = x.sort_values("ratio_pct", ascending=True)
    fig = go.Figure()
    if "prev_pct" in x.columns and x["prev_pct"].notna().any():
        fig.add_trace(go.Bar(
            x=x["prev_pct"], y=x["sec_name"], orientation="h", name="上期",
            marker_color="rgba(31,111,235,0.22)", marker_line_width=0,
            hovertemplate="<b>%{y}</b><br>上期 %{x:.2f}%<extra></extra>"))
    _cols = ["#1F6FEB" if n != "港股通合计" else "#E8710A" for n in x["sec_name"]]
    fig.add_trace(go.Bar(
        x=x["ratio_pct"], y=x["sec_name"], orientation="h", name="本期",
        marker_color=_cols, marker_line_width=0,
        text=[f"{v:.1f}%" for v in x["ratio_pct"]], textposition="outside",
        textfont=dict(size=9.5),
        hovertemplate="<b>%{y}</b><br>本期 %{x:.2f}%<extra></extra>"))
    fig.update_layout(barmode="overlay")
    _mx = float(x["ratio_pct"].max()) if len(x) else 10
    if "prev_pct" in x.columns and x["prev_pct"].notna().any():
        _mx = max(_mx, float(x["prev_pct"].max()))
    fig.update_xaxes(range=[0, _mx * 1.28], title_text="占基金净值 %",
                     tickfont=dict(size=9.5))
    fig.update_yaxes(tickfont=dict(size=10))
    _cap = "按行业分类的股票投资组合（季报原文 · 全部持仓）"
    if "prev_pct" in x.columns and x["prev_pct"].notna().any():
        _cap += "　深色＝本期，浅色＝上期"
    fig = _base(fig, _cap, max(260, 34 * len(x) + 70), legend=True)
    fig.update_layout(margin=dict(l=8, r=64, t=46, b=34))
    return fig


def fig_market_position(g: pd.DataFrame, n: int = 20) -> go.Figure:
    """已入库基金的股票仓位中位数（按季），最后一点标数值与样本数。"""
    if g is None or g.empty:
        return _base(go.Figure(), "暂无样本", 240)
    d = g.tail(n)
    fig = go.Figure(go.Scatter(
        x=list(d["quarter"]), y=list(d["median"]), mode="lines+markers",
        line=dict(color=_BLUE, width=2.2), marker=dict(size=4),
        hovertemplate="%{x}<br>中位 %{y:.1f}%<extra></extra>"))
    _x, _y = str(d["quarter"].iloc[-1]), float(d["median"].iloc[-1])
    fig.add_annotation(x=_x, y=_y, text=f"{_y:.1f}%（N={int(d['count'].iloc[-1])}）",
                       showarrow=False, xanchor="left", yanchor="middle",
                       xshift=4, font=dict(size=9, color=_BLUE))
    fig.update_yaxes(title_text="股票仓位中位数 %", tickfont=dict(size=9))
    _tk = _thin(list(d["quarter"]), 8)
    fig.update_xaxes(tickmode="array", tickvals=_tk, ticktext=_tk,
                     tickangle=-45, tickfont=dict(size=8.5))
    fig = _base(fig, "样本池基金的股票仓位中位数（按季）", 300)
    fig.update_layout(margin=dict(l=48, r=40, t=46, b=56))
    return fig


def fig_industry_delta(d: pd.DataFrame, top: int = 5) -> go.Figure:
    """全样本行业配置的环比加减仓（A股门类，单位 pp）。"""
    if d is None or d.empty:
        return _base(go.Figure(), "暂无行业环比", 240)
    up = d.nlargest(top, "差值")
    dn = d.nsmallest(top, "差值")
    x = pd.concat([dn, up]).sort_values("差值")
    cols = [_RED if v >= 0 else _GREEN for v in x["差值"]]
    fig = go.Figure(go.Bar(
        x=x["差值"], y=x.index, orientation="h", marker_color=cols,
        marker_line_width=0,
        text=[f"{v:+.2f}" for v in x["差值"]], textposition="outside",
        textfont=dict(size=9),
        hovertemplate="<b>%{y}</b><br>环比 %{x:+.2f}pp<extra></extra>"))
    _m = float(x["差值"].abs().max()) * 1.45 if len(x) else 1
    fig.update_xaxes(range=[-_m, _m], title_text="环比变化（pp）",
                     tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    fig = _base(fig, "全样本行业配置环比（配对中位数，A股门类）", 300)
    fig.update_layout(margin=dict(l=8, r=30, t=46, b=34))
    return fig


def fig_shares_waterfall(s: pd.Series) -> go.Figure:
    """份额变动瀑布：期初 + 申购 − 赎回 = 期末（亿份）。

    期初由「期末 − 申购 + 赎回」反推，这是会计恒等式，不是估算；
    季报本身只披露期间申购、期间赎回与期末份额。
    """
    if s is None or len(s) == 0:
        return _base(go.Figure(), "暂无份额变动数据", 240)
    end = float(s.get("end_share_yi", np.nan))
    buy = float(s.get("purchase_yi", np.nan))
    sell = float(s.get("redeem_yi", np.nan))
    if not np.isfinite(end):
        return _base(go.Figure(), "暂无份额变动数据", 240)
    buy = 0.0 if not np.isfinite(buy) else buy
    sell = 0.0 if not np.isfinite(sell) else sell
    begin = end - buy + sell
    fig = go.Figure(go.Waterfall(
        x=["期初份额", "期间申购", "期间赎回", "期末份额"],
        measure=["absolute", "relative", "relative", "total"],
        y=[begin, buy, -sell, end],
        text=[f"{begin:.1f}", f"+{buy:.1f}", f"-{sell:.1f}", f"{end:.1f}"],
        textposition="outside", textfont=dict(size=10),
        connector=dict(line=dict(color="#C3CBD4", width=1)),
        increasing=dict(marker_color=_RED),
        decreasing=dict(marker_color=_GREEN),
        totals=dict(marker_color="#2F3B49"),
        hovertemplate="%{x}<br>%{y:.2f} 亿份<extra></extra>"))
    _lo = min(0, begin, end - sell) * 1.15
    _hi = max(begin, end, begin + buy) * 1.22
    fig.update_yaxes(title_text="亿份", range=[_lo, _hi], tickfont=dict(size=9.5))
    fig.update_xaxes(tickfont=dict(size=10))
    _chg = (end / begin - 1) * 100 if begin > 0 else np.nan
    _cap = ("份额怎么变的：" + (f"净申购 {end - begin:+.1f} 亿份（{_chg:+.0f}%）"
                                if np.isfinite(_chg) else ""))
    fig = _base(fig, _cap, 300, legend=False)
    fig.update_layout(margin=dict(l=54, r=16, t=50, b=30))
    return fig


def fig_trades(c: pd.DataFrame, top: int = 10) -> go.Figure:
    """累计买入 / 卖出前 top 名（左买右卖，条长＝金额，单位亿元）。"""
    if c is None or c.empty:
        return _base(go.Figure(), "本期无重大买卖明细", 240)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.16,
                        subplot_titles=("累计买入前 %d" % top,
                                        "累计卖出前 %d" % top))
    for j, (direction, color) in enumerate((("买入", _RED), ("卖出", _GREEN))):
        d = c[c["direction"] == direction].nlargest(top, "amount_wan")
        if d.empty:
            continue
        d = d.iloc[::-1]
        fig.add_trace(go.Bar(
            x=d["amount_wan"] / 1e4, y=d["stock_name"], orientation="h",
            marker_color=color, marker_line_width=0,
            text=[f"{v / 1e4:.1f}" for v in d["amount_wan"]],
            textposition="outside", textfont=dict(size=9),
            customdata=d[["stock_code", "pct_nav"]].values,
            hovertemplate="<b>%{y}</b>（%{customdata[0]}）<br>"
                          "累计" + direction + " %{x:.2f} 亿元<br>"
                          "占期初净值 %{customdata[1]:.2f}%<extra></extra>"),
            row=1, col=j + 1)
    _mx = float(c["amount_wan"].max()) / 1e4 * 1.35
    fig.update_xaxes(range=[0, _mx], tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9.5))
    fig = _base(fig, "报告期内的重大买卖（季报原文 · 前 20 名）", 360)
    fig.update_layout(margin=dict(l=6, r=40, t=54, b=30),
                      showlegend=False)
    for an in fig.layout.annotations:
        an.font = dict(size=10)
    return fig


def fig_top10_bar(d: pd.DataFrame, top: int = 10) -> go.Figure:
    """前十大个股权重（占净值），条端写权重。"""
    if d is None or d.empty:
        return _base(go.Figure(), "暂无持仓明细", 240)
    x = d.nsmallest(top, "rnk").sort_values("ratio_pct", ascending=True)
    fig = go.Figure(go.Bar(
        x=x["ratio_pct"], y=x["stock_name"], orientation="h",
        marker_color="#2F3B49", marker_line_width=0,
        text=[f"{v:.2f}%" for v in x["ratio_pct"]], textposition="outside",
        textfont=dict(size=9),
        customdata=x[["stock_code"]].values,
        hovertemplate="<b>%{y}</b>（%{customdata[0]}）<br>"
                      "占净值 %{x:.2f}%<extra></extra>"))
    _mx = float(x["ratio_pct"].max()) if len(x) else 5
    fig.update_xaxes(range=[0, _mx * 1.25], title_text="占基金净值 %",
                     tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9.5))
    fig = _base(fig, "", max(240, 30 * len(x) + 50))
    fig.update_layout(margin=dict(l=6, r=54, t=12, b=30))
    return fig
