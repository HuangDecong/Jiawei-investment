# -*- coding: utf-8 -*-
"""「本期解读」用到的图。

原则和 `insight_charts` 一致，但这里全部改用 `canon` 的口径：
全持仓期回答「配了多少、主动还是被动」，前十大期回答「多少基金在动」，
两者不混在同一张图里。

每张图只留一个信息点，标题直接写成一句结论，读者不用先看图例。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as C


def _base(fig: go.Figure, title: str, h: int = 420, legend: bool = False) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=C.COLOR["text"]), x=0.005, y=0.97),
        height=h, margin=dict(l=6, r=10, t=46, b=10),
        paper_bgcolor=C.COLOR["bg"], plot_bgcolor=C.COLOR["bg"],
        font=dict(size=11, color=C.COLOR["text"]),
        showlegend=legend,
        hoverlabel=dict(font_size=11),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=9)),
    )
    fig.update_xaxes(gridcolor=C.COLOR["grid"], zerolinecolor=C.COLOR["line"])
    fig.update_yaxes(gridcolor=C.COLOR["grid"], zerolinecolor=C.COLOR["line"])
    return fig


# ---------------------------------------------------------------------------
# 1. 仓位：先看总开关，再看行业
# ---------------------------------------------------------------------------
def fig_position(c0: pd.Series, c1: pd.Series, q0: str, q1: str) -> go.Figure:
    """左：两期股票仓位分布；右：逐基金仓位变化分布。

    仓位是季报里最被忽略的一行。行业占比上升可能只是「整体加了仓」，
    不看仓位就把「加仓」误读成「看好这个行业」。
    """
    idx = c0.index.intersection(c1.index)
    a, b = c0.loc[idx].dropna(), c1.loc[idx].dropna()
    idx2 = a.index.intersection(b.index)
    a, b = a.loc[idx2], b.loc[idx2]
    d = (b - a).clip(-30, 30)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14,
                        subplot_titles=(f"股票仓位分布（{len(idx2):,} 只）",
                                        "逐基金仓位变化（pp）"))
    # 用箱线而不是小提琴：仓位分布高度集中（多数基金 85–95%），
    # 小提琴会画成一个「蝴蝶结」，箱体反而看不清；箱线 + 离群点更直观。
    for s, name, col in ((a, q0, "#9AA7B4"), (b, q1, C.COLOR["price"])):
        fig.add_trace(go.Box(y=s, name=name, marker_color=col, line_color=col,
                             boxpoints="outliers", marker=dict(size=3, opacity=0.5),
                             width=0.45, whiskerwidth=0.4,
                             hovertemplate=f"{name}<br>%{{y:.1f}}%<extra></extra>"),
                      row=1, col=1)
    fig.add_trace(go.Histogram(x=d, nbinsx=46, marker_color=C.COLOR["price"],
                               opacity=0.8, name="仓位变化",
                               hovertemplate="%{x:.1f}pp：%{y} 只<extra></extra>"),
                  row=1, col=2)
    fig.add_vline(x=0, line=dict(color=C.COLOR["line"], width=1), row=1, col=2)
    fig.update_yaxes(title_text="占净值比 %", row=1, col=1)
    fig.update_xaxes(title_text="仓位变化（pp）　←减仓　｜　加仓→", row=1, col=2)
    fig.update_yaxes(title_text="基金数", row=1, col=2)
    return _base(fig, "先看总开关：股票仓位变了没有", 330)


# ---------------------------------------------------------------------------
# 2. 配置变化：多少是价格，多少是机构
# ---------------------------------------------------------------------------
def fig_active_passive(d: pd.DataFrame, top: int = 11) -> go.Figure:
    """堆叠横条：灰＝价格带来的被动变化，彩＝机构主动，条长＝总变化。"""
    d = d.dropna(subset=["被动贡献", "主动贡献"]).copy()
    # 按「主动贡献」排序，不按总变化：按总变化排，第一名永远是被价格推着走的行业，
    # 而这张图要回答的是「机构动了哪」，主动项才是主角。
    d = d.reindex(d["主动贡献"].abs().sort_values(ascending=False).index).head(top)
    d = d.iloc[::-1]
    labels = [f"{i}　{v:+.2f}pp" for i, v in zip(d.index, d["变化"])]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=d["被动贡献"], orientation="h", name="价格（被动）",
                         marker_color="#C3CBD4",
                         hovertemplate="%{y}<br>价格 %{x:+.2f}pp<extra></extra>"))
    fig.add_trace(go.Bar(y=labels, x=d["主动贡献"], orientation="h", name="机构（主动）",
                         marker_color=[C.COLOR["up"] if v > 0 else C.COLOR["down"]
                                       for v in d["主动贡献"]],
                         hovertemplate="%{y}<br>主动 %{x:+.2f}pp<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=C.COLOR["line"], width=1))
    fig.update_layout(barmode="relative", bargap=0.3)
    fig.update_xaxes(title_text="对行业配置比例的影响（pp）　灰＋彩＝合计")
    return _base(fig, "配置变化拆开看：灰的是价格，彩的才是机构", 430, legend=True)


# ---------------------------------------------------------------------------
# 3. 广度：多少基金在动
# ---------------------------------------------------------------------------
def fig_breadth(d: pd.DataFrame, top: int = 14) -> go.Figure:
    """加仓基金占比的横向条形，按偏离 50% 的幅度排序。

    为什么不用「占比 × 分歧度」散点：分歧度 = 1−|2×占比−1|，
    完全由占比决定，两轴相关性 ±1，画出来是一条线，没有信息量。
    """
    d = d.dropna(subset=["加仓占比"]).copy()
    d["偏离"] = (d["加仓占比"] - 50).abs()
    d = d.nlargest(top, "偏离").sort_values("加仓占比")
    col = [C.COLOR["up"] if v >= 50 else C.COLOR["down"] for v in d["加仓占比"]]
    fig = go.Figure(go.Bar(
        y=d["行业"], x=d["加仓占比"], orientation="h", marker_color=col,
        text=[f"{v:.0f}%" for v in d["加仓占比"]], textposition="outside",
        textfont=dict(size=9),
        customdata=np.stack([d["加仓基金数"], d["减仓基金数"], d["平均变化pp"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>加仓基金占比 %{x:.1f}%<br>"
                       "加 %{customdata[0]:.0f} 只 / 减 %{customdata[1]:.0f} 只<br>"
                       "平均变化 %{customdata[2]:+.2f}pp<extra></extra>"),
    ))
    fig.add_vline(x=50, line=dict(color=C.COLOR["text_dim"], width=1, dash="dot"))
    fig.update_xaxes(title_text="加仓基金占比（%）　←多数在减　｜　多数在加→",
                     range=[0, 108])
    fig.update_yaxes(title_text="")
    return _base(fig, "谁在动：加仓基金占比最偏离 50% 的行业", 430)


# ---------------------------------------------------------------------------
# 4. 结构演变：31 个季度的行业配置漂移
# ---------------------------------------------------------------------------
def fig_structure_heatmap(mat: pd.DataFrame, top: int = 18,
                          title: str = "行业配置结构的漂移（等权平均，占股票市值 %）") -> go.Figure:
    """行＝行业（按最新一期排序），列＝季度，颜色＝平均配置比例。"""
    if mat.empty:
        return _base(go.Figure(), "结构数据未就绪")
    order = mat.iloc[-1].sort_values(ascending=False).index[:top]
    d = mat[order].T
    fig = go.Figure(go.Heatmap(
        z=d.values, x=list(d.columns), y=list(d.index),
        colorscale=[[0, "#F7FAFD"], [0.35, "#CFE0F5"], [0.7, "#4C8DDF"], [1, "#123E7C"]],
        hovertemplate="%{y}<br>%{x}：%{z:.1f}%<extra></extra>",
        colorbar=dict(thickness=9, len=0.75, tickfont=dict(size=9),
                      title=dict(text="%", font=dict(size=9))),
        xgap=1.2, ygap=1.2,
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=9))
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    return _base(fig, title, 520)


# ---------------------------------------------------------------------------
# 5. 个股：净增持 / 净减持
# ---------------------------------------------------------------------------
def fig_stock_flow(d: pd.DataFrame, top: int = 10) -> go.Figure:
    """左右对称条形：左边净减持、右边净增持（Δ持股基金数）。"""
    add = d.nlargest(top, "d_n").iloc[::-1]
    cut = d.nsmallest(top, "d_n")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.18,
                        subplot_titles=("被更多基金买入（Δ 持股基金数）",
                                        "被更多基金卖出（Δ 持股基金数）"))
    fig.add_trace(go.Bar(y=add["stock_name"], x=add["d_n"], orientation="h",
                         marker_color=C.COLOR["up"],
                         text=[f"{v:+.0f}" for v in add["d_n"]], textposition="outside",
                         textfont=dict(size=9),
                         customdata=np.stack([add["industry"], add["mv_yi"]], axis=-1),
                         hovertemplate=("<b>%{y}</b><br>%{customdata[0]}<br>"
                                        "Δ基金数 %{x:+.0f}<br>持仓 %{customdata[1]:.1f} 亿"
                                        "<extra></extra>")), row=1, col=1)
    fig.add_trace(go.Bar(y=cut["stock_name"], x=cut["d_n"], orientation="h",
                         marker_color=C.COLOR["down"],
                         text=[f"{v:+.0f}" for v in cut["d_n"]], textposition="outside",
                         textfont=dict(size=9),
                         customdata=np.stack([cut["industry"], cut["mv_yi"]], axis=-1),
                         hovertemplate=("<b>%{y}</b><br>%{customdata[0]}<br>"
                                        "Δ基金数 %{x:+.0f}<br>持仓 %{customdata[1]:.1f} 亿"
                                        "<extra></extra>")), row=1, col=2)
    fig.add_vline(x=0, line=dict(color=C.COLOR["line"], width=1), row=1, col=1)
    fig.add_vline(x=0, line=dict(color=C.COLOR["line"], width=1), row=1, col=2)
    fig.update_yaxes(tickfont=dict(size=9))
    return _base(fig, "个股层面：谁被机构加进来，谁被移出去", 430)


# ---------------------------------------------------------------------------
# 6. 分歧结构：广度 × 平均变化
# ---------------------------------------------------------------------------
def fig_breadth_vs_move(d: pd.DataFrame) -> go.Figure:
    """横轴＝多少基金在加，纵轴＝平均变化幅度（含价格漂移，已标注）。

    两轴独立：广度回答「共识度」，幅度回答「力度」。
    左上（少数大基金在加）与右下（多数小基金在加、净额却为负）含义相反。
    """
    if d.empty:
        return _base(go.Figure(), "数据未就绪")
    d = d.copy()
    cmap = {"共识形成": C.COLOR["up"], "系统性撤离": C.COLOR["down"],
            "多空对撞": "#E8710A", "分歧": C.COLOR["text_dim"]}
    # 只给「偏离中心」的点加文字：31 个全标会在 (50,0) 附近糊成一团
    notable = ((d["加仓占比"] - 50).abs() >= 15) | (d["平均变化pp"].abs() >= 1.2)
    fig = go.Figure()
    for tag, g in d.groupby("判读"):
        lab = [i if notable.get(i, False) else "" for i in g.index]
        fig.add_trace(go.Scatter(
            x=g["加仓占比"], y=g["平均变化pp"], mode="markers+text", name=f"{tag}（{len(g)}）",
            marker=dict(size=10 + np.sqrt(g["样本基金数"]) / 8,
                        color=cmap.get(tag, C.COLOR["text_dim"]), opacity=0.75,
                        line=dict(width=0.8, color="#FFFFFF")),
            text=lab, textposition="top center",
            textfont=dict(size=9, color=C.COLOR["text_dim"]),
            hovertemplate=("<b>%{text}</b><br>加仓基金占比 %{x:.1f}%<br>"
                           "平均变化 %{y:+.2f}pp<extra></extra>"),
        ))
    fig.add_vline(x=50, line=dict(color=C.COLOR["text_dim"], width=1, dash="dot"))
    fig.add_hline(y=0, line=dict(color=C.COLOR["line"], width=1))
    fig.update_xaxes(title_text="加仓基金占比（%）　←少数在加　｜　多数在加→", range=[0, 100])
    fig.update_yaxes(title_text="平均变化（pp）　含价格漂移")
    return _base(fig, "广度 × 幅度：四个象限对应四种完全不同的资金行为", 470, legend=True)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import canon as cn
    q = cn.latest()
    print("期次", q)
    pf = cn.pair_full()
    if pf:
        c0, c1 = cn.coverage(pf[0]), cn.coverage(pf[1])
        fig = fig_position(c0, c1, *pf)
        print("仓位图 ok", len(fig.data))
    pq = cn.pair_quarter()
    if pq:
        b = cn.breadth(*pq)
        print("广度", b.shape, b["判读"].value_counts().to_dict())
