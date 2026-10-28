# -*- coding: utf-8 -*-
"""把季报读透的三张图：主动/被动、共识/分歧、历史类比。

设计原则
--------
· **一张图只回答一个问题**，配 2–3 句「所以呢」，不堆指标；
· 颜色沿用全站约定：红＝加仓/上涨，绿＝减仓/下跌（A 股习惯）；
· 所有数字现算，不写死；标注样本与口径，避免「看起来很确定」。

三个问题
--------
1. `fig_active_passive`：这个行业的配置变化，是机构真买卖，还是价格涨跌？
2. `fig_dispersion_bubble`：这个行业是「一致加仓」「多空对撞」还是「一致撤离」？
3. `history_analog`：现在的持仓结构和历史上哪个季度最像？那之后一个季度发生了什么？
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config as C


def _base_layout(fig: go.Figure, title: str, h: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=C.COLOR["text"]), x=0.01),
        height=h, margin=dict(l=8, r=8, t=44, b=8),
        paper_bgcolor=C.COLOR["bg"], plot_bgcolor=C.COLOR["bg"],
        font=dict(size=11, color=C.COLOR["text"]),
        hoverlabel=dict(font_size=11),
    )
    fig.update_xaxes(gridcolor=C.COLOR["grid"], zerolinecolor=C.COLOR["line"])
    fig.update_yaxes(gridcolor=C.COLOR["grid"], zerolinecolor=C.COLOR["line"])
    return fig


# ---------------------------------------------------------------------------
# 1. 主动 vs 被动：配置变化里，机构到底动了多少
# ---------------------------------------------------------------------------
def fig_active_passive(active: pd.DataFrame, top: int = 12) -> go.Figure:
    """每个行业一根堆叠横条：灰＝价格带来的被动变化，彩＝机构主动，长度＝总变化。

    为什么用堆叠而不是分组：分组时两根条并排，读者还要自己相加；
    堆叠后条长就是「配置变了多少」，灰彩比例就是「价格 vs 机构」的贡献比。
    """
    d = active.dropna(subset=["passive", "active"]).copy()
    d["d_alloc"] = pd.to_numeric(d.get("d_alloc"), errors="coerce")
    d = d.reindex(d["d_alloc"].abs().sort_values(ascending=False).index).head(top)
    d = d.iloc[::-1]                      # 变化最大的排最上面
    # 把「合计变化」写进纵轴标签，省掉压在条上的数字（会互相遮挡）
    d.index = [f"{i}　{v:+.2f}pp" for i, v in zip(d.index, d["d_alloc"])]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=d.index, x=d["passive"], orientation="h", name="价格带来的被动变化（灰）",
        marker=dict(color="#C3CBD4"),
        hovertemplate="<b>%{y}</b><br>价格 %{x:+.2f}pp<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=d.index, x=d["active"], orientation="h", name="机构主动加减仓（彩）",
        marker=dict(color=[C.COLOR["up"] if v > 0 else C.COLOR["down"]
                           for v in d["active"]]),
        hovertemplate="%{y}<br>机构主动 %{x:+.2f}pp<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=C.COLOR["line"], width=1))
    fig.update_layout(barmode="relative", bargap=0.3,
                      legend=dict(orientation="h", y=-0.14, x=0, font=dict(size=9)))
    fig.update_xaxes(title_text="配置变化（pp）：灰＋彩＝合计　←减　｜　加→")
    fig.update_yaxes(title_text="")
    return _base_layout(fig, "配置变化拆开看：灰的是价格，彩的才是机构", 470)


# ---------------------------------------------------------------------------
# 2. 共识 / 分歧：加仓的基金多不多，方向一致不一致
# ---------------------------------------------------------------------------
def fig_breadth_vs_active(disp: pd.DataFrame, active: pd.DataFrame,
                          min_n: int = 200) -> go.Figure:
    """横轴＝加仓基金占比（广度），纵轴＝机构主动加减仓（金额/幅度）。

    为什么不用「加仓占比 × 分歧度」：分歧度 = 1 − |占比 − 50%| × 2，
    完全由占比决定，两轴相关性 ±1，画出来是一条线，不提供任何新信息。
    换成「广度 × 主动幅度」后，两轴独立，四个象限各有明确含义：

        左上＝少数大基金在加（广度低、净额为正）→ 不是共识，容易反转
        右上＝多数基金在加且净额为正 → 真共识
        左下＝多数在减且净额在减 → 一致撤离
        右下＝多数在加但净额为负 → 小基金接盘、大基金出货
    """
    if disp is None or disp.empty:
        return _base_layout(go.Figure(), "分歧度数据未就绪")
    d = disp.set_index("行业").copy()
    a = active.reindex(d.index)
    d["主动pp"] = pd.to_numeric(a.get("active"), errors="coerce")
    d = d.dropna(subset=["加仓占比", "主动pp"])
    d = d[d["样本基金数"] >= min(min_n, int(d["样本基金数"].max()))]
    if d.empty:
        return _base_layout(go.Figure(), "样本不足")

    cmap = {"共识形成": C.COLOR["up"], "多空对撞": "#E8710A",
            "系统性撤离": C.COLOR["down"], "分歧": C.COLOR["text_dim"]}
    fig = go.Figure()
    for tag, g in d.groupby("判读"):
        fig.add_trace(go.Scatter(
            x=g["加仓占比"], y=g["主动pp"], mode="markers+text", name=f"{tag}（{len(g)}）",
            marker=dict(size=9 + g["样本基金数"] / d["样本基金数"].max() * 26,
                        color=cmap.get(tag, C.COLOR["text_dim"]),
                        opacity=0.72, line=dict(width=0.6, color="#FFFFFF")),
            text=g.index, textposition="top center",
            textfont=dict(size=9, color=C.COLOR["text_dim"]),
            customdata=np.stack([g["样本基金数"], g["分歧度"], g["平均配置"]], axis=-1),
            hovertemplate=("<b>%{text}</b><br>加仓基金占比 %{x:.1f}%<br>"
                           "机构主动 %{y:+.2f}pp<br>分歧度 %{customdata[1]:.2f}<br>"
                           "平均配置 %{customdata[2]:.2f}%<br>"
                           "样本 %{customdata[0]:.0f} 只<extra></extra>"),
        ))
    fig.add_vline(x=50, line=dict(color=C.COLOR["line"], width=1, dash="dot"))
    fig.add_hline(y=0, line=dict(color=C.COLOR["line"], width=1))
    for _x, _y, _t in [(0.02, 0.97, "少数大基金在加<br>（广度低，易反转）"),
                       (0.98, 0.97, "多数在加、净额也加<br>（真共识）"),
                       (0.02, 0.03, "多数在减、净额也减<br>（一致撤离）"),
                       (0.98, 0.03, "多数在加、净额在减<br>（小基金接盘）")]:
        fig.add_annotation(xref="paper", yref="paper", x=_x, y=_y, text=_t, showarrow=False,
                           align="left" if _x < 0.5 else "right",
                           font=dict(size=9, color=C.COLOR["text_dim"]),
                           bgcolor="rgba(255,255,255,0.72)", borderpad=3)
    fig.update_xaxes(title_text="加仓基金占比（%）　←多数在减　｜　多数在加→", range=[0, 100])
    fig.update_yaxes(title_text="机构主动加减仓（pp）　←减　｜　加→")
    fig.update_layout(legend=dict(orientation="h", y=-0.16, x=0, font=dict(size=9)))
    return _base_layout(fig, "谁在买：横轴看「多少基金在加」，纵轴看「净买了多少」", 470)


# ---------------------------------------------------------------------------
# 3. 历史类比：现在像哪一年，之后发生了什么
# ---------------------------------------------------------------------------
def history_analog(b, k: int = 3, min_gap: int = 4) -> dict:
    """用「31 个行业配置向量」的相关性找历史相似季度，再看次季行业收益。

    · 相关性用横截面皮尔逊，避免量纲影响；
    · 排除最近 min_gap 期，防止与当前期重叠造成「自己像自己」；
    · 次季收益用行业相对基准的超额，和项目其他检验一致。
    """
    import backtest as bt
    alloc = b.alloc.reindex(columns=C.INDUSTRIES).dropna(how="all")
    qs = list(alloc.index)
    if len(qs) < 8:
        return {"ok": False, "why": "历史期数不足"}
    cur_q = b.latest
    cur = alloc.loc[cur_q].astype(float)
    hist = alloc.iloc[:-min_gap]
    corr = hist.apply(lambda r: float(np.corrcoef(r.astype(float), cur)[0, 1]), axis=1)
    top = corr.sort_values(ascending=False).head(k)
    fwd = bt.forward_excess(b.market, 1)
    rows, series = [], []
    for q, c in top.items():
        if q not in fwd.index:
            continue
        r = fwd.loc[q].dropna()
        if r.empty:
            continue
        rows.append(dict(相似季度=q, 相似度=round(c, 3),
                         次季上涨行业数=int((r > 0).sum()),
                         次季超额中位=round(float(r.median()), 2),
                         次季最强=r.idxmax(), 最强超额=round(float(r.max()), 2),
                         次季最弱=r.idxmin(), 最弱超额=round(float(r.min()), 2)))
        series.append(r.rename(q))
    if not rows:
        return {"ok": False, "why": "相似季度的次季收益缺失"}
    avg = pd.concat(series, axis=1).mean(axis=1).sort_values(ascending=False)
    return {"ok": True, "matches": pd.DataFrame(rows), "avg_next": avg,
            "current": cur_q, "n_hist": int(len(corr))}


def fig_history_analog(res: dict) -> go.Figure:
    """历史相似季度之后，各行业次季超额收益的平均值（红涨绿跌）。"""
    if not res.get("ok"):
        return _base_layout(go.Figure(), "历史类比：数据不足")
    s = res["avg_next"]
    s = pd.concat([s.head(8), s.tail(8)])
    colors = [C.COLOR["up"] if v > 0 else C.COLOR["down"] for v in s.values]
    fig = go.Figure(go.Bar(
        x=s.values, y=s.index, orientation="h",
        marker=dict(color=colors), text=[f"{v:+.2f}%" for v in s.values],
        textposition="outside", textfont=dict(size=9),
        hovertemplate="<b>%{y}</b><br>历史相似季度之后平均 %{x:+.2f}%<extra></extra>",
    ))
    fig.update_xaxes(title_text="次季超额收益（%，相似季度均值）")
    fig.update_yaxes(autorange="reversed")
    return _base_layout(fig, f"和 {res['current']} 最像的 {len(res['matches'])} 个季度，"
                             f"之后一个季度谁涨谁跌", 520)
