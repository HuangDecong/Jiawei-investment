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
def fig_position(P: pd.DataFrame, cur: pd.Series, q_cur: str) -> go.Figure:
    """左：15 个全持仓期的仓位中位数走势（带 25–75 分位带）；右：最新一期分布。

    早期只画「上期 vs 本期」两个箱线，看上去像只有两季数据；
    改成全历史趋势后，才能判断当前仓位在历史上是高是低。
    """
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                        horizontal_spacing=0.08,
                        subplot_titles=(f"股票仓位趋势（{len(P)} 个半年报期）",
                                        f"{q_cur} 仓位分布（{len(cur):,} 只）"))
    x = list(P.index)
    # 25–75 分位带
    fig.add_trace(go.Scatter(x=x + x[::-1], y=list(P["q3"]) + list(P["q1"])[::-1],
                             fill="toself", fillcolor="rgba(31,111,235,0.13)",
                             line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=P["中位"], mode="lines+markers", name="仓位中位",
                             line=dict(color=C.COLOR["price"], width=2),
                             marker=dict(size=6),
                             hovertemplate="%{x}　中位 %{y:.1f}%<extra></extra>",
                             showlegend=False), row=1, col=1)
    _last = P.index[-1]
    fig.add_annotation(x=_last, y=float(P["中位"].iloc[-1]),
                       text=f"{float(P['中位'].iloc[-1]):.1f}%", showarrow=True,
                       arrowhead=0, ax=0, ay=-26, font=dict(size=10, color=C.COLOR["price"]),
                       row=1, col=1)
    fig.update_yaxes(title_text="占净值比 %", row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=1, col=1)

    fig.add_trace(go.Histogram(x=cur, nbinsx=40, marker_color=C.COLOR["price"],
                               opacity=0.8, showlegend=False,
                               hovertemplate="%{x:.0f}%：%{y} 只<extra></extra>"),
                  row=1, col=2)
    _m = float(cur.median())
    fig.add_vline(x=_m, line=dict(color=C.COLOR["up"], width=1.5, dash="dash"),
                  annotation_text=f"中位 {_m:.1f}%", annotation_position="top",
                  annotation_font=dict(size=9, color=C.COLOR["up"]), row=1, col=2)
    fig.update_xaxes(title_text="仓位 %", row=1, col=2)
    fig.update_yaxes(title_text="基金数", row=1, col=2)
    return _base(fig, "", 330)


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
    return _base(fig, "", 430, legend=True)


# ---------------------------------------------------------------------------
# 3. 广度：多少基金在动
# ---------------------------------------------------------------------------
def fig_breadth(d: pd.DataFrame, top: int = 14) -> go.Figure:
    """加仓基金占比的横向条形，按偏离 50% 的幅度排序。

    为什么不用「占比 × 分歧度」散点：分歧度 = 1−|2×占比−1|，
    完全由占比决定，两轴相关性 ±1，画出来是一条线，没有信息量。

    分母说明（页面上也写着）：加仓占比 = 加仓 ÷ (加仓 + 减仓)，即
    「有实质变动的基金里多少在加」；不持有该行业的基金不进分母。
    """
    d = d.dropna(subset=["加仓占比"]).copy()
    d["偏离"] = (d["加仓占比"] - 50).abs()
    d = d.nlargest(top, "偏离").sort_values("加仓占比")
    col = [C.COLOR["up"] if v >= 50 else C.COLOR["down"] for v in d["加仓占比"]]
    fig = go.Figure(go.Bar(
        y=d["行业"], x=d["加仓占比"], orientation="h", marker_color=col,
        text=[f"{v:.0f}%" for v in d["加仓占比"]], textposition="outside",
        textfont=dict(size=9),
        customdata=np.stack([d["加仓基金数"], d["减仓基金数"], d["持有基金数"],
                             d["平均主动pp"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>加仓基金占比 %{x:.1f}%<br>"
                       "加 %{customdata[0]:.0f} 只 / 减 %{customdata[1]:.0f} 只"
                       "（持有 %{customdata[2]:.0f} 只）<br>"
                       "价格中性净额 %{customdata[3]:+.2f}pp<extra></extra>"),
    ))
    fig.add_vline(x=50, line=dict(color=C.COLOR["text_dim"], width=1, dash="dot"))
    fig.update_xaxes(title_text="加仓基金占比（%）　←多数在减　｜　多数在加→",
                     range=[0, 108])
    fig.update_yaxes(title_text="")
    fig = _base(fig, "", 430)
    fig.update_layout(margin=dict(l=6, r=30, t=16, b=34))
    return fig


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
def fig_stock_flow(d: pd.DataFrame, top: int = 10,
                   q0: str | None = None, q: str | None = None) -> go.Figure:
    """左右对称条形：左边净减持、右边净增持（Δ持股基金数）。

    q0/q 只用于标题与悬浮提示——个股榜必须标明「哪两期」，
    否则读者会以为是相邻两季（那会把「第 11 名掉出前十」误判成清仓）。
    """
    add = d.nlargest(top, "d_n").iloc[::-1]
    cut = d.nsmallest(top, "d_n")
    _tag = f"（{q0} → {q}）" if q0 and q else ""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.18,
                        subplot_titles=(f"被更多基金买入{_tag}　Δ 持股基金数",
                                        f"被更多基金卖出{_tag}　Δ 持股基金数"))
    for _df, _col, _c in ((add, 1, C.COLOR["up"]), (cut, 2, C.COLOR["down"])):
        fig.add_trace(go.Bar(
            y=_df["stock_name"], x=_df["d_n"], orientation="h", marker_color=_c,
            text=[f"{v:+.0f}" for v in _df["d_n"]], textposition="outside",
            textfont=dict(size=9),
            customdata=np.stack([_df["industry"], _df["n_funds_0"], _df["n_funds"],
                                 _df["mv_yi"]], axis=-1),
            hovertemplate=("<b>%{y}</b>　%{customdata[0]}<br>"
                           "持有基金数 %{customdata[1]:.0f} → %{customdata[2]:.0f}"
                           "（Δ %{x:+.0f}）<br>"
                           "本期持仓 %{customdata[3]:.1f} 亿<extra></extra>"),
        ), row=1, col=_col)
        fig.add_vline(x=0, line=dict(color=C.COLOR["line"], width=1), row=1, col=_col)
    fig.update_yaxes(tickfont=dict(size=9))
    # 条端数字画在坐标区外，右侧必须留白，否则会被裁掉。
    # 注意：必须放在 _base 之后——_base 会重写 margin。
    fig = _base(fig, "", 430)
    fig.update_layout(margin=dict(l=6, r=54, t=52, b=10))
    return fig


def fig_stock_industry(d: pd.DataFrame, top: int = 50,
                       q0: str | None = None, q: str | None = None) -> go.Figure:
    """加仓前 N 与减仓前 N 的行业构成：判断是「行业性」还是「个股性」。

    同一批钱如果都流向同一个行业，说明基金经理在做行业配置；
    散在十个行业里，才是选股。这是个股榜唯一能给出的增量信息。
    """
    a = d.nlargest(top, "d_n")["industry"].value_counts()
    c = d.nsmallest(top, "d_n")["industry"].value_counts()
    inds = list((a.add(c, fill_value=0)).sort_values(ascending=False).index)[:10]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=inds, x=[a.get(i, 0) for i in inds], orientation="h",
                         name=f"加仓前 {top} 只", marker_color=C.COLOR["up"],
                         hovertemplate="%{y}：%{x} 只<extra></extra>"))
    fig.add_trace(go.Bar(y=inds, x=[-c.get(i, 0) for i in inds], orientation="h",
                         name=f"减仓前 {top} 只", marker_color=C.COLOR["down"],
                         hovertemplate="%{y}：%{x} 只<extra></extra>"))
    fig.update_layout(barmode="relative", bargap=0.25)
    fig.add_vline(x=0, line=dict(color=C.COLOR["line"], width=1))
    fig.update_xaxes(title_text="← 减仓榜的只数　｜　加仓榜的只数 →")
    return _base(fig, "", 380, legend=True)


# ---------------------------------------------------------------------------
# 10. 单只基金：全部行业的结构漂移（堆叠面积）
# ---------------------------------------------------------------------------
# 31 个申万一级行业，按「上游—中游—科技—消费—金融—公用」分色系，
# 同色系内靠明度区分。不用随机调色板：行业相邻色如果在同一色系，
# 读者能一眼看出「这一块是周期还是成长」。
IND_COLORS = {
    # 上游资源：棕
    "石油石化": "#7A4A1E", "煤炭": "#8C5A2B", "有色金属": "#B07030",
    "钢铁": "#C98A3E", "基础化工": "#D9A45B",
    # 中游制造：蓝绿
    "电力设备": "#1E4F6B", "机械设备": "#2F6B8F", "国防军工": "#4A8CAD",
    "建筑材料": "#6FA9C4", "建筑装饰": "#94C2D6",
    # 科技成长：深蓝
    "电子": "#1F3F8F", "通信": "#3B5FC0", "计算机": "#5B7FE0", "传媒": "#8FA7EE",
    # 消费：红橙
    "食品饮料": "#8C1F1F", "医药生物": "#A8322D", "汽车": "#C94A3F",
    "家用电器": "#E0705F", "纺织服饰": "#E89A88", "轻工制造": "#B35A1F",
    "商贸零售": "#D97A3A", "社会服务": "#F0A868", "农林牧渔": "#7A3B10",
    "美容护理": "#C98B5A",
    # 金融地产：绿
    "银行": "#1A7F37", "非银金融": "#2EA043", "房地产": "#56B870",
    # 公用与其他：紫
    "公用事业": "#6E4AB8", "交通运输": "#8B6FD1", "环保": "#4A3578", "综合": "#A992E0",
}


def ind_color(name: str) -> str:
    return IND_COLORS.get(name, "#B7C1CC")


def fig_ind_structure(M: pd.DataFrame, title: str = "",
                      residual_label: str = "其他（港股/未映射/现金）",
                      label_top: int = 4) -> go.Figure:
    """全部 31 个行业的堆叠面积图（行＝期，列＝行业，单位 %）。

    为什么用堆叠面积而不是多条线：31 条线会互相遮挡，面积图能同时看到
    「谁在扩张、谁在收缩、总结构怎么变」。行业按均值从大到小排，
    大的在下、小的在上，视觉上稳定。

    残差带：31 个行业之和常常不到 100——差的部分是港股、未映射行业与现金。
    不补这一条，图上会出现一块白色空白，看起来像渲染错误；补上并标明，
    读者才知道「剩下的是没被行业分类覆盖的部分」。
    """
    if M.empty:
        return _base(go.Figure(), "数据未就绪")
    d = M.copy()
    order = d.mean().sort_values(ascending=False).index.tolist()
    d = d[order]
    fig = go.Figure()
    for ind in order:
        s = d[ind]
        fig.add_trace(go.Scatter(
            x=list(d.index), y=s.values, mode="lines", name=ind,
            stackgroup="one", line=dict(width=0.5, color="#FFFFFF"),
            fillcolor=ind_color(ind),
            hovertemplate=f"<b>{ind}</b><br>%{{x}}：%{{y:.2f}}%<extra></extra>"))
    _resid = (100 - d.sum(axis=1)).clip(lower=0)
    if float(_resid.max()) > 0.5:
        fig.add_trace(go.Scatter(
            x=list(d.index), y=_resid.values, mode="lines", name=residual_label,
            stackgroup="one", line=dict(width=0.5, color="#FFFFFF"),
            fillcolor="#E9EDF2",
            hovertemplate=f"<b>{residual_label}</b><br>%{{x}}：%{{y:.2f}}%<extra></extra>"))

    # 末端直接标注前几大行业：31 项图例放不下，读者需要一眼看到「现在最大的是谁」。
    # 位置必须按**堆叠顺序**（即 trace 顺序＝均值降序）累加，
    # 不能用「最新一期的排序」——两者不一致时标注会飘到别人的色块上。
    _last = d.index[-1]
    _cur = d.iloc[-1]
    _pos, _acc = {}, 0.0
    for ind in order:
        _v = float(_cur.get(ind, 0) or 0)
        _pos[ind] = (_acc, _v)
        _acc += _v
    _top_now = _cur.sort_values(ascending=False).index[:label_top]
    for ind in _top_now:
        _y0, _v = _pos.get(ind, (0.0, 0.0))
        if _v < 2:
            continue
        fig.add_annotation(x=_last, y=_y0 + _v / 2, text=f"{ind} {_v:.0f}%",
                           showarrow=False, xanchor="left", xshift=8,
                           font=dict(size=9, color="#FFFFFF"), bgcolor=ind_color(ind),
                           borderpad=2, opacity=0.96)
    fig.update_yaxes(title_text="占前十大 %" if residual_label.startswith("其他")
                     else "占净值比 %", range=[0, 100])
    fig.update_xaxes(tickangle=-45)
    fig = _base(fig, title, 430)
    # 末端标注需要右侧留白，否则会被裁掉
    fig.update_layout(margin=dict(l=6, r=118, t=40, b=8))
    return fig


def ind_legend_html(names: list[str]) -> str:
    """行业颜色索引：31 个小色块 + 名称，替代放不下的 plotly 图例。"""
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:3px;'
        f'margin:2px 7px 2px 0;font-size:10px;color:#37485C">'
        f'<span style="width:9px;height:9px;border-radius:2px;'
        f'background:{ind_color(n)};display:inline-block"></span>{n}</span>'
        for n in names)
    return f'<div style="line-height:1.9">{chips}</div>'


# ---------------------------------------------------------------------------
# 11. 拥挤度：31 个行业一根条，颜色即等级
# ---------------------------------------------------------------------------
CSI_COLORS = {
    "极度拥挤": "#C62828", "拥挤": "#E8710A", "中性": "#8A97A5",
    "偏冷": "#4C8DDF", "冷门": "#1F4E9C", "数据不足": "#C3CBD4",
}


def fig_csi_bar(cs: pd.DataFrame) -> go.Figure:
    """按 CSI 排序的横向条形图，颜色＝拥挤等级，条端标注分位。

    这张图替代原来的「四象限散点」承担「一眼看完 31 个行业」的职责：
    散点要读两个轴，条形只需要读长度和颜色。
    """
    if cs.empty:
        return _base(go.Figure(), "数据未就绪")
    d = cs.dropna(subset=["CSI"]).sort_values("CSI")
    cols = [CSI_COLORS.get(l, "#C3CBD4") for l in d["等级"]]
    fig = go.Figure(go.Bar(
        y=list(d.index), x=d["CSI"], orientation="h", marker_color=cols,
        text=[f'{v:.0f}　{l}' + ("†" if rel == "小样本" else "")
              for v, l, rel in zip(d["CSI"], d["等级"], d.get("可靠", [""] * len(d)))],
        textposition="outside", textfont=dict(size=9),
        customdata=np.stack([d["配置分位"], d["价格分位"], d["资金分位"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>CSI %{x:.0f}<br>"
                       "配置分位 %{customdata[0]:.0f}｜价格分位 %{customdata[1]:.0f}｜"
                       "资金分位 %{customdata[2]:.0f}<extra></extra>")))
    for v in (25, 45, 65, 80):
        fig.add_vline(x=v, line=dict(color=C.COLOR["line"], width=1, dash="dot"))
    fig.update_xaxes(title_text="拥挤度指数 CSI（0＝最冷，100＝最挤）　†＝小样本", range=[0, 126])
    fig.update_yaxes(tickfont=dict(size=9.5))
    fig = _base(fig, "", 640)
    fig.update_layout(margin=dict(l=6, r=86, t=16, b=34))
    return fig


# ---------------------------------------------------------------------------
# 6. 分歧结构：广度 × 平均变化
# ---------------------------------------------------------------------------
def fig_breadth_vs_move(d: pd.DataFrame) -> go.Figure:
    """横轴＝多少基金在加，纵轴＝价格中性后的净买卖（pp）。

    两轴独立：广度回答「共识度」，幅度回答「力度」。
    左上（少数大基金在加）与右下（多数小基金在加、净额却为负）含义相反。

    纵轴必须用「平均主动pp」而不是原始变化：原始变化含价格漂移，
    暴跌季所有行业都会落在左下角，看不出谁在被真买。
    """
    if d.empty:
        return _base(go.Figure(), "数据未就绪")
    d = d.copy()
    _y = "平均主动pp" if "平均主动pp" in d.columns else "平均变化pp"
    cmap = {"共识形成": C.COLOR["up"], "系统性撤离": C.COLOR["down"],
            "多空对撞": "#E8710A", "分歧": C.COLOR["text_dim"]}
    notable = ((d["加仓占比"] - 50).abs() >= 15) | (d[_y].abs() >= 0.8)
    fig = go.Figure()
    for tag, g in d.groupby("判读"):
        lab = [i if notable.get(i, False) else "" for i in g.index]
        fig.add_trace(go.Scatter(
            x=g["加仓占比"], y=g[_y], mode="markers+text", name=f"{tag}（{len(g)}）",
            marker=dict(size=10 + np.sqrt(g["样本基金数"]) / 8,
                        color=cmap.get(tag, C.COLOR["text_dim"]), opacity=0.75,
                        line=dict(width=0.8, color="#FFFFFF")),
            text=lab, textposition="top center",
            textfont=dict(size=9, color=C.COLOR["text_dim"]),
            customdata=np.stack([g["加仓基金数"], g["减仓基金数"]], axis=-1),
            hovertemplate=("<b>%{text}</b><br>加仓基金占比 %{x:.1f}%<br>"
                           "价格中性净额 %{y:+.2f}pp<br>"
                           "加 %{customdata[0]:.0f} / 减 %{customdata[1]:.0f}<extra></extra>"),
        ))
    fig.add_vline(x=50, line=dict(color=C.COLOR["text_dim"], width=1, dash="dot"))
    fig.add_hline(y=0, line=dict(color=C.COLOR["line"], width=1))
    fig.update_xaxes(title_text="加仓基金占比（%）　←少数在加　｜　多数在加→", range=[0, 100])
    fig.update_yaxes(title_text="价格中性净额（pp）")
    fig = _base(fig, "", 470, legend=True)
    fig.update_layout(margin=dict(l=64, r=16, t=16, b=44))
    return fig


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


# ---------------------------------------------------------------------------
# 7. 双周期对比：与上一季、与上年同期
# ---------------------------------------------------------------------------
def fig_compare_two(comp: pd.DataFrame, top: int = 16) -> go.Figure:
    """左右两栏：左边 Δ上一季，右边 Δ上年同期，行顺序一致，直接对着看。

    为什么不做成一张表：31 行 × 4 列的数字表看不出形状；
    两根条一组，谁在加速、谁在掉头，一眼就能分出来。
    """
    d = comp.dropna(subset=["Δ上季pp", "Δ上年pp"]).copy()
    d = d.reindex(d["Δ上季pp"].abs().sort_values(ascending=False).index).head(top)
    d = d.iloc[::-1]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=(f"与上一季比（{comp.attrs.get('q_prev', '')}→"
                                        f"{comp.attrs.get('q', '')}）",
                                        f"与上年同期比（{comp.attrs.get('q_year', '')}→"
                                        f"{comp.attrs.get('q', '')}）"))
    for col, (name, c) in enumerate([("Δ上季pp", C.COLOR["price"]),
                                     ("Δ上年pp", C.COLOR["accent"])]):
        fig.add_trace(go.Bar(
            y=d.index, x=d[name], orientation="h", showlegend=False,
            marker_color=[C.COLOR["up"] if v > 0 else C.COLOR["down"] for v in d[name]],
            text=[f"{v:+.2f}" for v in d[name]], textposition="outside",
            textfont=dict(size=9),
            hovertemplate=f"%{{y}}<br>{name} %{{x:+.2f}}pp<extra></extra>"),
            row=1, col=col + 1)
        fig.add_vline(x=0, line=dict(color=C.COLOR["line"], width=1), row=1, col=col + 1)
        fig.update_xaxes(title_text="pp", row=1, col=col + 1)
    # 不再放总标题：两个子图标题本身已经说清「与谁比」，
    # 再加一行总标题必然和子图标题重叠（调 margin 也压不住）。
    fig.update_layout(margin=dict(l=6, r=16, t=52, b=10))
    return _base(fig, "", 640)


# ---------------------------------------------------------------------------
# 8. 单行业全历史配置序列（拥挤度页）
# ---------------------------------------------------------------------------
def fig_level_history(M: pd.DataFrame, industry: str,
                      title_extra: str = "") -> go.Figure:
    """某个行业在 31 个报告期的配置比例走势（前十大口径，等权平均）。"""
    if M.empty or industry not in M.columns:
        return _base(go.Figure(), "数据未就绪")
    s = M[industry].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(s.index), y=s.values, mode="lines+markers",
                             line=dict(color=C.COLOR["price"], width=2),
                             marker=dict(size=5),
                             hovertemplate="%{x}　%{y:.2f}%<extra></extra>",
                             showlegend=False))
    _mean = float(s.mean())
    fig.add_hline(y=_mean, line=dict(color=C.COLOR["text_dim"], width=1, dash="dot"),
                  annotation_text=f"历史均值 {_mean:.2f}%", annotation_position="right",
                  annotation_font=dict(size=9, color=C.COLOR["text_dim"]))
    _hi, _lo = float(s.max()), float(s.min())
    fig.add_annotation(x=s.idxmax(), y=_hi, text=f"最高 {_hi:.1f}%", showarrow=True,
                       arrowhead=0, ay=-22, font=dict(size=9, color=C.COLOR["up"]))
    fig.add_annotation(x=s.idxmin(), y=_lo, text=f"最低 {_lo:.1f}%", showarrow=True,
                       arrowhead=0, ay=22, font=dict(size=9, color=C.COLOR["down"]))
    # 上下留 12% 余量：最高/最低点的文字标注画在点外侧，不留余量会被画到坐标区外
    _lo, _hi = float(s.min()), float(s.max())
    _pad = max((_hi - _lo) * 0.18, 0.6)
    fig.update_yaxes(title_text="占前十大组合 %", range=[_lo - _pad, _hi + _pad])
    fig.update_xaxes(tickangle=-45)
    fig = _base(fig, f"{industry} 配置比例的历史位置{title_extra}", 360)
    # 右侧要放「历史均值」标注，不留边距会被裁成「历…」
    fig.update_layout(margin=dict(l=6, r=72, t=46, b=10))
    return fig


# ---------------------------------------------------------------------------
# 9. 分位双轴：相对自己 vs 相对同行
# ---------------------------------------------------------------------------
def fig_dual_percentile(comp: pd.DataFrame, cs: pd.Series) -> go.Figure:
    """横条＝自身历史分位（左轴），折线＝当期横截面分位（右轴）。

    两把尺子量的是不同的事：
    · 自身分位高＝「它自己历史上从没这么重过」；
    · 横截面分位高＝「在 31 个行业里它也是最大的那几个」。
    只有两者都高，才是「又重又在历史极值」的真拥挤；
    自身高、横截面低（如石油石化），是「小行业里的相对极端」，
    对组合的影响远小于电子这种大行业。
    """
    d = comp.copy()
    d["横截面"] = cs.reindex(d.index)
    d = d.dropna(subset=["历史分位"]).sort_values("历史分位")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        y=d.index, x=d["历史分位"], orientation="h", name="自身历史分位（左轴）",
        marker_color=[C.COLOR["up"] if v >= 80 else
                      "#E8A33D" if v >= 60 else
                      "#B7C1CC" if v >= 40 else
                      "#7FA8DC" if v >= 20 else C.COLOR["down"]
                      for v in d["历史分位"]],
        hovertemplate="<b>%{y}</b><br>自身历史分位 %{x:.0f}<extra></extra>"),
        secondary_y=False)
    fig.add_trace(go.Scatter(
        y=d.index, x=d["横截面"], mode="markers", name="全市场横截面分位（右轴）",
        marker=dict(symbol="diamond", size=8, color=C.COLOR["accent"],
                    line=dict(width=0.8, color="#FFFFFF")),
        hovertemplate="<b>%{y}</b><br>横截面分位 %{x:.0f}<extra></extra>"),
        secondary_y=True)
    for v in (20, 80):
        fig.add_vline(x=v, line=dict(color=C.COLOR["line"], width=1, dash="dot"))
    fig.update_xaxes(title_text="分位（0＝历史最低，100＝历史最高）", range=[0, 104])
    fig.update_yaxes(title_text="", secondary_y=False)
    fig.update_yaxes(title_text="横截面分位", range=[0, 104], secondary_y=True)
    return _base(fig, "", 720, legend=True)


# ---------------------------------------------------------------------------
# 9. 拥挤度双轴：配置分位（机构有多重）× 指数分位（价格有多贵）
# ---------------------------------------------------------------------------
def fig_crowding_map(comp: pd.DataFrame, idx: pd.DataFrame) -> go.Figure:
    """横轴＝配置分位（31 期），纵轴＝指数点位分位（2019 年以来）。

    两个分位回答不同问题：
      横轴：机构在这个行业上押得比过去重还是轻；
      纵轴：这个行业的价格相对自己过去贵不贵。
    只有两者都高，才是「拥挤」——机构重仓且已经涨到历史高位。
    """
    if comp.empty or idx.empty:
        return _base(go.Figure(), "指数数据未就绪")
    # 调用方可能已经把这些列 join 进 comp，只补缺的，避免 columns overlap
    _need = [c for c in ("指数分位", "近3月涨幅%", "近1年涨幅%")
             if c not in comp.columns]
    d = comp.join(idx[_need], how="inner") if _need else comp.copy()
    d = d.dropna(subset=["历史分位", "指数分位"])
    if d.empty:
        return _base(go.Figure(), "指数数据未就绪")

    def _q(r):
        hi_c, hi_p = r["历史分位"] >= 50, r["指数分位"] >= 50
        if hi_c and hi_p:
            return "真拥挤（配置高+价格高）"
        if hi_c and not hi_p:
            return "抱团没赚（配置高+价格低）"
        if not hi_c and hi_p:
            return "机构踏空（配置低+价格高）"
        return "冷门（配置低+价格低）"

    d["象限"] = d.apply(_q, axis=1)
    cmap = {"真拥挤（配置高+价格高）": C.COLOR["up"],
            "抱团没赚（配置高+价格低）": "#E8710A",
            "机构踏空（配置低+价格高）": C.COLOR["price"],
            "冷门（配置低+价格低）": C.COLOR["down"]}
    fig = go.Figure()
    for tag, g in d.groupby("象限"):
        fig.add_trace(go.Scatter(
            x=g["历史分位"], y=g["指数分位"], mode="markers+text", name=f"{tag}：{len(g)}",
            marker=dict(size=9 + g["配置_前十"].clip(0, 20) * 1.3,
                        color=cmap.get(tag, C.COLOR["text_dim"]), opacity=0.75,
                        line=dict(width=0.7, color="#FFFFFF")),
            text=g.index, textposition="top center",
            textfont=dict(size=9, color=C.COLOR["text_dim"]),
            customdata=np.stack([g["配置_前十"], g["近3月涨幅%"], g["Δ上季pp"]], axis=-1),
            hovertemplate=("<b>%{text}</b><br>配置分位 %{x:.0f}（31 期）<br>"
                           "指数分位 %{y:.0f}<br>前十配置 %{customdata[0]:.2f}%<br>"
                           "近 3 月 %{customdata[1]:+.1f}%<br>"
                           "环比 %{customdata[2]:+.2f}pp<extra></extra>"),
        ))
    fig.add_vline(x=50, line=dict(color=C.COLOR["line"], width=1, dash="dot"))
    fig.add_hline(y=50, line=dict(color=C.COLOR["line"], width=1, dash="dot"))
    # 不再画四角文字：图例已经写明四个象限，角标还会压住落在角落的点
    fig.update_xaxes(title_text="配置分位（机构押得多重，31 期）", range=[-6, 108])
    fig.update_yaxes(title_text="指数分位（价格相对自己多贵）", range=[-6, 108])
    # 坐标轴标题要留出边距，否则左侧竖排标题会被裁掉
    fig.update_layout(margin=dict(l=64, r=16, t=52, b=54))
    return _base(fig, "两个分位一起看：只有右上角才是真拥挤", 520, legend=True)
