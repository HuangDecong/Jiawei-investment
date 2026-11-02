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
def fig_breadth(d: pd.DataFrame, top: int = 14,
                small: list[str] | None = None) -> go.Figure:
    """加仓基金占比的横向条形，按偏离 50% 的幅度排序。

    小样本行业（综合等）单独排在最后、涂灰并标 †：
    它的 62% 是「58 只基金里 36 只在加」，和银行「1,000 只里 620 只在加」
    完全不是一个量级的证据，混在中间排会误导。

    为什么不用「占比 × 分歧度」散点：分歧度 = 1−|2×占比−1|，
    完全由占比决定，两轴相关性 ±1，画出来是一条线，没有信息量。
    """
    small = small or []
    d = d.dropna(subset=["加仓占比"]).copy()
    if d.empty:
        return _base(go.Figure(), "数据未就绪")
    d["_sm"] = d["行业"].isin(small).astype(int)
    d["偏离"] = (d["加仓占比"] - 50).abs()
    # 正常样本取最偏离的 top 个；小样本一律全部保留在末尾
    _ok = d[d["_sm"] == 0].nlargest(top, "偏离")
    _sm = d[d["_sm"] == 1]
    d = pd.concat([_ok, _sm]).sort_values(["_sm", "加仓占比"],
                                          ascending=[False, True])
    col = [CSI_COLORS["小样本"] if sm else
           (C.COLOR["up"] if v >= 50 else C.COLOR["down"])
           for v, sm in zip(d["加仓占比"], d["_sm"])]
    fig = go.Figure(go.Bar(
        y=[i + ("†" if sm else "") for i, sm in zip(d["行业"], d["_sm"])],
        x=d["加仓占比"], orientation="h", marker_color=col,
        text=[f"{v:.0f}%" for v in d["加仓占比"]], textposition="outside",
        textfont=dict(size=11),
        customdata=np.stack([d["加仓基金数"], d["减仓基金数"], d["持有基金数"],
                             d["平均主动pp"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>加仓基金占比 %{x:.1f}%<br>"
                       "加 %{customdata[0]:.0f} 只 / 减 %{customdata[1]:.0f} 只"
                       "（持有 %{customdata[2]:.0f} 只）<br>"
                       "价格中性净额 %{customdata[3]:+.2f}pp<extra></extra>"),
    ))
    fig.add_vline(x=50, line=dict(color=C.COLOR["threshold"], width=1.4, dash="dash"),
                  annotation_text="50%：左＝多数在减，右＝多数在加",
                  annotation_position="top",
                  annotation_font=dict(size=10, color=C.COLOR["threshold"]),
                  annotation_bgcolor="rgba(255,255,255,0.85)")
    fig.update_xaxes(title_text="加仓基金占比（%）", range=[0, 112],
                     tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=12))
    fig = _base(fig, "", 430)
    fig.update_layout(margin=dict(l=6, r=40, t=26, b=42))
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
                      label_top: int = 4,
                      max_items: int | None = 10) -> go.Figure:
    """行业结构堆叠面积图（行＝期，列＝行业，单位 %）。

    为什么默认只画前 10 大 + 「其他 N 个」：31 个色块叠在一起时，
    后面 20 个行业每个只有 1–3pp，颜色块细到看不清，图例也排不下。
    合并之后每一层都有 4pp 以上的厚度，趋势才读得出来。
    想看全部 31 个，把 max_items 设为 None（页面上有开关）。

    残差带：各行业之和常常不到 100——差的部分是港股、未映射行业与现金。
    不补这一条，图上会出现一块白色空白，看起来像渲染错误。
    """
    if M.empty:
        return _base(go.Figure(), "数据未就绪")
    d = M.copy()
    order = d.mean().sort_values(ascending=False).index.tolist()
    merged_label = None
    if max_items and len(order) > max_items:
        keep = order[:max_items]
        rest = order[max_items:]
        merged_label = f"其他 {len(rest)} 个行业"
        d = d[keep].copy()
        d[merged_label] = M[rest].sum(axis=1)
        order = keep + [merged_label]

    fig = go.Figure()
    for ind in order:
        s_ = d[ind]
        fig.add_trace(go.Scatter(
            x=list(d.index), y=s_.values, mode="lines", name=ind,
            stackgroup="one", line=dict(width=0.6, color="#FFFFFF"),
            fillcolor="#C9D3DE" if ind == merged_label else ind_color(ind),
            hoverinfo="skip", showlegend=False))
    _resid = (100 - d.sum(axis=1)).clip(lower=0)
    if float(_resid.max()) > 0.5:
        fig.add_trace(go.Scatter(
            x=list(d.index), y=_resid.values, mode="lines", name=residual_label,
            stackgroup="one", line=dict(width=0.6, color="#FFFFFF"),
            fillcolor="#E9EDF2", hoverinfo="skip", showlegend=False))

    # 末端标注前几大：读者需要一眼看到「现在最大的是谁」。
    # 位置按堆叠顺序累加，不能用「最新一期的排序」。
    _last = d.index[-1]
    _cur = d.iloc[-1]
    _pos, _acc = {}, 0.0
    for ind in order:
        _v = float(_cur.get(ind, 0) or 0)
        _pos[ind] = (_acc, _v)
        _acc += _v
    _top_now = [i for i in _cur.sort_values(ascending=False).index][:label_top]
    for ind in _top_now:
        _y0, _v = _pos.get(ind, (0.0, 0.0))
        if _v < 4:
            continue
        fig.add_annotation(x=_last, y=_y0 + _v / 2, text=f"{ind} {_v:.0f}%",
                           showarrow=False, xanchor="right", xshift=-4,
                           font=dict(size=10, color="#FFFFFF"),
                           bgcolor="#C9D3DE" if ind == merged_label else ind_color(ind),
                           borderpad=2, opacity=0.96)
    fig.update_yaxes(title_text="占前十大 %" if residual_label.startswith("其他")
                     else "占净值比 %", range=[0, 100], tickfont=dict(size=11))
    # 悬停：一条透明轨迹承载「该季度全部行业」的数值，一次看全。
    _txt = []
    for q in d.index:
        row = M.loc[q].sort_values(ascending=False)
        if float(_resid.get(q, 0) or 0) > 0.02:
            row = pd.concat([row, pd.Series({residual_label: float(_resid[q])})])
        lines = [f"{k} {v:.2f}" for k, v in row.items() if v >= 0.02]
        more = len(row) - len(lines)
        _txt.append(f"<b>{q}</b>　单位 %<br>" + "　".join(lines[:14])
                    + (f"<br>…另有 {more} 个行业 &lt;0.02%" if more > 0 else ""))
    fig.add_trace(go.Scatter(
        x=list(d.index), y=[50] * len(d), mode="markers",
        marker=dict(size=1, color="rgba(0,0,0,0)"), showlegend=False,
        customdata=_txt, hovertemplate="%{customdata}<extra></extra>",
        name="全部行业"))
    fig.update_layout(hovermode="closest",
                      hoverlabel=dict(bgcolor="rgba(255,255,255,0.98)", font_size=11,
                                      bordercolor=C.COLOR["line"], align="left",
                                      namelength=0))
    _ticks = [q for q in d.index if str(q).endswith("Q1")] + [d.index[-1]]
    fig.update_xaxes(tickmode="array", tickvals=_ticks, ticktext=_ticks,
                     tickangle=-45, tickfont=dict(size=10))
    fig = _base(fig, title, 520)
    fig.update_layout(margin=dict(l=8, r=14, t=44, b=44))
    return fig


def fig_csi_change(d: pd.DataFrame, top: int = 31) -> go.Figure:
    """拥挤度变化：本期 CSI − 上期 CSI，红＝变挤、绿＝变松。

    为什么单独画：水位（CSI）高不可怕，可怕的是「水位还在涨」。
    把变化和水平放在一起看，才知道谁在往拥挤区走、谁在往外撤。
    """
    if d.empty or "ΔCSI" not in d.columns:
        return _base(go.Figure(), "数据未就绪")
    x = d.dropna(subset=["ΔCSI"]).sort_values("ΔCSI")
    if len(x) > top:
        x = pd.concat([x.head(top // 2), x.tail(top // 2)])
    _lv = x["等级"] if "等级" in x else pd.Series("", index=x.index)
    fig = go.Figure(go.Bar(
        y=list(x.index), x=x["ΔCSI"], orientation="h",
        # 小样本用灰色：它的 ΔCSI 是「小样本里的排名变动」，不是资金方向，
        # 涂成红/绿会让人以为有资金在大举进出。
        marker_color=[CSI_COLORS["小样本"] if lv == "小样本"
                      else (C.COLOR["up"] if v > 0 else C.COLOR["down"])
                      for v, lv in zip(x["ΔCSI"], _lv)],
        text=[f"{v:+.0f}" + ("†" if lv == "小样本" else "")
              for v, lv in zip(x["ΔCSI"], _lv)], textposition="outside",
        textfont=dict(size=9),
        customdata=np.stack([x["CSI"], x.get("上期CSI", pd.Series(np.nan, index=x.index)),
                             _lv], axis=-1),
        hovertemplate=("<b>%{y}</b><br>ΔCSI %{x:+.1f}<br>"
                       "本期 CSI %{customdata[0]:.0f}（%{customdata[2]}）<br>"
                       "上期 CSI %{customdata[1]:.0f}"
                       "<extra></extra>")))
    fig.add_vline(x=0, line=dict(color=C.COLOR["line"], width=1))
    fig.update_xaxes(title_text="ΔCSI（正＝变挤，负＝变松）")
    fig.update_yaxes(tickfont=dict(size=9.5))
    fig = _base(fig, "", 560)
    fig.update_layout(margin=dict(l=6, r=46, t=16, b=34))
    return fig


def ind_legend_html(names: list[str],
                    values: "pd.Series | None" = None,
                    changes: "pd.Series | None" = None,
                    cols: int = 4,
                    min_value: float = 0.05) -> str:
    """行业图例：色块 + 名称 + 当前占比（+ 区间变化）。

    只列「有持仓」的行业（默认 >0.05%）：单只基金常常只持有 8–12 个行业，
    把 31 个全列出来会有一半是 0.0%，图例比图还长，反而看不清。
    想核对全部 31 个，用上方的堆叠图或下面的季度条形图。
    """
    cells = []
    for n in names:
        v_raw = np.nan if values is None else float(values.get(n, np.nan))
        if values is not None and (not np.isfinite(v_raw) or v_raw < min_value):
            continue
        v = "" if values is None else f"{v_raw:.1f}%"
        chg = ""
        if changes is not None:
            c = float(changes.get(n, np.nan))
            if np.isfinite(c):
                _c = C.COLOR["up"] if c > 0 else C.COLOR["down"] if c < 0 else C.COLOR["text_dim"]
                chg = (f'<span style="color:{_c};font-weight:600;margin-left:6px;'
                       f'font-size:11.5px">{c:+.1f}pp</span>')
        cells.append(
            f'<div style="display:flex;align-items:center;gap:7px;padding:4px 5px;'
            f'border-bottom:1px solid {C.COLOR["grid"]};min-width:0">'
            f'<span style="width:11px;height:11px;border-radius:2px;flex:0 0 11px;'
            f'background:{ind_color(n)};display:inline-block"></span>'
            f'<span style="font-size:12.5px;color:#37485C;white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis">{n}</span>'
            f'<span style="margin-left:auto;font-size:12.5px;color:#1F2D3D;'
            f'font-variant-numeric:tabular-nums;font-weight:700">{v}{chg}</span></div>')
    if not cells:
        return ""
    return (f'<div style="display:grid;grid-template-columns:repeat({cols},minmax(0,1fr));'
            f'gap:0 16px;margin:2px 0 8px">' + "".join(cells) + '</div>')


# ---------------------------------------------------------------------------
# 11. 拥挤度：31 个行业一根条，颜色即等级
# ---------------------------------------------------------------------------
CSI_COLORS = {
    "极度拥挤": "#C62828", "拥挤": "#E8710A", "中性": "#8A97A5",
    "偏冷": "#4C8DDF", "冷门": "#1F4E9C", "数据不足": "#C3CBD4",
    # 小样本：持有基金数 <150 或前十配置 <0.5%。用灰色而不是任何暖色，
    # 因为它既不该被读成「拥挤」，也不该被读成「冷门」——它是不评级。
    "小样本": "#9AA6B2",
}


def fig_csi_bar(cs: pd.DataFrame) -> go.Figure:
    """按 CSI 排序的横向条形图，颜色＝拥挤等级，条端标注分位。

    这张图替代原来的「四象限散点」承担「一眼看完 31 个行业」的职责：
    散点要读两个轴，条形只需要读长度和颜色。
    """
    if cs.empty:
        return _base(go.Figure(), "数据未就绪")
    # 排序：正常样本按 CSI 升序（最拥挤画在最上面），不评级的小样本统一压在底部。
    # 纯按 CSI 排会让综合 83 插在石油石化 85 与银行 78 之间，灰色条夹在红条里，
    # 读者会以为「这个灰的也算拥挤」。
    d = cs.dropna(subset=["CSI"]).copy()
    d["_sm"] = (d["等级"].astype(str) == "小样本").astype(int)
    d = d.sort_values(["_sm", "CSI"], ascending=[False, True])
    cols = [CSI_COLORS.get(l, "#C3CBD4") for l in d["等级"]]
    # 把悬浮文本**预先拼成字符串**，而不是用 hovertemplate 的格式符：
    # 小样本行要说的话（「按三分位本来是 83 分，但只有 58 只基金持有」）
    # 是条件分支，模板语法表达不了。
    _hv = []
    for i, r in d.iterrows():
        _base_txt = (f"<b>{i}</b><br>CSI {r['CSI']:.1f} = "
                     f"({r['配置分位']:.0f} + {r['价格分位']:.0f} + {r['资金分位']:.0f}) / 3<br>"
                     f"配置分位 {r['配置分位']:.0f}｜价格分位 {r['价格分位']:.0f}｜"
                     f"资金分位 {r['资金分位']:.0f}")
        if r.get("可靠") == "小样本":
            _base_txt += (f"<br><span style='color:#C62828'>⚠ 小样本，不评级</span>"
                          f"<br>全市场仅 {r.get('持有基金数', float('nan')):.0f} 只基金持有，"
                          f"前十配置 {r.get('配置', float('nan')):.2f}%"
                          f"<br>按三分位本应判为「{r.get('原等级', '—')}」，"
                          f"但这个体量对组合影响不足 0.2pp，"
                          f"分位噪声远大于信息")
        else:
            _base_txt += f"<br>等级 {r['等级']}"
        _hv.append(_base_txt)
    fig = go.Figure(go.Bar(
        y=list(d.index), x=d["CSI"], orientation="h", marker_color=cols,
        text=[f'{v:.0f}　{l}' + ("†" if rel == "小样本" else "")
              for v, l, rel in zip(d["CSI"], d["等级"], d.get("可靠", [""] * len(d)))],
        textposition="outside", textfont=dict(size=10.5),
        customdata=np.array(_hv).reshape(-1, 1),
        hovertemplate="%{customdata[0]}<extra></extra>"))
    for v in (25, 45, 65, 80):
        fig.add_vline(x=v, line=dict(color=C.COLOR["line"], width=1, dash="dot"))
    fig.update_xaxes(title_text="拥挤度指数 CSI（0＝最冷，100＝最挤）　†＝小样本",
                     range=[0, 132], tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=11.5))
    fig = _base(fig, "", 730)
    fig.update_layout(margin=dict(l=8, r=92, t=14, b=40),
                      hoverlabel=dict(font_size=11))
    return fig


# ---------------------------------------------------------------------------
# 6. 分歧结构：广度 × 平均变化
# ---------------------------------------------------------------------------
def fig_breadth_vs_move(d: pd.DataFrame,
                        small: list[str] | None = None) -> go.Figure:
    """横轴＝多少基金在加，纵轴＝价格中性后的净买卖（pp）。

    两轴独立：广度回答「共识度」，幅度回答「力度」。
    左上（少数大基金在加）与右下（多数小基金在加、净额却为负）含义相反。

    纵轴必须用「平均主动pp」而不是原始变化：原始变化含价格漂移，
    暴跌季所有行业都会落在左下角，看不出谁在被真买。

    小样本行业画成灰点并标 †：它们的位置由几十只基金决定，
    放进「共识/撤离」的分组里会污染判断。
    """
    if d.empty:
        return _base(go.Figure(), "数据未就绪")
    d = d.copy()
    small = small or []
    # 广度表的索引是 RangeIndex，行业名在「行业」列里；
    # 直接用 d.index 会拿到整数，后面拼 † 就会 TypeError。
    _nm = (d["行业"].astype(str) if "行业" in d.columns
           else pd.Series(d.index.astype(str), index=d.index))
    d["_nm"] = _nm
    d["_sm"] = _nm.isin(small)
    _y = "平均主动pp" if "平均主动pp" in d.columns else "平均变化pp"
    cmap = {"共识形成": C.COLOR["up"], "系统性撤离": C.COLOR["down"],
            "多空对撞": "#E8710A", "分歧": C.COLOR["text_dim"]}
    # 只在「明显偏离中心」的点上写字：31 个点里有一半挤在 (55%, 0) 附近，
    # 全标会叠成一团黑字，反而看不清。小样本一律标（它们是灰色的，需要识别）。
    notable = ((d["加仓占比"] - 50).abs() >= 15) | (d[_y].abs() >= 1.0) | d["_sm"]
    fig = go.Figure()
    for tag, g in d.groupby("判读", dropna=False):
        lab = [str(nm) + ("†" if sm else "") if notable.get(ix, False) else ""
               for ix, nm, sm in zip(g.index, g["_nm"], g["_sm"])]
        fig.add_trace(go.Scatter(
            x=g["加仓占比"], y=g[_y], mode="markers+text", name=f"{tag}（{len(g)}）",
            marker=dict(size=11 + np.sqrt(g["样本基金数"]) / 9,
                        color=[CSI_COLORS["小样本"] if sm else cmap.get(tag, C.COLOR["text_dim"])
                               for sm in g["_sm"]],
                        opacity=0.85, line=dict(width=0.8, color="#FFFFFF")),
            text=lab, textposition="top center",
            textfont=dict(size=11, color="#37485C"),
            customdata=np.stack([g["加仓基金数"], g["减仓基金数"], g["持有基金数"]], axis=-1),
            hovertemplate=("<b>%{text}</b><br>加仓基金占比 %{x:.1f}%<br>"
                           "价格中性净额 %{y:+.2f}pp<br>"
                           "加 %{customdata[0]:.0f} / 减 %{customdata[1]:.0f}"
                           "（持有 %{customdata[2]:.0f}）<extra></extra>"),
        ))
    fig.add_vline(x=50, line=dict(color=C.COLOR["threshold"], width=1.4, dash="dash"),
                  annotation_text="50%：左＝多数在减，右＝多数在加",
                  annotation_position="top", annotation_font=dict(size=10, color=C.COLOR["threshold"]),
                  annotation_bgcolor="rgba(255,255,255,0.85)")
    fig.add_hline(y=0, line=dict(color=C.COLOR["text_dim"], width=1, dash="dot"),
                  annotation_text="净额 0：上＝净买入，下＝净卖出",
                  annotation_position="right", annotation_font=dict(size=10, color=C.COLOR["text_dim"]),
                  annotation_bgcolor="rgba(255,255,255,0.85)")
    fig.update_xaxes(title_text="加仓基金占比（%）", range=[10, 95],
                     tickfont=dict(size=11))
    fig.update_yaxes(title_text="价格中性净额（pp）", tickfont=dict(size=11))
    fig = _base(fig, "", 470, legend=True)
    fig.update_layout(margin=dict(l=64, r=26, t=16, b=44))
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
                       arrowhead=0, ay=-24, font=dict(size=10, color=C.COLOR["up"]))
    fig.add_annotation(x=s.idxmin(), y=_lo, text=f"最低 {_lo:.1f}%", showarrow=True,
                       arrowhead=0, ay=24, font=dict(size=10, color=C.COLOR["down"]))
    # 上下留 18% 余量：最高/最低点的文字标注画在点外侧，不留余量会被画到坐标区外
    _pad = max((_hi - _lo) * 0.22, 0.8)
    fig.update_yaxes(title_text="占前十大组合 %", range=[_lo - _pad, _hi + _pad],
                     tickfont=dict(size=11))
    # 31 个季度标签全标会糊成一片：只标每年 Q1 + 末期
    _tk = [q for q in s.index if str(q).endswith("Q1")] + [s.index[-1]]
    fig.update_xaxes(tickmode="array", tickvals=_tk, ticktext=_tk,
                     tickangle=-45, tickfont=dict(size=10))
    fig = _base(fig, f"{industry} 配置比例的历史位置{title_extra}", 380)
    # 右侧要放「历史均值」标注，不留边距会被裁成「历…」
    fig.update_layout(margin=dict(l=6, r=86, t=48, b=46),
                      hoverlabel=dict(font_size=11))
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


# ---------------------------------------------------------------------------
# 12. 拥挤度折线：31 期怎么变、什么时候穿过阈值
# ---------------------------------------------------------------------------
CSI_BANDS = [(80, 101, "rgba(198,40,40,0.16)", "≥80 极度拥挤"),
             (65, 80, "rgba(232,113,10,0.14)", "65–80 拥挤"),
             (45, 65, "rgba(138,151,165,0.04)", "45–65 中性"),
             (25, 45, "rgba(76,141,223,0.12)", "25–45 偏冷"),
             (0, 25, "rgba(31,78,156,0.16)", "<25 冷门")]


def fig_csi_trend(H: pd.DataFrame, sel: str | None = None,
                  show_band: bool = True) -> go.Figure:
    """31 期 CSI 折线：看「拥挤度怎么变」，而不是只看当期一个点。

    为什么必须画折线：CSI 是一个 0–100 的水位，单看当期数字无法判断
    「它是在涨还是在退」。同样的 88 分，从 60 涨上来和从 95 跌下来，
    含义完全相反。
    """
    if H.empty:
        return _base(go.Figure(), "数据未就绪")
    d = H.dropna(how="all")
    fig = go.Figure()

    # 阈值背景带：先把「哪里是危险区」画出来，再看线怎么走
    for lo, hi, color, lab in CSI_BANDS:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, line_width=0,
                      layer="below", showlegend=False)
    # 阈值标注放**图内左上**：放右侧会被坐标区裁掉（只露出「拥挤度」三个字），
    # 放左侧四条线互不遮挡，也不占右边距。
    for y, txt in [(80, "80 极度拥挤"), (65, "65 拥挤"), (45, "45 中性"), (25, "25 冷门")]:
        fig.add_hline(y=y, line=dict(color="#9AA6B2", width=0.9, dash="dot"),
                      annotation_text=txt, annotation_position="top left",
                      annotation_font=dict(size=8, color="#5A6B7D"),
                      annotation_bgcolor="rgba(255,255,255,0.75)")

    # 全市场中位：代表「整体水位」，是所有行业的共同背景
    med = d.median(axis=1)
    fig.add_trace(go.Scatter(
        x=list(d.index), y=med.values, mode="lines", name="全市场中位",
        line=dict(color="#7C8B9C", width=2.4),
        hovertemplate="%{x}<br>全市场中位 %{y:.1f}<extra></extra>"))

    # 不画 31 条线——那是一团蜘蛛网，谁在涨谁在跌都看不出来。
    # 改成「31 个行业的 25–75% 分位带」：它表达的是「市场整体的分布宽度」，
    # 只占一层视觉，不跟主角（选中行业）抢注意力。
    if show_band:
        q1 = d.quantile(0.25, axis=1)
        q3 = d.quantile(0.75, axis=1)
        xs = list(d.index)
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1],
            y=list(q3.values) + list(q1.values)[::-1],
            fill="toself", fillcolor="rgba(124,139,156,0.20)",
            line=dict(width=0), hoverinfo="skip", showlegend=True,
            name="31 行业中 25–75% 区间"))

    if sel and sel in d.columns:
        s = d[sel]
        _c = CSI_COLORS.get("中性", "#1F6FEB")
        fig.add_trace(go.Scatter(
            x=list(d.index), y=s.values, mode="lines+markers", name=sel,
            line=dict(color="#1F6FEB", width=2.8),
            marker=dict(size=3.5, color="#1F6FEB"),
            hovertemplate=f"<b>{sel}</b><br>%{{x}}：%{{y:.1f}}<extra></extra>"))
        # 穿过 80 的时点：上穿红三角、下穿绿三角，一眼看出「什么时候变挤/变松」
        up, dn = [], []
        for i in range(1, len(s)):
            a, b = s.iloc[i - 1], s.iloc[i]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            if a < 80 <= b:
                up.append((s.index[i], b))
            elif a >= 80 > b:
                dn.append((s.index[i], b))
        if up:
            fig.add_trace(go.Scatter(
                x=[x for x, _ in up], y=[y for _, y in up], mode="markers",
                name="上穿 80", marker=dict(symbol="triangle-up", size=10,
                                            color=C.COLOR["up"],
                                            line=dict(width=1, color="#FFFFFF")),
                hovertemplate="%{x}<br>上穿 80：%{y:.1f}<extra></extra>"))
        if dn:
            fig.add_trace(go.Scatter(
                x=[x for x, _ in dn], y=[y for _, y in dn], mode="markers",
                name="下穿 80", marker=dict(symbol="triangle-down", size=10,
                                            color=C.COLOR["down"],
                                            line=dict(width=1, color="#FFFFFF")),
                hovertemplate="%{x}<br>下穿 80：%{y:.1f}<extra></extra>"))

    fig.update_yaxes(title_text="CSI（0＝最冷，100＝最挤）", range=[0, 104],
                     tickfont=dict(size=10))
    # x 轴只标每年 Q1 + 末期：31 个标签全打出来会挤成一片，
    # 读图只需要知道「大概哪一年」。
    _ticks = [q for q in d.index if str(q).endswith("Q1")] + [d.index[-1]]
    fig.update_xaxes(tickmode="array", tickvals=_ticks, ticktext=_ticks,
                     tickangle=-45, tickfont=dict(size=9.5))
    fig = _base(fig, "", 520, legend=True)
    fig.update_layout(margin=dict(l=56, r=18, t=56, b=62),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                  x=0, font=dict(size=10)))
    return fig


# ---------------------------------------------------------------------------
# 13. 拥挤度全景热力图：31 个行业 × 31 期，一屏看全
# ---------------------------------------------------------------------------
_CSI_SCALE = [
    # 分段色阶（不是渐变）：颜色只表达「档位」，不表达「小数差多少」。
    # 饱和度刻意压低一档，31×31 格满屏时纯红纯蓝会刺眼。
    (0.00, "#2C5A96"), (0.25, "#2C5A96"),
    (0.25, "#93B7E0"), (0.45, "#93B7E0"),
    (0.45, "#F5F7F9"), (0.65, "#F5F7F9"),
    (0.65, "#EEA758"), (0.80, "#EEA758"),
    (0.80, "#C04B4B"), (1.00, "#C04B4B"),
]


def fig_csi_heatmap(H: pd.DataFrame, small: list[str] | None = None,
                    height: int = 700) -> go.Figure:
    """31 个行业 × 31 期的 CSI 热力图：颜色＝档位。

    为什么用它替代「31 条线叠加」：叠加 31 条线谁都看不清（蜘蛛网），
    而热力图一屏就能看出「哪一行什么时候变红」——这是横向对比最有效的编码。
    行按当期 CSI 降序排（不评级的排最后），列按时间。
    """
    if H.empty:
        return _base(go.Figure(), "数据未就绪")
    d = H.dropna(how="all")
    if d.empty:
        return _base(go.Figure(), "数据未就绪")
    small = small or []
    last = d.iloc[-1]
    order = [c for c in d.columns if c not in small]
    order.sort(key=lambda c: (-(last.get(c) if np.isfinite(last.get(c, np.nan)) else -1)))
    order += [c for c in d.columns if c in small]

    z = d[order].T.values
    ylab = [c + ("†" if c in small else "") for c in order]
    hv = [[f"<b>{ind}</b><br>{q}：CSI {z[i, j]:.1f}"
           + ("<br>小样本，不评级" if ind.replace("†", "") in small else "")
           for j, q in enumerate(d.index)] for i, ind in enumerate(ylab)]
    fig = go.Figure(go.Heatmap(
        z=z, x=list(d.index), y=ylab,
        colorscale=_CSI_SCALE, zmin=0, zmax=100,
        xgap=1, ygap=1,
        customdata=np.array(hv),
        hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title=dict(text="CSI", font=dict(size=10)),
                      thickness=9, len=0.5, x=1.012, xpad=4,
                      tickfont=dict(size=9.5), outlinewidth=0,
                      tickvals=[12, 35, 55, 72, 90],
                      ticktext=["冷门", "偏冷", "中性", "拥挤", "极度"])))
    _ticks = [q for q in d.index if str(q).endswith("Q1")] + [d.index[-1]]
    fig.update_xaxes(tickmode="array", tickvals=_ticks, ticktext=_ticks,
                     tickangle=-45, tickfont=dict(size=9.5), side="bottom")
    fig.update_yaxes(tickfont=dict(size=10), autorange="reversed")
    fig = _base(fig, "", height)
    # 右边必须留够：colorbar 在坐标区外，r=16 会把它切掉一半
    fig.update_layout(margin=dict(l=92, r=86, t=16, b=62))
    return fig


def fig_threshold(bins: pd.DataFrame, events: pd.DataFrame) -> go.Figure:
    """阈值实证：各档的次季超额收益 + 胜率 + 样本量。

    为什么把胜率和均值放一张图：只看均值会被少数极端值带偏，
    只看胜率会忽略幅度；两者一起看，才知道「这个档位值不值得下注」。
    """
    if bins.empty:
        return _base(go.Figure(), "数据未就绪")
    d = bins.reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    cols = [C.COLOR["up"] if v > 0 else C.COLOR["down"] for v in d["均值"]]
    fig.add_trace(go.Bar(
        x=d["档"], y=d["均值"], name="次季超额（%）", marker_color=cols,
        text=[f"{v:+.2f}%" for v in d["均值"]], textposition="outside",
        textfont=dict(size=11, color="#37485C"), width=0.52,
        customdata=np.stack([d["样本"], d["中位"], d["t"]], axis=-1),
        hovertemplate=("<b>%{x}</b><br>次季超额均值 %{y:+.2f}%<br>"
                       "中位 %{customdata[1]:+.2f}%<br>样本 %{customdata[0]:.0f}<br>"
                       "t 值 %{customdata[2]:+.2f}<extra></extra>")),
        secondary_y=False)
    fig.add_trace(go.Scatter(
        x=d["档"], y=d["胜率"], name="胜率（%）", mode="lines+markers",
        line=dict(color="#1F6FEB", width=2), marker=dict(size=9),
        # 不在线上写数字：双轴图里数字必然与柱子/坐标轴打架，
        # 胜率已经在下方表里逐档列出，这里只保留悬浮。
        hovertemplate="<b>%{x}</b><br>次季跑赢同行概率 %{y:.1f}%<extra></extra>"),
        secondary_y=True)
    fig.add_hline(y=0, line=dict(color=C.COLOR["line"], width=1), secondary_y=False)
    fig.add_hline(y=50, line=dict(color="#B9C4D0", width=1, dash="dot"),
                  secondary_y=True)
    fig.update_yaxes(title_text="次季超额收益（%）", secondary_y=False,
                     range=[-1.1, 1.1], dtick=0.5)
    fig.update_yaxes(title_text="胜率（%）", secondary_y=True, range=[36, 56],
                     dtick=5, title_standoff=12)
    fig = _base(fig, "", 400, legend=True)
    fig.update_layout(margin=dict(l=58, r=70, t=42, b=56),
                      xaxis=dict(tickfont=dict(size=11)),
                      hoverlabel=dict(font_size=11))
    return fig


def _band_of(v: float) -> str:
    if not np.isfinite(v):
        return "数据不足"
    if v >= 80:
        return "极度拥挤"
    if v >= 65:
        return "拥挤"
    if v >= 45:
        return "中性"
    if v >= 25:
        return "偏冷"
    return "冷门"


def fig_csi_snapshot(H: pd.DataFrame, quarter: str,
                     small: list[str] | None = None,
                     height: int = 720) -> go.Figure:
    """某一个季度、31 个行业的 CSI 一次全部排出来（横向条形）。

    为什么需要它：热力图能看「谁什么时候变红」，但每一格只有一个颜色，
    想知道「这一期到底谁 88 分、谁 71 分」必须逐个悬停，很费劲。
    这里把选中的那一期整列展开成条形，所有数值一次性标在条端。
    """
    if H.empty or quarter not in H.index:
        return _base(go.Figure(), "数据未就绪")
    small = small or []
    s = H.loc[quarter].dropna()
    if s.empty:
        return _base(go.Figure(), "数据未就绪")
    d = s.to_frame("CSI")
    d["_sm"] = d.index.isin(small)
    d = d.sort_values(["_sm", "CSI"], ascending=[True, False])
    cols = [CSI_COLORS["小样本"] if sm else CSI_COLORS.get(_band_of(v), "#C3CBD4")
            for v, sm in zip(d["CSI"], d["_sm"])]
    fig = go.Figure(go.Bar(
        y=[i + ("†" if sm else "") for i, sm in zip(d.index, d["_sm"])],
        x=d["CSI"], orientation="h", marker_color=cols,
        text=[f"{v:.0f}" for v in d["CSI"]], textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>%{x:.1f} 分<extra></extra>"))
    for x, lab in [(80, "80"), (65, "65"), (45, "45"), (25, "25")]:
        fig.add_vline(x=x, line=dict(color="#B9C4D0", width=0.9, dash="dot"))
    fig.update_xaxes(title_text=f"{quarter} 的 CSI（0＝最冷，100＝最挤）",
                     range=[0, 108], tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=11.5))
    fig = _base(fig, "", height)
    fig.update_layout(margin=dict(l=8, r=44, t=12, b=40))
    return fig
