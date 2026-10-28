# -*- coding: utf-8 -*-
"""③ 行业地图：每个行业现在站在自己历史的什么位置。

只回答一个问题：**贵不贵、挤不挤**。
答案不是「配置 8% 算高还是低」，而是「这个 8% 在它自己过去 7 年的区间里排第几」。

口径
----
· 时间序列只用**全持仓期**（半年报 Q2、年报 Q4）：季报只披露前十大，
  拿它算「配置比例」会系统性低估每个行业，分位数会整体漂移；
· 配置 = 占净值比（不是占已披露组合），所以之和≈股票仓位，可横向比；
· 分位 = 当前值在该行业历史序列中的百分位（0＝历史最低，100＝历史最高）；
· 集中度 HHI = Σ(行业占比/100)²，等价于「随机抽两块钱，落在同一行业的概率」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import canon as cn
import config as C


def _base(fig, title, h=430, legend=False):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=C.COLOR["text"]), x=0.005),
        height=h, margin=dict(l=6, r=10, t=46, b=10),
        paper_bgcolor=C.COLOR["bg"], plot_bgcolor=C.COLOR["bg"],
        font=dict(size=11, color=C.COLOR["text"]), showlegend=legend,
        legend=dict(orientation="h", y=-0.16, x=0, font=dict(size=9)),
        hoverlabel=dict(font_size=11),
    )
    fig.update_xaxes(gridcolor=C.COLOR["grid"], zerolinecolor=C.COLOR["line"])
    fig.update_yaxes(gridcolor=C.COLOR["grid"], zerolinecolor=C.COLOR["line"])
    return fig


@st.cache_data(show_spinner=False)
def _series() -> tuple[pd.DataFrame, pd.Series]:
    """全持仓期的行业占比序列（行＝期次，列＝行业）与 HHI。"""
    rows = {}
    for q in cn.full_quarters():
        p = cn.panel_full(q, normalize=False)      # 占净值比，不归一化
        u = cn.universe(q)
        idx = p.index.intersection(list(u)) if u else p.index
        if len(idx) < 300:
            continue
        rows[q] = (p.loc[idx].mean(), len(idx))
    # 样本不全的期次必须剔除：补抓历史时 2021 年只有 1400 只基金，
    # 而 2026 年有 4100 只，混在一起算分位，前面几期全是「小样本偏差」。
    if rows:
        _mx = max(v[1] for v in rows.values())
        rows = {k: v[0] for k, v in rows.items() if v[1] >= max(1000, _mx * 0.55)}
    M = pd.DataFrame(rows).T.sort_index()
    hhi = ((M / 100) ** 2).sum(axis=1) if not M.empty else pd.Series(dtype=float)
    return M, hhi


def _pct(M: pd.DataFrame) -> pd.DataFrame:
    """逐行业算历史分位（0–100）。"""
    return M.rank(pct=True) * 100


def render(h) -> None:
    M, hhi = _series()
    if M.empty or len(M) < 4:
        h('<div class="cg-note">全持仓期不足（需至少 4 个半年报），'
          '先运行 <span class="mono">tools/run_ak_all.py</span> 补历史。</div>')
        return
    q = M.index[-1]
    cur = M.iloc[-1]
    pct = _pct(M).iloc[-1]
    med = M.median()
    hi_q = M.idxmax()
    lo_q = M.idxmin()

    t = pd.DataFrame({
        "配置%": cur, "历史分位": pct, "历史中位": med,
        "历史最高": M.max(), "最高出现在": hi_q,
        "历史最低": M.min(), "最低出现在": lo_q,
        "样本期数": len(M),
    })
    t["较中位pp"] = t["配置%"] - t["历史中位"]
    t = t.sort_values("历史分位", ascending=False)
    t.index.name = "行业"

    # ---------------- KPI ----------------
    top3 = cur.nlargest(3)
    h(f'<div class="sec">行业地图 · {q}（全持仓口径，{len(M)} 个半年报期）</div>')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最拥挤", f"{pct.idxmax()} · {pct.max():.0f} 分位",
              help="当前配置在自己历史区间里排得最高的行业")
    c2.metric("最冷门", f"{pct.idxmin()} · {pct.min():.0f} 分位",
              help="当前配置在自己历史区间里排得最低的行业")
    c3.metric("行业集中度 HHI", f"{hhi.iloc[-1]:.4f}",
              delta=f"{hhi.iloc[-1] - hhi.iloc[-2]:+.4f}" if len(hhi) > 1 else None,
              delta_color="off")
    c4.metric("前三大行业合计", f"{top3.sum():.1f}%",
              delta=f"{top3.sum() - M.iloc[-2].nlargest(3).sum():+.1f}pp" if len(M) > 1 else None,
              delta_color="off")

    st.write("")

    # ---------------- 1. 分位地图 ----------------
    h('<div class="sec">① 一张图看清：谁在历史高位、谁在历史低位</div>')
    d = t.copy()
    # 只标注「分位极值」或「配置较大」的行业：31 个全标会在左下角糊成一团，
    # 而左下角那些 0.5% 配置的行业本来也不是这张图要讲的重点。
    lab = [i if (r["历史分位"] >= 80 or r["历史分位"] <= 20 or r["配置%"] >= 5)
           else "" for i, r in d.iterrows()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["配置%"], y=d["历史分位"], mode="markers+text", text=lab,
        # 分位接近 100 的点若把标签放在上方，会被画到坐标区外（电子就丢过标签），
        # 顶部的点改为把标签放在下方；其余仍放上方。
        textposition=["bottom center" if v >= 95 else "top center"
                      for v in d["历史分位"]],
        textfont=dict(size=9, color=C.COLOR["text_dim"]),
        marker=dict(size=np.minimum(9 + d["较中位pp"].abs() * 2.0, 30),
                    color=[C.COLOR["up"] if v > 0 else C.COLOR["down"]
                           for v in d["较中位pp"]],
                    opacity=0.72, line=dict(width=0.8, color="#FFFFFF")),
        customdata=np.stack([d["历史中位"], d["较中位pp"], d["最高出现在"],
                             d["最低出现在"]], axis=-1),
        hovertemplate=("<b>%{text}</b><br>当前配置 %{x:.2f}%<br>历史分位 %{y:.0f}<br>"
                       "历史中位 %{customdata[0]:.2f}%<br>较中位 %{customdata[1]:+.2f}pp<br>"
                       "最高 %{customdata[2]}　最低 %{customdata[3]}<extra></extra>"),
        showlegend=False))
    fig.add_hline(y=80, line=dict(color=C.COLOR["up"], width=1, dash="dot"),
                  annotation_text="80 分位：历史高位", annotation_position="right",
                  annotation_font=dict(size=9, color=C.COLOR["up"]))
    fig.add_hline(y=20, line=dict(color=C.COLOR["down"], width=1, dash="dot"),
                  annotation_text="20 分位：历史低位", annotation_position="right",
                  annotation_font=dict(size=9, color=C.COLOR["down"]))
    # 对数横轴：配置从 0.1% 到 22% 跨两个数量级，线性轴会把 30 个行业挤在左边
    fig.update_xaxes(title_text="当前配置（占净值比 %，对数轴）",
                     type="log", range=[-1.05, 1.65],
                     tickvals=[0.1, 0.3, 1, 3, 10, 30],
                     ticktext=["0.1", "0.3", "1", "3", "10", "30"])
    fig.update_yaxes(title_text="该行业自身的历史分位（0–100）", range=[-4, 108])
    fig.update_layout(margin=dict(l=6, r=46, t=46, b=10))
    st.plotly_chart(_base(fig, "横轴是配置多高，纵轴是「相对自己历史」有多高", 450),
                    width="stretch", config={"displayModeBar": False})
    _hi = t.head(4)
    _lo = t.tail(4).iloc[::-1]
    h(f'<div class="cg-note"><b>右上角＝又高又在历史极值</b>：'
      + "、".join(f'{i}（{r["配置%"]:.1f}%，{r["历史分位"]:.0f} 分位）' for i, r in _hi.iterrows())
      + f'；<b>下方＝历史低位</b>：'
      + "、".join(f'{i}（{r["配置%"]:.1f}%，{r["历史分位"]:.0f} 分位）' for i, r in _lo.iterrows())
      + '。<br><b>所以呢</b>：横轴告诉你「这个行业在组合里有多重要」，'
        '纵轴才告诉你「这个重要性是不是异常」。'
        '配置绝对值低但分位 95 的行业，是「从没人要到很多人要」的转折，'
        '往往比一直高配的行业更有信息量。</div>')

    # ---------------- 2. 集中度 ----------------
    h('<div class="sec">② 钱是不是越来越挤在少数行业里</div>')
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=list(hhi.index), y=hhi.values, mode="lines+markers",
                              name="行业集中度 HHI", line=dict(color=C.COLOR["accent"], width=2),
                              marker=dict(size=5),
                              hovertemplate="%{x}　HHI %{y:.4f}<extra></extra>"),
                   secondary_y=False)
    top3s = M.apply(lambda r: r.nlargest(3).sum(), axis=1)
    fig2.add_trace(go.Scatter(x=list(top3s.index), y=top3s.values, mode="lines+markers",
                              name="前三大行业合计 %", line=dict(color=C.COLOR["price"],
                                                           width=2, dash="dot"),
                              marker=dict(size=5),
                              hovertemplate="%{x}　前三大 %{y:.1f}%<extra></extra>"),
                   secondary_y=True)
    fig2.update_yaxes(title_text="HHI（0.05 以下＝分散）", secondary_y=False)
    fig2.update_yaxes(title_text="前三大%", secondary_y=True)
    # 只标半年报期的刻度：11 个季度全标会挤成一片，而这页本来就是半年频
    _tk = [x for x in hhi.index if str(x).endswith("Q2")]
    fig2.update_xaxes(tickangle=-45, tickmode="array", tickvals=_tk, ticktext=_tk)
    fig2.update_layout(margin=dict(l=6, r=72, t=46, b=10))
    st.plotly_chart(_base(fig2, "集中度：行业层面的抱团程度", 380, legend=True),
                    width="stretch", config={"displayModeBar": False})
    _d = hhi.iloc[-1] - hhi.iloc[0]
    h(f'<div class="cg-note">HHI 从 {hhi.iloc[0]:.4f}（{hhi.index[0]}）走到 '
      f'{hhi.iloc[-1]:.4f}（{hhi.index[-1]}），变化 <b>{_d:+.4f}</b>；'
      f'前三大行业合计 {top3s.iloc[0]:.1f}% → <b>{top3s.iloc[-1]:.1f}%</b>。'
      f'<br><b>所以呢</b>：HHI 上升说明钱在往少数行业集中，'
      f'这时「选对行业」比「选对个股」更重要，行业判断的权重应该提高；'
      f'HHI 下降则相反，个股 alpha 更值钱。'
      f'当前 HHI {hhi.iloc[-1]:.3f} 属于'
      f'{"偏集中" if hhi.iloc[-1] > hhi.median() else "偏分散"}'
      f'（历史中位 {hhi.median():.3f}）。</div>')

    # ---------------- 3. 明细 ----------------
    h('<div class="sec">③ 31 个行业的历史位置（按分位排序）</div>')
    show = t.reset_index()[["行业", "配置%", "历史分位", "历史中位", "较中位pp",
                            "历史最高", "最高出现在", "历史最低", "最低出现在"]]
    st.dataframe(
        show, width="stretch", hide_index=True, height=560,
        column_config={
            "配置%": st.column_config.NumberColumn(format="%.2f"),
            "历史分位": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100,
                help="当前配置在该行业历史序列中的百分位"),
            "历史中位": st.column_config.NumberColumn(format="%.2f"),
            "较中位pp": st.column_config.NumberColumn(format="%+.2f"),
            "历史最高": st.column_config.NumberColumn(format="%.2f"),
            "历史最低": st.column_config.NumberColumn(format="%.2f"),
        })
    h(f'<div class="tiny">口径：只用全持仓期（{M.index[0]}–{M.index[-1]}，'
      f'共 {len(M)} 个半年报），每期取样本基金等权平均的<b>占净值比</b>；'
      f'季报期（Q1/Q3）不参与，因为只披露前十大，会把每个行业的占比都压低。'
      f'分位＝当前值在自身历史中的排名，不是跨行业比较——'
      f'银行 20% 与电子 20% 的「贵」不是同一件事。</div>')
