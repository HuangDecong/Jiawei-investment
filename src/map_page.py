# -*- coding: utf-8 -*-
"""③ 行业拥挤度：像之前那样「左侧行业列表 + 右侧单行业详情」，数据换成 31 期全量。

为什么回到列表式
----------------
一张散点图回答不了「电子现在到底挤不挤、比上季如何、比去年如何」。
投资者真正会做的动作是：先扫一遍 31 个行业找异常，再点进去看这一个行业的历史。
所以：左侧列表按「历史分位」排序并带色点，右侧给该行业的
配置走势（31 期）、环比、同比、主动/被动与一句话判断。

口径
----
· 配置比例用**前十大归一化**：31 期都有，才能做环比与同比；
· 「全持仓」口径只在半年报期存在，用于水平与主动/被动，单列并标注期次；
· 分位＝当前值在**自身 31 期**历史中的排名（0＝历史最低，100＝历史最高）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import canon as cn
import config as C
import report_charts as rc


def _tone(p: float) -> tuple[str, str]:
    """分位 → (颜色, 等级名)。"""
    if p >= 80:
        return C.COLOR["level_极度拥挤"], "历史高位"
    if p >= 60:
        return C.COLOR["level_拥挤"], "偏高"
    if p >= 40:
        return C.COLOR["level_中性"], "中性"
    if p >= 20:
        return C.COLOR["level_偏拥挤"], "偏低"
    return C.COLOR["level_显著低配"], "历史低位"


def _verdict(r) -> str:
    """按「分位 + 环比 + 同比」给一句可执行判断（规则事前固定）。"""
    p, d1, d2 = r["历史分位"], r["Δ上季pp"], r["Δ上年pp"]
    if p >= 80 and d1 > 0 and d2 > 0:
        return ("分位已到历史高位，且环比、同比同向上行——拥挤度还在累积。"
                "这时买入等于在别人的浮盈上加仓，仓位要更轻、止损要更紧。")
    if p <= 20 and d1 < 0 and d2 < 0:
        return ("分位在历史低位，且环比、同比同向下行——机构在持续撤。"
                "不要因为「便宜」就买，先等主动项转正。")
    if np.sign(d1) != np.sign(d2):
        return ("环比与同比方向相反：这是季度内的摆动，不是趋势，"
                "别拿它做年度配置依据，只适合做交易。")
    if p >= 60:
        return "分位偏高但变动温和：拥挤度不极端，按个股逻辑处理，不必因行业减仓。"
    return "分位偏低且变动温和：没有明显的拥挤或抛弃信号，按个股逻辑处理即可。"


def render(h) -> None:
    if not (cn.available() or cn.fallback_available()):
        h('<div class="cg-note">未找到持仓数据。</div>')
        return
    comp = cn.compare_periods()
    if comp.empty:
        h('<div class="cg-note">数据不足。</div>')
        return
    M = cn.level_matrix("top10")
    cs = cn.cross_section_percentile()
    comp = comp.copy()
    comp["横截面分位"] = cs.reindex(comp.index)
    q, qp, qy = comp.attrs["q"], comp.attrs["q_prev"], comp.attrs["q_year"]
    qf, qf0 = comp.attrs["q_full"], comp.attrs["q_full0"]
    # 指数分位：行业指数点位在 2019 年以来的排名，回答「价格贵不贵」。
    # 它与「配置分位」（机构押得多重）是两个维度，必须分开看。
    idx = cn.index_stats()
    if not idx.empty:
        comp = comp.join(idx[["指数分位", "近3月涨幅%", "近1年涨幅%"]], how="left")

    # ---------------- 全市场概览 ----------------
    h(f'<div class="sec">行业地图 · {q}（{len(M)} 个报告期 · {len(comp)} 个申万一级行业）</div>')
    _c1, _c2, _c3, _c4 = st.columns(4)
    _top = comp["历史分位"].idxmax()
    _bot = comp["历史分位"].idxmin()
    _hhi = float(((M.iloc[-1] / 100) ** 2).sum())
    _hhi0 = float(((M.iloc[-2] / 100) ** 2).sum()) if len(M) > 1 else np.nan
    _c1.metric("最拥挤", f"{_top}", f"{comp.loc[_top, '历史分位']:.0f} 分位",
               delta_color="off", help="当前配置在自身历史中的排名最高")
    _c2.metric("最冷门", f"{_bot}", f"{comp.loc[_bot, '历史分位']:.0f} 分位",
               delta_color="off")
    _c3.metric("行业集中度 HHI", f"{_hhi:.4f}",
               f"{_hhi - _hhi0:+.4f}" if np.isfinite(_hhi0) else None, delta_color="off",
               help="Σ(行业占比)²，按前十大归一化口径算；与半年报全持仓口径"
                    "（更分散、HHI 更低）不可直接比较，只看趋势")
    _c4.metric("前三大行业合计", f"{M.iloc[-1].nlargest(3).sum():.1f}%",
               f"{M.iloc[-1].nlargest(3).sum() - M.iloc[-2].nlargest(3).sum():+.1f}pp"
               if len(M) > 1 else None, delta_color="off")

    st.write("")

    # ---------------- 左列表 + 右详情 ----------------
    L, R = st.columns([0.26, 0.74], gap="small")
    with L:
        # 选中行业的摘要卡放在列表上方：31 项 radio 很长，
        # 卡片若放列表下面会被顶到屏幕外，等于没有。
        _sel0 = st.session_state.get("sel_ind2")
        _rank0 = comp.sort_values("历史分位", ascending=False)
        _sel0 = _sel0 if _sel0 in _rank0.index else _rank0.index[0]
        _col0, _lv0 = _tone(float(comp.loc[_sel0, "历史分位"]))
        h(f'<div style="background:{C.COLOR["panel"]};border-left:3px solid {_col0};'
          f'border-radius:6px;padding:8px 10px;margin-bottom:6px">'
          f'<div style="font-size:14px;font-weight:700">{_sel0}</div>'
          f'<div style="margin:3px 0;color:{_col0};font-size:12px">● {_lv0}'
          f'（{float(comp.loc[_sel0, "历史分位"]):.0f} 分位）</div>'
          f'<div class="tiny">{len(M)} 期均值 {M[_sel0].mean():.2f}%　'
          f'区间 {M[_sel0].min():.2f}–{M[_sel0].max():.2f}%</div></div>')
        h('<div class="sec">31 个行业（按历史分位排序）</div>')
        _rank = comp.sort_values("历史分位", ascending=False)
        _opts = list(_rank.index)
        # radio 的 label 是纯文本，HTML 会被原样打印出来（试过，页面上直接显示成
        # <span style="...">●</span>）。所以这里只用文字：名次 + 行业 + 分位，
        # 颜色信息放到右侧详情卡（那里是真 HTML）。
        _labels = [f"{k:>2}. {i}　{float(_rank.loc[i, '历史分位']):.0f}"
                   for k, i in enumerate(_opts, 1)]
        sel = st.radio("行业", _opts,
                       format_func=lambda x: _labels[_opts.index(x)],
                       label_visibility="collapsed", key="sel_ind2")
        h(f'<div class="tiny" style="margin-top:8px">'
          f'数字＝该行业当前配置在 <b>{len(M)} 期</b>历史中的分位：'
          f'80+ 历史高位、60–80 偏高、40–60 中性、20–40 偏低、20 以下历史低位。'
          f'名次按分位从高到低。</div>')

    r = comp.loc[sel]
    with R:
        _d1, _d2, _d3, _d4, _d5, _d6 = st.columns(6)
        _d1.metric(f"前十配置 {q}", f"{r['配置_前十']:.2f}%",
                   help="前十大归一化口径：组合里最重要的 10 只股票，该行业占多少")
        _d2.metric("配置分位", f"{r['历史分位']:.0f}",
                   help=f"在自身 {len(M)} 期历史中的排名")
        _d3.metric("指数分位",
                   f"{r['指数分位']:.0f}" if np.isfinite(r.get("指数分位", np.nan)) else "—",
                   help="行业指数点位在 2019 年以来每个交易日中的排名，回答「贵不贵」")
        _d4.metric(f"环比 {qp}", f"{r['Δ上季pp']:+.2f}pp", delta_color="off")
        _d5.metric(f"同比 {qy}", f"{r['Δ上年pp']:+.2f}pp", delta_color="off")
        _d6.metric(f"全持仓 {qf}",
                   f"{r['配置_全持仓']:.2f}%" if np.isfinite(r.get("配置_全持仓", np.nan))
                   else "—",
                   help="全持仓口径（半年报），与前十大口径不可直接比较")

        st.plotly_chart(rc.fig_level_history(M, sel, f"（{len(M)} 期，前十大口径）"),
                        width="stretch", config={"displayModeBar": False})

        _act = r.get("主动pp", np.nan)
        h(f'<div class="cg-note">{sel} 本期配置 <b>{r["配置_前十"]:.2f}%</b>（前十大口径），'
          f'比上一季 <b>{r["Δ上季pp"]:+.2f}pp</b>、比去年同季 '
          f'<b>{r["Δ上年pp"]:+.2f}pp</b>；放在 {len(M)} 期历史里排第 '
          f'<b>{r["历史分位"]:.0f}</b> 分位。'
          + (f' 全持仓口径（{qf0}→{qf}）配置 {r["配置_全持仓"]:.2f}%，'
             f'其中机构主动 {_act:+.2f}pp、价格 {r["被动pp"]:+.2f}pp。'
             if np.isfinite(_act) else '')
          + f'<br><b>怎么用</b>：{_verdict(r)}</div>')

        _hist = pd.DataFrame({"配置%": M[sel]}).dropna().tail(8)
        _hist["环比pp"] = _hist["配置%"].diff().round(2)
        _hist = _hist.iloc[::-1].reset_index().rename(columns={"index": "报告期"})
        st.dataframe(_hist, width="stretch", hide_index=True, height=230,
                     column_config={
                         "配置%": st.column_config.NumberColumn(format="%.2f"),
                         "环比pp": st.column_config.NumberColumn(format="%+.2f")})

    # ---------------- 分位双轴 ----------------
    h('<div class="sec">② 两把尺子：相对自己历史 vs 相对全市场同行</div>')
    st.plotly_chart(rc.fig_dual_percentile(comp, comp["横截面分位"]),
                    width="stretch", config={"displayModeBar": False})
    _both_high = comp[(comp["历史分位"] >= 80) & (comp["横截面分位"] >= 80)]
    _self_only = comp[(comp["历史分位"] >= 80) & (comp["横截面分位"] < 50)]
    _mt_high = comp[(comp["历史分位"] < 50) & (comp["横截面分位"] >= 70)]
    h('<div class="cg-note">'
      f'<b>怎么读</b>：横条＝这个行业的配置在<b>它自己 31 期历史</b>里排第几；'
      f'菱形＝这个行业的配置在<b>当期 31 个行业之间</b>排第几。'
      f'两根尺子必须一起看。<br>'
      f'<b>① 又重又在历史极值（两者都 ≥80）</b>：'
      + ("、".join(_both_high.index) or "无")
      + '——这才是真拥挤，减仓优先级最高。<br>'
      f'<b>② 只是「小行业里的相对极端」（自身高、横截面 <50）</b>：'
      + ("、".join(f"{i}（自身 {r['历史分位']:.0f}、横截面 {r['横截面分位']:.0f}）"
                  for i, r in _self_only.iterrows()) or "无")
      + '——它自己历史上没这么重过，但绝对配置很小，对组合影响有限，不用因为「分位高」就卖。<br>'
      f'<b>③ 绝对配置不低、但相对自己历史很轻（自身 <50、横截面 ≥70）</b>：'
      + ("、".join(f"{i}（自身 {r['历史分位']:.0f}、横截面 {r['横截面分位']:.0f}）"
                  for i, r in _mt_high.iterrows()) or "无")
      + '——被冷落了很久，但仍是组合里的大块头，这类最容易出现「预期差」。<br>'
      f'<b>结论</b>：只看自身分位会高估小行业的拥挤、低估大行业的风险；'
      f'只看横截面分位会把「一直很重」当成「现在才拥挤」。两个一起看，'
      f'才知道该减谁、该等谁。</div>')

    # ---------------- 价格维度：配置分位 × 指数分位 ----------------
    if not idx.empty:
        h('<div class="sec">③ 第三把尺子：价格贵不贵——配置分位 × 指数分位</div>')
        st.plotly_chart(rc.fig_crowding_map(comp, idx), width="stretch",
                        config={"displayModeBar": False})
        # comp 在上面已经 join 过指数列，这里不能重复 join（会报 columns overlap）
        _d = comp.dropna(subset=["历史分位", "指数分位"])
        _crowd = _d[(_d["历史分位"] >= 50) & (_d["指数分位"] >= 50)]
        _hold = _d[(_d["历史分位"] >= 50) & (_d["指数分位"] < 50)]
        _miss = _d[(_d["历史分位"] < 50) & (_d["指数分位"] >= 50)]
        _cold = _d[(_d["历史分位"] < 50) & (_d["指数分位"] < 50)]
        h(f'<div class="cg-note"><b>怎么读</b>：横轴＝机构押得多重（配置分位，31 期）；'
          f'纵轴＝价格相对自己多贵（指数分位，{idx.attrs.get("start")} 起 '
          f'{idx.attrs.get("n_days"):,} 个交易日）。气泡大小＝前十配置比例。<br>'
          f'<b>右上·真拥挤（{len(_crowd)} 个）</b>：'
          + ("、".join(f"{i}（配置 {r['历史分位']:.0f} / 指数 {r['指数分位']:.0f}）"
                      for i, r in _crowd.nlargest(4, "指数分位").iterrows()) or "无")
          + '——机构重仓且价格已在历史高位，买它等于在拥挤处接棒。<br>'
          f'<b>右下·抱团没赚（{len(_hold)} 个）</b>：'
          + ("、".join(f"{i}（配置 {r['历史分位']:.0f} / 指数 {r['指数分位']:.0f}）"
                      for i, r in _hold.head(3).iterrows()) or "无")
          + '——仓位重但价格还在低位，要么是机会，要么是机构判断错了，'
            '要看下一季主动项有没有继续加。<br>'
          f'<b>左上·机构踏空（{len(_miss)} 个）</b>：'
          + ("、".join(f"{i}（配置 {r['历史分位']:.0f} / 指数 {r['指数分位']:.0f}）"
                      for i, r in _miss.head(3).iterrows()) or "无")
          + '——价格涨了但仓位没跟上，机构还没上车，这类要留意补涨或补跌。<br>'
          f'<b>左下·冷门（{len(_cold)} 个）</b>：仓位轻、价格低，没有资金关注。<br>'
          f'<b>结论</b>：只看配置分位，会把「抱团没涨」误判成拥挤；'
          f'加上价格分位，才知道右上角那 {len(_crowd)} 个才是真要减的，'
          f'右下角那些是「闷在里面」的，不用急着卖。</div>')

    # ---------------- 全行业一览 ----------------
    h('<div class="sec">④ 全行业一览：把 31 个行业放在一张表里</div>')
    _show = comp.reset_index()[[
        "行业", "配置_前十", "Δ上季pp", "Δ上年pp", "配置_全持仓", "主动pp",
        "历史分位", "横截面分位", "指数分位", "近3月涨幅%", "加仓占比", "判读"]].copy()
    _show["拥挤等级"] = [_tone(float(v))[1] if np.isfinite(v) else "—"
                        for v in _show["历史分位"]]
    _show = _show.rename(columns={
        "配置_前十": f"前十配置%({q})", "Δ上季pp": f"环比pp({qp}→{q})",
        "Δ上年pp": f"同比pp({qy}→{q})", "配置_全持仓": f"全持仓%({qf})",
        "主动pp": f"主动pp({qf0}→{qf})", "历史分位": "自身分位(31期)",
        "横截面分位": "横截面分位",
        "指数分位": "指数分位(2019起)", "近3月涨幅%": "指数近3月%",
        "加仓占比": f"加仓占比%({q})"})
    st.dataframe(
        _show.sort_values("自身分位(31期)", ascending=False),
        width="stretch", hide_index=True, height=520,
        column_config={
            f"前十配置%({q})": st.column_config.NumberColumn(format="%.2f"),
            f"环比pp({qp}→{q})": st.column_config.NumberColumn(format="%+.2f"),
            f"同比pp({qy}→{q})": st.column_config.NumberColumn(format="%+.2f"),
            f"全持仓%({qf})": st.column_config.NumberColumn(format="%.2f"),
            f"主动pp({qf0}→{qf})": st.column_config.NumberColumn(format="%+.2f"),
            "自身分位(31期)": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100,
                help="该行业配置在它自己 31 期历史中的排名"),
            "指数分位(2019起)": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100,
                help="行业指数点位在 2019 年以来每个交易日中的排名，回答「贵不贵」"),
            "指数近3月%": st.column_config.NumberColumn(format="%+.1f"),
            "横截面分位": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100,
                help="当期 31 个行业之间的排名"),
            f"加仓占比%({q})": st.column_config.NumberColumn(
                format="%.0f", help="分母是「该行业进入前十大」的基金数"),
            "判读": st.column_config.TextColumn(width="small"),
            "拥挤等级": st.column_config.TextColumn(width="small"),
        })

    # ---------------- 结构热力图 ----------------
    h(f'<div class="sec">把 {len(M)} 期铺开：每个行业的配置是怎么一步步走到今天的</div>')
    st.plotly_chart(rc.fig_structure_heatmap(M, top=20), width="stretch",
                    config={"displayModeBar": False})
    _chg = (M.iloc[-1] - M.iloc[0]).sort_values(ascending=False)
    h(f'<div class="cg-note">行是行业、列是报告期，颜色越深＝在前十大组合里占比越高。'
      f'从 {M.index[0]} 到 {M.index[-1]}，抬升最多 <b>{_chg.index[0]}'
      f'（{_chg.iloc[0]:+.1f}pp）</b>、让位最多 <b>{_chg.index[-1]}'
      f'（{_chg.iloc[-1]:+.1f}pp）</b>。'
      f'<br><b>所以呢</b>：单季 1–2pp 的波动多是噪音，'
      f'连续 3 个季度往同一方向走才是结构性变化。</div>')
    h(f'<div class="tiny">口径：全部 {len(M)} 期都用「每只基金取前十大 → 归一化 → 等权平均」，'
      f'这样季报（前十大）与半年报（全持仓）才可比；'
      f'全持仓口径的水平在表格里单列，不与本图混算。'
      f'样本 {len(cn.universe(q)):,} 只主动权益基金，已剔除主题基金与覆盖率异常记录。'
      f'<br><b>这条序列含价格漂移</b>：行业涨了，即使没人买，它在组合里的占比也会上升——'
      f'所以「分位高」只能说明<b>它在组合里很重</b>，不能直接说成「机构在买」。'
      f'要剥离价格，只有半年报的全持仓 GK 分解（主动 pp / 被动 pp），见「① 本期解读」。</div>')
