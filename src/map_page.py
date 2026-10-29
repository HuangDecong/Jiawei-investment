# -*- coding: utf-8 -*-
"""④ 行业拥挤度：可搜索、每个行业都有颜色标注、每个数字都能查到算法。

页面结构
--------
    本页结论（先给判断）
    ① 搜索 + 行业总览（31 个色块，红＝拥挤）
    ② 单行业详情（6 个指标 + 历史走势 + 近 8 期台账）
    ③ 31 个行业一根条（颜色＝等级）
    ④ 双分位：相对自己 vs 相对同行
    ⑤ 四象限：配置 × 价格
    ⑥ 全行业表

口径
----
· 配置比例用**前十大归一化**（31 期都有，才能做环比/同比/分位）；
· 「全持仓」只在半年报期存在，单列并标期次，不与前十大混算；
· 分位＝当前值在**自身 31 期**中的排名（0＝历史最低，100＝历史最高）；
· 拥挤度 CSI＝(配置分位＋价格分位＋资金分位)/3，等权、不拟合。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import canon as cn
import config as C
import report_charts as rc


def _tone(p: float) -> tuple[str, str]:
    """配置分位 → (颜色, 等级名)。"""
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
    """按「分位 + 环比 + 同比 + 资金」给一句可执行判断（规则事前固定）。"""
    p, d1, d2 = r["历史分位"], r["Δ上季pp"], r["Δ上年pp"]
    a, br = r.get("平均主动pp", np.nan), r.get("加仓占比", np.nan)
    if p >= 80 and np.isfinite(a) and a < 0:
        return ("分位在历史高位，但本期资金在撤（价格中性后净额为负）——"
                "这是「拥挤度见顶」的形态：仓位还没降下来，人已经开始走。"
                "持有的减半，不新建仓。")
    if p >= 80 and d1 > 0 and d2 > 0:
        return ("分位已到历史高位，且环比、同比同向上行——拥挤度还在累积。"
                "这时买入等于在别人的浮盈上加仓，仓位要更轻、止损要更紧。")
    if p <= 20 and d1 < 0 and d2 < 0:
        return ("分位在历史低位，且环比、同比同向下行——机构在持续撤。"
                "不要因为「便宜」就买，先等主动项转正。")
    if p <= 20 and np.isfinite(a) and a > 0.2 and np.isfinite(br) and br >= 60:
        return ("分位在历史低位，但资金已经开始回流、广度也够——"
                "这是「冷门转热」的早期形态，可以小仓位试，等下一季确认。")
    if np.sign(d1) != np.sign(d2):
        return ("环比与同比方向相反：这是季度内的摆动，不是趋势，"
                "别拿它做年度配置依据，只适合做交易。")
    if p >= 60:
        return "分位偏高但变动温和：拥挤度不极端，按个股逻辑处理，不必因行业减仓。"
    return "分位偏低且变动温和：没有明显的拥挤或抛弃信号，按个股逻辑处理即可。"


def _chips(cs: pd.DataFrame, comp: pd.DataFrame) -> str:
    """31 个行业的色块总览：颜色＝拥挤等级，点一下上方的下拉即可看详情。"""
    out = []
    for i, r in cs.iterrows():
        c = rc.CSI_COLORS.get(r["等级"], "#C3CBD4")
        _aq = comp.loc[i, "Δ上季pp"] if i in comp.index else np.nan
        out.append(
            f'<div style="flex:1 1 88px;min-width:82px;background:{c};color:#fff;'
            f'border-radius:6px;padding:6px 7px;line-height:1.35">'
            f'<div style="font-size:11px;font-weight:700">{i}'
            f'{"†" if r.get("可靠") == "小样本" else ""}</div>'
            f'<div style="font-size:15px;font-weight:700">{r["CSI"]:.0f}</div>'
            f'<div style="font-size:9px;opacity:.92">{r["等级"]}　'
            f'{_aq:+.2f}pp</div></div>')
    return ('<div style="display:flex;gap:5px;flex-wrap:wrap;margin:4px 0 6px">'
            + "".join(out) + '</div>')


def render(h) -> None:
    if not (cn.available() or cn.fallback_available()):
        h('<div class="cg-note">未找到持仓数据。</div>')
        return
    comp = cn.compare_periods()
    if comp.empty:
        h('<div class="cg-note">数据不足。</div>')
        return
    # join 会丢掉 DataFrame 的 attrs（pandas 不保证跨操作保留），
    # 所以先把期次信息存下来，后面一律用这个副本。
    A = dict(comp.attrs)
    M = cn.level_matrix("top10")
    cs = cn.cross_section_percentile()
    csi = cn.crowding_score()
    comp = comp.copy()
    comp["横截面分位"] = cs.reindex(comp.index)
    if not csi.empty:
        comp = comp.join(csi[["CSI", "等级", "配置分位", "价格分位", "资金分位"]], how="left")
    q, qp, qy = A["q"], A["q_prev"], A["q_year"]
    qf, qf0 = A["q_full"], A["q_full0"]
    idx = cn.index_stats()
    if not idx.empty:
        comp = comp.join(idx[["指数分位", "近3月涨幅%", "近1年涨幅%"]], how="left")

    # =====================================================================
    # 本页结论
    # =====================================================================
    _top = comp.nlargest(3, "CSI") if "CSI" in comp else pd.DataFrame()
    _bot = comp.nsmallest(3, "CSI") if "CSI" in comp else pd.DataFrame()
    _hi_chg = comp[(comp["历史分位"] >= 80) & (comp["平均主动pp"] < 0)] if "平均主动pp" in comp else pd.DataFrame()
    _lo_chg = comp[(comp["历史分位"] <= 30) & (comp["平均主动pp"] > 0.2) &
                   (comp["加仓占比"] >= 60)] if "平均主动pp" in comp else pd.DataFrame()
    _n_ext = int((comp["等级"] == "极度拥挤").sum()) if "等级" in comp else 0
    _n_cold = int((comp["等级"] == "冷门").sum()) if "等级" in comp else 0

    h('<div class="sec">本页结论：谁最挤、谁在变挤、谁在变冷</div>')
    _bl = []
    if len(_top):
        _bl.append(f'<b>最拥挤</b>：' + "、".join(
            f'{i}（CSI {r["CSI"]:.0f}，配置{r["历史分位"]:.0f}/价格'
            f'{r.get("指数分位", float("nan")):.0f}）' for i, r in _top.iterrows())
            + f'。31 个行业里极度拥挤 <b>{_n_ext}</b> 个、冷门 <b>{_n_cold}</b> 个。')
    if len(_hi_chg):
        _bl.append(f'<b>拥挤度见顶</b>（配置还在高位、资金已转负）：'
                   + "、".join(f'{i}（配置{r["历史分位"]:.0f}、当期{r["平均主动pp"]:+.2f}pp）'
                              for i, r in _hi_chg.iterrows())
                   + '。这类是<b>减仓优先级最高</b>的，不是买入候选。')
    if len(_lo_chg):
        _bl.append(f'<b>冷门转热</b>（配置在低位、资金开始进、多数在加）：'
                   + "、".join(f'{i}（配置{r["历史分位"]:.0f}、当期{r["平均主动pp"]:+.2f}pp、'
                              f'{r["加仓占比"]:.0f}%在加）' for i, r in _lo_chg.iterrows())
                   + '。这类是<b>预期差</b>的来源，可以小仓位试。')
    _bl.append('<b>怎么用</b>：拥挤度只决定「给多大仓位」，不决定「买不买」。'
               '本库数据已证伪「高拥挤必然下跌」，所以看到红色不要条件反射地卖，'
               '而是把它的仓位上限压下来；看到深蓝也不要马上买，先等资金转向。')
    h('<div class="cg-note" style="line-height:1.9">'
      + "<br>".join(f"{i}. {x}" for i, x in enumerate(_bl, 1)) + "</div>")

    # =====================================================================
    # ① 搜索 + 总览
    # =====================================================================
    h('<div class="sec">① 找行业：输入名称即可搜索，颜色＝拥挤等级</div>')
    c1, c2 = st.columns([0.42, 0.58], gap="small")
    with c1:
        _kw = st.text_input("搜索行业（支持部分匹配，如「电」）", value="", key="ind_kw",
                            placeholder="输入行业名，如 电子 / 医药 / 银行")
        _cand = [i for i in comp.index if _kw.strip() in i] if _kw.strip() else list(comp.index)
        if not _cand:
            _cand = list(comp.index)
            st.caption("没有匹配的行业，已显示全部。")
        _rank = comp.loc[_cand].sort_values("CSI", ascending=False) if "CSI" in comp else comp.loc[_cand]
        _opts = list(_rank.index)
        _labels = [f'{k}. {i}　CSI {comp.loc[i, "CSI"]:.0f}　{comp.loc[i, "等级"]}'
                   + ("　†小样本" if comp.loc[i].get("可靠") == "小样本" else "")
                   for k, i in enumerate(_opts, 1)]
        sel = st.selectbox("选择行业（可再输入搜索）", _opts,
                           format_func=lambda x: _labels[_opts.index(x)],
                           key="sel_ind3")
        _lv_c = rc.CSI_COLORS.get(comp.loc[sel, "等级"], "#C3CBD4")
        h(f'<div style="background:{_lv_c};color:#fff;border-radius:7px;'
          f'padding:8px 11px;margin-top:2px">'
          f'<div style="font-size:15px;font-weight:700">{sel}</div>'
          f'<div style="font-size:11px;opacity:.95">CSI {comp.loc[sel, "CSI"]:.0f}　'
          f'{comp.loc[sel, "等级"]}　配置分位 {comp.loc[sel, "历史分位"]:.0f}　'
          f'环比 {comp.loc[sel, "Δ上季pp"]:+.2f}pp</div></div>')
    with c2:
        h('<div class="tiny" style="margin-bottom:2px">'
          '31 个行业总览：<b>红＝极度拥挤、橙＝拥挤、灰＝中性、浅蓝＝偏冷、深蓝＝冷门</b>。'
          '数字是 CSI（0–100），下方小字是本期环比变化。要点开某个行业，'
          '用左侧下拉框搜索。<b>†</b>＝该行业全市场只有不到 150 只基金持有'
          '（前十配置 &lt;0.5%），分位噪音大，等级仅供参考。</div>')
        h(_chips(comp.dropna(subset=["CSI"]).sort_values("CSI", ascending=False), comp))

    # =====================================================================
    # ② 单行业详情
    # =====================================================================
    r = comp.loc[sel]
    h(f'<div class="sec">② {sel}：六个指标 + 历史位置</div>')
    _d = st.columns(6)
    _d[0].metric(f"前十配置 {q}", f"{r['配置_前十']:.2f}%",
                 help="前十大归一化口径：组合里最重要的 10 只股票，该行业占多少")
    _d[1].metric("配置分位", f"{r['历史分位']:.0f}",
                 help=f"在自身 {len(M)} 期历史中的排名（0＝最低，100＝最高）")
    _d[2].metric("价格分位",
                 f"{r['指数分位']:.0f}" if np.isfinite(r.get("指数分位", np.nan)) else "—",
                 help="行业指数点位在 2019 年以来每个交易日中的排名，回答「贵不贵」")
    _d[3].metric("CSI 拥挤度", f"{r.get('CSI', float('nan')):.0f}",
                 help="(配置分位+价格分位+资金分位)/3，等权；≥80 极度拥挤")
    _d[4].metric(f"环比 {qp}", f"{r['Δ上季pp']:+.2f}pp", delta_color="off",
                 help="含价格漂移；价格中性后的净额见下方")
    _d[5].metric(f"同比 {qy}", f"{r['Δ上年pp']:+.2f}pp", delta_color="off")

    st.plotly_chart(rc.fig_level_history(M, sel, f"（{len(M)} 期，前十大口径）"),
                    width="stretch", config={"displayModeBar": False})

    _act = r.get("平均主动pp", np.nan)
    h(f'<div class="cg-note">{sel} 本期配置 <b>{r["配置_前十"]:.2f}%</b>（前十大口径），'
      f'比上一季 <b>{r["Δ上季pp"]:+.2f}pp</b>、比去年同季 '
      f'<b>{r["Δ上年pp"]:+.2f}pp</b>；放在 {len(M)} 期历史里排第 '
      f'<b>{r["历史分位"]:.0f}</b> 分位。'
      + (f' 价格中性后，本期净买卖 <b>{_act:+.2f}pp</b>，'
         f'{r["加仓占比"]:.0f}% 的变动基金在加（分母是「有实质变动的基金」，'
         f'不是全部样本）。' if np.isfinite(_act) else '')
      + (f' 全持仓口径（{qf0}→{qf}）配置 {r["配置_全持仓"]:.2f}%，'
         f'其中机构主动 {r["主动pp"]:+.2f}pp、价格 {r["被动pp"]:+.2f}pp。'
         if np.isfinite(r.get("主动pp", np.nan)) else '')
      + f'<br><b>怎么用</b>：{_verdict(r)}</div>')

    _hist = pd.DataFrame({"配置%": M[sel]}).dropna().tail(8)
    _hist["环比pp"] = _hist["配置%"].diff().round(2)
    _hist = _hist.iloc[::-1].reset_index().rename(columns={"index": "报告期"})
    st.dataframe(_hist, width="stretch", hide_index=True, height=230,
                 column_config={
                     "配置%": st.column_config.NumberColumn(format="%.2f"),
                     "环比pp": st.column_config.NumberColumn(format="%+.2f")})

    # =====================================================================
    # ③ 31 个行业一根条
    # =====================================================================
    h('<div class="sec">③ 31 个行业排一排：长度＝拥挤度，颜色＝等级</div>')
    st.plotly_chart(rc.fig_csi_bar(comp.dropna(subset=["CSI"])), width="stretch",
                    config={"displayModeBar": False})
    _c = comp.dropna(subset=["CSI"])
    _hi = _c[_c["CSI"] >= 80]
    _lo = _c[_c["CSI"] < 25]
    h(f'<div class="cg-note"><b>怎么读</b>：条越长越挤，颜色跟着等级走。'
      f'虚线是分级线（25/45/65/80）。<br>'
      f'<b>≥80 极度拥挤（{len(_hi)} 个）</b>：'
      + ("、".join(f'{i}（{r["CSI"]:.0f}）' for i, r in _hi.iterrows()) or "无")
      + f'——仓位上限 10%，不新增。<br>'
      f'<b>&lt;25 冷门（{len(_lo)} 个）</b>：'
      + ("、".join(f'{i}（{r["CSI"]:.0f}）' for i, r in _lo.iterrows()) or "无")
      + '——没有资金关注，等转向信号（广度≥60 且净额转正）再买。<br>'
      f'<b>中间段</b>不是「没意见」，而是「不极端」：按个股逻辑处理，'
      f'不必因为行业分位高低调整仓位。</div>')

    # =====================================================================
    # ④ 双分位
    # =====================================================================
    # ---- 拥挤度变化：谁在变挤、谁在变松 ----
    h('<div style="font-size:12px;font-weight:700;margin:12px 0 3px">'
      '拥挤度变化：本期 CSI − 上期 CSI（红＝变挤，绿＝变松）</div>')
    cc = cn.csi_change(q)
    if not cc.empty:
        st.plotly_chart(rc.fig_csi_change(cc), width="stretch",
                        config={"displayModeBar": False})
        _up = cc.nlargest(3, "ΔCSI")
        _dn = cc.nsmallest(3, "ΔCSI")
        _hi_up = cc[(cc["CSI"] >= 80) & (cc["ΔCSI"] > 0)]
        h(f'<div class="cg-note">'
          f'<b>怎么读</b>：横轴是「本期 CSI − 上期 CSI」，不是水位而是水流。'
          f'右边越长＝这段时间挤得越快，左边越长＝在快速变松。悬停可看本期与上期的具体数值。<br>'
          f'<b>变挤最多（{cc.attrs.get("q_prev", "")}→{cc.attrs.get("q", "")}）</b>：'
          + "、".join(f'{i}（{r["上期CSI"]:.0f}→{r["CSI"]:.0f}，{r["ΔCSI"]:+.0f}）'
                      for i, r in _up.iterrows())
          + f'；<b>变松最多</b>：'
          + "、".join(f'{i}（{r["上期CSI"]:.0f}→{r["CSI"]:.0f}，{r["ΔCSI"]:+.0f}）'
                      for i, r in _dn.iterrows())
          + f'。<br><b>要特别盯的是「水位已高且还在涨」的 {len(_hi_up)} 个</b>：'
          + ("、".join(_hi_up.index) if len(_hi_up) else "无")
          + f'——它们才是真正在往拥挤区走的；已经变松的高水位行业（如电子、通信）'
            f'反而不用急着减，因为资金正在自己退潮。</div>')
    else:
        h('<div class="tiny">上期数据不足，无法计算变化。</div>')

    # =====================================================================
    # ④ 为什么用 CSI，不用 HHI
    # =====================================================================
    h('<div class="sec">④ 为什么拥挤度用 CSI，不用 HHI？</div>')
    _hhi_now = float(((comp["配置_前十"] / 100) ** 2).sum())
    _hhi_M = cn.level_matrix("top10")
    _hhi_hist = ((_hhi_M / 100) ** 2).sum(axis=1)
    _hhi_pct = float((_hhi_hist <= _hhi_now).mean() * 100) if len(_hhi_hist) else np.nan
    h('<div style="display:flex;gap:7px;flex-wrap:wrap">'
      '<div style="flex:1 1 320px;background:#FFFFFF;border:1px solid ' + C.COLOR["line"] +
      ';border-top:3px solid ' + C.COLOR["text_dim"] + ';border-radius:7px;padding:9px 11px">'
      '<div style="font-size:12px;font-weight:700">HHI（赫芬达尔指数）</div>'
      '<div style="font-size:11px;line-height:1.85;margin-top:4px">'
      '<b>公式</b>：HHI = Σ(w<sub>k</sub>/100)²，w 是 31 个行业的配置比例<br>'
      '<b>它是什么</b>：<b>一个组合</b>的集中度——把各行业权重平方后加总，'
      '权重越集中，HHI 越大。<br>'
      '<b>它回答</b>：「这个组合分散不分散？押注少数行业还是撒得很开？」<br>'
      '<b>它不回答</b>：「<b>哪个</b>行业挤？」——HHI 是一个标量，'
      '31 个行业最后只剩一个数，无法排序、无法定位。<br>'
      '<b>当前值</b>：全市场前十大口径 HHI <b>' + f'{_hhi_now:.4f}' + '</b>，'
      '处在 31 期的第 <b>' + f'{_hhi_pct:.0f}' + '</b> 分位（越高质量越集中）。'
      '</div></div>'
      '<div style="flex:1 1 320px;background:#FFFFFF;border:1px solid ' + C.COLOR["line"] +
      ';border-top:3px solid ' + C.COLOR["threshold"] + ';border-radius:7px;padding:9px 11px">'
      '<div style="font-size:12px;font-weight:700">CSI（拥挤度指数）</div>'
      '<div style="font-size:11px;line-height:1.85;margin-top:4px">'
      '<b>公式</b>：CSI = (配置分位 + 价格分位 + 资金分位) ÷ 3<br>'
      '<b>它是什么</b>：<b>单个行业</b>在市场里的拥挤程度，三个维度各自换算成 '
      '0–100 的分位后等权平均。<br>'
      '<b>它回答</b>：「这个行业被多少人在什么价位上挤着买？现在是不是人人都上车了？」<br>'
      '<b>它不回答</b>：「整个组合分散不分散」——那是 HHI 的事。<br>'
      '<b>为什么等权</b>：三维已统一为分位，量纲一致；31 个行业 × 31 期不足以'
      '稳定估计权重，任何加权都是过拟合。'
      '</div></div></div>')
    h('<div class="cg-note" style="margin-top:6px"><b>一句话区别</b>：'
      'HHI 是「<b>组合</b>的体温计」，只能告诉你发不发烧；'
      'CSI 是「<b>行业</b>的拥堵指数」，能告诉你哪条路堵、堵到什么程度。'
      '本页要回答的是「31 个行业里谁最挤」，所以主指标必须是 CSI；'
      'HHI 作为辅助，放在页头告诉你「整体是不是越来越集中」。'
      '<br><b>为什么不用换手率、成交额占比这类经典拥挤度指标</b>：'
      '① 本库的季报数据不含交易明细，成交额只能外接行情，会引入第二套口径；'
      '② 成交额占比在 A 股受涨跌停、次新股、指数调整影响极大，'
      '与「基金持仓拥挤」不是同一件事。本页衡量的是<b>持仓拥挤</b>，'
      '所以三个维度全部来自同一套持仓与指数数据，保证可复算。</div>')


    h('<div class="sec">⑤ 两把尺子：相对自己历史 vs 相对全市场同行</div>')
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
      f'<b>② 只是「小行业里的相对极端」（自身高、横截面 &lt;50）</b>：'
      + ("、".join(f"{i}（自身 {r['历史分位']:.0f}、横截面 {r['横截面分位']:.0f}）"
                  for i, r in _self_only.iterrows()) or "无")
      + '——它自己历史上没这么重过，但绝对配置很小，对组合影响有限。<br>'
      f'<b>③ 绝对配置不低、但相对自己历史很轻（自身 &lt;50、横截面 ≥70）</b>：'
      + ("、".join(f"{i}（自身 {r['历史分位']:.0f}、横截面 {r['横截面分位']:.0f}）"
                  for i, r in _mt_high.iterrows()) or "无")
      + '——被冷落了很久，但仍是组合里的大块头，这类最容易出现「预期差」。<br>'
      f'<b>结论</b>：只看自身分位会高估小行业的拥挤、低估大行业的风险；'
      f'只看横截面会把「一直很重」当成「现在才拥挤」。</div>')

    # =====================================================================
    # ⑤ 四象限
    # =====================================================================
    if not idx.empty:
        h('<div class="sec">⑥ 第三把尺子：价格贵不贵——配置分位 × 指数分位</div>')
        st.plotly_chart(rc.fig_crowding_map(comp, idx), width="stretch",
                        config={"displayModeBar": False})
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
          + '——仓位重但价格还在低位，要么是机会，要么是机构判断错了。<br>'
          f'<b>左上·机构踏空（{len(_miss)} 个）</b>：'
          + ("、".join(f"{i}（配置 {r['历史分位']:.0f} / 指数 {r['指数分位']:.0f}）"
                      for i, r in _miss.head(3).iterrows()) or "无")
          + '——价格涨了但仓位没跟上，留意补涨或补跌。<br>'
          f'<b>左下·冷门（{len(_cold)} 个）</b>：仓位轻、价格低，没有资金关注。<br>'
          f'<b>结论</b>：只看配置分位，会把「抱团没涨」误判成拥挤；'
          f'加上价格分位，才知道右上角那 {len(_crowd)} 个才是真要减的。</div>')

    # =====================================================================
    # ⑥ 全行业表
    # =====================================================================
    h('<div class="sec">⑦ 全行业一览：把 31 个行业放在一张表里</div>')
    _show = comp.reset_index()[[
        "行业", "配置_前十", "Δ上季pp", "Δ上年pp", "平均主动pp", "加仓占比",
        "配置_全持仓", "主动pp", "历史分位", "横截面分位", "指数分位",
        "近3月涨幅%", "CSI", "等级", "判读"]].copy()
    _show = _show.rename(columns={
        "配置_前十": f"前十配置%({q})", "Δ上季pp": f"环比pp({qp}→{q})",
        "Δ上年pp": f"同比pp({qy}→{q})", "平均主动pp": f"当期主动pp({qp}→{q})",
        "加仓占比": f"加仓占比%({q})", "配置_全持仓": f"全持仓%({qf})",
        "主动pp": f"半年主动pp({qf0}→{qf})", "历史分位": "配置分位(31期)",
        "横截面分位": "横截面分位", "指数分位": "价格分位(2019起)",
        "近3月涨幅%": "指数近3月%", "CSI": "CSI", "等级": "拥挤等级"})
    # 给「拥挤等级」与「判读」上色：一屏 31 行，纯文字很难扫出谁最挤
    _LV = {"极度拥挤": "#C62828", "拥挤": "#E8710A", "中性": "#8A97A5",
           "偏冷": "#4C8DDF", "冷门": "#1F4E9C", "数据不足": "#C3CBD4"}
    _JD = {"共识形成": "#C62828", "多空对撞": "#8A97A5",
           "系统性撤离": "#1A7F37", "数据不足": "#C3CBD4"}

    def _tint_lv(v):
        c = _LV.get(str(v))
        return f"background-color:{c}22;color:{c};font-weight:700" if c else ""

    def _tint_jd(v):
        c = _JD.get(str(v))
        return f"color:{c};font-weight:600" if c else ""

    _st = (_show.sort_values("CSI", ascending=False).style
           .map(_tint_lv, subset=["拥挤等级", "CSI"])
           .map(_tint_jd, subset=["判读"]))
    st.dataframe(
        _st,
        width="stretch", hide_index=True, height=540,
        column_config={
            f"前十配置%({q})": st.column_config.NumberColumn(format="%.2f"),
            f"环比pp({qp}→{q})": st.column_config.NumberColumn(format="%+.2f"),
            f"同比pp({qy}→{q})": st.column_config.NumberColumn(format="%+.2f"),
            f"当期主动pp({qp}→{q})": st.column_config.NumberColumn(
                format="%+.2f", help="价格中性后的净买卖，分母见「加仓占比」"),
            f"加仓占比%({q})": st.column_config.NumberColumn(
                format="%.0f", help="分子＝加仓基金数；分母＝加仓+减仓（有实质变动的基金）"),
            f"全持仓%({qf})": st.column_config.NumberColumn(format="%.2f"),
            f"半年主动pp({qf0}→{qf})": st.column_config.NumberColumn(
                format="%+.2f", help="GK 分解后的机构主动部分（半年报口径）"),
            "配置分位(31期)": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100,
                help="该行业配置在它自己 31 期历史中的排名"),
            "价格分位(2019起)": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100,
                help="行业指数点位在 2019 年以来每个交易日中的排名"),
            "CSI": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100,
                help="(配置分位+价格分位+资金分位)/3"),
            "横截面分位": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100,
                help="当期 31 个行业之间的排名"),
            "指数近3月%": st.column_config.NumberColumn(format="%+.1f"),
        })
    h(f'<div class="tiny">三组口径、三种期次，<b>不要相加</b>：'
      f'「前十配置/环比/同比/当期主动/广度」是<b>前十大（{qp}→{q}）</b>；'
      f'「全持仓/半年主动」是<b>全持仓（{qf0}→{qf}）</b>；'
      f'「配置分位/CSI」用 31 期前十大算，「价格分位」用 2019 年以来的日频指数算。'
      f'<br>样本 {len(cn.universe(q)):,} 只主动权益基金，已剔除主题基金与覆盖率异常记录。'
      f'<br><b>这条序列含价格漂移</b>：行业涨了，即使没人买，占比也会上升——'
      f'所以「分位高」只能说明<b>它在组合里很重</b>，不能直接说成「机构在买」。'
      f'要剥离价格，用「当期主动pp」与半年的 GK 分解。</div>')
