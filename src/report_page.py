# -*- coding: utf-8 -*-
"""② 本期解读：一张图一个信息点，每张图下面都把「读图 → 结论」写清楚。

和「① 结论总览」的分工
----------------------
    ① 结论总览：给判断与动作（跟什么、避什么、多少仓位、何时推翻）
    ② 本期解读：给证据与推导（这个数怎么来的、为什么这样读、换了口径会怎样）

所以本页**不再重复**投资建议表与操作规则，只保留每张图的解读与结论。
这样两页不会出现「同一句话两个数」，也不会让读者在两张表之间对不上。

四段结构
--------
    A 本页结论   B 六张图 + 读图 + 结论   C 行业台账   D 方法与边界
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import canon as cn
import config as C
import playbook as pb
import report_charts as rc


def _fmt(v, unit="", nd=1, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (int, float)) and abs(v) < 0.1 and nd < 2:
        nd = 2
    return (f"{v:+,.{nd}f}" if sign else f"{v:,.{nd}f}") + unit


def _kpis(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="kpi"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="s">{s}</div></div>' for k, v, s in items)
    return f'<div class="kpis">{cells}</div>'


def _read(h, read: str, concl: str) -> None:
    """统一的「读图 + 结论」块。每张图下面都必须有一块，否则图就是摆设。"""
    h(f'<div style="background:{C.COLOR["panel"]};border-left:3px solid {C.COLOR["price"]};'
      f'padding:9px 12px;border-radius:0 6px 6px 0;margin:4px 0 16px">'
      f'<div style="font-size:12px;line-height:1.9"><b>读图</b>：{read}</div>'
      f'<div style="font-size:12.5px;line-height:1.9;margin-top:5px">'
      f'<b style="color:{C.COLOR["threshold"]}">结论</b>：{concl}</div></div>')


def _read_row(r) -> str:
    """把「当期净额」与「加仓广度」合成一句读数，两者矛盾时直说。"""
    a = r.get("平均主动pp", np.nan)
    s_ = r.get("加仓占比", np.nan)
    if not np.isfinite(a) or not np.isfinite(s_):
        return "—"
    if a > 0 and s_ >= 60:
        return "真加仓"
    if a > 0 and s_ < 50:
        return "少数大基金加"
    if a < 0 and s_ <= 40:
        return "真撤退"
    if a < 0 and s_ > 50:
        return "小基金接盘"
    return "方向冲突"


def render(h, B=None, IND=None) -> None:
    """入口。B/IND 仅为兼容旧签名，本页不依赖 Wind 数据。"""
    if not (cn.available() or cn.fallback_available()):
        h('<div class="cg-note">未找到持仓数据。运行 '
          '<span class="mono">python tools/run_ak_all.py</span> 抓取，'
          '或 <span class="mono">python tools/export_canon.py</span> 生成压缩快照。</div>')
        return

    q = cn.latest()
    pf, pq, ps = cn.pair_full(), cn.pair_quarter(), cn.pair_same()
    n_funds = len(cn.universe(q))
    pos_hist = cn.position_series()
    pos = cn.position_change() if pf else {}
    dec = cn.decomp(*pf) if pf else pd.DataFrame()
    br = cn.breadth(*pq) if pq else pd.DataFrame()
    comp = cn.compare_periods(q)

    # =====================================================================
    # A. 本页结论
    # =====================================================================
    _pct = float(pos_hist["中位"].rank(pct=True).iloc[-1] * 100) if not pos_hist.empty else np.nan
    _concl = []
    if pos:
        _concl.append(
            f'<b>总开关没动</b>：仓位中位 {pos["仓位1"]:.1f}%（15 期第 {_pct:.0f} 分位），'
            f'变化 {pos["中位变化pp"]:+.2f}pp，{pos["加仓基金占比"]:.0f}% 的基金加仓。'
            f'所以下面的行业变化是<b>内部换仓</b>，不是新增资金——'
            f'某行业配置上升可以读成「机构更看好」，不必担心是「大盘涨、水涨船高」。')
    if not dec.empty:
        _u, _d = dec["主动贡献"].idxmax(), dec["主动贡献"].idxmin()
        _pas = float(dec["被动贡献"].abs().sum())
        _act = float(dec["主动贡献"].abs().sum())
        _concl.append(
            f'<b>配置变化大半是价格</b>：31 个行业里，价格带来的被动变化合计 '
            f'{_pas:.1f}pp、机构主动只占 {_act:.1f}pp，'
            f'被动占全部变动的 {_pas / (_pas + _act) * 100:.0f}%。'
            f'主动净买最多 <b>{_u}（{dec.loc[_u, "主动贡献"]:+.2f}pp）</b>、'
            f'净卖最多 <b>{_d}（{dec.loc[_d, "主动贡献"]:+.2f}pp）</b>。'
            f'<b>这两个数才是机构的态度</b>，配置比例的变化不是。')
    if not br.empty:
        _gu = br.loc[br["加仓占比"].idxmax()]
        _gd = br.loc[br["加仓占比"].idxmin()]
        _n_c = int((br["判读"] == "共识形成").sum())
        _n_s = int((br["判读"] == "系统性撤离").sum())
        _concl.append(
            f'<b>共识只在少数行业</b>：{_n_c} 个「共识形成」、{_n_s} 个「系统性撤离」，'
            f'其余 {len(br) - _n_c - _n_s} 个多空对撞。'
            f'最高 <b>{_gu["行业"]} {_gu["加仓占比"]:.0f}%</b>（{_gu["持有基金数"]:.0f} 只持有者中 '
            f'{_gu["加仓基金数"]:.0f} 只加、{_gu["减仓基金数"]:.0f} 只减），'
            f'最低 <b>{_gd["行业"]} {_gd["加仓占比"]:.0f}%</b>。'
            f'<b>只有「多数在加 + 净额为正」才叫共识</b>，加仓不到一半却净额为正的，'
            f'是少数大基金在撑，最容易反转。')
    if pq:
        try:
            _sf = cn.stock_flow_top10(*pq)          # 相邻季，前十大口径
            _sf = _sf[(_sf["n_funds"] >= 10) | (_sf["n_funds_0"] >= 10)]
            if not _sf.empty:
                _a3 = _sf.nlargest(3, "d_n")
                _c3 = _sf.nsmallest(3, "d_n")
                _ind_n = _a3["industry"].value_counts()
                _concl.append(
                    f'<b>个股层（{pq[0]}→{pq[1]}，环比）：加仓集中在 {_ind_n.index[0]}</b>——'
                    f'被最多基金加进来的三只是 '
                    + "、".join(f'{r["stock_name"]}（{r["d_n"]:+.0f} 家）'
                               for _, r in _a3.iterrows())
                    + '；被减最多的是 '
                    + "、".join(f'{r["stock_name"]}（{r["d_n"]:+.0f} 家）'
                               for _, r in _c3.iterrows())
                    + f'。<b>前十大口径下环比与同比都能比</b>（两期披露范围一致），'
                    f'两者经常相反：环比在减、同比在加的，是「这一年被加过、现在开始撤」，'
                    f'最需要警惕。')
        except Exception:
            pass

    h(f'<div class="sec">一、本页结论 · {q}（样本 {n_funds:,} 只主动权益基金）</div>')
    h('<div class="cg-note" style="font-size:12.5px;line-height:1.95">'
      + "<br>".join(f"{i}. {x}" for i, x in enumerate(_concl, 1)) + "</div>")
    h(_kpis([
        ("最新季报", q, f"全持仓期 {pf[1] if pf else '—'}"),
        ("样本基金", f"{n_funds:,}", "主动权益，剔主题基金"),
        ("股票仓位", f'{pos.get("仓位1", float("nan")):.1f}%' if pos else "—",
         f"15 期第 {_pct:.0f} 分位" if np.isfinite(_pct) else ""),
        ("共识 / 撤离", f"{int((br['判读'] == '共识形成').sum())} / "
                        f"{int((br['判读'] == '系统性撤离').sum())}" if not br.empty else "—",
         "行业数，见第 4 张图"),
    ]))
    h(f'<div class="tiny" style="margin-top:4px">本页只讲「数据说了什么」；'
      f'「该买什么、买多少、什么时候推翻」在「① 结论总览」，两页用同一套规则，不会打架。</div>')
    st.write("")

    # =====================================================================
    # 图1 仓位
    # =====================================================================
    if pos:
        h('<div class="sec">二、六张图看懂这一季</div>')
        h('<div style="font-size:12px;font-weight:700;margin:2px 0 4px">'
          '图 1　仓位：把 15 个半年报期连起来，才知道现在是高还是低</div>')
        _c = cn.coverage(pos["q"])
        _idx = _c.index.intersection(list(cn.universe(pos["q"])))
        st.plotly_chart(rc.fig_position(pos_hist, _c.loc[_idx].dropna(), pos["q"]),
                        width="stretch", config={"displayModeBar": False})
        h(f'<div class="tiny">口径：全持仓半年报（{pos["q0"]}→{pos["q"]}），'
          f'占净值比；只含股票，不含债券与现金。样本 {pos["n"]:,} 只。'
          f'<b>期次说明</b>：最新季报是 {q}，但季报只披露前十大、算不出股票仓位，'
          f'所以仓位只能用最近的全持仓期 {pos["q"]}；后文行业与广度已更新到 {q}。</div>')
        _read(h,
              f'左图是 {len(pos_hist)} 个半年报期的仓位中位数（阴影是 25–75 分位带），'
              f'右图是本期 {len(_idx):,} 只基金的仓位分布。'
              f'当前中位 <b>{pos["仓位1"]:.1f}%</b>，位于历史第 <b>{_pct:.0f}</b> 分位，'
              f'本期变化 <b>{pos["中位变化pp"]:+.2f}pp</b>，'
              f'{pos["加仓基金占比"]:.0f}% 的基金加仓、{100 - pos["加仓基金占比"]:.0f}% 减仓。',
              f'仓位是「总开关」：<b>整体没动，说明下面的行业变化是内部换仓，不是新增资金</b>。'
              f'这一条决定了后面的读法——某行业配置上升，可以放心读成「机构更看好它」，'
              f'而不是「大盘涨了、水涨船高」。反过来，仓位已在 {_pct:.0f} 分位的高位，'
              f'<b>指望「全体加仓推动普涨」是不现实的</b>；'
              f'想涨，只能靠行业之间的结构性机会。')

    # =====================================================================
    # 图2 主动/被动
    # =====================================================================
    if not dec.empty:
        h('<div style="font-size:12px;font-weight:700;margin:10px 0 4px">'
          '图 2　配置变化拆开看：灰色是价格涨出来的，彩色才是机构动手</div>')
        st.plotly_chart(rc.fig_active_passive(dec), width="stretch",
                        config={"displayModeBar": False})
        h(f'<div class="tiny">口径：<b>全持仓半年报 {dec.attrs["q0"]}→{dec.attrs["q"]}</b>，'
          f'权重归一化到 100%；收益用申万一级行业指数。样本 {dec.attrs["n_funds"]:,} 只。'
          f'<b>注意期次</b>：这是「上一个半年」的主动/被动，'
          f'与下面图 3 的「本季环比（{pq[0] if pq else ""}→{q}）」不是同一段时间，不要混。</div>')
        _pas = float(dec["被动贡献"].abs().sum())
        _act = float(dec["主动贡献"].abs().sum())
        _topname = dec["被动贡献"].abs().idxmax()
        _t = dec.loc[_topname]
        _a_max = dec["主动贡献"].idxmax()
        _a_min = dec["主动贡献"].idxmin()
        _read(h,
              f'把 31 个行业的配置变化拆成两半：价格带来的被动变化合计 <b>{_pas:.1f}pp</b>，'
              f'机构主动只占 <b>{_act:.1f}pp</b>，被动占了全部变动的 '
              f'<b>{_pas / (_pas + _act) * 100:.0f}%</b>。'
              f'最典型的是 <b>{_topname}</b>：变化 {_t["变化"]:+.2f}pp，'
              f'其中价格 {_t["被动贡献"]:+.2f}pp、机构 {_t["主动贡献"]:+.2f}pp；'
              f'机构主动加最多的是 <b>{_a_max}（{dec.loc[_a_max, "主动贡献"]:+.2f}pp）</b>、'
              f'减最多的是 <b>{_a_min}（{dec.loc[_a_min, "主动贡献"]:+.2f}pp）</b>。',
              f'看到「某行业配置暴跌」先别当利空——<b>跌幅大但主动项接近 0 的，'
              f'基本面含义接近于零</b>，只是那个行业跌了。真正该重视的是'
              f'<b>主动项绝对值大、且与广度同向</b>的行业；'
              f'反过来，一个行业配置涨了很多但机构没动，说明它是被「涨」上去的，'
              f'接下来回落时配置也会自己掉下来，不需要卖。'
              f'<b>这就是为什么「配置变化」不能直接当「机构观点」用。</b>')

    # =====================================================================
    # 图3 环比/同比
    # =====================================================================
    if not comp.empty:
        h('<div style="font-size:12px;font-weight:700;margin:10px 0 4px">'
          f'图 3　两个时间尺度：比上一季（{comp.attrs["q_prev"]}）与比去年同季'
          f'（{comp.attrs["q_year"]}）</div>')
        st.plotly_chart(rc.fig_compare_two(comp), width="stretch",
                        config={"displayModeBar": False})
        _dd = comp.dropna(subset=["Δ上季pp", "Δ上年pp"])
        _opp = _dd[(_dd["Δ上季pp"] * _dd["Δ上年pp"]) < 0]
        _su = _dd[(_dd["Δ上季pp"] > 0) & (_dd["Δ上年pp"] > 0)].nlargest(3, "Δ上年pp")
        _sd = _dd[(_dd["Δ上季pp"] < 0) & (_dd["Δ上年pp"] < 0)].nsmallest(3, "Δ上年pp")
        h(f'<div class="tiny">口径：前十大归一化（{comp.attrs["q"]} 对 '
          f'{comp.attrs["q_prev"]} 与 {comp.attrs["q_year"]}），<b>含价格漂移</b>——'
          f'行业涨跌本身会改变占比，所以这两栏是「结果」不是「动作」。</div>')
        _read(h,
              f'左栏是本季对上一季，右栏是本季对去年同季。'
              f'<b>方向一致的有「连续加仓」{len(_su)} 个</b>（'
              + ("、".join(f'{i}（环比 {r["Δ上季pp"]:+.2f}、同比 {r["Δ上年pp"]:+.2f}）'
                          for i, r in _su.iterrows()) or "无")
              + f'）与<b>「连续减仓」{len(_sd)} 个</b>（'
              + ("、".join(f'{i}（环比 {r["Δ上季pp"]:+.2f}、同比 {r["Δ上年pp"]:+.2f}）'
                          for i, r in _sd.iterrows()) or "无")
              + f'）；<b>方向相反的 {len(_opp)} 个</b>，'
                f'比如 ' + ("、".join(
                    f'{i}（环比 {r["Δ上季pp"]:+.2f}、同比 {r["Δ上年pp"]:+.2f}）'
                    for i, r in _opp.head(3).iterrows()) or "无") + '。',
              f'<b>方向一致才是趋势，方向相反只是摆动。</b>'
              f'环比说「在加」、同比说「在减」的行业，是短周期调仓，'
              f'只能做交易、不能做底仓；反过来，连续两期都在加的行业，'
              f'才值得按年度维度配置。只看环比会被季度噪音带偏，只看同比会错过拐点——'
              f'两条一起看，才知道谁是「刚启动」、谁是「已经走了一年」。'
              f'<b>但两栏都不能直接当「机构在买」</b>：这里的数字含价格，'
              f'真正的交易方向看下面图 4 的广度。')

    # =====================================================================
    # 图4 广度
    # =====================================================================
    if not br.empty:
        h('<div style="font-size:12px;font-weight:700;margin:10px 0 4px">'
          f'图 4　谁在动：{pq[0]} → {pq[1]} 的前十大口径（季频、价格中性）</div>')
        _c1, _c2 = st.columns([0.56, 0.44], gap="small")
        with _c1:
            st.plotly_chart(rc.fig_breadth(br), width="stretch",
                            config={"displayModeBar": False})
        with _c2:
            st.plotly_chart(rc.fig_breadth_vs_move(br), width="stretch",
                            config={"displayModeBar": False})
        h(f'<div class="tiny">口径：前十大归一化，逐基金算<b>价格中性</b>变化'
          f'（假设不交易时权重随行业指数漂移，实际与反事实之差才是交易）；'
          f'单只基金变化超过 ±0.1pp 才计数。'
          f'加仓占比的分母是<b>加仓 + 减仓</b>（有实质变动的基金），不是全部样本。'
          f'样本 {br.attrs["n_funds"]:,} 只。</div>')
        _hi = br.loc[br["加仓占比"].idxmax()]
        _lo = br.loc[br["加仓占比"].idxmin()]
        _lft = br[(br["加仓占比"] < 50) & (br["平均主动pp"] > 0)]
        _rgt = br[(br["加仓占比"] >= 60) & (br["平均主动pp"] < 0)]
        _read(h,
              f'左图按「加仓基金占比」挑出最偏离 50% 的 14 个行业：'
              f'<b>{_hi["行业"]}</b> 最高（{_hi["持有基金数"]:.0f} 只持有者中 '
              f'{_hi["加仓基金数"]:.0f} 只加、{_hi["减仓基金数"]:.0f} 只减，占 '
              f'{_hi["加仓占比"]:.0f}%），<b>{_lo["行业"]}</b> 最低'
              f'（{_lo["加仓占比"]:.0f}% 在加）。右图把「多少基金在加」和'
              f'「价格中性净额」放在一起，左上角有 <b>{len(_lft)}</b> 个行业是'
              f'「加仓基金不到一半、但净额为正」——少数大基金在撑；'
              f'右下角有 <b>{len(_rgt)}</b> 个是「多数在加、净额却为负」——小基金接盘。',
              f'<b>广度是分辨「共识」与「假象」的唯一办法。</b>'
              f'同一个「平均加仓 2pp」，900 只基金各加 2pp 和 200 只大基金各加 9pp '
              f'含义完全相反：前者是共识、能持续，后者一停手就反转。'
              f'所以只跟右上方（广度高 + 净额正）的行业；'
              f'左上角那批只能观察，不能当信号；右下角是<b>最危险</b>的形态——'
              f'大资金已经在撤，小基金还在接。')

    # =====================================================================
    # 图5 个股（相邻季 + 上年同季，两把尺子都给）
    # =====================================================================
    try:
        _qoq = cn.pair_quarter()          # 2026Q2 → 2026Q3
        _yoy = cn.pair_same()             # 2025Q3 → 2026Q3
        _basis = st.radio(
            "个股榜的对比基准",
            [f"环比：{_qoq[0]} → {_qoq[1]}（看这一季谁在动）" if _qoq else "环比",
             f"同比：{_yoy[0]} → {_yoy[1]}（看这一年谁被换掉）" if _yoy else "同比"],
            horizontal=True, key="stock_basis", index=0)
        _pair = _qoq if _qoq and _basis.startswith("环比") else _yoy
        if _pair:
            sf = cn.stock_flow_top10(*_pair)
            _sf_other = cn.stock_flow_top10(*(_yoy if _basis.startswith("环比") else _qoq))
            h('<div style="font-size:12px;font-weight:700;margin:10px 0 4px">'
              f'图 5　落到个股：{_pair[0]} → {_pair[1]}，谁被加进来、谁被移出去</div>')
            if not sf.empty:
                sf_f = sf[(sf["n_funds"] >= 10) | (sf["n_funds_0"] >= 10)]
                st.plotly_chart(rc.fig_stock_flow(sf_f, 10, *_pair), width="stretch",
                                config={"displayModeBar": False})
                _a, _c = sf_f.nlargest(3, "d_n"), sf_f.nsmallest(3, "d_n")
                _ind_a = _a["industry"].value_counts()
                h(f'<div class="tiny">口径：<b>前十大</b>（两期都只取每只基金最重的 10 只，'
                  f'披露范围一致才可比），只含主动权益样本 '
                  f'{sf.attrs["n_funds"][1]:,} 只；已剔除两期持有基金数都不到 10 家的小样本。'
                  f'「−100」不等于清仓，也可能是掉出前十。</div>')
                # 两个基准的对照表：同一只股票，环比与同比可能完全相反
                _cmp = pd.DataFrame({
                    "本期基金数": sf["n_funds"],
                    "环比Δ": sf["d_n"],
                    "同比Δ": _sf_other["d_n"].reindex(sf.index) if not _sf_other.empty else np.nan,
                    "行业": sf["industry"],
                })
                _show = _cmp.reindex(
                    list(_a.index) + list(_c.index)).drop_duplicates()
                _show = _show.rename(columns={"index": "代码"})
                _show.index.name = "代码"
                st.dataframe(_show.reset_index(), width="stretch", hide_index=True,
                             height=250,
                             column_config={
                                 "本期基金数": st.column_config.NumberColumn(format="%.0f"),
                                 "环比Δ": st.column_config.NumberColumn(format="%+.0f"),
                                 "同比Δ": st.column_config.NumberColumn(
                                     format="%+.0f", help="与上年同季比，正数＝这一年被更多基金持有"),
                             })
                _read(h,
                      f'被最多基金新买入：' + "、".join(
                          f'{r["stock_name"]}（{r["n_funds_0"]:.0f}→{r["n_funds"]:.0f} 家，'
                          f'Δ{r["d_n"]:+.0f}，{r["industry"]}）'
                          for _, r in _a.iterrows())
                      + f'；被最多基金卖出：' + "、".join(
                          f'{r["stock_name"]}（{r["n_funds_0"]:.0f}→{r["n_funds"]:.0f} 家，'
                          f'Δ{r["d_n"]:+.0f}，{r["industry"]}）'
                          for _, r in _c.iterrows()) + '。',
                      f'<b>环比看「这一季谁在动」，同比看「这一年谁被换掉」，两者经常相反。</b>'
                      f'上表同时给了两个 Δ：环比大正、同比大负的，是「刚被买进来的新宠」；'
                      f'环比负、同比正的，是「这一年被加过、现在开始撤」——后者最危险，'
                      f'因为浮盈盘还没走完。'
                      + (f' 本期加仓前三集中在 {_ind_a.index[0]}'
                         f'（{_ind_a.iloc[0]} 只），说明更像<b>行业性配置</b>，'
                         f'跟着行业方向做即可。'
                         if len(_ind_a) <= 1 else
                         f' 本期加仓前三分散在 {len(_ind_a)} 个行业，'
                         f'说明更偏<b>个股选择</b>，需要逐只判断。'))
                # 行业构成
                h('<div style="font-size:12px;font-weight:700;margin:8px 0 3px">'
                  f'图 6　{_pair[0]}→{_pair[1]} 加仓/减仓前 50 只的行业构成</div>')
                st.plotly_chart(rc.fig_stock_industry(sf_f, 50, *_pair), width="stretch",
                                config={"displayModeBar": False})
                _top50a = sf_f.nlargest(50, "d_n")["industry"].value_counts()
                _top50c = sf_f.nsmallest(50, "d_n")["industry"].value_counts()
                _read(h,
                      f'把加仓前 50 只按行业归类：<b>{_top50a.index[0]}</b> 占 '
                      f'{_top50a.iloc[0]} 只（{_top50a.iloc[0] / 50 * 100:.0f}%），'
                      f'其次是 {_top50a.index[1] if len(_top50a) > 1 else "—"} '
                      f'{_top50a.iloc[1] if len(_top50a) > 1 else 0} 只；'
                      f'减仓前 50 只里最多的是 <b>{_top50c.index[0]}</b> '
                      f'{_top50c.iloc[0]} 只。',
                      f'这条信息决定你该怎么用个股榜：'
                      + (f'加仓高度集中在 {_top50a.index[0]}，'
                         f'<b>直接买这个行业的 ETF 比挑个股更省事</b>，'
                         f'因为涨跌主要由行业决定。'
                         if _top50a.iloc[0] / 50 > 0.3 else
                         '加仓分散在多个行业，<b>说明基金经理在选股而不是配行业</b>，'
                         '这时个股榜才有独立价值，需要逐只判断。')
                      + f'减仓端同理：如果减仓集中在某个行业，那是行业性撤退，'
                        f'不要在这个行业里挑「被错杀」的个股。')
    except Exception as _e:
        h(f'<div class="tiny">个股榜暂不可用：{type(_e).__name__}</div>')

    # =====================================================================
    # C. 行业台账
    # =====================================================================
    if not comp.empty:
        h('<div class="sec">三、行业台账：一行一个行业，水平 + 环比 + 同比 + 主动/被动</div>')
        t = comp.copy()
        t["读数"] = t.apply(_read_row, axis=1)
        q_, qp_, qy_ = comp.attrs["q"], comp.attrs["q_prev"], comp.attrs["q_year"]
        qf_, qf0_ = comp.attrs["q_full"], comp.attrs["q_full0"]
        show = t.reset_index()[[
            "行业", "配置_全持仓", "主动pp", "被动pp", "配置_前十",
            "Δ上季pp", "Δ上年pp", "平均主动pp", "历史分位", "加仓占比", "判读", "读数"]]
        show = show.rename(columns={
            "配置_全持仓": f"配置%({qf_})", "主动pp": f"半年主动pp({qf0_}→{qf_})",
            "被动pp": f"半年被动pp({qf0_}→{qf_})", "配置_前十": f"前十配置%({q_})",
            "Δ上季pp": f"环比pp({qp_}→{q_})", "Δ上年pp": f"同比pp({qy_}→{q_})",
            "平均主动pp": f"当期主动pp({qp_}→{q_})", "历史分位": "配置分位(31期)",
            "加仓占比": f"加仓占比%({q_})"})
        # 判读/读数上色：31 行文字里找「真加仓」很费眼，颜色能一眼扫出来。
        # 红＝资金在进（真加仓/共识），绿＝在撤（真撤退/系统性撤离），
        # 橙＝少数大基金在撑，灰＝方向冲突。
        _tint_map = {"真加仓": "#C62828", "共识形成": "#C62828",
                     "少数大基金加": "#E8710A", "小基金接盘": "#E8710A",
                     "真撤退": "#1A7F37", "系统性撤离": "#1A7F37",
                     "方向冲突": "#6B7A8C", "多空对撞": "#6B7A8C"}

        def _tint(v):
            c = _tint_map.get(str(v))
            return (f"background-color:{c}1F;color:{c};font-weight:600") if c else ""

        _styled = show.style.map(_tint, subset=["判读", "读数"])
        st.dataframe(
            _styled, width="stretch", hide_index=True, height=560,
            column_config={
                f"配置%({qf_})": st.column_config.NumberColumn(
                    format="%.2f", help="全持仓口径，占净值比；之和＝股票仓位"),
                f"半年主动pp({qf0_}→{qf_})": st.column_config.NumberColumn(
                    format="%+.2f", help="GK 分解后的机构主动部分（半年报口径）"),
                f"半年被动pp({qf0_}→{qf_})": st.column_config.NumberColumn(
                    format="%+.2f", help="价格涨跌带来的部分（半年报口径）"),
                f"前十配置%({q_})": st.column_config.NumberColumn(
                    format="%.2f", help="前十大归一化口径，31 期都有，用于环比/同比"),
                f"环比pp({qp_}→{q_})": st.column_config.NumberColumn(
                    format="%+.2f", help="含价格漂移，是结果不是动作"),
                f"同比pp({qy_}→{q_})": st.column_config.NumberColumn(format="%+.2f"),
                f"当期主动pp({qp_}→{q_})": st.column_config.NumberColumn(
                    format="%+.2f", help="价格中性后的净买卖"),
                "配置分位(31期)": st.column_config.ProgressColumn(
                    format="%.0f", min_value=0, max_value=100,
                    help="该行业当前配置在自身 31 期历史中的排名"),
                f"加仓占比%({q_})": st.column_config.NumberColumn(
                    format="%.0f", help="分母是「加仓+减仓」的基金数，不是全部样本"),
                "判读": st.column_config.TextColumn(width="small"),
                "读数": st.column_config.TextColumn(
                    width="small",
                    help="真加仓＝当期净额为正且多数在加；少数大基金加＝净额为正但广度不足；"
                         "真撤退＝净额为负且多数在减；小基金接盘＝多数在加但净额为负；"
                         "方向冲突＝介于两者之间"),
            })
        h(f'<div class="tiny">三组列、三种期次，<b>不要相加</b>：'
          f'「配置/半年主动/半年被动」是<b>全持仓半年报</b>口径（{qf0_}→{qf_}），'
          f'最准但一年两次；「前十配置/环比/同比/当期主动/加仓占比」是'
          f'<b>前十大归一化</b>口径（{q_} 对 {qp_} 与 {qy_}），31 期都有、跨季可比；'
          f'「配置分位」用 31 期前十大口径算。'
          f'三套口径各回答一个问题，混着看必然矛盾。</div>')

    # =====================================================================
    # D. 方法与免责
    # =====================================================================
    h('<div class="sec">四、这套判断的边界（不写清楚就等于骗人）</div>')
    h('<div class="cg-note" style="line-height:1.9">'
      '· <b>它是描述，不是预测</b>：加仓广度对次季行业收益的 RankIC 长期在 0 附近、'
      'p 值不显著（实测见「因子检验」页）；「拥挤即风险」只有方向、没有统计显著性。'
      '它能告诉你机构在做什么，不能告诉你接下来会涨什么。<br>'
      '· <b>季报有时滞</b>：季末到数据可得约 3–4 周（法定 15 个工作日），'
      '本期数据抓取于季末后 28 天，你看到时下季度已过约 1/3。'
      '所有「本期」都是「上一个已披露期」，不是「今天」。<br>'
      '· <b>季报只有前十大</b>：Q1/Q3 每只基金只披露 10 只股票，'
      '所以季报期的配置一律用来算广度，不用来判断绝对水平。<br>'
      '· <b>主动/被动只在半年报做</b>：季报的前十大会换人，逐季做 GK 分解噪声极大，'
      '本库不做这种「看起来更精细、实际更不可靠」的事。<br>'
      '· <b>操作建议是规则输出，不是投资建议</b>：规则事前固定、触发条件可验证，'
      '但你的风险预算、持有期限、税负都不在模型里。'
      '</div>')
