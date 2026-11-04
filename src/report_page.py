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
import report_charts as rc
import ui


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


def _read(h, fact: str, econ: str, invest: str = "") -> None:
    """图下方的三段式分析：事实 → 经济含义 → 投资含义。

    不再写「横轴是…纵轴是…」那种读图说明——读者要看的是
    「这个数说明了什么、为什么会这样、我该怎么做」。
    """
    ui.analysis(h, fact, econ, invest)


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
    pf, pq = cn.pair_full(), cn.pair_quarter()
    n_funds = len(cn.universe(q))
    pos_hist = cn.position_series()
    _npos = len(pos_hist)          # 半年报期数：文案里不写死
    _nq = len(cn.quarters())       # 前十大口径的期数
    pos = cn.position_change() if pf else {}
    dec = cn.decomp(*pf) if pf else pd.DataFrame()
    br = cn.breadth(*pq) if pq else pd.DataFrame()
    comp = cn.compare_periods(q)
    # 小样本行业（持有基金数<150 或前十配置<0.5%）在广度类图里单独标灰、排最后：
    # 它们的「62% 在加」是 58 只基金里 36 只，和银行的 1,000 只不是同一量级的证据。
    _csi = cn.crowding_score(q)
    _SMALL = ([i for i in _csi.index if _csi.loc[i, "等级"] == "小样本"]
              if not _csi.empty and "等级" in _csi.columns else [])

    # =====================================================================
    # A. 本页结论
    # =====================================================================
    _pct = float(pos_hist["中位"].rank(pct=True).iloc[-1] * 100) if not pos_hist.empty else np.nan
    _concl = []
    if pos:
        _concl.append(
            f'<b>总开关没动</b>：仓位中位 {pos["仓位1"]:.1f}%（{_npos} 期第 {_pct:.0f} 分位），'
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
                _top_ind, _top_cnt = _ind_n.index[0], int(_ind_n.iloc[0])
                _concl.append(
                    f'<b>个股层（{pq[0]}→{pq[1]}，环比）</b>：被最多基金加进来的是 '
                    + "、".join(f'{r["stock_name"]}（{r["d_n"]:+.0f} 家，{r["industry"]}）'
                               for _, r in _a3.iterrows())
                    + '；被减最多的是 '
                    + "、".join(f'{r["stock_name"]}（{r["d_n"]:+.0f} 家，{r["industry"]}）'
                               for _, r in _c3.iterrows())
                    # 只有「前三里同一个行业占 2 只及以上」才能说行业性配置。
                    # 三只各属三个行业时说「集中在电子」，是拿排序第一的行业名硬套，
                    # 读者一看名单就知道不对（电子、医药、有色各一只）。
                    + (f'。<b>加仓前三里 {_top_ind} 占 {_top_cnt} 只，是行业性加仓</b>，'
                       f'跟行业方向做即可，不必逐只研究。'
                       if _top_cnt >= 2 else
                       f'。<b>这三只分属 {len(_ind_n)} 个行业，不属于行业性配置，而是选股</b>，'
                       f'必须逐只判断基本面——买行业 ETF 替代不了。'))
        except Exception:
            pass

    h(f'<div class="sec">一、本页结论 · {q}'
      f'<span style="font-weight:400;font-size:12.5px;color:{C.COLOR["text_dim"]};'
      f'margin-left:8px">样本 {n_funds:,} 只主动权益基金</span></div>')
    h(ui.concl_board(_concl))
    h(_kpis([
        ("最新季报", q, f"全持仓期 {pf[1] if pf else '—'}"),
        ("样本基金", f"{n_funds:,}", "主动权益，剔主题基金"),
        ("股票仓位", f'{pos.get("仓位1", float("nan")):.1f}%' if pos else "—",
         f"{_npos} 期第 {_pct:.0f} 分位" if np.isfinite(_pct) else ""),
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
          f'图 1　仓位：把 {_npos} 个半年报期连起来，才知道现在是高还是低</div>')
        _c = cn.coverage(pos["q"])
        _idx = _c.index.intersection(list(cn.universe(pos["q"])))
        _v = _c.loc[_idx].dropna()
        # 直方图必须与左栏同口径（剔除股票仓位 <40% 的打新/偏债型），
        # 否则同一张图里会并排出现两个中位数（89.0% 与 89.5%），
        # 读者会以为其中一个是错的。
        _v = _v[_v >= cn.POS_MIN]
        st.plotly_chart(rc.fig_position(pos_hist, _v, pos["q"]),
                        width="stretch", config={"displayModeBar": False})
        h(f'<div class="tiny">口径：全持仓半年报（{pos["q0"]}→{pos["q"]}），'
          f'占净值比；只含股票，不含债券与现金。样本 '
          f'{int(pos_hist["基金数"].iloc[-1]):,} 只'
          f'（只统计股票仓位 ≥40% 的基金：2016–2018 年的打新/偏债型灵活配置'
          f'股票只有两三成，混进来会把早期中位数压到 76%，让今天显得「史上最高」）。'
          f'<b>期次说明</b>：最新季报是 {q}，但季报只披露前十大、算不出股票仓位，'
          f'所以仓位只能用最近的全持仓期 {pos["q"]}；后文行业与广度已更新到 {q}。</div>')
        _pmin, _pmax = float(pos_hist["中位"].min()), float(pos_hist["中位"].max())
        _pq1 = float(pos_hist["q1"].iloc[-1]); _pq3 = float(pos_hist["q3"].iloc[-1])
        _read(h,
              f'{_npos} 个半年报期的仓位中位在 {_pmin:.1f}%–{_pmax:.1f}% 之间，'
              f'当前 <b>{pos["仓位1"]:.1f}%</b>，处于第 <b>{_pct:.0f}</b> 分位；'
              f'本期只动了 <b>{pos["中位变化pp"]:+.2f}pp</b>，'
              f'{pos["加仓基金占比"]:.0f}% 加仓、{100 - pos["加仓基金占比"]:.0f}% 减仓，'
              f'两边几乎各占一半；中间 50% 的基金落在 {_pq1:.0f}%–{_pq3:.0f}% 之间。',
              f'股票型基金有 80% 的仓位下限，灵活配置型还要留现金应对赎回，'
              f'所以<b>仓位的可调空间本来就窄</b>——{_npos} 期里中位数的极差只有 '
              f'{_pmax - _pmin:.1f}pp。这意味着「机构看好后市」很少表现为整体加仓，'
              f'而是表现为<b>行业之间的再分配</b>：卖一个行业去买另一个，总仓位几乎不动。'
              f'仓位是<b>存量博弈</b>的证据，也是为什么后面所有分析都要看「结构」而不是「总量」。',
              f'仓位在第 {_pct:.0f} 分位，说明<b>可用的增量资金已经不多</b>：'
              f'需要新增资金推动的普涨行情缺乏基础，收益只能来自结构。'
              f'对组合的含义有两条：① 不要用仓位高低判断顶底（它一年只动几个点）；'
              f'② 高分位时把仓位集中给「可以跟」的行业，不要平均撒——'
              f'平均撒等于买了一个高仓位的指数，承担了拥挤风险却拿不到结构收益。')

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
              f'31 个行业的配置变化里，价格带来的被动变化合计 <b>{_pas:.1f}pp</b>，'
              f'机构主动只有 <b>{_act:.1f}pp</b>，被动占全部变动的 '
              f'<b>{_pas / (_pas + _act) * 100:.0f}%</b>。'
              f'最典型的是 <b>{_topname}</b>：变化 {_t["变化"]:+.2f}pp，'
              f'其中价格 {_t["被动贡献"]:+.2f}pp、机构 {_t["主动贡献"]:+.2f}pp；'
              f'机构主动加最多的是 <b>{_a_max}（{dec.loc[_a_max, "主动贡献"]:+.2f}pp）</b>、'
              f'减最多的是 <b>{_a_min}（{dec.loc[_a_min, "主动贡献"]:+.2f}pp）</b>。',
              f'基金不交易，权重也会变：某行业涨 20%、其他不变，它的占比会自动上升约 2pp。'
              f'<b>这就是价格漂移</b>，它在行业波动大的 A 股尤其明显——所以「配置比例变化」'
              f'大部分不是基金经理的动作，而是市场的动作。把两者混在一起看，'
              f'会把「行业涨了」误读成「机构在买」，从而在情绪高点跟着加仓。'
              f'只有剥离价格之后剩下的<b>主动项</b>，才等同于「净买入」。',
              f'看到「某行业配置暴跌」先别当利空：<b>跌幅大但主动项接近 0 的，'
              f'基本面含义接近于零</b>，只是那个行业跌了。反过来，一个行业配置涨了很多、'
              f'机构却没动，说明它是被「涨」上去的，回落时占比也会自己掉下来，不需要卖。'
              f'真正值得跟随的是<b>主动项绝对值大、且与广度同向</b>的行业；'
              f'主动项与广度打架时（少数大基金在加），只能观察不能跟。')

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
              f'方向一致的有「连续加仓」<b>{len(_su)} 个</b>（'
              + ("、".join(f'{i}（环比 {r["Δ上季pp"]:+.2f}、同比 {r["Δ上年pp"]:+.2f}）'
                          for i, r in _su.iterrows()) or "无")
              + f'）与「连续减仓」<b>{len(_sd)} 个</b>（'
              + ("、".join(f'{i}（环比 {r["Δ上季pp"]:+.2f}、同比 {r["Δ上年pp"]:+.2f}）'
                          for i, r in _sd.iterrows()) or "无")
              + f'）；方向相反的 <b>{len(_opp)} 个</b>，例如 '
              + ("、".join(f'{i}（环比 {r["Δ上季pp"]:+.2f}、同比 {r["Δ上年pp"]:+.2f}）'
                          for i, r in _opp.head(3).iterrows()) or "无") + '。',
              f'两个时间尺度对应两种完全不同的行为。<b>环比</b>里混着季度再平衡、'
              f'应对申赎、以及一次性的调仓，噪音大；<b>同比</b>把季节性抹掉，'
              f'更接近「这一年机构信了什么」。国内基金经理的考核周期多为一年，'
              f'所以同比方向更能代表配置意图，环比更容易被交易性因素污染。'
              f'两者同向，说明是持续行为；一正一负，说明只是季度内的摆动。',
              f'做<b>年度底仓</b>看同比 + 广度：连续一年被加、且多数基金在加的行业，'
              f'才是真正的配置方向。做<b>季度交易</b>看环比 + 当期主动净额：'
              f'它捕捉的是刚刚发生的边际变化，但必须用价格中性口径，否则会把下跌当成减仓。'
              f'方向相反的行业<b>不要当趋势</b>——它只能做交易，不能做底仓。'
              f'另外提醒：这两栏都含价格漂移，是「结果」不是「动作」。')

    # =====================================================================
    # 图4 广度
    # =====================================================================
    if not br.empty:
        h('<div style="font-size:12px;font-weight:700;margin:10px 0 4px">'
          f'图 4　谁在动：{pq[0]} → {pq[1]} 的前十大口径（季频、价格中性）</div>')
        _c1, _c2 = st.columns([0.56, 0.44], gap="small")
        with _c1:
            st.plotly_chart(rc.fig_breadth(br, 12, _SMALL), width="stretch",
                            config={"displayModeBar": False})
        with _c2:
            st.plotly_chart(rc.fig_breadth_vs_move(br, _SMALL), width="stretch",
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
              f'{_hi["行业"]} 的加仓占比最高（{_hi["持有基金数"]:.0f} 只持有者中 '
              f'{_hi["加仓基金数"]:.0f} 只加、{_hi["减仓基金数"]:.0f} 只减，占 '
              f'{_hi["加仓占比"]:.0f}%），{_lo["行业"]} 最低（{_lo["加仓占比"]:.0f}% 在加）。'
              f'「加仓基金不到一半、但净额为正」的有 <b>{len(_lft)}</b> 个（少数大基金在撑），'
              f'「多数在加、净额却为负」的有 <b>{len(_rgt)}</b> 个（小基金接盘）。',
              f'<b>广度回答「共识」，净额回答「力度」</b>，两者分离的根源是基金规模差异：'
              f'一只百亿基金加 3pp，等于几十只小基金各加 3pp 的钱。所以「多少人在加」'
              f'和「加进去多少钱」不是一回事。<b>更关键的是：广度是滞后指标。</b>'
              f'基金经理在排名压力下行为高度同质化，行业涨了才敢加，于是广度往往在'
              f'上涨之后才冲到 70% 以上——这时边际买家其实已经在减少。'
              f'因子检验里广度对次季收益的 IC 为负且不显著（见 ⑤ 因子检验），正是这个机制。',
              f'只跟<b>右上角</b>（广度高 + 净额正）的行业：共识与资金同向，行情有延续性。'
              f'<b>右下角最危险</b>——大资金已经在撤、小基金还在接，这是典型的派发形态；'
              f'左上角（少数大基金加）只能观察，因为一个人停手就结束。'
              f'还有一条反直觉的用法：广度冲到 75% 以上时，不要加仓，'
              f'要开始想「谁还没上车」——没有后来者，就没有下一段涨幅。')

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
                h(f'<div class="tiny">口径：<b>前十大</b>。'
                  f'季报（Q1/Q3）法定只披露<b>前十大重仓股</b>；半年报/年报（Q2/Q4）披露'
                  f'<b>全部持仓</b>。这里对四个季度统一只取「每只基金最重的 10 只」，'
                  f'所以披露范围一致，<b>环比与同比都能算</b>——这不是跨口径拼接，'
                  f'而是把全持仓也截到前十大，与季报对齐。'
                  f'只含主动权益样本 {sf.attrs["n_funds"][1]:,} 只；'
                  f'已剔除两期持有基金数都不到 10 家的小样本。'
                  f'注意「−100」不等于清仓，也可能是掉出前十。</div>')
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
                      f'个股层的加仓，反映的是基金经理对<b>产业景气、订单、估值</b>的判断，'
                      f'比行业配置更具体。但它有一个陷阱：<b>当一只股票被几百只基金同时加进来，'
                      f'它的价格里已经包含了这部分共识</b>，此时跟随买入的赔率并不好。'
                      f'真正有价值的是「环比与同比打架」的股票——它们暴露了持仓周期的切换。',
                      f'环比大正、同比大负：刚被买进来的新宠，逻辑还没被验证，只能小仓位试。'
                      f'环比负、同比大正：<b>这一年被大幅加过、现在开始撤</b>，浮盈盘还没走完，'
                      f'最需要警惕，持有的应减仓。两者都为正：机构在持续加，'
                      f'但要看行业——同行业的股票一起被加，说明是行业性配置，买 ETF 即可；'
                      f'散在不同行业才是选股，才值得逐只研究基本面。')
                # 行业构成
                h('<div style="font-size:12px;font-weight:700;margin:8px 0 3px">'
                  f'图 6　{_pair[0]}→{_pair[1]} 加仓/减仓前 50 只的行业构成</div>')
                st.plotly_chart(rc.fig_stock_industry(sf_f, 50, *_pair), width="stretch",
                                config={"displayModeBar": False})
                _top50a = sf_f.nlargest(50, "d_n")["industry"].value_counts()
                _top50c = sf_f.nsmallest(50, "d_n")["industry"].value_counts()
                _read(h,
                      f'加仓前 50 只里，<b>{_top50a.index[0]}</b> 占 '
                      f'{_top50a.iloc[0]} 只（{_top50a.iloc[0] / 50 * 100:.0f}%），'
                      f'其次是 {_top50a.index[1] if len(_top50a) > 1 else "—"} '
                      f'{_top50a.iloc[1] if len(_top50a) > 1 else 0} 只；'
                      f'减仓前 50 只里最多的是 <b>{_top50c.index[0]}</b> '
                      f'{_top50c.iloc[0]} 只。',
                      f'这组分布决定这批买入的<b>性质</b>：集中在少数行业，说明基金经理是在'
                      f'做行业配置，个股只是载体，涨跌由行业景气决定；分散在多个行业，'
                      f'说明是在做选股，收益来自个体差异。两种行为的风险来源完全不同——'
                      f'前者怕行业逻辑被证伪，后者怕个股基本面出问题。',
                      f'加仓集中在 {_top50a.index[0]} 时，<b>直接买这个行业的 ETF 比挑个股更省事</b>，'
                      f'因为收益主要由行业决定，选股的边际贡献很小；'
                      f'加仓分散时，个股榜才有独立价值，需要逐只判断，'
                      f'而且要注意「看起来在选股、其实在押同一个赛道」的伪分散。'
                      f'减仓端同理：减仓集中在某个行业时，那是行业性撤退，'
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
            "平均主动pp": f"当期主动pp({qp_}→{q_})", "历史分位": f"配置分位({_nq}期)",
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
                    format="%.2f", help=f"前十大归一化口径，{_nq} 期都有，用于环比/同比"),
                f"环比pp({qp_}→{q_})": st.column_config.NumberColumn(
                    format="%+.2f", help="含价格漂移，是结果不是动作"),
                f"同比pp({qy_}→{q_})": st.column_config.NumberColumn(format="%+.2f"),
                f"当期主动pp({qp_}→{q_})": st.column_config.NumberColumn(
                    format="%+.2f", help="价格中性后的净买卖"),
                f"配置分位({_nq}期)": st.column_config.ProgressColumn(
                    format="%.0f", min_value=0, max_value=100,
                    help=f"该行业当前配置在自身 {_nq} 期历史中的排名"),
                f"加仓占比%({q_})": st.column_config.NumberColumn(
                    format="%.0f", help="分母是「加仓+减仓」的基金数，不是全部样本"),
                "判读": st.column_config.TextColumn(width="small"),
                "行业": st.column_config.TextColumn(width="small", pinned=True),
                "读数": st.column_config.TextColumn(
                    width="small",
                    help="真加仓＝当期净额为正且多数在加；少数大基金加＝净额为正但广度不足；"
                         "真撤退＝净额为负且多数在减；小基金接盘＝多数在加但净额为负；"
                         "方向冲突＝介于两者之间"),
            })
        h(f'<div class="tiny">三组列、三种期次，<b>不要相加</b>：'
          f'「配置/半年主动/半年被动」是<b>全持仓半年报</b>口径（{qf0_}→{qf_}），'
          f'最准但一年两次；「前十配置/环比/同比/当期主动/加仓占比」是'
          f'<b>前十大归一化</b>口径（{q_} 对 {qp_} 与 {qy_}），{_nq} 期都有、跨季可比；'
          f'「配置分位」用 {_nq} 期前十大口径算。'
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
