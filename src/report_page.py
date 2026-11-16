# -*- coding: utf-8 -*-
"""② 本期解读（全市场）：一张图一个信息点，每张图下面都把「读图 → 结论」写清楚。

和「① 结论总览」的分工
----------------------
    ① 结论总览：给判断与动作（跟什么、避什么、多少仓位、何时推翻）
    ② 本期解读（全市场）：给证据与推导（这个数怎么来的、为什么这样读、换了口径会怎样）

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
    """图下方的一行式读数：事实 → 结论。

    2026-11-16 精简：六张图各配一张三段式卡片，整页多出约 1500px 纯文字，
    读者滚到第三张就忘了第一张。改成一行后，判断留给上方的四条结论，
    这里只回答「这张图看到了什么」。
    """
    parts = [p for p in (fact, econ, invest) if p]
    if not parts:
        return
    h(f'<div class="tiny" style="margin:2px 0 13px;line-height:1.95">'
      + '　'.join(parts) + '</div>')


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
    # 广度排名/计数一律剔除小样本：它们的「26% 在加」是几十只基金里的几个人，
    # 既不该当「最低」，也不该计成一次「系统性撤离」（③ 页对它们不评级）。
    _br_ok = (br[~br["行业"].isin(_SMALL)]
              if (not br.empty and _SMALL) else br)

    # =====================================================================
    # A. 本页结论
    # =====================================================================
    _pct = float(pos_hist["中位"].rank(pct=True).iloc[-1] * 100) if not pos_hist.empty else np.nan

    # ------------------------------------------------------------------
    # 结论先行：读者第一屏就要知道「这一季发生了什么」
    # 四条全部由下面的图算出，不引入任何新口径（同一批数字，只是先给判断）
    # ------------------------------------------------------------------
    _pos_now = float(pos.get("仓位1", np.nan)) if pos else np.nan
    _pos_chg = float(pos.get("中位变化pp", np.nan)) if pos else np.nan
    _buy_pct = float(pos.get("加仓基金占比", np.nan)) if pos else np.nan
    _act = dec["主动贡献"] if not dec.empty else pd.Series(dtype=float)
    _pas = dec["被动贡献"] if not dec.empty else pd.Series(dtype=float)
    _chg = dec["变化"] if not dec.empty else pd.Series(dtype=float)
    _act_buy = float(_act[_act > 0].sum()) if len(_act) else np.nan
    _n_cons = int((_br_ok["判读"] == "共识形成").sum()) if not _br_ok.empty else 0
    _n_out = int((_br_ok["判读"] == "系统性撤离").sum()) if not _br_ok.empty else 0
    _top_add = (_br_ok.sort_values("平均主动pp", ascending=False).head(2)
                if not _br_ok.empty else pd.DataFrame())
    _pos_word = ("加了杠杆" if _pos_chg >= 0.5 else
                 "降了仓位" if _pos_chg <= -0.5 else "基本没动")
    _s1 = (f'总仓位 <b>{_pos_now:.1f}%</b>，环比 <b>{_pos_chg:+.1f}pp</b>'
           f'（{_buy_pct:.0f}% 的基金在加）→ <b>{_pos_word}</b>：'
           + ("这是增量资金入场，仓位本身就在抬升。"
              if _pos_chg >= 0.5 else
              "这是降仓，先看是主动减还是净值跌出来的（图 2 拆）。"
              if _pos_chg <= -0.5 else
              "存量换仓，不是增量入场——行业间的钱是「搬」过去的，不是「新」进来的。"))
    # 与 ⑥ 方法页用同一个数（31 个行业先平均再取绝对值），否则读者会看到两个价格占比
    _pe: dict = {}
    _pe_note = "与 ⑥ 同口径"
    try:
        _pe = cn.price_effect() or {}
        _pe_txt = (f'行业层面价格漂移平均占 |原始变化| 的 <b>{_pe["share"]:.0f}%</b>'
                   if "share" in _pe else "价格漂移对配置变化影响很大")
        if "share" in _pe:
            _pe_note = (f'单季 {_pe["q0"]}→{_pe["q"]}，31 个行业先平均再取绝对值，'
                        f'与 ⑥ 同口径')
    except Exception:  # noqa: BLE001
        _pe_txt = "价格漂移对配置变化影响很大"
    _s2 = (f'31 个行业里，主动买入合计 <b>{_act_buy:.1f}pp</b>；{_pe_txt}'
           f'（{_pe_note}）→ <b>先扣价格，再谈加减仓</b>：'
           f'某行业占比下降，可能只是它跌了。')
    _s3 = (f'{len(_br_ok)} 个可评级行业中，<b>{_n_cons} 个「共识形成」、'
           f'{_n_out} 个「系统性撤离」</b>；'
           + (f'加得最齐的是 <b>{_top_add.iloc[0]["行业"]}</b>'
              f'（{_top_add.iloc[0]["加仓占比"]:.0f}% 在加）' if len(_top_add) else "")
           + (f'、<b>{_top_add.iloc[1]["行业"]}</b>'
              f'（{_top_add.iloc[1]["加仓占比"]:.0f}%）' if len(_top_add) > 1 else "")
           + f'（图 4）；共识是<b>局部</b>的，不是全面加仓。')
    # 「谁占主导」必须由数据决定：价格占比 38% 时说「价格比主动更大」就是错话
    _dom = ("价格贡献比主动买卖更大" if np.isfinite(_pe.get("share", np.nan))
            and _pe["share"] >= 50 else "主动买卖是主导，价格只是配角")
    _s4 = (f"<b>这一季是「结构性换仓」——仓位不动、钱在行业间搬，{_dom}。</b>"
           "所以别用「XX 行业被加仓了」当买入理由，"
           "要看主动项（扣掉价格）和广度（多少人在加）。")
    _items = [("①", "仓位", _s1), ("②", "主动 vs 价格", _s2),
              ("③", "行业共识", _s3), ("④", "怎么用", _s4)]
    h('<div style="background:#FFFFFF;border:1px solid ' + C.COLOR["line"]
      + ';border-left:4px solid ' + C.COLOR["price"] + ';border-radius:10px;'
      'padding:11px 15px 12px;margin:2px 0 12px">'
      + f'<div style="font-size:13.5px;font-weight:700;color:#1B2A3A;margin-bottom:7px">'
        f'先说结论（{q}，样本 {n_funds:,} 只）</div>'
      + '<div style="display:flex;flex-wrap:wrap;gap:10px 22px">'
      + "".join(
          f'<div style="flex:1 1 46%;min-width:340px;display:flex;gap:8px">'
          f'<div style="flex:0 0 20px;font-size:12px;font-weight:700;color:'
          f'{C.COLOR["price"]};padding-top:1px">{n}</div>'
          f'<div style="font-size:12.5px;line-height:1.85;color:#26374A">'
          f'<b>{t}：</b>{txt}</div></div>'
          for n, t, txt in _items)
      + '</div></div>')

    h(f'<div class="sec">二、本页导读 · {q}'
      f'<span style="font-weight:400;font-size:12.5px;color:{C.COLOR["text_dim"]};'
      f'margin-left:8px">样本 {n_funds:,} 只主动权益基金</span></div>')
    # 这一页是「证据页」，不是第二个结论页：① 已经给出判断与动作，
    # 这里只回答四个可复算的问题，避免同一句话在两页各写一遍。
    _q4 = [("①", "总仓位动了吗", "存量换仓还是增量入场"),
           ("②", "配置变化里多少是价格", "把被动漂移从主动买卖里剥出来"),
           ("③", "共识落在哪些行业", "多少人加 vs 加进去多少钱"),
           ("④", "个股层换了谁", "行业性配置还是选股")]
    h('<div style="display:flex;flex-wrap:wrap;gap:8px;margin:2px 0 10px">'
      + "".join(
          f'<div style="flex:1 1 190px;min-width:170px;background:#FFFFFF;'
          f'border:1px solid {C.COLOR["line"]};border-radius:9px;padding:8px 11px">'
          f'<div style="font-size:11px;color:{C.COLOR["text_dim"]};font-weight:700">'
          f'问题 {n}</div>'
          f'<div style="font-size:13px;font-weight:700;color:#1B2A3A;line-height:1.5;'
          f'margin:1px 0">{t}</div>'
          f'<div style="font-size:11px;color:{C.COLOR["text_dim"]};line-height:1.55">'
          f'{sub}</div></div>'
          for n, t, sub in _q4)
      + '</div>')
    h(_kpis([
        ("最新季报", q, f"全持仓期 {pf[1] if pf else '—'}"),
        ("样本基金", f"{n_funds:,}", "主动权益，剔主题基金"),
        ("股票仓位", f'{pos.get("仓位1", float("nan")):.1f}%' if pos else "—",
         f"{_npos} 期第 {_pct:.0f} 分位" if np.isfinite(_pct) else ""),
        ("共识 / 撤离", f"{int((_br_ok['判读'] == '共识形成').sum())} / "
                        f"{int((_br_ok['判读'] == '系统性撤离').sum())}" if not br.empty else "—",
         "行业数，见第 4 张图"),
    ]))
    h(f'<div class="tiny" style="margin-top:4px">判断、动作与仓位在「① 结论总览」，'
      f'本页只给证据与推导，两页共用同一套数字。'
      f'单只基金的季报原文见「④ 单只基金」——那是<b>一只基金</b>，这里是 '
      f'{n_funds:,} 只的横截面，样本不同、不重复。</div>')
    # 同一页四张图四个样本：不写清楚，读者会拿「3,923」去质疑「4,270」。
    _npos_n = int(pos_hist["基金数"].iloc[-1]) if not pos_hist.empty else 0
    _ndec_n = int(dec.attrs.get("n_funds", 0)) if not dec.empty else 0
    _nbr_n = int(br.attrs.get("n_funds", 0)) if not br.empty else 0
    h('<div class="tiny" style="margin:2px 0 6px">'
      f'<b>四组样本不同，不能互相套用</b>：仓位 {_npos_n:,} 只（股票仓位 ≥40%）；'
      f'半年报主动/被动 {_ndec_n:,} 只（两期都可比）；广度 {_nbr_n:,} 只（当期有实质变动）；'
      f'个股 {n_funds:,} 只（全部主动权益）。差异来自「哪些基金在两个时点都有可比数据」，'
      f'不是漏数。</div>')
    st.write("")

    # =====================================================================
    # 图1 仓位
    # =====================================================================
    if pos:
        h('<div class="sec">三、六张图看懂这一季</div>')
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
        _buy = float(pos["加仓基金占比"])
        # 41/59 不能写成「几乎各占一半」：写清楚偏哪边，读者才不会被措辞误导。
        _bal = ("接近对半" if 45 <= _buy <= 55
                else ("加仓略多" if _buy > 55 else "减仓略多"))
        _read(h,
              f'{_npos} 个半年报期的仓位中位在 {_pmin:.1f}%–{_pmax:.1f}% 之间，'
              f'当前 <b>{pos["仓位1"]:.1f}%</b>，处于第 <b>{_pct:.0f}</b> 分位；'
              f'本期只动了 <b>{pos["中位变化pp"]:+.2f}pp</b>，'
              f'{_buy:.0f}% 加仓、{100 - _buy:.0f}% 减仓，{_bal}；'
              f'中间 50% 的基金落在 {_pq1:.0f}%–{_pq3:.0f}% 之间。',
              f'股票型有 80% 仓位下限、灵活配置还要留现金，{_npos} 期极差只有 '
              f'{_pmax - _pmin:.1f}pp——「看好」很少表现为整体加仓，而是行业间再分配。'
              f'当前第 {_pct:.0f} 分位，增量资金有限，收益只能来自结构。',
              f'高分位时把仓位集中给「可以跟」的行业，不平均撒。')

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
              f'机构主动只有 <b>{_act:.1f}pp</b>'
              f'（两者都是 31 个行业 |贡献| 的合计，被动是主动的 '
              f'<b>{_pas / max(_act, 1e-9):.1f} 倍</b>；这是半年报口径，'
              f'不要和结论里的单季占比直接比）。'
              f'最典型的是 <b>{_topname}</b>：变化 {_t["变化"]:+.2f}pp，'
              f'其中价格 {_t["被动贡献"]:+.2f}pp、机构 {_t["主动贡献"]:+.2f}pp；'
              f'机构主动加最多的是 <b>{_a_max}（{dec.loc[_a_max, "主动贡献"]:+.2f}pp）</b>、'
              f'减最多的是 <b>{_a_min}（{dec.loc[_a_min, "主动贡献"]:+.2f}pp）</b>。',
              f'不交易权重也会变：某行业涨 20%、其他不变，占比会自动升约 2pp。'
              f'混在一起看，会把「行业涨了」误读成「机构在买」；'
              f'扣掉价格后剩下的<b>主动项</b>，才等同于「净买入」。',
              f'看到配置暴跌先看主动项：主动≈0 只是跌出来的，不要跟着砍。')

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
          f'{comp.attrs["q_prev"]} 与 {comp.attrs["q_year"]}），<b>含价格漂移</b>（全样本合计，两期基金不完全相同）——'
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
              f'环比混着季度再平衡与申赎、噪音大；同比抹掉季节性，更接近'
              f'「这一年机构信了什么」。两者同向＝持续行为，一正一负＝季度摆动。',
              f'底仓看同比 + 广度，交易看环比 + 当期主动；方向相反的行业只做交易、不做底仓。'
              f'两栏都含价格漂移，是「结果」不是「动作」。')

    # =====================================================================
    # 图4 广度
    # =====================================================================
    if not br.empty:
        h('<div style="font-size:12px;font-weight:700;margin:10px 0 4px">'
          f'图 4　谁在动：{pq[0]} → {pq[1]} 的前十大口径（季频、价格中性）</div>')
        # 31 个行业的标签在 600px 里必然叠字（截图反馈过）：两张图改为通栏上下排，
        # 条形图看「谁偏离 50% 最远」，散点看「共识与力度是否同向」。
        st.plotly_chart(rc.fig_breadth(br, 14, _SMALL), width="stretch",
                        config={"displayModeBar": False})
        st.plotly_chart(rc.fig_breadth_vs_move(br, _SMALL), width="stretch",
                        config={"displayModeBar": False})
        h(f'<div class="tiny">口径：前十大归一化，逐基金算<b>价格中性</b>变化'
          f'（假设不交易时权重随行业指数漂移，实际与反事实之差才是交易）；'
          f'单只基金变化超过 ±0.1pp 才计数。'
          f'加仓占比的分母是<b>加仓 + 减仓</b>（有实质变动的基金），不是全部样本。'
          f'样本 {br.attrs["n_funds"]:,} 只。</div>')
        # 广度排名要剔除小样本：几十只基金里 3 只加仓就能把占比推到 80%，
        # 也可能因为 1 只减仓把「最低」安到一个只有 0.1% 仓位的行业上。
        # 图里已经把这些行业标灰，文字却拿它们当「最高/最低」，会自相矛盾。
        _br_ok = br[~br["行业"].isin(_SMALL)] if _SMALL else br
        _hi = _br_ok.loc[_br_ok["加仓占比"].idxmax()]
        _lo = _br_ok.loc[_br_ok["加仓占比"].idxmin()]
        _lft = _br_ok[(_br_ok["加仓占比"] < 50) & (_br_ok["平均主动pp"] > 0)]
        _rgt = _br_ok[(_br_ok["加仓占比"] >= 60) & (_br_ok["平均主动pp"] < 0)]
        _read(h,
              f'{_hi["行业"]} 加仓占比最高（{_hi["加仓占比"]:.0f}%：'
              f'{_hi["加仓基金数"]:.0f} 加 / {_hi["减仓基金数"]:.0f} 减），'
              f'{_lo["行业"]} 最低（{_lo["加仓占比"]:.0f}% 在加）。'
              + (f'背离 {len(_lft) + len(_rgt)} 个（少数大基金撑量 / 小基金接盘）。'
                 if (len(_lft) or len(_rgt)) else '两类背离都没出现。'),
              f'广度答「多少人在加」，净额答「加了多少钱」，规模差异让两者分离。'
              f'广度还是滞后指标：涨了才敢加，冲到 70% 以上时边际买家已在减少。',
              f'只跟<b>右上角</b>；右下角（大资金撤、小基金接）是派发，左上角只观察。')

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
                # 图 5 已经把「加最多 / 减最多」画出来了，这里再列同一批股票
                # 就是重复。表只留图上没有的信息：**环比与同比方向相反**的股票——
                # 它们暴露了持仓周期的切换，也是正文里「最需要警惕」的那一类。
                _conf = _cmp[(_cmp["环比Δ"] * _cmp["同比Δ"]) < 0].copy()
                if len(_conf):
                    _conf["_abs"] = _conf["同比Δ"].abs()
                    _conf = _conf.sort_values("_abs", ascending=False).drop(columns="_abs")
                    _conf.index.name = "代码"
                    with st.expander(
                            f"环比与同比打架的 {len(_conf)} 只：一个时间尺度在加、另一个在减",
                            expanded=False):
                        st.dataframe(_conf.reset_index(), width="stretch",
                                     hide_index=True, height=min(320, 40 + 33 * len(_conf)),
                                     column_config={
                                         "本期基金数": st.column_config.NumberColumn(format="%.0f"),
                                         "环比Δ": st.column_config.NumberColumn(format="%+.0f"),
                                         "同比Δ": st.column_config.NumberColumn(
                                             format="%+.0f",
                                             help="与上年同季比，正数＝这一年被更多基金持有"),
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
                      f'被几百只基金同时加进来，价格里已含这部分共识，追买赔率不好；'
                      f'真正有价值的是环比与同比打架的股票。',
                      f'环比大正/同比大负＝刚被买进来的新宠，只能小仓试；'
                      f'环比负/同比大正＝这一年被加过、现在开始撤，持有的应减仓。')
                # 行业构成
                h('<div style="font-size:12px;font-weight:700;margin:8px 0 3px">'
                  f'图 6　{_pair[0]}→{_pair[1]} 加仓/减仓前 50 只的行业构成</div>')
                st.plotly_chart(rc.fig_stock_industry(sf_f, 50, *_pair), width="stretch",
                                config={"displayModeBar": False})
                _top50a = sf_f.nlargest(50, "d_n")["industry"].value_counts()
                _top50c = sf_f.nsmallest(50, "d_n")["industry"].value_counts()
                # 「港股」不是申万一级行业，是未映射行业的港股通持仓；
                # 直接写「其次是港股」会让人以为港股是一个行业。
                _lab = lambda x: ("港股（未映射行业）" if str(x) == "港股" else str(x))
                _read(h,
                      f'加仓前 50 只里，<b>{_lab(_top50a.index[0])}</b> 占 '
                      f'{_top50a.iloc[0]} 只（{_top50a.iloc[0] / 50 * 100:.0f}%），'
                      f'其次是 {_lab(_top50a.index[1]) if len(_top50a) > 1 else "—"} '
                      f'{_top50a.iloc[1] if len(_top50a) > 1 else 0} 只；'
                      f'减仓前 50 只里最多的是 <b>{_lab(_top50c.index[0])}</b> '
                      f'{_top50c.iloc[0]} 只。'
                      f'（港股＝未映射申万一级的港股通持仓，单列一行，不并入 A 股行业。）',
                      f'集中在少数行业＝行业配置，个股只是载体，买 ETF 更省事；'
                      f'分散在多个行业＝选股，才值得逐只研究基本面。',
                      f'加仓集中在某行业时，买该行业 ETF 更省事；分散时再逐只研究。')
    except Exception as _e:
        h(f'<div class="tiny">个股榜暂不可用：{type(_e).__name__}</div>')

    # =====================================================================
    # C. 行业台账
    # =====================================================================
    if not comp.empty:
        h('<div class="sec">四、行业台账：一行一个行业，水平 + 环比 + 同比 + 主动/被动</div>')
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
        with st.expander("行业台账：31 行 × 12 列（可排序，写在正文会占 560px）",
                     expanded=False):
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
    h('<div class="sec">五、本页特有的两条边界</div>')
    h('<div class="cg-note" style="line-height:1.9">'
      '· <b>主动/被动只在半年报做</b>：季报的前十大每季换人，逐季做 GK 分解噪声极大，'
      '本库不做这种「看起来更精细、实际更不可靠」的事。<br>'
      '· <b>图 2 与图 3 不是同一段时间</b>：图 2 是上一个半年（全持仓），'
      '图 3 起是本季环比（前十大）；两张图的数字不能直接相减。<br>'
      '其余边界（季报只有前十大、3–4 周时滞、幸存者偏差）见「① 结论总览 · 四条硬约束」'
      '与「⑥ 方法与口径 · 三套口径」，本页不重复。</div>')

    ui.source_footer(h, "report", "gk_active")
