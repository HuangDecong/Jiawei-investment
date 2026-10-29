# -*- coding: utf-8 -*-
"""「本期解读」页：一页话讲清「发生了什么 → 为什么 → 怎么做」。

写作原则
--------
1. **结论先行**：页面最上面就是判断与操作建议，读者不用往下翻才知道要看什么；
2. **每张图配「读图 + 结论」**：读图说这张图在讲什么，结论说所以该做什么；
3. **每个数字标注口径与期次**：全持仓（半年）与前十大（季度）不混着说；
4. **只写能证伪的判断**：不写「值得关注」「密切跟踪」，每条建议都带触发条件。

四段结构
--------
    A 结论与操作（顶栏）   B 五张图 + 解读 + 结论   C 行业台账   D 方法与免责
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import canon as cn
import config as C
import report_charts as rc


def _fmt(v, unit="", nd=1, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    # 绝对值小于 0.1 时自动多给一位小数：否则 -0.04 会显示成「-0.0」，
    # 读者会以为真的是零，而实际上这是一个方向明确的减仓。
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


def _box(h, title: str, body: str, color: str) -> None:
    h(f'<div style="border:1px solid {C.COLOR["line"]};border-top:3px solid {color};'
      f'border-radius:6px;padding:9px 11px;height:100%">'
      f'<div style="font-size:12px;font-weight:700;color:{color};margin-bottom:5px">{title}</div>'
      f'<div style="font-size:11.5px;line-height:1.8">{body}</div></div>')


def _names(df: pd.DataFrame, n: int = 3) -> str:
    return "、".join(list(df.index)[:n]) if len(df) else "无"


def _advice(comp: pd.DataFrame, pos: pd.DataFrame, br: pd.DataFrame) -> dict:
    """把三条规则算成一张操作清单。规则事前固定，不看结果调参。

    · 超配：机构主动加（≥0.3pp）＋ 多数基金也在加（≥60%）＋ 还没到历史高位（≤80）
    · 不追：已到历史高位（≥90），不管主动项多好——拥挤本身就是风险
    · 回避：机构主动减（≤−0.5pp）＋ 广度不足（<60% 在加）
    · 观察：主动为正但广度<50%，少数大基金在撑，等一季确认
    """
    pct = float(pos["中位"].rank(pct=True).iloc[-1] * 100)
    div = float(br["分歧度"].median()) if not br.empty else np.nan

    # 仓位区间：仓位越靠历史高位、分歧越大，给的区间越低
    if pct >= 85:
        lo, hi, stance = 55, 65, "偏高（历史高位）"
    elif pct >= 70:
        lo, hi, stance = 65, 75, "中性偏高"
    elif pct >= 30:
        lo, hi, stance = 70, 80, "中性"
    else:
        lo, hi, stance = 75, 85, "偏低（历史低位）"
    base = (lo, hi)
    cut = bool(np.isfinite(div) and div > 0.65 and pct >= 70)
    if cut:
        lo, hi = lo - 5, hi - 5
        stance += "，且分歧大→再降 5pp"

    buy = comp[(comp["主动pp"] >= 0.3) & (comp["加仓占比"] >= 60) &
               (comp["历史分位"] <= 80)].sort_values("主动pp", ascending=False)
    hot = comp[(comp["历史分位"] >= 90) & ~comp.index.isin(buy.index)].sort_values(
        "历史分位", ascending=False)
    avoid = comp[(comp["主动pp"] <= -0.5) & (comp["加仓占比"] < 60)].sort_values(
        "主动pp")
    watch = comp[(comp["主动pp"] > 0) & (comp["加仓占比"] < 50) &
                 ~comp.index.isin(buy.index)].sort_values("主动pp", ascending=False)
    return dict(pct=pct, div=div, lo=lo, hi=hi, stance=stance, base=base, cut=cut,
                buy=buy, hot=hot, avoid=avoid, watch=watch)


def _read_row(r) -> str:
    """把「机构主动方向」与「加仓广度」合成一句读数，两者矛盾时直说。"""
    a = r.get("主动pp", np.nan)
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
    adv = _advice(comp, pos_hist, br) if not comp.empty else None

    # ============================================================
    # A. 结论与操作（顶栏）
    # ============================================================
    h(f'<div class="sec">一、先给结论 · {q} 主动权益基金季报（样本 {n_funds:,} 只）</div>')

    _concl = []
    if adv:
        _concl.append(
            f'<b>仓位在历史高位、但机构没加仓。</b>股票仓位中位 '
            f'{pos["仓位0"]:.1f}% → {pos["仓位1"]:.1f}%，'
            f'只有 {pos["中位变化pp"]:+.2f}pp，处在 {len(pos_hist)} 个半年报期的第 '
            f'<b>{adv["pct"]:.0f} 分位</b>。高仓位意味着<b>后续增量资金有限</b>，'
            f'行业之间的钱是「挪」出来的，不是「新」进来的。')
    if not dec.empty:
        _u, _d = dec["主动贡献"].idxmax(), dec["主动贡献"].idxmin()
        _concl.append(
            f'<b>主动净买入最多的是 {_u}（{dec.loc[_u, "主动贡献"]:+.2f}pp），'
            f'主动净卖出最多的是 {_d}（{dec.loc[_d, "主动贡献"]:+.2f}pp）。</b>'
            f'注意区分：<b>配置比例变化大 ≠ 机构在动手</b>，'
            f'电子半年变化 {dec.loc["电子", "变化"]:+.2f}pp 里，'
            f'{dec.loc["电子", "被动贡献"]:+.2f}pp 是价格涨出来的，'
            f'机构只动了 {dec.loc["电子", "主动贡献"]:+.2f}pp。'
            if "电子" in dec.index else '')
    if not br.empty:
        _vc = br["判读"].value_counts()
        _concl.append(
            f'<b>共识集中在少数行业，多数行业在多空对撞。</b>'
            + "、".join(f"{k} {v} 个" for k, v in _vc.items())
            + f'。<b>只有「多数基金在加 + 净额为正」才是共识</b>，'
              f'加仓基金不到一半却净额为正的，是少数大基金在撑，最容易反转。')

    if _concl:
        h('<div class="cg-note" style="font-size:13px;line-height:1.95">'
          + "<br>".join(f"{i}. {x}" for i, x in enumerate(_concl, 1)) + "</div>")

    h(_kpis([
        ("最新季报", q, f"全持仓期 {pf[1] if pf else '—'}"),
        ("样本基金", f"{n_funds:,}", "主动权益，剔主题基金"),
        ("股票仓位", _fmt(pos.get("仓位1"), "%", 1) if pos else "—",
         f"历史第 {adv['pct']:.0f} 分位" if adv else ""),
        ("共识 / 对撞", f"{int((br['判读'] == '共识形成').sum())} / "
                        f"{int((br['判读'] == '多空对撞').sum())}" if not br.empty else "—",
         "行业数，见第 4 张图"),
    ]))

    st.write("")

    # ---------------- 二、投资建议 ----------------
    if adv:
        _ind = cn.index_stats()
        _adv = comp.copy()
        if not _ind.empty:
            _adv = _adv.join(_ind[["指数分位", "近3月涨幅%"]], how="left")
        h('<div class="sec">二、投资建议：买什么、卖什么、为什么</div>')
        h('<div class="tiny" style="margin-bottom:6px">'
          '三条筛选条件事前固定，全部可复核：'
          '<b>①机构主动</b>（剔除价格后的真实加减仓）'
          '<b>②加仓广度</b>（多少基金在动手）'
          '<b>③价格分位</b>（这个行业相对自己历史贵不贵）。'
          '只有三条同时通过的才叫「可以跟」。</div>')

        _buy = _adv[(_adv["主动pp"] >= 0.3) & (_adv["加仓占比"] >= 60)]
        if "指数分位" in _buy:
            _buy = _buy[_buy["指数分位"].fillna(50) < 75]
        _buy = _buy.sort_values("主动pp", ascending=False)

        _out = _adv[(_adv["主动pp"] <= -0.3) & (_adv["加仓占比"] <= 45)].sort_values("主动pp")

        _crowd = pd.DataFrame()
        if "指数分位" in _adv:
            _crowd = _adv[(_adv["历史分位"] >= 80) & (_adv["指数分位"] >= 85)].sort_values(
                "指数分位", ascending=False)

        _wait = _adv[(_adv["主动pp"] > 0) & (_adv["加仓占比"] < 50)].sort_values(
            "主动pp", ascending=False)

        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            _txt = "<br>".join(
                f'· <b>{i}</b>　主动 {r["主动pp"]:+.2f}、{r["加仓占比"]:.0f}% 在加、'
                f'指数 {r.get("指数分位", float("nan")):.0f} 分位'
                for i, r in _buy.head(4).iterrows()) or "本期一个都没有"
            _box(h, f"① 可以跟（{len(_buy)} 个）", _txt, C.COLOR["up"])
        with c2:
            _txt = "<br>".join(
                f'· <b>{i}</b>　主动 {r["主动pp"]:+.2f}、仅 {r["加仓占比"]:.0f}% 在加'
                for i, r in _out.head(4).iterrows()) or "本期无"
            _box(h, f"② 要回避（{len(_out)} 个）", _txt, C.COLOR["down"])
        with c3:
            _txt = "<br>".join(
                f'· <b>{i}</b>　配置 {r["历史分位"]:.0f} 分位、指数 '
                f'{r["指数分位"]:.0f} 分位、近3月 {r["近3月涨幅%"]:+.0f}%'
                for i, r in _crowd.head(4).iterrows()) or "本期无"
            _box(h, f"③ 别追高（{len(_crowd)} 个）", _txt, C.COLOR["level_拥挤"])
        with c4:
            _txt = "<br>".join(
                f'· <b>{i}</b>　主动 {r["主动pp"]:+.2f}，但只有 {r["加仓占比"]:.0f}% 在加'
                for i, r in _wait.head(4).iterrows()) or "本期无"
            _box(h, f"④ 等确认（{len(_wait)} 个）", _txt, C.COLOR["text_dim"])

        h(f'<div class="cg-note" style="margin-top:10px">'
          f'<b>为什么这样分</b>：'
          f'「可以跟」＝机构真金白银在加（主动项为正）<b>且</b>多数基金跟着加（广度≥60%）'
          f'<b>且</b>价格还没到历史极值（指数分位<75），三条缺一条就降级；'
          f'「要回避」＝主动项为负、广度也不足，<b>不要因为跌得多就去接</b>；'
          f'「别追高」＝配置和价格都在历史高位，此时利好大多已反映在持仓里，'
          f'赔率不够，不是看空；'
          f'「等确认」＝净额为正但广度不足，是<b>少数大基金在撑</b>，等下一季主动项还在才跟。'
          f'</div>')

    # ---------------- 三、操作建议 ----------------
    if adv:
        h('<div class="sec">三、操作建议：怎么执行、什么时候推翻</div>')
        c1, c2 = st.columns([0.34, 0.66], gap="small")
        with c1:
            _box(h, "总仓位", f'权益仓位参考 <b>{adv["lo"]}%–{adv["hi"]}%</b><br>'
                              f'算法：仓位 {adv["pct"]:.0f} 分位 → 基础区间 '
                              f'{adv["base"][0]}–{adv["base"][1]}%；'
                              f'分歧度中位 {adv["div"]:.2f}'
                              f'（{">0.65，再降 5pp" if adv["cut"] else "≤0.65，不调整"}）'
                              f' → <b>{adv["lo"]}–{adv["hi"]}%</b>。<br>'
                              f'<span style="color:{C.COLOR["text_dim"]}">'
                              f'不是说必须这么配，而是提醒：现在整体加仓的赔率不高。</span>',
                 C.COLOR["price"])
        with c2:
            # 注意：DataFrame 不渲染 HTML，条件里不能写 <b>，否则页面上直接显示标签
            _rules = pd.DataFrame([
                dict(动作="买入", 触发条件="主动项连续 2 期为正 且 广度>60% 且 指数分位<70",
                     执行="单行业 ≤ 权益仓位 15%，分 3 批建"),
                dict(动作="卖出", 触发条件="主动项连续 2 期为负，或 广度<40% 且 指数分位>80",
                     执行="先减半，下一季仍不改善再清"),
                dict(动作="止损", 触发条件="行业指数自建仓回撤>15% 且 主动项转负",
                     执行="减半，不等反弹"),
                dict(动作="推翻", 触发条件="下季主动项反转，或样本基金数减少 20% 以上",
                     执行="重新跑一次本页"),
            ])
            st.dataframe(_rules, width="stretch", hide_index=True, height=200,
                         column_config={
                             "动作": st.column_config.TextColumn(width="small"),
                             "触发条件": st.column_config.TextColumn(width="large"),
                             "执行": st.column_config.TextColumn(width="medium")})
        h('<div class="tiny">规则写死在代码里（<span class="mono">report_page._advice</span>），'
          '不看结果调参；换一季数据会自动重算，你只需要对着条件核对有没有触发。</div>')

    # ============================================================
    # B. 图 + 解读 + 结论
    # ============================================================
    # ---------- 图1 仓位 ----------
    if pos:
        h('<div class="sec">三、五张图看懂这一季</div>')
        h('<div style="font-size:12px;font-weight:700;margin:2px 0 4px">'
          '图 1　仓位：把 15 个半年报期连起来，才知道现在是高还是低</div>')
        _c = cn.coverage(pos["q"])
        _idx = _c.index.intersection(list(cn.universe(pos["q"])))
        st.plotly_chart(rc.fig_position(pos_hist, _c.loc[_idx].dropna(), pos["q"]),
                        width="stretch", config={"displayModeBar": False})
        _p = adv["pct"] if adv else np.nan
        _read(h,
              f'左图是 {len(pos_hist)} 个半年报期的仓位中位数（阴影是 25–75 分位带），'
              f'右图是本期 {len(_idx):,} 只基金的仓位分布。'
              f'当前中位 <b>{pos["仓位1"]:.1f}%</b>，位于历史第 <b>{_p:.0f}</b> 分位，'
              f'本期变化 <b>{pos["中位变化pp"]:+.2f}pp</b>，'
              f'{pos["加仓基金占比"]:.0f}% 的基金加仓、{100 - pos["加仓基金占比"]:.0f}% 减仓。',
              f'仓位是「总开关」：<b>整体没动，说明下面的行业变化是内部换仓，不是新增资金</b>。'
              f'这一条决定了后面的读法——某行业配置上升，可以放心读成「机构更看好它」，'
              f'而不是「大盘涨了、水涨船高」。反过来，仓位已在 '
              f'{_p:.0f} 分位的高位，<b>指望「全体加仓推动普涨」是不现实的</b>。')

    # ---------- 图2 主动/被动 ----------
    if not dec.empty:
        h('<div style="font-size:12px;font-weight:700;margin:10px 0 4px">'
          '图 2　配置变化拆开看：灰色是价格涨出来的，彩色才是机构动手</div>')
        st.plotly_chart(rc.fig_active_passive(dec), width="stretch",
                        config={"displayModeBar": False})
        _pas = float(dec["被动贡献"].abs().sum())
        _act = float(dec["主动贡献"].abs().sum())
        _topname = dec["被动贡献"].abs().idxmax()
        _t = dec.loc[_topname]
        _read(h,
              f'把 31 个行业的配置变化拆成两半：价格带来的被动变化合计 <b>{_pas:.1f}pp</b>，'
              f'机构主动只占 <b>{_act:.1f}pp</b>，被动占了全部变动的 '
              f'<b>{_pas / (_pas + _act) * 100:.0f}%</b>。'
              f'最典型的是 <b>{_topname}</b>：变化 {_t["变化"]:+.2f}pp，'
              f'其中价格 {_t["被动贡献"]:+.2f}pp、机构 {_t["主动贡献"]:+.2f}pp。',
              f'看到「某行业配置暴跌」先别当利空——<b>跌幅大但主动项接近 0 的，'
              f'基本面含义接近于零</b>，只是那个行业跌了。真正该重视的是'
              f'<b>主动项绝对值大、且与广度同向</b>的行业；'
              f'反过来，一个行业配置涨了 10pp 但机构没动，说明它是被「涨」上去的，'
              f'接下来回落时配置也会自己掉下来，不需要卖。')

    # ---------- 图3 环比/同比 ----------
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
        _read(h,
              f'左栏是本季对上一季，右栏是本季对去年同季。'
              f'<b>方向一致的有「连续加仓」{len(_su)} 个</b>（'
              + ("、".join(_su.index) or "无")
              + f'）与<b>「连续减仓」{len(_sd)} 个</b>（'
              + ("、".join(_sd.index) or "无")
              + f'）；<b>方向相反的 {len(_opp)} 个</b>，'
                f'比如 ' + ("、".join(
                    f'{i}（环比 {r["Δ上季pp"]:+.2f}、同比 {r["Δ上年pp"]:+.2f}）'
                    for i, r in _opp.head(3).iterrows()) or "无") + '。',
              f'<b>方向一致才是趋势，方向相反只是摆动。</b>'
              f'环比说「在加」、同比说「在减」的行业，是短周期调仓，'
              f'只能做交易、不能做底仓；反过来，连续两期都在加的行业，'
              f'才值得按年度维度配置。只看环比会被季度噪音带偏，只看同比会错过拐点——'
              f'两条一起看，才知道谁是「刚启动」、谁是「已经走了一年」。')

    # ---------- 图4 广度 ----------
    if not br.empty:
        h('<div style="font-size:12px;font-weight:700;margin:10px 0 4px">'
          f'图 4　谁在动：{pq[0]} → {pq[1]} 的前十大口径（季频、跨季可比）</div>')
        _c1, _c2 = st.columns([0.56, 0.44], gap="small")
        with _c1:
            st.plotly_chart(rc.fig_breadth(br), width="stretch",
                            config={"displayModeBar": False})
        with _c2:
            st.plotly_chart(rc.fig_breadth_vs_move(br), width="stretch",
                            config={"displayModeBar": False})
        _hi = br.loc[br["加仓占比"].idxmax()]
        _lo = br.loc[br["加仓占比"].idxmin()]
        _lft = br[(br["加仓占比"] < 50) & (br["平均变化pp"] > 0)]
        _read(h,
              f'左图按「加仓基金占比」挑出最偏离 50% 的 14 个行业：'
              f'<b>{_hi["行业"]}</b> 最高（持有它的 {int(_hi["样本基金数"]):,} 只里 '
              f'{_hi["加仓占比"]:.0f}% 在加），<b>{_lo["行业"]}</b> 最低'
              f'（{_lo["加仓占比"]:.0f}% 在加）。右图把「多少基金在加」和「净买了多少」'
              f'放在一起，左上角有 <b>{len(_lft)}</b> 个行业是「加仓基金不到一半、'
              f'但净额为正」——少数大基金在撑。',
              f'<b>广度是分辨「共识」与「假象」的唯一办法。</b>'
              f'同一个「平均加仓 2pp」，900 只基金各加 2pp 和 200 只大基金各加 9pp '
              f'含义完全相反：前者是共识、能持续，后者一停手就反转。'
              f'所以只跟右上方（广度高 + 净额正）的行业，'
              f'左上角那批只能观察，不能当信号。')

    # ---------- 图5 个股 ----------
    if ps:
        h('<div style="font-size:12px;font-weight:700;margin:10px 0 4px">'
          f'图 5　落到个股：{ps[0]} → {ps[1]}，谁被加进来、谁被移出去</div>')
        try:
            sf = cn.stock_flow(*ps)
            if not sf.empty:
                sf = sf[(sf["n_funds"] >= 10) | (sf["n_funds_0"] >= 10)]
                st.plotly_chart(rc.fig_stock_flow(sf), width="stretch",
                                config={"displayModeBar": False})
                _a, _c = sf.nlargest(3, "d_n"), sf.nsmallest(3, "d_n")
                _ind_a = _a["industry"].value_counts()
                _read(h,
                      f'被最多基金新买入：' + "、".join(
                          f'{r["stock_name"]}（{r["d_n"]:+.0f}，{r["industry"]}）'
                          for _, r in _a.iterrows())
                      + f'；被最多基金卖出：' + "、".join(
                          f'{r["stock_name"]}（{r["d_n"]:+.0f}，{r["industry"]}）'
                          for _, r in _c.iterrows())
                      + f'。口径是<b>同一披露类型的两期</b>（季报对上年同季），'
                        f'样本 {sf.attrs["n_funds"][1]:,} 只基金。',
                      f'个股榜要配着行业看：<b>如果加仓的个股集中在同一个行业，'
                      f'说明是行业配置行为；散在不同行业，才是真正的选股行为。</b>'
                      f'本期加仓前三里 {len(_ind_a)} 个行业、'
                      f'最大占比是 {_ind_a.index[0] if len(_ind_a) else "—"}，'
                      + ("说明这批买入更像<b>行业性加仓</b>，跟行业方向即可，"
                         "不必逐只研究。"
                         if len(_ind_a) <= 2 else
                         "说明更偏<b>个股选择</b>，需要逐只判断基本面。")
                      + '另外，个股层不能用相邻季对比——「从第 11 名掉出前十大」'
                        '会被误判成清仓。')
        except Exception as _e:
            h(f'<div class="tiny">个股榜暂不可用：{type(_e).__name__}</div>')

    # ============================================================
    # C. 行业台账
    # ============================================================
    if not comp.empty:
        h('<div class="sec">四、行业台账：一行一个行业，水平 + 环比 + 同比 + 主动/被动</div>')
        t = comp.copy()
        t["读数"] = t.apply(_read_row, axis=1)
        q_, qp_, qy_ = comp.attrs["q"], comp.attrs["q_prev"], comp.attrs["q_year"]
        qf_, qf0_ = comp.attrs["q_full"], comp.attrs["q_full0"]
        show = t.reset_index()[[
            "行业", "配置_全持仓", "主动pp", "被动pp", "配置_前十",
            "Δ上季pp", "Δ上年pp", "历史分位", "加仓占比", "判读", "读数"]]
        show = show.rename(columns={
            "配置_全持仓": f"配置%({qf_})", "主动pp": f"主动pp({qf0_}→{qf_})",
            "被动pp": f"被动pp({qf0_}→{qf_})", "配置_前十": f"前十配置%({q_})",
            "Δ上季pp": f"环比pp({qp_}→{q_})", "Δ上年pp": f"同比pp({qy_}→{q_})",
            "历史分位": "分位(31期)", "加仓占比": f"加仓占比%({q_})"})
        st.dataframe(
            show, width="stretch", hide_index=True, height=560,
            column_config={
                f"配置%({qf_})": st.column_config.NumberColumn(
                    format="%.2f", help="全持仓口径，占净值比；之和＝股票仓位"),
                f"主动pp({qf0_}→{qf_})": st.column_config.NumberColumn(
                    format="%+.2f", help="GK 分解后的机构主动部分（半年）"),
                f"被动pp({qf0_}→{qf_})": st.column_config.NumberColumn(
                    format="%+.2f", help="价格涨跌带来的部分（半年）"),
                f"前十配置%({q_})": st.column_config.NumberColumn(
                    format="%.2f", help="前十大归一化口径，31 期都有，用于环比/同比"),
                f"环比pp({qp_}→{q_})": st.column_config.NumberColumn(format="%+.2f"),
                f"同比pp({qy_}→{q_})": st.column_config.NumberColumn(format="%+.2f"),
                "分位(31期)": st.column_config.ProgressColumn(
                    format="%.0f", min_value=0, max_value=100,
                    help="该行业当前配置在自身 31 期历史中的排名"),
                f"加仓占比%({q_})": st.column_config.NumberColumn(
                    format="%.0f", help="分母是「该行业进入前十大」的基金数"),
                "判读": st.column_config.TextColumn(width="small"),
                "读数": st.column_config.TextColumn(
                    width="small",
                    help="真加仓＝机构主动加且多数基金在加；少数大基金加＝净额为正但广度不足；"
                         "真撤退＝主动减且多数在减；小基金接盘＝多数在加但净额为负；"
                         "方向冲突＝介于两者之间"),
            })
        h(f'<div class="tiny">三组列、三种期次，<b>不要相加</b>：'
          f'「配置/主动/被动」是<b>全持仓半年报</b>口径（{qf0_}→{qf_}），最准但一年两次；'
          f'「前十配置/环比/同比」是<b>前十大归一化</b>口径（{q_} 对 {qp_} 与 {qy_}），'
          f'31 期都有、跨季可比；「分位」用 31 期前十大口径算。'
          f'三套口径各回答一个问题，混着看必然矛盾。</div>')

    # ============================================================
    # D. 方法与免责
    # ============================================================
    h('<div class="sec">五、这套判断的边界（不写清楚就等于骗人）</div>')
    h('<div class="cg-note" style="line-height:1.9">'
      '· <b>它是描述，不是预测</b>：加仓占比对次季行业收益的 RankIC≈+0.05、p≈0.37（14 期），'
      '「拥挤即风险」在 30 期数据上也被证伪（低分位 −1.42% vs 高分位 +0.83%）。'
      '它能告诉你机构在做什么，不能告诉你接下来会涨什么。<br>'
      '· <b>季报有时滞</b>：季末到披露约 56 天，你看到时季度已过 62%。'
      '所有「本期」都是「上一个已披露期」，不是「今天」。<br>'
      '· <b>季报只有前十大</b>：Q1/Q3 每只基金只披露 10 只股票，'
      '所以季报期的配置一律用来算广度，不用来判断绝对水平。<br>'
      '· <b>操作建议是规则输出，不是投资建议</b>：四条规则事前固定、触发条件可验证，'
      '但你的风险预算、持有期限、税负都不在模型里。'
      '</div>')
