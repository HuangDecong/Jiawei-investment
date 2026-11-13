# -*- coding: utf-8 -*-
"""④ 单只基金 · 季报全解：一只基金、一个季度，季报里到底说了什么。

设计原则
--------
1. **只用披露值**。本页每个数字都能在季报原文里找到对应项；唯一用到恒等式的是
   「期初＝期末−申购+赎回」与「净资产＝份额×份额净值」，这是会计恒等式，不是估算。
   缺数据就写「本期季报未披露」，绝不拿前十大反推、覆盖率归一之类的替代口径顶上。
2. **季度不裁剪**。资产配置与份额由数据源一次返回全历史（老基金 40 期以上），
   所以选择器给全部季度；行业/持仓是按年抓的，翻到早年时现场补抓那一年，
   补过就落库，不会重复。
3. **一屏一个结论**。钱在哪、押什么行业、份额怎么变、换了哪些重仓、买卖了什么，
   每节一张图 + 一段「事实 → 经济含义 → 投资含义」，不做多维交叉。
4. **口径写在标题里**。全持仓 / 前十名 / 半年报，读者不需要翻方法页。

与「持仓结构」视图的分工
------------------------
本页（季报全解）回答「这一季季报说了什么」——资产配置、证监会门类行业、
份额、前十名、累计买卖、债券、财务指标，全部是季报原文披露值。
同一页可切到「持仓结构」——用前十大/全持仓反推申万 31 行行业结构、与同行比、
看历史漂移。两者一个看「原文」，一个看「结构」，口径不同，不重复。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import canon as cn
import qreport as Q
import qreport_charts as qc
import ui


def _funds() -> pd.DataFrame:
    p = cn.profiles()
    if p is None or p.empty:
        return pd.DataFrame(columns=["code", "name", "scale_yi"])
    d = p.copy()
    d["code"] = d["code"].astype(str).str.zfill(6)
    if "scale_yi" not in d.columns:
        d["scale_yi"] = np.nan
    return d.sort_values("scale_yi", ascending=False, na_position="last")


def _no_data_note(code: str) -> str:
    """「确实没有数据」的判据：逐条给出可核对的事实，不做「可能已清盘」的猜测。

    只有同时满足三条才走到这里：① 不缺依赖；② 没有任何接口报错；
    ③ 本地各表确实查不到该基金的任何季度。所以这里可以断言「源站没有记录」，
    但仍要把「没收录档案 / 成立太晚 / 抓不到」三种情形分开说。
    """
    try:
        prof = cn.profiles()
    except Exception:  # noqa: BLE001
        prof = None
    if prof is None or prof.empty:
        return "本库尚未收录基金档案，无法判断成立日；先补录档案再重试"
    row = prof[prof["code"].astype(str).str.zfill(6) == code]
    if row.empty:
        return "本库尚未收录该基金档案（无法核对成立日），先补录档案再重试"
    try:
        inc = pd.to_datetime(row["inception"].iloc[0], errors="coerce")
    except Exception:  # noqa: BLE001
        inc = pd.NaT
    try:
        q = str(cn.csi_history().index[-1])
        y, k = int(q[:4]), int(q[-1])
        end = pd.Timestamp(year=y, month={1: 3, 2: 6, 3: 9, 4: 12}[k], day=1) \
            + pd.offsets.MonthEnd(0)
    except Exception:  # noqa: BLE001
        q, end = "最新报告期", pd.NaT
    if pd.notna(inc) and pd.notna(end) and inc > end:
        return f"该基金成立于 {inc:%Y-%m-%d}，晚于最近报告期 {q}，尚未到首个披露期"
    return (f"已成立（{inc:%Y-%m-%d}）但源站未返回 {q} 及之前的记录；"
            f"可点右上「刷新」重试，本库不做推测") if pd.notna(inc) else \
        f"源站未返回 {q} 及之前的记录；可点右上「刷新」重试"


def _pick_quarter(d: dict) -> str:
    qs = set()
    for k in ("q_asset", "q_industry", "q_shares", "q_hold"):
        v = d.get(k)
        if v is not None and not v.empty:
            qs |= set(v["quarter"].dropna())
    return sorted(qs)[-1] if qs else ""


def _pick_row(df: pd.DataFrame, q: str):
    """同一季度可能有多条记录（基金转型/清盘期的临时披露日，如 2019-05-23）。

    优先取标准季末日（03-31/06-30/09-30/12-31），没有再退回最早一条，
    否则「2019Q2」可能拿到转型日的临时数据，股票仓位显示成 0.02%。
    """
    if df is None or df.empty or "quarter" not in df.columns:
        return None
    x = df[df["quarter"] == q].copy()
    if x.empty:
        return None
    x["_end"] = x["period"].astype(str).str.endswith(
        ("03-31", "06-30", "09-30", "12-31"))
    return x.sort_values(["_end", "period"], ascending=[False, True]).iloc[0]


def _prev_row(df: pd.DataFrame, row) -> pd.Series | None:
    """同一张表里、严格早于当前行的最后一条（按 period 排序）。"""
    if df is None or df.empty or row is None:
        return None
    x = df[df["period"].astype(str) < str(row["period"])].sort_values("period")
    return x.iloc[-1] if len(x) else None


def _chips(items: list[tuple[str, str]]) -> str:
    """一行「本期可得」小标签：绿＝有，灰＝季报不披露。"""
    out = []
    for name, state in items:
        ok = state == "有"
        bg, fg = ("#E8F5EC", "#1A7F37") if ok else ("#F2F5F8", "#8A97A5")
        out.append(
            f'<span style="display:inline-block;background:{bg};color:{fg};'
            f'font-size:11.5px;font-weight:600;border-radius:10px;'
            f'padding:2px 9px;margin:0 6px 4px 0">{name}{"" if ok else " · 无"}</span>')
    return f'<div style="margin:0 0 6px">{"".join(out)}</div>'


def render(h, code: str | None = None, show_picker: bool = True) -> None:
    funds = _funds()
    if funds.empty:
        h('<div class="cg-note">基金档案尚未就绪。</div>')
        return

    # ---------------- 选择器 ----------------
    if show_picker or not code:
        c1, c2, c3 = st.columns([0.52, 0.26, 0.22])
        with c1:
            opts = ui.fund_options(funds)
            _i = 0
            if code:
                _hit = [j for j, s in enumerate(opts.tolist())
                        if s.startswith(str(code) + "　")]
                _i = _hit[0] if _hit else 0
            pick = st.selectbox("选基金（可输入代码或名称搜索）", opts.tolist(),
                                index=_i, key="qr_fund")
            code = pick.split("　")[0].strip()
        with c2:
            st.caption("首次选择会现场抓取季报（约 3–6 秒），之后 20 小时内读缓存。")
        with c3:
            st.caption(f"本地已缓存 {Q.coverage()['funds']:,} 只基金")

    code = str(code).zfill(6)
    with st.spinner("读取季报数据…"):
        d = Q.fetch_fund(code)
    _errs = d.get("errors") or []
    if _errs:
        _labels = [str(e.get("label", "")) for e in _errs]
        _kinds = {str(e.get("kind", "")) for e in _errs}
        # 文案里必须出现「缺少依赖」四个字：读者与审计脚本都按这个词判断
        # 「是部署问题，不是基金没数据」。只写「无法现场抓取」会被误解。
        _head = ("部署缺少依赖，季报无法现场抓取" if "missing_dep" in _kinds
                 else "部分数据未取到")
        _more = f"等 {len(_errs)} 项" if len(_errs) > 3 else ""
        h(f'<div class="cg-note"><b>{_head}</b>：{"、".join(_labels[:3])}{_more}'
          f'　<span style="color:#8A97A5">（缺的部分显示「暂无」，不做替代口径）</span></div>')
        # 详情折叠：一行汇总在页头，具体原因与处置点开再看。
        with st.expander(f"查看这 {len(_errs)} 项的原因与处置", expanded=False):
            for e in _errs:
                st.markdown(f"- **{e.get('label', '')}**：{e.get('reason', '')}"
                            f"　→ {e.get('action', '')}")

    qs = Q.all_quarters(code)
    if not qs:
        if _errs:
            # 「抓取失败」与「数据源确实没有」是两件事，绝不能共用一句话。
            h('<div class="cg-note">这次是<b>接口未取到</b>，不是「数据源里没有这只基金」；'
              '按上面的处置重试即可。已缓存的历史期次仍会正常显示。</div>')
        else:
            h(f'<div class="cg-note">数据源没有该基金的季报记录：{_no_data_note(code)}</div>')
        return

    prof = cn.profiles()
    row = prof[prof["code"].astype(str).str.zfill(6) == code]
    nm = ui.safe_str(row["name"].iloc[0], code) if len(row) else code
    typ = ui.safe_str(row["ftype"].iloc[0]) if len(row) else "—"
    mgr = ui.safe_str(row["managers"].iloc[0]) if len(row) else "—"
    bench = (ui.safe_str(row["benchmark"].iloc[0])
             if len(row) and "benchmark" in row.columns else "—")

    h(f'<div class="sec">{nm}（{code}）· 季报全解</div>')
    _fetched = "—"
    for _k in ("q_asset", "q_shares", "q_industry", "q_hold"):
        _v = d.get(_k)
        if _v is not None and not _v.empty and "fetched_at" in _v.columns:
            _fetched = str(_v["fetched_at"].max())
            break
    h(f'<div class="tiny" style="margin:-2px 0 8px">{typ}｜基金经理 {str(mgr)[:24]}｜'
      f'业绩基准 {bench[:60] if bench and bench != "nan" else "—"}<br>'
      f'抓取 {_fetched}｜来源：基金季报（东方财富基金档案转引）｜'
      f'<b>全部为披露值，未做替代口径</b>　'
      f'<a href="https://fundf10.eastmoney.com/jjgg_{code}.html" target="_blank">公告原文</a>'
      f' · <a href="https://fundf10.eastmoney.com/zcpz_{code}.html" target="_blank">资产配置</a>'
      f' · <a href="https://fundf10.eastmoney.com/gmbd_{code}.html" target="_blank">份额变动</a>'
      f'</div>')

    # ---------------- 季度：全部历史，翻到早年就补抓 ----------------
    cq1, cq2 = st.columns([0.42, 0.58])
    with cq1:
        _opts = list(reversed(qs))
        # 换基金时，上一只基金选的季度可能不在新基金的期次里；
        # 不先归位，selectbox 会拿着一个不存在的值去匹配选项，直接报错。
        if st.session_state.get("qr_q") not in _opts:
            st.session_state["qr_q"] = _opts[0]
        q = st.selectbox(f"季度（共 {len(qs)} 期，{qs[0]}–{qs[-1]}）", _opts,
                         index=0, key="qr_q")
    with cq2:
        st.caption("资产配置与份额是全历史；行业/持仓按年抓取，选到早年季度会现场补抓"
                   "该年（约 3 秒），补过就存库，不会重复。")

    cov = Q.coverage_quarters(code)
    _year = int(str(q)[:4])
    if (q not in cov["q_industry"] and q not in cov["q_hold"]
            and q not in cov.get("q_change", set())):
        with st.spinner(f"首次查看 {_year} 年，正在补抓该年行业与持仓明细…"):
            _r = Q.ensure_year(code, _year)
        if _r.get("fetched"):
            d = Q.load_fund(code)
            cov = Q.coverage_quarters(code)
            if not _r.get("rows"):
                h(f'<div class="cg-note">{q} 在数据源里没有行业/持仓明细'
                  f'（该基金当年可能尚未成立、已清盘，或是纯债基金）。'
                  f'资产配置与份额仍按原样显示。</div>')

    a = d["q_asset"]
    a_q = _pick_row(a, q)
    s_all = d["q_shares"].sort_values("period") if not d["q_shares"].empty else pd.DataFrame()
    s_q = _pick_row(s_all, q)
    hold_q = d["q_hold"][d["q_hold"]["quarter"] == q] if not d["q_hold"].empty else pd.DataFrame()
    ind_q = d["q_industry"][d["q_industry"]["quarter"] == q] if not d["q_industry"].empty else pd.DataFrame()
    chg_q = d["q_change"][d["q_change"]["quarter"] == q] if not d["q_change"].empty else pd.DataFrame()
    bond_q = d["q_bond"][d["q_bond"]["quarter"] == q] if not d["q_bond"].empty else pd.DataFrame()
    fin_q = (d["q_fin"][d["q_fin"]["quarter"] == q]
             if d.get("q_fin") is not None and not d["q_fin"].empty else pd.DataFrame())

    # 本期可得性：先把「季报本来就不披露」和「数据源没取到」分开说清楚，
    # 否则读者会以为是自己选错了季度。
    h(_chips([
        ("资产配置", "有" if a_q is not None else "无"),
        ("行业分类", "有" if len(ind_q) else "无"),
        ("份额变动", "有" if s_q is not None else "无"),
        ("前十大", "有" if len(hold_q) else "无"),
        ("累计买卖", "有" if len(chg_q) else "无"),
        ("财务指标", "有" if len(fin_q) else "无"),
        ("债券", "有" if len(bond_q) else "无"),
    ]) + f'<div class="tiny" style="margin:-2px 0 6px">「无」有两种：季报本来就不披露'
         f'（累计买卖、财务指标只在半年报/年报有），或该基金当期没有这类资产'
         f'（纯股基没有债券）。本页不填补、不估算。</div>')

    # 数据完整度与「补齐」：行业/持仓按年抓，默认只抓最近两年。
    # 想要长历史不必一年一年点，这里一次补齐（写库后永久生效）。
    _yq = {int(str(x)[:4]) for x in qs}
    _have_y = {int(str(x)[:4]) for x in (cov["q_industry"] | cov["q_hold"])}
    _gap_years = sorted(_yq - _have_y - Q.tried_years(code))
    if _gap_years:
        _g1, _g2 = st.columns([0.62, 0.38])
        with _g1:
            h(f'<div class="tiny">行业/持仓已覆盖 '
              f'{len(set(qs) & (cov["q_industry"] | cov["q_hold"]))}/{len(qs)} 期；'
              f'缺 {len(_gap_years)} 个年份（{_gap_years[0]}–{_gap_years[-1]}）。'
              f'按年补抓，抓到就写库，下次不用再等。</div>')
        with _g2:
            if st.button(f"补齐缺失年份（{len(_gap_years)} 年，约 "
                         f"{max(3, 3 * len(_gap_years))} 秒）",
                         key="qr_fill_hist"):
                _pb = st.progress(0.0, text="正在补抓…")
                for _i, _y in enumerate(_gap_years, 1):
                    Q.ensure_year(code, _y)
                    _pb.progress(_i / len(_gap_years), text=f"已补到 {_y} 年")
                st.rerun()

    # ---------------- 核心数字 ----------------
    if a_q is not None:
        nav = float(a_q["nav_yi"])
        _pa = _prev_row(a, a_q)
        nav_chg = ((nav / float(_pa["nav_yi"]) - 1) * 100
                   if _pa is not None and np.isfinite(_pa["nav_yi"]) and
                   float(_pa["nav_yi"]) > 0 else np.nan)
        stock = float(a_q["stock_pct"])
        cash = float(a_q["cash_pct"])
        bond = float(a_q["bond_pct"]) if np.isfinite(a_q["bond_pct"]) else np.nan
        d_stock = (stock - float(_pa["stock_pct"])
                   if _pa is not None and np.isfinite(_pa["stock_pct"]) else np.nan)
    else:
        nav = nav_chg = stock = cash = bond = d_stock = np.nan

    net_buy = (float(s_q["purchase_yi"]) - float(s_q["redeem_yi"])) \
        if s_q is not None and np.isfinite(s_q["purchase_yi"]) and \
        np.isfinite(s_q["redeem_yi"]) else np.nan

    # 份额表与资产表是否同一口径（都是「基金整体」）。
    # 东财的份额接口按份额类别返回；若历史数据尚未合并，期末净资产会显著小于
    # 资产配置里的净资产。此时份额类数字只能当「本份额类别」看，规模拆解必须停用，
    # 否则会算出「219 亿 ÷ 0.03 亿份 = 7300 元/份」这种明显错误的份额净值。
    _share_same = bool(
        s_q is not None
        and Q.shares_consistent(s_q.get("end_nav_yi", np.nan), nav))

    top10 = hold_q.nsmallest(10, "rnk") if len(hold_q) else pd.DataFrame()
    conc = float(top10["ratio_pct"].sum()) if len(top10) else np.nan
    conc_of_stock = conc / stock * 100 if np.isfinite(conc) and stock > 0 else np.nan

    _top_ind = (ind_q.nlargest(1, "ratio_pct").iloc[0] if len(ind_q) else None)

    def _yi(v: float) -> str:
        """亿元显示：小基金不能被四舍五入成「0 亿」。"""
        if not np.isfinite(v):
            return "—"
        if v < 1:
            return f"{v:.2f}"
        if v < 10:
            return f"{v:.1f}"
        return f"{v:,.0f}"

    k = []
    k.append(("净资产", f"{_yi(nav)} 亿" if np.isfinite(nav) else "暂无",
              (f"环比 {nav_chg:+.1f}%" if np.isfinite(nav_chg) else "—")))
    k.append(("股票仓位", f"{stock:.1f}%" if np.isfinite(stock) else "暂无",
              (f"较上期 {d_stock:+.1f}pp｜现金 {cash:.1f}%" if np.isfinite(d_stock)
               else (f"现金 {cash:.1f}%" if np.isfinite(cash) else "—"))))
    k.append(("净申购", f"{net_buy:+.1f} 亿份" if np.isfinite(net_buy) else "暂无",
              (f"申购 {s_q['purchase_yi']:.1f}｜赎回 {s_q['redeem_yi']:.1f}"
               + ("" if _share_same else "（本份额类别）")
               if s_q is not None else "—")))
    k.append(("前十大集中度", f"{conc:.1f}%" if np.isfinite(conc) else "暂无",
              (f"占股票仓位 {conc_of_stock:.0f}%" if np.isfinite(conc_of_stock)
               else "—")))
    k.append(("最重行业", f"{_top_ind['sec_name'][:6]}" if _top_ind is not None else "暂无",
              (f"占净值 {_top_ind['ratio_pct']:.1f}%" if _top_ind is not None else "—")))
    h(ui.stat_strip(k))

    # ---------------- 一句话：这一季最重要的三件事 ----------------
    # 不再把 KPI 里的五个数字重念一遍（读者刚看完），只讲「变了什么、为什么变」。
    _ev = []
    # 行业切换：本期相对上一期行业表的最大加减
    _iq = sorted(d["q_industry"]["quarter"].dropna().unique()) \
        if not d["q_industry"].empty else []
    _iprev = (_iq[_iq.index(q) - 1] if q in _iq and _iq.index(q) > 0 else None)
    if _iprev:
        _ta = Q.tag_industry(ind_q)
        _tb = Q.tag_industry(d["q_industry"][d["q_industry"]["quarter"] == _iprev])
        _sa = _ta[_ta["market"] == "A股"].groupby("sec_name")["ratio_pct"].sum()
        _sb = _tb[_tb["market"] == "A股"].groupby("sec_name")["ratio_pct"].sum()
        _dl = (_sa - _sb).dropna()
        if len(_dl) >= 2:
            _u, _d = _dl.idxmax(), _dl.idxmin()
            if abs(_dl[_u]) >= 2 or abs(_dl[_d]) >= 2:
                _ev.append(f'<b>行业</b> {_d} {_dl[_d]:+.0f}pp、{_u} {_dl[_u]:+.0f}pp'
                           f'（{_iprev}→{q}）')
    # 前十换手：新进 + 升入
    _hq = sorted(d["q_hold"]["quarter"].dropna().unique()) if not d["q_hold"].empty else []
    _hprev = (_hq[_hq.index(q) - 1] if q in _hq and _hq.index(q) > 0 else None)
    _df0 = Q.top10_diff(d, q, _hprev) if _hprev else pd.DataFrame()
    if len(_df0):
        _nnew = int(_df0["状态"].isin(["新进前十", "升入前十"]).sum())
        _big = _df0[_df0["状态"] == "升入前十"].nlargest(1, "d_pct")
        _txt = f'<b>前十</b> 换掉 {_nnew} 席'
        if len(_big):
            _txt += (f'，{_big.iloc[0]["名称"]} {_big.iloc[0]["上期%"]:.1f}%→'
                     f'{_big.iloc[0]["本期%"]:.1f}%')
        _ev.append(_txt)
    if not _ev:
        _ev = [f'{q} 季报数据不完整，无法给出变化摘要。']

    # 规模已在上一行 KPI 与下面的瀑布图里拆过（净申购 vs 净值涨跌），
    # 这里只说「行业」与「前十」两件事，避免同一组数字写三遍。
    _econ = ("<b>行业</b>大幅切换说明观点变了（不是市场涨跌能解释的）；"
             "<b>前十换手</b>高说明它在主动调整，而不是被动持有。"
             "两者同时出现，才是一次方向调整；都没动，就是「拿着」。")
    _inv = ("规模靠申购 + 行业切换 + 前十大换手同时出现，意味着这是一次<b>主动的方向调整</b>："
            "要顺着新进的行业和股票去验证逻辑，而不是看它过去三年的业绩。"
            "反过来，三项都没动，说明这一季它只是「拿着」，此时业绩好坏更多来自市场，不是它的判断。")
    ui.analysis(h, "；".join(_ev) + "。", _econ, _inv,
                tag1="变了什么", tag2="怎么读", tag3="对你的动作")

    _CIRCLED = "①②③④⑤⑥⑦⑧⑨"
    _nsec = [0]

    def _sec(title: str, sub: str = "") -> None:
        _nsec[0] += 1
        h(ui.section(f"{_CIRCLED[_nsec[0] - 1]} {title}", sub))

    # ---------------- ① 钱放在哪 ----------------
    _sec("钱放在哪：股票 / 债券 / 现金",
         f"全部 {a['quarter'].nunique() if not a.empty else 0} 期；占基金净值，{q} 已高亮标注")
    st.plotly_chart(qc.fig_asset_mix(a, n=None, q=q), width="stretch",
                    config={"displayModeBar": False, "scrollZoom": False})
    if a_q is not None:
        _sum3 = stock + cash + (bond if np.isfinite(bond) else 0)
        h(f'<div class="tiny" style="margin:-4px 0 6px">'
          f'本期：股票 {stock:.1f}% ＋ 现金 {cash:.1f}%'
          + (f" ＋ 债券 {bond:.1f}%" if np.isfinite(bond) else "")
          + f' ＝ {_sum3:.1f}%'
          + ('，超过 100% 的部分来自应付/回购等负债项，季报原样披露，不做归一。'
             if _sum3 > 101 else '。')
          + '　分组柱不堆叠：三者之和本来就不等于 100%，堆叠会让人误以为刚好分完。'
          f'</div>')
        # 仓位在「自己的历史」里排第几——和同行比是横向，和自己比才是纵向。
        # 只用到当期为止的期数：回看 2018Q1 时把 2026 年的仓位算进分位，等于用了未来数据。
        _aps = a.sort_values("period")
        _aps = _aps[_aps["quarter"] <= q]
        _pct_stock = Q.self_pct(_aps["stock_pct"], stock)
        _pct_cash = Q.self_pct(_aps["cash_pct"], cash)
        _fact = (f'股票仓位 {stock:.1f}%，在本基金 {len(_aps)} 期里排第 '
                 f'{_pct_stock:.0f} 分位；现金 {cash:.1f}%（第 {_pct_cash:.0f} 分位）'
                 + (f'、债券 {bond:.1f}%' if np.isfinite(bond) else '') + '。')
        if np.isfinite(d_stock):
            _fact += f'较上期 {d_stock:+.1f}pp。'
        _econ = ('仓位是风险预算，不是观点：股票型基金有 80% 的仓位下限，'
                 '可调空间只有十几个点，所以「看好」很少表现为整体加仓，'
                 '而是表现为行业之间的再分配。现金比例高，可能是主动留弹药，'
                 '也可能是申购进来还没买——用份额数据可以区分。')
        _inv = ('仓位在自己历史的高分位（≥80）＋ 前十大集中度高：组合最脆弱，'
                '任何一次风格切换都会被放大；'
                '低分位（≤20）＋ 现金多：有加仓空间，但也可能是被赎回逼出来的。'
                '判断方向要看下一季它是把现金变成股票，还是继续趴着。')
        ui.analysis(h, _fact, _econ, _inv,
                    tag1="仓位位置", tag2="说明什么", tag3="对你的动作")
    else:
        h('<div class="cg-note">本期无资产配置数据（季报未披露或数据源缺失）。</div>')

    # 规模变化拆解：净资产变动 = 份额效应 + 净值效应 + 交叉项（恒等式）
    if a_q is not None and s_q is not None and np.isfinite(nav) and not _share_same:
        h(f'<div class="cg-note" style="margin:-4px 0 10px">'
          f'本期份额表与资产配置的净资产口径不一致'
          f'（{float(s_q.get("end_nav_yi", float("nan"))):.2f} 亿 vs {nav:.2f} 亿），'
          f'说明该期份额只覆盖部分份额类别——<b>本页不做规模拆解</b>，'
          f'避免用「基金整体净资产 ÷ 单一类别份额」算出错误的份额净值。'
          f'份额数字仅作「本份额类别」参考。</div>')
    if a_q is not None and s_q is not None and np.isfinite(nav) and _share_same:
        _pa = _prev_row(a, a_q)
        if _pa is not None and np.isfinite(float(_pa["nav_yi"])):
            _nav0 = float(_pa["nav_yi"])
            _e_sh = float(s_q["end_share_yi"])
            _b_sh = _e_sh - float(s_q["purchase_yi"]) + float(s_q["redeem_yi"])
            if _b_sh > 0 and _nav0 > 0 and _e_sh > 0:
                _np0, _np1 = _nav0 / _b_sh, nav / _e_sh
                _eff_s = (_e_sh - _b_sh) * _np0
                _eff_n = _b_sh * (_np1 - _np0)
                _cross = (_e_sh - _b_sh) * (_np1 - _np0)
                st.plotly_chart(qc.fig_scale_bridge(_nav0, _eff_s, _eff_n, _cross,
                                                    nav, _np0, _np1),
                                width="stretch",
                                config={"displayModeBar": False})
                h(f'<div class="tiny" style="margin:-6px 0 10px">'
                  f'<b>规模变化拆解</b>（恒等式，不是估算）：净资产 {nav - _nav0:+.1f} 亿 ＝ '
                  f'净申购 {_eff_s:+.1f} ＋ 净值上涨 {_eff_n:+.1f} ＋ 交叉 {_cross:+.1f}。'
                  f'这一拆的意义：<b>规模涨不等于业绩好</b>——靠申购堆出来的规模要建仓，'
                  f'会摊薄原有持仓；靠净值涨出来的规模只是账面浮盈，不产生买入压力。</div>')

    # ---------------- ② 行业押注 ----------------
    _sec("行业押注：季报按行业分类的全部股票",
         "证监会门类口径，覆盖全部股票，不是前十大")
    _ind_qs = sorted(d["q_industry"]["quarter"].dropna().unique()) \
        if not d["q_industry"].empty else []
    _prev_q = (_ind_qs[_ind_qs.index(q) - 1]
               if q in _ind_qs and _ind_qs.index(q) > 0 else None)
    ind_prev = (d["q_industry"][d["q_industry"]["quarter"] == _prev_q]
                if _prev_q else None)
    _iv = Q.industry_view(ind_q, ind_prev) if len(ind_q) else pd.DataFrame()
    _tag = Q.tag_industry(ind_q) if len(ind_q) else pd.DataFrame()
    _hk = (float(_tag[_tag["market"] == "港股通"]["ratio_pct"].sum())
           if len(_tag) else 0.0)
    if len(_iv):
        _it = Q.industry_trend(d, 6)
        _c1, _c2 = st.columns([0.5, 0.5], gap="small")
        with _c1:
            st.plotly_chart(qc.fig_industry(_iv), width="stretch",
                            config={"displayModeBar": False})
        with _c2:
            if len(_it) > 1:
                st.plotly_chart(qc.fig_industry_area(_it), width="stretch",
                                config={"displayModeBar": False})
            else:
                h('<div class="cg-note" style="margin-top:40px">只有一期行业表，'
                  '看不出行业押注的挪动。翻到更早的季度会自动补抓。</div>')
        _tot = float(ind_q["ratio_pct"].sum())
        _gap = stock - _tot
        # 差 0.004 会格式化成「-0.00pp」，看着像错数；四舍五入到 0 就直接写 0.00。
        _gap_s = "0.00" if abs(_gap) < 0.005 else f"{_gap:+.2f}"
        if np.isfinite(stock) and abs(_gap) < 0.5:  # caliber-ok: 0.5pp 数值核对容差
            _gap_txt = (f'行业合计 <b>{_tot:.1f}%</b>，与股票仓位 {stock:.1f}% 一致'
                        f'（差 {_gap_s}pp）——这张表覆盖了全部股票。')
        else:
            _gap_txt = (f'行业合计 <b>{_tot:.1f}%</b>，与股票仓位 {stock:.1f}% 差 '
                        f'{_gap:+.1f}pp。差额有三种来源：① 证监会门类未覆盖的标的'
                        f'（北交所 920/83x/87x/43x 与新上市未映射代码）；'
                        f'② 港股通按恒生分类、与 A 股门类不同源，只单列不并入；'
                        f'③ 季报本身的四舍五入。季报原样披露，不做归一。')
        if _hk > 0.01:
            _gap_txt += (f'　港股通 {_hk:.1f}% 用的是<b>恒生行业分类</b>，与 A 股的证监会门类'
                         f'不同源，已汇总成一条（橙色），不逐条并列。')
        # 环比：只对本期和上期都有的门类做差，避免「新出现/消失」被读成加减仓
        if len(_iv) and "prev_pct" in _iv.columns and _iv["prev_pct"].notna().any():
            _dd = (_iv.dropna(subset=["prev_pct"])
                   .assign(Δ=lambda x: x["ratio_pct"] - x["prev_pct"])
                   .sort_values("Δ"))
            if len(_dd):
                _dn, _up = _dd.iloc[0], _dd.iloc[-1]
                _gap_txt += (f'　环比（{_prev_q}→{q}）：加仓最多 <b>{_up["sec_name"]}</b>'
                             f' {_up["Δ"]:+.2f}pp，减仓最多 <b>{_dn["sec_name"]}</b>'
                             f' {_dn["Δ"]:+.2f}pp；只比两期都有数据的门类。')
        h(f'<div class="tiny" style="margin:-4px 0 10px">{_gap_txt}</div>')
    else:
        h('<div class="cg-note">该期无行业配置数据（季报未披露或数据源缺失）。</div>')

    # ---------------- ③ 份额 ----------------
    _sec("份额怎么变：申购、赎回与期末规模",
         ("基金整体、A/C 合并；单位亿份；期初＝期末−申购+赎回（会计恒等式）"
          if _share_same else
          "本份额类别（非基金整体）；单位亿份；期初＝期末−申购+赎回"))
    if s_q is not None:
        _c1, _c2 = st.columns([0.52, 0.48], gap="small")
        with _c1:
            st.plotly_chart(qc.fig_shares_waterfall(s_q), width="stretch",
                            config={"displayModeBar": False})
        with _c2:
            _sp = s_all.dropna(subset=["end_share_yi"]).copy()
            if len(_sp) > 1:
                _sp["净申购"] = _sp["purchase_yi"] - _sp["redeem_yi"]
                _sp = _sp.tail(20)
                import plotly.graph_objects as go
                from report_charts import _base
                _f = go.Figure()
                _f.add_trace(go.Bar(
                    x=list(_sp["quarter"]), y=list(_sp["净申购"]),
                    marker_color=["#C62828" if v >= 0 else "#1A7F37"
                                  for v in _sp["净申购"]],
                    marker_line_width=0,
                    hovertemplate="%{x}<br>净申购 %{y:+.2f} 亿份<extra></extra>"))
                _f.add_trace(go.Scatter(
                    x=list(_sp["quarter"]), y=list(_sp["end_share_yi"]),
                    mode="lines+markers", name="期末份额",
                    line=dict(color="#2F3B49", width=1.8), marker=dict(size=3.5),
                    yaxis="y2",
                    hovertemplate="%{x}<br>期末份额 %{y:.2f} 亿份<extra></extra>"))
                _tk = qc._thin(list(_sp["quarter"]))
                _f.update_xaxes(tickmode="array", tickvals=_tk, ticktext=_tk,
                                tickfont=dict(size=9))
                _f.update_yaxes(title_text="净申购（亿份）", tickfont=dict(size=9))
                _f.update_layout(yaxis2=dict(title="期末份额（亿份）", overlaying="y",
                                             side="right", showgrid=False,
                                             tickfont=dict(size=9)))
                _f = _base(_f, "净申购与期末份额（近 20 期）", 320, legend=True)
                _f.update_layout(margin=dict(l=48, r=52, t=48, b=52),
                                 legend=dict(orientation="h", y=-0.2, x=0,
                                             font=dict(size=9)))
                st.plotly_chart(_f, width="stretch",
                                config={"displayModeBar": False})
            else:
                h('<div class="cg-note" style="margin-top:40px">份额历史不足两期。</div>')
        _ps = _prev_row(s_all, s_q)
        _p = float(s_q["purchase_yi"])
        _r = float(s_q["redeem_yi"])
        _p_ok, _r_ok = np.isfinite(_p), np.isfinite(_r)
        # 源数据个别期次只披露期末份额，申购/赎回为空；直接格式化会写出 nan。
        _pt = f"{_p:.2f}" if _p_ok else "未披露"
        _rt = f"{_r:.2f}" if _r_ok else "未披露"
        _nt = f"{_p - _r:+.2f}" if (_p_ok and _r_ok) else "无法计算"
        _fact = (f'本期申购 {_pt} 亿份、赎回 {_rt} 亿份，净申购 '
                 f'{_nt} 亿份，期末 {float(s_q["end_share_yi"]):.2f} 亿份。')
        if (_p_ok and _p < 0) or (_r_ok and _r < 0):
            _fact += ('　<b>注意</b>：源数据该期的申购/赎回为负，'
                      '本库原样保留、不做修正，也不据此算换手。')
        if (_p_ok and _r_ok and _ps is not None
                and np.isfinite(_ps["end_share_yi"])
                and float(_ps["end_share_yi"]) > 0):
            _rate = (float(s_q["end_share_yi"]) / float(_ps["end_share_yi"]) - 1) * 100
            _flow = (_p + _r)
            _turn_sum = _flow / float(_ps["end_share_yi"]) * 100
            # 换手强度按「买卖各算一半」定义：直接说「申购+赎回合计＝94%」
            # 会把 187% 的进出说成 94%，与词典里的公式对不上。
            _turn = _turn_sum / 2
            _fact += (f'期末份额环比 {_rate:+.1f}%；期间申购+赎回合计为期初份额的 '
                      f'{_turn_sum:.0f}%，单边换手强度 <b>{_turn:.0f}%</b>'
                      f'（（申购+赎回）÷2÷期初，见词典「份额换手强度」）。')
        ui.analysis(h,
                    _fact,
                    '份额是<b>唯一的「用钱投票」指标</b>：净值涨跌由市场决定，'
                    '申购赎回由持有人决定。净申购持续为正，说明渠道在推、客户在追；'
                    '持续为负，说明产品在被赎回，基金经理可能被迫卖出流动性最好的股票。',
                    '净申购大正 + 仓位下降：新钱进来但还没建仓，下一季有买入压力；'
                    '净申购大负 + 仓位没降：赎回把仓位被动推高，看似「加仓」其实是缩表。'
                    '这两种情况都不能读成「基金经理看多」。')
    else:
        h('<div class="cg-note">该期无份额变动数据。</div>')

    # ---------------- ④ 重仓与换仓 ----------------
    _n_hold = len(hold_q)
    _rest = hold_q[hold_q["rnk"] > 10] if _n_hold else pd.DataFrame()
    _n_hk = int(hold_q["stock_code"].astype(str).str.startswith("HK").sum()) \
        if _n_hold else 0
    _sub = (f"数据源本期返回 {_n_hold} 只"
            + (f"（含港股 {_n_hk} 只，代码带 HK 前缀，不计入 A 股行业）"
               if _n_hk else "")
            + f"；法定只要求披露前十，多出的为自愿披露或数据源附加明细。"
            + f"条形与表格取前 10；"
            + (f"另有 {len(_rest)} 只合计 {float(_rest['ratio_pct'].sum()):.1f}%。"
               if len(_rest) else "已全部列出。"))
    # A+H 两地上市：官方季报按「合并公允价值」排序、A/H 分行披露，
    # 本库只能按单只占比排序，遇到这类公司名次会与原文不同。
    # 不解释会被当成算错，所以显式标出来（全库有 473 个基金×季度属于这种情况）。
    _ah = (hold_q[hold_q["stock_name"].duplicated(keep=False)]
           .groupby("stock_name")["stock_code"]
           .filter(lambda x: x.astype(str).str.startswith("HK").any()
                   and (~x.astype(str).str.startswith("HK")).any())
           .unique() if _n_hold else [])
    if len(_ah):
        _sub += (f'　<b>注意</b>：本期同时持有 {"、".join(list(_ah)[:3])} 的 A 股与 H 股。'
                 f'官方季报对这类公司按「A+H 合并公允价值」排序、分行披露，'
                 f'本页按单只占净值比排序，名次可能与原文不同（数值不受影响）。')
    _sec(f"重仓股（{q}）：前十大占净值 {conc:.1f}%" if np.isfinite(conc)
         else f"重仓股（{q}）", _sub)
    if len(top10):
        cc1, cc2 = st.columns([0.46, 0.54])
        with cc1:
            st.plotly_chart(qc.fig_top10_bar(top10), width="stretch",
                            config={"displayModeBar": False})
        with cc2:
            _tb = top10.nsmallest(10, "rnk")[["rnk", "stock_code", "stock_name",
                                              "ratio_pct", "shares_wan", "mv_wan"]]
            st.dataframe(_tb.rename(columns={
                "rnk": "#", "stock_code": "代码", "stock_name": "名称",
                "ratio_pct": "占净值%", "shares_wan": "持股(万股)",
                "mv_wan": "市值(万元)"}), hide_index=True, height=380,
                width="stretch", column_config={
                    "#": st.column_config.NumberColumn(width="small", format="%d"),
                    "占净值%": st.column_config.NumberColumn(format="%.2f"),
                    "持股(万股)": st.column_config.NumberColumn(format="%.1f"),
                    "市值(万元)": st.column_config.NumberColumn(format="%.0f")})
    else:
        h('<div class="cg-note">该期无持仓明细。</div>')

    # 集中度：自己和自己比（同行之间比集中度没有意义，规模与策略差异太大）
    _hc = Q.hold_conc_series(d) if _n_hold or not d["q_hold"].empty else pd.DataFrame()
    if len(_hc):
        # 只和「当期及以前」比：回看 2018Q1 时不能用 2026 年的集中度当历史，
        # 那等于把未来信息塞进了分位。
        _hc = _hc[_hc["quarter"] <= q]
    if len(_hc) >= 2 and np.isfinite(conc):
        _pct_c = Q.self_pct(_hc["top10"], conc)
        ui.analysis(
            h,
            f'前十大合计占净值 <b>{conc:.1f}%</b>，在已取到的 {len(_hc)} 期里排第 '
            f'<b>{_pct_c:.0f}</b> 分位'
            + (f'；相当于股票仓位的 {conc_of_stock:.0f}%' if np.isfinite(conc_of_stock) else '')
            + f'。披露 {_n_hold} 只，其余部分更分散。',
            '集中度是「靠少数股票还是靠一篮子」的直接度量。集中度高，'
            '意味着净值由少数几只股票决定，研究得过来、但个股风险大；'
            '集中度低，等于买了一个小型指数，个股风险被摊薄，超额收益也被摊薄。',
            '集中度在自身高分位（≥80）时，任何一只重仓股出问题都会被放大，'
            '要逐只核对逻辑；在低分位（≤20）时，它更像指数，'
            '此时「选股能力」不是买它的理由，应该看行业配置。')

    # 前十大逐季进出：季报没有「重大买卖」时，这是唯一的换仓证据
    _hqs = sorted(d["q_hold"]["quarter"].dropna().unique()) if _n_hold else []
    _prev_hq = (_hqs[_hqs.index(q) - 1] if q in _hqs and _hqs.index(q) > 0 else None)
    _diff = Q.top10_diff(d, q, _prev_hq) if _prev_hq else pd.DataFrame()
    if len(_diff):
        _n_new = int((_diff["状态"] == "新进前十").sum())
        _n_in = int((_diff["状态"] == "升入前十").sum())
        _n_out = int((_diff["状态"] == "退出前十").sum())
        _n_clr = int((_diff["状态"] == "清仓").sum())
        # 「股数增加」要覆盖所有有基数可比的股票（增持 + 升入），
        # 只数「增持」会与下面「升入 +384%」自相矛盾。
        _dsh = pd.to_numeric(_diff["d_sh"], errors="coerce")
        _n_up = int((_dsh > 0.005).sum())
        _n_dn = int((_dsh < -0.005).sum())
        _sec(f"这一季换了谁：{_prev_hq} → {q} 前十大进出",
             "上期若为半年报/年报（全持仓），用它做基准：区分「新进」（上期未持有）"
             "与「升入」（上期已持有、未进前十）；股数变化含送转股")
        # 图与表是同一批数据的两种看法：并排时图被压到半栏、表被截断，
        # 读者两份都要看却都看不清。改为图通栏、表折叠——要核对精确数字再展开。
        st.plotly_chart(qc.fig_top10_diff(_diff), width="stretch",
                        config={"displayModeBar": False})
        with st.expander(f"看精确数字：{len(_diff)} 只股票的占比与股数变化",
                         expanded=False):
            _dt = _diff.reset_index()[["代码", "名称", "状态", "本期%",
                                       "上期%", "d_pct", "d_sh"]]
            _dt = _dt.sort_values("d_pct", ascending=False)
            _dt["d_sh"] = _dt["d_sh"].map(
                lambda v: "—" if not np.isfinite(v) else f"{v:+.0%}")
            _dt["d_pct"] = _dt["d_pct"].map(lambda v: f"{v:+.2f}")
            for _c in ("本期%", "上期%"):
                _dt[_c] = _dt[_c].map(lambda v: "—" if not np.isfinite(v) else f"{v:.2f}")
            st.dataframe(_dt.rename(columns={"d_pct": "Δpp", "d_sh": "股数变化"}),
                         hide_index=True, height=380, width="stretch")
        _new = _diff[_diff["状态"] == "新进前十"].nlargest(2, "本期%")
        _in = _diff[_diff["状态"] == "升入前十"].nlargest(2, "d_pct")
        _out = _diff[_diff["状态"].isin(["退出前十", "清仓"])].nlargest(2, "上期%")
        _big_up = _diff[_diff["状态"].isin(["增持", "升入前十"])].nlargest(2, "d_sh")
        _big_dn = _diff[_diff["状态"] == "减持"].nsmallest(2, "d_sh")
        _turn = (_n_new + _n_in + _n_out + _n_clr) / 20 * 100
        _ft = (f'十席里：新进 <b>{_n_new}</b> 只（上期未持有）、升入 <b>{_n_in}</b> 只'
               f'（上期已持有但不在前十）、退出 <b>{_n_out}</b> 只'
               + (f'、清仓 <b>{_n_clr}</b> 只' if _n_clr else '')
               + f'；股数增加 {_n_up} 只、减少 {_n_dn} 只'
               f'（十席换手 <b>{_turn:.0f}%</b>＝(换进 {_n_new + _n_in} 席'
               f'＋换出 {_n_out + _n_clr} 席)÷20，等价于换掉 '
               f'{(_n_new + _n_in + _n_out + _n_clr) / 2:.0f}/10 席）。')
        if len(_new):
            _ft += '　新进：' + "、".join(f'{r["名称"]}（{r["本期%"]:.2f}%）'
                                        for _, r in _new.iterrows()) + '。'
        if len(_in):
            _ft += '　升入：' + "、".join(
                f'{r["名称"]}（{r["上期%"]:.2f}%→{r["本期%"]:.2f}%，'
                f'股数 {r["d_sh"]:+.0%}）' for _, r in _in.iterrows()) + '。'
        if len(_out):
            _ft += (f'　退出/清仓（共 {_n_out + _n_clr} 只，最大两只）：' + "、".join(
                f'{r["名称"]}（上期 {r["上期%"]:.2f}%）'
                for _, r in _out.iterrows()) + '。')
        if len(_big_up):
            _ft += '　股数增加最多：' + "、".join(
                f'{r["名称"]}（{r["d_sh"]:+.0%}，占净值 {r["d_pct"]:+.2f}pp）'
                for _, r in _big_up.iterrows()) + '。'
        if len(_big_dn):
            _ft += '　股数减少最多：' + "、".join(
                f'{r["名称"]}（{r["d_sh"]:+.0%}，占净值 {r["d_pct"]:+.2f}pp）'
                for _, r in _big_dn.iterrows()) + '。'
        # 规模暴增会把所有持仓的权重「摊薄」：股数没减，占比也会掉。
        if np.isfinite(nav_chg) and nav_chg > 15:
            _ft += (f'　注意：本期净资产 {nav_chg:+.0f}%，'
                    f'即使一股没卖，老持仓的占比也会被新钱摊薄——'
                    f'看到权重下降先看股数，股数没减就不是减持。')
        ui.analysis(h, _ft,
                    '「新进」与「升入」是两件事：前者是这一季新建仓，后者是原有持仓被加到前十——'
                    '后者的股数变化可以逐股核对，含金量更高。'
                    '「退出前十」不等于清仓（Q1/Q3 只披露前十，可能仍持有）；'
                    '只有在本期是半年报/年报（全持仓）时，彻底消失才算「清仓」。',
                    '优先跟「升入前十 + 股数大幅增加」的股票：有上期基数、有真实加仓，'
                    '是基金经理主动提高权重的核心持仓；'
                    '「新进」要看是不是行业性建仓（同行业多只一起进），是的话买行业 ETF 更省事。'
                    '「清仓」出现在半年报里，是最强的否定信号，比「退出前十」确定得多。')

    # ---------------- ⑤ 买卖与利润 ----------------
    _sec("这一季买卖了什么（半年报/年报口径）",
         "季报「报告期内股票投资组合的重大变动」，披露累计买卖前 20 名，图中取前 10")
    if len(chg_q):
        st.plotly_chart(qc.fig_trades(chg_q), width="stretch",
                        config={"displayModeBar": False})
        _b = chg_q[chg_q["direction"] == "买入"]["amount_wan"].sum() / 1e4
        _s = chg_q[chg_q["direction"] == "卖出"]["amount_wan"].sum() / 1e4
        _top_b = chg_q[chg_q["direction"] == "买入"].nlargest(1, "amount_wan")
        _top_s = chg_q[chg_q["direction"] == "卖出"].nlargest(1, "amount_wan")
        _ft = f'前 20 名合计：买入 {_b:.1f} 亿元、卖出 {_s:.1f} 亿元。'
        if len(_top_b):
            _ft += (f'　买得最多的是 {_top_b.iloc[0]["stock_name"]}'
                    f'（{_top_b.iloc[0]["amount_wan"] / 1e4:.1f} 亿元）。')
        if len(_top_s):
            _ft += (f'　卖得最多的是 {_top_s.iloc[0]["stock_name"]}'
                    f'（{_top_s.iloc[0]["amount_wan"] / 1e4:.1f} 亿元）。')
        ui.analysis(h, _ft,
                    '这张表只覆盖「超过披露门槛」的交易，不等于全部换手：'
                    '买入前 20 名合计通常只占净值的一小部分。它的价值在于<b>方向</b>——'
                    '同一只股票同时出现在买入和卖出榜，说明在做波段；只出现在一边，才是真的建仓/清仓。',
                    '买入榜集中在一个行业，说明是行业性加仓，跟行业做即可；'
                    '分散在多个行业，说明是选股，要逐只判断。'
                    '卖出榜里出现「刚买不久的新宠」，是最该警惕的信号：逻辑可能已被证伪。')
    else:
        h('<div class="cg-note">本期无重大买卖明细。<b>这是制度决定的，不是缺数据</b>：'
          '「累计买入/卖出前 20 名」只在半年报与年报（Q2/Q4）披露，'
          '一季报与三季报没有这张表。本页不用持仓差分代替它——'
          '上一节的「前十大进出」是同一目的下唯一有披露依据的替代视角。</div>')

    if len(fin_q):
        _fv = fin_q.drop_duplicates("item").set_index("item")["value"]

        def _g(k):
            return float(_fv.get(k, np.nan))

        _pf, _rl = _g("本期利润") / 1e8, _g("本期已实现收益") / 1e8
        _npv, _wp = _g("期末基金份额净值"), _g("加权平均基金份额本期利润")
        _ratio = (_rl / _pf * 100) if np.isfinite(_rl) and np.isfinite(_pf) and _pf else np.nan
        h(f'<div style="font-size:13px;font-weight:700;margin:4px 0 4px">'
          f'主要财务指标（半年报/年报口径，报告期累计）</div>')
        h(ui.stat_strip([
            ("本期利润", f"{_pf:,.1f} 亿" if np.isfinite(_pf) else "暂无",
             "含未卖出股票的浮盈"),
            ("本期已实现收益", f"{_rl:,.1f} 亿" if np.isfinite(_rl) else "暂无",
             "卖掉的部分才计入"),
            ("兑现比例", f"{_ratio:.0f}%" if np.isfinite(_ratio) else "—",
             "已实现 ÷ 本期利润"),
            ("期末份额净值", f"{_npv:.3f} 元" if np.isfinite(_npv) else "暂无",
             (f"加权平均份额利润 {_wp:.2f} 元" if np.isfinite(_wp) else "—")),
        ]))
        h('<div class="tiny" style="margin:-4px 0 10px">'
          '「本期利润」含未卖出股票的浮盈，「已实现收益」只算卖掉的部分；'
          '兑现比例低说明收益还挂在账上，下一季如果回撤会把利润吐回去。'
          '一季报/三季报不要求披露这两个数，选到 Q1/Q3 时本段省略。</div>')

    if len(bond_q):
        h(f'<div style="font-size:13px;font-weight:700;margin:10px 0 4px">'
          f'债券持仓（季报「按债券品种分类」与「前五名债券」）</div>')
        st.dataframe(bond_q[["bond_code", "bond_name", "ratio_pct", "mv_wan"]]
                     .rename(columns={"bond_code": "代码", "bond_name": "名称",
                                      "ratio_pct": "占净值%", "mv_wan": "市值(万元)"}),
                     hide_index=True, height=min(220, 36 * len(bond_q) + 40),
                     width="stretch",
                     column_config={"占净值%": st.column_config.NumberColumn(format="%.2f"),
                                    "市值(万元)": st.column_config.NumberColumn(format="%.0f")})

    # ---------------- ⑥ 同行 ----------------
    _ms = Q.market_series()
    _peer = Q.industry_peer(q, _prev_q) if _prev_q else pd.DataFrame()
    if len(_ms) and int(_ms["count"].iloc[-1]) >= 30:
        _mrow0 = _ms[_ms["quarter"] == q]
        _nq_peer = (int(_mrow0["count"].iloc[-1]) if len(_mrow0)
                    else int(_ms["count"].iloc[-1]))
        # 这不是「全市场」：q_asset 只含已抓取基金，且抓取按规模优先，属大基金口径。
        # 写成「全样本」会让读者把 N=4,920 当成全市场 1 万多只基金，必须点明。
        _sec("已入库基金在做什么（大基金口径，不是全市场）",
             f"{q} 有 {_nq_peer:,} 个份额代码可算中位数（已入库 {len(funds):,} 个，"
             f"覆盖 {_nq_peer / max(len(funds), 1):.1%}；A/C 分别计，按规模优先抓取）；"
             f"行业环比为配对样本")
        _c1, _c2 = st.columns([0.5, 0.5])
        with _c1:
            st.plotly_chart(qc.fig_market_position(_ms), width="stretch",
                            config={"displayModeBar": False})
        with _c2:
            st.plotly_chart(qc.fig_industry_delta(_peer), width="stretch",
                            config={"displayModeBar": False})
        # 必须用「所选季度」的中位数：拿 2018 年的仓位去和 2026 年的中位数比，
        # 差值没有意义（这十年全样本仓位中枢上移了近 10pp）。
        _mrow = _ms[_ms["quarter"] == q]
        _mm = float(_mrow["median"].iloc[-1]) if len(_mrow) else float("nan")
        _txt = (f'{q} 已入库 {int(_mrow["count"].iloc[-1]) if len(_mrow) else 0:,} 个份额代码的'
                f'股票仓位中位数 <b>{_mm:.1f}%</b>，本基金 {stock:.1f}%'
                f'（{stock - _mm:+.1f}pp）。' if np.isfinite(_mm) else "")
        if len(_peer):
            _u = _peer["差值"].nlargest(2)
            _d = _peer["差值"].nsmallest(2)
            _lo = int(_peer["样本"].min()) if "样本" in _peer.columns else 0
            _hi = int(_peer["样本"].max()) if "样本" in _peer.columns else 0
            _txt += ('已入库基金里加仓最多的是 ' + "、".join(f"{i}（{v:+.2f}pp）" for i, v in _u.items())
                     + '，减仓最多的是 ' + "、".join(f"{i}（{v:+.2f}pp）" for i, v in _d.items())
                     + f'（各门类配对样本 {_lo:,}–{_hi:,} 只——同一只基金两期都有行业表才计入，'
                       f'避免「换了一批基金」被读成「加减仓」）。')
        if np.isfinite(stock):
            _pr, _pn = Q.peer_rank(q, stock)
            if np.isfinite(_pr) and _pn >= 30:
                _txt += f'　本基金在已入库基金中处于第 <b>{_pr:.0f}</b> 分位（N={_pn:,}，非全市场）。'
        h(f'<div class="tiny" style="margin:-4px 0 12px">{_txt}</div>')

    # ---------------- ⑦ 全历史速览 ----------------
    _sec(f"全历史速览：{len(qs)} 期全部列出（一行一季）",
         "每一格都是当期披露值；「本期利润」法定只在半年报/年报披露，季报期写「未披露」而不是留空")
    _rows = []
    for _qq in qs:
        _ra = a[a["quarter"] == _qq]
        _rs = d["q_shares"][d["q_shares"]["quarter"] == _qq] \
            if not d["q_shares"].empty else pd.DataFrame()
        _rh = d["q_hold"][d["q_hold"]["quarter"] == _qq] \
            if not d["q_hold"].empty else pd.DataFrame()
        _ri = d["q_industry"][d["q_industry"]["quarter"] == _qq] \
            if not d["q_industry"].empty else pd.DataFrame()
        _rf = (d["q_fin"][d["q_fin"]["quarter"] == _qq]
               if d.get("q_fin") is not None and not d["q_fin"].empty
               else pd.DataFrame())
        _ra = _pick_row(a, _qq)
        _hhi = Q.industry_hhi(_ri) if len(_ri) else np.nan
        _pf = (float(_rf[_rf["item"] == "本期利润"]["value"].iloc[0]) / 1e8
               if len(_rf) and (_rf["item"] == "本期利润").any() else np.nan)
        _top = (float(_rh.nsmallest(10, "rnk")["ratio_pct"].sum())
                if len(_rh) else np.nan)
        _rows.append({
            "季度": _qq,
            "净资产(亿)": round(float(_ra["nav_yi"]), 2) if _ra is not None else np.nan,
            "股票仓位%": round(float(_ra["stock_pct"]), 2) if _ra is not None else np.nan,
            "债券%": (round(float(_ra["bond_pct"]), 2)
                      if _ra is not None and np.isfinite(_ra["bond_pct"]) else np.nan),
            "现金%": round(float(_ra["cash_pct"]), 2) if _ra is not None else np.nan,
            "净申购(亿份)": (round(float(_rs["purchase_yi"].iloc[0])
                                  - float(_rs["redeem_yi"].iloc[0]), 2)
                             if len(_rs) else np.nan),
            "前十大%": round(_top, 2) if np.isfinite(_top) else np.nan,
            "行业HHI": round(_hhi, 4) if np.isfinite(_hhi) else np.nan,
            "本期利润(亿，累计)": round(_pf, 2) if np.isfinite(_pf) else np.nan,
        })
    _tf = pd.DataFrame(_rows).sort_values("季度", ascending=False)
    for _c in _tf.columns[1:]:
        _tf[_c] = pd.to_numeric(_tf[_c], errors="coerce")
    # 预格式化字符串：NumberColumn 遇到缺失值会把 None 原样打印成 "None"
    _show = _tf.copy()
    for _c in _show.columns[1:]:
        if _c == "净申购(亿份)":
            _show[_c] = _tf[_c].map(lambda v: "—" if not np.isfinite(v) else f"{v:+.2f}")
        elif _c == "行业HHI":
            _show[_c] = _tf[_c].map(lambda v: "—" if not np.isfinite(v) else f"{v:.4f}")
        elif _c.startswith("本期利润"):
            _show[_c] = _tf[_c].map(lambda v: "未披露" if not np.isfinite(v) else f"{v:.2f}")
        else:
            _show[_c] = _tf[_c].map(lambda v: "—" if not np.isfinite(v) else f"{v:.2f}")
    h('<div class="tiny" style="margin:2px 0 4px">'
      f'下表已列出全部 <b>{len(_show)}</b> 期（{qs[0]}–{qs[-1]}），表格内可滚动；'
      '净资产、仓位、债券、现金、份额、前十、行业集中度全部为披露值，'
      '缺失处只有两种：「未披露」（法规不要求）或「—」（该期确实没有这张表）。</div>')
    st.dataframe(_show, hide_index=True, width="stretch",
                 height=min(560, 34 * len(_show) + 40))

    st.download_button(
        f"下载本基金季报速览（{len(_tf)} 季，CSV）",
        _tf.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"季报速览_{code}.csv", mime="text/csv", key="dl_qr")

    # ---------------- 边界 ----------------
    _cov_txt = "｜".join(
        f"{_lab} {d[_k]['quarter'].nunique()} 期"
        for _lab, _k in (("资产配置", "q_asset"), ("行业配置", "q_industry"),
                         ("份额", "q_shares"), ("持仓", "q_hold"),
                         ("买卖", "q_change"), ("债券", "q_bond"),
                         ("财务", "q_fin"))
        if d.get(_k) is not None and not d[_k].empty)
    h(f'<div class="tiny" style="margin-top:12px">本基金已取到：{_cov_txt}'
      f'｜可读季度 {len(qs)} 期（{qs[0]}–{qs[-1]}）</div>')
    h('<div class="cg-note" style="margin-top:6px">'
      '<b>本页覆盖</b>：资产配置（股/债/现金/净资产）、按行业分类的全部股票持仓、'
      '份额变动、前十名股票、累计买卖前 20 名、债券明细、半年报财务指标——全部为季报披露值。'
      '<br><b>口径处理</b>：港股通持仓用的是恒生行业分类，与 A 股的证监会门类不同源，'
      '本页把它汇总成「港股通合计」一条，既不丢信息，也不制造假的可比性。'
      '<br><b>本页不覆盖</b>：① 管理人报告文字（运作分析、市场展望）——数据源只提供数字，'
      '不提供正文；② 持有人结构（户数、机构占比）——半年报/年报才披露；'
      '③ 季度财务指标——法规不要求季报披露。缺失处一律留空，不用替代口径填充。</div>')

    ui.source_footer(h, "fund", "share_change")
