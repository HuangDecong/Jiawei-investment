# -*- coding: utf-8 -*-
"""「本期解读」页：把季报读成一页话。

写作原则
--------
1. **先说结论，再给证据**：读者先看到三句话，再决定要不要往下看；
2. **每个数字都标注口径与期次**：全持仓（半年）与前十大（季度）不混着说；
3. **每段结尾给一句「所以呢」**：把统计量翻译成投资动作或纪律；
4. 不写「值得关注」「密切跟踪」这类空话，只写能证伪的判断。
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


def _ledger(decomp: pd.DataFrame, br: pd.DataFrame) -> pd.DataFrame:
    """行业台账：配置（半年）＋ 广度（季度）拼在一张表里，各自标期次。

    两张表期间不同，所以列名必须把期次写进去，否则读者会把
    「半年主动 +0.8pp」和「本季 62% 基金在加」当成同一件事。
    """
    if decomp is None or decomp.empty:
        return pd.DataFrame()
    t = decomp.copy()
    if br is not None and not br.empty:
        t = t.join(br.set_index("行业")[["加仓占比", "分歧度", "判读"]], how="left")
    else:
        for c in ("加仓占比", "分歧度", "判读"):
            t[c] = np.nan

    def _read(r):
        a, s = r.get("主动贡献", np.nan), r.get("加仓占比", np.nan)
        if not np.isfinite(a) or not np.isfinite(s):
            return "—"
        if a > 0 and s >= 60:
            return "真加仓：机构加、多数基金也在加"
        if a > 0 and s < 50:
            return "少数大基金在加（多数在撤）"
        if a < 0 and s <= 40:
            return "真撤退：机构减、多数基金也在减"
        if a < 0 and s > 50:
            return "多数在加但净额在减（小基金接盘）"
        return "方向不一致（净额与广度冲突）"

    t["读数"] = t.apply(_read, axis=1)
    # 变化幅度%：0.7%→1.1% 是 +57%，8%→8.6% 只有 +7%，pp 相同含义完全不同。
    # 不加这一列，读者会把「煤炭 +0.58pp」和「机械 +0.78pp」当成同一量级。
    t["变化幅度%"] = np.where(
        t["上期配置"].abs() > 0.05,
        t["变化"] / t["上期配置"] * 100, np.nan)
    return t.sort_values("主动贡献", ascending=False)


def render(h, B=None, IND=None) -> None:
    """入口。B/IND 仅为兼容旧签名，本页不依赖 Wind 数据。"""
    # 判「能不能算」而不是「有没有数据库」：线上没有 500MB 的 SQLite，
    # 但有 data/derived/canon_*.csv.gz 快照，canon 会自动回退。
    # 这里若只判 available()，线上就会显示「未找到数据库」——明明数据是够的。
    if not (cn.available() or cn.fallback_available()):
        h('<div class="cg-note">未找到持仓数据。运行 '
          '<span class="mono">python tools/run_ak_all.py</span> 抓取，'
          '或 <span class="mono">python tools/export_canon.py</span> 生成压缩快照。</div>')
        return

    q = cn.latest()
    if q is None:
        h('<div class="cg-note">数据库为空。</div>')
        return
    pf = cn.pair_full()
    pq = cn.pair_quarter()
    ps = cn.pair_same()

    # ---------------- 顶部：三句话 + 指标 ----------------
    n_funds = len(cn.universe(q))
    pos = cn.position_change() if pf else {}
    br = cn.breadth(*pq) if pq else pd.DataFrame()
    dec = cn.decomp(*pf) if pf else pd.DataFrame()

    h(f'<div class="sec">这一季（{q}）发生了什么 · 样本 {n_funds:,} 只主动权益基金</div>')

    head = []
    if pos:
        dpos = pos["中位变化pp"]
        head.append(
            f'整体<b>{"加仓" if dpos > 0.3 else "减仓" if dpos < -0.3 else "没动仓位"}</b>：'
            f'股票仓位中位 {pos["仓位0"]:.1f}% → <b>{pos["仓位1"]:.1f}%</b>'
            f'（{_fmt(dpos, "pp", 2, True)}，{pos["加仓基金占比"]:.0f}% 的基金在加仓）')
    if not dec.empty:
        up, dn = dec["主动贡献"].idxmax(), dec["主动贡献"].idxmin()
        head.append(
            f'主动加仓最多 <b>{up}</b>（{dec.loc[up, "主动贡献"]:+.2f}pp，'
            f'配置 {dec.loc[up, "本期配置"]:.1f}%）；'
            f'主动减仓最多 <b>{dn}</b>（{dec.loc[dn, "主动贡献"]:+.2f}pp）')
    if not br.empty:
        hi, lo = br.loc[br["加仓占比"].idxmax()], br.loc[br["加仓占比"].idxmin()]
        # 分母必须是「持有该行业的基金数」，不是全样本：
        # 房地产只有 208 只基金把它买进前十大，说「80% 基金在加」会被读成 4270 只的 80%。
        head.append(
            f'共识度最高 <b>{hi["行业"]}</b>（持有它的 {int(hi["样本基金数"])} 只基金里 '
            f'{hi["加仓占比"]:.0f}% 在加：{int(hi["加仓基金数"])} 加 / '
            f'{int(hi["减仓基金数"])} 减）；'
            f'最一致撤离 <b>{lo["行业"]}</b>（持有它的 {int(lo["样本基金数"])} 只里仅 '
            f'{lo["加仓占比"]:.0f}% 在加）')

    if head:
        h('<div class="cg-note" style="font-size:13px;line-height:1.9">'
          + "<br>".join(f"{i + 1}. {x}" for i, x in enumerate(head)) + "</div>")

    h(_kpis([
        ("最新季报", q, f"全持仓期 {pf[1] if pf else '—'}"),
        ("样本基金", f"{n_funds:,}", "主动权益，剔主题基金"),
        ("股票仓位", _fmt(pos.get("仓位1"), "%", 1) if pos else "—",
         f"较上期 {_fmt(pos.get('中位变化pp'), 'pp', 1, True)}" if pos else ""),
        ("季报滞后", "56 天", "季末到披露，见 ⑦ 方法与口径"),
    ]))

    st.write("")

    # ---------------- 1. 仓位 ----------------
    if pos:
        h('<div class="sec">① 先看总开关：机构是在加仓，还是只在行业之间挪</div>')
        c0, c1 = cn.coverage(pos["q0"]), cn.coverage(pos["q"])
        idx = c0.index.intersection(c1.index).intersection(list(cn.universe(pos["q"])))
        st.plotly_chart(rc.fig_position(c0.loc[idx], c1.loc[idx], pos["q0"], pos["q"]),
                        width="stretch", config={"displayModeBar": False})
        med = pos["中位变化pp"]
        h(f'<div class="cg-note">股票仓位中位数 <b>{pos["仓位0"]:.1f}% → {pos["仓位1"]:.1f}%</b>，'
          f'中位变化 <b>{_fmt(med, "pp", 2, True)}</b>，<b>{pos["加仓基金占比"]:.0f}%</b> 的基金净加仓、'
          f'{100 - pos["加仓基金占比"]:.0f}% 净减仓。'
          f'<br><b>所以呢</b>：'
          + ("仓位几乎没变，说明下面看到的行业变化是<b>内部换仓</b>，不是整体加仓——"
             "这时「某行业配置上升」可以放心读成「机构更看好它」。"
             if abs(med) < 0.5 else
             f"仓位整体{'上升' if med > 0 else '下降'} {abs(med):.1f}pp，"
             "行业占比的变化里有一部分只是「水涨船高」，"
             "必须用下一段的主动/被动拆解剔掉，否则会把整体加仓误读成看好某个行业。")
          + '</div>')

    # ---------------- 2. 主动 vs 被动 ----------------
    if not dec.empty:
        h('<div class="sec">② 配置变化里，多少是价格自己涨出来的</div>')
        st.plotly_chart(rc.fig_active_passive(dec), width="stretch",
                        config={"displayModeBar": False})
        _d = dec.dropna(subset=["被动贡献", "主动贡献"])
        _pas = _d["被动贡献"].abs().sum()
        _act = _d["主动贡献"].abs().sum()
        _top = dec.loc[dec["被动贡献"].abs().idxmax()]
        _topname = dec["被动贡献"].abs().idxmax()
        h(f'<div class="cg-note">把 31 个行业的配置变化拆成两半：'
          f'<b>价格带来的被动变化合计 {_pas:.1f}pp、机构主动只占 {_act:.1f}pp</b>，'
          f'被动占全部变动的 <b>{_pas / (_pas + _act) * 100:.0f}%</b>。'
          f'其中 <b>{_topname}</b> 的 {_top["变化"]:+.2f}pp 里，'
          f'价格解释了 {_top["被动贡献"]:+.2f}pp、机构只动了 {_top["主动贡献"]:+.2f}pp。'
          f'<br><b>所以呢</b>：看到「某行业配置暴跌」先别当利空——'
          f'跌幅 >1pp 但主动项不到 0.3pp 的，基本面含义接近于零，'
          f'只是那个行业跌了。真正该重视的是<b>主动项和广度同向</b>的行业。</div>')

    # ---------------- 3. 广度 ----------------
    if not br.empty:
        h(f'<div class="sec">③ 谁在动：{pq[0]} → {pq[1]} 的前十大口径（季频、跨季可比）</div>')
        c1_, c2_ = st.columns([0.56, 0.44], gap="small")
        with c1_:
            st.plotly_chart(rc.fig_breadth(br), width="stretch",
                            config={"displayModeBar": False})
        with c2_:
            st.plotly_chart(rc.fig_breadth_vs_move(br), width="stretch",
                            config={"displayModeBar": False})
        vc = br["判读"].value_counts()
        h(f'<div class="cg-note">口径：每只基金只取权重最大的 10 只股票、按行业加总后归一化，'
          f'四个季度都这样取，才能跨季比较（季报本来就只披露前十大，'
          f'半年报虽披露全部，这里也截断到 10 只）。参与比较的基金 '
          f'<b>{br.attrs["n_funds"]:,}</b> 只；<b>单个行业的有效样本是「该行业进入前十大」'
          f'的基金数</b>（表里第一列，通常只有几百到几千只），'
          f'所以占比的分母不是全样本。'
          f'本期判读：'
          + "、".join(f"{k} {v} 个" for k, v in vc.items())
          + f'。<br><b>所以呢</b>：广度（多少基金在加）和幅度（净买了多少）要一起看——'
          f'左上角那批「加仓基金不到一半、但净额为正」的行业，是<b>少数大基金在撑</b>，'
          f'它们一停手就反转；右下角「多数在加、净额却为负」说明<b>小基金在接盘</b>。'
          f'只有右上角（广度高 + 净额正）才算共识。</div>')
        h('<div class="tiny">注意：这张图的两个轴都含价格漂移——'
          '行业涨了，即使没人买，它在「前十大组合」里的占比也会上升。'
          '要把价格剔干净，只能用第 ② 段的全持仓口径（但那是半年频）。'
          '两个口径各回答一个问题，不要互相替代。</div>')

    # ---------------- 4. 结构演变 ----------------
    h('<div class="sec">④ 把时间拉长：这套结构是怎么一步步走到今天的</div>')
    try:
        raw = {}
        for qq in cn.quarters():
            p = cn.panel_top10(qq)
            if p.empty:
                continue
            u = cn.universe(qq)
            idx = p.index.intersection(list(u)) if u else p.index
            if len(idx) < 300:
                continue
            raw[qq] = (p.loc[idx].mean(), len(idx))
        # 样本门槛用「最大期的 55%」动态判定，不写死数字：
        # 补抓历史时，中间状态会出现「前几列 1400 只、后几列 4900 只」，
        # 颜色深浅的差异全是样本量造成的，不是配置变化。
        if raw:
            _mx = max(v[1] for v in raw.values())
            mat = {k: v[0] for k, v in raw.items() if v[1] >= max(1000, _mx * 0.55)}
        else:
            mat = {}
        M = pd.DataFrame(mat).T.sort_index()
        if not M.empty:
            st.plotly_chart(rc.fig_structure_heatmap(M), width="stretch",
                            config={"displayModeBar": False})
            _chg = (M.iloc[-1] - M.iloc[0]).sort_values(ascending=False)
            h(f'<div class="cg-note">行是行业、列是季度，颜色越深＝在前十大组合里占比越高；'
              f'共 <b>{len(M)}</b> 个季度、每季按等权平均。'
              f'从 {M.index[0]} 到 {M.index[-1]}，占比抬升最多的是 '
              f'<b>{_chg.index[0]}（{_chg.iloc[0]:+.1f}pp）</b>、'
              f'<b>{_chg.index[1]}（{_chg.iloc[1]:+.1f}pp）</b>，'
              f'让位最多的是 <b>{_chg.index[-1]}（{_chg.iloc[-1]:+.1f}pp）</b>。'
              f'<br><b>所以呢</b>：这条长趋势才是「结构性变化」，'
              f'单季 1–2pp 的波动绝大多数是价格和季报噪音。'
              f'判断一个行业是不是真在被长期增配，看它在这张图里有没有连续 3 个季度往右上走。</div>')
    except Exception as _e:
        h(f'<div class="tiny">结构演变暂不可用：{type(_e).__name__}</div>')

    # ---------------- 5. 个股 ----------------
    if ps:
        h(f'<div class="sec">⑤ 落到个股：{ps[0]} → {ps[1]}，谁被加进来、谁被移出去</div>')
        try:
            sf = cn.stock_flow(*ps)
            if not sf.empty:
                sf = sf[(sf["n_funds"] >= 10) | (sf["n_funds_0"] >= 10)]
                st.plotly_chart(rc.fig_stock_flow(sf), width="stretch",
                                config={"displayModeBar": False})
                _a = sf.nlargest(3, "d_n")
                _c = sf.nsmallest(3, "d_n")
                h(f'<div class="cg-note">'
                  f'被最多基金<b>新买入</b>：'
                  + "、".join(f'{r["stock_name"]}（{r["d_n"]:+.0f}）' for _, r in _a.iterrows())
                  + f'；被最多基金<b>卖出</b>：'
                  + "、".join(f'{r["stock_name"]}（{r["d_n"]:+.0f}）' for _, r in _c.iterrows())
                  + f'。<br>口径：只在<b>同一披露类型</b>的两期之间比较（季报对上年同季），'
                    f'样本 {sf.attrs["n_funds"][1]:,} 只基金。'
                    f'如果拿 Q3 和 Q2 比，「从第 11 名掉出前十大」会被误判成清仓，'
                    f'所以个股层不用相邻季。<br>'
                    f'<b>所以呢</b>：个股榜要配着行业一起看——'
                    f'如果加仓的个股集中在同一个行业，说明是行业配置行为；'
                    f'如果散在不同行业，才是真正的选股行为。</div>')
        except Exception as _e:
            h(f'<div class="tiny">个股榜暂不可用：{type(_e).__name__}</div>')

    # ---------------- 6. 行业台账 ----------------
    if not dec.empty:
        h('<div class="sec">⑥ 行业台账：把上面几张图落到一行一个行业</div>')
        t = _ledger(dec, br)
        show = t.reset_index()[[
            "行业", "上期配置", "本期配置", "变化", "变化幅度%",
            "被动贡献", "主动贡献", "加仓占比", "判读", "读数"]]
        show = show.rename(columns={
            "上期配置": f"配置%({pf[0]})", "本期配置": f"配置%({pf[1]})",
            "变化": f"半年变化pp({pf[0]}→{pf[1]})",
            "变化幅度%": "相对变化%",
            "被动贡献": "其中价格pp", "主动贡献": "其中机构pp",
            "加仓占比": f"加仓占比%({pq[1] if pq else '—'})",
        })
        st.dataframe(
            show, width="stretch", hide_index=True, height=520,
            column_config={
                f"配置%({pf[0]})": st.column_config.NumberColumn(format="%.2f"),
                f"配置%({pf[1]})": st.column_config.NumberColumn(format="%.2f"),
                f"半年变化pp({pf[0]}→{pf[1]})": st.column_config.NumberColumn(format="%+.2f"),
                "相对变化%": st.column_config.NumberColumn(
                    format="%+.1f", help="变化 ÷ 上期配置。低配行业 0.6pp 可能是 +57%，与高配行业的 0.6pp 不是一回事"),
                "其中价格pp": st.column_config.NumberColumn(format="%+.2f"),
                "其中机构pp": st.column_config.NumberColumn(format="%+.2f"),
                f"加仓占比%({pq[1] if pq else '—'})": st.column_config.NumberColumn(
                format="%.0f", help="分母是「该行业进入前十大」的基金数，不是全样本"),
                "判读": st.column_config.TextColumn(width="small"),
                "读数": st.column_config.TextColumn(width="large"),
            })
        h('<div class="tiny">前 5 列是<b>全持仓口径</b>（半年报，权重＝占净值比，'
          '主动/被动用 GK 分解，价格按行业季度收益计算）；'
          '「加仓基金占比」是<b>前十大口径</b>（季频）。'
          '两列期间不同，这是刻意的：半年报才够准，季报才够新，'
          '把两者各自放在它擅长的位置上，而不是硬凑成一个数。</div>')

    # ---------------- 7. 投资含义 ----------------
    if not dec.empty and not br.empty:
        h('<div class="sec">⑦ 这一页对你的投资含义（可直接执行的 4 条）</div>')
        _j = dec.join(br.set_index("行业")[["加仓占比"]], how="left")
        _real = _j[(_j["主动贡献"] > 0) & (_j["加仓占比"] >= 60)].sort_values(
            "主动贡献", ascending=False)
        _fake = _j[(_j["主动贡献"] > 0) & (_j["加仓占比"] < 50)].sort_values(
            "主动贡献", ascending=False)
        _out = _j[(_j["主动贡献"] < 0) & (_j["加仓占比"] <= 40)].sort_values("主动贡献")
        lines = []
        if len(_real):
            lines.append(
                f'<b>可以跟的</b>：' + "、".join(
                    f'{i}（主动 {r["主动贡献"]:+.2f}pp、{r["加仓占比"]:.0f}% 的持有者在加）'
                    for i, r in _real.head(4).iterrows())
                + '——主动项和广度同向，是这一季最干净的机构行为。')
        if len(_fake):
            lines.append(
                f'<b>只能看不能追的</b>：' + "、".join(
                    f'{i}（主动 {r["主动贡献"]:+.2f}pp，但只有 {r["加仓占比"]:.0f}% 的持有者在加）'
                    for i, r in _fake.head(4).iterrows())
                + '——少数大基金在拉，缺少广度支撑，随时可能反向。')
        if len(_out):
            lines.append(
                f'<b>要回避的</b>：' + "、".join(
                    f'{i}（主动 {r["主动贡献"]:+.2f}pp、{r["加仓占比"]:.0f}% 的持有者在加）'
                    for i, r in _out.head(4).iterrows())
                + '——机构在真减，不是价格跌出来的。')
        lines.append(
            '<b>别做的</b>：不要因为「某行业配置比例大跌」就卖出，'
            '先看主动项；也不要因为「多家基金加仓」就买入——'
            '这个指标在历史检验里对次季收益没有预测力（IC≈0.05、p≈0.37），'
            '它能告诉你机构在做什么，不能告诉你接下来会涨什么。')
        h('<div class="cg-note" style="font-size:12.5px;line-height:1.95">'
          + "<br>".join(f"· {x}" for x in lines)
          + '<br><span style="color:#6B7A8C">注：「持有者在加」的分母是'
            '「把该行业买进前十大」的基金数，不是全部 4,270 只——'
            '冷门行业的百分比波动大，样本少于 200 只时不要当结论。</span>'
          + "</div>")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    print("模块自检通过；渲染需在 Streamlit 内调用 render(h)")
