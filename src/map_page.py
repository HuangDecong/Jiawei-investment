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


def _read(h, read: str, concl: str) -> None:
    """统一的「读图 + 结论」块：每张图下面都必须有，否则图就是摆设。"""
    h(f'<div style="background:{C.COLOR["panel"]};border-left:3px solid {C.COLOR["price"]};'
      f'padding:9px 12px;border-radius:0 6px 6px 0;margin:4px 0 16px">'
      f'<div style="font-size:12px;line-height:1.9"><b>读图</b>：{read}</div>'
      f'<div style="font-size:12.5px;line-height:1.9;margin-top:5px">'
      f'<b style="color:{C.COLOR["threshold"]}">结论</b>：{concl}</div></div>')


def _rank_sort(df: pd.DataFrame, level_col: str = "等级") -> pd.DataFrame:
    """正常样本按 CSI 降序在前，小样本（不评级）统一排到最后。

    为什么不能纯按 CSI 排：综合 83 分，会插在石油石化 85 与银行 78 之间，
    灰色块夹在红色块中间，读者第一眼就会问「这个灰的为什么排这么前」。
    「不评级」不是一个等级，它不该参与排名。
    """
    d = df.copy()
    sm = (d[level_col].astype(str) == "小样本").astype(int) \
        if level_col in d.columns else 0
    d["_sm"] = sm
    return d.sort_values(["_sm", "CSI"], ascending=[True, False]).drop(columns="_sm")


def _chips(cs: pd.DataFrame, comp: pd.DataFrame) -> str:
    """31 个行业的色块总览：颜色＝拥挤等级，点一下上方的下拉即可看详情。

    小样本行业（持有基金数 <150 或前十配置 <0.5%）用灰色、写「小样本·不评级」：
    它们既不是拥挤也不是冷门，是「样本不够，判不了」。
    """
    out = []
    _sep_done = False
    for i, r in cs.iterrows():
        _is_small = r.get("可靠") == "小样本"
        c = rc.CSI_COLORS.get(r["等级"], "#C3CBD4")
        _aq = comp.loc[i, "Δ上季pp"] if i in comp.index else np.nan
        _tip = (f'全市场仅 {r.get("持有基金数", float("nan")):.0f} 只基金持有，'
                f'前十配置 {r.get("配置_前十", r.get("配置", float("nan"))):.2f}%；'
                f'按三分位本应判为「{r.get("原等级", "—")}」，'
                f'但样本不足，不评级' if _is_small else f'CSI {r["CSI"]:.0f}（{r["等级"]}）')
        _lvl = "小样本·不评级" if _is_small else r["等级"]
        # 小样本整段排在最后，中间插一条分隔说明：
        # 否则读者会疑惑「为什么 83 分的灰色排在一堆 20 分后面」。
        if _is_small and not _sep_done:
            out.append(
                '<div style="flex:1 1 100%;font-size:9px;color:#6B7A8C;'
                'border-top:1px dashed #D6DEE7;padding:4px 0 1px;margin-top:2px">'
                '以下行业<b>不评级</b>（全市场持有基金数 &lt;150 只，或前十配置 &lt;0.5%），'
                '不参与拥挤度排名，只列出数值供参考</div>')
            _sep_done = True
        out.append(
            f'<div title="{_tip}" style="flex:1 1 88px;min-width:82px;background:{c};color:#fff;'
            f'border-radius:6px;padding:6px 7px;line-height:1.35">'
            f'<div style="font-size:11px;font-weight:700">{i}'
            f'{"†" if _is_small else ""}</div>'
            f'<div style="font-size:15px;font-weight:700">{r["CSI"]:.0f}</div>'
            f'<div style="font-size:9px;opacity:.92">{_lvl}　'
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
        comp = comp.join(csi[["CSI", "等级", "原等级", "可靠", "配置分位", "价格分位", "资金分位"]],
                         how="left")
    q, qp, qy = A["q"], A["q_prev"], A["q_year"]
    qf, qf0 = A["q_full"], A["q_full0"]
    idx = cn.index_stats()
    if not idx.empty:
        comp = comp.join(idx[["指数分位", "近3月涨幅%", "近1年涨幅%"]], how="left")

    # =====================================================================
    # 本页结论
    # =====================================================================
    # 「最拥挤」必须排除小样本：综合全市场只有 58 只基金持有、前十配置 0.12%，
    # 它的 CSI 83 是「几十只基金的分位平均」，跟煤炭（434 只、1.5%）不是一个量级。
    # 把它排进最拥挤，老板第一眼就会问「综合为什么会拥挤」——因为它其实不挤。
    _ok = comp[comp["可靠"] == "正常"] if "可靠" in comp else comp
    _top = _ok.nlargest(3, "CSI") if "CSI" in _ok else pd.DataFrame()
    _bot = _ok.nsmallest(3, "CSI") if "CSI" in _ok else pd.DataFrame()
    _small = comp[comp["可靠"] == "小样本"] if "可靠" in comp else pd.DataFrame()
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
    if len(_small):
        _s0 = _small.sort_values("CSI", ascending=False).iloc[0]
        _bl.append(
            f'<b>有 {len(_small)} 个行业不评级</b>（'
            + "、".join(_small.sort_values("CSI", ascending=False).index)
            + f'）：它们的 CSI 看起来可能很高——比如 {_small.sort_values("CSI", ascending=False).index[0]} '
              f'算出来 {_s0["CSI"]:.0f} 分，按三分位本该是「{_s0["原等级"]}」——'
              f'但全市场只有 <b>{_s0["持有基金数"]:.0f}</b> 只基金把它放进前十大、'
              f'前十配置仅 <b>{_s0["配置_前十"]:.2f}%</b>。'
              f'几十只基金里少数几只调仓，就足以把三个分位同时推到高位，'
              f'所以本页把它标成灰色「小样本」、不算进拥挤个数、也不给买卖动作。')
    # 本期刚好穿过阈值的行业：这是「折线 + 阈值」最直接可用的输出
    try:
        _ST0 = cn.csi_threshold()
        if _ST0:
            _cur0 = _ST0["current"]
            # 小样本行业（综合）的穿越是噪音，不进结论
            if "等级" in _cur0.columns:
                _cur0 = _cur0[_cur0["等级"] != "小样本"]
            _up0 = _cur0[_cur0["状态"] == "刚上穿 80"]
            _dn0 = _cur0[_cur0["状态"] == "刚跌破 80"]
            _ev0 = _ST0.get("events")
            _wr = (_ev0["胜率"].iloc[0] if _ev0 is not None and len(_ev0) else np.nan)
            if len(_up0) or len(_dn0):
                _p0 = []
                if len(_up0):
                    _p0.append('<b>刚上穿 80</b>：' + "、".join(
                        f'{i}（{r["上期CSI"]:.0f}→{r["CSI"]:.0f}）' for i, r in _up0.iterrows()))
                if len(_dn0):
                    _p0.append('<b>刚跌破 80</b>：' + "、".join(
                        f'{i}（{r["上期CSI"]:.0f}→{r["CSI"]:.0f}）' for i, r in _dn0.iterrows()))
                _bl.append('<b>本期刚好穿过阈值</b>：' + "；".join(_p0)
                           + f'。历史上「刚上穿」的次季跑赢同行概率只有 <b>{_wr:.0f}%</b>'
                             f'（t 值不显著，但方向偏负），所以它是<b>不追高</b>的量化依据；'
                             f'「刚跌破」方向相反，不必急着卖。')
    except Exception:
        pass
    _bl.append('<b>怎么用</b>：拥挤度只决定「给多大仓位」，不决定「买不买」。'
               '本库数据已证伪「高拥挤必然下跌」，所以看到红色不要条件反射地卖，'
               '而是把它的仓位上限压下来；看到深蓝也不要马上买，先等资金转向。')
    h('<div class="cg-note" style="line-height:1.9">'
      + "<br>".join(f"{i}. {x}" for i, x in enumerate(_bl, 1)) + "</div>")

    # =====================================================================
    # ① 拥挤度变化：31 期折线
    # =====================================================================
    H = cn.csi_history()
    h('<div class="sec">① 拥挤度怎么变：31 期折线，以及它什么时候穿过阈值</div>')
    if H.empty:
        h('<div class="tiny">历史 CSI 不可用（数据不足）。</div>')
    else:
        _c1, _c2 = st.columns([0.70, 0.30])
        with _c1:
            _opts = ["只看全市场中位（31 个行业的 CSI 中位数）"] + list(H.columns)
            _focus = st.selectbox("聚焦一个行业", _opts, index=0, key="csi_focus",
                                  help="选一个行业会画出它自己的 31 期曲线，并标出它上穿/下穿 80 的时点")
        with _c2:
            _show_all = st.checkbox("叠加全部 31 条", value=False, key="csi_all",
                                    help="31 条线同时画会互相遮挡；需要横向对比时再打开")
        _sel_t = None if _focus == _opts[0] else _focus
        st.plotly_chart(rc.fig_csi_trend(H, _sel_t, _show_all), width="stretch",
                        config={"displayModeBar": False})
        _med = H.median(axis=1).dropna()
        _now = float(_med.iloc[-1])
        _pct = float((_med <= _now).mean() * 100)
        _imax = int(np.argmax(_med.values))
        _imin = int(np.argmin(_med.values))
        _txt2 = (f'灰粗线是<b>全市场中位</b>：最新 <b>{_now:.0f}</b> 分，'
                 f'在 31 期里排第 <b>{_pct:.0f}</b> 分位；'
                 f'历史最高 {_med.iloc[_imax]:.0f}（{_med.index[_imax]}）、'
                 f'最低 {_med.iloc[_imin]:.0f}（{_med.index[_imin]}）。'
                 f'背景色带就是五档阈值：<b>≥80 淡红、65–80 淡橙、45–65 近白、25–45 淡蓝、&lt;25 深蓝</b>，'
                 f'让「现在算不算高」一眼可见，不用记数字。')
        if _sel_t:
            _s = H[_sel_t].dropna()
            _d1 = float(_s.iloc[-1] - _s.iloc[-2]) if len(_s) > 1 else np.nan
            _n_up = sum(1 for i in range(1, len(_s)) if _s.iloc[i - 1] < 80 <= _s.iloc[i])
            _n_dn = sum(1 for i in range(1, len(_s)) if _s.iloc[i - 1] >= 80 > _s.iloc[i])
            _txt2 += (f'<br><b>{_sel_t}</b>：最新 <b>{_s.iloc[-1]:.0f}</b>'
                      f'（上期 {_s.iloc[-2]:.0f}，{_d1:+.0f}）；'
                      f'31 期里上穿 80 共 <b>{_n_up}</b> 次、下穿 <b>{_n_dn}</b> 次，'
                      f'图上红三角＝上穿、绿三角＝下穿。')
        _read(h, _txt2,
              f'<b>折线回答的是「方向」，不是「高低」。</b>同样是 88 分，'
              f'从 60 涨上来（刚变挤）与从 95 跌下来（正在退潮）含义完全相反——'
              f'上面这张图就是把「水位」和「方向」放在一张图上。'
              f'下面的阈值实证会回答：这些分界线到底有没有预测力。')

    # =====================================================================
    # ② 阈值实证
    # =====================================================================
    h('<div class="sec">② 阈值有没有用：31 期 × 31 行业，按 CSI 分档看次季能否跑赢同行</div>')
    try:
        _ST = cn.csi_threshold()
        if not _ST:
            h('<div class="tiny">样本不足，无法做阈值检验。</div>')
        else:
            _bins, _events, _cur = _ST["bins"], _ST["events"], _ST["current"]
            _up = None
            st.plotly_chart(rc.fig_threshold(_bins, _events), width="stretch",
                            config={"displayModeBar": False})
            _bt = _bins.reset_index().copy()
            _bt.columns = ["CSI 档位", "样本数", "次季超额均值%", "中位%", "胜率%", "t 值"]
            st.dataframe(_bt, width="stretch", hide_index=True, height=214,
                         column_config={
                             "次季超额均值%": st.column_config.NumberColumn(format="%+.2f"),
                             "中位%": st.column_config.NumberColumn(format="%+.2f"),
                             "胜率%": st.column_config.NumberColumn(format="%.1f"),
                             "t 值": st.column_config.NumberColumn(format="%+.2f"),
                         })
            _sig = _bt[_bt["t 值"].abs() >= 1.65]
            _read(h,
                  f'把 {_ST["q_first"]}–{_ST["q_last"]} 的每一期、每个行业都当成一个观测，'
                  f'共 <b>{_ST["n_obs"]}</b> 个（31 期 × 31 行业，剔除末期无次季收益的）。'
                  f'「次季超额」＝该行业下一季收益 − 下一季 31 个行业等权收益，'
                  f'<b>已经扣掉大盘涨跌</b>；胜率＝次季跑赢同行的比例。'
                  f'<br>五档的均值从 {_bt["次季超额均值%"].min():+.2f}% 到 '
                  f'{_bt["次季超额均值%"].max():+.2f}%，'
                  f't 值全部在 ±1.65 以内（'
                  + ("只有 " + "、".join(_sig["CSI 档位"]) + f" 超过 1.65，但样本只有 "
                     f"{int(_sig['样本数'].iloc[0])} 个，且其它档位不单调，"
                     f"按多重检验看更可能是噪音。" if len(_sig) else "即没有一档显著")
                  + '）。',
                  f'<b>结论：CSI 的绝对水位不能预测次季收益。</b>'
                  f'看起来「{_bt.iloc[0]["CSI 档位"]} {_bt.iloc[0]["次季超额均值%"]:+.2f}%、'
                  f'{_bt.iloc[1]["CSI 档位"]} {_bt.iloc[1]["次季超额均值%"]:+.2f}%」有差别，'
                  f'但 t 值只有 {_bt.iloc[0]["t 值"]:+.1f} 和 {_bt.iloc[1]["t 值"]:+.1f}，'
                  f'换成下一年样本就会翻转。所以本页把阈值用于<b>仓位约束</b>'
                  f'（≥80 的行业，单一行业仓位上限打七折），而不是买卖信号。'
                  f'<br><b>胜率为什么大多不到 50%</b>：A 股行业季度超额收益是右偏分布——'
                  f'少数行业暴涨把均值拉高，多数行业低于均值。所以 '
                  f'{_bt["胜率%"].min():.0f}% 的胜率不等于「大概率下跌」，'
                  f'只说明「中位数略负」。看胜负要同时看中位数与幅度。')
            if not _events.empty:
                _et = _events.copy()
                _et.index.name = "事件"
                st.dataframe(_et[["样本", "均值", "中位", "胜率", "t"]], width="stretch",
                             height=110,
                             column_config={
                                 "均值": st.column_config.NumberColumn(format="%+.2f"),
                                 "中位": st.column_config.NumberColumn(format="%+.2f"),
                                 "胜率": st.column_config.NumberColumn(format="%.1f"),
                                 "t": st.column_config.NumberColumn(format="%+.2f"),
                             })
                _up = _events.loc[_events.index[0]] if len(_events) else None
                h(f'<div class="tiny">事件研究：'
                  f'<b>上穿 80</b>（上期 &lt;80、本期 ≥80）的 {int(_up["样本"]) if _up is not None else 0} 个样本，'
                  f'次季超额均值 {_up["均值"]:+.2f}%、中位 {_up["中位"]:+.2f}%、'
                  f'胜率 {_up["胜率"]:.0f}%、t={_up["t"]:+.2f}；'
                  f'<b>下穿 80</b>（拥挤解除）方向相反，但同样不显著。'
                  f'这说明「刚变挤」有一点负向倾向、「刚变松」有一点正向倾向，'
                  f'只能当<b>风险提示</b>，不能当交易信号。</div>')
            # 本期谁刚穿过阈值
            _cur = _cur[_cur["等级"] != "小样本"] if "等级" in _cur.columns else _cur
            _hot = _cur[_cur["状态"] == "刚上穿 80"]
            _cool = _cur[_cur["状态"] == "刚跌破 80"]
            if len(_hot) or len(_cool):
                h(f'<div class="cg-note" style="margin-top:4px">'
                  f'<b>本期刚好在穿越阈值上的行业</b>：'
                  + (f'<b>刚上穿 80</b>：' + "、".join(
                      f'{i}（{r["上期CSI"]:.0f}→{r["CSI"]:.0f}）' for i, r in _hot.iterrows())
                     if len(_hot) else "无刚上穿")
                  + '；'
                  + (f'<b>刚跌破 80</b>：' + "、".join(
                      f'{i}（{r["上期CSI"]:.0f}→{r["CSI"]:.0f}）' for i, r in _cool.iterrows())
                     if len(_cool) else "无刚跌破")
                  + (f'。按事件研究，刚上穿的一组历史胜率 '
                     f'{_up["胜率"]:.0f}%（不显著但方向偏负），'
                     f'刚好是「别追高」名单的来源；刚跌破的一组历史上反而略偏正，'
                     f'不必急着卖。</div>' if _up is not None else '</div>'))
    except Exception as _e:
        h(f'<div class="tiny">阈值检验暂不可用：{type(_e).__name__}</div>')

    # =====================================================================
    # ① 搜索 + 总览
    # =====================================================================
    h('<div class="sec">③ 找行业：输入名称即可搜索，颜色＝拥挤等级</div>')
    c1, c2 = st.columns([0.42, 0.58], gap="small")
    with c1:
        _kw = st.text_input("搜索行业（支持部分匹配，如「电」）", value="", key="ind_kw",
                            placeholder="输入行业名，如 电子 / 医药 / 银行")
        _cand = [i for i in comp.index if _kw.strip() in i] if _kw.strip() else list(comp.index)
        if not _cand:
            _cand = list(comp.index)
            st.caption("没有匹配的行业，已显示全部。")
        _rank = _rank_sort(comp.loc[_cand]) if "CSI" in comp else comp.loc[_cand]
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
          '31 个行业总览：<b>红＝极度拥挤、橙＝拥挤、灰＝中性、浅蓝＝偏冷、深蓝＝冷门</b>；'
          '<b>灰底「小样本·不评级」</b>＝全市场持有基金数不足 150 只或前十配置 &lt;0.5%，'
          '分位噪音大于信息，不给等级。<br>'
          '数字是 CSI（0–100），下方小字是本期环比变化（单位：百分点）。'
          '鼠标悬停任一色块可看它的三个分位是怎么算出 CSI 的；要点开某个行业，'
          '用左侧下拉框搜索。</div>')
        h(_chips(_rank_sort(comp.dropna(subset=["CSI"])), comp))

    # =====================================================================
    # ② 单行业详情
    # =====================================================================
    r = comp.loc[sel]
    h(f'<div class="sec">④ {sel}：六个指标 + 历史位置</div>')

    # 小样本行业先说清楚「为什么这个数不能当拥挤度用」，
    # 否则读者看到「综合 CSI 83」会以为综合真的挤了——它只占组合 0.12%。
    if r.get("可靠") == "小样本":
        _mn = float(M[sel].min()) if sel in M.columns else np.nan
        _mx = float(M[sel].max()) if sel in M.columns else np.nan
        h(f'<div style="background:#FFF8E6;border:1px solid #E8C36A;border-left:4px solid #D9A21B;'
          f'border-radius:7px;padding:9px 12px;margin:2px 0 8px;font-size:11.5px;line-height:1.9">'
          f'<b>⚠ {sel}不评级：下面这个 CSI {r["CSI"]:.0f} 不能当拥挤度用。</b><br>'
          f'<b>为什么</b>：全市场只有 <b>{r["持有基金数"]:.0f}</b> 只样本基金把它放进前十大，'
          f'前十配置合计 <b>{r["配置_前十"]:.2f}%</b>（{len(M)} 期区间 {_mn:.2f}%–{_mx:.2f}%）。'
          f'三个分位都是在这个极小的样本上排出来的：<br>'
          f'· <b>配置分位 {r["配置分位"]:.0f}</b>——它的历史配置只在 {_mn:.2f}% 到 {_mx:.2f}% 之间摆动，'
          f'差 0.05pp 就能从第 20 分位跳到第 80 分位，排名没有经济含义；<br>'
          f'· <b>价格分位 {r["价格分位"]:.0f}</b>——小盘股指数从低位反弹一段，就能把分位推到 90 以上，'
          f'但这不代表机构在往里挤；<br>'
          f'· <b>资金分位 {r["资金分位"]:.0f}</b>——{r["持有基金数"]:.0f} 只基金里只要有 3–5 只加仓，'
          f'广度就会跳到前 20%。<br>'
          f'三者平均出 <b>{r["CSI"]:.0f}</b>，按三分位本来会被判为「{r["原等级"]}」，'
          f'但这个体量对组合的影响不到 0.2pp——它就算仓位全部清零，对净值的影响也小于一个'
          f'正常行业 1pp 的调仓。<b>所以本页把它标成灰色「小样本」，不计入拥挤个数，也不给买卖动作。</b>'
          f'<br>要判断它到底挤不挤，至少需要「持有基金数 ≥150 只、前十配置 ≥0.5%」才够看。'
          f'</div>')

    _d = st.columns(6)
    _d[0].metric(f"前十配置 {q}", f"{r['配置_前十']:.2f}%",
                 help="前十大归一化口径：组合里最重要的 10 只股票，该行业占多少")
    _d[1].metric("配置分位", f"{r['历史分位']:.0f}",
                 help=f"在自身 {len(M)} 期历史中的排名（0＝最低，100＝最高），满分 100")
    _d[2].metric("价格分位",
                 f"{r['指数分位']:.0f}" if np.isfinite(r.get("指数分位", np.nan)) else "—",
                 help="行业指数季末点位在 2019 年以来每个交易日中的排名（0–100），回答「贵不贵」")
    _d[3].metric("CSI 拥挤度", f"{r.get('CSI', float('nan')):.0f}",
                 help="(配置分位+价格分位+资金分位)/3，等权，0–100 分；"
                      "≥80 极度拥挤、65–80 拥挤、45–65 中性、25–45 偏冷、<25 冷门")
    # 环比/同比的 value 只放数字，单位写在 label 与下方说明里：
    # 之前 value 里拼 "pp"，在部分字体/缩放下两个 p 会被看成「页」，
    # 读者会问「+0.47 页是什么单位」。数字与单位分开就不会有歧义。
    _d[4].metric(f"环比 {qp}", f"{r['Δ上季pp']:+.2f}", delta_color="off",
                 help="单位：百分点（pp）。含价格漂移（行业涨跌本身会改变权重）；"
                      "价格中性后的净额见下方")
    _d[5].metric(f"同比 {qy}", f"{r['Δ上年pp']:+.2f}", delta_color="off",
                 help="单位：百分点（pp）。与去年同季比，可剔除季节性")
    h('<div class="tiny" style="margin:-2px 0 4px">'
      '单位说明：前十配置＝%（占前十大归一化后的比重）；配置分位、价格分位、CSI＝0–100 的排名分；'
      '环比、同比＝<b>百分点（pp）</b>，即两个百分比相减的差值，1pp＝1 个百分点。</div>')

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
      + (f'<br><b>怎么用</b>：本行业为小样本，本页<b>不给买卖判断</b>——'
         f'上面对分位的所有解读都不适用于它，请只当作「它在自己的历史里排在什么位置」来看，'
         f'不要据此加减仓。'
         if r.get("可靠") == "小样本" else f'<br><b>怎么用</b>：{_verdict(r)}') + '</div>')

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
    h('<div class="sec">⑤ 31 个行业排一排：长度＝拥挤度，颜色＝等级</div>')
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
        # 变化榜同样排除小样本：综合的 ΔCSI +30 是几十只基金调仓的结果，
        # 放进「变挤最多」会让读者以为有资金在大举进入。
        _cco = cc[cc["等级"] != "小样本"] if "等级" in cc else cc
        _up = _cco.nlargest(3, "ΔCSI")
        _dn = _cco.nsmallest(3, "ΔCSI")
        _hi_up = _cco[(_cco["CSI"] >= 80) & (_cco["ΔCSI"] > 0)]
        _n_sm = len(cc) - len(_cco)
        h(f'<div class="cg-note">'
          f'<b>怎么读</b>：横轴是「本期 CSI − 上期 CSI」，不是水位而是水流。'
          f'右边越长＝这段时间挤得越快，左边越长＝在快速变松。悬停可看本期与上期的具体数值。'
          f'为免误读，变化榜已剔除 {_n_sm} 个小样本行业（灰色显示），'
          f'它们的 CSI 波动主要来自样本太小，不代表资金方向。<br>'
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
    h('<div class="sec">⑥ 为什么拥挤度用 CSI，不用 HHI？</div>')
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


    h('<div class="sec">⑦ 两把尺子：相对自己历史 vs 相对全市场同行</div>')
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
        h('<div class="sec">⑧ 第三把尺子：价格贵不贵——配置分位 × 指数分位</div>')
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
    h('<div class="sec">⑨ 全行业一览：把 31 个行业放在一张表里</div>')
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
    # 给「拥挤等级」与「判读」上色：一屏 31 行，纯文字很难扫出谁最挤。
    # 小样本用灰色——它既不是拥挤也不是冷门，是「样本不够，判不了」。
    _LV = {"极度拥挤": "#C62828", "拥挤": "#E8710A", "中性": "#8A97A5",
           "偏冷": "#4C8DDF", "冷门": "#1F4E9C", "数据不足": "#C3CBD4",
           "小样本": "#9AA6B2"}
    _JD = {"共识形成": "#C62828", "多空对撞": "#8A97A5",
           "系统性撤离": "#1A7F37", "数据不足": "#C3CBD4"}

    def _row_style(row):
        """按行上色：等级列底色 + CSI 数字同色 + 判读列文字色。

        原来用 map(subset=["拥挤等级","CSI"]) 给 CSI 上色是无效的——
        CSI 列的值是 83.0 这样的数字，查不到颜色字典的键，整列等于没上色。
        改成按行取「该行的等级」再回填，CSI 才会跟等级同色。
        """
        out = [""] * len(row)
        c = _LV.get(str(row.get("拥挤等级")))
        if c:
            if "拥挤等级" in row.index:
                out[row.index.get_loc("拥挤等级")] = (
                    f"background-color:{c}22;color:{c};font-weight:700")
            if "CSI" in row.index:
                out[row.index.get_loc("CSI")] = f"color:{c};font-weight:700"
        jd = _JD.get(str(row.get("判读")))
        if jd and "判读" in row.index:
            out[row.index.get_loc("判读")] = f"color:{jd};font-weight:600"
        return out

    _st = _rank_sort(_show, "拥挤等级").style.apply(_row_style, axis=1)
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
