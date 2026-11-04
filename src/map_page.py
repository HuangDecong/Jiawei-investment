# -*- coding: utf-8 -*-
"""③ 行业拥挤度：可搜索、每个行业都有颜色标注、每个数字都能查到算法。

页面结构
--------
    本页结论（先给判断）
    ① 拥挤度怎么变：全样本期折线 + 31 行业全景热力图
    ② 阈值有没有用：分档 → 次季超额（含事件研究）
    ③ 找行业：搜索 + 31 个色块（颜色＝等级）
    ④ 单行业详情（6 个指标 + 历史走势 + 近 8 期台账）
    ⑤ 31 个行业排一排：**选一期**，最拥挤的排在最上面
    ⑥ 为什么用 CSI 不用 HHI
    ⑦ 双分位：相对自己 vs 相对同行
    ⑧ 四象限：配置 × 价格
    ⑨ 全行业表

口径
----
· 配置比例用**前十大归一化**（每期都有，才能做环比/同比/分位）；
· 「全持仓」只在半年报期存在，单列并标期次，不与前十大混算；
· 分位＝当前值在**自身全部期次**中的排名（0＝历史最低，100＝历史最高）；
· 拥挤度 CSI＝(配置分位＋价格分位＋资金分位)/3，等权、不拟合。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import canon as cn
import config as C
import report_charts as rc
import ui


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


def _num(v, nd: int = 0, dash: str = "—") -> str:
    """把可能为 NaN 的数值格式化成字符串：NaN → 破折号，绝不输出 "nan"。"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return dash
    return f"{v:.{nd}f}" if np.isfinite(v) else dash


def _read(h, fact: str, econ: str, invest: str = "") -> None:
    """图下方的三段式分析：事实 → 经济含义 → 投资含义。"""
    ui.analysis(h, fact, econ, invest)


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
        _tip = (f'全市场仅 {_num(r.get("持有基金数"))} 只基金持有，'
                f'前十配置 {_num(r.get("配置_前十", r.get("配置")), 2)}%；'
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
            f'{_num(r.get("指数分位"), 0)}）' for i, r in _top.iterrows())
            + f'。31 个行业里极度拥挤 <b>{_n_ext}</b> 个、冷门 <b>{_n_cold}</b> 个。')
    if len(_bot):
        _bl.append('最冷门：' + "、".join(
            f'{i}（CSI {r["CSI"]:.0f}，配置{r["历史分位"]:.0f}/价格'
            f'{_num(r.get("指数分位"), 0)}）' for i, r in _bot.iterrows())
            + '。冷门不等于机会——没有资金关注时，估值再低也可能继续低。')
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
    h(ui.concl_board(_bl))

    # =====================================================================
    # ① 拥挤度变化：全样本期折线
    # =====================================================================
    H = cn.csi_history()
    _nq = len(H) if not H.empty else len(cn.quarters())
    h(f'<div class="sec">① 拥挤度怎么变：{_nq} 期折线 + 它自己的历史分位</div>')
    _gap = len(cn.quarters()) - _nq if not H.empty else 0
    if _gap > 0:
        h(f'<div class="tiny" style="margin:-2px 0 5px">'
          f'CSI 自 <b>{H.index[0]}</b> 起可算：配置分位要求至少 8 个历史观测，'
          f'更早的 {_gap} 期（{cn.quarters()[0]}–{cn.quarters()[_gap - 1]}）'
          f'只作为分位的历史底稿，不进入曲线——否则只有 4 个观测时，'
          f'「恰好是历史最高」会被写成 100 分，图上会出现一片假的红。</div>')
    if H.empty:
        h('<div class="tiny">历史 CSI 不可用（数据不足）。</div>')
    else:
        # 四个控件分两行：一行放两个下拉，一行放滑块与开关。
        # 挤在一行时滑块只剩 100px，「拥挤阈值」的标题会被压成两行。
        _r1c1, _r1c2 = st.columns([0.56, 0.44])
        with _r1c1:
            _opts = [f"只看全市场中位（31 个行业的 CSI 中位数）"] + list(H.columns)
            _focus = st.selectbox("聚焦一个行业", _opts, index=0, key="csi_focus",
                                  help=f"选一个行业会画出它自己的 {_nq} 期曲线，"
                                       f"并标出它上穿/下穿阈值的时点")
        with _r1c2:
            # 分位口径自己选：全历史最稳，窗口越短越敏感、也越容易被单季波动带偏
            _win_lab = st.selectbox(
                "分位窗口", ["全部历史", "近 20 期（5 年）", "近 12 期（3 年）"],
                index=0, key="csi_win",
                help="分位＝当前值在「这个窗口」里的排名。全历史更稳；"
                     "近 20/12 期只看近期，适合判断「这一轮周期里它走到哪了」，"
                     "但一两次季度波动就可能把分位推到 0 或 100")
        _r2c1, _r2c2 = st.columns([0.66, 0.34])
        with _r2c1:
            _thr = st.slider("拥挤阈值（画在哪条线算过度拥挤）", 60, 95, 80,
                             step=5, key="csi_thr",
                             help="改它会同步改上穿/下穿标记、「历史触及」统计与提示语；"
                                  "五档底色仍按固定档位（80/65/45/25）")
        with _r2c2:
            _band = st.checkbox("显示 31 行业分位带", value=True, key="csi_band",
                                help="灰色带是 31 个行业每一期的 25–75% 分位区间，"
                                     "表示「市场整体的分布宽度」；不画 31 条线是因为会糊成一片")
        _sel_t = None if _focus == _opts[0] else _focus
        _window = {"全部历史": None, "近 20 期（5 年）": 20, "近 12 期（3 年）": 12}[_win_lab]
        # 排名与广度在「当期有值」的全部行业里算——包括选中行业自己，
        # 否则它不在池子里时会出现「第 27/26」这种不可能的名次。
        _lastrow = H.iloc[-1]
        _ok = [i for i in H.columns if np.isfinite(_lastrow.get(i, np.nan))]
        st.plotly_chart(rc.fig_csi_trend(H, _sel_t, _band, rank_universe=_ok,
                                         window=_window, thr=float(_thr)),
                        width="stretch", config={"displayModeBar": False})

        # ---- 现在处于什么分位：时间维度（自身历史）× 截面维度（全行业） ----
        _med = H.median(axis=1).dropna()
        _series = H[_sel_t].dropna() if _sel_t else _med
        _name = _sel_t or "全市场中位"
        _cur = float(_series.iloc[-1])
        _ns = len(_series)
        # 分位与排名必须与图里同一套算法（pct_of），否则同一页会出现两个分位
        _own, _nw, _kth = rc.pct_of(_series, _cur, _window)
        _rank = int((_lastrow[_ok] > _cur).sum()) + 1
        _nvalid = len(_ok)
        _chg4 = _cur - float(_series.iloc[-5]) if _ns >= 5 else np.nan
        _n80 = int((_series >= _thr).sum())
        _win_txt = f"近 {_nw} 期" if _window else f"自身 {_nw} 期历史"
        # 该行业的最后一个有效值可能不在最新一期（三项分位不齐时是 NaN）。
        # 那样「当前 CSI」其实是旧值，必须标出来，否则会被当成最新读数。
        _stale = bool(len(_series)) and _series.index[-1] != H.index[-1]
        _stale_txt = f"｜最新有值 {_series.index[-1]}" if _stale else ""
        _imax = int(np.argmax(_series.values))
        _imin = int(np.argmin(_series.values))
        _lv_now = ""
        if _sel_t and "等级" in csi.columns and _sel_t in csi.index:
            _lv_now = str(csi.loc[_sel_t, "等级"])
        # 横截面的「广度」：当期有多少行业已经进入拥挤区（≥65）。
        # 全市场视角下这比「中位排第几」有意义——中位天然在 16 名附近，不是信息。
        _crowd = _lastrow[_ok][_lastrow[_ok] >= 65]
        _crowd_txt = f"{len(_crowd)}/{_nvalid}"

        st.plotly_chart(rc.fig_csi_position(_series, _name, window=_window,
                                            thr=float(_thr)),
                        width="stretch", config={"displayModeBar": False})
        _kpis = [("当前 CSI", f"{_cur:.0f}",
                  f"{_name}" + (f"｜{_lv_now}" if _lv_now else "") + _stale_txt),
                 (_win_txt + "分位" if _window else "自身历史分位", f"{_own:.0f}",
                  f"{_nw} 期中第 {_kth} 低")]
        if _sel_t:
            _kpis.append(("全行业排名", f"第 {_rank}/{_nvalid}", "同一时点横截面"))
        else:
            _kpis.append(("≥65 分行业", _crowd_txt,
                          f"占 {len(_crowd) / max(_nvalid, 1) * 100:.0f}%，拥挤广度"))
        _kpis += [("近 4 期变化", (f"{_chg4:+.0f}" if np.isfinite(_chg4) else "—"),
                   "与前 4 个季度比"),
                  (f"历史触及 ≥{_thr:.0f}", f"{_n80} 期",
                   f"占 {_n80 / _ns * 100:.0f}%")]
        h(ui.stat_strip(_kpis))

        if _sel_t:
            _d1 = float(_series.iloc[-1] - _series.iloc[-2]) if _ns > 1 else np.nan
            _n_up = sum(1 for i in range(1, _ns)
                        if _series.iloc[i - 1] < _thr <= _series.iloc[i])
            _n_dn = sum(1 for i in range(1, _ns)
                        if _series.iloc[i - 1] >= _thr > _series.iloc[i])
            _txt2 = (f'上半图是<b>{_sel_t}</b> 的拥挤度水位：最新 <b>{_cur:.0f}</b> 分（{_lv_now}），'
                     f'上期 {_series.iloc[-2]:.0f}（{_d1:+.0f}）。两个分位同时给出来：'
                     f'<b>{_win_txt}第 {_own:.0f} 分位</b>（由低到高第 {_kth} 位，'
                     f'共 {_nw} 期）、<b>全行业第 {_rank}/{_nvalid}</b>。'
                     f'历史最高 {_series.iloc[_imax]:.0f}（{_series.index[_imax]}）、'
                     f'最低 {_series.iloc[_imin]:.0f}（{_series.index[_imin]}）；'
                     f'{_ns} 期里上穿 {_thr:.0f} 共 <b>{_n_up}</b> 次、'
                     f'下穿 <b>{_n_dn}</b> 次（红三角＝上穿、绿三角＝下穿）。')
        else:
            _txt2 = (f'灰粗线是<b>全市场中位</b>：最新 <b>{_cur:.0f}</b> 分，'
                     f'处在{_win_txt}的第 <b>{_own:.0f}</b> 分位；'
                     f'历史最高 {_series.iloc[_imax]:.0f}（{_series.index[_imax]}）、'
                     f'最低 {_series.iloc[_imin]:.0f}（{_series.index[_imin]}）。'
                     f'下半图是<b>拥挤广度</b>：当期 {_nvalid} 个行业里有 '
                     f'<b>{len(_crowd)} 个</b>已经 ≥65 分'
                     f'（{len(_crowd) / max(_nvalid, 1) * 100:.0f}%）。'
                     f'中位只说明「整体水位」，广度才说明「是几个行业在挤，还是全都挤」；'
                     f'单看中位看不出谁在变挤，要在上面选一个行业。')

        # 投资含义随所选行业变化：同一个分数，上行与下行对应完全相反的动作
        _inv3 = (f' 你选的 <b>{_sel_t}</b> 最新 {_cur:.0f} 分、'
                 f'上期 {_series.iloc[-2]:.0f} 分（{_d1:+.0f}）：'
                 + ("<b>还在变挤</b>——仓位上限压到平时的七成，等它转向再恢复。"
                    if _d1 > 3 else
                    "<b>正在变松</b>——不要因为分数还高就恐慌卖出，资金已经在退。"
                    if _d1 < -3 else
                    "<b>基本走平</b>——按水位办事即可，不需要因为这张图调仓。")
                 ) if _sel_t and _ns > 1 else ''
        _read(h, _txt2,
              f'<b>上半图回答「方向」，下半图回答「位置」。</b>'
              f'「自身历史分位」是时间维度：这个行业现在的拥挤度，在它自己过去 {_nw} 个季度里排第几——'
              f'回答「对它自己来说，现在是不是异常」。'
              + (f'当前选的是<b>近 {_window} 期窗口</b>（只跟最近这段时间比），'
                 f'窗口越短越敏感，适合看「这一轮周期走到哪了」。'
                 if _window else '当前选的是<b>全部历史</b>，最稳，适合做仓位决策。')
              + f'它用<b>扩张窗口</b>逐期计算'
              f'（第 i 期只跟第 i 期及以前比），不用未来数据，所以曲线是可以「当时就看见」的。'
              f'「全行业排名」是截面维度：同一时点 31 个行业里排第几——回答「在同行里，它是不是最挤的那个」。'
              f'两者会背离：小行业可能「自身 95 分位、全行业第 20 名」（配置只有 1%，挤也挤不到哪去），'
              f'大行业也可能「自身 60 分位、全行业第 3 名」（它一直很重，但没比过去更重）。'
              f'背离时以<b>全行业排名</b>为准——组合的风险来自「大家都挤在一起」，不是「它自己创了新高」。',
              f'<b>看方向做仓位，看水位做上限。</b>分数还在往上走、且已经过 80 的，'
              f'本季不要再加，甚至把已有仓位减半（还有人在往没有空间的地方挤，'
              f'转向时卖出会非常集中）；分数从 90 以上往下掉到 70 附近的，'
              f'反而是「压力已经释放」，不必急着卖。'
              f'<br><b>两个分位怎么用</b>：自身历史分位 ≥80 且全行业排名前 5 → '
              f'<b>先减它</b>（对己对人都到顶）；自身分位高但全行业排名靠后 → '
              f'<b>不用动</b>（是小体量的统计噪音）；自身分位不高但全行业前 3 → '
              f'<b>盯着</b>（它一直是大家的底仓，转向时冲击最大）。' + _inv3
              + f' 需要提醒的是：{_nq} 期样本下「上穿 {_thr:.0f}」的次季胜率不到五成、'
                't 值不显著，所以这条线只能约束仓位，不能当买卖信号。'
              + (' 你当前选的是**近 %d 期窗口**：分位只说明「在这段时间里排第几」，'
                 '窗口越短越敏感，也越容易被一两次季度波动推到 0 或 100——'
                 '做仓位决策时建议同时看「全部历史」那一档。' % _window
                 if _window else ''))

        # ---- 全景热力图：替代「31 条线叠加」 ----
        _small = ([i for i in csi.index if csi.loc[i, "等级"] == "小样本"]
                  if not csi.empty and "等级" in csi.columns else [])
        h('<div style="font-size:14px;font-weight:700;margin:14px 0 4px">'
          '31 个行业全景：颜色＝档位，一屏看出「谁、什么时候」变红</div>')
        st.plotly_chart(rc.fig_csi_heatmap(H, _small, bad=cn.csi_invalid()),
                        width="stretch",
                        config={"displayModeBar": False})
        h('<div class="tiny" style="margin:-4px 0 6px">'
          '<b>空白格＝数据不足，不评级</b>：① 序列最前 7 期历史不足 8 个观测；'
          '② 美容护理 / 石油石化 / 环保是 2021 年新设的一级行业，指数 2022 年才有，'
          '之前只有「配置 + 资金」两项，凑不出可比的三项平均。'
          '灰色 † 是小样本（持有 &lt;150 只基金或配置 &lt;0.5%）。</div>')

        # ---- 最新一期读数：只用文字，不重复画图 ----
        # 原来这里还有一张「同期横向对比」条形图，和 ⑤ 的排名图是同一份数据画两遍，
        # 而且横向条形图的 y 轴是从下往上排，最拥挤的反而落在最底部，读起来别扭。
        # 现在季度选择合并到 ⑤，一张图既能看当期、也能回看历史。
        _sq = H.loc[H.index[-1]].dropna().sort_values(ascending=False)
        _sq_ok = _sq[[i for i in _sq.index if i not in _small]]
        _chg = H.iloc[-1] - H.iloc[-2] if len(H) > 1 else pd.Series(dtype=float)
        _upq = _chg.dropna().nlargest(2)
        _dnq = _chg.dropna().nsmallest(2)
        h(f'<div class="cg-note" style="font-size:12.5px">'
          f'最新一期（{H.index[-1]}）最挤的是 '
          f'<b>{_sq_ok.index[0]} {_sq_ok.iloc[0]:.0f}</b>'
          + (f'、次之 {_sq_ok.index[1]} {_sq_ok.iloc[1]:.0f}' if len(_sq_ok) > 1 else '')
          + f'；最冷的是 <b>{_sq_ok.index[-1]} {_sq_ok.iloc[-1]:.0f}</b>。'
          + (f' 相对上一期，变挤最多 {_upq.index[0]}（{_upq.iloc[0]:+.0f}）、'
             f'变松最多 {_dnq.index[0]}（{_dnq.iloc[0]:+.0f}）。' if len(_upq) else '')
          + f' 想看某一期的完整排名，用下面 ⑤ 的季度选择器；'
            f'灰色 † 为小样本，不参与排名。</div>')

        # 最近 4 期里变化最大的几行
        _d4 = (H.iloc[-1] - H.iloc[-5]).dropna() if len(H) >= 5 else pd.Series(dtype=float)
        _up4 = _d4.nlargest(3)
        _dn4 = _d4.nsmallest(3)
        _red = H.iloc[-1][H.iloc[-1] >= 80].sort_values(ascending=False)
        _long = ((H >= 80).sum()).nlargest(3)
        _read(h,
              f'最新一期（{H.index[-1]}）31 个行业里只有 <b>{len(_red)} 个</b> 的 CSI ≥80'
              f'（{"、".join(f"{i} {v:.0f}" for i, v in _red.items()) or "无"}），'
              f'其余都在 80 以下。最近 4 期变红最快的是 '
              + ("、".join(f'{i}（{r:+.0f} 分）' for i, r in _up4.items()) or "—")
              + f'，退得最快的是 '
              + ("、".join(f'{i}（{r:+.0f} 分）' for i, r in _dn4.items()) or "—")
              + f'。历史上待在 ≥80 最久的是 '
              + ("、".join(f'{i}（{int(v)} 期）' for i, v in _long.items()) or "—") + '。',
              f'拥挤从来不是均匀发生的。行业景气的传导有先后：上游先涨（价格信号最快），'
              f'中游跟上（订单兑现），下游最后（需求确认）。反映到持仓上，就是'
              f'<b>红色区域沿产业链依次亮起</b>——所以看热力图要看「红色的移动方向」，'
              f'而不是「现在哪里红」。红色停留在同一行越久，说明这个行业的共识越牢固，'
              f'也越接近「所有人都已上车」的状态；反过来，一行从红退回橙、再退回白，'
              f'通常是景气见顶后资金在分批撤。',
              f'本期只有 {len(_red)} 个行业 ≥80，说明这是<b>结构性拥挤、不是系统性风险</b>：'
              f'市场没有出现「全行业抱团」，因此不需要整体降仓，只需要在这几个行业上'
              f'压缩仓位上限。如果某期出现 15 个以上同时红，那才是「全市场都在抱团」的顶部特征，'
              f'届时应把权益仓位整体下调。'
              f'红色在移动时，正确的动作是<b>跟着移动方向找下一个还没变红的环节</b>，'
              f'而不是追已经红透的那个。')

    # =====================================================================
    # ② 阈值实证
    # =====================================================================
    h(f'<div class="sec">② 阈值有没有用：{_nq} 期 × 31 行业，按 CSI 分档看次季能否跑赢同行</div>')
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
            # 显著性说法必须跟着数据走：一档过线时再说「全部不显著」就是自相矛盾
            if len(_sig):
                _sig_txt = ("有 <b>" + str(len(_sig)) + " 档</b>的 |t| ≥1.65（"
                            + "、".join(f'{r["CSI 档位"]} t={r["t 值"]:+.2f}'
                                        for _, r in _sig.iterrows())
                            + f'），但五档不单调、样本只有 {int(_sig["样本数"].min())}–'
                              f'{int(_sig["样本数"].max())} 个，按 5 次检验的多重比较看，'
                              f'更可能是一次偶然。')
            else:
                _sig_txt = "五档的 t 值全部落在 ±1.65 以内，<b>没有一档显著</b>。"
            _b1 = _bt.iloc[1]
            _b1_sig = abs(float(_b1["t 值"])) >= 1.65
            _read(h,
                  f'把 {_ST["q_first"]}–{_ST["q_last"]} 的每一期、每个行业都当成一个观测，'
                  f'共 <b>{_ST["n_obs"]}</b> 个（{_nq} 期 × 31 行业，剔除无次季收益的期次）。'
                  f'「次季超额」＝该行业下一季收益 − 下一季 31 个行业等权收益，'
                  f'<b>已经扣掉大盘涨跌</b>；胜率＝次季跑赢同行的比例。'
                  f'<br>五档的均值从 {_bt["次季超额均值%"].min():+.2f}% 到 '
                  f'{_bt["次季超额均值%"].max():+.2f}%；' + _sig_txt,
                  f'<b>结论：CSI 的绝对水位没有稳定的预测力。</b>'
                  f'看起来「{_bt.iloc[0]["CSI 档位"]} {_bt.iloc[0]["次季超额均值%"]:+.2f}%、'
                  f'{_bt.iloc[1]["CSI 档位"]} {_bt.iloc[1]["次季超额均值%"]:+.2f}%」有差别，'
                  f'但档位之间不单调（中间三档忽高忽低），'
                  f'换成下一段样本就会翻转。所以本页把阈值用于<b>仓位约束</b>'
                  f'（≥80 的行业，单一行业仓位上限打七折），而不是买卖信号。'
                  f'<br><b>胜率为什么大多不到 50%</b>：A 股行业季度超额收益是右偏分布——'
                  f'少数行业暴涨把均值拉高，多数行业低于均值。所以 '
                  f'{_bt["胜率%"].min():.0f}% 的胜率不等于「大概率下跌」，'
                  f'只说明「中位数略负」。看胜负要同时看中位数与幅度。',
                  f'<b>所以这张表只能这么用</b>：① ≥80 的行业把「单行业仓位上限」'
                  f'打到平时的七折（{_bt.iloc[-1]["CSI 档位"]} 档的均值 '
                  f'{_bt.iloc[-1]["次季超额均值%"]:+.2f}% 说明高位并不必然下跌，'
                  f'砍的是风险预算、不是收益预期）；② 25–45 档不要因为「便宜」就加——'
                  f'它的均值 {_b1["次季超额均值%"]:+.2f}%、t 值 {_b1["t 值"]:+.2f}，'
                  + ("这一档的负超额在统计上已经过线，说明「便宜」本身不是买入理由，"
                     "但只有一档、且不单调，不能反手做空。"
                     if _b1_sig else "没有任何统计优势。")
                  + '③ 任何一档都不构成「买/卖」信号——真正可交易的是'
                    '「上穿/下穿 80 的方向」，而它也只有四成胜率。'
                    '把这张表当成「别把仓位押在拥挤行业上」的量化依据，'
                    '不要当成选行业模型。')
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
                 help="行业指数季末点位在 2015 年以来每个交易日中的排名（0–100），回答「贵不贵」")
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
    # 最早一行没有上一期，diff 是 NaN。直接交给 Streamlit 会显示成「None」，
    # 看起来像数据缺失，其实是「无上期可比」——改成空字符串。
    _hist["环比pp"] = _hist["环比pp"].map(
        lambda v: "" if pd.isna(v) else f"{v:+.2f}")
    _hist["配置%"] = _hist["配置%"].map(lambda v: f"{v:.2f}")
    st.dataframe(_hist, width="stretch", hide_index=True, height=300)

    # =====================================================================
    # ③ 31 个行业一根条
    # =====================================================================
    h('<div class="sec">⑤ 31 个行业排一排：选一期，最拥挤的排在最上面</div>')
    h('<div class="tiny" style="margin:-2px 0 6px">长度＝CSI，颜色＝拥挤等级；'
      '小样本（†）压在最底部、不参与排名。鼠标悬停任一条可看 CSI 由哪三个分位加出来。</div>')
    _qbar = None
    if not H.empty:
        _qbar = st.selectbox("看哪一期", list(H.index)[::-1],
                             index=0, key="csi_bar_q",
                             help="默认最新一期；选历史期可回看当时谁最挤、谁最冷——"
                                  "同一行业在不同期的分数可以直接对比")
    _bar = cn.csi_parts(_qbar) if _qbar else pd.DataFrame()
    if _bar.empty:
        _bar = comp.dropna(subset=["CSI"]).copy()
    _qtitle = _qbar or "最新一期"
    st.plotly_chart(rc.fig_csi_bar(_bar, f"{_qtitle}　CSI 排名（最拥挤在最上面）"),
                    width="stretch", config={"displayModeBar": False})
    _c = _bar.dropna(subset=["CSI"])
    _c_ok = _c[_c["等级"].astype(str) != "小样本"] if "等级" in _c else _c
    _hi = _c_ok[_c_ok["CSI"] >= 80]
    _lo = _c_ok[_c_ok["CSI"] < 25]
    _sm_n = len(_c) - len(_c_ok)
    _is_latest = (_qbar is None) or (_qbar == (H.index[-1] if not H.empty else ""))
    _read(h,
          f'{_qtitle}：≥80 极度拥挤的有 <b>{len(_hi)} 个</b>：'
          + ("、".join(f'{i}（{r["CSI"]:.0f}）' for i, r in _hi.iterrows()) or "无")
          + f'；&lt;25 冷门的有 <b>{len(_lo)} 个</b>：'
          + ("、".join(f'{i}（{r["CSI"]:.0f}）' for i, r in _lo.iterrows()) or "无")
          + f'；其余 {len(_c_ok) - len(_hi) - len(_lo)} 个落在中间段。'
          + (f' 另有 {_sm_n} 个小样本行业（†）不评级。' if _sm_n else ''),
          f'CSI 把「配置、价格、资金」三个维度压成一个 0–100 的分数，'
          f'分数高说明三件事同时成立：机构配得多、价格在历史高位、资金还在进。'
          f'这三个条件凑齐时，行业的<b>边际买家最少</b>——想买的人已经买完了，'
          f'剩下的只有卖出选项。这就是「拥挤」的经济含义，与「贵」不是一回事：'
          f'贵只说明价格高，拥挤说明<b>持仓结构脆弱</b>。'
          + ('' if _is_latest else
             f'<br>你现在看的是<b>历史期 {_qbar}</b>：它的分位只用到 {_qbar} 及以前的数据，'
             f'所以「当时的分位」与「今天回看的分位」是同一个数，没有用到未来信息。'),
          f'≥80 的行业<b>压缩仓位上限</b>（单行业不超过权益的 10%），但不等于清仓——'
          f'因子检验显示高 CSI 对未来收益没有显著负向预测力，所以「卖」没有依据，'
          f'「不加」有依据。&lt;25 的行业不要因为便宜就买，要等「广度≥60% 且净额转正」'
          f'两个信号同时出现，因为冷门可能是「没人要」而不是「没人发现」。'
          f'中间段不必调整，按个股逻辑处理。'
          + ('' if _is_latest else
             f'<br><b>历史期怎么用</b>：只用来看「这一行的分数是怎么走到今天的」——'
             f'比如某个行业在 {_qbar} 就已经 85 分，说明它挤了很久，'
             f'今天的 80 分是「退潮中的高位」，而不是「刚变挤」。'
             f'不要拿历史期的分数当今天的买卖依据。'))

    # =====================================================================
    # ④ 双分位
    # =====================================================================
    # ---- 拥挤度变化：谁在变挤、谁在变松 ----
    h('<div style="font-size:14px;font-weight:700;margin:14px 0 4px">'
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
        _read(h,
              f'变挤最多（{cc.attrs.get("q_prev", "")}→{cc.attrs.get("q", "")}）：'
              + "、".join(f'{i}（{r["上期CSI"]:.0f}→{r["CSI"]:.0f}，{r["ΔCSI"]:+.0f}）'
                          for i, r in _up.iterrows())
              + f'；变松最多：'
              + "、".join(f'{i}（{r["上期CSI"]:.0f}→{r["CSI"]:.0f}，{r["ΔCSI"]:+.0f}）'
                          for i, r in _dn.iterrows())
              + f'。其中「水位已高且还在涨」的有 <b>{len(_hi_up)} 个</b>：'
              + ("、".join(_hi_up.index) if len(_hi_up) else "无")
              + (f'。变化榜已剔除 {_n_sm} 个小样本行业（灰色），它们的波动来自样本太小。'
                 if _n_sm else '。'),
              f'水位（CSI）和流向（ΔCSI）是两件事。一个行业可以「很挤但正在变松」——'
              f'那是资金已经在撤、只是仓位还没降下来，危险程度远低于「很挤且还在变挤」。'
              f'后者意味着还有人在往已经没有空间的地方挤，一旦转向，卖出会非常集中。'
              f'这就是为什么变化比水平更适合做<b>择时信号</b>，而水平只适合做<b>仓位约束</b>。',
              f'真正要动手减的是「水位 ≥80 且 ΔCSI>0」的那 {len(_hi_up)} 个'
              f'（{"、".join(_hi_up.index) if len(_hi_up) else "本期没有"}）。'
              f'已经变松的高水位行业（电子、通信）反而不用急着卖——资金正在自己退潮，'
              f'你卖的时点未必比它好。变松最快且已跌到低位的行业可以进观察名单，'
              f'但要等资金转向确认，不能仅凭「跌得多」就买。')

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
      f'处在 {len(_hhi_hist)} 期的第 <b>' + f'{_hhi_pct:.0f}' + '</b> 分位（越高质量越集中）。'
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
      f'<b>为什么等权</b>：三维已统一为分位，量纲一致；31 个行业 × {_nq} 期不足以'
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
    _read(h,
          f'两边都 ≥80（自身历史高位 + 当期绝对重仓）的有 <b>{len(_both_high)} 个</b>：'
          + ("、".join(_both_high.index) or "无")
          + f'。自身分位高、但横截面不到 50 的有 {len(_self_only)} 个；'
          f'自身分位低、横截面 ≥70 的有 {len(_mt_high)} 个（'
          + ("、".join(_mt_high.index) or "无") + '）。',
          f'「重不重」有两个完全不同的参照系。相对自己：这个行业在它过去 {_nq} 期里算不算高位，'
          f'回答「机构是不是刚开始重视它」；相对同行：它在当期 31 个行业里排第几，'
          f'回答「它是不是组合里的大块头」。小行业可以在自己历史上创纪录，'
          f'绝对配置却只有 1%，对净值几乎没有影响；大行业可以一直是第一大重仓，'
          f'自身分位反而不高。<b>两个参照系缺一个，判断就会偏</b>。',
          f'两边都高的才是<b>真拥挤</b>——既是历史极值、又是绝对重仓，减仓优先级最高。'
          f'只有自身分位高的，多半是小行业的统计假象，不必因为它「创了新高」就卖。'
          f'自身分位低、横截面高的，是「被冷落的大块头」，最容易出现预期差，'
          f'值得放进观察名单——但要有资金转向的信号才动手。')

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
        _read(h,
              f'右上·真拥挤 <b>{len(_crowd)} 个</b>：'
              + ("、".join(f"{i}（配置 {r['历史分位']:.0f} / 指数 {r['指数分位']:.0f}）"
                          for i, r in _crowd.nlargest(4, "指数分位").iterrows()) or "无")
              + f'；右下·抱团没赚 {len(_hold)} 个；左上·机构踏空 {len(_miss)} 个：'
              + ("、".join(_miss.head(3).index) or "无")
              + f'；左下·冷门 {len(_cold)} 个。',
              f'配置分位和价格分位回答两个独立的问题：<b>机构押得多重</b>、<b>价格贵不贵</b>。'
              f'两者交叉出四种完全不同的处境。右上角是「重仓 + 高位」——机构已经赚到了，'
              f'价格也反映了乐观预期，此时买入是替别人接棒；右下角是「重仓 + 低位」——'
              f'机构押了但市场没认，可能是左侧布局，也可能是判断错了；'
              f'左上角是「轻仓 + 高位」——价格先涨、仓位没跟，机构踏空；'
              f'左下角是「轻仓 + 低位」，没人关注也没人赚钱。',
              f'只看配置分位，会把「抱团没涨」误判成拥挤，过早卖出还可能踏空；'
              f'只看价格分位，会把「一直很重」当成「现在才贵」。两个一起看，'
              f'才知道该减谁：右上角那 {len(_crowd)} 个才是真要压仓位的。'
              f'右下的 {len(_hold)} 个不要急着卖——它们是「闷在里面」的，'
              f'一旦行业景气反转，弹性和右上角一样大，只是时点未知。')

    # =====================================================================
    # ⑥ 全行业表
    # =====================================================================
    h('<div class="sec">⑨ 全行业一览：把 31 个行业放在一张表里</div>')
    # 15 列在笔记本上必然要横向拖动，读者很难「一眼扫完」。
    # 默认只给 11 列核心字段，需要审计再切到全部列。
    _wide = st.radio("表格列数", ["核心 11 列（推荐）", "全部 15 列（含全持仓与半年主动）"],
                     horizontal=True, key="map_tbl_cols")
    _cols = ["行业", "配置_前十", "Δ上季pp", "Δ上年pp", "平均主动pp", "加仓占比",
             "历史分位", "指数分位", "CSI", "等级", "判读"]
    if _wide.startswith("全部"):
        _cols = _cols + ["配置_全持仓", "主动pp", "横截面分位", "近3月涨幅%"]
    _show = comp.reset_index()[_cols].copy()
    _show = _show.rename(columns={
        "配置_前十": f"前十配置%({q})", "Δ上季pp": f"环比pp({qp}→{q})",
        "Δ上年pp": f"同比pp({qy}→{q})", "平均主动pp": f"当期主动pp({qp}→{q})",
        "加仓占比": f"加仓占比%({q})", "配置_全持仓": f"全持仓%({qf})",
        "主动pp": f"半年主动pp({qf0}→{qf})", "历史分位": f"配置分位({_nq}期)",
        "横截面分位": "横截面分位", "指数分位": "价格分位(2015起)",
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
            f"配置分位({_nq}期)": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100,
                help=f"该行业配置在它自己 {_nq} 期历史中的排名"),
            "价格分位(2015起)": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100,
                help="行业指数点位在 2015 年以来每个交易日中的排名"),
            "CSI": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100,
                help="(配置分位+价格分位+资金分位)/3", width="small"),
            "拥挤等级": st.column_config.TextColumn(width="small"),
            "判读": st.column_config.TextColumn(width="small"),
            "横截面分位": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100,
                help="当期 31 个行业之间的排名"),
            "指数近3月%": st.column_config.NumberColumn(format="%+.1f"),
        })
    h(f'<div class="tiny">三组口径、三种期次，<b>不要相加</b>：'
      f'「前十配置/环比/同比/当期主动/广度」是<b>前十大（{qp}→{q}）</b>；'
      f'「全持仓/半年主动」是<b>全持仓（{qf0}→{qf}）</b>；'
      f'「配置分位/CSI」用 {_nq} 期前十大算，「价格分位」用 2015 年以来的日频指数算。'
      f'<br>样本 {len(cn.universe(q)):,} 只主动权益基金，已剔除主题基金与覆盖率异常记录。'
      f'<br><b>这条序列含价格漂移</b>：行业涨了，即使没人买，占比也会上升——'
      f'所以「分位高」只能说明<b>它在组合里很重</b>，不能直接说成「机构在买」。'
      f'要剥离价格，用「当期主动pp」与半年的 GK 分解。</div>')
