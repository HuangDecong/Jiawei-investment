# -*- coding: utf-8 -*-
"""
rigor.py  ——  方法严谨性工具箱（第 1 轮自主迭代新增）
======================================================================
本模块集中处理"统计方法是否严谨"这一类问题，逐项对应明确的假设与检验标准：

  M1 多重检验校正        bh_fdr / bonferroni
     问题：同时检验 10+ 个因子，单看 p<0.05 会大量假阳性。
     假设：各因子的 IC 序列在零假设下 p 值独立或正相关（BH 对正相关稳健）。
     标准：报告 BH-FDR 的 q 值，以 q<0.10 为发现阈值。

  M2 Newey-West 滞后阶数敏感性   nw_lag_sensitivity
     问题：NW 滞后阶数选 2 缺乏依据。
     标准：滞后 0/1/2/4/8 依次估计，若 t 值符号或显著性随阶数跳变，
           说明结论不稳健，必须标注。

  M3 交易成本与换手率            long_short_with_cost
     问题：分层多空是理论值，未扣成本、未考虑换手。
     口径：季度调仓；换手率 = 相邻两期成分变动的比例（双边计）。
           成本 = 换手率 × 单边成本 × 2（多空两条腿）；同时给 5/10/20bp 三档。

  M4 压力测试                    stress_test
     问题：只有"描述当前集中度"，没有"如果某行业回撤 X% 会怎样"。
     口径：以主动权益基金行业配置比例为权重的组合，施加情景冲击。

  M5 子样本稳定性                subperiod_stability
     问题：IC 只看全样本均值，掩盖时变。
     口径：前半样本 / 后半样本 / 逐年，报告 IC 与方向一致率。

  M6 因子拥挤度监控              factor_crowding_watch
     问题：因子自身也会拥挤。
     口径：复用 FCI 三个分量，对候选因子逐个计算。

所有函数只做计算，不写文件、不打印结论——由 update_pipeline 统一汇报。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from factor_engine import cz, rank_ic_panel, safe_spearman
from backtest import nw_tstat, perf_series


# ============================================================================
# M1 多重检验校正
# ============================================================================
def bh_fdr(pvals, alpha: float = 0.10) -> dict:
    """Benjamini-Hochberg FDR 校正。

    返回 q 值（每个假设的校正后错误发现率）与 reject 掩码。
    零假设：该因子的 IC 均值为 0。
    """
    p = pd.Series(pvals).astype(float)
    ok = p.notna()
    idx = p.index[ok]
    pv = p[ok].sort_values()
    m = len(pv)
    if m == 0:
        return dict(q=pd.Series(dtype=float), reject=pd.Series(dtype=bool), m=0)
    ranks = np.arange(1, m + 1)
    q = (pv.to_numpy() * m / ranks)
    # 单调化：从大到小取累积最小
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    qs = pd.Series(q, index=pv.index)
    return dict(q=qs.reindex(idx), reject=(qs <= alpha).reindex(idx),
                m=m, alpha=alpha)


def bonferroni(pvals, alpha: float = 0.05) -> dict:
    p = pd.Series(pvals).astype(float)
    m = int(p.notna().sum())
    thr = alpha / m if m else np.nan
    return dict(threshold=thr, reject=(p <= thr), m=m, alpha=alpha)


# ============================================================================
# M2 Newey-West 滞后阶数敏感性
# ============================================================================
def nw_lag_sensitivity(ic: pd.Series, lags=(0, 1, 2, 4, 8)) -> pd.DataFrame:
    """对同一 IC 序列用不同 NW 滞后阶数估计 t 值，检查结论是否随阶数跳变。"""
    x = pd.Series(ic).dropna()
    rows = []
    for L in lags:
        t, p = nw_tstat(x, lags=L)
        rows.append(dict(NW滞后阶数=L, t值=t, p值=p,
                         显著性=("**" if p < 0.01 else "*" if p < 0.05
                                 else "." if p < 0.10 else "不显著")
                         if p == p else "样本不足"))
    df = pd.DataFrame(rows)
    if not df.empty:
        signs = np.sign(df["t值"].dropna())
        df["全部同号"] = bool(len(set(signs)) <= 1)
    return df


# ============================================================================
# M3 交易成本与换手率
# ============================================================================
def long_short_with_cost(factor: pd.DataFrame, fwd: pd.DataFrame,
                         n_layer: int = C.LAYER_N, min_obs: int = 15,
                         costs_bp=(5, 10, 20)) -> dict:
    """分层多空组合（高拥挤 − 低拥挤），计算换手率并扣除交易成本。

    换手率口径：季度调仓，双边计。
      turnover_t = (|多头腿成分变动| + |空头腿成分变动|) / (2 × 每腿行业数)
    毛收益 = 高拥挤腿 − 低拥挤腿（等权）。
    净收益 = 毛收益 − turnover_t × 单边成本 × 2（买卖各一次）
    """
    members = {}
    gross = {}
    for q in factor.index:
        a, bq = factor.loc[q], fwd.loc[q]
        m = a.notna() & bq.notna()
        if int(m.sum()) < min_obs:
            continue
        aa, bb = a[m], bq[m]
        layers = pd.qcut(aa.rank(method="first"), n_layer, labels=False) + 1
        hi = set(aa.index[layers == n_layer])
        lo = set(aa.index[layers == 1])
        members[q] = (hi, lo)
        gross[q] = float(bb[list(hi)].mean() - bb[list(lo)].mean())

    qs = list(members)
    if len(qs) < 3:
        return dict(table=pd.DataFrame(), perf={}, turnover=pd.Series(dtype=float))
    rows = []
    prev = None
    for q in qs:
        hi, lo = members[q]
        if prev is None:
            tv = np.nan
        else:
            phi, plo = prev
            tv = (len(hi ^ phi) + len(lo ^ plo)) / (len(hi) + len(lo))
        rows.append(dict(quarter=q, gross=gross[q], turnover=tv,
                         n_hi=len(hi), n_lo=len(lo)))
        prev = (hi, lo)
    t = pd.DataFrame(rows).set_index("quarter")
    t["turnover"] = t["turnover"].fillna(t["turnover"].dropna().mean()
                                         if t["turnover"].notna().any() else 1.0)

    perf = {}
    perf["毛收益"] = perf_series(t["gross"])
    for bp in costs_bp:
        net = t["gross"] - t["turnover"] * (bp / 10000.0) * 100 * 2
        perf[f"净收益_{bp}bp"] = perf_series(net)
    t["平均换手率"] = t["turnover"].mean()
    return dict(table=t, perf=perf,
                turnover=t["turnover"],
                avg_turnover=float(t["turnover"].mean()))


# ============================================================================
# M4 压力测试
# ============================================================================
def stress_test(b, ind: dict, mkt: dict, scenarios=None) -> pd.DataFrame:
    """以主动权益基金的行业配置比例为权重，施加情景冲击。

    组合口径：假设一个"复制主动权益基金行业配置"的组合（等权于配置比例）。
    冲击：对指定行业施加给定涨跌幅，其余行业不变。
    输出：组合冲击幅度、以及"若只持第一大重仓行业"的对照。
    """
    if scenarios is None:
        scenarios = [
            ("电子回撤30%", {"电子": -30.0}),
            ("电子回撤50%", {"电子": -50.0}),
            ("TMT整体回撤30%（电子+通信+计算机+传媒）",
             {"电子": -30.0, "通信": -30.0, "计算机": -30.0, "传媒": -30.0}),
            ("前三大重仓行业回撤25%（电子+通信+机械设备）",
             {"电子": -25.0, "通信": -25.0, "机械设备": -25.0}),
            ("市场普跌20%", {i: -20.0 for i in C.INDUSTRIES}),
            ("拥挤行业回撤30% / 低配行业持平（风格反转）",
             {i: -30.0 for i in ind["拥挤等级"].loc[b.latest]
              [ind["拥挤等级"].loc[b.latest] == "极度拥挤"].index}),
        ]
    w = b.alloc.loc[b.latest] / 100.0
    top1 = b.alloc.loc[b.latest].idxmax()
    rows = []
    for name, shock in scenarios:
        pnl = sum(w.get(k, 0.0) * (v / 100.0) for k, v in shock.items())
        conc = sum(w.get(k, 0.0) for k in shock)
        rows.append(dict(情景=name, 受影响行业权重=f"{conc * 100:.2f}%",
                         组合冲击=f"{pnl * 100:+.2f}%",
                         仅持第一大重仓行业冲击=
                         f"{(shock.get(top1, 0.0) / 100.0) * 100:+.2f}%",
                         涉及行业="、".join(shock.keys())))
    return pd.DataFrame(rows)


# ============================================================================
# M5 子样本稳定性
# ============================================================================
def subperiod_stability(ic: pd.Series, n_parts: int = 3) -> pd.DataFrame:
    """把 IC 序列等分为 n_parts 段，检查符号与量级是否稳定。"""
    x = pd.Series(ic).dropna()
    if len(x) < n_parts * 3:
        return pd.DataFrame()
    parts = np.array_split(np.arange(len(x)), n_parts)
    rows = []
    for i, p in enumerate(parts, 1):
        seg = x.iloc[p]
        rows.append(dict(区段=f"第{i}/{n_parts}段",
                         区间=f"{seg.index[0]} ~ {seg.index[-1]}",
                         期数=len(seg),
                         IC均值=seg.mean(),
                         IC_IR=seg.mean() / seg.std(ddof=1) if seg.std(ddof=1) else np.nan,
                         胜率=(seg > 0).mean() * 100))
    all_sign = np.sign([r["IC均值"] for r in rows])
    for r in rows:
        r["与全样本同号"] = bool(np.sign(r["IC均值"]) == np.sign(x.mean()))
    return pd.DataFrame(rows)


# ============================================================================
# M6 因子拥挤度监控（逐因子）
# ============================================================================
def factor_crowding_watch(ind: dict, b, factors=None) -> pd.DataFrame:
    """对每个候选因子计算"因子自身拥挤度"的三个分量（最新期）。

    · 空头集中度：因子值最低 1/3 行业，其配置比例的 HHI
    · 因子相关性：该因子与其余候选因子的平均 |Spearman|
    · 极端占比  ：因子值处于 ±1σ 之外的行业占比
    """
    factors = factors or [k for k in ind if k in
                          ("F1_超配比例", "F2_超配Zscore", "F3_超配历史分位",
                           "F4b_超配动量Z", "F9_交易拥挤度", "F12_配置系数",
                           "F13_筹码盈利比例", "F11_综合拥挤度_推荐权重")]
    q = b.latest
    alloc = b.alloc.loc[q] / 100.0
    n_short = max(int(C.NI / 3), 5)
    rows = []
    for f in factors:
        row = ind[f].loc[q].dropna()
        if len(row) < n_short + 3:
            continue
        leg = row.nsmallest(n_short).index
        w = alloc.reindex(leg).dropna()
        hhi = float(((w / w.sum()) ** 2).sum()) if w.sum() > 0 else np.nan
        cs = []
        for o in factors:
            if o == f:
                continue
            a2 = row.copy()
            b2 = ind[o].loc[q].reindex(a2.index).dropna()
            a2 = a2.reindex(b2.index)
            if len(b2) >= 8:
                _c = safe_spearman(a2, b2)
                if _c == _c:
                    cs.append(abs(_c))
        row = row.astype(float)
        z = (row - row.mean()) / row.std(ddof=1) if row.std(ddof=1) else row * np.nan
        rows.append(dict(因子=f, 空头集中度HHI=hhi,
                         平均因子相关性=float(np.mean(cs)) if cs else np.nan,
                         极端占比=f"{float((z.abs() > 1).mean() * 100):.2f}%",
                         有效行业数=len(row)))
    return pd.DataFrame(rows)
