# -*- coding: utf-8 -*-
"""
factor_engine.py  ——  因子计算引擎（F1 ~ F11）
======================================================================
无未来函数（No-Lookahead）强制口径
----------------------------------
  · 因子观测期为第 t 季度。
  · 凡涉及"历史分布"的统计量（均值/标准差/分位数/P90 阈值/百分位）
    一律使用 **严格早于 t 的扩展窗口**：`s.shift(1).expanding(min_periods=8)`。
    因此 t = 0..7 期没有 Z / 分位 / P90 取值（返回 NaN），这是刻意的，
    不是缺陷——避免用未来信息定义"极端"。
  · 收益率对齐：第 t 期因子 → 第 t+1 期行业相对沪深300 超额收益。
    季报法定披露约 15 个工作日，t+1 季度首月即可获得 t 期持仓，故不构成前视。
    稳健性另测 t → t+2。

因子清单
--------
  F1  行业超配比例          超配(i,t) = 配置比例(%) − 沪深300权重(%)
  F2  超配 Z-score          扩窗标准化，衡量相对自身历史的极端程度
  F3  超配历史分位          扩窗百分位，不依赖分布假设
  F4  超配动量              超配(i,t) − 超配(i,t−1)
  F4b 超配动量 Z            动量的扩窗标准化，捕捉"加速"
  F5  行业 HHI              Σw²，w = 配置比例/100；附标准化 (HHI−1/31)/(1−1/31)
  F6  CR3 / CR5 / CR10      前 n 大行业配置比例之和
  F7  个股集中度（筹码）     Top20 平均持股基金数 / Top20 市值占比 / ≥100 / ≥50 / ≥20
  F8  仓位代理（集中度代理） 前十占净值比 均值 / 中位数 / 重仓股数均值
  F9  交易拥挤度            成交额占比 + 扩窗分位 + 环比加速度（不用 ETF 数据）
  F10 共振得分              S1 持仓拥挤 + S2 交易拥挤 + S3 筹码拥挤
  F11 综合拥挤度            权重合成；行业层面 + 市场层面两条
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config as C

INDUSTRIES = C.INDUSTRIES
NI = C.NI
MIN_HIST = C.MIN_HIST


# ============================================================================
# 0. 无前视统计工具
# ============================================================================
def expanding_z(s: pd.Series, min_periods: int = MIN_HIST) -> pd.Series:
    """扩窗 Z：用 **shift(1)** 之后的历史均值/标准差标准化当前值。"""
    m = s.shift(1).expanding(min_periods=min_periods).mean()
    sd = s.shift(1).expanding(min_periods=min_periods).std()
    return (s - m) / sd.replace(0, np.nan)


def expanding_pct(s: pd.Series, min_periods: int = MIN_HIST) -> pd.Series:
    """扩窗百分位(0~100)：当前值在**此前**历史中的分位。"""
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for t in range(len(v)):
        if t < min_periods or np.isnan(v[t]):
            continue
        hist = v[:t]
        hist = hist[~np.isnan(hist)]
        if len(hist) == 0:
            continue
        out[t] = (hist < v[t]).sum() / len(hist) * 100.0
    return pd.Series(out, index=s.index)


def expanding_quantile(s: pd.Series, q: float, min_periods: int = MIN_HIST) -> pd.Series:
    """扩窗分位阈值（如 P90）：只用历史数据。"""
    return s.shift(1).expanding(min_periods=min_periods).quantile(q)


def cz(df: pd.DataFrame) -> pd.DataFrame:
    """横截面标准化（逐期、跨行业）。"""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1, ddof=1), axis=0)


def forward_excess(market: dict, lag: int = 1) -> pd.DataFrame:
    """第 t 期因子的前瞻超额收益矩阵：lag=1 → 用第 t+1 期超额收益。"""
    return market["excess"].shift(-lag)


def forward_cum_excess(market: dict, horizon: int) -> pd.DataFrame:
    """第 t 期起未来 horizon 个季度的累计相对超额收益(%)。"""
    r = 1 + market["ret"] / 100.0
    h = 1 + market["hs300"] / 100.0
    res = None
    for k in range(1, horizon + 1):
        term = r.shift(-k).div(h.shift(-k), axis=0)       # 按季度行对齐
        res = term if res is None else res * term
    return (res - 1) * 100


def forward_cum_market(market: dict, horizon: int) -> pd.Series:
    """第 t 期起未来 horizon 个季度的累计沪深300 收益(%)。"""
    h = 1 + market["hs300"] / 100.0
    res = None
    for k in range(1, horizon + 1):
        term = h.shift(-k)
        res = term if res is None else res * term
    return (res - 1) * 100


def safe_spearman(x, y) -> float:
    """安全的 Spearman：任一侧为常数或样本过少时返回 NaN（不抛警告）。

    scipy 的 `stats.spearmanr` 在输入为常数序列时会发 ConstantInputWarning
    并把相关系数置为 NaN。本项目多处做"因子两两相关"扫描，出现常数列是
    正常情形（如某指标在某一期对全部行业取同一值），故统一走本封装。
    """
    xa = np.asarray(pd.Series(x).dropna(), dtype=float)
    ya = np.asarray(pd.Series(y).reindex(pd.Series(x).dropna().index).dropna(),
                    dtype=float)
    if len(xa) < 8 or len(ya) < 8:
        return np.nan
    if np.nanstd(xa) == 0 or np.nanstd(ya) == 0:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(stats.spearmanr(xa, ya).statistic)


def rank_ic_panel(factor: pd.DataFrame, fwd: pd.DataFrame, min_obs: int = 8) -> pd.Series:
    """逐期截面 Spearman RankIC。"""
    out = {}
    for q in factor.index:
        a, b = factor.loc[q], fwd.loc[q]
        m = a.notna() & b.notna()
        if m.sum() < min_obs:
            out[q] = np.nan
            continue
        xa, xb = a[m].to_numpy(dtype=float), b[m].to_numpy(dtype=float)
        # 常数列（如某因子在该期对全部行业取同一值）的 Spearman 无定义，
        # 直接返回 NaN，避免 scipy 抛 ConstantInputWarning 污染运行日志。
        if np.nanstd(xa) == 0 or np.nanstd(xb) == 0:
            out[q] = np.nan
            continue
        out[q] = safe_spearman(xa, xb)
    return pd.Series(out, index=factor.index, name="RankIC")


def zscore_ts(s: pd.Series) -> pd.Series:
    """时序扩窗 Z（用于市场层面分量的合成）。"""
    return expanding_z(s, MIN_HIST)


# ============================================================================
# 1. 主构建
# ============================================================================
def build_all(b) -> dict:
    """返回 dict：'ind' = 行业横截面因子，'mkt' = 市场层面时序因子，'raw' = 中间量。"""
    over = b.overweight.reindex(columns=INDUSTRIES)
    alloc = b.alloc.reindex(columns=INDUSTRIES)
    share = b.market["turn_share"].reindex(columns=INDUSTRIES)
    Q = list(over.index)
    ind, mkt = {}, {}

    # =============== F1 行业超配比例 ===============
    ind["F1_超配比例"] = over.copy()

    # =============== F2 超配 Z-score（扩窗） ===============
    ind["F2_超配Zscore"] = pd.DataFrame(
        {i: expanding_z(over[i]) for i in INDUSTRIES}, index=Q)[INDUSTRIES]
    # 对照：全样本 Z（含前视，仅用于量化前视偏差大小）
    ind["_F2_全样本Z_仅对照"] = pd.DataFrame(
        {i: (over[i] - over[i].mean()) / over[i].std(ddof=1) for i in INDUSTRIES},
        index=Q)[INDUSTRIES]

    # =============== F3 超配历史分位（扩窗） ===============
    ind["F3_超配历史分位"] = pd.DataFrame(
        {i: expanding_pct(over[i]) for i in INDUSTRIES}, index=Q)[INDUSTRIES]

    # =============== F1d/F2d/F3d 动态基准变体（第 2 轮迭代新增） ===============
    # 背景：工作簿 Sheet4 的超配比例 = 配置比例 − **静态**沪深300 权重（34 期恒定），
    #       用 2026 年的权重衡量 2018 年，会把"因价格上涨自然上升的权重"
    #       误判为"基金经理主动加仓"。
    # 改进：基准改用 **漂移法重建的动态季度权重**（tools/build_dynamic_weights.py），
    #       锚点 2026Q2（该期动态 ≡ 静态，故当前状态结论不变，仅修正历史）。
    # 口径隔离：F1 保留静态（与工作簿 Sheet4 逐格可核对），F1d 为动态，
    #          两者并列进入检验，由数据决定哪个口径更有效。严禁混用。
    _wq = getattr(b, "weight_matrix", None)
    _w_static = b.hs300_weights.reindex(C.INDUSTRIES)
    _dyn_ok = (_wq is not None
               and getattr(b, "n_dynamic_quarters", 0) > 0)
    over_d = (alloc - _wq.reindex(index=Q, columns=INDUSTRIES)) if _dyn_ok else over.copy()
    ind["F1d_超配比例_动态基准"] = over_d
    ind["F2d_超配Z_动态基准"] = pd.DataFrame(
        {i: expanding_z(over_d[i]) for i in INDUSTRIES}, index=Q)[INDUSTRIES]
    ind["F3d_超配分位_动态基准"] = pd.DataFrame(
        {i: expanding_pct(over_d[i]) for i in INDUSTRIES}, index=Q)[INDUSTRIES]
    # 静态/动态基准权重并列，供审计
    ind["B_基准权重_静态%"] = pd.DataFrame(
        [ _w_static.to_numpy(dtype=float) ] * len(Q), index=Q, columns=INDUSTRIES)
    ind["B_基准权重_动态%"] = (_wq.reindex(index=Q, columns=INDUSTRIES)
                             if _dyn_ok else ind["B_基准权重_静态%"])
    mkt["动态基准生效期数"] = int(getattr(b, "n_dynamic_quarters", 0))

    # =============== F4 / F4b 超配动量 ===============
    mom = over.diff()
    ind["F4_超配动量"] = mom
    ind["F4b_超配动量Z"] = pd.DataFrame(
        {i: expanding_z(mom[i]) for i in INDUSTRIES}, index=Q)[INDUSTRIES]

    # =============== 长历史扩窗因子（解锁 F2/F3/F4b/F15 的样本边界）===============
    # 背景：主口径持仓仅 2024Q1~2026Q2 共 10 期，扩窗 MIN_HIST=8 起步 →
    #       F2/F3/F4b 只有最后 2 期有值、F15 全 NaN，无法做 IC 检验（遗留边界）。
    # 解法：用偏股混合单类型超配长历史（b.over_ext，2022Q1~2026Q2 共 17 期，口径统一，
    #       = 偏股混合配置占比 − 动态 hs300 权重）做扩窗标准化，使这些因子自 2024Q1
    #       起即可取值（F2e/F3e 有 9 期、F4be 有 8 期），IC 可检验。
    # 命名：F2e/F3e/F4be/F15e —— "e" = extended（长历史版）。与主口径因子并列、不覆盖；
    #       两者经济含义一致，差异仅在扩窗所用的超配历史口径（偏股混合单类型 vs 三类）。
    _oe = getattr(b, "over_ext", None)
    if _oe is not None and getattr(b, "over_ext_available", False):
        _oe = _oe.reindex(index=Q, columns=INDUSTRIES)
        ind["F2e_超配Z_长历史"] = pd.DataFrame(
            {i: expanding_z(_oe[i]) for i in INDUSTRIES}, index=Q)[INDUSTRIES]
        ind["F3e_超配分位_长历史"] = pd.DataFrame(
            {i: expanding_pct(_oe[i]) for i in INDUSTRIES}, index=Q)[INDUSTRIES]
        _oe_mom = _oe.diff()
        ind["F4be_动量Z_长历史"] = pd.DataFrame(
            {i: expanding_z(_oe_mom[i]) for i in INDUSTRIES}, index=Q)[INDUSTRIES]
        ind["F15e_拥挤背离_长历史"] = cz(_oe) - cz(ind["F2e_超配Z_长历史"])

    # =============== F5 HHI（时序） ===============
    mkt["F5_HHI"] = b.cr_hhi["HHI"].copy()
    mkt["F5b_HHI标准化"] = b.cr_hhi["HHI标准化"].copy()
    mkt["F5c_HHI扩窗分位"] = expanding_pct(mkt["F5_HHI"])
    # 复算校验
    hhi_re = ((alloc / 100.0) ** 2).sum(axis=1)
    ind["F5_HHI贡献"] = (alloc / 100.0) ** 2          # 行业对 HHI 的贡献，可横截面比较

    # =============== F6 CR3 / CR5 / CR10（时序） ===============
    mkt["F6_CR3"] = b.cr_hhi["CR3"].copy()
    mkt["F6_CR5"] = b.cr_hhi["CR5"].copy()
    mkt["F6_CR10"] = b.cr_hhi["CR10"].copy()
    mkt["F6_Top1比例"] = b.cr_hhi["Top1比例"].copy()
    mkt["F6_Top1行业"] = b.cr_hhi["Top1行业"].copy()

    # =============== F7 个股集中度（筹码拥挤，时序） ===============
    sc = b.stock_crowd
    f7 = pd.DataFrame({
        "Top20平均持股基金数": sc["Top20平均持股基金数"],
        "Top20持仓市值占比%": sc["Top20持仓市值占比"],
        "持股基金数≥100个股数": sc["持股基金数_ge100"],
        "持股基金数≥50个股数": sc["持股基金数_ge50"],
        "持股基金数≥20个股数": sc["持股基金数_ge20"],
        "个股总数": sc["个股总数"]})
    mkt["F7_个股集中度"] = f7
    sub = ["Top20平均持股基金数", "Top20持仓市值占比%", "持股基金数≥100个股数"]
    z7 = pd.DataFrame({c: zscore_ts(f7[c]) for c in sub}, index=Q)
    mkt["F7_个股集中度Z"] = z7.mean(axis=1)
    mkt["F7_子因子Z"] = z7

    # =============== F8 仓位代理（时序；弃用"隐含股票仓位"） ===============
    pos = b.position
    f8 = pd.DataFrame({
        "前十占净值比均值": pos["前十占净值比均值"],
        "前十占净值比中位数": pos["前十占净值比中位数"],
        "重仓股数均值": pos["重仓股数均值"]}, index=Q)
    mkt["F8_仓位代理"] = f8
    mkt["F8_仓位代理Z"] = zscore_ts(f8["前十占净值比均值"])
    mkt["F8b_仓位动量"] = pd.DataFrame({
        "前十占净值比均值_动量": f8["前十占净值比均值"].diff(),
        "前十占净值比中位数_动量": f8["前十占净值比中位数"].diff()}, index=Q)
    mkt["F8_样本基金数"] = pos["样本基金数"].reindex(Q)
    mkt["F8_沪深300季末点位"] = pos["沪深300季末点位"].reindex(Q)

    # =============== F9 交易拥挤度（行业成交额，不使用 ETF 数据） ===============
    ind["F9a_成交额占比%"] = share.copy()
    f9b = pd.DataFrame({i: expanding_pct(share[i]) for i in INDUSTRIES}, index=Q)[INDUSTRIES]
    ind["F9b_成交额占比分位"] = f9b
    accel = share - share.rolling(4, min_periods=4).mean().shift(1)
    ind["F9c_成交额占比加速度"] = accel
    # 交易拥挤信号：分位 ≥90 且 环比加速
    sig_tr = ((f9b >= 90) & (accel > 0)).astype(float)
    sig_tr = sig_tr.mask(f9b.isna() | accel.isna())
    ind["F9d_交易拥挤信号"] = sig_tr
    # 供 F11 使用的标准化合成（60% 分位 + 40% 加速度）
    ind["F9_交易拥挤度"] = (0.6 * cz(f9b) + 0.4 * cz(accel))

    # =============== F10 共振得分 ===============
    p90 = pd.DataFrame({i: expanding_quantile(over[i], 0.90) for i in INDUSTRIES},
                       index=Q)[INDUSTRIES]
    ind["F1b_超配P90阈值"] = p90          # 扩窗 P90 阈值（画图与信号判定共用）
    S1 = (over > p90).astype(float).mask(p90.isna())
    S2 = sig_tr.copy()
    # S3 筹码拥挤：任一抱团指标创扩窗新高
    hi = {}
    for c in sub:
        prev_max = f7[c].shift(1).expanding(min_periods=MIN_HIST).max()
        hi[c] = (f7[c] >= prev_max)
    hist_ok = f7[sub[0]].shift(1).expanding(min_periods=MIN_HIST).count() >= MIN_HIST
    s3_ts = pd.Series(False, index=Q)
    for c in sub:
        s3_ts |= hi[c].fillna(False)
    s3_ts &= hist_ok.fillna(False)
    S3 = pd.DataFrame(np.repeat(s3_ts.to_numpy(dtype=float)[:, None], NI, axis=1),
                      index=Q, columns=INDUSTRIES)
    S3 = S3.mask(~hist_ok.to_numpy()[:, None].repeat(NI, axis=1))
    ind["F10_S1_持仓拥挤"] = S1
    ind["F10_S2_交易拥挤"] = S2
    ind["F10_S3_筹码拥挤"] = S3
    score = S1.fillna(0) + S2.fillna(0) + S3.fillna(0)
    ind["F10_共振得分"] = score.mask(S1.isna() | S2.isna() | S3.isna())
    mkt["F10_S3筹码拥挤_时序"] = s3_ts.astype(float).mask(~hist_ok.fillna(False))

    # =============== F12 配置系数 = 配置比例 / 基准权重 ===============
    # 判断：> 2.5 极度拥挤；1.8~2.5 拥挤；< 0.5 显著低配
    Wb = b.weight_matrix.reindex(index=Q, columns=INDUSTRIES)
    with np.errstate(divide="ignore", invalid="ignore"):
        cfg = alloc / Wb.replace(0, np.nan)
    ind["F12_配置系数"] = cfg                     # 动态基准（若动态权重可用）
    ind["F12b_基准权重%"] = Wb
    # 静态基准对照（= 工作簿「配置比例/静态沪深300权重」）
    with np.errstate(divide="ignore", invalid="ignore"):
        cfg_s = alloc / _w_static.reindex(INDUSTRIES).replace(0, np.nan)
    ind["F12s_配置系数_静态基准"] = cfg_s
    ind["_静态与动态是否恒等"] = bool(
        float((Wb - ind["B_基准权重_静态%"]).abs().max().max()) < 1e-9)
    mkt["F12_基准权重来源"] = b.weight_source.reindex(Q)

    # =============== F13 筹码盈利比例（行业指数成本线代理） ===============
    # 精确口径需要「行业成分股逐股持仓成本 × 现价 × 流通市值」，属持仓数据库范畴。
    # 本项目用**行业指数层面**的可算代理：
    #   ① 由季度涨跌幅累乘得到行业指数相对点位 P（P0 = 100）；
    #   ② 取滚动 window 个季度的成交额 T_s，视为该价位上累积的筹码量；
    #   ③ 筹码盈利比例 = Σ_{P_s < P_t} T_s / Σ_{s∈window} T_s × 100%
    # 即"滚动窗口内存量筹码中，建仓价低于现价的比重"。
    # 该口径为**指数层面近似**，不等于成分股加权口径，已在文档与看板标注。
    def _profit_ratio(window: int) -> pd.DataFrame:
        lvl = 100.0 * (1 + b.market["ret"].reindex(index=Q, columns=INDUSTRIES)
                       / 100.0).cumprod()
        out = {}
        for col in INDUSTRIES:
            P, T = lvl[col], b.market["turn"][col].reindex(Q)
            vals = []
            for t in range(len(Q)):
                lo = max(0, t - window + 1)
                ps, ts = P.iloc[lo:t + 1], T.iloc[lo:t + 1]
                m = ps.notna() & ts.notna()
                if m.sum() < 2:
                    vals.append(np.nan)
                    continue
                ps, ts = ps[m], ts[m]
                cur = ps.iloc[-1]
                tot = ts.sum()
                vals.append(float(ts[ps < cur].sum() / tot * 100) if tot > 0 else np.nan)
            out[col] = vals
        return pd.DataFrame(out, index=Q)

    ind["F13_筹码盈利比例"] = _profit_ratio(12)          # 主口径：滚动 12 季度
    ind["F13b_筹码盈利比例_全历史"] = _profit_ratio(10 ** 9)  # 对照：自 2018Q1 起全历史

    # =============== F14 ETF资金流强度 = 行业ETF净申赎 / 期初总份额·净值 ===============
    # 数据源：ETF 日度 场内流通份额 + 单位净值（真实数据，份额折算已复权）
    # 强度(%) = Σ期内净申赎(亿元) / 期初AUM(亿元) × 100
    etf = getattr(b, "etf_daily", None)
    f14 = pd.DataFrame(np.nan, index=Q, columns=INDUSTRIES)
    mkt["F14_ETF明细"] = None
    if etf is not None and not etf.empty:
        e = etf.copy()
        e["季度"] = pd.to_datetime(e["日期"]).dt.to_period("Q").astype(str)
        e["季度"] = (e["季度"].str[:4] + "Q" + e["季度"].str[-1])
        rows, daily_rows = [], []
        for (ind_name, q), g in e.groupby(["申万一级", "季度"]):
            g = g.sort_values("日期")
            if ind_name not in INDUSTRIES:
                continue
            aum0 = float((g["复权份额"].iloc[0] * g["复权净值"].iloc[0]))
            net = float(g["净申赎亿元"].sum(skipna=True))
            strength = net / aum0 * 100.0 if aum0 > 0 else np.nan
            # 明细表保留所有季度（含持仓数据之后的季度，如 2026Q3）
            rows.append(dict(季度=q, 行业=ind_name, ETF数=int(g["代码"].nunique()),
                             期初AUM亿元=round(aum0, 2), 区间净申赎亿元=round(net, 2),
                             资金流强度=round(strength, 2) if strength == strength else np.nan))
            # 因子矩阵只在持有持仓数据的季度上填值（其余季度留 NaN）
            if q in f14.index:
                f14.loc[q, ind_name] = strength
        mkt["F14_ETF明细"] = pd.DataFrame(rows)
        # 日度累计强度（用于 ETF 资金流页面）
        for code, g in e.groupby("代码"):
            g = g.sort_values("日期").copy()
            aum0 = float(g["复权份额"].iloc[0] * g["复权净值"].iloc[0])
            g["累计净申赎亿元"] = g["净申赎亿元"].fillna(0).cumsum()
            g["累计资金流强度%"] = (g["累计净申赎亿元"] / aum0 * 100).round(C.DEC_PCT)
            daily_rows.append(g)
        if daily_rows:
            mkt["F14_ETF日度"] = pd.concat(daily_rows, ignore_index=True)
    ind["F14_ETF资金流强度"] = f14

    # =============== F11 行业综合拥挤度（本次升级后的推荐权重） ===============
    # 推荐权重：F1=0.20, F12=0.25, F3=0.15, F13=0.15, F14=0.15, F4b=0.10
    comp = {k: ind[k] for k in C.W_INDUSTRY if k in ind}
    rec, zs = weighted_z_composite(comp, C.W_INDUSTRY, Q, INDUSTRIES, C.MIN_COMPONENTS)
    ind["F11_综合拥挤度_推荐权重"] = rec
    ind["_F11_分量Z"] = pd.DataFrame(
        {k: v.stack() for k, v in zs.items()}) if zs else pd.DataFrame()

    # (b) 等权（仅用可用的分量）
    eq_w = {k: 1.0 for k in C.W_INDUSTRY if k in ind}
    eq, _ = weighted_z_composite(comp, eq_w, Q, INDUSTRIES, C.MIN_COMPONENTS)
    ind["F11_综合拥挤度_等权"] = eq

    # (c) 历史对照：沿用旧口径（F2/F3/F4b/F9），便于与上一版结果对比
    old_keys = {"F2_超配Zscore": 0.35, "F3_超配历史分位": 0.30,
                "F4b_超配动量Z": 0.10, "F9_交易拥挤度": 0.10}
    old_comp = {k: ind[k] for k in old_keys if k in ind}
    oldc, _ = weighted_z_composite(old_comp, old_keys, Q, INDUSTRIES, 2)
    ind["F11_综合拥挤度_旧口径对照"] = oldc

    # (c2) IC 加权：权重 = 各分量截至 t−2 期的扩窗平均 RankIC（严格无前视）
    fwd = forward_excess(b.market, lag=1)
    ic_w = {k: rank_ic_panel(cz(comp[k]), fwd).shift(2)
            .expanding(min_periods=max(MIN_HIST - 2, 2)).mean() for k in comp}
    Wdf = pd.DataFrame(ic_w, index=Q)
    Wn = Wdf.div(Wdf.abs().sum(axis=1), axis=0)
    # 以「推荐权重」为兜底：早期/缺失期 IC 无定义时不做等权，而沿用推荐权重比例
    base = pd.Series({k: C.W_INDUSTRY[k] / sum(C.W_INDUSTRY.values()) for k in comp})
    Wn_filled = Wn.where(Wn.notna().all(axis=1), np.nan)
    for k in comp:
        Wn_filled[k] = Wn_filled[k].fillna(base[k])
    row_sum = Wn_filled.sum(axis=1).replace(0, np.nan)
    Wn_filled = Wn_filled.div(row_sum, axis=0).fillna(base)
    icw, _ = weighted_z_composite(comp, Wn_filled.iloc[-1].to_dict(), Q, INDUSTRIES,
                                  C.MIN_COMPONENTS)
    # 逐期权重不同 → 手工按 (T, n, I) 广播计算
    names = list(comp)
    stack = np.stack([cz(comp[k]).to_numpy(dtype=float) for k in names])
    avail = ~np.isnan(stack)
    wts = np.repeat(Wn_filled[names].to_numpy(dtype=float)[:, :, None], NI, axis=2)
    W = avail * wts.transpose(1, 0, 2)
    Wsum = W.sum(axis=0)
    icw_arr = np.nansum(W * np.nan_to_num(stack), axis=0) / np.where(Wsum > 0, Wsum, np.nan)
    icw_arr[avail.sum(axis=0) < C.MIN_COMPONENTS] = np.nan
    ind["F11_综合拥挤度_IC加权"] = pd.DataFrame(icw_arr, index=Q, columns=INDUSTRIES)
    ind["_F11_IC权重"] = Wn_filled

    # =============== 市场层面综合拥挤度 ===============
    mcomp = {}
    mcomp["F5_HHI"] = mkt["F5_HHI"]
    mcomp["F6_CR3"] = mkt["F6_CR3"]
    mcomp["F7_个股集中度Z"] = mkt["F7_个股集中度Z"]
    mcomp["F8_仓位代理Z"] = mkt["F8_仓位代理Z"]
    MZ = pd.DataFrame({k: zscore_ts(mcomp[k]) for k in mcomp}, index=Q)
    mkt["F11_市场综合拥挤度Z"] = sum(C.W_MARKET[k] * MZ[k] for k in C.W_MARKET)
    mkt["_F11_市场分量Z"] = MZ
    mkt["F11_市场综合拥挤度分位"] = expanding_pct(mkt["F11_市场综合拥挤度Z"])

    # =============== FCI 因子拥挤指数 ===============
    # FCI = ( z_空头集中度 + z_ETF资金流 + z_因子相关性 ) / 3
    #
    # ① z_空头集中度：每期取综合拥挤度最低的 1/3 行业（"空头腿"），
    #    计算其配置比例的 HHI。空头腿越集中，说明"低拥挤"这一侧越拥挤，
    #    因子越容易被反向踩踏。再做时序扩窗 z。
    # ② z_ETF资金流：全市场 ETF 净申赎强度（Σ净申赎 / Σ期初AUM）的时序扩窗 z。
    # ③ z_因子相关性：F11 与其余候选因子的平均 |Spearman| 秩相关，
    #    越高说明该因子与既有因子越冗余 → 因子本身越拥挤。时序扩窗 z。
    n_short = max(int(NI / 3), 5)
    short_hhi = pd.Series(np.nan, index=Q, name="空头集中度HHI")
    for q in Q:
        s_row = ind["F11_综合拥挤度_推荐权重"].loc[q].dropna()
        if len(s_row) < n_short + 3:
            continue
        leg = s_row.nsmallest(n_short).index
        w = alloc.loc[q, leg] / 100.0
        tot = w.sum()
        if tot > 0:
            short_hhi[q] = float(((w / tot) ** 2).sum())

    etf_flow_ts = pd.Series(np.nan, index=Q, name="ETF净申赎强度")
    if mkt["F14_ETF明细"] is not None and not mkt["F14_ETF明细"].empty:
        d = mkt["F14_ETF明细"]
        for q, g in d.groupby("季度"):
            if q in etf_flow_ts.index:
                tot_aum = g["期初AUM亿元"].sum()
                if tot_aum > 0:
                    etf_flow_ts[q] = float(g["区间净申赎亿元"].sum() / tot_aum * 100)

    others = ["F1_超配比例", "F12_配置系数", "F3_超配历史分位", "F13_筹码盈利比例",
              "F4b_超配动量Z", "F9_交易拥挤度", "F10_共振得分"]
    corr_ts = pd.Series(np.nan, index=Q, name="因子相关性")
    for q in Q:
        base = ind["F11_综合拥挤度_推荐权重"].loc[q]
        cs = []
        for o in others:
            if o not in ind:
                continue
            a2 = base.dropna()
            b2 = ind[o].loc[q].reindex(a2.index).dropna()
            a2 = a2.reindex(b2.index)
            if len(b2) >= 8:
                _c = safe_spearman(a2, b2)
                if _c == _c:
                    cs.append(abs(_c))
        if cs:
            corr_ts[q] = float(np.mean(cs))

    fci_comp = pd.DataFrame({
        "z_空头集中度": zscore_ts(short_hhi),
        "z_ETF资金流": zscore_ts(etf_flow_ts),
        "z_因子相关性": zscore_ts(corr_ts)})
    fci = fci_comp.mean(axis=1)
    fci[fci_comp.notna().sum(axis=1) < 2] = np.nan
    mkt["FCI_因子拥挤指数"] = fci
    mkt["FCI_分量"] = fci_comp
    mkt["FCI_原始值"] = pd.DataFrame({"空头集中度HHI": short_hhi,
                                     "ETF净申赎强度": etf_flow_ts,
                                     "因子相关性": corr_ts})
    mkt["FCI_分位"] = expanding_pct(fci)

    # =============== 拥挤等级 ===============
    # ⚠️ 必须把 F1（超配的**绝对水平**）一并传入，用于约束「相对历史」口径的语义。
    # 原因见 _levels 文档字符串。
    ind["拥挤等级"] = _levels(ind["F2_超配Zscore"], ind["F3_超配历史分位"],
                              ind["F12_配置系数"], OV=ind["F1_超配比例"])
    ind["口径背离"] = divergence_flag(ind["F2_超配Zscore"], ind["F3_超配历史分位"],
                                  ind["F12_配置系数"])

    # =============== F15 拥挤背离因子 ===============
    # 诊断发现：F1（超配的**绝对水平**）IC = −0.056（符合"拥挤=风险"），
    #           而 F2（超配相对**自身历史**的极端度）IC = +0.084（趋势延续）。
    # 两者符号相反 → 它们刻画的是**两个不同的风险维度**。
    # 取二者之差 = "绝对超配高、但相对自身历史并不极端" → **新增的拥挤**
    # （新资金刚进来，历史分位还没抬起来），理论上是最危险的一类。
    #   F15(i,t) = z_cs(F1) − z_cs(F2)
    ind["F15_拥挤背离"] = cz(over) - cz(ind["F2_超配Zscore"])
    # 动态基准版本：F15d = z_cs(F1d) − z_cs(F2d)
    ind["F15d_拥挤背离_动态基准"] = cz(over_d) - cz(ind["F2d_超配Z_动态基准"])

    # =============== 正交化残差因子（作为一等因子） ===============
    # 目的：检验基金持仓数据是否含**独立于价格类风格**的增量信息。
    # 做法：逐期对 3 个**同期可观测**的风格控制变量做横截面 OLS 取残差。
    #       控制变量 = 第 t 期收益 / 过去 4 季收益 / 过去 4 季波动（均截至 t 期，无未来信息）。
    _ctrl_names = ["CTRL_第t期收益", "CTRL_过去4季收益", "CTRL_过去4季波动"]

    def _residualize(fac, ctrls):
        out = pd.DataFrame(np.nan, index=fac.index, columns=fac.columns, dtype=float)
        for q in fac.index:
            y = fac.loc[q]
            m = y.notna()
            for c in ctrls:
                m &= c.loc[q].notna()
            if int(m.sum()) < 15:
                continue
            cols = m.index[m]
            X = np.column_stack([np.ones(len(cols))] +
                                [c.loc[q].reindex(cols).to_numpy(dtype=float)
                                 for c in ctrls])
            try:
                beta, *_ = np.linalg.lstsq(X, y.reindex(cols).to_numpy(dtype=float),
                                           rcond=None)
            except np.linalg.LinAlgError:
                continue
            out.loc[q, cols] = y.reindex(cols).to_numpy(dtype=float) - X @ beta
        return out

    # =============== 风格控制变量（正交化用） ===============


    ind["CTRL_第t期收益"] = b.market["ret"].reindex(columns=INDUSTRIES)
    ind["CTRL_过去4季收益"] = b.market["ret"].rolling(4, min_periods=4).apply(
        lambda a: (np.prod(1 + a / 100) - 1) * 100, raw=True).shift(1)
    ind["CTRL_过去4季波动"] = b.market["ret"].rolling(4, min_periods=4).std().shift(1)

    # ---- 正交化残差因子：必须在 CTRL_* 定义之后执行 ----
    # 注意：Spearman 对逐期常数平移不变，因此若控制变量为空，
    #       残差 = 去均值后仍是同序 → IC 完全不变（这曾导致一个静默 bug）。
    #       故此处强制校验控制变量数量。
    _ctrl = [ind[c] for c in _ctrl_names if c in ind]
    assert len(_ctrl) == len(_ctrl_names), \
        f"正交化控制变量缺失：期望 {_ctrl_names}，实际 {len(_ctrl)} 个"
    for _src, _tgt in [("F1_超配比例", "R_F1_超配比例_残差"),
                       ("F2_超配Zscore", "R_F2_超配Z_残差"),
                       ("F3_超配历史分位", "R_F3_超配分位_残差"),
                       ("F4b_超配动量Z", "R_F4b_动量Z_残差"),
                       ("F11_综合拥挤度_推荐权重", "R_F11_综合_残差")]:
        if _src in ind:
            ind[_tgt] = _residualize(ind[_src], _ctrl)

    raw = dict(over=over, alloc=alloc, share=share, accel=accel, p90=p90,
               hhi_recalc=hhi_re, hs300_w=b.hs300_weights)
    return dict(ind=ind, mkt=mkt, raw=raw)


# ============================================================================
# 1b. 加权 z 合成（缺失分量自动按可用权重重新归一化）
# ============================================================================
def weighted_z_composite(components: dict, weights: dict, index, columns,
                         min_comp: int = 2):
    """对每个分量做横截面 z-score 后按权重合成。

    关键设计：某期某行业若缺失部分分量（如 F14 仅有 2026Q3 的 ETF 数据），
    则**只在可用分量之间重新归一化权重**，而不是把缺失当 0 或直接整行丢弃。
    但当可用分量数 < min_comp 时该单元置 NaN（避免单分量代表"综合"）。

    返回 (composite_df, z_dict)
    """
    names = [k for k in weights if k in components]
    zs = {k: cz(components[k]) for k in names}
    if not names:
        return pd.DataFrame(index=index, columns=columns, dtype=float), {}
    stack = np.stack([zs[k].to_numpy(dtype=float) for k in names])       # (n,T,I)
    w = np.array([weights[k] for k in names], dtype=float)[:, None, None]
    avail = ~np.isnan(stack)
    W = avail * w
    Wsum = W.sum(axis=0)
    out = np.nansum(W * np.nan_to_num(stack), axis=0) / np.where(Wsum > 0, Wsum, np.nan)
    out[avail.sum(axis=0) < min_comp] = np.nan
    return pd.DataFrame(out, index=index, columns=columns), zs


# ============================================================================
# 2. 拥挤等级
# ============================================================================
def _levels(Z: pd.DataFrame, PCT: pd.DataFrame,
            CF: pd.DataFrame | None = None,
            OV: pd.DataFrame | None = None) -> pd.DataFrame:
    """按任务书判定顺序生成等级标签（**已加入绝对方向约束**）。

    ① 极度拥挤：超配分位 ≥ 90  **或** 配置系数 F12 > 2.5
    ② 拥挤    ：Z > 1.5        **或** 配置系数 F12 > 1.8
    ③ 偏拥挤  ：分位 75 ~ 90
    ④ 显著低配：Z < −1.5       **或** 配置系数 F12 < 0.5
    ⑤ 中性    ：其余

    ---------------------------------------------------------------------------
    ⚠️ 本轮自主发现并修正的方法论缺陷（机构级必须留痕）
    ---------------------------------------------------------------------------
    Z 与分位都是**相对该行业自身历史**的口径，本身不含"绝对方向"。
    早期实现只用「分位 ≥ 90 ⇒ 极度拥挤」，在长期单边低配的行业上会得出
    **与事实完全相反**的标签。实测（2026Q2，修正基准后）：

        银行    配置 1.12% / 基准 9.36% → 超配 −8.24pct（深度低配）
                但其分位 = 100%（历史上"最不低配"的一期）
                ⇒ 旧规则判为「极度拥挤」❌

    这类误判的后果最严重：它会让使用者去规避一个**已经严重低配**的行业，
    方向恰好相反。同理，「超配但相对自身历史偏低」的行业（如医药生物
    +1.97pct / 分位 0%）被旧规则判为「显著低配」，同样是方向错误。

    正确语义：**先按绝对偏离定方向，再按相对历史定强度**。
        · 超配 > 0 的行业，只能落在「中性 → 偏拥挤 → 拥挤 → 极度拥挤」一侧；
        · 超配 < 0 的行业，只能落在「中性 → 显著低配」一侧。
    两者冲突的单元（如"绝对超配但历史分位极低"）一律归入**中性**，
    并已由 `口径背离` 标志单独标注、在看板中并列呈现两个口径。

    参数 OV 为超配比例（F1）。为向后兼容，OV=None 时退化为旧行为，
    但当前所有生产调用都已传入 OV。
    """
    lv = pd.DataFrame("中性", index=Z.index, columns=Z.columns, dtype=object)
    valid = Z.notna() & PCT.notna()
    if CF is not None:
        cf = CF.reindex(index=Z.index, columns=Z.columns)
    else:
        cf = pd.DataFrame(np.nan, index=Z.index, columns=Z.columns)
    # 绝对方向：True = 确实超配；False = 确实低配；NaN → 不施加方向约束
    if OV is not None:
        ov = OV.reindex(index=Z.index, columns=Z.columns)
        over = ov > 0.0
        under = ov < 0.0
    else:
        over = pd.DataFrame(True, index=Z.index, columns=Z.columns)
        under = pd.DataFrame(True, index=Z.index, columns=Z.columns)

    # ---- 超配一侧：拥挤程度（低 → 高）----
    lv = lv.mask(valid & over & (PCT >= C.THRESH_PCT_CROWDED), "偏拥挤")
    lv = lv.mask(valid & over & ((Z > C.THRESH_Z_CROWDED) |
                                 (cf > C.THRESH_CF_CROWDED)), "拥挤")
    lv = lv.mask(valid & over & ((PCT >= C.THRESH_PCT_EXTREME) |
                                 (cf > C.THRESH_CF_EXTREME)), "极度拥挤")
    # ---- 低配一侧：低配程度 ----
    lv = lv.mask(valid & under & ((Z < C.THRESH_Z_UNDER) |
                                  (cf < C.THRESH_CF_UNDER)), "显著低配")
    lv = lv.mask(~valid, "数据不足")
    return lv


def divergence_flag(Z: pd.DataFrame, PCT: pd.DataFrame,
                    CF: pd.DataFrame) -> pd.DataFrame:
    """标记「口径背离」：Z/分位与 F12 指向相反的行业-季度。

    典型：医药生物 2026Q2 —— 超配分位 0%（自身历史极低位）但配置系数 2.51
    （相对基准极重配）。这类单元不能只用一个标签表达，必须同时呈现两个口径。
    """
    z_crowded = (Z > 0) | (PCT >= 50)
    z_under = (Z < 0) | (PCT < 50)
    cf_crowded = CF > 1.5
    cf_under = CF < 0.8
    return ((z_crowded & cf_under) | (z_under & cf_crowded)).fillna(False)


def level_series(lv_row: pd.Series) -> pd.Series:
    return lv_row


# ============================================================================
# 3. 市场汇总
# ============================================================================
def market_summary(ind: dict, mkt: dict, b, quarter: str | None = None) -> dict:
    """给定季度的市场层面汇总卡片数据。"""
    q = quarter or list(mkt["F5_HHI"].index)[-1]
    lv = ind["拥挤等级"].loc[q]
    n_ext = int((lv == "极度拥挤").sum())
    n_crowd = int(lv.isin(["极度拥挤", "拥挤"]).sum())
    n_under = int((lv == "显著低配").sum())
    n_neutral = int((lv == "中性").sum())
    hhi = float(mkt["F5_HHI"].loc[q])
    hhi_level = next(name for thr, name in C.HHI_LEVELS if hhi < thr)
    cr3 = float(mkt["F6_CR3"].loc[q])
    top1 = mkt["F6_Top1行业"].loc[q]
    top1_pct = float(mkt["F6_Top1比例"].loc[q])
    hist_mean = float(mkt["F5_HHI"].mean())
    # 当前 HHI 在历史中的分位
    # ⚠️ 分母必须是**有效观测数**，不能用 len(series)。
    # mkt["F5_HHI"] 的索引覆盖全部 30 个季度（2019Q1~2026Q2），但 HHI 只在
    # 有基金持仓的 10 个季度上有值，其余为 NaN。早期实现写
    #   (hh < hhi).sum() / len(hh)
    # 分母含 20 个空季度 → HHI 明明创历史新高，分位却只显示 30%（9/30）。
    # 这是一个会让使用者把"历史极值"误读为"中位水平"的静默错误。
    #
    # 分位口径（与项目其余处一致）：分位 = 严格小于当前值的观测数 ÷ 有效观测数。
    # 因此**全样本最大值**的分位为 (n−1)/n（如 10 期 → 90%），而非 100%；
    # 这与分级阈值（90 / 75 / 10）同构，最大值恰好触发"极度拥挤"。
    hh = mkt["F5_HHI"].dropna()
    hhi_pct = float((hh < hhi).sum() / len(hh) * 100) if len(hh) else np.nan
    m = dict(quarter=q, HHI=hhi, HHI_norm=float(mkt["F5b_HHI标准化"].loc[q]),
             HHI_level=hhi_level, HHI_hist_mean=hist_mean, HHI_pct=hhi_pct,
             CR3=cr3, CR5=float(mkt["F6_CR5"].loc[q]),
             CR10=float(mkt["F6_CR10"].loc[q]),
             top1=top1, top1_pct=top1_pct,
             n_extreme=n_ext, n_crowded=n_crowd, n_under=n_under, n_neutral=n_neutral,
             n_funds=float(mkt["F8_样本基金数"].loc[q]),
             market_crowd_z=float(mkt["F11_市场综合拥挤度Z"].loc[q])
             if mkt["F11_市场综合拥挤度Z"].loc[q] == mkt["F11_市场综合拥挤度Z"].loc[q] else None,
             extreme_industries=list(lv.index[lv == "极度拥挤"]),
             under_industries=list(lv.index[lv == "显著低配"]))
    for name, fn in C.MARKET_STATE_RULES:
        if fn(m):
            m["state"] = name
            break
    return m


# ============================================================================
# 4. 面板展开（长表，供导出与看板使用）
# ============================================================================
PANEL_COLS = [
    "F1_超配比例", "F2_超配Zscore", "F3_超配历史分位", "F4_超配动量", "F4b_超配动量Z",
    "F5_HHI贡献", "F9a_成交额占比%", "F9b_成交额占比分位", "F9c_成交额占比加速度",
    "F9d_交易拥挤信号", "F9_交易拥挤度",
    "F10_S1_持仓拥挤", "F10_S2_交易拥挤", "F10_S3_筹码拥挤", "F10_共振得分",
    "F1d_超配比例_动态基准", "F2d_超配Z_动态基准", "F3d_超配分位_动态基准",
    "B_基准权重_静态%", "B_基准权重_动态%",
    "F12_配置系数", "F12s_配置系数_静态基准", "F12b_基准权重%",
    "F13_筹码盈利比例", "F13b_筹码盈利比例_全历史",
    "F14_ETF资金流强度", "F15_拥挤背离", "F15d_拥挤背离_动态基准",
    "R_F1_超配比例_残差", "R_F2_超配Z_残差", "R_F3_超配分位_残差",
    "R_F4b_动量Z_残差", "R_F11_综合_残差",
    "F11_综合拥挤度_推荐权重", "F11_综合拥挤度_等权", "F11_综合拥挤度_IC加权",
    "F11_综合拥挤度_旧口径对照",
]


def to_panel(ind: dict, b, extra=True) -> pd.DataFrame:
    """把行业因子摊平成 (季度 × 行业) 长表。"""
    frames = []
    for c in PANEL_COLS:
        if c not in ind:
            continue
        d = ind[c].stack().rename(c)
        frames.append(d)
    if extra:
        for c, nm in [("alloc", "配置比例%"), ("overweight", "超配比例%")]:
            src = b.alloc if c == "alloc" else b.overweight
            frames.append(src.stack().rename(nm))
        frames.append(ind["拥挤等级"].stack().rename("拥挤等级"))
        if "口径背离" in ind:
            frames.append(ind["口径背离"].stack().rename("口径背离"))
        frames.append(b.market["ret"].stack().rename("行业涨跌幅%"))
        frames.append(b.market["turn_share"].stack().rename("成交额占比%"))
    panel = pd.concat(frames, axis=1)
    panel.index.names = ["季度", "行业"]
    panel = panel.reset_index()
    panel = panel.rename(columns={"level_0": "季度", "level_1": "行业"})
    return panel


def market_indicators(mkt: dict, b) -> pd.DataFrame:
    """市场层面指标时序长表。"""
    d = pd.DataFrame({
        "季度": mkt["F5_HHI"].index,
        "HHI": mkt["F5_HHI"].to_numpy(),
        "HHI标准化": mkt["F5b_HHI标准化"].to_numpy(),
        "HHI扩窗分位": mkt["F5c_HHI扩窗分位"].to_numpy(),
        "CR3": mkt["F6_CR3"].to_numpy(),
        "CR5": mkt["F6_CR5"].to_numpy(),
        "CR10": mkt["F6_CR10"].to_numpy(),
        "Top1行业": mkt["F6_Top1行业"].to_numpy(),
        "Top1比例": mkt["F6_Top1比例"].to_numpy(),
        "Top20平均持股基金数": mkt["F7_个股集中度"]["Top20平均持股基金数"].to_numpy(),
        "Top20持仓市值占比": mkt["F7_个股集中度"]["Top20持仓市值占比%"].to_numpy(),
        "持股基金数≥100个股数": mkt["F7_个股集中度"]["持股基金数≥100个股数"].to_numpy(),
        "个股集中度Z": mkt["F7_个股集中度Z"].to_numpy(),
        "前十占净值比均值": mkt["F8_仓位代理"]["前十占净值比均值"].to_numpy(),
        "前十占净值比中位数": mkt["F8_仓位代理"]["前十占净值比中位数"].to_numpy(),
        "仓位代理Z": mkt["F8_仓位代理Z"].to_numpy(),
        "仓位动量": mkt["F8b_仓位动量"]["前十占净值比均值_动量"].to_numpy(),
        "样本基金数": mkt["F8_样本基金数"].to_numpy(),
        "沪深300季末点位": mkt["F8_沪深300季末点位"].to_numpy(),
        "市场综合拥挤度Z": mkt["F11_市场综合拥挤度Z"].to_numpy(),
        "S3筹码拥挤": mkt["F10_S3筹码拥挤_时序"].to_numpy(),
        "FCI因子拥挤指数": mkt["FCI_因子拥挤指数"].to_numpy(),
        "FCI分位": mkt["FCI_分位"].to_numpy(),
        "基准权重来源": mkt["F12_基准权重来源"].astype(str).to_numpy(),
    })
    for _c in ["空头集中度HHI", "ETF净申赎强度", "因子相关性"]:
        d[f"FCI_{_c}"] = mkt["FCI_原始值"][_c].to_numpy()
    d["沪深300涨跌幅%"] = b.market["hs300"].reindex(d["季度"]).to_numpy()
    return d
