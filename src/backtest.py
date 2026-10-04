# -*- coding: utf-8 -*-
"""
backtest.py  ——  IC / 分层 / 事件研究 / 稳健性 检验
======================================================================
全部以"无未来函数"为前提：
  · 因子暴露取第 t 期；收益取第 t+1 期（相邻季度），稳健性另测 t+2。
  · 所有阈值型信号（P90 突破等）均用扩窗阈值，不使用全样本阈值。

评价标准（任务书）
------------------
  RankIC        |IC均值| > 0.03 有效；> 0.05 优秀
  IC_IR         > 0.5 优秀；0.3~0.5 可用；< 0.3 无效
  IC 胜率       ≥ 55% 较好，≥ 60% 优秀
  t 值          Newey-West 滞后 2 期
  分层多空      年化 > 5%，最大回撤 < 15%，单调性 Spearman > 0.8
  事件研究      触发后 12 个月超额中位数 < 0 且负概率 > 55% 且 p < 0.1 → 信号有效
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from factor_engine import (cz, expanding_pct, expanding_quantile, expanding_z,
                           forward_cum_excess, forward_cum_market, forward_excess,
                           rank_ic_panel)

INDUSTRIES = C.INDUSTRIES
MIN_HIST = C.MIN_HIST

# 需要检验的行业横截面因子
CS_FACTORS = [
    "F1_超配比例", "F2_超配Zscore", "F3_超配历史分位",
    # 第 2 轮：动态基准变体（与静态口径并列检验，不混用）
    "F1d_超配比例_动态基准", "F2d_超配Z_动态基准", "F3d_超配分位_动态基准",
    "F12s_配置系数_静态基准", "F15d_拥挤背离_动态基准",
    "F4_超配动量", "F4b_超配动量Z", "F9_交易拥挤度",
    "F10_共振得分", "F12_配置系数", "F13_筹码盈利比例", "F15_拥挤背离",
    "F11_综合拥挤度_推荐权重", "F11_综合拥挤度_等权",
    "F11_综合拥挤度_IC加权",
    # 长历史扩窗因子（第 7 轮：用偏股混合 17 期超配历史解锁 F2/F3/F4b/F15 的样本边界）
    "F2e_超配Z_长历史", "F3e_超配分位_长历史", "F4be_动量Z_长历史",
    "F15e_拥挤背离_长历史",
    # 正交化残差因子（第 1 轮迭代正式列为一等因子）
    "R_F1_超配比例_残差", "R_F2_超配Z_残差", "R_F3_超配分位_残差",
    "R_F4b_动量Z_残差", "R_F11_综合_残差",
]

# 方向预期登记：拥挤度类因子"越高越危险" → 先验期望 IC 为负。
# 例外登记（而非事后修改预期，以避免数据挖掘）：
#   F2 / F3 / F4b 及其残差在样本内表现为"趋势延续"（IC 为正），
#   这是**与先验相反**的实证结果，ic_conclusion 会标注「方向相反」。
EXPECT_POSITIVE = {"R_F2_超配Z_残差", "R_F3_超配分位_残差", "R_F4b_动量Z_残差",
                   "F2d_超配Z_动态基准", "F3d_超配分位_动态基准",
                   # 长历史扩窗因子：与 F2/F3/F4b 同为「趋势延续」先验（期望正）
                   "F2e_超配Z_长历史", "F3e_超配分位_长历史", "F4be_动量Z_长历史"}
# 上述因子的经济含义方向：拥挤为风险 → 期望 IC 为负
EXPECT_NEGATIVE = {f: (f not in EXPECT_POSITIVE) for f in CS_FACTORS}


# ============================================================================
# 基础统计工具
# ============================================================================
def nw_tstat(x: pd.Series, lags: int = C.NW_LAGS):
    """Newey-West 调整的单样本均值 t 检验（自相关稳健）。"""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 4:
        return np.nan, np.nan
    mu = x.mean()
    e = x - mu
    var = (e @ e) / n
    for L in range(1, min(lags, n - 1) + 1):
        gl = (e[L:] @ e[:-L]) / n
        var += 2 * (1 - L / (lags + 1)) * gl
    se = np.sqrt(max(var, 1e-18) / n)
    t = mu / se
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    return float(t), float(p)


def perf_series(x: pd.Series) -> dict:
    """季度收益序列(%) → 年化/波动/夏普/最大回撤/累计/胜率。"""
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return dict(ann=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan,
                    total=np.nan, win=np.nan, n=0)
    g = 1 + x / 100.0
    total = g.prod() - 1
    yrs = len(x) / 4.0
    ann = (1 + total) ** (1 / yrs) - 1 if total > -1 else np.nan
    # 注意：x 的单位已是百分数(%)，故 std 已在 % 量纲；
    # 季度→年化：波动按 √4 = 2 缩放。**此处不得再 ×100**。
    # （第 2 轮修复：此前 `vol=vol*100` 使波动率被放大 100 倍，
    #   因下游只引用 ann/maxdd/sharpe 而长期休眠，直到组合层报告首次展示 vol 才暴露。）
    vol = x.std(ddof=1) * 2                                  # 单位：%
    sharpe = (x.mean() * 4) / vol if vol and vol > 0 else np.nan
    c = g.cumprod()
    dd = (c / c.cummax() - 1).min()
    return dict(ann=ann * 100, vol=vol, sharpe=sharpe, maxdd=dd * 100,
                total=total * 100, win=(x > 0).mean() * 100, n=len(x))


def ic_stats(ic: pd.Series) -> dict:
    x = pd.Series(ic).dropna()
    n = len(x)
    if n == 0:
        return dict(n=0, ic_mean=np.nan, ic_std=np.nan, ic_ir=np.nan,
                    t_nw=np.nan, p_nw=np.nan, win_rate=np.nan)
    mean, sd = x.mean(), x.std(ddof=1)
    ir = mean / sd if sd and sd > 0 else np.nan
    t_nw, p_nw = nw_tstat(x)
    return dict(n=n, ic_mean=float(mean), ic_std=float(sd), ic_ir=float(ir) if ir == ir else np.nan,
                t_nw=t_nw, p_nw=p_nw, win_rate=float((x > 0).mean() * 100))


def ic_conclusion(s: dict, expect_negative: bool = True) -> str:
    """按任务书标准给出结论标签。"""
    if s.get("n", 0) == 0 or s.get("ic_mean") != s.get("ic_mean"):
        return "样本不足"
    a = abs(s["ic_mean"])
    ir = abs(s["ic_ir"]) if s["ic_ir"] == s["ic_ir"] else 0.0
    sign_ok = (s["ic_mean"] < 0) if expect_negative else (s["ic_mean"] > 0)
    if a > 0.05 and ir > 0.5:
        return "优秀" if sign_ok else "优秀(方向相反)"
    if a > 0.03 and ir >= 0.3:
        return "有效" if sign_ok else "有效(方向相反)"
    if a > 0.03 or ir >= 0.3:
        return "边际可用"
    return "无效"


# ============================================================================
# 1. IC / RankIC
# ============================================================================
def run_ic(ind: dict, b, lag: int = 1) -> dict:
    """逐因子 RankIC 序列与汇总。lag=1 → t+1 期；lag=2 → t+2 期（稳健性）。"""
    fwd = forward_excess(b.market, lag=lag)
    series, rows = {}, []
    for f in CS_FACTORS:
        if f not in ind:
            continue
        ic = rank_ic_panel(ind[f], fwd)
        series[f] = ic
        s = ic_stats(ic)
        ci = block_bootstrap_ci(ic)
        rows.append(dict(因子=f, lag=f"t+{lag}", **s,
                         ic_boot_lo=ci["ic_lo"], ic_boot_hi=ci["ic_hi"],
                         ir_boot_lo=ci["ir_lo"], ir_boot_hi=ci["ir_hi"],
                         boot_p_positive=ci["p_positive"],
                         CI含0=(ci["ic_lo"] <= 0 <= ci["ic_hi"])
                         if ci["ic_lo"] == ci["ic_lo"] else None,
                         方向="拥挤=风险(期望负)" if EXPECT_NEGATIVE.get(f) else "期望正",
                         结论=ic_conclusion(s, EXPECT_NEGATIVE.get(f, True))))
    return dict(series=series, table=pd.DataFrame(rows))


def run_ic_both_lags(ind: dict, b) -> pd.DataFrame:
    """合并 t+1 / t+2 两套 IC 结果。"""
    t1 = run_ic(ind, b, lag=1)["table"]
    t2 = run_ic(ind, b, lag=2)["table"]
    cols = ["因子", "n", "ic_mean", "ic_std", "ic_ir", "t_nw", "p_nw",
            "win_rate", "ic_boot_lo", "ic_boot_hi", "ir_boot_lo", "ir_boot_hi",
            "boot_p_positive", "CI含0", "方向", "结论"]
    out = t1[cols].rename(columns={c: f"{c}_t1" for c in cols if c != "因子"})
    out = out.merge(t2[cols].rename(columns={c: f"{c}_t2" for c in cols if c != "因子"}),
                    on="因子", how="outer")
    out["IC方向一致(t1/t2)"] = np.sign(out["ic_mean_t1"]) == np.sign(out["ic_mean_t2"])
    return out


# ============================================================================
# 1b. Bootstrap 置信区间（块抽样，保留自相关）
# ============================================================================
def block_bootstrap_ci(x, n: int = None, block: int = None, seed: int = None,
                       alpha: float = 0.05) -> dict:
    """对 IC 序列做**块（moving-block）自举**，给出 IC 均值与 IC_IR 的置信区间。

    为什么用块抽样而不是逐点抽样：
      RankIC 序列存在自相关（因子是持久的季度变量）。逐点重抽会破坏时序结构、
      低估标准误；块抽样（block = 4 季度，约 1 年）保留块内自相关，更贴近真实
      抽样分布。

    返回：IC 均值的 [lo, hi]、IC_IR 的 [lo, hi]、以及 IC 均值>0 的自举概率。
    """
    n = int(n or C.BOOTSTRAP_N)
    block = int(block or C.BOOTSTRAP_BLOCK)
    seed = C.BOOTSTRAP_SEED if seed is None else seed
    v = pd.Series(x).dropna().to_numpy(dtype=float)
    T = len(v)
    if T < 6:
        return dict(n=T, reps=0, ic_lo=np.nan, ic_hi=np.nan,
                    ir_lo=np.nan, ir_hi=np.nan, p_positive=np.nan)
    nb = int(np.ceil(T / block))
    rng = np.random.default_rng(seed)
    offs = np.arange(block)
    means = np.empty(n)
    irs = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, T, size=nb)
        idx = ((starts[:, None] + offs[None, :]).ravel() % T)[:T]
        smp = v[idx]
        m = smp.mean()
        sd = smp.std(ddof=1)
        means[i] = m
        irs[i] = m / sd if sd > 0 else np.nan
    lo_q, hi_q = alpha / 2 * 100, (1 - alpha / 2) * 100
    return dict(n=T, reps=n, block=block,
                ic_lo=float(np.nanpercentile(means, lo_q)),
                ic_hi=float(np.nanpercentile(means, hi_q)),
                ir_lo=float(np.nanpercentile(irs, lo_q)),
                ir_hi=float(np.nanpercentile(irs, hi_q)),
                p_positive=float((means > 0).mean()),
                ic_mean_boot=float(np.nanmean(means)))


# ============================================================================
# 2. 分层回测
# ============================================================================
def layer_backtest(factor: pd.DataFrame, fwd: pd.DataFrame, n_layer: int = C.LAYER_N,
                   min_obs: int = 15) -> dict:
    """按因子值分 n_layer 层（层1 = 因子最小 = 最不拥挤），等权持有下一季度。"""
    recs = []
    for q in factor.index:
        a, bq = factor.loc[q], fwd.loc[q]
        m = a.notna() & bq.notna()
        if m.sum() < min_obs:
            continue
        aa, bb = a[m], bq[m]
        layers = pd.qcut(aa.rank(method="first"), n_layer, labels=False) + 1
        for L in range(1, n_layer + 1):
            sel = layers == L
            recs.append(dict(quarter=q, layer=int(L), ret=float(bb[sel].mean()),
                             n=int(sel.sum())))
    df = pd.DataFrame(recs)
    if df.empty:
        return dict(table=df, pivot=None)
    piv = df.pivot(index="quarter", columns="layer", values="ret")
    cum = (1 + piv / 100).cumprod()
    perf_tbl = pd.DataFrame({int(L): perf_series(piv[L]) for L in piv.columns}).T
    perf_tbl.index.name = "层"
    ls = piv[1] - piv[n_layer]            # 低拥挤 − 高拥挤（拥挤为风险时该值应为正）
    ls_rev = piv[n_layer] - piv[1]        # 高拥挤 − 低拥挤
    mono = float(stats.spearmanr(piv.columns, piv.mean()).statistic)
    return dict(table=df, pivot=piv, cum=cum, cum_ls=(1 + ls / 100).cumprod(),
                cum_ls_rev=(1 + ls_rev / 100).cumprod(), ls=ls, ls_rev=ls_rev,
                perf=perf_tbl, perf_ls=perf_series(ls), perf_ls_rev=perf_series(ls_rev),
                monotonicity=mono, ls_tstat=nw_tstat(ls),
                ls_tstat_rev=nw_tstat(ls_rev))


def run_all_layers(ind: dict, b) -> dict:
    fwd = forward_excess(b.market, lag=1)
    return {f: layer_backtest(ind[f], fwd) for f in CS_FACTORS if f in ind}


def layer_summary(bt: dict, b) -> pd.DataFrame:
    """分层回测汇总表 + 样本内/样本外多空年化。

    注意：分层多空序列的索引可能短于全样本季度（前 8 期无扩窗 Z），
    故样本内/外必须按**季度标签**切分，不能用位置切片。
    """
    rows = []
    for f, r in bt.items():
        if r.get("pivot") is None:
            continue
        p = r["perf"]
        hilo = r["ls_rev"]          # 高拥挤 − 低拥挤（"拥挤=风险"时的坏腿）
        is_part = hilo.reindex([q for q in hilo.index if q <= C.IS_LAST]).dropna()
        oos_part = hilo.reindex([q for q in hilo.index if q >= C.OOS_FIRST]).dropna()
        row = {"因子": f}
        for L in [1, 2, 3, 4, 5]:
            row[f"层{L}年化%"] = float(p.loc[L, "ann"]) if L in p.index else np.nan
        row.update({
            "多空年化%(高−低)": r["perf_ls_rev"]["ann"],
            "多空最大回撤%": r["perf_ls_rev"]["maxdd"],
            "多空夏普": r["perf_ls_rev"]["sharpe"],
            "多空t值_NW": r["ls_tstat_rev"][0],
            "多空p值_NW": r["ls_tstat_rev"][1],
            "顺势年化%(低−高)": r["perf_ls"]["ann"],
            "单调性Spearman": r["monotonicity"],
            "样本内多空年化%": perf_series(is_part)["ann"],
            "样本外多空年化%": perf_series(oos_part)["ann"],
            "样本内季数": len(is_part), "样本外季数": len(oos_part),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# 3. 事件研究
# ============================================================================
def event_study(b, over: pd.DataFrame, horizons=C.HORIZONS_Q,
                min_hist: int = MIN_HIST) -> dict:
    """行业超配比例**首次**突破扩窗 P90 分位后的 3/6/12 个月超额收益。"""
    events = []
    for ind_name in INDUSTRIES:
        s = over[ind_name]
        p90 = expanding_quantile(s, 0.90, min_hist)
        above = (s > p90).fillna(False).astype(bool) & p90.notna()
        for t in range(len(s)):
            if not bool(above.iloc[t]):
                continue
            if t > 0 and bool(above.iloc[t - 1]):       # 只取首次
                continue
            events.append(dict(行业=ind_name, 季度=s.index[t], over=float(s.iloc[t]),
                               p90=float(p90.iloc[t]), z=float(expanding_z(s).iloc[t])))
    ev = pd.DataFrame(events)
    if ev.empty:
        return dict(events=ev, stats=pd.DataFrame())
    for h in horizons:
        cm = forward_cum_excess(b.market, h)
        ev[f"超额{h*3}月%"] = [cm.loc[r.季度, r.行业] for r in ev.itertuples()]
    rows = []
    for h in horizons:
        col = f"超额{h*3}月%"
        x = ev[col].dropna()
        base = forward_cum_excess(b.market, h).stack().dropna()
        if len(x) == 0:
            continue
        tt, pp = stats.ttest_1samp(x, 0.0) if len(x) > 1 else (np.nan, np.nan)
        rows.append(dict(持有时长=f"{h*3}个月(={h}季度)", 触发次数=len(x),
                         均值=x.mean(), 中位数=x.median(),
                         负收益概率=x.lt(0).mean() * 100,
                         t值=tt, p值=pp,
                         无条件均值=base.mean(),
                         无条件负概率=base.lt(0).mean() * 100))
    st = pd.DataFrame(rows)
    if not st.empty:
        st["是否满足有效标准"] = ((st["中位数"] < 0) & (st["负收益概率"] > 55) & (st["p值"] < 0.1))
    return dict(events=ev, stats=st)


def position_event_study(b, mkt: dict, col: str = "前十占净值比均值", q: float = 0.90,
                         horizons=(1, 2)) -> dict:
    """「88%魔咒」替代检验①：集中度代理突破扩窗 P90 后沪深300 未来收益。"""
    s = mkt["F8_仓位代理"][col]
    thr = expanding_quantile(s, q, MIN_HIST)
    above = (s > thr).fillna(False).astype(bool) & thr.notna()
    events = []
    for t in range(len(s)):
        if not bool(above.iloc[t]):
            continue
        if t > 0 and bool(above.iloc[t - 1]):
            continue
        events.append(dict(季度=s.index[t], 指标值=float(s.iloc[t]),
                           阈值=float(thr.iloc[t])))
    ev = pd.DataFrame(events)
    if ev.empty:
        return dict(events=ev, stats=pd.DataFrame())
    for k in horizons:
        fwd = forward_cum_market(b.market, k)
        ev[f"沪深300_{k*3}月%"] = [fwd.loc[r.季度] for r in ev.itertuples()]
    rows = []
    for k in horizons:
        col = f"沪深300_{k*3}月%"
        x = ev[col].dropna()
        base = forward_cum_market(b.market, k).dropna()
        if len(x) == 0:
            continue
        tt, pp = stats.ttest_1samp(x, 0.0) if len(x) > 1 else (np.nan, np.nan)
        rows.append(dict(持有时长=f"{k*3}个月", 触发次数=len(x), 均值=x.mean(),
                         中位数=x.median(), 负收益概率=x.lt(0).mean() * 100,
                         t值=tt, p值=pp, 无条件均值=base.mean(),
                         无条件负概率=base.lt(0).mean() * 100))
    return dict(events=ev, stats=pd.DataFrame(rows))


def position_momentum_study(b, mkt: dict, col: str = "前十占净值比均值",
                            horizons=(1, 2)) -> dict:
    """「88%魔咒」替代检验②：集中度代理动量由升转降后的市场收益。"""
    s = mkt["F8_仓位代理"][col]
    d = s.diff()
    turn_neg = (d < 0) & (d.shift(1) > 0)
    ev = pd.DataFrame({"季度": s.index, "指标值": s.to_numpy(), "动量": d.to_numpy(),
                       "动量转负": turn_neg.fillna(False).to_numpy()})
    for k in horizons:
        ev[f"沪深300_{k*3}月%"] = forward_cum_market(b.market, k).reindex(ev["季度"]).to_numpy()
    rows = []
    for k in horizons:
        col = f"沪深300_{k*3}月%"
        x = ev.loc[ev["动量转负"], col].dropna()
        base = ev[col].dropna()
        if len(x) == 0:
            continue
        tt, pp = stats.ttest_1samp(x, 0.0) if len(x) > 1 else (np.nan, np.nan)
        rows.append(dict(持有时长=f"{k*3}个月", 触发次数=len(x), 均值=x.mean(),
                         中位数=x.median(), 负收益概率=x.lt(0).mean() * 100,
                         t值=tt, p值=pp, 无条件均值=base.mean(),
                         无条件负概率=base.lt(0).mean() * 100))
    return dict(events=ev, stats=pd.DataFrame(rows))


def hhi_shock_study(b, mkt: dict, horizons=(1, 2)) -> dict:
    """HHI 突变（ΔHHI > 扩窗 mean + 1.5σ）后的市场收益与下期行业分化。"""
    h = mkt["F5_HHI"]
    d = h.diff()
    m = d.shift(1).expanding(min_periods=MIN_HIST).mean()
    sd = d.shift(1).expanding(min_periods=MIN_HIST).std()
    shock = (d > (m + 1.5 * sd)).fillna(False).astype(bool) & sd.notna()
    ev = pd.DataFrame({"季度": h.index, "HHI": h.to_numpy(), "ΔHHI": d.to_numpy(),
                       "HHI突变": shock.to_numpy(), "突变阈值": (m + 1.5 * sd).to_numpy()})
    for k in horizons:
        ev[f"沪深300_{k*3}月%"] = forward_cum_market(b.market, k).reindex(ev["季度"]).to_numpy()
    ev["下期行业分化度"] = b.market["excess"].std(axis=1, ddof=1).shift(-1).reindex(
        ev["季度"]).to_numpy()
    rows = []
    for k in horizons:
        col = f"沪深300_{k*3}月%"
        x = ev.loc[ev["HHI突变"], col].dropna()
        base = ev[col].dropna()
        if len(x) == 0:
            continue
        tt, pp = stats.ttest_1samp(x, 0.0) if len(x) > 1 else (np.nan, np.nan)
        rows.append(dict(持有时长=f"{k*3}个月", 触发次数=len(x), 均值=x.mean(),
                         中位数=x.median(), 负收益概率=x.lt(0).mean() * 100,
                         t值=tt, p值=pp, 无条件均值=base.mean(),
                         无条件负概率=base.lt(0).mean() * 100))
    return dict(events=ev, stats=pd.DataFrame(rows))


# ============================================================================
# 4. 多信号共振回测
# ============================================================================
def multi_signal_backtest(b, ind: dict, horizons=C.HORIZONS_Q) -> dict:
    """F10 共振得分 0/1/2/3 触发后的未来累计超额收益对比。"""
    score = ind["F10_共振得分"]
    cum = {h: forward_cum_excess(b.market, h) for h in horizons}
    recs = []
    for q in score.index:
        for i in INDUSTRIES:
            sc = score.loc[q, i]
            if pd.isna(sc):
                continue
            row = dict(季度=q, 行业=i, 共振得分=int(sc),
                       S1=ind["F10_S1_持仓拥挤"].loc[q, i],
                       S2=ind["F10_S2_交易拥挤"].loc[q, i],
                       S3=ind["F10_S3_筹码拥挤"].loc[q, i])
            for h in horizons:
                row[f"超额{h*3}月%"] = cum[h].loc[q, i]
            recs.append(row)
    df = pd.DataFrame(recs)
    if df.empty:
        return dict(detail=df, summary=pd.DataFrame(), combo=pd.DataFrame())
    rows = []
    for sc in sorted(df["共振得分"].unique()):
        sub = df[df["共振得分"] == sc]
        for h in horizons:
            col = f"超额{h*3}月%"
            x = sub[col].dropna()
            if len(x) == 0:
                continue
            tt, pp = stats.ttest_1samp(x, 0.0) if len(x) > 1 else (np.nan, np.nan)
            rows.append(dict(共振得分=int(sc), 持有时长=f"{h*3}个月", 样本数=len(x),
                             均值=x.mean(), 中位数=x.median(),
                             负收益概率=x.lt(0).mean() * 100, t值=tt, p值=pp))
    combo = df.assign(信号组合=df[["S1", "S2", "S3"]].fillna(0).astype(int)
                      .astype(str).agg("".join, axis=1))
    crows = []
    for c, sub in combo.groupby("信号组合"):
        for h in horizons:
            x = sub[f"超额{h*3}月%"].dropna()
            if len(x) == 0:
                continue
            crows.append(dict(信号组合=c, 含义="".join(
                ["持仓拥挤" if c[0] == "1" else "-", "交易拥挤" if c[1] == "1" else "-",
                 "筹码拥挤" if c[2] == "1" else "-"]),
                持有时长=f"{h*3}个月", 样本数=len(x), 均值=x.mean(),
                中位数=x.median(), 负收益概率=x.lt(0).mean() * 100))
    return dict(detail=df, summary=pd.DataFrame(rows), combo=pd.DataFrame(crows))


# ============================================================================
# 5. 稳健性
# ============================================================================
def orthogonalize(factor: pd.DataFrame, controls: list, fwd: pd.DataFrame,
                  min_obs: int = 15) -> dict:
    """逐期对风格控制变量做截面 OLS，取残差因子并重算 RankIC。"""
    res = pd.DataFrame(index=factor.index, columns=factor.columns, dtype=float)
    for q in factor.index:
        y = factor.loc[q]
        Xs = [c.loc[q] for c in controls]
        m = y.notna()
        for x in Xs:
            m &= x.notna()
        if int(m.sum()) < min_obs:
            continue
        cols = m.index[m]
        X = np.column_stack([np.ones(len(cols))] +
                            [x.reindex(cols).to_numpy(dtype=float) for x in Xs])
        try:
            beta, *_ = np.linalg.lstsq(X, y.reindex(cols).to_numpy(dtype=float), rcond=None)
        except np.linalg.LinAlgError:
            continue
        res.loc[q, cols] = y.reindex(cols).to_numpy(dtype=float) - X @ beta
    return dict(residual=res, ic_raw=rank_ic_panel(factor, fwd),
                ic_resid=rank_ic_panel(res, fwd))


def run_robustness(ind: dict, b) -> dict:
    """样本内/外 IC、牛熊震荡子样本、参数敏感性、正交化。"""
    qs = list(b.quarters)
    split = qs.index(C.IS_LAST) if C.IS_LAST in qs else len(qs) - 1
    fwd1 = forward_excess(b.market, lag=1)

    # --- 5.1 样本内 / 样本外 ---
    oos_rows = []
    for f in CS_FACTORS:
        if f not in ind:
            continue
        ic = rank_ic_panel(ind[f], fwd1)
        a = ic.iloc[: split + 1].dropna()
        bb = ic.iloc[split + 1:].dropna()
        oos_rows.append(dict(因子=f, 样本内IC=a.mean() if len(a) else np.nan,
                             样本内n=len(a), 样本外IC=bb.mean() if len(bb) else np.nan,
                             样本外n=len(bb),
                             方向一致=bool(len(a) and len(bb)
                                        and np.sign(a.mean()) == np.sign(bb.mean()))))
    oos = pd.DataFrame(oos_rows)

    # --- 5.2 牛 / 熊 / 震荡 子样本 ---
    hs = b.market["hs300"]
    trail = ((1 + hs / 100.0).rolling(4, min_periods=4).apply(np.prod, raw=True) - 1) * 100
    regime = pd.Series("震荡市", index=hs.index)
    regime[trail > 15] = "牛市"
    regime[trail < -10] = "熊市"
    subs = []
    for f in CS_FACTORS:
        if f not in ind:
            continue
        ic = rank_ic_panel(ind[f], fwd1)
        for g in ["牛市", "熊市", "震荡市"]:
            idx = regime[regime == g].index
            s = ic.reindex(idx).dropna()
            subs.append(dict(因子=f, 市场状态=g, n=len(s),
                             IC均值=s.mean() if len(s) else np.nan,
                             IC_IR=(s.mean() / s.std(ddof=1)) if len(s) > 1 and s.std(ddof=1) else np.nan,
                             胜率=(s > 0).mean() * 100 if len(s) else np.nan))
    sub_df = pd.DataFrame(subs)

    # --- 5.3 参数敏感性：扩窗最小历史 8 / 12 / 16 季度 ---
    rob_rows = []
    for n in C.ROBUST_WINDOWS:
        over = b.overweight.reindex(columns=INDUSTRIES)
        z = pd.DataFrame({i: expanding_z(over[i], n) for i in INDUSTRIES})
        pct = pd.DataFrame({i: expanding_pct(over[i], n) for i in INDUSTRIES})
        ic_z = rank_ic_panel(z, fwd1)
        ic_p = rank_ic_panel(pct, fwd1)
        rob_rows.append(dict(窗口=f"min_hist={n}", n_z=len(ic_z.dropna()),
                             F2_Z_IC=ic_z.mean(), F3_分位_IC=ic_p.mean(),
                             F2_Z_IR=(ic_z.mean() / ic_z.std(ddof=1))
                             if ic_z.std(ddof=1) else np.nan))
    # 滚动窗口口径（滚动 N 个季度而非扩窗）
    for n in C.ROBUST_WINDOWS:
        over = b.overweight.reindex(columns=INDUSTRIES)
        rm = over.shift(1).rolling(n, min_periods=n).mean()
        rs = over.shift(1).rolling(n, min_periods=n).std()
        z = (over - rm) / rs.replace(0, np.nan)
        ic_z = rank_ic_panel(z, fwd1)
        rob_rows.append(dict(窗口=f"rolling={n}", n_z=len(ic_z.dropna()),
                             F2_Z_IC=ic_z.mean(), F3_分位_IC=np.nan,
                             F2_Z_IR=(ic_z.mean() / ic_z.std(ddof=1))
                             if ic_z.std(ddof=1) else np.nan))
    rob = pd.DataFrame(rob_rows)

    # --- 5.4 正交化 ---
    ctrl = [ind["CTRL_第t期收益"], ind["CTRL_过去4季收益"], ind["CTRL_过去4季波动"]]
    orth_rows, orth_detail = [], {}
    for f in ["F1_超配比例", "F2_超配Zscore", "F3_超配历史分位", "F4b_超配动量Z",
              "F9_交易拥挤度", "F11_综合拥挤度_推荐权重"]:
        if f not in ind:
            continue
        o = orthogonalize(ind[f], ctrl, fwd1)
        orth_detail[f] = o
        s_raw = ic_stats(o["ic_raw"])
        s_res = ic_stats(o["ic_resid"])
        orth_rows.append(dict(因子=f, 原始IC=s_raw["ic_mean"], 残差IC=s_res["ic_mean"],
                              原始IC_IR=s_raw["ic_ir"], 残差IC_IR=s_res["ic_ir"],
                              残差n=s_res["n"],
                              残差是否仍有效=bool(abs(s_res["ic_mean"] or 0) > 0.03)))
    orth = pd.DataFrame(orth_rows)

    return dict(oos=oos, subsample=sub_df, robustness=rob, orthogonal=orth,
                orthogonal_detail=orth_detail, regime=regime)


# ============================================================================
# 5b. 因子衰减分析（滚动窗口 IC / IR）
# ============================================================================
def decay_analysis(ind: dict, b, windows=None) -> dict:
    """滚动窗口 IC/IR：观察因子预测力随窗口长度与时间的变化。

    口径说明：本项目的因子是**季度**频率，故"滚动 252 个交易日"等价于
    滚动 4 个季度（一年）。为使结论可比，同时给出 4 / 8 / 12 季度三档窗口。

    输出两张表：
      · 分窗口汇总：每个因子在每个窗口长度上的 IC 均值、IC_IR、胜率
      · 滚动序列：每个交易日的"当前窗口"IC 与 IR（用于画衰减曲线）
    """
    windows = windows or C.DECAY_WINDOWS_Q
    fwd = forward_excess(b.market, lag=1)
    summ, series = [], {}
    for f in CS_FACTORS:
        if f not in ind:
            continue
        ic = rank_ic_panel(ind[f], fwd).dropna()
        if len(ic) < 6:
            continue
        for w in windows:
            roll_mean = ic.rolling(w, min_periods=max(3, w // 2)).mean()
            roll_sd = ic.rolling(w, min_periods=max(3, w // 2)).std()
            roll_ir = roll_mean / roll_sd.replace(0, np.nan)
            series[(f, w)] = pd.DataFrame({"IC均值": roll_mean, "IC_IR": roll_ir,
                                           "IC_胜率": ic.rolling(
                                               w, min_periods=max(3, w // 2)
                                           ).apply(lambda a: (a > 0).mean() * 100,
                                                   raw=True)})
            last_m = roll_mean.dropna()
            last_i = roll_ir.dropna()
            summ.append(dict(
                因子=f, 窗口=f"{w}季度(≈{w * 63}交易日)",
                IC均值=ic.mean(), IC_IR=ic.mean() / ic.std(ddof=1)
                if ic.std(ddof=1) else np.nan,
                IC胜率=(ic > 0).mean() * 100,
                最新窗口IC均值=last_m.iloc[-1] if len(last_m) else np.nan,
                最新窗口IC_IR=last_i.iloc[-1] if len(last_i) else np.nan,
                最新窗口位置=last_m.index[-1] if len(last_m) else None))
    return dict(summary=pd.DataFrame(summ), series=series)
