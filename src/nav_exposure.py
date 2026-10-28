# -*- coding: utf-8 -*-
"""日度行业暴露（4.4）：用基金日度净值对 31 个申万一级行业指数做滚动约束回归。

为什么需要它
------------
季报持仓有 15 个工作日 + 披露滞后的固有延迟（约 56 天），
等看到持仓时季度已经过了 62%。日度净值是唯一能把这个滞后压到 1–2 周的信息源：
净值每天的涨跌，本质上是它所持行业当日收益的加权和，反解权重就能得到「今天」的暴露。

模型
----
    r_fund,t = Σ_k w_k · r_ind,k,t + ε_t      w_k ≥ 0, Σ w_k ≤ 1

· 非负约束：基金不能做空行业（公募股票多头）；
· Σ ≤ 1：剩余部分视为现金/债券/港股等未建模资产，不强行归一；
· 窗口 60 个交易日，滚动估计；
· 用 `scipy.optimize.lsq_linear`（带边界的最小二乘），比手写 NNLS 稳定。

诚实边界
--------
· 31 个行业日收益高度共线（相关性常 >0.8），回归系数不稳定是常态，
  所以必须报告 R² 分布；R² < 0.5 的基金超过三成时，本方法整体回退季频。
· 基金季报只披露前十大，季报口径的「真值」本身有偏，校准只能做方向性检验。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C  # noqa: E402

CACHE = os.path.join(C.DATA, "cache")
IDX_PATH = os.path.join(CACHE, "sw_index_daily_ret.csv")
NAV_DIR = os.path.join(CACHE, "fund_nav")


def _sw_codes() -> dict[str, str]:
    """行业名 → 申万指数代码（801xxx）。"""
    import akshare as ak
    sw = ak.sw_index_first_info()
    out = {}
    for _, r in sw.iterrows():
        code = str(r["行业代码"]).split(".")[0]
        name = str(r["行业名称"])
        if name in C.INDUSTRIES:
            out[name] = code
    return out


def build_index_returns(force: bool = False) -> pd.DataFrame:
    """31 个申万一级指数的日收益矩阵（index=日期，columns=行业）。"""
    if os.path.exists(IDX_PATH) and not force:
        d = pd.read_csv(IDX_PATH, index_col=0, parse_dates=True)
        if d.shape[1] >= 20:
            return d
    import akshare as ak
    os.makedirs(CACHE, exist_ok=True)
    mp = _sw_codes()
    cols = {}
    for name, code in mp.items():
        try:
            x = ak.index_hist_sw(symbol=code)
            x["日期"] = pd.to_datetime(x["日期"])
            s = x.set_index("日期")["收盘"].astype(float).sort_index()
            cols[name] = s.pct_change()
        except Exception as e:
            print(f"  指数失败 {name}: {type(e).__name__}")
    d = pd.DataFrame(cols).dropna(how="all")
    d.to_csv(IDX_PATH, encoding="utf-8")
    print(f"申万日收益：{d.shape[0]} 交易日 × {d.shape[1]} 行业 → {IDX_PATH}")
    return d


def fetch_nav(code: str, force: bool = False) -> pd.Series:
    """单只基金日收益（用单位净值算，避免「日增长率」口径差异）。"""
    os.makedirs(NAV_DIR, exist_ok=True)
    p = os.path.join(NAV_DIR, f"{code}.csv")
    if os.path.exists(p) and not force:
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        return d.iloc[:, 0]
    import akshare as ak
    x = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
    x["净值日期"] = pd.to_datetime(x["净值日期"])
    s = x.set_index("净值日期")["单位净值"].astype(float).sort_index()
    r = s.pct_change().dropna()
    r.to_csv(p, encoding="utf-8")
    return r


def rolling_exposure(fund_ret: pd.Series, ind_ret: pd.DataFrame,
                     window: int = 60, step: int = 5) -> pd.DataFrame:
    """滚动约束回归，返回每个窗口末日的行业权重估计。"""
    from scipy.optimize import lsq_linear
    idx = fund_ret.index.intersection(ind_ret.index)
    f = fund_ret.reindex(idx).to_numpy(float)
    X = ind_ret.reindex(idx).to_numpy(float)
    cols = list(ind_ret.columns)
    out = []
    for end in range(window, len(idx) + 1, step):
        y = f[end - window:end]
        Xw = X[end - window:end]
        if np.isnan(y).any() or np.isnan(Xw).any():
            continue
        # 目标：min ||Xw·w − y||，w∈[0,1]，再加一行「惩罚和>1」的软约束
        A = np.vstack([Xw, np.ones((1, Xw.shape[1])) * 0.1])
        b = np.concatenate([y, [0.1]])
        try:
            res = lsq_linear(A, b, bounds=(0, 1), max_iter=200)
        except Exception:
            continue
        w = res.x.copy()
        # 软约束不保证 Σw ≤ 1：超过 1 时等比缩放回来，避免出现「120% 仓位」的估计
        if w.sum() > 1:
            w = w / w.sum()
        pred = Xw @ w
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        row = {c: float(v) for c, v in zip(cols, w)}
        row["_r2"] = r2
        row["_sum"] = float(w.sum())
        row["_date"] = idx[end - 1]
        out.append(row)
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).set_index("_date")


def exposure_quality(codes: list[str], window: int = 60) -> pd.DataFrame:
    """批量估计并返回每只基金最新窗口的 R²、权重和、Top 行业。"""
    ind = build_index_returns()
    rows = []
    for c in codes:
        try:
            nav = fetch_nav(c)
        except Exception:
            continue
        e = rolling_exposure(nav, ind, window=window)
        if e.empty:
            continue
        last = e.iloc[-1]
        w = last.drop(["_r2", "_sum"])
        top = w.sort_values(ascending=False).head(3)
        rows.append(dict(
            code=c, n_days=len(nav), r2=round(float(last["_r2"]), 3),
            w_sum=round(float(last["_sum"]), 3),
            top1=top.index[0], top1_w=round(float(top.iloc[0]) * 100, 1),
            top2=top.index[1] if len(top) > 1 else None,
            top2_w=round(float(top.iloc[1]) * 100, 1) if len(top) > 1 else None,
            top3=top.index[2] if len(top) > 2 else None,
            top3_w=round(float(top.iloc[2]) * 100, 1) if len(top) > 2 else None,
        ))
    return pd.DataFrame(rows)


def _quarter_end(q: str) -> pd.Timestamp:
    y, k = int(q[:4]), int(q[-1])
    return pd.Timestamp(year=y, month=k * 3, day=1) + pd.offsets.MonthEnd(0)


def drift_since_quarter(q: str, codes: list[str] | None = None,
                        min_r2: float = 0.5, window: int = 60) -> pd.DataFrame:
    """季末 → 今天，基金行业暴露的漂移（同一模型内的前后对比）。

    为什么不用「日度权重 − 季报占比」
    --------------------------------
    两者稀疏度不同：回归会把权重集中在少数行业、把其余压到 0，
    而季报占比天然分散在 4–10 个行业。直接相减会得到 −10 到 −16pp 的
    系统性负值，那不是调仓，是口径差。

    正确做法是在**同一个模型内**做前后对比：
    · t0 = 季末最后一个交易日的 60 日窗口回归权重；
    · t1 = 最新交易日的 60 日窗口回归权重；
    · 漂移 = t1 − t0，两边同尺度、同稀疏度，差值才是真实的仓位移动。

    用途：季报披露滞后约 56 天，等看到持仓时季度已过 62%；
    这个漂移把「季末之后又挪了多少」补上，把时效性从 56 天压到 1–2 周。
    """
    import ak_data as akd
    ind = build_index_returns()
    if codes is None:
        con = akd._con()
        codes = [r[0] for r in con.execute(
            "SELECT code FROM coverage WHERE quarter=? ORDER BY coverage DESC LIMIT 60", (q,))]
        con.close()
    qe = _quarter_end(q)
    rows, r2s = {}, []
    for c in codes:
        try:
            nav = fetch_nav(c)
            e = rolling_exposure(nav, ind, window=window)
        except Exception:
            continue
        if len(e) < 2:
            continue
        before = e[e.index <= qe]
        if before.empty:
            continue
        t0, t1 = before.iloc[-1], e.iloc[-1]
        if float(t0["_r2"]) < min_r2 or float(t1["_r2"]) < min_r2:
            continue
        w0 = t0.drop(["_r2", "_sum"]).astype(float)
        w1 = t1.drop(["_r2", "_sum"]).astype(float)
        if w0.sum() <= 0 or w1.sum() <= 0:
            continue
        rows[c] = (w1 / w1.sum() * 100 - w0 / w0.sum() * 100).reindex(C.INDUSTRIES)
        r2s.append(float(t1["_r2"]))
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows).T
    out = pd.DataFrame({
        "行业平均漂移pp": d.mean().round(2),
        "平均绝对漂移pp": d.abs().mean().round(2),
        "同向基金占比%": ((np.sign(d).eq(np.sign(d.mean()))).mean() * 100).round(1),
        "加仓基金数": (d > 0.5).sum(),
        "减仓基金数": (d < -0.5).sum(),
    }).reset_index().rename(columns={"index": "行业"})
    out = out.sort_values("行业平均漂移pp", ascending=False)
    out.attrs.update(q=q, q_end=str(qe.date()), n_funds=len(rows),
                     r2_median=float(np.median(r2s)) if r2s else np.nan)
    return out


if __name__ == "__main__":
    ind = build_index_returns()
    print("行业数", ind.shape[1], "区间", ind.index.min().date(), "→", ind.index.max().date())
    codes = sys.argv[1:] or ["005827", "003095", "110029"]
    q = exposure_quality(codes)
    print(q.to_string(index=False))
    if not q.empty:
        print(f"\nR² 中位 {q['r2'].median():.3f}；R²<0.5 占比 "
              f"{(q['r2'] < 0.5).mean() * 100:.0f}%")
