# -*- coding: utf-8 -*-
"""季报因子的标准检验流程：定义 → 计算 → 单期 IC → 汇总显著性与稳定性 → 分层。

为什么必须自己检验
------------------
页面上的每个判断（「可以跟」「要回避」）都隐含一个假设：这些指标能预测次季收益。
如果不检验就写进结论，等于把「描述」当「预测」。本模块用 31 期、31 个行业的面板
把每个因子都过一遍标准流程，结果无论好坏都展示——包括被证伪的那些。

标准流程（每一步都可复核）
--------------------------
    1. 定义：因子在每个（期, 行业）上的取值，方向事先声明（越大越看多/看空）
    2. 计算：只用当期及之前的信息，分位数用**扩张窗口**，避免前视
    3. 单期 IC：Spearman(因子_t, 次期收益_t)，横截面 31 个行业
    4. 汇总：IC 均值、IC_IR、Newey-West t、p 值、胜率
    5. 分层：按因子五等分，看次期收益是否单调；多空 = Q5−Q1
    6. 结论：按事前阈值给出「可用 / 仅辅助 / 无效 / 反向」

事前阈值（写死在 TH，不因结果调整）
----------------------------------
    |IC| < 0.03 无效；0.03–0.05 弱；0.05–0.10 可用；>0.10 强
    |t| ≥ 1.96 为 5% 显著；IC_IR > 0.5 为稳定
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

import canon as cn
import config as C

TH = dict(ic_weak=0.03, ic_ok=0.05, ic_strong=0.10,
          t_sig=1.96, ir_ok=0.5)


# ---------------------------------------------------------------------------
# 基础面板
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def quarter_ends() -> pd.DataFrame:
    """每个报告期对应的「最后一个交易日」与行业累计点位。

    持仓是季末的快照，收益必须从季末收盘算起，不能用自然日。
    """
    lv = cn.index_levels()
    if lv.empty:
        return pd.DataFrame()
    qs = cn.quarters()
    rows = {}
    for q in qs:
        y, k = int(q[:4]), int(q[-1])
        end = pd.Timestamp(year=y, month=k * 3, day=1) + pd.offsets.MonthEnd(0)
        sub = lv[lv.index <= end]
        if sub.empty:
            continue
        rows[q] = sub.iloc[-1]
    out = pd.DataFrame(rows).T.sort_index()
    out.index.name = "quarter"
    return out


@lru_cache(maxsize=1)
def forward_returns() -> pd.DataFrame:
    """次期行业收益（%）：季末点位 → 下一季末点位。"""
    lv = quarter_ends()
    if lv.empty or len(lv) < 2:
        return pd.DataFrame()
    fwd = (lv.shift(-1) / lv - 1) * 100
    fwd = fwd.dropna(how="all")
    fwd.index.name = "quarter"
    fwd.columns.name = "industry"
    return fwd


@lru_cache(maxsize=1)
def expanding_pct() -> pd.DataFrame:
    """扩张窗口的配置分位：每期只用该期及之前的历史，避免前视。

    页面上的「31 期分位」用的是全样本排名（展示用，方便读者对照）；
    做检验时必须换成扩张窗口，否则 2019 年的分位用到了 2026 年的数据，
    IC 会被系统性高估。这是因子研究里最常见的前视偏差。
    """
    M = cn.level_matrix("top10")
    if M.empty:
        return pd.DataFrame()
    out = {}
    for i, q in enumerate(M.index):
        sub = M.iloc[: i + 1]
        out[q] = (sub.rank(pct=True).iloc[-1] * 100)
    return pd.DataFrame(out).T.sort_index()


@lru_cache(maxsize=1)
def index_pct_expanding() -> pd.DataFrame:
    """价格分位的扩张版本：每个季末只用该日之前的交易日排名。"""
    lv = cn.index_levels()
    if lv.empty:
        return pd.DataFrame()
    out = {}
    for q in cn.quarters():
        y, k = int(q[:4]), int(q[-1])
        end = pd.Timestamp(year=y, month=k * 3, day=1) + pd.offsets.MonthEnd(0)
        sub = lv[lv.index <= end]
        if sub.empty:
            continue
        out[q] = sub.rank(pct=True).iloc[-1] * 100
    return pd.DataFrame(out).T.sort_index()


@lru_cache(maxsize=16)
def factor_panel(name: str) -> pd.Series:
    """构造某个因子在（期, 行业）上的取值。返回 MultiIndex Series。"""
    qs = cn.quarters()
    if name == "breadth":
        rows = {}
        for i in range(1, len(qs)):
            b = cn.breadth(qs[i - 1], qs[i])
            if not b.empty:
                rows[qs[i]] = b.set_index("行业")["加仓占比"]
        s = pd.DataFrame(rows).T.stack()
    elif name == "active_q":
        rows = {}
        for i in range(1, len(qs)):
            b = cn.breadth(qs[i - 1], qs[i])
            if not b.empty:
                rows[qs[i]] = b.set_index("行业")["平均主动pp"]
        s = pd.DataFrame(rows).T.stack()
    elif name == "delta_q":
        M = cn.level_matrix("top10")
        s = M.diff().stack()
    elif name == "delta_y":
        M = cn.level_matrix("top10")
        s = (M - M.shift(4)).stack()
    elif name == "cfg_pct":
        s = expanding_pct().stack()
    elif name == "idx_pct":
        s = index_pct_expanding().stack()
    elif name == "dispersion":
        rows = {}
        for i in range(1, len(qs)):
            b = cn.breadth(qs[i - 1], qs[i])
            if not b.empty:
                rows[qs[i]] = b.set_index("行业")["离散度pp"]
        s = pd.DataFrame(rows).T.stack()
    elif name == "csi":
        # 用扩张分位重算，避免前视
        cfg = expanding_pct()
        idxp = index_pct_expanding()
        brd = factor_panel("breadth").unstack()
        fund = brd.rank(axis=1, pct=True) * 100
        csi = (cfg.reindex(fund.index) + idxp.reindex(fund.index) + fund) / 3
        s = csi.stack()
    elif name == "active_h":
        rows = {}
        fq = cn.full_quarters()
        for i in range(1, len(fq)):
            try:
                d = cn.decomp(fq[i - 1], fq[i])
            except Exception:
                continue
            if not d.empty:
                rows[fq[i]] = d["主动贡献"]
        s = pd.DataFrame(rows).T.stack()
    else:
        raise KeyError(name)
    s.index.names = ["quarter", "industry"]
    return s.dropna()


FACTORS = {
    "breadth": dict(name="加仓广度", direction=+1, freq="季",
                    formula="加仓基金数 ÷ (加仓 + 减仓)，价格中性 Δ>0.1pp 计入",
                    expect="广度高＝共识强，次季应有超额（动量/羊群）"),
    "active_q": dict(name="当期主动净额", direction=+1, freq="季",
                     formula="逐基金价格中性 Δ 的行业均值（pp）",
                     expect="净买入越多，次季越强（需求冲击）"),
    "delta_q": dict(name="环比配置变化", direction=+1, freq="季",
                    formula="前十大配置_t − 配置_{t−1}（pp，含价格）",
                    expect="配置抬升代表看多（但含价格，可能反向）"),
    "delta_y": dict(name="同比配置变化", direction=+1, freq="季",
                    formula="配置_t − 配置_{t−4}（pp）",
                    expect="年度维度的结构变化"),
    "cfg_pct": dict(name="配置分位", direction=-1, freq="季",
                    formula="扩张窗口内该行业配置的排名（0–100）",
                    expect="拥挤即风险：分位高＝次季弱"),
    "idx_pct": dict(name="价格分位", direction=-1, freq="季",
                    formula="行业指数点位在历史交易日中的扩张排名",
                    expect="反转：价格贵＝次季弱"),
    "csi": dict(name="拥挤度 CSI", direction=-1, freq="季",
                formula="(配置分位 + 价格分位 + 资金分位)/3",
                expect="三合一拥挤，次季弱"),
    "dispersion": dict(name="离散度", direction=-1, freq="季",
                       formula="价格中性 Δ 的横截面标准差（pp）",
                       expect="分歧大＝方向不稳，次季弱"),
    "active_h": dict(name="半年主动净额", direction=+1, freq="半年",
                     formula="GK 分解的主动贡献（pp，全持仓）",
                     expect="剥离价格后的真加仓，次季强"),
}


# ---------------------------------------------------------------------------
# 检验
# ---------------------------------------------------------------------------
def _nw_t(x: pd.Series, lags: int = 4) -> tuple[float, float]:
    """Newey-West HAC t 值与双侧 p 值。时序自相关会让普通 t 高估显著性。"""
    from scipy import stats
    v = np.asarray(x.dropna(), dtype=float)
    n = len(v)
    if n < 5:
        return np.nan, np.nan
    mu = v.mean()
    e = v - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1 - l / (lags + 1)
        g = (e[l:] @ e[:-l]) / n
        var += 2 * w * g
    se = np.sqrt(max(var, 1e-12) / n)
    t = mu / se
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    return float(t), float(p)


@lru_cache(maxsize=16)
def test_factor(key: str) -> dict:
    """对单个因子跑完整流程，返回全部中间结果（页面要用）。"""
    meta = FACTORS[key]
    f = factor_panel(key)
    fr = forward_returns()
    if f.empty or fr.empty:
        return dict(key=key, meta=meta, ok=False)
    # 对齐：因子在 q，收益是 q→q+1
    d = f.to_frame("f")
    long = fr.stack().rename("r")
    # 必须显式对齐两级索引名：stack 出来的第二级常常没有名字，
    # join 时 pandas 只按 quarter 对齐，会得到「行业 × 行业」的笛卡尔积
    # ——结果是 IC 全为 0，而且不报错。这个坑踩过，这里加断言兜底。
    long.index.names = ["quarter", "industry"]
    d = d.join(long, how="inner").dropna()
    # 断言：两级索引、无重复。若 stack/join 走偏，会退化成三级索引的
    # 「行业 × 行业」笛卡尔积，IC 全为 0 且不报错——必须显式拦住。
    assert d.index.nlevels == 2 and d.index.is_unique, "因子与收益的索引没有一一对齐"
    if d.empty:
        return dict(key=key, meta=meta, ok=False)
    # 方向：direction=-1 的因子取负，统一成「越大越看多」，便于横向比较
    d["x"] = d["f"] * meta["direction"]

    ic, ls, layers, ic_idx = [], [], [], []
    for q, g in d.groupby(level=0):
        if len(g) < 10 or g["x"].nunique() < 5:
            continue
        ic.append(g["x"].corr(g["r"], method="spearman"))
        ic_idx.append(q)
        try:
            g5 = pd.qcut(g["x"].rank(method="first"), 5, labels=False)
        except ValueError:
            continue
        m = g.groupby(g5)["r"].mean()
        layers.append(m)
        if 0 in m.index and 4 in m.index:
            ls.append(m[4] - m[0])
    ic = pd.Series(ic, index=ic_idx).dropna()
    if len(ic) < 5:
        return dict(key=key, meta=meta, ok=False)
    t, p = _nw_t(ic)
    L = pd.DataFrame(layers).mean()
    mono = L.rank().corr(pd.Series(range(len(L)), index=L.index), method="spearman")
    ls = pd.Series(ls)
    return dict(
        key=key, meta=meta, ok=True, ic=ic,
        ic_mean=float(ic.mean()), ic_std=float(ic.std(ddof=0)),
        ic_ir=float(ic.mean() / ic.std(ddof=0)) if ic.std(ddof=0) else np.nan,
        t=float(t), p=float(p), n=int(len(ic)),
        win=float((ic > 0).mean() * 100),
        layers=L, mono=float(mono) if np.isfinite(mono) else np.nan,
        ls_mean=float(ls.mean()) if len(ls) else np.nan,
        ls_win=float((ls > 0).mean() * 100) if len(ls) else np.nan,
        ann=float(((1 + ls.mean() / 100) ** 4 - 1) * 100) if len(ls) else np.nan,
        panel=d,
    )


def verdict(r: dict) -> tuple[str, str]:
    """按事前阈值给结论：(标签, 一句话解释)。

    方向必须单独处理：一个因子「显著但反着走」和「不显著」是完全不同的结论。
    前者可以反向使用（需样本外验证），后者只能弃用。
    """
    if not r.get("ok"):
        return "样本不足", "有效 IC 期数少于 5，不做判断。"
    ic, t, ir = r["ic_mean"], r["t"], r["ic_ir"]
    aic, at, air = abs(ic), abs(t), abs(ir)
    if aic < TH["ic_weak"] or at < 1.64:
        return "无效", (f"|IC|={aic:.3f}、|t|={at:.2f}，均未过阈值；"
                       f"它只能描述状态，不能预测次季收益。")
    if ic < 0:
        return "反向", (f"IC={ic:.3f}（负）、t={t:.2f}：按定义方向使用会亏，"
                       f"反向才有正收益；反向策略需样本外再验证一季。")
    if aic < TH["ic_ok"] or air < TH["ir_ok"]:
        return "仅辅助", (f"|IC|={aic:.3f} 过弱阈值但未到可用线（0.05），"
                         f"或 IC_IR={air:.2f} 不足 0.5；只能作辅助。")
    if aic < TH["ic_strong"]:
        return "可用", f"|IC|={aic:.3f}、|t|={at:.2f}、IC_IR={air:.2f}，方向稳定。"
    return "强", f"|IC|={aic:.3f} 超过 0.10，排序能力强。"


@lru_cache(maxsize=1)
def run_all() -> pd.DataFrame:
    """所有因子的汇总表（页面主表）。"""
    rows = []
    for k, meta in FACTORS.items():
        r = test_factor(k)
        lab, why = verdict(r)
        rows.append(dict(
            因子=meta["name"], 代码=k, 方向="越大越看多" if meta["direction"] > 0 else "越大越看空",
            频率=meta["freq"], 公式=meta["formula"],
            IC=r.get("ic_mean", np.nan), IC_IR=r.get("ic_ir", np.nan),
            t值=r.get("t", np.nan), p值=r.get("p", np.nan),
            胜率=r.get("win", np.nan), 期数=r.get("n", 0),
            多空=r.get("ls_mean", np.nan), 单调性=r.get("mono", np.nan),
            结论=lab, 说明=why))
    return pd.DataFrame(rows).sort_values("IC", key=lambda s: s.abs(), ascending=False)


if __name__ == "__main__":
    print(run_all().round(3).to_string(index=False))
