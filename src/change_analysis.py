# -*- coding: utf-8 -*-
"""环比变化分析（Change-driven analytics）——全站唯一的环比口径来源。

设计原则
--------
1. **主判断只用 Δ超配Z（长历史口径）**：ΔF2e = F2e_t − F2e_{t−1}。
   实测（2024Q2–2026Q2 因子 → 2024Q3–2026Q3 收益，n=9）：
   Δ配置pp 的 RankIC = +0.008（t=0.14，p=0.89）——无信息；
   Δ超配Z 的 RankIC = +0.124（t=3.53，p=0.008）——有信息。
   因此页面任何"谁在加仓/减仓"的排序与判定都走 ΔZ；
   Δ配置pp 只作为悬浮提示里的辅助信息。

2. **水平用长历史口径**：F2e / F3e（偏股混合单类型，2016Q2 起），
   而不是主口径 F2/F3（2024Q1 起，只有 3 期有效）。
   两个口径不可混用；页面在每处标注"长历史"。

3. **四象限用绝对阈值**，不用 31 个行业的横截面中位数：
   中位数会把"深度低配但在加仓"的行业标成"加仓拥挤"，与实际含义脱节。
   阈值：Z ≥ 2.0 或分位 ≥ 80 → 高位；Z ≤ −1.5 或分位 ≤ 20 → 低位。
   方向：|ΔZ| < 0.2 视为方向不明，不参与四象限命名。

4. **无未来函数**：T 期因子只用 T 期及之前已披露数据（F2e 为扩窗 Z），
   收益取 T+1 期（forward_excess 的 index=T，value=T+1 期超额收益）。

5. **所有统计结论必须带**：样本期、n、t、p、判定词（显著/边际/不显著）。
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

import backtest as bt
import config as C

# ---------------------------------------------------------------------------
# 四象限定义（模块顶部，避免依赖 import 顺序）
# ---------------------------------------------------------------------------
Z_HI, Z_LO = 2.0, -1.5          # 高位 / 低位 的 Z 阈值
PCT_HI, PCT_LO = 80.0, 20.0     # 高位 / 低位 的分位阈值
DZ_MIN = 0.20                   # |ΔZ| 小于该值视为「方向不明」

QUAD_META = {
    "HH": dict(name="高位加仓", short="高位加仓",
               tag="最后一棒风险：拥挤已极值仍在加码",
               color_key="up"),
    "HC": dict(name="高位减仓", short="高位减仓",
               tag="回避信号：拥挤开始松动",
               color_key="down"),
    "LH": dict(name="低位加仓", short="低位加仓",
               tag="预期差：需连续两季确认才算成立",
               color_key="up"),
    "LC": dict(name="低位减仓", short="低位减仓",
               tag="系统性放弃：机构在无人问津处继续撤",
               color_key="down"),
    "MID": dict(name="中性区", short="中性区·不构成信号",
                tag="区位或方向不明确，不做四象限命名",
                color_key="text_dim"),
}
QUAD_ORDER = ["HH", "HC", "LH", "LC", "MID"]


# ---------------------------------------------------------------------------
# 基础统计
# ---------------------------------------------------------------------------
def t_p(x: Iterable[float]) -> tuple[float, float, int]:
    """普通单样本 t 检验（与任务书基准口径一致）。返回 (t, p, n)。"""
    s = pd.Series(list(x)).dropna().astype(float)
    n = len(s)
    if n < 3 or s.std(ddof=1) == 0:
        return np.nan, np.nan, n
    t = float(s.mean() / (s.std(ddof=1) / np.sqrt(n)))
    p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1)))
    return t, p, n


def verdict(p: float) -> str:
    """判定词：显著 / 边际 / 不显著（5% 与 10% 两个门槛）。"""
    if pd.isna(p):
        return "样本不足"
    if p < 0.05:
        return "显著"
    if p < 0.10:
        return "边际"
    return "不显著"


def perf(s: pd.Series) -> dict:
    """多空季度收益（%）→ 年化、t、p、胜率、累计。"""
    s = pd.Series(s).dropna().astype(float)
    n = len(s)
    if n == 0:
        return dict(n=0, ann=np.nan, ann_arith=np.nan, t=np.nan, p=np.nan,
                    win=np.nan, total=np.nan, nw_t=np.nan, nw_p=np.nan)
    total = float((1 + s / 100).prod() - 1)
    ann = ((1 + total) ** (4 / n) - 1) * 100 if total > -1 else np.nan
    t, p, _ = t_p(s)
    nw_t, nw_p = bt.nw_tstat(s)
    return dict(n=n, ann=ann, ann_arith=float(s.mean() * 4),
                t=t, p=p, win=float((s > 0).mean() * 100),
                total=total * 100, nw_t=nw_t, nw_p=nw_p)


# ---------------------------------------------------------------------------
# 1. 变化表（31 行业）
# ---------------------------------------------------------------------------
def delta_table(b, ind: dict, q: str | None = None) -> pd.DataFrame:
    """构建当期变化表：ΔZ 为主，Δ配置pp 为辅。

    返回 index = 行业，列：
        z / pct          当期水平（长历史口径）
        dz / d_pct       环比变化（ΔZ 为主判断）
        alloc / alloc0   本期 / 上期配置比例（主口径，%）
        d_alloc          Δ配置pp（仅辅助）
        over / over0     本期 / 上期超配（主口径，%）
        lvl / lvl0       本期 / 上期拥挤等级
        zone             区位：high / low / mid
        quad             四象限键：HH/HC/LH/LC/MID
        dir              方向：加仓 / 减仓 / 方向不明
        conflict         ΔZ 与 Δ配置pp 符号是否相反
        gate             可信度闸门标记
    """
    q = q or b.latest
    qs = list(b.quarters)
    if q not in qs:
        raise ValueError(f"{q} 不在季度序列中")
    i = qs.index(q)
    q0 = qs[i - 1] if i > 0 else None

    z = ind["F2e_超配Z_长历史"]
    pct = ind["F3e_超配分位_长历史"]
    dz = z.diff()
    dpct = pct.diff()

    alloc = b.alloc.reindex(columns=C.INDUSTRIES)
    over = b.overweight.reindex(columns=C.INDUSTRIES)
    lvl = ind["拥挤等级"].reindex(columns=C.INDUSTRIES)

    t = pd.DataFrame(index=pd.Index(C.INDUSTRIES, name="行业"))
    t["z"] = z.loc[q]
    t["pct"] = pct.loc[q]
    t["dz"] = dz.loc[q]
    t["d_pct"] = dpct.loc[q]
    t["alloc"] = alloc.loc[q] if q in alloc.index else np.nan
    t["alloc0"] = alloc.loc[q0] if q0 in alloc.index else np.nan
    t["d_alloc"] = t["alloc"] - t["alloc0"]
    t["over"] = over.loc[q] if q in over.index else np.nan
    t["over0"] = over.loc[q0] if q0 in over.index else np.nan
    t["turn_pct"] = (ind["F9b_成交额占比分位"].loc[q]
                     if "F9b_成交额占比分位" in ind else np.nan)
    t["lvl"] = lvl.loc[q] if q in lvl.index else np.nan
    t["lvl0"] = lvl.loc[q0] if q0 in lvl.index else np.nan

    t["zone"] = [zone_of(a, c) for a, c in zip(t["z"], t["pct"])]
    t["dir"] = [
        "加仓" if pd.notna(x) and x >= DZ_MIN
        else "减仓" if pd.notna(x) and x <= -DZ_MIN
        else "方向不明"
        for x in t["dz"]
    ]
    t["quad"] = [quad_of(a, d) for a, d in zip(t["zone"], t["dir"])]
    t["conflict"] = (
        (pd.to_numeric(t["dz"], errors="coerce") * pd.to_numeric(t["d_alloc"], errors="coerce")) < 0
    )
    t["gate"] = ""
    t.attrs["q"], t.attrs["q0"] = q, q0
    return t


def zone_of(z, pct) -> str:
    """区位判定：high / low / mid（绝对阈值，不用横截面中位数）。"""
    hi = (pd.notna(z) and z >= Z_HI) or (pd.notna(pct) and pct >= PCT_HI)
    lo = (pd.notna(z) and z <= Z_LO) or (pd.notna(pct) and pct <= PCT_LO)
    if hi and not lo:
        return "high"
    if lo and not hi:
        return "low"
    return "mid"


def quad_of(zone: str, direction: str) -> str:
    if direction == "方向不明" or zone == "mid":
        return "MID"
    if zone == "high":
        return "HH" if direction == "加仓" else "HC"
    return "LH" if direction == "加仓" else "LC"


# ---------------------------------------------------------------------------
# 2. 环比可信度闸门
# ---------------------------------------------------------------------------
def credibility_gate(b, ind: dict, t: pd.DataFrame,
                     fund_series: pd.Series | None = None) -> pd.DataFrame:
    """单季 |Δ配置| > 5pp 时触发检查（共 4 项证据）。

    ① 样本基金数是否突变（>15%）；
    ② 配置变化能否被「价格漂移」解释（用 attribution 的被动项）；
    ③ 是否与 ΔZ 符号相反（仓位降但相对历史仍在高位，或反之）；
    ④ 是否「配置大动而历史分位几乎未动」——典型的口径断层特征。

    判定逻辑：③④ 同时成立才是「待核实」；若②显示主要是价格漂移，
    则降为「价格驱动」，不阻断结论。
    """
    import attribution as atr

    fund_series = fund_series if fund_series is not None else pd.Series(dtype=float)
    q, q0 = t.attrs.get("q"), t.attrs.get("q0")
    f1 = fund_series.get(q, np.nan) if q in getattr(fund_series, "index", []) else np.nan
    f0 = fund_series.get(q0, np.nan) if q0 in getattr(fund_series, "index", []) else np.nan
    fund_chg = (f1 / f0 - 1) * 100 if pd.notna(f1) and pd.notna(f0) and f0 else np.nan

    # 价格漂移（被动项）：只在主口径季报齐全时可用
    try:
        dec = atr.decompose(b.alloc, b.market["ret"])
        passive = {k: atr.split_one(dec, q, k).get("被动贡献", np.nan)
                   for k in t.index}
    except Exception:
        passive = {k: np.nan for k in t.index}

    if "gate" not in t.columns:
        t["gate"] = ""
    if "gate_note" not in t.columns:
        t["gate_note"] = ""
    if "passive" not in t.columns:
        t["passive"] = np.nan
    if "active" not in t.columns:
        t["active"] = np.nan
    t["passive"] = [passive.get(k, np.nan) for k in t.index]
    t["active"] = t["d_alloc"] - t["passive"]

    for name in t.index:
        r = t.loc[name]
        if pd.isna(r["d_alloc"]) or abs(r["d_alloc"]) <= 5.0:
            continue
        notes = []
        pv = passive.get(name, np.nan)
        active = (r["d_alloc"] - pv) if pd.notna(pv) else np.nan
        price_driven = pd.notna(active) and abs(active) < 1.0
        level = "提示"
        if price_driven:
            notes.append(f"价格漂移 {pv:+.2f}pp 解释了 {abs(pv) / abs(r['d_alloc']) * 100:.0f}%，"
                         f"主动项 {active:+.2f}pp")
        if pd.notna(fund_chg) and abs(fund_chg) > 15:
            notes.append(f"样本基金数环比 {fund_chg:+.1f}%，需核对口径")
            level = "待核实"
        if bool(r["conflict"]):
            notes.append(f"Δ配置 {r['d_alloc']:+.2f}pp 与 ΔZ {r['dz']:+.2f}σ 符号相反")
        needs_check = (pd.notna(r["dz"]) and abs(r["dz"]) < 0.5
                       and not price_driven)
        if needs_check:
            notes.append("配置大动但历史分位几乎未动，且价格无法解释，存在口径断层可能")
            level = "待核实"
        if pd.isna(fund_chg):
            notes.append("本期样本基金数未采集，无法用基金数核对")
            # 基金数缺失时，若价格已能解释，不升级为待核实
            if not price_driven and level == "提示":
                level = "待核实"
        t.at[name, "gate"] = level
        t.at[name, "gate_note"] = "；".join(notes)
    t["gate_note"] = t["gate_note"].fillna("")
    return t


# ---------------------------------------------------------------------------
# 3. 等级迁移
# ---------------------------------------------------------------------------
LEVEL_ORDER = ["极度拥挤", "拥挤", "偏拥挤", "中性", "显著低配", "数据不足"]


def level_migration(t: pd.DataFrame) -> pd.DataFrame:
    """上期等级 × 本期等级的迁移矩阵（行=上期，列=本期）。"""
    rows = [x for x in LEVEL_ORDER if x in set(t["lvl0"].dropna())]
    cols = [x for x in LEVEL_ORDER if x in set(t["lvl"].dropna())]
    m = pd.crosstab(t["lvl0"], t["lvl"]).reindex(index=rows, columns=cols).fillna(0)
    return m.astype(int)


LEVEL_RANK = {"极度拥挤": 5, "拥挤": 4, "偏拥挤": 3, "中性": 2,
              "显著低配": 1, "数据不足": np.nan}


def net_upgrade(t: pd.DataFrame) -> int:
    """等级净升级数 = 升级家数 − 降级家数（按 LEVEL_RANK 排序）。"""
    up = down = 0
    for _, r in t.iterrows():
        a, b_ = LEVEL_RANK.get(r["lvl0"], np.nan), LEVEL_RANK.get(r["lvl"], np.nan)
        if pd.isna(a) or pd.isna(b_):
            continue
        if b_ > a:
            up += 1
        elif b_ < a:
            down += 1
    return up - down


def net_upgrade_series(ind: dict, quarters: list[str], window: int = 12,
                       level_key: str = "F11e_等级_长历史") -> pd.Series:
    """近 N 期「净升级行业数」序列。

    默认用长历史等级（F11e，30+ 期有效）：主口径「拥挤等级」只有 2026Q1 起
    的 2~3 期可比较，画不成趋势。长历史口径与主口径同一套阈值，仅超配序列
    更长（偏股混合 2016Q2 起），在图中明确标注。
    """
    lv = ind.get(level_key) if level_key in ind else ind["拥挤等级"]
    valid = [q for q in quarters
             if q in lv.index and int(lv.loc[q].ne("数据不足").sum()) >= 10]
    out = {}
    for i in range(1, len(valid)):
        q, q0 = valid[i], valid[i - 1]
        if q not in lv.index or q0 not in lv.index:
            continue
        a = lv.loc[q0].map(LEVEL_RANK)
        b_ = lv.loc[q].map(LEVEL_RANK)
        d = (b_ > a).sum() - (b_ < a).sum()
        out[q] = int(d)
    return pd.Series(out).tail(window)


# ---------------------------------------------------------------------------
# 4. IC 对比：水平 / Δ配置 / ΔZ
# ---------------------------------------------------------------------------
def ic_compare(b, ind: dict, start: str = "2024Q1") -> pd.DataFrame:
    """三种口径的 RankIC 对比（t+1 期超额收益）。

    * 配置水平     = alloc（主口径）
    * Δ配置pp      = alloc.diff()
    * Δ超配Z       = F2e.diff()（长历史口径）
    * 超配Z 水平   = F2e（长历史口径，对照）
    """
    fwd = bt.forward_excess(b.market, lag=1)
    z = ind["F2e_超配Z_长历史"]
    specs = [
        ("配置水平（主口径）", b.alloc.reindex(columns=C.INDUSTRIES), "水平"),
        ("超配Z 水平（长历史）", z, "水平"),
        ("Δ配置pp（主口径）", b.alloc.diff().reindex(columns=C.INDUSTRIES), "环比"),
        ("Δ超配Z（长历史）", z.diff(), "环比"),
    ]
    rows = []
    for name, fac, kind in specs:
        ic = bt.rank_ic_panel(fac, fwd).dropna()
        ic_short = ic.loc[[x for x in ic.index if x >= start]]
        for tag, s in (("2024Q1 起", ic_short), ("全部可用", ic)):
            if len(s) == 0:
                continue
            t, p, n = t_p(s)
            nw_t, nw_p = bt.nw_tstat(s)
            rows.append(dict(
                口径=name, 类型=kind, 区间=tag, n=n,
                RankIC=float(s.mean()), IC_IR=float(s.mean() / s.std(ddof=1)) if s.std(ddof=1) else np.nan,
                t=t, p=p, t_NW=nw_t, p_NW=nw_p,
                胜率=float((s > 0).mean() * 100),
                # 判定与表内展示的 NW p 保持一致（普通 t 的 p 仅作参考）
                判定=verdict(nw_p),
                起始=str(s.index[0]), 终止=str(s.index[-1]),
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. 多空组合
# ---------------------------------------------------------------------------
def long_short(b, fac: pd.DataFrame, k: int = 8, lag: int = 1) -> pd.Series:
    """每期按因子排序，做多 top-k、做空 bottom-k（等权），持有下期。

    收益单位：%。行业超额收益之差（同期同一基准，基准项相互抵消）。
    """
    fwd = bt.forward_excess(b.market, lag=lag)
    out = {}
    for q in fac.index:
        if q not in fwd.index:
            continue
        f = fac.loc[q].dropna()
        r = fwd.loc[q].reindex(f.index).dropna()
        f = f.reindex(r.index)
        if len(f) < 2 * k:
            continue
        s = f.sort_values()
        out[q] = float(r[s.index[-k:]].mean() - r[s.index[:k]].mean())
    return pd.Series(out).sort_index()


def long_short_bundle(b, ind: dict, k: int = 8) -> dict:
    """ΔZ 多空：全样本、分区制、净值曲线。"""
    dz = ind["F2e_超配Z_长历史"].diff()
    s = long_short(b, dz, k=k)
    full = perf(s)
    periods = [("2019Q3", "2022Q4", "2019Q3–2022Q4 段"),
               ("2023Q1", "2026Q2", "2023Q1–2026Q2 段")]
    seg = []
    for a, z_, label in periods:
        x = s.loc[a:z_]
        st = perf(x)
        seg.append(dict(区间=label, n=st["n"], 年化=st["ann"], t=st["t"],
                        p=st["p"], 胜率=st["win"], 判定=verdict(st["p"])))
    return dict(series=s, full=full, segments=pd.DataFrame(seg), k=k)


def cost_sensitivity(b, ind: dict, k: int = 8,
                     bps_list=(0, 10, 20, 30)) -> pd.DataFrame:
    """交易成本敏感性：按季均单边换手折算净年化。

    换手定义：每期多头/空头组合各取 k 只等权；单边换手 = 0.5 × Σ|w_t − w_{t−1}|
    （仅计多头；空头同理，合计双边成本按 2×单边计）。
    """
    dz = ind["F2e_超配Z_长历史"].diff()
    fwd = bt.forward_excess(b.market, lag=1)
    gross, turns = {}, []
    prev = None
    for q in dz.index:
        if q not in fwd.index:
            continue
        f = dz.loc[q].dropna()
        r = fwd.loc[q].reindex(f.index).dropna()
        f = f.reindex(r.index)
        if len(f) < 2 * k:
            continue
        s = f.sort_values()
        hi = list(s.index[-k:])
        w = pd.Series(1.0 / k, index=hi)
        if prev is not None:
            allidx = w.index.union(prev.index)
            turns.append(0.5 * float((w.reindex(allidx, fill_value=0)
                                      - prev.reindex(allidx, fill_value=0)).abs().sum()))
        prev = w
        gross[q] = float(r[hi].mean() - r[s.index[:k]].mean())
    g = pd.Series(gross).sort_index()
    base = perf(g)
    turn_q = float(np.mean(turns)) if turns else np.nan
    rows = []
    for bp in bps_list:
        # 双边成本：多头 + 空头，各按单边 bps 计
        drag = turn_q * 4 * (bp / 10000) * 2 * 100 if pd.notna(turn_q) else np.nan
        rows.append(dict(单边成本_bp=bp,
                         年化毛=base["ann"],
                         年换手=round(turn_q * 4 * 100, 1) if pd.notna(turn_q) else np.nan,
                         成本拖累=drag,
                         年化净=base["ann"] - drag if pd.notna(drag) else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. 动量正交化（证伪）
# ---------------------------------------------------------------------------
def momentum_neutral(b, ind: dict, k: int = 8, lookback: int = 4) -> dict:
    """对「过去 4 季涨幅」做横截面回归，取残差再做多空。

    目的：检验 ΔZ 的多空收益是否只是价格动量的代理。
    """
    ret = b.market["ret"].reindex(columns=C.INDUSTRIES)
    mom = (1 + ret / 100).rolling(lookback).apply(np.prod, raw=True).sub(1) * 100
    dz = ind["F2e_超配Z_长历史"].diff()
    resid = pd.DataFrame(index=dz.index, columns=dz.columns, dtype=float)
    corr = []
    for q in dz.index:
        if q not in mom.index:
            continue
        y = dz.loc[q]
        x = mom.loc[q]
        d = pd.concat([y, x], axis=1).dropna()
        if len(d) < 10:
            continue
        X = np.column_stack([np.ones(len(d)), d.iloc[:, 1].to_numpy(float)])
        beta = np.linalg.lstsq(X, d.iloc[:, 0].to_numpy(float), rcond=None)[0]
        resid.loc[q] = y - (beta[0] + beta[1] * x)
        corr.append(float(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman")))
    s_raw = long_short(b, dz, k=k)
    s_res = long_short(b, resid, k=k)
    return dict(raw=perf(s_raw), resid=perf(s_res),
                corr_mean=float(np.mean(corr)) if corr else np.nan,
                corr_n=len(corr),
                n_periods=int(resid.dropna(how="all").shape[0]))


# ---------------------------------------------------------------------------
# 7. 样本充分性体检
# ---------------------------------------------------------------------------
def sample_audit(b, ind: dict) -> pd.DataFrame:
    """每个因子究竟几期有效——避免「43 季度」被误读成「43 期可用」。"""
    specs = [
        ("F2 超配Z（主口径）", "F2_超配Zscore", "主口径持仓 2024Q1 起，扩窗需 8 期"),
        ("F2e 超配Z（长历史）", "F2e_超配Z_长历史", "偏股混合 2016Q2 起，2018Q3 起可用"),
        ("F3e 超配分位（长历史）", "F3e_超配分位_长历史", "同上"),
        ("F4be 动量Z（长历史）", "F4be_动量Z_长历史", "同上"),
        ("F11e 综合拥挤度（长历史）", "F11e_综合拥挤度_长历史", "同上"),
        ("F1 超配比例（主口径）", "F1_超配比例", "主口径持仓"),
        ("F13 筹码盈利", "F13_筹码盈利比例", "需季报个股明细"),
    ]
    rows = []
    for label, key, note in specs:
        if key not in ind:
            rows.append(dict(因子=label, 有效期数=0, 末期非空=0, 总季度=len(b.quarters), 说明=note))
            continue
        s = ind[key]
        rows.append(dict(因子=label,
                         有效期数=int(s.dropna(how="all").shape[0]),
                         末期非空=int(s.iloc[-1].notna().sum()),
                         总季度=len(b.quarters), 说明=note))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. 行业变化 → 个股映射
# ---------------------------------------------------------------------------
def load_stock_pool(path: str | None = None) -> pd.DataFrame:
    """读取现有抱团股明细（hot_stocks_top20_*.xlsx，取最新一期）。"""
    if path is None:
        import glob
        cand = sorted(glob.glob(os.path.join(C.DATA_RAW, "hot_stocks_top20_*.xlsx")))
        path = cand[-1] if cand else ""
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["股票代码", "股票名称", "申万一级",
                                     "持股基金数", "持仓市值亿元"])
    d = pd.read_excel(path)
    d = d.rename(columns={"申万一级": "行业"})
    for c in ("持股基金数", "持仓市值亿元"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def stock_map(t: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """把行业环比方向映射到个股，给出可执行结论。

    排序：按所属行业 ΔZ 升序（撤得最狠的排最前，作为风险清单）。
    """
    if pool is None or pool.empty:
        return pd.DataFrame()
    d = pool.copy()
    d["行业"] = d.get("行业", pd.Series(dtype=str)).astype(str)
    d = d.join(t[["dz", "dir", "quad", "alloc", "lvl", "lvl0"]], on="行业")
    d["行业环比方向"] = d["dir"].fillna("—")
    # 2026Q3 个股级持股基金数未采集：保留列但留空，不以估算替代
    d["持股基金数环比"] = "暂无数据"
    d = d.sort_values(["dz", "持仓市值亿元"], ascending=[True, False])

    def _concl(r):
        if pd.isna(r.get("dz")):
            return "观察｜所属行业当期环比数据缺失"
        dz = float(r["dz"])
        q = t.attrs.get("q", "")
        if dz <= -0.8:
            return f"减｜所属{r['行业']}ΔZ {dz:+.2f}σ（{q}），本季不加仓"
        if dz <= -0.2:
            return f"观察｜所属{r['行业']}ΔZ {dz:+.2f}σ，等下一季确认再动"
        if dz >= 0.8:
            return f"加｜所属{r['行业']}ΔZ {dz:+.2f}σ，行业在加仓，个股可跟"
        if dz >= 0.2:
            return f"持｜所属{r['行业']}ΔZ {dz:+.2f}σ，小幅加仓，维持"
        return "观察｜行业方向不明，不加不减"

    d["结论"] = d.apply(_concl, axis=1)
    return d


# ---------------------------------------------------------------------------
# 9. 结论卡（判断 / 数字 / 上期 / 失效）
# ---------------------------------------------------------------------------
def _fmt(v, tpl="{:.2f}", na="—"):
    try:
        return na if v is None or pd.isna(v) else tpl.format(v)
    except Exception:
        return na


def top_movers(t: pd.DataFrame, min_alloc: float = 0.5) -> dict:
    """选出加仓最猛 / 减仓最猛。

    两条约束，缺一不可：
      ① 规模过滤：配置 < min_alloc% 的行业容量过小（如「综合」0.14%），
         ΔZ 再大也不具备可交易意义，不进入判断卡，只在变化榜中保留；
      ② 同号约束：加仓卡只取 ΔZ > 0 且 Δ配置 > 0 的行业，减仓卡只取
         ΔZ < 0 且 Δ配置 < 0 的行业。避免把「配置在加、相对历史在降」
         的有色金属（Δ配置 +0.43pp / ΔZ −0.85σ）说成「减仓最猛」。
    若某方向没有同号行业，退回纯 ΔZ 排序，并在卡片中显示冲突提示。
    """
    d = t.dropna(subset=["dz"]).copy()
    big = d[d["alloc"].fillna(0) >= min_alloc]
    pool = big if len(big) >= 5 else d

    add_same = pool[(pool["dz"] >= DZ_MIN) & (pool["d_alloc"] > 0)]
    cut_same = pool[(pool["dz"] <= -DZ_MIN) & (pool["d_alloc"] < 0)]
    add_pool = add_same if len(add_same) else pool
    cut_pool = cut_same if len(cut_same) else pool

    add = add_pool.sort_values("dz", ascending=False).iloc[0]
    cut = cut_pool.sort_values("dz").iloc[0]
    return dict(add=add, cut=cut, min_alloc=min_alloc,
                add_same=bool(len(add_same)), cut_same=bool(len(cut_same)),
                n_add=int((d["dz"] >= DZ_MIN).sum()),
                n_cut=int((d["dz"] <= -DZ_MIN).sum()),
                n_flat=int((d["dz"].abs() < DZ_MIN).sum()),
                n_conflict=int(d["conflict"].sum()))


def compare_rows(t: pd.DataFrame, names: Iterable[str] | None = None) -> pd.DataFrame:
    """指定的若干行业的前后期对照（用于文字结论的「上期」列）。"""
    names = list(names) if names is not None else list(t.index)
    return t.reindex(names)


def max_single_cut(b, ind: dict, top: int = 3) -> pd.DataFrame:
    """长历史口径的历史单季减仓排名（Δ配置pp），用于「历史第几大」的表述。

    口径：偏股混合单类型 alloc_ext（2016Q2 起 38 期），与主口径 11 期不同。
    """
    da = b.alloc_ext.diff().stack().dropna().sort_values()
    out = da.head(top).rename("d_alloc").reset_index()
    out.columns = ["季度", "行业", "d_alloc"]
    return out
