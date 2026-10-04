# -*- coding: utf-8 -*-
"""
research3.py  ——  第 3 轮自主迭代
======================================================================
承接 README「R2-5 下一轮计划」的 6 项，逐项解决第 2 轮暴露出的缺口：

  R3-A  Walk-forward 滚动样本外          walk_forward_test / wf_regime_breakdown
        问题：R2 的"设计期样本外"只有**一次**切分（≤2022Q4 / ≥2023Q1），
              结果是**单点**；而该留出期恰好是 AI 主题单边行情，
              12/15 的高通过率无法区分"因子有效"与"这段时间好做"。
        做法：每期只用**截至 t-1** 的数据挑设定，在 t 期实现一次前瞻 IC；
              滚动 20+ 次 → 得到"选设定流程"样本外表现的**分布**，而非单点。

  R3-B  时间戳前移的严格设计检验        timestamped_design_test
        问题：F15 的构造灵感来自**全样本**上 F1 与 F2 符号相反这一观察。
        做法：只用 ≤OBS_END（默认 2019Q4）的数据确认该观察，
              再对 ≥TEST_START（默认 2020Q1）一次性检验。
              这才是真正的"当时可做"的检验。

  R3-C  正交双因子组合                  dual_factor_combination
        问题：F15（IC −0.12）与 F3d（IC +0.11）方向相反、量的级相当。
              若不检验组合，等于默认"只用单个因子"。
        做法：方向按**设计期**符号对齐（不用全样本），等权组合 + 正交组合，
              走同一套三重评估（IC / 设计期样本外 / 二层正交 / 组合层策略）。

  R3-D  基准修正                        benchmark_variants
        问题：F15 策略战胜等权基准、却不战胜沪深300 —— 因为它天然低配
              第一大权重行业（电子）。这可能是"真 alpha"，也可能只是
              "低配大牛股的 beta"。
        做法：构造"市值加权但剔除 Top1/Top3 行业"的基准，
              剔除后再看超额是否仍然存在。

  R3-E  数据质量网关定时化              tools/scheduled_check.py（见该文件）
  R3-F  真实指数成分权重渠道探测        见 update_pipeline 输出与 README R3-3

约定：本模块只做计算与判定，不写文件；汇报由 update_pipeline 统一承担。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from factor_engine import cz, rank_ic_panel, forward_excess, safe_spearman
from backtest import nw_tstat, ic_stats
from research import _cap_normalize


# ============================================================================
# 公共：候选池与 IC 矩阵
# ============================================================================
WF_CANDIDATES = [
    "F1_超配比例", "F1d_超配比例_动态基准",
    "F2_超配Zscore", "F2d_超配Z_动态基准",
    "F3_超配历史分位", "F3d_超配分位_动态基准",
    "F12_配置系数", "F12s_配置系数_静态基准",
    "F13_筹码盈利比例", "F15_拥挤背离", "F15d_拥挤背离_动态基准",
    "F11_综合拥挤度_推荐权重",
]


def ic_matrix(ind: dict, b, candidates=None) -> pd.DataFrame:
    """构造 IC 面板：行=季度，列=候选因子（无未来函数，来自 rank_ic_panel）。"""
    candidates = candidates or WF_CANDIDATES
    fwd = forward_excess(b.market, lag=1)
    d = {}
    for f in candidates:
        if f in ind:
            d[f] = rank_ic_panel(ind[f], fwd)
    return pd.DataFrame(d)


# ============================================================================
# R3-A  Walk-forward 滚动样本外
# ============================================================================
def walk_forward_test(ind: dict, b, candidates=None, min_dev: int = 8,
                      trailing: int | None = None,
                      dev_end: str | None = None) -> dict:
    """滚动 walk-forward 检验（检验的是「**挑设定的流程**」能否复制）。

    对每一个有前瞻 IC 的季度 t：
      1. 设计窗 = 严格早于 t 的 IC 观测（可选只取最近 `trailing` 期）；
      2. 在设计窗内按 |IC 均值| 最大挑一个因子（要求该因子在设计窗内
         至少有 `min_dev` 个有效观测，否则不参与挑选）；
      3. 记录该因子在 t 期的**实现** IC，并按设计窗符号对齐
         （`方向对齐后 = sign(设计期IC) × 当期IC`）→ 这正是"照着设计窗
         的方向下注，实际赚到多少"。
      4. `dev_end` 若给定，则只在 t > dev_end 的季度上累计（模拟"从某时点
         开始真正实盘"），但仍允许 t 之前的所有历史用于设计。

    关键设计：**没有任何一步看到 t 期及以后的信息**，因此这条序列是
    真正的样本外序列，可以对其做统计推断（均值、NW t 值、胜率）。
    """
    ic = ic_matrix(ind, b, candidates)
    rows = []
    for i, t in enumerate(ic.index):
        prior = ic.iloc[:i]
        if trailing:
            prior = prior.iloc[-trailing:]
        cnt = prior.notna().sum()
        dm = prior.mean().where(cnt >= min_dev).dropna()
        cur = ic.loc[t].dropna()
        if dm.empty or cur.empty:
            continue
        fstar = dm.abs().idxmax()
        if fstar not in cur.index:
            continue
        sign = float(np.sign(dm[fstar]))
        rows.append(dict(季度=t, 选出因子=fstar,
                         设计期IC=float(dm[fstar]), 设计期n=int(cnt[fstar]),
                         当期IC=float(cur[fstar]), 方向对齐IC=sign * float(cur[fstar])))
    df = pd.DataFrame(rows)
    if df.empty:
        return dict(table=df, stats={}, series=pd.Series(dtype=float))
    if dev_end:
        df["是否留出"] = df["季度"] > dev_end
    else:
        df["是否留出"] = True
    hold = df[df["是否留出"]]
    sr = hold.set_index("季度")["方向对齐IC"]
    t_nw, p_nw = nw_tstat(sr, lags=2) if len(sr) > 3 else (np.nan, np.nan)
    st = dict(
        总期数=len(df), 留出期数=len(hold),
        留出期均值=float(sr.mean()) if len(sr) else np.nan,
        留出期中位数=float(sr.median()) if len(sr) else np.nan,
        留出期NW_t=t_nw, 留出期p=p_nw,
        留出期胜率=float((sr > 0).mean() * 100) if len(sr) else np.nan,
        留出期t检验p=float(stats.ttest_1samp(sr, 0).pvalue) if len(sr) > 2 else np.nan,
        选出因子种类=int(hold["选出因子"].nunique()) if len(hold) else 0,
        选出因子分布=(hold["选出因子"].value_counts().to_dict() if len(hold) else {}),
    )
    return dict(table=df, stats=st, series=sr)


def wf_window_sensitivity(ind: dict, b, windows=(None, 8, 12, 16),
                          candidates=None, min_dev: int = 6) -> pd.DataFrame:
    """Walk-forward 对"设计窗长度"的敏感性：结果不应随窗口剧烈变化。"""
    rows = []
    for w in windows:
        r = walk_forward_test(ind, b, candidates, min_dev=min_dev, trailing=w)
        s = r["stats"]
        if not s:
            continue
        rows.append(dict(设计窗=("扩窗（全部历史）" if w is None else f"滚动 {w} 季"),
                         留出期数=s["留出期数"], IC均值=s["留出期均值"],
                         IC_IR=(s["留出期均值"] / np.nan)
                         if False else np.nan,
                         NW_t=s["留出期NW_t"], p值=s["留出期p"],
                         胜率=s["留出期胜率"], 选出因子数=s["选出因子种类"]))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["IC_IR"] = np.nan
    return df


def wf_regime_breakdown(ind: dict, b, wf: dict | None = None,
                        candidates=None) -> pd.DataFrame:
    """把 walk-forward 的方向对齐 IC 按**市场状态**与**时间区块**分解。

    目的：回答"R2 的 12/15 高通过率有多少来自单一行情"——
    若样本外表现集中在某一区块/某一状态，则结论的普适性必须打折。
    """
    wf = wf or walk_forward_test(ind, b, candidates)
    df = wf["table"]
    if df.empty:
        return pd.DataFrame()

    hs = b.market["hs300"]
    trail = ((1 + hs / 100.0).rolling(4, min_periods=4).apply(np.prod, raw=True) - 1) * 100
    regime = pd.Series("震荡市", index=hs.index)
    regime[trail > 15] = "牛市"
    regime[trail < -10] = "熊市"

    out = []
    hold = df[df["是否留出"]]
    for g in ["牛市", "熊市", "震荡市"]:
        idx = [q for q in hold["季度"] if regime.get(q) == g]
        s = hold.set_index("季度").loc[idx, "方向对齐IC"].dropna()
        if len(s) == 0:
            continue
        out.append(dict(维度="市场状态", 分组=g, 期数=len(s),
                        IC均值=s.mean(), 中位数=s.median(),
                        胜率=(s > 0).mean() * 100,
                        年化贡献=s.mean() * 4 * 100))
    parts = np.array_split(np.arange(len(hold)), 3)
    for i, p in enumerate(parts, 1):
        seg = hold.iloc[p]
        s = seg["方向对齐IC"].dropna()
        if len(s) == 0:
            continue
        out.append(dict(维度="时间区块", 分组=f"第{i}/3 段（{seg['季度'].iloc[0]}~{seg['季度'].iloc[-1]}）",
                        期数=len(s), IC均值=s.mean(), 中位数=s.median(),
                        胜率=(s > 0).mean() * 100, 年化贡献=s.mean() * 4 * 100))
    return pd.DataFrame(out)


# ============================================================================
# R3-B  时间戳前移的严格设计检验
# ============================================================================
OBS_END = "2019Q4"       # 允许用于"观察"的最后一期
TEST_START = "2020Q1"    # 检验期起点


def timestamped_design_test(ind: dict, b, obs_end: str = OBS_END,
                            test_start: str = TEST_START,
                            candidates=None) -> dict:
    """只用 ≤obs_end 的数据确认构造依据，再对 ≥test_start 一次性检验。

    与 R2 的 design_oos_test 的区别：
      · R2 观察窗到 2022Q4（更长，但仍是"事后"选择）；
      · 这里刻意把观察窗压到 **2019Q4**（此时 F15 的构造依据
        "F1 与 F2 符号相反"是否已经成立？）——如果成立，
        说明这个设计在当时是可得的，不是事后归纳。
    """
    ic = ic_matrix(ind, b, candidates)
    dev = ic.loc[[q for q in ic.index if q <= obs_end]].dropna(how="all")
    test = ic.loc[[q for q in ic.index if q >= test_start]].dropna(how="all")
    if dev.empty or test.empty:
        return dict(premise=None, table=pd.DataFrame(), detail={})

    # ---- 构造依据检查：F1 与 F2 在观察窗内是否符号相反 ----
    prem_rows = []
    for f in ["F1_超配比例", "F2_超配Zscore", "F3_超配历史分位",
              "F1d_超配比例_动态基准", "F2d_超配Z_动态基准"]:
        if f in dev.columns:
            prem_rows.append(dict(因子=f, 观察窗IC均值=float(dev[f].mean()),
                                  观察窗n=int(dev[f].notna().sum()),
                                  方向=("负" if dev[f].mean() < 0 else "正")))
    premise = pd.DataFrame(prem_rows)
    f1_sign = None
    f2_sign = None
    if "F1_超配比例" in dev.columns and "F2_超配Zscore" in dev.columns:
        f1_sign = float(np.sign(dev["F1_超配比例"].mean()))
        f2_sign = float(np.sign(dev["F2_超配Zscore"].mean()))
    premise_ok = bool(f1_sign is not None and f2_sign is not None
                      and f1_sign * f2_sign < 0)

    # ---- 观察窗可得的最优设定（当时若照此选择，会选谁） ----
    cand_rank = (dev.mean().to_frame("观察窗IC均值")
                 .assign(观察窗n=dev.notna().sum(),
                         _a=lambda d: d["观察窗IC均值"].abs())
                 .sort_values("_a", ascending=False).drop(columns="_a"))

    # ---- 检验期一次性检验 ----
    rows = []
    for f in [c for c in (candidates or WF_CANDIDATES) if c in test.columns]:
        if int(test[f].notna().sum()) < 4:
            continue
        s = test[f].dropna()
        st = ic_stats(s)
        t, p = nw_tstat(s, lags=2)
        rows.append(dict(因子=f, 检验期IC均值=st["ic_mean"], 检验期n=st["n"],
                         检验期IC_IR=st["ic_ir"], NW_t=t, p值=p,
                         胜率=st["win_rate"],
                         符号与观察窗一致=(
                             bool(np.sign(st["ic_mean"]) == np.sign(dev[f].mean()))
                             if f in dev.columns else None)))
    tbl = pd.DataFrame(rows)
    if not tbl.empty:
        tbl = tbl.sort_values("检验期IC均值", key=lambda c: -c.abs())
    return dict(premise=premise, premise_ok=premise_ok,
                cand_rank=cand_rank, table=tbl,
                obs_end=obs_end, test_start=test_start,
                ic_dev=dev, ic_test=test)


# ============================================================================
# R3-C  正交双因子组合
# ============================================================================
def dual_factor_combination(ind: dict, b, f_a: str = "F15_拥挤背离",
                            f_b: str = "F3d_超配分位_动态基准",
                            sign_from: str | None = None,
                            min_dev: int = 8) -> dict:
    """把两个方向相反、量级相当的因子合成。

    步骤（全部只用 sign_from 及之前的信息决定方向，避免前视）：
      1. 方向对齐：sign_a = −sign(设计窗 IC_a)、sign_b = +sign(设计窗 IC_b)……严格说
         应取"使两者都预期正"的符号：d_a = sign_a · z_cs(F_a) 使得设计窗内
         E[IC(d_a)] > 0。故 sign_k = sign(设计窗 IC_k)。
      2. 等权组合：F16a = z_cs(d_a) + z_cs(d_b)
      3. 正交组合：F16b = z_cs(d_a) + z_cs(resid(d_b | d_a))，逐期截面 OLS 取残差。
         （若 d_b 的增量信息在 d_a 之外，正交组合应优于等权组合。）

    输出每个候选与原两因子的 IC / IC_IR / 相关度，供直接比较。
    """
    fwd = forward_excess(b.market, lag=1)
    ic_a = rank_ic_panel(ind[f_a], fwd)
    ic_b = rank_ic_panel(ind[f_b], fwd)
    if sign_from is None:
        # 自动找出「两因子都已有 ≥min_dev 个 IC 观测」的最早季度：
        # 只有到这个时点，方向才可能在**当时**被确定，否则就是事后选符号。
        both = (ic_a.expanding().count() >= min_dev) & (ic_b.expanding().count() >= min_dev)
        cand_qs = [q for q in ic_a.index if bool(both.get(q, False))]
        if not cand_qs:
            return dict(table=pd.DataFrame(), corr=None, sign_a=None, sign_b=None,
                        sign_from=None, note="两因子从未同时具备足够观测")
        sign_from = cand_qs[0]
    dev_a = ic_a.loc[[q for q in ic_a.index if q <= sign_from]].dropna()
    dev_b = ic_b.loc[[q for q in ic_b.index if q <= sign_from]].dropna()
    if len(dev_a) < 4 or len(dev_b) < 4:
        return dict(table=pd.DataFrame(), corr=None, sign_a=None, sign_b=None,
                    sign_from=sign_from, note="设计窗观测不足")
    sign_a = float(np.sign(dev_a.mean())) or -1.0
    sign_b = float(np.sign(dev_b.mean())) or 1.0

    d_a = cz(ind[f_a]) * sign_a
    d_b = cz(ind[f_b]) * sign_b

    # 逐期截面残差：d_b 对 d_a 回归取残差
    resid = pd.DataFrame(np.nan, index=d_b.index, columns=d_b.columns, dtype=float)
    for q in d_b.index:
        y, x = d_b.loc[q], d_a.loc[q]
        m = y.notna() & x.notna()
        if int(m.sum()) < 15:
            continue
        X = np.column_stack([np.ones(int(m.sum())), x[m].to_numpy(dtype=float)])
        beta, *_ = np.linalg.lstsq(X, y[m].to_numpy(dtype=float), rcond=None)
        resid.loc[q, m.index[m]] = y[m].to_numpy(dtype=float) - X @ beta

    f16a = cz(d_a) + cz(d_b)
    f16b = cz(d_a) + cz(resid)

    # 逐期截面相关（原始两因子）
    corrs = []
    for q in ind[f_a].index:
        x, y = ind[f_a].loc[q], ind[f_b].loc[q]
        m = x.notna() & y.notna()
        if int(m.sum()) >= 8:
            _c = safe_spearman(x[m], y[m])
            if _c == _c:
                corrs.append(abs(_c))
    corr_mean = float(np.mean(corrs)) if corrs else np.nan

    rows = []
    for name, fac in [(f"{f_a}（原始，方向未对齐）", ind[f_a]),
                      (f"{f_b}（原始，方向未对齐）", ind[f_b]),
                      ("d_a 方向对齐", d_a), ("d_b 方向对齐", d_b),
                      ("F16a 等权组合", f16a), ("F16b 正交组合", f16b)]:
        st = ic_stats(rank_ic_panel(fac, fwd))
        t, p = nw_tstat(rank_ic_panel(fac, fwd).dropna(), lags=2)
        rows.append(dict(构造=name, IC均值=st["ic_mean"], IC_IR=st["ic_ir"],
                         n=st["n"], NW_t=t, p值=p, 胜率=st["win_rate"]))
    tbl = pd.DataFrame(rows)
    return dict(table=tbl, sign_a=sign_a, sign_b=sign_b, corr=corr_mean,
                sign_from=sign_from,
                factors={"F16a_等权双因子": f16a, "F16b_正交双因子": f16b,
                         "d_a": d_a, "d_b": d_b, "resid_b": resid})


# ============================================================================
# R3-D  基准修正：剔除最大权重行业
# ============================================================================
def make_base_weights(b, kind: str, q: str | None = None) -> tuple[pd.Series, str]:
    """构造基准权重向量。

    kind:
      equal            等权 31 行业
      hs300            沪深300 动态季度权重
      hs300_ex_top1    沪深300 动态权重，剔除当期最大权重行业后归一
      hs300_ex_top3    沪深300 动态权重，剔除当期前 3 大权重行业后归一
      equal_ex_top1    等权，剔除当期最大权重行业
    """
    q = q or b.latest
    n = len(C.INDUSTRIES)
    wq = b.weight_matrix.reindex(index=b.quarters, columns=C.INDUSTRIES).loc[q]
    if kind == "equal":
        return pd.Series(1.0 / n, index=C.INDUSTRIES), "等权 31 行业"
    if kind == "hs300":
        return (wq / 100.0).astype(float), "沪深300 动态权重"
    if kind in ("hs300_ex_top1", "hs300_ex_top3"):
        k = 1 if kind.endswith("top1") else 3
        drop = wq.nlargest(k).index
        w = wq.drop(index=drop)
        w = w / w.sum()
        # 注意：剔除对象是**逐期变化**的（早期是银行、近年是电子），
        # 标签只报告该期实际剔除的行业，不能写成固定行业名。
        return w.fillna(0.0), (f"沪深300 动态权重 ＋ 逐期剔除当期前 {k} 大权重行业"
                               f"（{q} 期为：{'、'.join(drop)}）")
    if kind == "equal_ex_top1":
        drop = wq.nlargest(1).index
        w = pd.Series(0.0, index=C.INDUSTRIES)
        w[w.index.difference(drop)] = 1.0 / (n - 1)
        return w, f"等权（剔除最大权重行业：{'、'.join(drop)}）"
    raise ValueError(f"未知基准类型：{kind}")


def benchmark_variants(ind: dict, b, factor: str = "F15_拥挤背离",
                       lam: float = 0.5, cap: float = 0.10,
                       cost_bp: float = 20.0,
                       kinds=("equal", "equal_ex_top1", "hs300",
                              "hs300_ex_top1", "hs300_ex_top3")) -> pd.DataFrame:
    """在同一策略权重下，换用多个基准计算超额 → 区分真 alpha 与"低配大牛股"的 beta。

    关键：策略的**起始权重固定为等权 31 行业**（不随基准变化），
    只有"用来算超额的基准"在变。这样超额差异完全来自基准选择，
    不掺杂策略自身的差异。
    """
    ret = b.market["ret"].reindex(index=b.quarters, columns=C.INDUSTRIES).astype(float)
    base0, _ = make_base_weights(b, "equal")
    rows = []
    for kind in kinds:
        bm, label = make_base_weights(b, kind)
        # 基准要逐期可得 → 用每期各自的权重（top 剔除是逐期变化的）
        wts, prev, recs = {}, None, []
        for i, q in enumerate(b.quarters):
            bw, _lab = make_base_weights(b, kind, q=q)
            z = cz(ind[factor].loc[[q]]).iloc[0]
            if z.notna().sum() < 20:
                continue
            z = z.fillna(0.0)
            kap = -1.0   # 默认按经济先验方向（F15 类）
            raw = base0 * np.exp(kap * lam * z)
            w = _cap_normalize(raw, cap)
            wts[q] = w
            if i + 1 >= len(b.quarters):
                break
            r_next = ret.iloc[i + 1]
            m = r_next.notna()
            port = float((w[m] * r_next[m] / 100.0).sum())
            bench = float((bw.reindex(r_next.index).fillna(0.0)[m] * r_next[m] / 100.0).sum())
            tv = np.nan if prev is None else float((w - prev).abs().sum()) / 2.0
            recs.append(dict(quarter=f"{q}→{b.quarters[i + 1]}", 组合=port * 100,
                             基准=bench * 100, 超额=(port - bench) * 100, turnover=tv))
            prev = w
        # 空行保护：因子无可用观测时 recs 为空，set_index 会抛 KeyError。
        # 必须先判空，让该基准变体被跳过而不是让整个看板崩溃。
        if not recs:
            continue
        t = pd.DataFrame(recs).set_index("quarter")
        if t.empty:
            continue
        t["turnover"] = t["turnover"].fillna(t["turnover"].dropna().mean()
                                             if t["turnover"].notna().any() else 1.0)
        t["超额_净"] = (t["组合"] - t["turnover"] * (cost_bp / 10000.0) * 100 * 2
                        - t["基准"])
        ex = t["超额_净"]
        rows.append(dict(基准=kind, 基准说明=label, 期数=len(t),
                         基准年化=((1 + t["基准"] / 100).prod() ** (4 / len(t)) - 1) * 100,
                         组合净年化=((1 + (t["组合"] - t["turnover"] * (cost_bp / 10000.0)
                                          * 100 * 2) / 100).prod() ** (4 / len(t)) - 1) * 100,
                         # ex 已是百分数(%)，年化=均值×4，年化波动=std×2，**不得再 ×100**
                         # ⚠️ 同时给「算术年化」与「几何年化」：波动大时二者差异巨大，
                         #    只报算术均值会系统性高估可实现的超额。
                         超额净年化_算术=ex.mean() * 4,
                         超额净年化_几何=((1 + ex / 100).prod() ** (4 / len(ex)) - 1) * 100
                         if (1 + ex / 100).prod() > 0 else np.nan,
                         超额波动=ex.std(ddof=1) * 2 if ex.std(ddof=1) else np.nan,
                         IR=(ex.mean() * 4) / (ex.std(ddof=1) * 2)
                         if ex.std(ddof=1) else np.nan,
                         超额胜率=(ex > 0).mean() * 100,
                         最大回撤=((1 + ex / 100).cumprod()
                                 / (1 + ex / 100).cumprod().cummax() - 1).min() * 100))
    return pd.DataFrame(rows)


def premise_availability(ind: dict, b, f_a: str = "F1_超配比例",
                         f_b: str = "F2_超配Zscore",
                         min_obs: int = 8, candidates=None) -> dict:
    """找出「构造依据」最早可得的时点，并从该时点起做一次性检验。

    构造依据（F15 的定义）：`F1（超配水平）与 F2（超配Z）的 IC 符号相反`。
    本函数逐期向前推进，找出**最早**满足下列条件的季度 t*：
      · F1 与 F2 在该期及以前都有 ≥ min_obs 个 IC 观测；
      · 两者 IC 均值符号相反。

    然后以 **t* 为观察截止**、**t*+1 起为检验期** 做事后无法再修改的检验。
    这才是"当时真的能做出来吗"的答案。
    若直到样本末都没有满足条件的 t*，则说明构造依据在全样本期内
    **并非事先可得**，F15 属于事后归纳（in-sample construct）。
    """
    ic = ic_matrix(ind, b, candidates)
    if f_a not in ic.columns or f_b not in ic.columns:
        return dict(found=False, reason="因子缺失")
    qs = list(ic.index)
    t_star = None
    hist = []
    for i, q in enumerate(qs):
        prior = ic.iloc[:i + 1]
        ca, cb = prior[f_a].notna().sum(), prior[f_b].notna().sum()
        if ca < min_obs or cb < min_obs:
            hist.append(dict(季度=q, F1累计IC=prior[f_a].mean(), F1_n=int(ca),
                             F2累计IC=prior[f_b].mean(), F2_n=int(cb),
                             符号相反=False))
            continue
        ma, mb = prior[f_a].mean(), prior[f_b].mean()
        ok = bool(np.sign(ma) * np.sign(mb) < 0)
        hist.append(dict(季度=q, F1累计IC=ma, F1_n=int(ca),
                         F2累计IC=mb, F2_n=int(cb), 符号相反=ok))
        if ok and t_star is None:
            t_star = q
    if t_star is None:
        return dict(found=False, t_star=None, hist=pd.DataFrame(hist),
                    reason=f"全样本期内 F1/F2 从未同时满足「各自 ≥{min_obs} 个观测」"
                           f"且「符号相反」→ 构造依据并非事先可得，F15 属事后归纳")
    idx = qs.index(t_star)
    test_qs = qs[idx + 1:]
    rows = []
    for f in [c for c in (candidates or WF_CANDIDATES) if c in ic.columns]:
        s_test = ic.loc[test_qs, f].dropna()
        if len(s_test) < 4:
            continue
        st = ic_stats(s_test)
        t, p = nw_tstat(s_test, lags=2)
        rows.append(dict(因子=f, 检验期IC均值=st["ic_mean"], 检验期n=st["n"],
                         检验期IC_IR=st["ic_ir"], NW_t=t, p值=p,
                         胜率=st["win_rate"]))
    tbl = pd.DataFrame(rows)
    if not tbl.empty:
        tbl = tbl.sort_values("检验期IC均值", key=lambda c: -c.abs())
    return dict(found=True, t_star=t_star, test_start=test_qs[0] if test_qs else None,
                test_end=test_qs[-1] if test_qs else None,
                n_test=len(test_qs), hist=pd.DataFrame(hist), table=tbl)


def excess_attribution(ind: dict, b, factor: str = "F15_拥挤背离",
                       lam: float = 0.5, cap: float = 0.10,
                       base: str = "equal", benchmark: str = "hs300",
                       cost_bp: float = 20.0) -> dict:
    """超额收益的**行业归因**——直接回答「超额是否只来自避开某一个大牛行业」。

    逐期超额可精确分解为各行业贡献之和：
        excess(t) = Σ_i [ w_i(t) − bm_i(t) ] × r_i(t+1)
    其中 w = 策略权重、bm = 基准权重（均为第 t 期已知）。
    汇总到行业维度后：
        · 若超额几乎全部来自"对某行业的负权重 × 该行业的高收益"（负贡献），
          则超额本质是**低配大牛股的 beta**，不是选行业的能力；
        · 若多个行业同时有正贡献且分散，则更接近**真 alpha**。
    """
    ret = b.market["ret"].reindex(index=b.quarters, columns=C.INDUSTRIES).astype(float)
    base0, base_lab = make_base_weights(b, base)
    rows, detail = [], []
    prev = None
    for i, q in enumerate(b.quarters):
        z = cz(ind[factor].loc[[q]]).iloc[0]
        if z.notna().sum() < 20 or i + 1 >= len(b.quarters):
            continue
        z = z.fillna(0.0)
        w = _cap_normalize(base0 * np.exp(-lam * z), cap)
        bw, _lab = make_base_weights(b, benchmark, q=q)
        bw = bw.reindex(C.INDUSTRIES).fillna(0.0)
        r = ret.iloc[i + 1]
        m = r.notna()
        diff = (w.reindex(C.INDUSTRIES).fillna(0.0) - bw)[m]
        cont = (diff * r[m] / 100.0) * 100          # 单位：pct
        tv = np.nan if prev is None else float((w - prev).abs().sum()) / 2.0
        rows.append(dict(季度=q, 超额pct=float(cont.sum()),
                         换手=tv))
        detail.append(cont.rename(q))
        prev = w
    if not detail:
        return dict(table=pd.DataFrame(), detail=pd.DataFrame(), base=base_lab)
    det = pd.concat(detail, axis=1)
    t = pd.DataFrame(rows).set_index("季度")
    t["换手"] = t["换手"].fillna(t["换手"].dropna().mean()
                                 if t["换手"].notna().any() else 1.0)
    t["净超额pct"] = t["超额pct"] - t["换手"] * (cost_bp / 10000.0) * 100 * 2

    tot = det.sum(axis=1).sort_values()          # 各行业累计贡献（pct）
    out = pd.DataFrame({
        "行业": tot.index,
        "累计贡献pct": tot.to_numpy(),
        "占超额比%": (tot / t["超额pct"].sum() * 100).to_numpy()
        if t["超额pct"].sum() else np.nan,
        "平均权重偏离pct": (det.index.map(
            lambda i: np.nan) if False else
            pd.Series({i: float(np.nan) for i in tot.index})).to_numpy(),
    })
    # 平均权重偏离：逐期 (w − bm) 的均值
    devs = []
    prev = None
    for i, q in enumerate(b.quarters):
        z = cz(ind[factor].loc[[q]]).iloc[0]
        if z.notna().sum() < 20 or i + 1 >= len(b.quarters):
            continue
        z = z.fillna(0.0)
        w = _cap_normalize(base0 * np.exp(-lam * z), cap)
        bw, _lab = make_base_weights(b, benchmark, q=q)
        devs.append((w.reindex(C.INDUSTRIES).fillna(0.0) - bw.reindex(C.INDUSTRIES).fillna(0.0))
                    * 100)
        prev = w
    if devs:
        out["平均权重偏离pct"] = out["行业"].map(pd.concat(devs, axis=1).mean(axis=1))
    neg = out[out["累计贡献pct"] < 0]
    pos = out[out["累计贡献pct"] > 0]
    summary = dict(
        总额外收益pct=float(t["超额pct"].sum()),
        净超额pct=float(t["净超额pct"].sum()),
        正贡献行业数=len(pos), 负贡献行业数=len(neg),
        最大负贡献行业=(neg["行业"].iloc[0] if len(neg) else None),
        最大负贡献pct=float(neg["累计贡献pct"].iloc[0]) if len(neg) else 0.0,
        最大正贡献行业=(pos["行业"].iloc[-1] if len(pos) else None),
        最大正贡献pct=float(pos["累计贡献pct"].iloc[-1]) if len(pos) else 0.0,
        # 剔除最大负贡献行业后，超额是否仍为正 → 判定"只靠避开某行业"
        剔除最大负贡献后超额pct=float(t["超额pct"].sum()
                                 - (neg["累计贡献pct"].iloc[0] if len(neg) else 0.0)),
        基准=f"{base_lab} 对比 {benchmark}",
    )
    return dict(table=out.sort_values("累计贡献pct", ascending=False),
                detail=det, period=t, summary=summary,
                base=base_lab, benchmark=benchmark)


def tilt_vs_cap_decomposition(ind: dict, b, factor: str = "F15_拥挤背离",
                              cap: float = 0.10, lam: float = 0.5,
                              bases=("equal", "hs300"),
                              cost_bp: float = 20.0) -> pd.DataFrame:
    """分离「单行业上限（cap）约束」与「因子倾斜」各自贡献的超额。

    这是组合层检验里**最容易被忽略的混淆项**：
    只要对基准施加 cap（例如把 19.25% 的持仓压到 10%），
    就已经产生了一个与因子无关的主动偏离。
    若不设 λ=0 的对照，就会把 cap 的效果误算成因子的 alpha。

    对每个起点 base：
      · λ=0  → 只有 cap 在作用（因子完全不用）
      · λ=lam → cap + 因子倾斜
      两者之差 = 因子倾斜的净增量。
    """
    from research import industry_rotation_strategy as rot
    rows = []
    for base in bases:
        r0 = rot(ind, b, factor, lam=0.0, cap=cap, base=base, cost_bp=cost_bp)
        r1 = rot(ind, b, factor, lam=lam, cap=cap, base=base, cost_bp=cost_bp)
        if not r0["perf"] or not r1["perf"]:
            continue
        e0 = r0["perf"]["超额净"]["ann"]
        e1 = r1["perf"]["超额净"]["ann"]
        rows.append(dict(
            策略起点=r0["base"], 基准=r0["base"],
            仅cap约束_超额=e0, cap加因子倾斜_超额=e1,
            因子倾斜净增量=e1 - e0,
            仅cap_IR=r0["perf"]["信息比IR"], 含因子_IR=r1["perf"]["信息比IR"],
            仅cap_换手率=r0["perf"]["平均换手率"] * 100,
            含因子_换手率=r1["perf"]["平均换手率"] * 100,
            平均最大行业权重=r1["perf"]["平均最大行业权重"]))
    return pd.DataFrame(rows)


def benchmark_cross_consistency(ind: dict, b, factor: str = "F15_拥挤背离",
                                lam: float = 0.5, cap: float = 0.10,
                                cost_bp: float = 20.0) -> pd.DataFrame:
    """把「策略起点」×「超额基准」的 4 种组合摆在一起，消除口径混淆。

    区分清楚：
      · 策略起点 = 倾斜的出发点（等权 / 沪深300）
      · 超额基准 = 计算超额时用的参照（等权 / 沪深300）
    两者不必相同；只有把四种组合都列出，才能看出"结论对口径有多敏感"。
    """
    from research import industry_rotation_strategy as rot
    rows = []
    for start in ("equal", "hs300"):
        for bm in ("equal", "hs300"):
            r = rot(ind, b, factor, lam=lam, cap=cap, base=start, cost_bp=cost_bp)
            if not r["perf"]:
                continue
            t = r["table"]
            if t.empty:
                continue
            # 基准序列按**位置**与 t 对齐：industry_rotation_strategy 的
            # weights 比 table 多最后一项（该期已建仓但无下一期收益）。
            ret = b.market["ret"].reindex(index=b.quarters, columns=C.INDUSTRIES)
            ks = list(r["weights"])
            assert len(ks) >= len(t), f"权重期数 {len(ks)} < 净值期数 {len(t)}"
            bench_vals = []
            for j in range(len(t)):
                q = ks[j]
                bw, _ = make_base_weights(b, bm, q=q)
                rn = ret.loc[ks[j + 1]]
                m = rn.notna()
                bench_vals.append(
                    float((bw.reindex(rn.index).fillna(0.0)[m] * rn[m] / 100).sum()) * 100)
            bdf = pd.Series(bench_vals, index=t.index)
            ex = (t["组合收益_净"] - bdf).dropna()
            if len(ex) < 4:
                continue
            rows.append(dict(策略起点=("等权" if start == "equal" else "沪深300逐期"),
                             超额基准=("等权" if bm == "equal" else "沪深300逐期"),
                             期数=len(ex),
                             超额净_算术=ex.mean() * 4,
                             超额净_几何=((1 + ex / 100).prod() ** (4 / len(ex)) - 1) * 100,
                             IR=(ex.mean() * 4) / (ex.std(ddof=1) * 2)
                             if ex.std(ddof=1) else np.nan,
                             超额胜率=(ex > 0).mean() * 100))
    return pd.DataFrame(rows)
