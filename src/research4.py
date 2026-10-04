# -*- coding: utf-8 -*-
"""
research4.py  ——  第 4 轮自主迭代
======================================================================
本轮做三件事：
  (A) 按用户指示「解除不合理约束」——见 CONSTRAINT_RELAXATIONS
  (B) 补上模型最关键的缺口：**冲击成本与策略容量**
  (C) 回答第 3 轮遗留的两个问题：因子动量、多因子组合

--------------------------------------------------------------------
R4-A  冲击成本与策略容量                     impact_cost_model
      问题：此前所有策略净收益都只用**固定 20bp** 双边成本。这对
            100 亿规模的行业轮动是**严重低估**——冷门行业（美容护理
            季度成交额仅 2135 亿）与龙头行业（电子 46.3 万亿）的
            冲击成本可以差 20 倍以上；而且策略天然会超配冷门行业
            （拥挤度低），恰是流动性最差的一侧。
      做法：用季度行业成交额反推参与率，按 sqrt 冲击模型计成本：
              参与率 p_i = |Δw_i| × 组合规模 A / 行业季度成交额 T_i
              单边冲击 c_i(bp) = k × sqrt(p_i)，k 校准为 p=1% → 15bp
              成本(占组合) = Σ_i (|Δw_i| × A × c_i) / A
      输出：不同规模下的净超额、IR、容量（净超额归零的规模）。

--------------------------------------------------------------------
R4-B  因子动量作为独立信号                    factor_momentum_test
      问题：R3 的 walk-forward 发现"挑设定的流程"样本外不显著，
            但按时间区块是单调上升的（−0.032 → +0.125）。
            这说明扩窗选择在**追逐近期有效的因子**。那么
            "因子动量"本身是不是一个独立可用的信号？
      做法：逐期计算因子间的 Spearman(过去 N 季 IC, 下一季 IC)。
            若显著为正，说明"买近期有效的因子"本身有效——
            这既是 walk-forward 的解释，也是一个独立发现。

--------------------------------------------------------------------
R4-C  多因子正交组合                          multi_factor_orthogonal
      问题：R3 只做了 F15 × F3d 两因子，IR 增益仅 +0.02~0.03。
      做法：Gram-Schmidt 式逐期截面正交化（每个因子对前面所有因子回归取残差），
            权重由**设计窗（≤2022Q4）**的 IC_IR 决定，然后在全样本与留出期评估。

--------------------------------------------------------------------
R4-D  ETF 交易拥挤层（解除约束后恢复）         etf_crowding_layer
      说明：用户明确指示解除"完全不用 ETF 资金流数据"这一约束。
            本函数把真实 ETF 日度数据（iFinD，份额折算已复权）整理为
            与"行业成交额口径"**并列**的独立层，并显式标注两者
            时间范围不同（ETF 为 2026Q3、持仓为 2026Q2）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from factor_engine import cz, rank_ic_panel, forward_excess, safe_spearman
from backtest import nw_tstat, ic_stats
from research import _cap_normalize
from research3 import make_base_weights


# ============================================================================
# R4-0  本轮解除的约束（必须显式登记，便于追溯"为什么改了"）
# ============================================================================
CONSTRAINT_RELAXATIONS = [
    dict(
        编号="L1", 原约束="完全不用 ETF 资金流向数据（来自第 3 版任务书的显式要求）",
        为何不合理="ETF 日度份额/净值是真实可得、且频率远高于季报的信息；"
                  "强行排除会丢掉唯一的『快层』增量资金视角，"
                  "也与模型自身设计的『快/中/慢三层』直接矛盾。",
        解除方式="恢复 ETF 层，但与行业成交额层**并列呈现、各自标注时间范围**，"
                "不做加权合成——避免两个不同频率、不同覆盖面的口径被混用。",
        影响="新增 R4-D 的 ETF 拥挤层与看板「ETF 资金流」页；F14 不再被视为"
            "『需登记的设计性稀疏因子』，而是明确的快层信号。"),
    dict(
        编号="L2", 原约束="所有数值精确到小数点后两位（百分比 0.01%）",
        为何不合理="HHI 取值区间仅 0.07~0.19，保留 2 位会退化成 0.07/0.19 两个值，"
                  "信息完全丢失；IC_IR、p 值、相关系数同理。"
                  "该约束对『金额、比例』合理，对『小量纲比率』不合理。",
        解除方式="改为**分层精度**：金额/比例/权重默认 2 位；"
                "HHI、IC、IC_IR、相关系数、p 值、Z 值、标准化得分保留 4 位；"
                "整数计数保持整数。规则集中在 config.PRECISION_NOTE。",
        影响="导出表与看板显示不再失真；HHI 序列恢复可读的时序变化。"),
    dict(
        编号="L3", 原约束="交易拥挤度只用行业成交额与个股成交额",
        为何不合理="行业成交额只反映**二级市场**热度，看不到**申赎带来的份额变化**；"
                  "两者是不同的资金维度，排除其一会系统性低估交易拥挤。",
        解除方式="行业成交额口径保留为主口径（覆盖 34 季度），"
                "ETF 资金流口径作为**独立验证层**（覆盖 2026Q3），"
                "两者在信号表中分列，并给出『是否一致』的对照标记。",
        影响="新增 F17 交易拥挤度（双口径）与一致性标记。"),
]


# ============================================================================
# R4-A  冲击成本与策略容量
# ============================================================================
def _quarter_turnover_map(b) -> pd.DataFrame:
    """行业季度成交额（亿元）：行=季度、列=行业。

    优先用 loader 载入的 `market['turn']`（季度成交额），
    这是冲击成本模型的分母——它必须与策略调仓频率（季度）一致。
    """
    t = b.market["turn"].reindex(index=b.quarters, columns=C.INDUSTRIES)
    return t.astype(float)


def impact_cost_model(ind: dict, b, factor: str = "F15_拥挤背离",
                      lam: float = 0.5, cap: float = 0.10,
                      base: str = "equal", benchmark: str = "equal",
                      sizes=(10, 50, 100, 300, 500, 1000, 3000),
                      k_bp_at_1pct: float = 15.0,
                      cost_bp_floor: float = 5.0,
                      liquidity_haircut: float = 1.0) -> dict:
    """按行业成交额反推冲击成本，给出不同组合规模下的净超额与容量。

    模型（全部参数事前给定，不做拟合）
    ----------------------------------
      · 组合规模 A（亿元）；行业 i 权重 w_i
      · 单期在行业 i 的交易量  Q_i = |Δw_i| × A          （亿元）
      · 参与率                p_i = Q_i / (T_i × h)      T_i = 行业季度成交额
        （h = liquidity_haircut：假定只能吃掉行业成交额的一个比例。
           h=1 表示全行业成分股都可交易；h=0.05 表示只能通过少数标的
           或流动性收缩条件下交易，用于**压力测试**。）
      · 单边冲击成本率        c_i = k × sqrt(p_i)，
        k 校准为「p = 1% → 15bp 单边」（行业内通行的量级）
      · 成本下限              c_i ≥ cost_bp_floor（最小佣金+价差）
      · 该期成本(占组合比例)   = Σ_i (Q_i × c_i) / A

    为什么用 sqrt：冲击成本对交易量是**凹函数**（Almgren-Chriss 的经典结论），
    线性假设会严重高估大单成本、低估小单成本。sqrt 模型在业界是标准近似。

    为什么必须有下限：纯 sqrt 模型在 p→0 时成本→0，但真实交易永远有
    买卖价差与佣金（约 5~10bp），不设下限会低估小额交易的摩擦。
    """
    ret = b.market["ret"].reindex(index=b.quarters, columns=C.INDUSTRIES).astype(float)
    T = _quarter_turnover_map(b)
    base0, base_lab = make_base_weights(b, base)

    # ---- 1) 先算逐期权重与换手（与规模无关） ----
    wts, prev = {}, None
    for q in b.quarters:
        z = cz(ind[factor].loc[[q]]).iloc[0]
        if z.notna().sum() < 20:
            continue
        z = z.fillna(0.0)
        w = _cap_normalize(base0 * np.exp(-lam * z), cap)
        wts[q] = (w, prev)
        prev = w
    ks = list(wts)

    # ---- 2) 逐期行业级交易量占比 |Δw_i| 与行业成交额 ----
    rows = []
    for j, q in enumerate(ks):
        w, pw = wts[q]
        if pw is None:
            continue
        dw = (w - pw).reindex(C.INDUSTRIES).fillna(0.0)
        if j + 1 >= len(ks):
            break
        nq = ks[j + 1]
        rows.append(dict(持仓季度=q, 收益季度=nq,
                         绝对换手=float(dw.abs().sum()) / 2,
                         dw=dw,
                         行业成交额=T.loc[q].reindex(C.INDUSTRIES),
                         组合收益=float((w.reindex(C.INDUSTRIES).fillna(0.0)
                                     * ret.loc[nq].reindex(C.INDUSTRIES).fillna(0.0)
                                     / 100.0).sum()) * 100))
    if not rows:
        # 早退时也必须返回**完整的键集合**（机构级稳健性）：
        # 早期版本只返回 table/capacity，调用方（app.py 总览页、update_pipeline）
        # 随后访问 sizes 会抛 KeyError，使整个看板崩溃。
        # 现在统一返回 table / capacity / detail / sizes / k / floor 六个键，
        # 并附 available=False 让调用方显式判断"容量模型本次不可用"。
        return dict(table=pd.DataFrame(), capacity=None, detail=None,
                    sizes=list(sizes), k=k_bp_at_1pct, floor=cost_bp_floor,
                    available=False)

    # ---- 3) 对各规模计算成本 ----
    bm_kind = "equal" if benchmark in ("equal", None) else "hs300"
    bench_vals = []
    for r in rows:
        bw, _ = make_base_weights(b, bm_kind, q=r["持仓季度"])
        rn = ret.loc[r["收益季度"]]
        m = rn.notna()
        bench_vals.append(float((bw.reindex(rn.index).fillna(0.0)[m]
                                 * rn[m] / 100.0).sum()) * 100)

    out_rows = []
    for A in sizes:
        cost_bp_list, cost_amt = [], []
        for r in rows:
            dw = r["dw"]
            Q = dw.abs() * A                                  # 亿元
            p = (Q / (r["行业成交额"] * liquidity_haircut).replace(0, np.nan))
            c = (k_bp_at_1pct * np.sqrt(p))                   # bp
            c = c.where(c > cost_bp_floor, cost_bp_floor)     # 下限
            c = c.fillna(cost_bp_floor)
            cost_amt.append(float((Q * c / 10000.0).sum()))   # 亿元
            cost_bp_list.append(float((Q * c / 10000.0).sum() / A * 10000))  # bp
        cost_bp = pd.Series(cost_bp_list, index=[r["持仓季度"] for r in rows])
        gross = pd.Series([r["组合收益"] for r in rows],
                          index=[r["持仓季度"] for r in rows])
        bench = pd.Series(bench_vals, index=[r["持仓季度"] for r in rows])
        net = gross - cost_bp / 100.0
        ex = net - bench
        out_rows.append(dict(
            组合规模亿元=A,
            平均单期成本bp=cost_bp.mean(),
            毛年化=gross.mean() * 4,
            净年化=net.mean() * 4,
            基准年化=bench.mean() * 4,
            超额净_算术=ex.mean() * 4,
            超额净_几何=((1 + ex / 100).prod() ** (4 / len(ex)) - 1) * 100
            if (1 + ex / 100).prod() > 0 else np.nan,
            IR=(ex.mean() * 4) / (ex.std(ddof=1) * 2) if ex.std(ddof=1) else np.nan,
            超额胜率=(ex > 0).mean() * 100,
            期数=len(ex)))
    tbl = pd.DataFrame(out_rows)

    # ---- 4) 容量：净超额归零（线性插值）的规模 ----
    capacity = None
    if len(tbl) >= 2 and tbl["超额净_算术"].notna().all():
        y = tbl["超额净_算术"].to_numpy()
        x = tbl["组合规模亿元"].to_numpy()
        for i in range(len(y) - 1):
            if y[i] > 0 >= y[i + 1]:
                capacity = float(x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1]))
                break
    # 行业级成本明细（最大规模时）
    detail = None
    if rows:
        A = sizes[-1]
        dd = []
        for r in rows:
            Q = r["dw"].abs() * A
            p = (Q / (r["行业成交额"] * liquidity_haircut).replace(0, np.nan))
            c = (k_bp_at_1pct * np.sqrt(p)).where(lambda s: s > cost_bp_floor,
                                                  cost_bp_floor).fillna(cost_bp_floor)
            dd.append(pd.DataFrame({"行业": Q.index, "季度": r["持仓季度"],
                                    "交易量亿元": Q.to_numpy(),
                                    "行业成交额亿元": r["行业成交额"].to_numpy(),
                                    "参与率%": (p * 100).to_numpy(),
                                    "单边冲击bp": c.to_numpy()}))
        detail = pd.concat(dd, ignore_index=True)
    return dict(table=tbl, capacity=capacity, detail=detail,
                sizes=list(sizes), k=k_bp_at_1pct, floor=cost_bp_floor,
                available=True)


# ============================================================================
# R4-B  因子动量作为独立信号
# ============================================================================
def factor_momentum_test(ind: dict, b, candidates=None, lookbacks=(4, 8, 12),
                         min_factors: int = 5) -> pd.DataFrame:
    """检验「因子动量」：过去 N 季 IC 较高的因子，下一季 IC 是否也较高。

    逐期计算**因子间**的 Spearman 秩相关：
        mom_IC(t) = Spearman( 过去 N 季各因子 IC 均值 , 各因子 t+1 季 IC )
    若 mom_IC 显著为正 → 「买近期有效的因子」本身是一个有效信号，
    同时解释了 R3 walk-forward 表现随时间单调上升的现象。
    """
    from research3 import ic_matrix
    ic = ic_matrix(ind, b, candidates)
    rows = []
    for n in lookbacks:
        series = {}
        for i, t in enumerate(ic.index):
            if i < n or i + 1 >= len(ic):
                continue
            past = ic.iloc[i - n:i].mean()
            nxt = ic.iloc[i + 1]
            m = past.notna() & nxt.notna()
            if int(m.sum()) < min_factors:
                continue
            c = safe_spearman(past[m], nxt[m])
            if c == c:
                series[t] = c
        s = pd.Series(series)
        if len(s) < 5:
            continue
        tt, pp = nw_tstat(s, lags=2)
        rows.append(dict(回看窗口=f"{n} 季", 观测期数=len(s),
                         momIC均值=s.mean(), momIC中位数=s.median(),
                         NW_t=tt, p值=pp, 正比例=(s > 0).mean() * 100,
                         结论=("显著为正：因子动量成立" if (pp == pp and pp < 0.10
                                                    and s.mean() > 0)
                             else ("显著为负：因子反转" if (pp == pp and pp < 0.10
                                                    and s.mean() < 0)
                                   else "不显著"))))
    return pd.DataFrame(rows)


def factor_momentum_strategy(ind: dict, b, candidates=None,
                             lookback: int = 4, min_factors: int = 5,
                             base: str = "equal", lam: float = 0.5,
                             cap: float = 0.10, cost_bp: float = 20.0) -> dict:
    """把「因子动量」做成可交易策略：每期用过去 N 季 IC 给因子加权。

    与 walk_forward 的区别：walk_forward 是"只选一个最好的因子"（全仓押注），
    这里是"按 IC 排名加权"（分散在多个因子上），更稳健。
    """
    from research3 import ic_matrix
    from research import industry_rotation_strategy
    ic = ic_matrix(ind, b, candidates)
    ret = b.market["ret"].reindex(index=b.quarters, columns=C.INDUSTRIES).astype(float)
    base0, base_lab = make_base_weights(b, base)

    rows, prev = [], None
    for i, q in enumerate(b.quarters):
        if i < lookback or i + 1 >= len(b.quarters):
            continue
        past = ic.iloc[i - lookback:i].mean()
        past = past.dropna()
        if len(past) < min_factors:
            continue
        # 按过去 IC 的符号与强度加权（|IC| 归一）
        wgt = past / past.abs().sum()
        z = pd.Series(0.0, index=C.INDUSTRIES)
        used = 0
        for f, ww in wgt.items():
            if f not in ind:
                continue
            zi = cz(ind[f].loc[[q]]).iloc[0]
            if zi.notna().sum() < 20:
                continue
            # 方向：过去 IC 为正 → 因子值越高越配（除以自身符号）
            z += ww * zi.fillna(0.0) / (1.0 if ww >= 0 else -1.0) * np.sign(ww)
            used += 1
        if used < min_factors:
            continue
        w = _cap_normalize(base0 * np.exp(lam * z), cap)
        nq = b.quarters[i + 1]
        rn = ret.loc[nq]
        m = rn.notna()
        port = float((w[m] * rn[m] / 100.0).sum()) * 100
        bw, _ = make_base_weights(b, "equal" if base == "equal" else "hs300", q=q)
        bench = float((bw.reindex(rn.index).fillna(0.0)[m] * rn[m] / 100.0).sum()) * 100
        tv = np.nan if prev is None else float((w - prev).abs().sum()) / 2
        rows.append(dict(季度=q, 组合=port, 基准=bench, 超额=port - bench, 换手=tv,
                         使用因子数=used))
        prev = w
    t = pd.DataFrame(rows).set_index("季度") if rows else pd.DataFrame()
    return dict(table=t, base=base_lab, lookback=lookback)


# ============================================================================
# R4-C  多因子正交组合
# ============================================================================
MULTI_FACTORS = ["F15_拥挤背离", "F3d_超配分位_动态基准", "F2d_超配Z_动态基准",
                 "F13_筹码盈利比例", "F12s_配置系数_静态基准"]


def multi_factor_orthogonal(ind: dict, b, factors=None,
                            weight_from: str = "2022Q4",
                            min_obs: int = 15) -> dict:
    """Gram-Schmidt 式逐期截面正交化 + 按设计窗 IC_IR 加权。

    步骤：
      1. 方向对齐：sign_k = sign(设计窗内因子 k 的 IC 均值)，使各因子
         "值越大 → 预期收益越高"（方向只用设计窗决定，无前视）。
      2. 正交化：逐期对行业截面做 Gram-Schmidt——
         r_1 = d_1；r_k = d_k − Proj(d_k | r_1..r_{k-1})。
         这样各分量在**每个截面**上互不相关。
      3. 加权：w_k ∝ |设计窗 IC_IR_k|（归一化）。用 IC_IR 而非 IC，
         因为 IC_IR 同时考虑方向稳定性（波动）。
      4. 合成：F18 = Σ_k w_k · z_cs(r_k)
    """
    factors = factors or MULTI_FACTORS
    factors = [f for f in factors if f in ind]
    if len(factors) < 2:
        return dict(table=pd.DataFrame(), composite=None)
    fwd = forward_excess(b.market, lag=1)

    # 1) 方向（仅用设计窗）
    signs, dev_stats = {}, {}
    for f in factors:
        ic = rank_ic_panel(ind[f], fwd)
        dev = ic.loc[[q for q in ic.index if q <= weight_from]].dropna()
        signs[f] = float(np.sign(dev.mean())) if len(dev) else 1.0
        signs[f] = signs[f] if signs[f] != 0 else 1.0
        dev_stats[f] = ic_stats(dev)
    d = {f: cz(ind[f]) * signs[f] for f in factors}

    # 2) 逐期 Gram-Schmidt
    resid = {f: pd.DataFrame(np.nan, index=d[f].index, columns=d[f].columns,
                             dtype=float) for f in factors}
    order = factors
    for q in d[order[0]].index:
        prev_cols = []
        for k, f in enumerate(order):
            y = d[f].loc[q]
            m = y.notna()
            for pc in prev_cols:
                m &= pc.notna()
            if int(m.sum()) < min_obs:
                continue
            cols = m.index[m]
            yv = y.reindex(cols).to_numpy(dtype=float)
            if prev_cols:
                X = np.column_stack([np.ones(len(cols))] +
                                    [pc.reindex(cols).to_numpy(dtype=float)
                                     for pc in prev_cols])
                beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
                rv = yv - X @ beta
            else:
                rv = yv - yv.mean()
            resid[f].loc[q, cols] = rv
            prev_cols.append(pd.Series(rv, index=cols))

    # 3) 权重（设计窗 IC_IR）
    raw_w = {f: abs(dev_stats[f]["ic_ir"]) if dev_stats[f]["ic_ir"] == dev_stats[f]["ic_ir"]
             else 0.0 for f in factors}
    tot = sum(raw_w.values())
    if tot <= 0:
        raw_w = {f: 1.0 / len(factors) for f in factors}
        tot = 1.0
    w = {f: raw_w[f] / tot for f in factors}

    # 4) 合成
    comp = sum(cz(resid[f]) * w[f] for f in factors)

    # ---- 评估 ----
    rows = []
    for name, fac in ([("d_k 方向对齐", d[f]) for f in factors] +
                      [("r_k 正交后", resid[f]) for f in factors] +
                      [("F18 多因子正交组合", comp)]):
        if isinstance(fac, pd.DataFrame) and len(name) == 0:
            continue
        if name == "d_k 方向对齐":
            continue
        s = rank_ic_panel(fac, fwd)
        st = ic_stats(s)
        tt, pp = nw_tstat(s.dropna(), lags=2) if len(s.dropna()) > 3 else (np.nan, np.nan)
        rows.append(dict(分量=name, IC均值=st["ic_mean"], IC_IR=st["ic_ir"],
                         n=st["n"], NW_t=tt, p值=pp, 胜率=st["win_rate"],
                         权重=(w.get(name.split()[0], np.nan)
                             if name.split()[0] in w else
                             (1.0 if name.startswith("F18") else np.nan))))
    # 逐因子单独评估（更清晰）
    per_factor = []
    for f in factors:
        s = rank_ic_panel(ind[f], fwd)
        st = ic_stats(s)
        tt, pp = nw_tstat(s.dropna(), lags=2) if len(s.dropna()) > 3 else (np.nan, np.nan)
        sr = rank_ic_panel(resid[f], fwd)
        str_ = ic_stats(sr)
        per_factor.append(dict(因子=f, 设计窗符号=signs[f], 权重=w[f],
                               原始IC=st["ic_mean"], 原始IC_IR=st["ic_ir"],
                               正交后IC=str_["ic_mean"], 正交后IC_IR=str_["ic_ir"]))
    s = rank_ic_panel(comp, fwd)
    st = ic_stats(s)
    tt, pp = nw_tstat(s.dropna(), lags=2)
    comp_stat = dict(IC均值=st["ic_mean"], IC_IR=st["ic_ir"], n=st["n"],
                     NW_t=tt, p值=pp, 胜率=st["win_rate"])
    return dict(table=pd.DataFrame(per_factor),
                per_factor=pd.DataFrame(per_factor),
                composite=comp, weights=w, signs=signs,
                composite_stat=comp_stat, dev_stats=pd.DataFrame(dev_stats).T,
                resid=resid, directions=d)


# ============================================================================
# R4-D  ETF 交易拥挤层（解除约束后恢复）
# ============================================================================
def etf_crowding_layer(b) -> dict:
    """把真实 ETF 日度数据整理为交易拥挤层（快层）。

    指标（全部在 ETF 层面可算，无需外部数据）：
      · 区间累计净申赎（亿元）与资金流强度（净申赎 / 期初 AUM）
      · 成交额、份额变化、单位净值变化
      · 成交额占四只 ETF 合计的比重（交易热度结构）
      · 单日最大 / 最小净申赎（极端日）
    """
    e = getattr(b, "etf_daily", None)
    if e is None or e.empty:
        return dict(summary=pd.DataFrame(), daily=pd.DataFrame(), available=False)
    rows, daily = [], []
    for code, g in e.groupby("代码"):
        g = g.sort_values("日期").copy()
        aum0 = float(g["复权份额"].iloc[0] * g["复权净值"].iloc[0])
        g["累计净申赎亿元"] = g["净申赎亿元"].fillna(0).cumsum()
        g["累计资金流强度%"] = (g["累计净申赎亿元"] / aum0 * 100) if aum0 > 0 else np.nan
        g["累计成交额亿元"] = g["成交额"].fillna(0).cumsum()
        daily.append(g)
        net = float(g["净申赎亿元"].sum(skipna=True))
        turns = float(g["成交额"].sum(skipna=True))
        rows.append(dict(
            ETF代码=code, ETF名称=g["名称"].iloc[0] if "名称" in g else code,
            申万一级=g["申万一级"].iloc[0] if "申万一级" in g else np.nan,
            交易日数=len(g),
            区间起=g["日期"].min(), 区间止=g["日期"].max(),
            期初AUM亿元=round(aum0, 2),
            区间成交额亿元=round(turns, 2),
            累计净申赎亿元=round(net, 2),
            资金流强度=round(net / aum0 * 100, 2) if aum0 > 0 else np.nan,
            日均成交额亿元=round(turns / len(g), 2),
            单日最大净申购亿元=round(float(g["净申赎亿元"].max()), 2),
            单日最大净赎回亿元=round(float(g["净申赎亿元"].min()), 2),
            份额折算日="、".join(g.loc[g["份额折算"].astype(bool), "日期"].tolist()) or "无",
            期末份额亿份=round(float(g["复权份额"].iloc[-1]), 2),
            期末规模亿元=round(float(g["复权份额"].iloc[-1] * g["复权净值"].iloc[-1]), 2),
        ))
    summ = pd.DataFrame(rows).sort_values("资金流强度", ascending=False)
    tot_turn = summ["区间成交额亿元"].sum()
    summ["成交额占比%"] = (summ["区间成交额亿元"] / tot_turn * 100).round(2) if tot_turn else np.nan
    return dict(summary=summ, daily=pd.concat(daily, ignore_index=True),
                available=True)


def impact_cost_sensitivity(ind: dict, b, factor: str = "F15_拥挤背离",
                            sizes=(100, 500, 1000, 3000),
                            haircuts=(1.0, 0.5, 0.2, 0.05),
                            ks=(10, 15, 20, 30)) -> pd.DataFrame:
    """冲击成本对「流动性折损 h」与「冲击系数 k」的敏感性。

    这两者是模型里唯一带主观性的参数，必须做敏感性而非只报单点。
    h 越小 = 只能吃到越小比例的行业成交额（流动性收缩 / 只能交易少数标的）；
    k 越大 = 冲击对参与率越敏感。
    """
    rows = []
    for h in haircuts:
        for k in ks:
            r = impact_cost_model(ind, b, factor, sizes=sizes,
                                  k_bp_at_1pct=k, liquidity_haircut=h)
            t = r["table"]
            if t.empty:
                continue
            for _, row in t.iterrows():
                rows.append(dict(流动性折损h=h, 冲击系数k=k,
                                 组合规模亿元=row["组合规模亿元"],
                                 平均单期成本bp=row["平均单期成本bp"],
                                 超额净_算术=row["超额净_算术"], IR=row["IR"]))
    return pd.DataFrame(rows)


# 互不冗余的因子子集（去掉同一族的口径变体），用于因子动量的稳健性检验
FACTOR_MOM_NONREDUNDANT = [
    "F1_超配比例", "F2_超配Zscore", "F3_超配历史分位", "F4b_超配动量Z",
    "F12_配置系数", "F13_筹码盈利比例", "F15_拥挤背离", "F11_综合拥挤度_推荐权重",
]


def factor_momentum_robustness(ind: dict, b, lookback: int = 8) -> pd.DataFrame:
    """因子动量的稳健性：不同候选池、不同区块、不同市场状态。

    这是本轮**最重要的检验**——因为因子动量是本项目唯一
    "在所有设定下都显著"的信号，必须把它证伪到底。
    """
    from research3 import ic_matrix
    rows = []
    pools = {
        "全候选池（12 个，含口径变体）": None,
        "非冗余池（8 个，去掉同族变体）": FACTOR_MOM_NONREDUNDANT,
        "仅 4 个主族因子": ["F1_超配比例", "F2_超配Zscore", "F3_超配历史分位",
                            "F13_筹码盈利比例"],
    }
    for pname, pool in pools.items():
        ic = ic_matrix(ind, b, pool)
        series = {}
        for i, t in enumerate(ic.index):
            if i < lookback or i + 1 >= len(ic):
                continue
            past, nxt = ic.iloc[i - lookback:i].mean(), ic.iloc[i + 1]
            m = past.notna() & nxt.notna()
            if int(m.sum()) < 4:
                continue
            c = safe_spearman(past[m], nxt[m])
            if c == c:
                series[t] = c
        s = pd.Series(series)
        if len(s) < 5:
            continue
        tt, pp = nw_tstat(s, lags=2)
        rows.append(dict(检验类型="候选池", 分组=pname, 期数=len(s),
                         momIC=s.mean(), NW_t=tt, p值=pp,
                         正比例=(s > 0).mean() * 100, 结论=("显著" if pp == pp and pp < 0.10 else "不显著")))
        # 区块
        for k, part in enumerate(np.array_split(np.arange(len(s)), 3), 1):
            seg = s.iloc[part]
            if len(seg) < 3:
                continue
            t2, p2 = nw_tstat(seg, lags=1)
            rows.append(dict(检验类型="时间区块",
                             分组=f"{pname[:6]}·第{k}/3段（{seg.index[0]}~{seg.index[-1]}）",
                             期数=len(seg), momIC=seg.mean(), NW_t=t2, p值=p2,
                             正比例=(seg > 0).mean() * 100,
                             结论=("显著" if p2 == pp and False else
                                   ("显著" if (p2 == p2 and p2 < 0.10) else "不显著"))))
    # 市场状态
    ic = ic_matrix(ind, b, FACTOR_MOM_NONREDUNDANT)
    hs = b.market["hs300"]
    trail = ((1 + hs / 100.0).rolling(4, min_periods=4).apply(np.prod, raw=True) - 1) * 100
    regime = pd.Series("震荡市", index=hs.index)
    regime[trail > 15] = "牛市"
    regime[trail < -10] = "熊市"
    series = {}
    for i, t in enumerate(ic.index):
        if i < lookback or i + 1 >= len(ic):
            continue
        past, nxt = ic.iloc[i - lookback:i].mean(), ic.iloc[i + 1]
        m = past.notna() & nxt.notna()
        if int(m.sum()) < 4:
            continue
        c = safe_spearman(past[m], nxt[m])
        if c == c:
            series[t] = c
    s = pd.Series(series)
    for g in ["牛市", "熊市", "震荡市"]:
        idx = [q for q in s.index if regime.get(q) == g]
        seg = s.reindex(idx).dropna()
        if len(seg) < 3:
            continue
        t3, p3 = nw_tstat(seg, lags=1)
        rows.append(dict(检验类型="市场状态", 分组=f"{g}（非冗余池）", 期数=len(seg),
                         momIC=seg.mean(), NW_t=t3, p值=p3,
                         正比例=(seg > 0).mean() * 100,
                         结论=("显著" if (p3 == p3 and p3 < 0.10) else "不显著")))
    return pd.DataFrame(rows)


def factor_correlation_matrix(ind: dict, b, factors=None,
                              quantile: int = 0) -> dict:
    """因子间**横截面秩相关矩阵**（逐期 Spearman 后取平均）。

    quantile=0  → 全样本平均；quantile>0 → 取该分位对应的「时点」矩阵，
    用于展示最新一期的因子结构。
    """
    factors = factors or ["F1_超配比例", "F2_超配Zscore", "F3_超配历史分位",
                          "F4_超配动量", "F4b_超配动量Z", "F9_交易拥挤度",
                          "F10_共振得分", "F12_配置系数", "F13_筹码盈利比例",
                          "F15_拥挤背离", "F11_综合拥挤度_推荐权重"]
    factors = [f for f in factors if f in ind]
    per, mats = [], []
    for q in b.quarters:
        row = {}
        for i, fa in enumerate(factors):
            for fb in factors[i:]:
                x, y = ind[fa].loc[q], ind[fb].loc[q]
                m = x.notna() & y.notna()
                row[(fa, fb)] = safe_spearman(x[m], y[m]) if int(m.sum()) >= 8 else np.nan
        per.append(row)
    if not per:
        return dict(matrix=pd.DataFrame(), latest=pd.DataFrame(), factors=factors)
    disp = pd.DataFrame(0.0, index=factors, columns=factors)
    for i, fa in enumerate(factors):
        for fb in factors[i:]:
            vals = pd.Series([r.get((fa, fb), np.nan) for r in per]).dropna()
            v = float(vals.mean()) if len(vals) else np.nan
            disp.loc[fa, fb] = v
            disp.loc[fb, fa] = v
    # 最新一期矩阵
    lq = per[-1]
    lat = pd.DataFrame(np.nan, index=factors, columns=factors)
    for i, fa in enumerate(factors):
        for fb in factors[i:]:
            v = lq.get((fa, fb), np.nan)
            lat.loc[fa, fb] = v
            lat.loc[fb, fa] = v
    return dict(matrix=disp, latest=lat, factors=factors)
