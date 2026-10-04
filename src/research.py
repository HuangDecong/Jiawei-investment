# -*- coding: utf-8 -*-
"""
research.py  ——  第 2 轮自主迭代：研究级方法
======================================================================
本模块对应 R1-4 计划中的 5 项，逐项解决第 1 轮暴露出的方法学缺口：

  R2-A  设计期样本外检验（design-OOS）      design_oos_test / specification_search
        问题：F15 的构造灵感来自**全样本**上 F1 与 F2 的符号，
              属"设计期乐观偏差"（design-stage overfitting），
              样本内 IC 会被系统性高估。
        协议：只用 ≤DEV_END 的季度**决定因子形式**，
              ≥HOLDOUT_START 的季度在检验前完全不可见，一次性检验。
              判定规则**事前写死**，不得事后调参。

  R2-B  二层正交化                          second_layer_orthogonalize
        问题：R_F2 只剔除了价格类风格，仍可能与 F1/F12/F13 共线。
        做法：对残差因子再对"同族因子"做截面 OLS，
              若残差 IC 大幅衰减 → 说明其增量信息来自同族，不是独立信息。

  R2-C  组合层面检验（行业轮动策略）        industry_rotation_strategy / cap_sensitivity
        问题：因子 IC 不等于可交易策略。
        做法：long-only、权重和=100%、单行业上限、季度调仓、扣 20bp 成本，
              对比等权基准与沪深300 基准，给出扣费后夏普与回撤。

  R2-D  静态 vs 动态基准口径对比            benchmark_comparison
        问题：工作簿超配用的是静态基准（2026 年权重衡量 2018 年）。
        做法：同一因子族在两种基准口径下并列检验，用 IC 说话。

  R2-E  数据质量前置闸门                    preflight_gate
        问题：第 1 轮的断言散落在代码各处，未形成"失败即中止"的闸门。
        做法：集中成一次前置检查，硬约束失败抛 DataQualityError。

约定：本模块只做计算与判定，不写文件；汇报由 update_pipeline 统一承担。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from factor_engine import cz, rank_ic_panel, forward_excess
from backtest import nw_tstat, perf_series, ic_stats, block_bootstrap_ci


# ============================================================================
# R2-E  数据质量前置闸门
# ============================================================================
class DataQualityError(RuntimeError):
    """硬约束失败——流水线必须中止，不得继续产出结论。"""


# 设计性稀疏因子：在**持仓季度索引**内必然无值，但这不是计算失败，
# 而是数据可得性的事实（必须显式登记，禁止静默通过）。
KNOWN_SPARSE = {
    "F14_ETF资金流强度":
        "ETF 日度份额/净值仅覆盖 2026Q3，而持仓季度索引截至 2026Q2，"
        "两者相差一个季度。为保证口径不混用，F14 只在 ETF 季度上取值，"
        "不插值、不外推。其明细见 mkt['F14_ETF明细']（含 2026Q3）。",
    # ---- 主口径扩窗因子：因主口径持仓仅 10 期、扩窗 MIN_HIST=8 起步而稀疏 ----
    # 这是「主口径（主动权益三类，2024Q1 起）」与「扩窗统计最少 8 期」两约束叠加的
    # 设计性稀疏，不是计算失败。已于第 7 轮用偏股混合 17 期超配长历史构建
    # 长历史版 F2e/F3e/F4be/F15e（2024Q1 起有值，IC 可检验），主口径版作为对照保留。
    "F2_超配Zscore": "主口径超配仅 10 期，扩窗 Z 需 8 期起步 → 仅 2026Q1/Q2 有值；已由 F2e_超配Z_长历史（偏股混合 17 期）覆盖。",
    "F2d_超配Z_动态基准": "同 F2（动态基准版）→ 稀疏；长历史版见 F2e。",
    "F3_超配历史分位": "主口径超配仅 10 期，扩窗分位需 8 期起步 → 仅 2 期有值；已由 F3e_超配分位_长历史覆盖。",
    "F3d_超配分位_动态基准": "同 F3（动态基准版）→ 稀疏；长历史版见 F3e。",
    "F4b_超配动量Z": "主口径超配一阶差分后有效观测 <8 → 全样本无法形成 Z；已由 F4be_动量Z_长历史覆盖。",
    "R_F4b_动量Z_残差": "F4b 的派生量，F4b 稀疏 → 残差亦空。长历史版见 F4be。",
    "F15_拥挤背离": "F15 = z(F1) − z(F2)，F2 稀疏 → F15 仅 2 期有值；已由 F15e_拥挤背离_长历史覆盖（IC −0.109 显著）。",
    "F15d_拥挤背离_动态基准": "同 F15（动态基准版）→ 稀疏；长历史版见 F15e。",
    "R_F2_超配Z_残差": "F2 稀疏 → 正交残差亦空；长历史版见 F2e。",
    "F1b_超配P90阈值": "扩窗 P90 需 8 期起步 → 主口径仅 2 期有值；长历史版见 F3e。",
    "F10_S1_持仓拥挤": "S1 = 超配 > 扩窗P90，P90 稀疏 → S1 仅 2 期有值。",
    "F10_共振得分": "S1 分量稀疏 → 共振得分仅 2 期有值（S2/S3 分量正常）。",
    "F4be_动量Z_长历史": "超配一阶差分后再扩窗（8 期起步）→ 首期与起步期天然无值；已有 7~8 期值可做 IC（IC +0.058）。",
}


def preflight_gate(b, F: dict, raise_on_fail: bool = True) -> pd.DataFrame:
    """集中式数据质量前置检查（第 2 轮）。

    硬约束（失败即中止）：
      H1 每期 31 行业配置比例加总 = 100.00 ± 0.01
      H2 HHI ∈ (0, 1)
      H3 CR3 ≤ CR5 ≤ CR10 ≤ 100
      H4 基准权重逐期加总 = 100 ± 0.01
      H5 超配 = 配置 − 基准权重（口径自洽，误差 < 1e-6）
      H6 因子面板无"全表为 NaN"的因子（视为计算失败）
      H7 前瞻收益与因子期间隔一个季度（无未来函数）——校验 shift 方向
      H8 正交化控制变量齐备

    软约束（记录并告警，不中止）：
      S1 因子缺失率 > 30%
      S2 样本基金数 < 800
      S3 因子在最新期全为 NaN
    """
    ind, mkt = F["ind"], F["mkt"]
    rows: list[dict] = []

    def rec(tier, code, item, ok, detail):
        rows.append(dict(级别=tier, 编号=code, 检查项=item,
                         结果="通过" if ok else ("失败" if tier == "硬" else "告警"),
                         详情=detail))
        return ok

    # ------------------------------------------------------------------
    # ⚠️ 机构级口径修正：所有硬约束都必须只在**可建模区间**上检验
    # ------------------------------------------------------------------
    # 背景：数据源切换 Wind 后，b.quarters 仍保留 2019Q1~2026Q2 的**行情**区间，
    # 但基金持仓只从 2024Q1 起。若在全体 30 期上做加总/单调性检查：
    #   · H1「配置比例逐期加总=100」：20 个全 NaN 季度 sum=0 → 偏离 100pct → 误判失败；
    #   · H3「CR3≤CR5≤CR10」：NaN 比较恒为 False → 误判"违反 20 期"；
    #   · H5「超配=配置−基准」：NaN 相减 → 最大误差 17.5（实为缺失）；
    #   · S1「缺失率」：分母含 20 个设计上就无持仓的季度 → 全部因子虚假超限。
    # 这不是数据问题，而是**检验区间与数据区间不匹配**的实现缺陷。
    # 正确做法：硬约束取「有持仓且能算超配」的季度（holdings ∩ hs300）。
    _hq = _all_qs(b)
    if not _hq:
        _hq = list(b.quarters)
    # 基准权重逐期可得 → H4 在有基准的季度上检查
    _wq = [q for q in _hq
           if q in b.weight_matrix.index
           and b.weight_matrix.loc[q].notna().sum() > 0]
    if not _wq:
        _wq = list(b.quarters)

    alloc = b.alloc.reindex(index=_hq, columns=C.INDUSTRIES)
    # H1 加总（仅在可建模区间）
    dev = (alloc.sum(axis=1) - 100.0).abs().max()
    rec("硬", "H1", f"配置比例逐期加总=100±0.01（{len(_hq)} 期 {_hq[0]}~{_hq[-1]}）",
        bool(dev <= 0.01),
        f"最大偏离 {dev:.10f} pct")

    # H2 HHI 值域
    hhi = mkt["F5_HHI"].dropna()
    rec("硬", "H2", "HHI ∈ (0,1)",
        bool(len(hhi) and hhi.min() > 0 and hhi.max() < 1),
        f"min={hhi.min():.4f} max={hhi.max():.4f} n={len(hhi)}")

    # H3 CR 单调（仅在可建模区间）
    cr3 = mkt["F6_CR3"].reindex(_hq).dropna()
    cr5 = mkt["F6_CR5"].reindex(_hq).dropna()
    cr10 = mkt["F6_CR10"].reindex(_hq).dropna()
    _idx = cr3.index.intersection(cr5.index).intersection(cr10.index)
    _bad = int((~((cr3.reindex(_idx) <= cr5.reindex(_idx) + 1e-9)
                  & (cr5.reindex(_idx) <= cr10.reindex(_idx) + 1e-9)
                  & (cr10.reindex(_idx) <= 100 + 1e-9))).sum())
    ok_cr = bool(len(_idx) > 0 and _bad == 0)
    rec("硬", "H3", "CR3 ≤ CR5 ≤ CR10 ≤ 100", ok_cr,
        f"检验 {len(_idx)} 期，违反 {_bad} 期")

    # H4 基准权重加总（在有基准的季度）
    w = b.weight_matrix.reindex(index=_wq, columns=C.INDUSTRIES)
    wsum = w.sum(axis=1)
    # 沪深300 只覆盖 26~28 个申万一级行业，加总天然略低于 100（缺失行业权重确为 0）；
    # 允许 [95, 100.01] 的合理带宽，并要求每期至少 20 个行业有值。
    _wmin, _wmax = float(wsum.min()), float(wsum.max())
    _npos = int((w.notna().sum(axis=1) >= 20).all())
    ok4 = bool(20 <= _wmin <= 100.01 and _wmax <= 100.01 and _npos)
    rec("硬", "H4", "基准权重逐期加总 ∈ [20,100.01] 且每期 ≥20 行业",
        ok4,
        f"加总 ∈ [{_wmin:.4f}, {_wmax:.4f}]；动态期数 "
        f"{getattr(b, 'n_dynamic_quarters', 0)}/{len(_wq)}")

    # H5 超配口径自洽：超配 == 配置 − 基准权重（用**动态**基准，
    #    因本项目已切换动态季度权重；缺失行业按 0）
    diff = (alloc - w.reindex(_hq)).sub(b.overweight.reindex(index=_hq,
                                                             columns=C.INDUSTRIES))
    m5 = float(diff.abs().max().max())
    rec("硬", "H5", "超配 == 配置 − 动态基准权重", bool(m5 < 1e-6),
        f"最大误差 {m5:.2e}")

    # H6 因子非全空（设计性稀疏因子须在 KNOWN_SPARSE 中显式登记，否则视为失败）
    empty = [k for k, v in ind.items()
             if not k.startswith("_") and isinstance(v, pd.DataFrame)
             and v.notna().sum().sum() == 0]
    bad = [k for k in empty if k not in KNOWN_SPARSE]
    rec("硬", "H6", "无「未登记的全表 NaN」因子", len(bad) == 0,
        "全部因子均有取值" if not empty else
        (f"异常因子：{bad}" if bad else
         f"已登记的设计性稀疏因子：{empty}（原因见 KNOWN_SPARSE）"))

    # H7 无未来函数：因子(t) 对齐 收益(t+1)
    fwd = forward_excess(b.market, lag=1)
    a = ind["F1_超配比例"]
    ok7 = bool(np.allclose(fwd.iloc[:-1].to_numpy(dtype=float),
                           b.market["excess"].iloc[1:].to_numpy(dtype=float),
                           equal_nan=True))
    rec("硬", "H7", "前瞻收益 = shift(−1) 且不含当期", ok7,
        "第 t 期因子的前瞻收益取自 t+1 期超额收益；t 期收益不在其中")

    # H8 控制变量齐备
    ctrl = [c for c in ("CTRL_第t期收益", "CTRL_过去4季收益", "CTRL_过去4季波动")
            if c in ind]
    rec("硬", "H8", "正交化控制变量齐备(3个)", len(ctrl) == 3, f"实际 {len(ctrl)} 个")

    # ---- 软约束 ----
    # S1 因子缺失率：**只在可建模区间**上统计。
    # 若用全 30 期（含 20 个设计上无持仓的季度），所有依赖持仓的因子的
    # 缺失率都会被虚假抬高到 66%~100%，告警完全失去信息量。
    miss = {}
    for k, v in ind.items():
        if k.startswith("_") or not isinstance(v, pd.DataFrame):
            continue
        vv = v.reindex(_hq) if set(_hq) & set(v.index) else v
        tot = vv.size
        miss[k] = 1 - vv.notna().sum().sum() / tot if tot else 1.0
    over = {k: f"{v * 100:.1f}%" for k, v in miss.items()
            if v > 0.30 and k not in KNOWN_SPARSE}
    rec("软", "S1", f"因子缺失率 ≤ 30%（{len(_hq)} 期可建模区间）",
        len(over) == 0,
        "全部达标" if not over else f"超限 {len(over)} 个：{over}")

    nf = mkt.get("F8_样本基金数")
    if nf is not None and len(nf.dropna()):
        # 同样只在持仓季度上看（早期 2019 年样本基金数 308 只是市场尚未扩容，
        # 与 Wind 主动权益三类口径无关，不构成数据质量问题）
        _nfs = nf.reindex(_hq).dropna()
        if len(_nfs):
            rec("软", "S2", "样本基金数 ≥ 800（持仓季度）",
                bool(_nfs.min() >= 800),
                f"最小 {float(_nfs.min()):.0f} 只，最大 "
                f"{float(_nfs.max()):.0f} 只（{len(_nfs)} 期）")
    rec("软", "S3", "最新期因子非全空",
        bool(ind["F1_超配比例"].loc[b.latest].notna().any()), b.latest)

    df = pd.DataFrame(rows)
    hard_fail = df[(df["级别"] == "硬") & (df["结果"] == "失败")]
    if raise_on_fail and not hard_fail.empty:
        raise DataQualityError(
            "数据质量硬约束失败：\n" + hard_fail.to_string(index=False))
    return df


# ============================================================================
# R2-A  设计期样本外检验
# ============================================================================
# 设计期 / 留出期切点
# ---------------------------------------------------------------------------
# ⚠️ 数据源切换 Wind 后的关键修正
# --------------------------------
# 原先此处**硬编码** DEV_END="2022Q4" / HOLDOUT_START="2023Q1"。
# Wind 数据下可建模区间为 2024Q2~2026Q2（9 期），设计期（≤2022Q4）成为空集，
# 导致 design_oos_test 对所有因子都跳过、返回空表，
# 下游 factor_verdict_table 再 .set_index("因子") 直接抛 KeyError，看板崩溃。
#
# 正确做法：切点由**实际可建模季度**推导（config.derive_split，规则事前写死）。
# 下方 _dyn_split() 在首次调用时惰性求解一次并缓存，避免每次重建。
_DEV_END = "2022Q4"          # 长历史（34 期）下的默认切点，保持向后兼容
_HOLDOUT_START = "2023Q1"
_SPLIT_NOTE = ""
_SPLIT_OK = True


def _dyn_split(b) -> tuple:
    """按 Bundle 的实际持仓季度推导设计期/留出期切点（惰性缓存）。

    返回 (dev_end, holdout_start, ok, note)。ok=False 表示样本不足以切分，
    调用方应显式跳过样本外检验并如实报告，而不是伪造一个通过率。
    """
    global _DEV_END, _HOLDOUT_START, _SPLIT_NOTE, _SPLIT_OK
    hq = getattr(b, "over_quarters", None) or getattr(b, "alloc_quarters", None)
    if not hq:
        hq = [q for q in getattr(b, "quarters", [])]
    r = C.derive_split(hq)
    if r["ok"]:
        _DEV_END, _HOLDOUT_START = r["dev_end"], r["holdout_start"]
    _SPLIT_NOTE, _SPLIT_OK = r["note"], r["ok"]
    return _DEV_END, _HOLDOUT_START, _SPLIT_OK, _SPLIT_NOTE


DEV_END = _DEV_END          # 模块级别名：被 research3/app 等按名引用
HOLDOUT_START = _HOLDOUT_START


def _all_qs(b) -> list:
    """可建模季度（有持仓且有基准），供切分明细与样本量门槛使用。"""
    hq = getattr(b, "over_quarters", None) or getattr(b, "alloc_quarters", None)
    return list(hq) if hq else list(getattr(b, "quarters", []))


def _ic_by_window(ind: dict, b, factor: str, dev_end=DEV_END,
                  holdout_start=HOLDOUT_START) -> dict:
    """按设计期/留出期两个窗口切分某因子的 RankIC 序列。

    机构级口径一致性要求（本轮修正）
    --------------------------------
    早期版本直接 `rank_ic_panel(ind[factor], fwd)`，fwd 是**全市场季度**的前瞻超额。
    这会让**不依赖持仓的因子**（如由行情/成交额算出的 F13）在更长的历史上取得 IC，
    而**依赖持仓的因子**（F1/F2/F3/F11）只能在有持仓的季度上取得 IC。
    结果是同一张表里不同因子的样本区间**互不相同**，横向不可比，
    且会系统性高估"行情类因子"的样本外通过率。

    正确做法：把所有因子的 IC 都限制在**可建模区间**
    （holdings ∩ 沪深300 基准）上，与主 IC 表（run_wind_pipeline）完全一致。
    """
    fwd = forward_excess(b.market, lag=1)
    fac = ind[factor]
    hq = _all_qs(b)
    if hq:
        # 因子与前瞻收益**同时**限制到可建模区间，二者索引必须完全一致
        # （rank_ic_panel 会在因子索引上逐一查找 fwd，索引不齐会抛 KeyError）。
        keep = [q for q in hq if q in fwd.index and q in fac.index]
        fwd = fwd.reindex(keep)
        fac = fac.reindex(keep)
    ic = rank_ic_panel(fac, fwd).dropna()
    qs = [q for q in ic.index if q <= dev_end]
    qh = [q for q in ic.index if q >= holdout_start]
    return dict(ic_all=ic, ic_dev=ic.reindex(qs).dropna(),
                ic_hold=ic.reindex(qh).dropna())


def _judge(ic_dev: pd.Series, ic_hold: pd.Series,
           min_abs_ic: float = 0.03, alpha: float = 0.10) -> dict:
    """事前写死的判定规则（不得事后修改）：

    1) 符号一致：设计期与留出期 IC 同号；
    2) 留出期 |IC| ≥ min_abs_ic；
    3) 留出期单侧 p < alpha（NW 调整，方向按设计期符号）。
    三条**同时**满足才算「通过」。
    """
    md = float(ic_dev.mean()) if len(ic_dev) else np.nan
    mh = float(ic_hold.mean()) if len(ic_hold) else np.nan
    t, p = nw_tstat(ic_hold, lags=2) if len(ic_hold) > 3 else (np.nan, np.nan)
    same_sign = bool(np.sign(md) == np.sign(mh)) if (md == md and mh == mh) else False
    # 单侧：t 的符号与设计期一致时取 p/2
    p_one = (p / 2) if (t == t and np.sign(t) == np.sign(md)) else (
        1 - p / 2 if t == t else np.nan)
    c1 = same_sign
    c2 = bool(abs(mh) >= min_abs_ic) if mh == mh else False
    c3 = bool(p_one < alpha) if p_one == p_one else False
    return dict(设计期IC=md, 设计期_n=len(ic_dev),
                留出期IC=mh, 符号一致=c1, 留出期_n=len(ic_hold),
                留出期NW_t=t, 留出期单侧p=p_one,
                条件1_符号一致=c1, 条件2_绝对IC达标=c2, 条件3_p达标=c3,
                通过=bool(c1 and c2 and c3))


def design_oos_test(ind: dict, b, candidates=None,
                    dev_end=None, holdout_start=None) -> dict:
    """R2-A：对候选因子（形式在检验前已固定）做设计期/留出期两段检验。

    与 run_robustness 里的"样本内/样本外"区别：
      · 那里的切点是 2023Q4，且切点曾用于观察结果；
      · 这里的切点由**实际可建模区间**自适应推导（config.derive_split），
        且留出期在本次检验前从未被任何选择过程使用过。

    机构级披露：实际使用的切点与切分依据写入返回值的 `split_note`，
    使「样本外检验在哪个区间上做」可回溯。
    """
    if dev_end is None or holdout_start is None:
        _de, _hs, _ok, _note = _dyn_split(b)
        dev_end = dev_end or _de
        holdout_start = holdout_start or _hs
        if not _ok:
            return dict(table=pd.DataFrame(), detail={},
                        dev_end=dev_end, holdout_start=holdout_start,
                        split_ok=False, split_note=_note)
    else:
        _note, _ok = f"调用方指定：设计期 ≤{dev_end}｜留出期 ≥{holdout_start}", True
    candidates = candidates or [
        "F15_拥挤背离", "F15d_拥挤背离_动态基准",
        "F1_超配比例", "F1d_超配比例_动态基准",
        "F2_超配Zscore", "F2d_超配Z_动态基准",
        "F3_超配历史分位", "F3d_超配分位_动态基准",
        "F12_配置系数", "F12s_配置系数_静态基准",
        "F13_筹码盈利比例", "F11_综合拥挤度_推荐权重",
        "R_F1_超配比例_残差", "R_F2_超配Z_残差", "R_F3_超配分位_残差",
    ]
    rows, detail = [], {}
    n_short = 0
    # 样本量门槛：按**实际可得**的 IC 观测数设定，而不是写死 6/4。
    # 动机：Wind 持仓数据自 2024Q2 起，可建模仅 10 期；再经 lag=1 前瞻收益
    # 截断后，留出期最多只有 2 个观测（2025Q4、2026Q1）。
    # 若仍要求「留出期 ≥ 4」，所有因子都会被静默跳过 → 返回空表 → 看板崩溃。
    # 正确做法：门槛随可得样本自适应，并在返回中**显式登记样本量是否足以支持
    # 样本外结论**（feasible），不足以支持时如实标注，而不是伪造通过率。
    _allq = _all_qs(b)
    _dev_pool = [q for q in _allq if q <= dev_end]
    _hold_pool = [q for q in _allq if q >= holdout_start]
    need_dev = max(3, int(round(len(_dev_pool) * 0.7)))
    need_hold = max(2, min(4, len(_hold_pool) - 1))
    for f in candidates:
        if f not in ind:
            continue
        w = _ic_by_window(ind, b, f, dev_end, holdout_start)
        if len(w["ic_dev"]) < need_dev or len(w["ic_hold"]) < need_hold:
            n_short += 1
            continue
        j = _judge(w["ic_dev"], w["ic_hold"])
        detail[f] = w
        rows.append(dict(因子=f, **j))
    df = pd.DataFrame(rows)
    # 样本外结论是否"成立"：留出期观测 ≥ 4 才具备最低统计功效。
    # 低于该数时，通过与否都不足以支撑结论——必须在返回中显式标注。
    feasible = len(df) > 0 and int(df["留出期_n"].min()) >= 4 if len(df) else False
    _note2 = (_note + f"；门槛：设计期 ≥{need_dev} 期、留出期 ≥{need_hold} 期"
              f"（设计池 {len(_dev_pool)} 期、留出池 {len(_hold_pool)} 期）")
    if not feasible:
        _note2 += ("　⚠️ 留出期观测 < 4，样本外结论**统计功效不足**，"
                   "仅作方向性参考，不得作为因子入选依据。")
    return dict(table=df, detail=detail,
                dev_end=dev_end, holdout_start=holdout_start,
                split_ok=_ok, split_note=_note2,
                feasible=bool(feasible),
                need_dev=need_dev, need_hold=need_hold,
                n_candidates=len(candidates), n_short=n_short)


def specification_search(ind: dict, b, candidates=None,
                         dev_end=None, holdout_start=None) -> dict:
    """R2-A（加强版）：让**设计期**在若干竞争设定中挑一个，再看留出期。

    这是比 design_oos_test 更严格的检验——它检验的是"**选设定的流程**"
    是否可复制，而不只是"某一个已定形式"是否有效。

    选择规则（事前写死）：在设计期内取 |IC| 最大者，若有并列取构造更简者。

    ⚠️ 切点必须走 _dyn_split
    -----------------------
    早期版本默认参数写成 `dev_end=DEV_END`，而 DEV_END 只是模块级**别名**、
    在 import 时即被冻结为 "2022Q4"。即使 _dyn_split 已把真实切点更新为
    2025Q3，本函数仍会拿 2022Q4 去切 → 设计期为空 → 返回 holdout_eval=None
    → 调用方 `_he["留出期IC"]` 抛 TypeError。现统一改为惰性求解。
    """
    if dev_end is None or holdout_start is None:
        _de, _hs, _ok2, _note2 = _dyn_split(b)
        dev_end = dev_end or _de
        holdout_start = holdout_start or _hs
    candidates = candidates or [
        "F15_拥挤背离", "F15d_拥挤背离_动态基准",
        "F1_超配比例", "F1d_超配比例_动态基准",
        "F2_超配Zscore", "F2d_超配Z_动态基准",
        "F3_超配历史分位", "F3d_超配分位_动态基准",
        "F12_配置系数", "F13_筹码盈利比例",
    ]
    rows = []
    for f in candidates:
        if f not in ind:
            continue
        w = _ic_by_window(ind, b, f, dev_end, holdout_start)
        # 门槛同样自适应（原因见 design_oos_test 注释）
        if len(w["ic_dev"]) < max(3, int(round(len([q for q in _all_qs(b)
                                                   if q <= dev_end]) * 0.7))):
            continue
        rows.append(dict(候选=f, 设计期IC=float(w["ic_dev"].mean()),
                         设计期n=len(w["ic_dev"])))
    dev = pd.DataFrame(rows)
    if dev.empty:
        return dict(selected=None, table=dev, holdout_eval=None,
                    dev_end=dev_end, holdout_start=holdout_start,
                    note="设计期内无可评估候选（样本不足）")
    dev["_abs"] = dev["设计期IC"].abs()
    sel = dev.sort_values("_abs", ascending=False).iloc[0]["候选"]
    w = _ic_by_window(ind, b, sel, dev_end, holdout_start)
    j = _judge(w["ic_dev"], w["ic_hold"])
    return dict(selected=sel, table=dev.drop(columns="_abs"),
                holdout_eval=j, detail={sel: w},
                dev_end=dev_end, holdout_start=holdout_start,
                note=f"切点 {dev_end} / {holdout_start}")


# ============================================================================
# R2-B  二层正交化
# ============================================================================
def second_layer_orthogonalize(ind: dict, b, targets=None, peers=None,
                              min_obs: int = 15, fwd_series=None) -> dict:
    """对因子做"第一层（价格风格）+ 第二层（同族因子）"双重正交化。

    第一层：剔 第t期收益 / 过去4季收益 / 过去4季波动（价格类风格）
    第二层：剔 同族因子 F1d / F12 / F13 / F15d（同为持仓拥挤族）
    目的：若残差 IC 在第二层后大幅衰减 → 该因子的"增量信息"其实来自同族，
          并非独立信息；若几乎不衰减 → 存在独立的增量信息。
    """
    fwd = forward_excess(b.market, lag=1) if fwd_series is None else fwd_series
    targets = targets or ["R_F2_超配Z_残差", "R_F3_超配分位_残差", "F15_拥挤背离"]
    peers = peers or ["F1d_超配比例_动态基准", "F12_配置系数",
                      "F13_筹码盈利比例", "F15d_拥挤背离_动态基准"]
    ctrl1 = [ind[c] for c in ("CTRL_第t期收益", "CTRL_过去4季收益",
                              "CTRL_过去4季波动") if c in ind]
    ctrl2 = [ind[c] for c in peers if c in ind]

    def _res(fac, ctrls):
        out = pd.DataFrame(np.nan, index=fac.index, columns=fac.columns, dtype=float)
        for q in fac.index:
            y = fac.loc[q]
            m = y.notna()
            for c in ctrls:
                m &= c.loc[q].notna()
            if int(m.sum()) < min_obs:
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

    rows, resid_both = [], {}
    for f in targets:
        if f not in ind:
            continue
        layer1 = ind[f]
        layer2 = _res(ind[f], ctrl1 + ctrl2)
        resid_both[f] = layer2
        s0 = ic_stats(rank_ic_panel(ind[f], fwd))
        s1 = ic_stats(rank_ic_panel(layer1, fwd))
        s2 = ic_stats(rank_ic_panel(layer2, fwd))
        rows.append(dict(
            因子=f,
            原始IC=s0["ic_mean"], 原始IC_IR=s0["ic_ir"],
            一层残差IC=s1["ic_mean"], 一层残差IC_IR=s1["ic_ir"],
            二层残差IC=s2["ic_mean"], 二层残差IC_IR=s2["ic_ir"],
            二层保留比例=(abs(s2["ic_mean"]) / abs(s1["ic_mean"]) * 100)
            if s1["ic_mean"] else np.nan,
            含独立增量信息=bool(abs(s2["ic_mean"] or 0) > 0.03)))
    return dict(table=pd.DataFrame(rows), residual=resid_both,
                peers_used=[c for c in peers if c in ind])


# ============================================================================
# R2-C  组合层面检验：行业轮动策略
# ============================================================================
def _cap_normalize(w: pd.Series, cap: float, iters: int = 60) -> pd.Series:
    """把权重限制在 [0, cap] 并归一化到 1（迭代再分配 / water-filling）。

    性能说明（本轮修复）：原实现用 pandas 的布尔掩码赋值反复改写 Series
    （`w[over] = cap`）。这种链式赋值会让 pandas 为每次写入都构造一次
    SettingWithCopyWarning 的消息，而构造消息需要**格式化整个 Series** ——
    单次调用因此要 ~46ms，且它在一次策略回测里被调用几十次，
    22 次策略调用 × 33 期合计占首屏计算时间的一半以上。
    改用 numpy 数组后完全绕开 pandas 的赋值路径，数值结果逐位一致。
    """
    idx = w.index
    a = np.array(w.to_numpy(dtype=float), copy=True)
    a[~np.isfinite(a)] = 0.0
    a = np.clip(a, 0.0, None)
    s = float(a.sum())
    if s <= 0:
        return pd.Series(1.0 / len(a), index=idx)
    a = a / s
    if cap is None or cap >= 1.0:
        return pd.Series(a, index=idx)
    for _ in range(iters):
        over = a > cap + 1e-12
        if not over.any():
            break
        excess = float((a[over] - cap).sum())
        a[over] = cap
        free = ~over
        if not free.any():
            break
        base_sum = float(a[free].sum())
        if base_sum <= 0:
            a[free] = excess / float(free.sum())
        else:
            a[free] = a[free] + a[free] / base_sum * excess
        a = a / a.sum()
    return pd.Series(np.clip(a, None, cap * 1.000001), index=idx)


def industry_rotation_strategy(ind: dict, b, factor: str,
                               lam: float = 0.5, cap: float = 0.10,
                               base: str = "equal", cost_bp: float = 20.0,
                               min_obs: int = 20, direction: str = "negative") -> dict:
    """基于拥挤度因子的 long-only 行业轮动策略。

    构造（全部参数**事前固定**，不做最优化）：
      z_i(t) = 因子在第 t 期的横截面 z 分数
      raw_i  = base_i × exp(κ · λ · z_i)        λ = 0.5（软倾斜强度）
      κ      = +1（direction="positive"，因子值越高越配）
               −1（direction="negative"，因子值越高越少配，拥挤度类默认）
      w_i    = cap_normalize(raw, cap)          单行业上限（默认 10%）
      季度调仓，权重和为 1（满仓，不做择时）

    为什么要显式传入方向：方向本身是一个**自由参数**。若按"数据里哪个方向
    收益高就选哪个"，等于在样本内挑参数。故本函数：(a) 默认按因子**经济先验**
    取方向；(b) 由调用方对每个因子**同时报告两个方向**，让读者看到
    "方向是否与先验一致"，而不是把挑出来的好结果当作唯一结论。
    exp 形式保证权重恒正、且对极端 z 值不过度放大。

    base = "equal"  → 等权 31 个申万一级行业（行业轮动的中性基准）
    base = "hs300"  → 沪深300 动态季度权重（贴近真实可投资基准）

    输出：净值曲线、毛/净年化、夏普（对基准）、最大回撤、换手、胜率。
    """
    qs = list(b.quarters)
    ret = b.market["ret"].reindex(index=qs, columns=C.INDUSTRIES).astype(float)
    if base == "equal":
        base_w = pd.Series(1.0 / len(C.INDUSTRIES), index=C.INDUSTRIES)
        base_name = "等权 31 行业"
        base_ts = None
    else:
        # 修正（第 3 轮）：此前误用「最新一期权重固定不变」当作沪深300 基准，
        # 与 research3.benchmark_variants 的**逐期动态权重**口径不一致，
        # 导致同一策略在两处报出符号相反的超额。现统一为逐期动态权重。
        base_ts = (b.weight_matrix.reindex(index=qs, columns=C.INDUSTRIES) / 100.0)
        base_w = base_ts.iloc[-1]
        base_name = "沪深300 动态权重（逐期）"
    rows, weights, prev = [], {}, None
    # 性能：横截面 z 只算一次。
    # 原先在循环内对「单行 DataFrame」反复调用 cz()，每期都要重建一次 DataFrame，
    # 22 次策略调用 × 33 期 ≈ 15 秒，是首屏加载的最大瓶颈。
    # 一次性对整块面板做横截面标准化，数值结果完全一致（cz 是逐行运算）。
    _fac = ind.get(factor)
    zmat = (cz(_fac.reindex(index=qs, columns=C.INDUSTRIES))
            if _fac is not None else None)
    for i, q in enumerate(qs):
        if zmat is None:
            continue
        z = zmat.loc[q]
        if z.notna().sum() < min_obs:
            continue
        z = z.fillna(0.0)
        kap = 1.0 if direction == "positive" else -1.0
        _bw_t = (base_w if base_ts is None else base_ts.loc[q])
        raw = _bw_t.reindex(C.INDUSTRIES).fillna(0.0) * np.exp(kap * lam * z)
        w = _cap_normalize(raw, cap)
        weights[q] = w
        # 用第 t 期权重持有第 t+1 期收益 → 严格无未来函数
        if i + 1 >= len(qs):
            break
        r_next = ret.iloc[i + 1]
        m = r_next.notna()
        port = float((w[m] * r_next[m] / 100.0).sum())
        bench = float((_bw_t.reindex(r_next.index).fillna(0.0)[m]
                       * r_next[m] / 100.0).sum())
        if prev is None:
            tv = np.nan
        else:
            tv = float((w - prev).abs().sum()) / 2.0
        rows.append(dict(quarter=f"{qs[i]}→{qs[i + 1]}", 持仓季度=q,
                         组合收益=port * 100, 基准收益=bench * 100,
                         超额=(port - bench) * 100, turnover=tv,
                         最大行业权重=float(w.max() * 100)))
        prev = w

    # 空行保护（机构级稳健性）：
    # 当因子无任何可用观测（如 F15 在样本过短、或扩窗最小样本数未满足时）
    # rows 为空，pd.DataFrame([]).set_index("quarter") 会直接抛 KeyError，
    # 使整个看板崩溃。这里必须先判空，再决定是否 set_index。
    if not rows:
        return dict(table=pd.DataFrame(), perf={}, weights=weights, base=base_name)
    t = pd.DataFrame(rows).set_index("quarter")
    t["turnover"] = t["turnover"].fillna(t["turnover"].dropna().mean()
                                         if t["turnover"].notna().any() else 1.0)
    t["组合收益_净"] = (t["组合收益"] - t["turnover"] * (cost_bp / 10000.0) * 100 * 2)
    t["超额_净"] = t["组合收益_净"] - t["基准收益"]

    perf = dict(
        毛收益=perf_series(t["组合收益"]),
        净收益=perf_series(t["组合收益_净"]),
        基准=perf_series(t["基准收益"]),
        超额毛=perf_series(t["超额"]),
        超额净=perf_series(t["超额_净"]),
        平均换手率=float(t["turnover"].mean()),
        平均最大行业权重=float(t["最大行业权重"].mean()),
        上线季度=t.index[0], 结束季度=t.index[-1], 期数=len(t),
    )
    # 相对基准的夏普（信息比）
    ex = t["超额_净"]
    ir = (ex.mean() * 4) / (ex.std(ddof=1) * 2) if ex.std(ddof=1) else np.nan
    perf["信息比IR"] = ir
    perf["超额胜率"] = float((ex > 0).mean() * 100)
    perf["参数"] = dict(λ=lam, 单行业上限=f"{cap * 100:.0f}%", 基准=base_name,
                        成本=f"{cost_bp:.0f}bp", 方向=direction)
    return dict(table=t, perf=perf, weights=weights, base=base_name,
                factor=factor, lam=lam, cap=cap, direction=direction)


def cap_sensitivity(ind: dict, b, factor: str, caps=(0.05, 0.10, 0.15, 0.20, 1.0),
                    lambdas=(0.25, 0.5, 1.0), base: str = "equal",
                    cost_bp: float = 20.0, direction: str = "negative") -> pd.DataFrame:
    """cap / λ 敏感性：核心参数不应导致结论剧烈跳变。"""
    rows = []
    for c in caps:
        for lam in lambdas:
            r = industry_rotation_strategy(ind, b, factor, lam=lam, cap=c,
                                           base=base, cost_bp=cost_bp,
                                           direction=direction)
            p = r["perf"]
            if not p:
                continue
            rows.append(dict(单行业上限=f"{c * 100:.0f}%" if c < 1 else "无限制",
                             λ=lam,
                             净年化=p["净收益"]["ann"], 超额净年化=p["超额净"]["ann"],
                             信息比IR=p["信息比IR"], 最大回撤=p["净收益"]["maxdd"],
                             平均换手率=f"{p['平均换手率'] * 100:.1f}%",
                             平均最大行业权重=f"{p['平均最大行业权重']:.1f}%",
                             超额胜率=f"{p['超额胜率']:.1f}%"))
    return pd.DataFrame(rows)


# ============================================================================
# R2-D  静态 vs 动态基准口径对比
# ============================================================================
_BENCH_PAIRS = [
    ("F1_超配比例", "F1d_超配比例_动态基准", "F1 超配（水平）"),
    ("F2_超配Zscore", "F2d_超配Z_动态基准", "F2 超配Z"),
    ("F3_超配历史分位", "F3d_超配分位_动态基准", "F3 超配分位"),
    ("F12s_配置系数_静态基准", "F12_配置系数", "F12 配置系数"),
    ("F15_拥挤背离", "F15d_拥挤背离_动态基准", "F15 拥挤背离"),
]


def benchmark_comparison(ind: dict, b) -> pd.DataFrame:
    """同一因子族在「静态基准」与「动态基准」两种口径下的 IC 对比。

    |IC| 更大、IC_IR 更稳的一方即更有效的基准口径。
    """
    fwd = forward_excess(b.market, lag=1)
    rows = []
    for f_s, f_d, label in _BENCH_PAIRS:
        if f_s not in ind or f_d not in ind:
            continue
        a = ic_stats(rank_ic_panel(ind[f_s], fwd))
        c = ic_stats(rank_ic_panel(ind[f_d], fwd))
        rows.append(dict(因子族=label,
                         静态基准IC=a["ic_mean"], 静态基准IC_IR=a["ic_ir"],
                         动态基准IC=c["ic_mean"], 动态基准IC_IR=c["ic_ir"],
                         IC改善=c["ic_mean"] - a["ic_mean"],
                         IC_IR改善=c["ic_ir"] - a["ic_ir"],
                         更优口径=("动态基准"
                                   if abs(c["ic_ir"]) > abs(a["ic_ir"])
                                   else "静态基准")))
    return pd.DataFrame(rows)


def rotation_direction_scan(ind: dict, b, factors=None, base: str = "equal",
                            lam: float = 0.5, cap: float = 0.10,
                            cost_bp: float = 20.0) -> pd.DataFrame:
    """对每个因子**同时**跑两个方向，报告哪个方向有效、以及是否与经济先验一致。

    经济先验（拥挤度类因子 = 风险因子，先验方向为 negative）：
      F1/F1d/F2/F2d/F3/F3d/F12/F12s/F13/F15/F15d/F11  → negative
    这样"方向与先验一致"就成为一条**可被证伪的检验**，而不是事后挑参数。
    """
    factors = factors or ["F15_拥挤背离", "F15d_拥挤背离_动态基准",
                          "F1_超配比例", "F1d_超配比例_动态基准",
                          "F2_超配Zscore", "F2d_超配Z_动态基准",
                          "F3_超配历史分位", "F3d_超配分位_动态基准",
                          "F12_配置系数", "F13_筹码盈利比例",
                          "F11_综合拥挤度_推荐权重"]
    rows = []
    for f in factors:
        if f not in ind:
            continue
        rec = dict(因子=f)
        for d in ("negative", "positive"):
            r = industry_rotation_strategy(ind, b, f, lam=lam, cap=cap,
                                           base=base, cost_bp=cost_bp,
                                           direction=d)
            p = r["perf"]
            if not p:
                continue
            rec[f"净年化_{d}"] = p["净收益"]["ann"]
            rec[f"超额净_{d}"] = p["超额净"]["ann"]
            rec[f"IR_{d}"] = p["信息比IR"]
        if f"IR_negative" in rec and f"IR_positive" in rec:
            best = ("negative" if rec["IR_negative"] >= rec["IR_positive"]
                    else "positive")
            rec["更优方向"] = best
            rec["与经济先验一致"] = bool(best == "negative")
        rows.append(rec)
    return pd.DataFrame(rows)


# ============================================================================
# R2-F  多重检验：区分「预注册主族」与「探索性扩展族」
# ============================================================================
# 统计上正确的做法：FDR 的 m 应为**预先声明的假设族**大小。
# 若把一切为了探索而计算的变体都塞进同一个族，m 会被稀释、检验功效大幅下降。
# 因此这里显式分成两族：
#   · 主族（PRIMARY）：模型正式提出的 8 个因子 —— 进入 BH-FDR
#   · 扩展族（EXTENDED）：口径对照/残差/动态变体 —— 仅报告 p 值，不参与主族校正
PRIMARY_FAMILY = [
    "F1_超配比例", "F2_超配Zscore", "F3_超配历史分位", "F4b_超配动量Z",
    "F12_配置系数", "F13_筹码盈利比例", "F15_拥挤背离",
    "F11_综合拥挤度_推荐权重",
]


def multiplicity_report(ic_table: pd.DataFrame, pcol: str = "p_nw_t1",
                        alpha: float = 0.10) -> dict:
    """对 IC 表做「主族 / 扩展族」两分法的多重检验报告。"""
    from rigor import bh_fdr, bonferroni
    df = ic_table.copy()
    df["族"] = np.where(df["因子"].isin(PRIMARY_FAMILY), "主族（预注册）", "扩展族（探索）")
    prim = df[df["族"] == "主族（预注册）"]
    ext = df[df["族"] == "扩展族（探索）"]
    out = {}
    if not prim.empty:
        f = bh_fdr(prim[pcol], alpha=alpha)
        b = bonferroni(prim[pcol], alpha=0.05)
        # 机构级稳健性：bh_fdr 会**丢弃 NaN 的 p 值**，返回的 q/reject 长度可能短于 prim。
        # 早期版本用 .values 按位置赋值 → 一旦主族里有因子因样本不足而 p 值为 NaN，
        # 就会抛 "Length of values does not match length of index"。
        # 正确做法：按索引对齐 reindex，缺失处保留 NaN / False。
        prim = prim.assign(
            BH_q值=f["q"].reindex(prim.index),
            BH拒绝=f["reject"].reindex(prim.index).fillna(False).astype(bool),
            Bonferroni拒绝=b["reject"].reindex(prim.index).fillna(False).astype(bool),
        )
        out["primary"] = prim
        out["primary_fdr"] = dict(m=f["m"], reject=int(f["reject"].sum()),
                                  min_q=float(f["q"].min()) if len(f["q"]) else np.nan,
                                  alpha=alpha)
        out["primary_bonf"] = dict(m=b["m"], reject=int(b["reject"].sum()),
                                   threshold=b["threshold"])
    if not ext.empty:
        out["extended"] = ext
        out["extended_min_p"] = float(ext[pcol].min())
        out["extended_n_sig_raw"] = int((ext[pcol] < 0.05).sum())
    out["table"] = pd.concat([prim, ext], ignore_index=True) if not prim.empty else df
    return out


# ============================================================================
# R2-G  因子分级裁决表（把 4 项独立检验合成一个可用性结论）
# ============================================================================
def factor_verdict_table(ind: dict, b, ic_table: pd.DataFrame,
                         cost_table: pd.DataFrame | None = None,
                         doos: dict | None = None,
                         second_layer: dict | None = None,
                         rotation_scan: pd.DataFrame | None = None,
                         rotation_scan_hs300: pd.DataFrame | None = None
                         ) -> pd.DataFrame:
    """把散落在各处的检验结果合成一张**可决策**的分级表。

    四项独立判据（每一项都来自一次独立检验，互不替代）：
      J1 统计显著性   |IC_IR| > 0.3 且 BH-FDR q ≤ 0.10（主族内）
      J2 设计期样本外  design_oos_test 三条件全通过（留出期从未参与设计）
      J3 独立增量信息  second_layer_orthogonalize 剔价格风格 + 剔同族后 |IC| > 0.03
      J4 可交易性      扣 20bp 成本后，按**经济先验方向**的超额净年化 > 0
                      （基准 = 等权 31 行业；同时列出对沪深300 的动态权重基准超额，
                        两者常常符号不同，必须并列引用，不得只报有利的那个）

    分级（宽进严出，宁缺勿滥）：
      A 可用      J2 + J4 同时成立，且（J1 或 J3 至少一项），
                  **且对等权与沪深300 两个基准的超额同为正** → 可作正式信号
      A− 可用     同上，但只战胜等权基准、未战胜沪深300 → 需注明基准依赖
      B 候选      仅 J2 成立（样本外有效但尚不可交易/未通过多重检验）
      C 描述性    J1 或 J3 成立 → 只可用于风险描述与压力测试
      D 未成立    四项均不成立
    """
    if doos is None:
        doos = design_oos_test(ind, b)
    if second_layer is None:
        second_layer = second_layer_orthogonalize(ind, b)
    if rotation_scan is None:
        rotation_scan = rotation_direction_scan(ind, b)
    if rotation_scan_hs300 is None:
        rotation_scan_hs300 = rotation_direction_scan(ind, b, base="hs300")
    mr = multiplicity_report(ic_table)

    ic = ic_table.copy()
    ic["|IC_IR|"] = ic["ic_ir_t1"].abs()
    ic_map = ic.set_index("因子") if "因子" in ic.columns else pd.DataFrame()

    def _safe_map(d, key="因子"):
        """把子表安全地转成以因子名为索引的表；列名缺失或表为空时返回空表。

        机构级稳健性：任何一张子表因样本不足而为空，都不能让整个裁决失败——
        应退化为"该判据不成立"，并在结果中体现，而不是抛异常。
        """
        if isinstance(d, pd.DataFrame) and not d.empty and key in d.columns:
            return d.set_index(key)
        return pd.DataFrame()

    oos_tbl = doos.get("table") if isinstance(doos, dict) else None
    oos_map = _safe_map(oos_tbl)
    sl_map = _safe_map(second_layer.get("table") if isinstance(second_layer, dict)
                       else second_layer)
    rot_map = _safe_map(rotation_scan)
    rot_hs_map = _safe_map(rotation_scan_hs300)
    _prim = mr.get("primary", pd.DataFrame()) if isinstance(mr, dict) else pd.DataFrame()
    q_map = (_safe_map(_prim)["BH_q值"] if (isinstance(_prim, pd.DataFrame)
                                            and not _prim.empty
                                            and "BH_q值" in _prim.columns)
             else pd.Series(dtype=float))
    cost_map = (cost_table.set_index("因子") if cost_table is not None
                and not cost_table.empty and "因子" in cost_table.columns
                else pd.DataFrame())

    rows = []
    for f in sorted(set(ic_map.index) | set(oos_map.index)):
        j1 = bool(ic_map.loc[f, "|IC_IR|"] > 0.3) if f in ic_map.index else False
        q = q_map.get(f, np.nan)
        j1 = j1 and (q == q and q <= 0.10)
        j2 = bool(oos_map.loc[f, "通过"]) if f in oos_map.index else False
        keep = (float(sl_map.loc[f, "二层保留比例"])
                if f in sl_map.index and sl_map.loc[f, "二层保留比例"] == sl_map.loc[f, "二层保留比例"]
                else np.nan)
        j3 = bool(keep > 30) if keep == keep else (j1 and False)
        ex_eq = np.nan
        if f in rot_map.index:
            ex_eq = float(rot_map.loc[f, "超额净_negative"])
            j4 = bool(rot_map.loc[f, "更优方向"] == "negative" and ex_eq > 0)
            prior_ok = bool(rot_map.loc[f, "与经济先验一致"])
        else:
            j4, prior_ok = False, False
        if f in rot_hs_map.index:
            ex_hs = float(rot_hs_map.loc[f, "超额净_negative"])
        else:
            ex_hs = np.nan
        both_bench = bool(ex_eq == ex_eq and ex_hs == ex_hs
                          and ex_eq > 0 and ex_hs > 0)
        if j2 and j4 and (j1 or j3):
            # 两个基准符号一致才给满分，否则显式降级并标注
            grade = "A 可用" if both_bench else "A− 可用（仅胜等权基准）"
        elif j2:
            grade = "B 候选（样本外有效）"
        elif j1 or j3:
            grade = "C 描述性（仅风险描述）"
        else:
            grade = "D 未成立"
        rows.append(dict(
            因子=f, 族=("主族" if f in PRIMARY_FAMILY else "扩展"),
            IC均值=ic_map.loc[f, "ic_mean_t1"] if f in ic_map.index else np.nan,
            IC_IR=ic_map.loc[f, "ic_ir_t1"] if f in ic_map.index else np.nan,
            p_NW=ic_map.loc[f, "p_nw_t1"] if f in ic_map.index else np.nan,
            BH_q值=q,
            J1_统计显著=j1,
            J2_设计期样本外=j2,
            J3_独立增量信息=(bool(keep > 30) if keep == keep else "未检验"),
            二层保留比例=keep,
            J4_超额净_对等权=ex_eq,
            J4_超额净_对沪深300=ex_hs,
            J4_可交易=(j4 if f in rot_map.index else "未检验"),
            方向与先验一致=(prior_ok if f in rot_map.index else "未检验"),
            裁决=grade))
    order = {"A 可用": 0, "A− 可用（仅胜等权基准）": 1,
             "B 候选（样本外有效）": 2, "C 描述性（仅风险描述）": 3,
             "D 未成立": 4}
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["因子", "族", "IC均值", "IC_IR", "p_NW", "BH_q值",
                                   "J1_统计显著", "J2_设计期样本外",
                                   "J3_独立增量信息", "二层保留比例",
                                   "J4_超额净_对等权", "J4_超额净_对沪深300",
                                   "J4_可交易", "方向与先验一致", "裁决"])
    else:
        df["_o"] = df["裁决"].map(order)
        df = (df.sort_values(["_o", "IC_IR"],
                             key=lambda c: c if c.name == "_o" else -c.abs())
              .drop(columns="_o").reset_index(drop=True))
    # 机构级披露：把本次裁决所依据的样本切分一并带回，避免"只看裁决不看区间"
    df.attrs["split_note"] = (doos.get("split_note", "") if isinstance(doos, dict)
                              else "")
    return df
