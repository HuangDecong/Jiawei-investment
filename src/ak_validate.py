# -*- coding: utf-8 -*-
"""分歧度 / 基金行为的时序验证：IC、分组、剔 Q4、分段、正交化、成本。

为什么单独一个模块
------------------
`institution.six_checks` 是通用检验器，但分歧度是「由持仓变动构造的行业面板」，
构造方式与用途都和项目原有的 15 个因子不同，所以因子面板的构建必须写清楚：

· 每一期 q 的因子值 = 从 q0 到 q 的基金行为统计（q0 是同类披露口径的期，见 `ak_stock`）；
· 因子在 q 期「已知」，用于预测 q+1 期行业超额收益（`bt.forward_excess(..., lag=1)`）；
· 所有跨期比较都按每只基金每期覆盖率归一化，否则季报/半年报切换会造出假信号。

样本门槛
--------
本模块强制要求「有效期数 ≥ 12」才输出显著性判断；不足时只输出描述统计，
并在返回里标 `insufficient=True`，页面与报告据此降级。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C  # noqa: E402
import ak_data as akd  # noqa: E402
import institution as inst  # noqa: E402


METRICS = {
    "加仓占比": "该行业被加仓基金数 ÷ 有效样本基金数（%）",
    "分歧度": "1 − |加仓占比 − 50%| × 2，0=一致、1=对半分裂",
    "平均配置变化": "样本基金该行业占净值比的均值变化（pp，已按覆盖率归一化）",
    "净加仓强度": "（加仓基金数 − 减仓基金数）÷ 有效样本基金数",
}


def _pairs() -> list[tuple[str, str]]:
    """所有可算的相邻期对（q0 → q），只保留两边都有数据的。"""
    qs = akd.latest_quarters_all()
    if len(qs) < 2:
        return []
    return [(qs[i - 1], qs[i]) for i in range(1, len(qs))]


def build_panel(metric: str = "加仓占比") -> pd.DataFrame:
    """行业 × 季度 的因子面板。index 为期次（信息可得时点）。"""
    rows = {}
    for q0, q in _pairs():
        d = akd.dispersion_ak(q, q0)
        if d.empty:
            continue
        d = d.set_index("行业")
        if metric == "加仓占比":
            v = d["加仓占比"]
        elif metric == "分歧度":
            v = d["分歧度"]
        elif metric == "平均配置变化":
            v = d["平均配置"]
        elif metric == "净加仓强度":
            v = (d["加仓基金数"] - d["减仓基金数"]) / d["样本基金数"].replace(0, np.nan) * 100
        else:
            raise ValueError(metric)
        rows[q] = v.reindex(C.INDUSTRIES)
    if not rows:
        return pd.DataFrame(columns=C.INDUSTRIES)
    out = pd.DataFrame(rows).T.sort_index()
    out.attrs["metric"] = metric
    return out


def _same_type_pairs() -> list[tuple[str, str]]:
    """同类披露口径的期对（Q2 对 Q2、Q3 对 Q3…），用于稳健性复检。"""
    import ak_stock as aks
    qs = akd.latest_quarters_all()
    out = []
    for q in qs:
        p = aks.comparable_pair(q, qs)
        if p and p[1] in qs:
            out.append((p[1], p[0]))
    return out


def build_panel_same_type(metric: str = "加仓占比") -> pd.DataFrame:
    """只用同类口径的期对构造面板——期数会少一半以上，但没有任何口径混比。"""
    rows, pairs = {}, {}
    for q0, q in _same_type_pairs():
        d = akd.dispersion_ak(q, q0)
        if d.empty:
            continue
        d = d.set_index("行业")
        v = d["加仓占比"] if metric == "加仓占比" else d["分歧度"]
        rows[q] = v.reindex(C.INDUSTRIES)   # 用 q 作索引，才能与次季收益对齐
        pairs[q] = q0
    if not rows:
        return pd.DataFrame(columns=C.INDUSTRIES)
    out = pd.DataFrame(rows).T.sort_index()
    out.attrs["pairs"] = pairs
    return out


def coverage_by_quarter() -> pd.DataFrame:
    """逐期样本覆盖：基金数、平均覆盖率、行业数。用于判断哪几期可用。"""
    rows = []
    for q in akd.latest_quarters_all():
        con = akd._con()
        if con is None:
            continue
        try:
            n_f = con.execute("SELECT COUNT(DISTINCT code) FROM fund_holdings WHERE quarter=?",
                              (q,)).fetchone()[0]
            cov = con.execute("SELECT AVG(coverage) FROM coverage WHERE quarter=?",
                              (q,)).fetchone()[0]
        except Exception:
            n_f, cov = 0, None
        con.close()
        rows.append(dict(期次=q, 基金数=int(n_f or 0),
                         平均覆盖率=round(float(cov), 1) if cov else None,
                         披露口径="全持仓" if q[-1] in ("2", "4") else "前十大"))
    return pd.DataFrame(rows)


def run(metric: str = "加仓占比", b=None) -> dict:
    """跑六项检验。返回 dict 含面板、检验表、是否样本不足。"""
    import data_loader as dl
    if b is None:
        b = dl.load_all()
    panel = build_panel(metric)
    panel = panel.dropna(how="all")
    if panel.empty:
        return dict(metric=metric, panel=panel, checks=None, insufficient=True,
                    n=0, note="无可用期次（尚未抓到跨期数据）")
    r = inst.six_checks(b, panel, metric)
    r["checks"] = r.get("检验")   # six_checks 用中文键「检验」，这里统一成 checks
    r["panel"] = panel
    r["insufficient"] = len(panel) < 12

    # 稳健性：只用「同披露口径」的期对（Q3 对上年 Q3、Q2 对上年 Q2）。
    # 期数会少一半以上，但不存在前十大与全持仓混比的问题——
    # 两种构造的 IC 同向才算稳，反向则说明结果由口径切换驱动。
    try:
        import backtest as bt
        st = build_panel_same_type(metric)
        if not st.empty:
            fwd = bt.forward_excess(b.market, 1)
            icx = bt.rank_ic_panel(st, fwd).dropna()
            if len(icx) >= 3:
                t_nw, p_nw = bt.nw_tstat(icx)
                r["same_type"] = dict(
                    n=int(len(icx)), ic=float(icx.mean()), t=float(t_nw),
                    p=float(p_nw), direction="同向" if np.sign(icx.mean()) ==
                    np.sign(r.get("ic_mean", np.nan)) else "反向",
                    pairs=st.attrs.get("pairs", {}))
    except Exception:
        pass
    r["n_periods"] = len(panel)
    if r["insufficient"]:
        r["note"] = f"仅 {len(panel)} 期，未达 12 期门槛，不得称显著"
    return r


if __name__ == "__main__":
    pd.set_option("display.width", 240)
    print("可用期次:", akd.latest_quarters_all())
    print("\n覆盖：")
    print(coverage_by_quarter().to_string(index=False))
    for m in METRICS:
        r = run(m)
        print(f"\n=== {m} === 期数 {r.get('n_periods')} 状态 {r.get('状态', '—')}")
        if r.get("checks") is not None:
            print(r["checks"].to_string(index=False))
        else:
            print(r.get("note"))
