# -*- coding: utf-8 -*-
"""run_wind_pipeline.py —— Wind 新数据下的全链条研究验证入口。

目的：在数据源切换到 Wind 主动权益三类口径后，端到端验证
      数据 → 因子 → IC 检验 → 因子裁决 的完整链路，并输出结论。

用法：
    python tools/run_wind_pipeline.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data_loader as dl  # noqa: E402
import factor_engine as fe  # noqa: E402
import research as rs  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


def main() -> None:
    print("=" * 78)
    print("Wind 数据源下的全链条验证")
    print("=" * 78)

    b = dl.load_all()
    print(f"\n【1】数据加载")
    print(f"  区间：{b.quarters[0]} ~ {b.latest}  共 {b.n_quarters} 期")
    print(f"  具备持仓：{len(b.alloc_quarters)} 期  可算超配：{len(b.over_quarters)} 期")

    v = dl.validate(b, write_log=True)
    print(f"\n【2】数据校验 R1–R10")
    print(f"  硬性校验通过：{v['passed']}")
    print(f"  R1 最大偏离 = {v['R1_最大的偏离']:.8f} pct")
    print(f"  R6 最大绝对差 = {v['R6_最大绝对差']}")

    f = fe.build_all(b)
    ind = f["ind"]
    print(f"\n【3】因子构建")
    print(f"  行业因子 {len([k for k in ind if not k.startswith('_')])} 个")

    # 有效样本季度：有持仓 + 有未来收益
    hq = b.over_quarters
    print(f"\n【4】RankIC 检验（可算超配的 {len(hq)} 期）")

    # 未来一期行业收益（滞后 1 期，避免前视）
    fwd = rs.forward_excess(b.market, lag=1)
    targets = ["F1_超配比例", "F2_超配Zscore", "F3_超配历史分位",
               "F4b_超配动量Z", "F11_综合拥挤度_推荐权重",
               "F12_配置系数", "F15_拥挤背离"]

    rows = []
    ic_store = {}
    for name in targets:
        if name not in ind:
            continue
        fac = ind[name].reindex(hq)
        try:
            ic = rs.rank_ic_panel(fac, fwd.reindex(hq), min_obs=8)
            ic_store[name] = ic
            st = rs.ic_stats(ic)
            rows.append(dict(
                因子=name,
                RankIC均值=round(st["ic_mean"], 4) if st["n"] else np.nan,
                IC标准差=round(st["ic_std"], 4) if st["n"] else np.nan,
                ICIR=round(st["ic_ir"], 3) if st["n"] else np.nan,
                t_NW=round(st["t_nw"], 2) if st["n"] else np.nan,
                p_NW=round(st["p_nw"], 3) if st["n"] else np.nan,
                胜率=round(st["win_rate"], 1) if st["n"] else np.nan,
                N期=st["n"],
            ))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(因子=name, RankIC均值=np.nan, IC标准差=np.nan,
                             ICIR=np.nan, t_NW=np.nan, p_NW=np.nan,
                             胜率=np.nan, N期=0))

    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    print("\n  注：N=样本期数（仅 9 个持仓季度，统计功效极低；")
    print("      t_NW 为 Newey-West 调整后 t 值；p < 0.1 才有讨论价值。")

    print(f"\n【5】集中度时序（核心结论）")
    cr = b.cr_hhi.reindex(hq).reset_index().rename(columns={"index": "季度"})
    cols = [c for c in ["季度", "CR3", "CR5", "CR10", "HHI",
                        "Top1行业", "Top1比例"] if c in cr.columns]
    print(cr[cols].round(3).to_string(index=False))

    print(f"\n【6】超配比例 Top 行业演化")
    ov = b.overweight.reindex(hq)
    for q in hq:
        r = ov.loc[q].dropna().sort_values(ascending=False)
        print(f"  {q}  超配Top3: "
              + " | ".join(f"{k}{v:+.1f}" for k, v in r.head(3).items())
              + f"    低配Top2: "
              + " | ".join(f"{k}{v:+.1f}" for k, v in r.tail(2).items()))

    print(f"\n【7】数据质量告警")
    for n in b.notes[:10]:
        print(f"  · {n}")

    print("\n" + "=" * 78)
    print("✅ 全链条验证完成")
    print("=" * 78)


if __name__ == "__main__":
    main()
