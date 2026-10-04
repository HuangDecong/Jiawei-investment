# -*- coding: utf-8 -*-
"""
build_dynamic_weights.py  ——  重建沪深300 动态季度行业权重（漂移法）
======================================================================
背景（第 1 轮遗留问题）
------------------------
`data/raw/hs300_weights.xlsx` 是从工作簿「配置比例 − 超配比例」反解出的
**静态权重**（34 期恒定，逐行业标准差 2.6e-14，加总 = 100.000000）。
它反映的是**最新（2026Q2）的沪深300 行业权重**。

用 2026 年的权重去衡量 2018 年的超配，会把"因价格上涨而自然上升的权重"
误判为"基金经理主动加仓"。这是 v1/v2 已知的系统性偏差。

方法（漂移法 / drift reconstruction）
------------------------------------
在**只掌握某一时点权重**、以及各行业与指数的**收益率序列**的条件下，
可以反推其它时点的权重 —— 假设区间内没有成分调整与权重再平衡：

    G_i(t→T) = Π_{s=t+1..T} [ (1 + r_i,s) / (1 + r_hs300,s) ]     相对价格累积因子
    w_i(t)   = w_i(T) / G_i(t→T)                                   基准时点反推
    w_i(t)   ← w_i(t) / Σ_j w_j(t) × 100                            归一到 100%

这是指数权重领域公认的近似，**能捕捉主导效应（价格漂移）**，
但**无法捕捉**：① 半年度成分调整（6 月/12 月）；② 自由流通市值加权下的
股本变动；③ 新股纳入。故它是一个**有偏但方向正确**的近似，必须显式声明。

锚点时点选择
------------
以 **T = 2026Q2**（静态权重的时点）为锚，向前反推。这样：
  · t = 2026Q2 时重建权重 ≡ 静态权重（可作为自校验，误差应为 0）
  · t < 2026Q2 时权重随相对收益向前回退

为什么不以 2018Q1 为锚向后推？因为我们**没有** 2018Q1 的真实权重，
而 2026Q2 的真实权重是已知的（从工作簿反解、且加总严格为 100）。锚点必须选已知的一端。

输出
----
    data/raw/hs300_weights_quarterly.xlsx   长表：季度 | 行业 | 权重%
    output/dynamic_weights_report.xlsx      校验与对比报告
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import config as C            # noqa: E402
import data_loader as DL      # noqa: E402

ANCHOR_Q = "2026Q2"
QUERY_DATE = "2026-09-30"
SOURCE_DB = "同花顺 iFinD + 用户提供工作簿"
SOURCE_TABLE = ("行业季度涨跌幅：iFinD 申万一级行业指数(801xxx.SL)季线；"
                "沪深300：iFinD 000300.SH 季线；"
                "锚点权重：工作簿 Sheet2「配置比例」− Sheet4「超配比例」反解")


def load_static_weights() -> pd.Series:
    df = pd.read_excel(C.F_HS300_WEIGHTS)
    col = "权重%" if "权重%" in df.columns else df.columns[-1]
    return df.set_index("行业")[col].reindex(C.INDUSTRIES).astype(float)


def main() -> int:
    # 首次运行时 hs300_weights_quarterly.xlsx 尚不存在，loader 会回退静态口径，
    # 正好给出我们需要的基础序列（行业收益 / 沪深300 收益 / 静态权重）。
    b = DL.load_all(verbose=False)
    quarters = list(b.quarters)
    static = b.hs300_weights.reindex(C.INDUSTRIES).astype(float)

    ret = b.market["ret"].reindex(index=quarters, columns=C.INDUSTRIES).astype(float)
    hs300 = b.market["hs300"].reindex(quarters).astype(float)

    if ANCHOR_Q not in quarters:
        raise SystemExit(f"锚点季度 {ANCHOR_Q} 不在数据区间内：{quarters[0]}~{quarters[-1]}")
    ia = quarters.index(ANCHOR_Q)

    # ---- 相对价格累积因子 G_i(t→anchor) ----
    rel = (1 + ret / 100.0).div(1 + hs300 / 100.0, axis=0)   # 每期相对收益
    G = pd.DataFrame(1.0, index=quarters, columns=C.INDUSTRIES)
    # 从锚点向前累积：G(t) = Π_{s=t+1..anchor} rel_s
    acc = pd.Series(1.0, index=C.INDUSTRIES)
    G.iloc[ia] = 1.0
    for t in range(ia - 1, -1, -1):
        acc = acc * rel.iloc[t + 1]
        G.iloc[t] = acc

    # ---- 反推权重 ----
    recon = G.apply(lambda r: static / r, axis=1)
    recon = recon.div(recon.sum(axis=1), axis=0) * 100.0

    # ---- 自校验：锚点处应 ≡ 静态权重 ----
    anchor_err = float((recon.loc[ANCHOR_Q] - static).abs().max())
    rowsum_err = float((recon.sum(axis=1) - 100.0).abs().max())

    # ---- 写出长表 ----
    long = recon.stack().rename("权重%").reset_index()
    long.columns = ["季度", "行业", "权重%"]
    # 注意：这是「输入数据文件」，保留 4 位小数以便锚点严格对齐静态权重
    # （3dp 会使锚点产生 ≤0.0005pct 的舍入噪声）；
    # 所有「导出报告与看板显示」仍按任务要求取 2 位小数。
    long["权重%"] = long["权重%"].round(4)
    long["数据来源"] = SOURCE_DB
    long["数据表"] = SOURCE_TABLE
    long["查询日期"] = QUERY_DATE
    long["口径"] = "漂移法重建（锚点 2026Q2，未计成分调整）"
    with pd.ExcelWriter(C.F_HS300_WEIGHTS_Q, engine="openpyxl") as w:
        long.to_excel(w, sheet_name="hs300_weights_quarterly", index=False)

    # ---- 对比报告 ----
    cmp_rows = []
    for q in quarters:
        d = recon.loc[q] - static
        cmp_rows.append(dict(季度=q,
                             **{k: round(float(recon.loc[q, k]), 2)
                                for k in ["电子", "银行", "医药生物", "食品饮料",
                                          "电力设备", "通信"]},
                             与静态最大偏离=round(float(d.abs().max()), 4),
                             偏离最大行业=d.abs().idxmax()))
    cmp = pd.DataFrame(cmp_rows)

    chg = pd.DataFrame({
        "行业": C.INDUSTRIES,
        "2018Q1重建%": recon.loc[quarters[0]].round(2).to_numpy(),
        "2026Q2静态%": static.round(2).to_numpy(),
        "变化pct": (static - recon.loc[quarters[0]]).round(2).to_numpy(),
    }).sort_values("变化pct", ascending=False)

    with pd.ExcelWriter(C.O_DYNAMIC_WEIGHTS if hasattr(C, "O_DYNAMIC_WEIGHTS")
                        else os.path.join(C.OUTPUT, "dynamic_weights_report.xlsx"),
                        engine="openpyxl") as w:
        cmp.to_excel(w, sheet_name="逐期权重与偏离", index=False)
        chg.to_excel(w, sheet_name="两端对比", index=False)
        recon.round(4).to_excel(w, sheet_name="重建权重_宽表")
        pd.DataFrame([dict(项="锚点季度", 值=ANCHOR_Q),
                      dict(项="锚点自校验最大误差(pct)", 值=round(anchor_err, 10)),
                      dict(项="逐期加总最大偏离(pct)", 值=round(rowsum_err, 10)),
                      dict(项="区间", 值=f"{quarters[0]}~{quarters[-1]}"),
                      dict(项="方法", 值="漂移法（相对收益反推 + 归一）"),
                      dict(项="已知局限1", 值="未计半年度成分调整（6月/12月）"),
                      dict(项="已知局限2", 值="未计自由流通市值加权下的股本变动"),
                      dict(项="已知局限3", 值="未计新股纳入"),
                      dict(项="数据来源", 值=SOURCE_DB),
                      dict(项="查询日期", 值=QUERY_DATE)]
                     ).to_excel(w, sheet_name="方法与校验", index=False)

    print("=" * 74)
    print("沪深300 动态季度权重（漂移法）已生成")
    print("=" * 74)
    print(f"  输出：{C.F_HS300_WEIGHTS_Q}")
    print(f"  区间：{quarters[0]} ~ {quarters[-1]}（{len(quarters)} 期 × {len(C.INDUSTRIES)} 行业）")
    print(f"  锚点自校验：{ANCHOR_Q} 重建权重与静态权重最大误差 = {anchor_err:.2e}（应为 0）")
    print(f"  逐期加总最大偏离 = {rowsum_err:.2e} pct（应为 0）")
    print()
    print("两端对比（漂移幅度最大的 10 个行业）：")
    print(chg.head(10).to_string(index=False))
    print()
    print("两端对比（漂移幅度最小/反向的 5 个行业）：")
    print(chg.tail(5).to_string(index=False))
    print()
    print("关键行业时序（每 8 期）：")
    print(cmp.iloc[::8].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
