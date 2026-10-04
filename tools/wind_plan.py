# -*- coding: utf-8 -*-
"""
wind_plan.py —— 生成 Wind 抓取计划表（机构级可审计）

按 Wind 单次返回的**行数上限**反推分批粒度，输出可直接执行的抓取清单。

实测结论（本项目）：
  * 单次返回约可容纳 120~200 行；
  * 31 行业 × 3 年 = 372 行 → **会被截断**；
  * 10 行业 × 3 年 = 120 行 → 安全；
  * 单季 × 31 行业 = 31 行 → 安全但效率低。
故制定如下分层抓取策略。

用法：python tools/wind_plan.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import config as C  # noqa: E402

IND = C.INDUSTRIES                      # 31 个
# 按 10 个一组切分行业（10×3年=120 行，安全）
IND_GROUPS = [IND[i:i + 10] for i in range(0, len(IND), 10)]
# 按 3 年一段切分历史
YR_RANGES = [(y, min(y + 2, 2026)) for y in range(2010, 2027, 3)]


def main() -> None:
    plan = []
    # ① returns / turnover：行业组 × 年段
    for task, metric in [("returns", "区间涨跌幅"), ("turnover", "区间成交额")]:
        for g_i, grp in enumerate(IND_GROUPS):
            for y0, y1 in YR_RANGES:
                plan.append(dict(
                    task=task, group=g_i, years=f"{y0}-{y1}",
                    industries=grp,
                    question=(f"查询{'、'.join(grp)}共{len(grp)}个申万一级行业指数"
                              f"从{y0}年一季度到{y1}年四季度每个季度的{metric}")))
    # ② holdings：逐季（单季 31 行，安全）
    for y0, y1 in YR_RANGES:
        for y in range(y0, y1 + 1):
            for q in range(1, 5):
                if y == 2026 and q > 2:
                    break
                plan.append(dict(
                    task="holdings", quarter=f"{y}Q{q}",
                    question=f"查询{y}-{{03-31|06-30|09-30|12-31}}，"
                             f"公募基金前十大重仓股持股总市值按申万一级行业汇总"))
    # ③ hs300：按年段
    for y0, y1 in YR_RANGES:
        plan.append(dict(task="hs300", years=f"{y0}-{y1}",
                         question=f"查询{y0}年至{y1}年各季度末，"
                                  f"沪深300指数成份股按申万一级行业分组的权重合计占比"))

    dest = os.path.join(ROOT, "data", "cache", "_plan.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    print(f"计划已生成：{len(plan)} 条 -> {dest}")
    for t in ("returns", "turnover", "holdings", "hs300"):
        n = sum(1 for p in plan if p["task"] == t)
        print(f"  {t:<9} {n:>3} 条")


if __name__ == "__main__":
    main()
