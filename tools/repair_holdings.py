# -*- coding: utf-8 -*-
"""修复「同一季度混了多套持仓快照」的期次。

问题
----
东财持仓接口会把同一季度内多个披露日（如 2025-05-07 与 2025-06-30）的
持仓混在一张表里：同一只股票出现两行、市值对应不同净资产。
按股票代码去重后，会得到「A 快照的这只 + B 快照的那只」的混合组合，
合计可能超过股票仓位（015346 的 2025Q2 曾算到 128.87%）。

判定
----
名次（rnk）不连续，说明有重复股票在落库时被丢掉；正常期次一定是 1..N。

修复
----
按 (code, year) 重抓该年：`qreport._ak_hold` 会用资产配置表里的候选净资产
识别快照，只保留季末那一套。

用法：
    python tools/repair_holdings.py --dry
    python tools/repair_holdings.py --workers 3
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd  # noqa: E402

import qreport as Q  # noqa: E402


def broken() -> pd.DataFrame:
    Q.ensure_db()
    with Q._con() as con:
        h = pd.read_sql("SELECT code, quarter, rnk, ratio_pct FROM q_hold", con)
    rows = []
    for (c, q), g in h.groupby(["code", "quarter"]):
        rk = sorted(int(x) for x in g["rnk"].dropna())
        if rk and (rk[0] != 1 or rk != list(range(1, len(rk) + 1))):
            rows.append({"code": c, "quarter": q, "rows": len(rk),
                         "max_rnk": rk[-1], "sum_pct": round(g["ratio_pct"].sum(), 2)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    b = broken()
    print(f"名次不连续：{len(b)} 个基金×季度，涉及 {b['code'].nunique()} 只基金")
    if b.empty:
        return 0
    print(b.head(10).to_string(index=False))
    if args.dry:
        return 0

    jobs = sorted({(r.code, int(str(r.quarter)[:4])) for r in b.itertuples()})
    print(f"需要重抓 {len(jobs)} 个「基金×年份」")
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(Q.ensure_year, c, y, True): (c, y) for c, y in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            c, y = futs[f]
            try:
                f.result()
                ok += 1
            except Exception as e:  # noqa: BLE001
                fail += 1
                print("  失败", c, y, type(e).__name__, e)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} 成功 {ok} 失败 {fail}")
    Q._load_cached.cache_clear()
    b2 = broken()
    print(f"修复后仍不连续：{len(b2)} 个；合计>100 的：{int((b2['sum_pct'] > 100).sum()) if len(b2) else 0}")
    if len(b2):
        print(b2.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
