# -*- coding: utf-8 -*-
"""修复季报库的「有股票仓位、却没有行业/持仓明细」的基金。

为什么会有这种缺口
------------------
批量抓取时东财偶发返回「列错位」的表（截止时间列里是基金代码），
解析后 quarter 成了 '000126'，落库就成了脏行；清理脏行后这只基金
在这一期就「什么都没有」，而 q_log 还记着 ok=1，页面 20 小时内不会重抓。

本脚本只做三件事：
  1. 找出「股票仓位 ≥ 阈值、但某期没有行业表」的基金；
  2. 对这些基金按年重抓（force），成功的写库；
  3. 把 q_log 的 ok 重置，让页面下次打开时按正常路径再校验一遍。

用法：
    python tools/repair_qreport.py            # 只修 2024–2026
    python tools/repair_qreport.py --dry      # 只列清单，不抓
    python tools/repair_qreport.py --all      # 全历史（很慢，慎用）
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


def targets(min_stock: float = 20.0, since: str = "2025Q1") -> pd.DataFrame:
    """返回 [(code, year, quarter)]：有股票仓位但缺行业表的期次。"""
    Q.ensure_db()
    with Q._con() as con:
        a = pd.read_sql("SELECT code, quarter, stock_pct FROM q_asset "
                        "WHERE quarter >= ?", con, params=(since,))
        ind = pd.read_sql("SELECT DISTINCT code, quarter FROM q_industry", con)
        hold = pd.read_sql("SELECT DISTINCT code, quarter FROM q_hold", con)
    a["stock_pct"] = pd.to_numeric(a["stock_pct"], errors="coerce")
    a = a[a["stock_pct"] >= min_stock]
    m = a.merge(ind.assign(has_ind=1), on=["code", "quarter"], how="left")
    m = m.merge(hold.assign(has_hold=1), on=["code", "quarter"], how="left")
    bad = m[m["has_ind"].isna() | m["has_hold"].isna()].copy()
    bad["year"] = bad["quarter"].str[:4].astype(int)
    return bad[["code", "quarter", "year", "stock_pct",
                "has_ind", "has_hold"]].sort_values(["code", "quarter"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--all", action="store_true", help="不限 2024 起")
    ap.add_argument("--since", default="2025Q1", help="从哪个季度开始找缺口")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    t = targets(since=("1900Q1" if args.all else args.since))
    if t.empty:
        print("没有缺口：所有有股票仓位的期次都有行业与持仓明细。")
        return 0
    print(f"缺口：{t['code'].nunique()} 只基金、{len(t)} 个「基金×季度」")
    print(t.groupby("quarter").size().to_string())
    if args.dry:
        print(t.head(30).to_string(index=False))
        return 0

    jobs = sorted({(r.code, int(r.year)) for r in t.itertuples()})
    print(f"需要重抓 {len(jobs)} 个「基金×年份」")
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(Q.ensure_year, c, y, True): (c, y) for c, y in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            c, y = futs[f]
            try:
                r = f.result()
                ok += 1
                if i % 20 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}  {c} {y} rows={r.get('rows')}")
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"  失败 {c} {y}: {type(e).__name__}: {e}")
    print(f"完成：成功 {ok}、失败 {fail}")

    # 让页面按正常路径重新校验（否则 q_log 的 ok=1 会继续走 20 小时缓存）
    with Q._con() as con:
        for c in sorted(set(t["code"])):
            con.execute("DELETE FROM q_log WHERE code=?", (c,))
    Q._load_cached.cache_clear()
    print("已清除这些基金的抓取缓存标记")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
