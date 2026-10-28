"""补「中间缺期」：基金在该年有部分季度、但漏了某几季。

为什么需要单独补
----------------
管道按「年份」判断是否已抓（`done` 是 `quarter LIKE '2024%'` 的 code 集合）。
只要一只基金在 2024 年有任意一期入库，整只基金就被标记为「该年已完成」，
于是它漏掉的那几季永远不会被补——实测有 77 只基金、261 个基金×期次属这种情况。

做法
----
1. 读库，算出每只基金在「首次期次 → 末次期次」区间内缺哪些季；
2. 按年份聚合，每年调一次 `fetch(force=True, codes=...)`；
3. 补完再算一次，打印剩余缺口。

只补缺口、不动已有数据：`fetch` 内部是 `INSERT OR REPLACE`，
重抓整年不会产生重复行，也不会删除已有期次。
"""
from __future__ import annotations

import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

DB = os.path.join(ROOT, "data", "db", "fund_ak.db")


def gaps() -> tuple[list[str], dict[str, list[str]]]:
    """返回 (全部期次, {年份: [需补的基金代码]})。"""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    qs = [r[0] for r in con.execute(
        "SELECT DISTINCT quarter FROM fund_holdings ORDER BY quarter")]
    d = con.execute("SELECT code, quarter FROM fund_holdings").fetchall()
    con.close()

    have: dict[str, set[str]] = {}
    for code, q in d:
        have.setdefault(code, set()).add(q)

    by_year: dict[str, set[str]] = {}
    for code, s in have.items():
        idx = sorted(qs.index(q) for q in s if q in qs)
        if not idx:
            continue
        lo, hi = idx[0], idx[-1]
        for k in range(lo, hi + 1):
            if qs[k] not in s:
                by_year.setdefault(qs[k][:4], set()).add(code)
    return qs, {y: sorted(c) for y, c in by_year.items()}


def main() -> None:
    from ak_fund_pipeline import fetch

    qs, by_year = gaps()
    total = sum(len(v) for v in by_year.values())
    print(f"期次 {len(qs)}（{qs[0]}–{qs[-1]}），待补 {total} 个基金×年：")
    for y in sorted(by_year):
        print(f"  {y}: {len(by_year[y])} 只")

    for y in sorted(by_year):
        codes = ",".join(by_year[y])
        print(f"\n===== 补 {y}（{len(by_year[y])} 只）=====", flush=True)
        fetch(year=y, codes=codes, force=True)

    qs2, by_year2 = gaps()
    left = sum(len(v) for v in by_year2.values())
    print(f"\n补完：剩余缺口 {left} 个基金×年")
    if left:
        for y in sorted(by_year2):
            print(f"  仍缺 {y}: {len(by_year2[y])} 只")


if __name__ == "__main__":
    main()
