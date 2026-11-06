# -*- coding: utf-8 -*-
"""修 `coverage` 表：让它真的等于该期明细的 Σ占净值比。

为什么需要修
------------
`coverage` 的文档定义是「每只基金每期 Σ占净值比」（见 ak_fund_pipeline 开头）。
A/C 份额合并修复之前，一部分基金把两个份额类别的明细各加了一遍，
coverage 约为真实值的 2 倍——例如 169102 2019Q1：表里 104.04%，
明细实际只有 50.69%（前十大的量级），比值 2.05。

这个错值不是「看着别扭」而已，它会往下游传：
  · `ak_data.build_industry_weights(normalize=True)` 拿它当归一化分母，
    行业占比会被整体压小一半，分歧度、行业权重都跟着错；
  · `audit_selfcheck` 的「覆盖率 = 该基金该期明细合计」直接失败（639 组）。

做法
----
  1) 先把整张表备份成 `coverage_bak_YYYYMMDD`（已存在就不再覆盖，
     保留第一次修复前的原貌，方便回滚）；
  2) 只改 |coverage − Σweight| > 0.05 的 (code, quarter)，
     n_stock / n_hk 一并按明细重算；
  3) 明细里根本没有的 (code, quarter) 不动——那是「没抓过明细」，
     不是「算错了」，不能用 0 覆盖掉原始披露值。

用法
----
    python tools/repair_coverage.py            # 只报告，不写库
    python tools/repair_coverage.py --apply    # 写回并打印前后对比
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "db", "fund_ak.db")
TOL = 0.05          # 0.05pp 以内算浮点误差，不动


def _diff_sql() -> str:
    """不一致的 (code, quarter)：明细存在、但 coverage 与 Σweight 差 > TOL。"""
    return f"""
    SELECT c.code, c.quarter, c.coverage, c.n_stock, c.n_hk,
           s.s, s.n, s.nhk
    FROM coverage c
    JOIN (SELECT code, quarter, SUM(weight_pct) s, COUNT(*) n,
                 SUM(COALESCE(is_hk, 0)) nhk
          FROM fund_holdings GROUP BY code, quarter) s
      ON s.code = c.code AND s.quarter = c.quarter
    WHERE ABS(c.coverage - s.s) > {TOL}
    """


def report(con: sqlite3.Connection) -> list[tuple]:
    rows = con.execute(_diff_sql()).fetchall()
    if not rows:
        print("✅ coverage 与明细合计一致（容差 %.2fpp），无需修复" % TOL)
        return rows
    d = [r[2] - r[5] for r in rows]          # 表内 − 明细
    dbl = sum(1 for r in rows if r[2] > 1.5 * r[5] and r[5] > 1)
    print(f"发现 {len(rows):,} 组不一致（共 "
          f"{con.execute('SELECT COUNT(*) FROM coverage').fetchone()[0]:,} 组）")
    print(f"  表内偏高（旧值 > 明细）{sum(1 for x in d if x > 0):,} 组，"
          f"偏低 {sum(1 for x in d if x < 0):,} 组")
    print(f"  其中「约 2 倍」形态（A/C 双计）{dbl:,} 组")
    print(f"  偏差中位 {sorted(d)[len(d) // 2]:+.2f}pp，"
          f"最大 {max(d, key=abs):+.2f}pp")
    per_q: dict[str, int] = {}
    for r in rows:
        per_q[r[1]] = per_q.get(r[1], 0) + 1
    top = sorted(per_q.items(), key=lambda kv: -kv[1])[:5]
    print("  最集中的 5 期：" + "、".join(f"{q} {n} 组" for q, n in top))
    return rows


def apply(con: sqlite3.Connection, rows: list[tuple]) -> None:
    tag = dt.date.today().strftime("%Y%m%d")
    bak = f"coverage_bak_{tag}"
    exists = con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (bak,)).fetchone()[0]
    if not exists:
        con.execute(f"CREATE TABLE {bak} AS SELECT * FROM coverage")
        con.commit()
        print(f"已备份原表 → {bak}")
    else:
        print(f"备份表 {bak} 已存在，保留最早的原貌")
    con.executemany(
        "UPDATE coverage SET coverage=?, n_stock=?, n_hk=? "
        "WHERE code=? AND quarter=?",
        [(round(r[5], 4), int(r[6]), int(r[7]), r[0], r[1]) for r in rows])
    con.commit()
    print(f"已写回 {len(rows):,} 组")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="写回数据库")
    args = ap.parse_args()
    if not os.path.exists(DB):
        print(f"找不到数据库：{DB}")
        return 1
    con = sqlite3.connect(DB, timeout=60)
    try:
        rows = report(con)
        if rows and args.apply:
            apply(con, rows)
            print("—— 修复后复核 ——")
            left = report(con)
            return 0 if not left else 1
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
