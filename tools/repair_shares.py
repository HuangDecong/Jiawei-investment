# -*- coding: utf-8 -*-
"""修复「份额变动」口径：把按份额类别返回的数据合并成基金整体。

问题
----
东财 gmbd 接口按**份额类别**返回（A 类、C 类各自一份），
而同一代码的资产配置页（zcpz）按**基金整体**返回。
直接用 gmbd 的「期末净资产」做规模拆解，会得到
「净资产 67 亿 ÷ 份额 0.03 亿份 = 2240 元/份」这种明显错误的数。

判定
----
同一 (code, period) 上，`q_shares.end_nav_yi` 应等于 `q_asset.nav_yi`
（都是基金整体净资产）。偏差超过 0.5% 或份额侧为 0，就判定为口径不一致。

修复
----
重新调用 `qreport._f10_shares`（内部按 MERGECODE 合并全部份额类别）并覆盖写库。

用法：
    python tools/repair_shares.py --dry      # 只统计
    python tools/repair_shares.py            # 全量修复（约 5–10 分钟）
    python tools/repair_shares.py --limit 50 # 先修 50 只试试
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd  # noqa: E402

import qreport as Q  # noqa: E402


def bad_codes() -> pd.DataFrame:
    Q.ensure_db()
    with Q._con() as con:
        a = pd.read_sql("SELECT code, period, nav_yi FROM q_asset", con)
        s = pd.read_sql("SELECT code, period, end_nav_yi FROM q_shares", con)
    m = a.merge(s, on=["code", "period"], how="inner")
    m["nav_yi"] = pd.to_numeric(m["nav_yi"], errors="coerce")
    m["end_nav_yi"] = pd.to_numeric(m["end_nav_yi"], errors="coerce")
    m = m[m["nav_yi"] > 0]
    m["r"] = m["end_nav_yi"] / m["nav_yi"]
    m["bad"] = [not Q.shares_consistent(e, n)
                for e, n in zip(m["end_nav_yi"], m["nav_yi"])]
    g = m.groupby("code")["bad"].any()
    return m[m["code"].isin(g[g].index)]


def fix_one(code: str) -> tuple[str, int, str]:
    d = Q._f10_shares(code)
    if d is None or d.empty:
        return code, 0, "空表"
    with Q._con() as con:
        n = Q._write(con, "q_shares", code, d,
                     ["purchase_yi", "redeem_yi", "end_share_yi", "end_nav_yi",
                      "nav_chg_pct"],
                     datetime.now().strftime("%Y-%m-%d %H:%M"),
                     "东财基金档案·份额变动（A/C 合并）", [])
    return code, n, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    m = bad_codes()
    codes = sorted(m["code"].unique())
    # 已经按 A/C 合并过、却仍然对不上：这是源数据本身的问题（单类基金的历史
    # 期末净资产与 zcpz 不一致、或某行字段错位），重抓也不会变，跳过。
    import sqlite3
    _con = sqlite3.connect(os.path.join(ROOT, "data", "db", "fund_q.db"))
    _mg = {r[0] for r in _con.execute(
        "SELECT DISTINCT code FROM q_shares WHERE src LIKE '%合并%'")}
    _con.close()
    _skip = [c for c in codes if c in _mg]
    codes = [c for c in codes if c not in _mg]
    print(f"跳过已合并但仍不一致（源数据缺陷）{len(_skip)} 只")
    print(f"口径不一致：{len(codes)} 只基金、{len(m)} 个基金×期")
    print(m.groupby("code").size().describe().round(1).to_string())
    if args.dry:
        print(m.sort_values("r").head(10).to_string(index=False))
        return 0
    if args.limit:
        codes = codes[:args.limit]
        print(f"只修前 {len(codes)} 只")

    ok = fail = 0
    fails = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fix_one, c): c for c in codes}
        for i, f in enumerate(as_completed(futs), 1):
            c = futs[f]
            try:
                _, n, msg = f.result()
                if n:
                    ok += 1
                else:
                    fail += 1
                    fails.append((c, msg or "无数据"))
            except Exception as e:  # noqa: BLE001
                fail += 1
                fails.append((c, f"{type(e).__name__}: {e}"))
            if i % 200 == 0 or i == len(codes):
                print(f"  {i}/{len(codes)}  成功 {ok} 失败 {fail}")

    # 清 lru_cache：否则页面还读旧快照
    Q._load_cached.cache_clear()
    print(f"完成：成功 {ok}、失败 {fail}")
    for c, msg in fails[:10]:
        print("  -", c, msg)

    # 复核
    m2 = bad_codes()
    print(f"修复后仍不一致：{m2['code'].nunique()} 只、{len(m2)} 个基金×期")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
