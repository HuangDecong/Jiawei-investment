# -*- coding: utf-8 -*-
"""补全基金档案里缺失的名称与类型（东财名录一次请求即可）。

为什么单独做
------------
`fund_profile` 里有 368 只基金 name/ftype 为空（历史抓取时雪球页缺字段）。
它们仍在样本池里（332 只）、也出现在「④ 单只基金」的下拉里，显示成
「880006　—　—」——不是分析错误，但看起来像数据缺了一块。

数据来源
--------
东方财富基金名录 `fund_name_em`：基金代码、简称、类型，一次返回 2.8 万条。
只补空值，不覆盖已有名称；类型按本库命名习惯归一（混合型-灵活 → 混合型-灵活配置）。

用法
----
    python tools/fix_profile_names.py --dry   # 只看会补多少
    python tools/fix_profile_names.py
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

DB = os.path.join(ROOT, "data", "db", "fund_ak.db")
OUT = os.path.join(ROOT, "data", "derived")

_TYPE_MAP = {
    "混合型-灵活": "混合型-灵活配置",
    "股票型": "股票型-普通",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    import akshare as ak

    d = ak.fund_name_em()
    d["基金代码"] = d["基金代码"].astype(str).str.zfill(6)
    d = d.drop_duplicates("基金代码").set_index("基金代码")

    con = sqlite3.connect(DB)
    p = pd.read_sql("SELECT code, name, ftype FROM fund_profile", con,
                    dtype={"code": str})
    miss = p[p["name"].isna() | (p["name"].astype(str).str.strip() == "")]
    hit = [c for c in miss["code"] if c in d.index]
    print(f"缺名称 {len(miss)} 只；名录可补 {len(hit)} 只；"
          f"仍缺 {len(miss) - len(hit)} 只")
    if args.dry or not hit:
        con.close()
        return

    rows = []
    for c in hit:
        r = d.loc[c]
        nm = str(r["基金简称"]).strip()
        ft = str(r["基金类型"]).strip()
        ft = _TYPE_MAP.get(ft, ft)
        rows.append((nm, nm, ft, c))
    with con:
        con.executemany(
            "UPDATE fund_profile SET name=?, full_name=?, ftype=? "
            "WHERE code=? AND (name IS NULL OR TRIM(name)='')", rows)
    left = con.execute(
        "SELECT COUNT(*) FROM fund_profile WHERE name IS NULL OR TRIM(name)=''"
    ).fetchone()[0]
    print(f"已补 {len(rows)} 只；补后仍缺 {left} 只")

    # 同步线上兜底快照（与 tools/export_canon.py 完全同一套取数与合并方式）
    p = pd.read_sql("SELECT code, name, ftype, inception, scale_yi, company, "
                    "managers, benchmark FROM fund_profile", con, dtype={"code": str})
    m = pd.read_sql("SELECT code, type FROM fund_meta", con, dtype={"code": str})
    p = p.merge(m, on="code", how="outer")
    fn = os.path.join(OUT, "canon_profile.csv.gz")
    p.to_csv(fn, index=False, compression="gzip", encoding="utf-8")
    print(f"快照已更新：{fn}（{len(p):,} 行）")
    con.close()


if __name__ == "__main__":
    main()
