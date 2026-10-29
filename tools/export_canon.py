"""把 canon 需要的数据压成 4 个小文件，供无数据库环境（线上部署）使用。

为什么不能直接上传 fund_ak.db
------------------------------
完整库 500MB+，超过 GitHub 单文件 100MB 上限。而线上页面要算的东西其实只有四类：

    canon_top10     每只基金每期的「前十大 → 行业」占比（已聚合，11 行/基金/期）
    canon_full      全持仓期的「行业占净值比」（已聚合，约 12 行/基金/期）
    canon_coverage  每只基金每期的覆盖率（股票仓位）
    canon_stocks    每期个股快照（持股基金数、持股数、市值）
    canon_profile   基金档案（类型、成立日、规模、经理、比较基准）

聚合后的文本高度重复，gzip 压缩比很高，5 个文件合计约 5–8MB，
和仓库里已有的 fund_industry_panel.csv.gz 是同一思路。

用法
----
    python tools/export_canon.py
"""

from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "db", "fund_ak.db")
OUT = os.path.join(ROOT, "data", "derived")


def _write(df: pd.DataFrame, name: str) -> None:
    p = os.path.join(OUT, name)
    df.to_csv(p, index=False, compression="gzip", encoding="utf-8")
    print(f"{name:<26} {len(df):>9,} 行  {os.path.getsize(p) / 1e6:>6.2f} MB")


def main() -> None:
    if not os.path.exists(DB):
        print("未找到数据库")
        return
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    qs = [r[0] for r in con.execute(
        "SELECT DISTINCT quarter FROM fund_holdings ORDER BY quarter")]
    print(f"期次 {len(qs)}：{qs[0]} → {qs[-1]}")

    # 1) 前十大 → 行业（口径 B）
    #    先在库里按基金取前 10 只，再聚合到行业，避免把 200 万行明细搬出来
    d = pd.read_sql("""
        SELECT code, quarter, industry, SUM(weight_pct) AS w
        FROM (
            SELECT code, quarter, industry, weight_pct,
                   ROW_NUMBER() OVER (PARTITION BY code, quarter
                                      ORDER BY weight_pct DESC) rn
            FROM fund_holdings
        )
        WHERE rn <= 10 AND industry IS NOT NULL AND industry != ''
        GROUP BY code, quarter, industry
    """, con, dtype={"code": str})
    _write(d, "canon_top10.csv.gz")

    # 2) 全持仓 → 行业（口径 A，只用半年报/年报）
    fq = [q for q in qs if q[-1] in ("2", "4")]
    ph = ",".join("?" * len(fq))
    d2 = pd.read_sql(f"""
        SELECT code, quarter, industry, SUM(weight_pct) AS w
        FROM fund_holdings
        WHERE quarter IN ({ph}) AND industry IS NOT NULL AND industry != ''
        GROUP BY code, quarter, industry
    """, con, params=fq, dtype={"code": str})
    _write(d2, "canon_full.csv.gz")

    # 3) 覆盖率 = 股票仓位
    d3 = pd.read_sql(
        "SELECT code, quarter, SUM(weight_pct) AS cov FROM fund_holdings "
        "GROUP BY code, quarter", con, dtype={"code": str})
    _write(d3, "canon_coverage.csv.gz")

    # 4) 个股快照
    d4 = pd.read_sql("""
        SELECT quarter, stock_code, MAX(stock_name) AS stock_name,
               MAX(industry) AS industry, MAX(is_hk) AS is_hk,
               COUNT(DISTINCT code) AS n_funds,
               SUM(shares) AS shares, SUM(mv) AS mv
        FROM fund_holdings GROUP BY quarter, stock_code
    """, con, dtype={"stock_code": str})
    _write(d4, "canon_stocks.csv.gz")

    # 5) 基金档案 + 类型（样本池判定要用 fund_meta.type）
    try:
        p = pd.read_sql("SELECT code, name, ftype, inception, scale_yi, company, "
                        "managers, benchmark FROM fund_profile", con, dtype={"code": str})
        m = pd.read_sql("SELECT code, type FROM fund_meta", con, dtype={"code": str})
        p = p.merge(m, on="code", how="outer")
        _write(p, "canon_profile.csv.gz")
    except Exception as e:
        print("档案导出失败：", e)

    # 6) 单只基金页需要：最近两个全持仓期的前 12 大持仓（股票级）
    last2 = fq[-2:]
    ph2 = ",".join("?" * len(last2))
    d6 = pd.read_sql(f"""
        SELECT code, quarter, stock_code, stock_name, industry,
               weight_pct, shares, mv, is_hk
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY code, quarter
                                         ORDER BY weight_pct DESC) rn
            FROM fund_holdings WHERE quarter IN ({ph2})
        )
        WHERE rn <= 12
    """, con, params=last2, dtype={"code": str, "stock_code": str})
    _write(d6, "canon_fund_top10.csv.gz")

    # 7) 个股榜（前十大口径、只含样本基金）：31 期全有，环比与同比都能算
    #    为什么必须限定样本基金：不加的话指数基金、ETF 联接也会被算成
    #    「持有该股的基金数」，个股榜会被被动资金污染。
    #
    #    实现上用 pandas 逐期算，不用 SQL 的 JOIN + ROW_NUMBER：
    #    4M 行明细按 (quarter, code) 关联样本表的执行计划很差，
    #    实测 20 分钟没跑完；改成「读一期 → 过滤样本 → 组内取前 10」只要 1–2 分钟。
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import canon as cn
    rows = []
    for q in qs:
        d = pd.read_sql(
            "SELECT code, stock_code, stock_name, industry, is_hk, "
            "weight_pct, shares, mv FROM fund_holdings WHERE quarter=?",
            con, params=(q,), dtype={"code": str, "stock_code": str})
        if d.empty:
            continue
        d = d[d["code"].isin(cn.universe(q))]
        d["rn"] = d.groupby("code")["weight_pct"].rank(method="first", ascending=False)
        d = d[d["rn"] <= 10]
        g = d.groupby("stock_code").agg(
            stock_name=("stock_name", "max"), industry=("industry", "max"),
            is_hk=("is_hk", "max"), n_funds=("code", "nunique"),
            shares=("shares", "sum"), mv=("mv", "sum")).reset_index()
        g["quarter"] = q
        rows.append(g)
        print(f"  {q} 个股 {len(g):,} 只", flush=True)
    d7 = pd.concat(rows, ignore_index=True)
    _write(d7, "canon_stock_top10.csv.gz")

    con.close()
    tot = sum(os.path.getsize(os.path.join(OUT, f))
              for f in os.listdir(OUT) if f.startswith("canon_"))
    print(f"合计 {tot / 1e6:.2f} MB → {OUT}")


if __name__ == "__main__":
    main()
