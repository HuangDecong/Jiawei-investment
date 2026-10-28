# -*- coding: utf-8 -*-
"""导出「基金 × 申万一级 × 报告期」的聚合面板，供无数据库环境使用。

为什么需要
----------
完整持仓明细入库后是 218 万行、504MB，超过 GitHub 单文件 100MB 上限，不能进仓库。
但页面上的分歧度只需要「每只基金每期在每个行业上的占比 + 该期覆盖率」，
聚合后约 60 万行、压缩后 5–10MB，可以随仓库走。

用法：python tools/export_panel.py
输出：data/derived/fund_industry_panel.csv.gz
     列：code, quarter, industry, weight_pct, coverage
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "db", "fund_ak.db")
OUT = os.path.join(ROOT, "data", "derived", "fund_industry_panel.csv.gz")


def main():
    if not os.path.exists(DB):
        print("数据库不存在，先运行 tools/run_ak_all.py")
        return
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    print("聚合中……")
    d = pd.read_sql(
        "SELECT code, quarter, industry, SUM(weight_pct) AS weight_pct "
        "FROM fund_holdings WHERE industry IS NOT NULL GROUP BY code, quarter, industry",
        con)
    cov = pd.read_sql("SELECT code, quarter, coverage FROM coverage", con)
    con.close()
    d = d.merge(cov, on=["code", "quarter"], how="left")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d.to_csv(OUT, index=False, compression="gzip", encoding="utf-8")
    mb = os.path.getsize(OUT) / 1e6
    print(f"已导出 {len(d):,} 行 → {OUT}（{mb:.1f} MB）")
    print(f"期次 {d['quarter'].nunique()}、基金 {d['code'].nunique()}、"
          f"行业 {d['industry'].nunique()}")


if __name__ == "__main__":
    main()
