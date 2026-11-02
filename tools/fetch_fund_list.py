# -*- coding: utf-8 -*-
"""刷新全市场基金名录（用于「样本漏斗」的分母）。

数据来源：东方财富基金列表（akshare.fund_name_em），含所有份额代码。
产出：data/derived/fund_list.csv.gz（约 280KB，随仓库发布）。
canon.market_fund_counts() 读它算「全市场基金数 / 主动权益基金数」。

什么时候需要重跑：新发/清盘会让名录每季变动几十只；
如果「样本漏斗」的 15,5xx 只与交易所/中基协公布的数量差得较远，就重跑一次。

用法：python tools/fetch_fund_list.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "derived", "fund_list.csv.gz")


def main() -> None:
    try:
        import akshare as ak
    except Exception as e:
        print("需要 akshare：pip install akshare  （", type(e).__name__, "）")
        return
    d = ak.fund_name_em()
    d = (d[["基金代码", "基金简称", "基金类型"]]
         .rename(columns={"基金代码": "code", "基金简称": "name", "基金类型": "ftype"}))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d.to_csv(OUT, index=False, compression="gzip", encoding="utf-8")
    print(f"fund_list.csv.gz  {len(d):,} 条份额  {os.path.getsize(OUT)/1e3:.0f} KB")

    base = (d["name"].astype(str).str.replace(r"[A-Z]$", "", regex=True)
            .str.replace(r"\((后端|前端)\)$", "", regex=True))
    act = d[d["ftype"].isin(["股票型", "混合型-偏股", "混合型-灵活", "混合型-平衡"])]
    print(f"去份额后缀后：全市场 {base.nunique():,} 只；"
          f"主动权益 {act.assign(b=base[act.index]).b.nunique():,} 只")


if __name__ == "__main__":
    main()
