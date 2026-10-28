# -*- coding: utf-8 -*-
"""基金持仓批量管道（AKShare / 东方财富公开口径）。

为什么新增这条管道
------------------
Wind MCP 只能按「单基金 × 单期」查询，全市场 4.6 万次跑不完。
AKShare 直连东方财富公开披露数据，单基金一次返回 2026 年各季度全部股票投资明细，
约 0.3 秒/只，全市场 3000–4000 只可在半小时内完成。东财口径与 Wind 同为
「季报/半年报披露的前十大或全持仓 + 占净值比」，可交叉验证。

口径与边界
----------
· 季报期（Q1/Q3）只披露前十大，半年报/年报（Q2/Q4）披露全部（东财截断到 100 只）；
  因此每只基金每期都要记录 coverage = Σ占净值比，跨期比较必须先归一化。
· 东财明细含港股（5 位代码），申万一级只覆盖 A 股，港股在行业汇总时剔除，
  但仍计入 coverage（因为它占净值）。
· 股票→申万一级映射来自申万指数成分（AKShare `index_component_sw`），
  映射文件落盘后可复核。

命令
----
    python tools/ak_fund_pipeline.py --build-map
    python tools/ak_fund_pipeline.py --fetch --limit 200
    python tools/ak_fund_pipeline.py --report
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(ROOT, "data", "db")
DB_PATH = os.path.join(DB_DIR, "fund_ak.db")
MAP_PATH = os.path.join(ROOT, "data", "raw", "sw_industry_map.json")

TYPES = ("混合型-偏股", "混合型-灵活", "股票型")

SCHEMA = """
CREATE TABLE IF NOT EXISTS fund_holdings (
    code TEXT, name TEXT, quarter TEXT,
    stock_code TEXT, stock_name TEXT,
    weight_pct REAL, shares REAL, mv REAL,
    industry TEXT, is_hk INTEGER,
    source TEXT, fetched_at TEXT,
    PRIMARY KEY (code, quarter, stock_code)
);
CREATE INDEX IF NOT EXISTS idx_fh_q ON fund_holdings(quarter);
CREATE INDEX IF NOT EXISTS idx_fh_code ON fund_holdings(code);

CREATE TABLE IF NOT EXISTS fund_meta (
    code TEXT PRIMARY KEY, name TEXT, type TEXT, fetched_at TEXT
);
CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT, status TEXT, rows INTEGER, ts TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS coverage (
    code TEXT, quarter TEXT, coverage REAL, n_stock INTEGER, n_hk INTEGER,
    PRIMARY KEY (code, quarter)
);
"""


def connect():
    os.makedirs(DB_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    return con


def build_map():
    import akshare as ak
    sw = ak.sw_index_first_info()
    mp = {}
    for _, r in sw.iterrows():
        code = str(r["行业代码"]).split(".")[0]
        name = r["行业名称"]
        try:
            c = ak.index_component_sw(symbol=code)
            for _, x in c.iterrows():
                mp[str(x["证券代码"]).zfill(6)] = name
        except Exception as e:
            print(f"  映射失败 {name}: {e.__class__.__name__}")
    os.makedirs(os.path.dirname(MAP_PATH), exist_ok=True)
    json.dump(mp, open(MAP_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"申万映射：{len(mp)} 只股票 / {len(set(mp.values()))} 个行业 → {MAP_PATH}")


def _quarter(label: str) -> str | None:
    """'2026年2季度股票投资明细' → '2026Q2'。"""
    import re
    m = re.search(r"(\d{4})年(\d)季度", str(label))
    return f"{m.group(1)}Q{m.group(2)}" if m else None


def _dedup_funds(df: pd.DataFrame) -> pd.DataFrame:
    """同一基金多份额（A/C/E）只保留一条：按简称去掉末尾份额字母后去重。"""
    import re
    d = df[df["基金类型"].isin(TYPES)].copy()
    d["base"] = d["基金简称"].astype(str).str.replace(
        r"[（(]?[A-EI-Z][)）]?$", "", regex=True).str.strip()
    d = d.drop_duplicates(subset=["base"], keep="first")
    return d[["基金代码", "基金简称", "基金类型"]]


def fetch(limit: int | None = None, year: str = "2026", codes: str | None = None):
    import akshare as ak
    if not os.path.exists(MAP_PATH):
        print("缺少申万映射，先运行 --build-map")
        return
    mp = json.load(open(MAP_PATH, encoding="utf-8"))
    con = connect()
    done = {r[0] for r in con.execute("SELECT DISTINCT code FROM fund_holdings")}
    names = _dedup_funds(ak.fund_name_em())
    todo = [r for r in names.itertuples(index=False) if r[0] not in done]
    if codes:
        want = [c.strip() for c in codes.split(",") if c.strip()]
        meta = {r[0]: (r[0], r[1], r[2]) for r in names.itertuples(index=False)}
        todo = [meta.get(c, (c, c, "指定")) for c in want if c not in done]
    if limit:
        todo = todo[:limit]
    print(f"待抓 {len(todo)} 只（已完成 {len(done)} 只），目标年份 {year}")
    ok = fail = 0
    t0 = time.time()
    for i, (code, name, typ) in enumerate(todo, 1):
        rows = []
        for attempt in range(2):
            try:
                d = ak.fund_portfolio_hold_em(symbol=code, date=year)
                break
            except Exception:
                d = pd.DataFrame()
                time.sleep(0.3)
        if d.empty:
            fail += 1
            con.execute("INSERT INTO fetch_log (code,status,rows,ts,note) VALUES (?,?,?,?,?)",
                        (code, "empty", 0, dt.datetime.now().isoformat(timespec="seconds"), ""))
            continue
        d["q"] = d["季度"].map(_quarter)
        d = d.dropna(subset=["q"])
        for q, g in d.groupby("q"):
            cov = float(pd.to_numeric(g["占净值比例"], errors="coerce").fillna(0).sum())
            n_hk = int(g["股票代码"].astype(str).str.len().ne(6).sum())
            con.execute("INSERT OR REPLACE INTO coverage VALUES (?,?,?,?,?)",
                        (code, q, round(cov, 4), len(g), n_hk))
            for r in g.itertuples(index=False):
                sc = str(getattr(r, "股票代码")).zfill(6)
                ind = mp.get(sc)
                rows.append((code, name, q, sc, getattr(r, "股票名称"),
                             float(pd.to_numeric(getattr(r, "占净值比例"), errors="coerce") or 0),
                             float(pd.to_numeric(getattr(r, "持股数"), errors="coerce") or 0),
                             float(pd.to_numeric(getattr(r, "持仓市值"), errors="coerce") or 0),
                             ind, 0 if ind else 1, "eastmoney_akshare",
                             dt.datetime.now().isoformat(timespec="seconds")))
        if rows:
            con.executemany(
                "INSERT OR REPLACE INTO fund_holdings VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            con.execute("INSERT OR REPLACE INTO fund_meta VALUES (?,?,?,?)",
                        (code, name, typ, dt.datetime.now().isoformat(timespec="seconds")))
            ok += 1
        con.execute("INSERT INTO fetch_log (code,status,rows,ts,note) VALUES (?,?,?,?,?)",
                    (code, "ok" if rows else "empty", len(rows),
                     dt.datetime.now().isoformat(timespec="seconds"), ""))
        if i % 50 == 0:
            con.commit()
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  成功 {ok} 失败 {fail}  "
                  f"{el:.0f}s  预计剩余 {el / i * (len(todo) - i) / 60:.0f} 分钟")
    con.commit()
    con.close()
    print(f"完成：成功 {ok}、空/失败 {fail}，耗时 {(time.time()-t0)/60:.1f} 分钟")


def report():
    con = connect()
    cov = pd.read_sql(
        "SELECT quarter, COUNT(DISTINCT code) 基金数, ROUND(AVG(coverage),1) 平均覆盖率,"
        " SUM(n_stock) 记录数 FROM coverage GROUP BY quarter ORDER BY quarter", con)
    print("逐期覆盖：\n", cov.to_string(index=False))
    tot = con.execute("SELECT COUNT(DISTINCT code) FROM fund_holdings").fetchone()[0]
    print(f"\n已入库基金 {tot} 只")
    ind = pd.read_sql(
        "SELECT COUNT(*) n, SUM(industry IS NULL) 未映射 FROM fund_holdings", con)
    print("记录：", ind.to_dict("records"))
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-map", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--year", default="2026")
    ap.add_argument("--codes", default=None, help="逗号分隔的基金代码，只抓这些")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.build_map:
        build_map()
    elif a.fetch:
        fetch(a.limit, a.year, a.codes)
    elif a.report:
        report()
    else:
        ap.print_help()
