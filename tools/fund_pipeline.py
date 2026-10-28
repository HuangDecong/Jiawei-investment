# -*- coding: utf-8 -*-
"""机构持仓数据管道（半自动，可断点续传）。

为什么是「半自动」
------------------
本项目运行环境没有 WindPy / iFinDPy（已实测：两个模块均未安装，仅有
C:/Wind/Wind.NET.Client 客户端），因此无法在 Python 进程内直接调用 Wind。
取数经 MCP 工具完成，本脚本负责：
    ① 生成待取清单（--plan）
    ② 把 MCP 返回落盘的 JSON 入库、校验、写日志（--ingest）
    ③ 输出覆盖统计与缺口（--report）
断点续传靠「已入库季度集合」实现：重复执行 --ingest 不会重复写，缺失季度会被列出。
若后续拿到 WindPy 授权，只需实现 `fetch_quarter_windpy()` 并把返回的
rows 交给 `_ingest_quarter()`，即可变成一条命令全自动。

数据表
------
stock_holdings  季度、排名、代码、名称、持股基金数、持仓市值（亿元）
fetch_log       每次取数的任务、季度、查询、时间、行数、状态、备注
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(ROOT, "data", "db")
DB_PATH = os.path.join(DB_DIR, "institution.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS stock_holdings (
    quarter   TEXT NOT NULL,
    rank      INTEGER,
    code      TEXT,
    name      TEXT NOT NULL,
    n_funds   INTEGER,
    mv_yi     REAL,
    source    TEXT,
    fetched_at TEXT,
    PRIMARY KEY (quarter, name)
);
CREATE INDEX IF NOT EXISTS idx_sh_quarter ON stock_holdings(quarter);
CREATE INDEX IF NOT EXISTS idx_sh_name ON stock_holdings(name);

CREATE TABLE IF NOT EXISTS fetch_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    task      TEXT,
    quarter   TEXT,
    query     TEXT,
    ts        TEXT,
    rows      INTEGER,
    status    TEXT,
    note      TEXT
);
"""


def connect() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    return con


def quarter_list(start: str = "2016Q1", end: str = "2026Q3") -> list[str]:
    def key(q):
        y, _, n = q.partition("Q")
        return int(y), int(n)
    out = []
    y, n = key(start)
    ey, en = key(end)
    while (y, n) <= (ey, en):
        out.append(f"{y}Q{n}")
        n += 1
        if n == 5:
            y, n = y + 1, 1
    return out


def _ingest_quarter(con: sqlite3.Connection, q: str, rows: list[dict],
                    source: str, queried_at: str) -> tuple[int, str]:
    """入库单个季度并校验。返回 (写入行数, 状态说明)。"""
    if not rows:
        return 0, "空返回"
    ok = [r for r in rows
          if r.get("name") and (r.get("n_funds") or 0) > 0 and (r.get("mv_yi") or 0) > 0]
    if len(ok) < 10:
        return 0, f"有效行不足（{len(ok)}/{len(rows)}）"
    con.executemany(
        """INSERT INTO stock_holdings
           (quarter, rank, code, name, n_funds, mv_yi, source, fetched_at)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(quarter, name) DO UPDATE SET
             rank=excluded.rank, code=excluded.code,
             n_funds=excluded.n_funds, mv_yi=excluded.mv_yi,
             source=excluded.source, fetched_at=excluded.fetched_at""",
        [(q, r.get("rank"), r.get("code", ""), r["name"],
          int(r["n_funds"]), float(r["mv_yi"]), source, queried_at) for r in ok])
    return len(ok), "ok"


def ingest(path: str) -> None:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    quarters = payload.get("quarters", {})
    source = payload.get("source", "Wind")
    fetched_at = payload.get("query_date", dt.date.today().isoformat())
    con = connect()
    total = 0
    for q in sorted(quarters, key=lambda x: (int(x[:4]), int(x[-1]))):
        rows = quarters[q]
        n, status = _ingest_quarter(con, q, rows, source, fetched_at)
        total += n
        con.execute(
            "INSERT INTO fetch_log (task,quarter,query,ts,rows,status,note) VALUES (?,?,?,?,?,?,?)",
            ("stock_holdings", q, "前50只重仓股（按持股基金数）",
             dt.datetime.now().isoformat(timespec="seconds"), n, status, ""))
    for q, err in (payload.get("errors") or {}).items():
        con.execute(
            "INSERT INTO fetch_log (task,quarter,query,ts,rows,status,note) VALUES (?,?,?,?,?,?,?)",
            ("stock_holdings", q, "前50只重仓股（按持股基金数）",
             dt.datetime.now().isoformat(timespec="seconds"), 0, "failed", str(err)[:200]))
    con.commit()
    con.close()
    print(f"入库完成：{len(quarters)} 个季度、{total} 行 → {DB_PATH}")


def report() -> None:
    con = connect()
    df = con.execute(
        """SELECT quarter, COUNT(*) n, MIN(n_funds) nf_min, MAX(n_funds) nf_max,
                  ROUND(SUM(mv_yi),1) mv_sum
           FROM stock_holdings GROUP BY quarter ORDER BY quarter""").fetchall()
    if not df:
        print("库存为空。先运行 --ingest。")
        return
    print(f"{'季度':8s} {'行数':>4s} {'基金数范围':>14s} {'持仓市值合计(亿)':>16s}")
    for q, n, lo, hi, mv in df:
        print(f"{q:8s} {n:>4d} {lo:>6d}–{hi:<6d} {mv:>16.1f}")
    have = {r[0] for r in df}
    missing = [q for q in quarter_list() if q not in have]
    print(f"\n覆盖 {len(have)}/{len(quarter_list())} 期；缺失：{missing[:12]}"
          f"{'...' if len(missing) > 12 else ''}")
    bad = con.execute(
        "SELECT quarter,status,note FROM fetch_log WHERE status!='ok' ORDER BY quarter").fetchall()
    if bad:
        print("\n失败/跳过记录：")
        for q, s, note in bad[:20]:
            print(f"  {q} {s} {note}")
    con.close()


def plan(start: str, end: str) -> None:
    con = connect()
    have = {r[0] for r in con.execute("SELECT DISTINCT quarter FROM stock_holdings")}
    con.close()
    todo = [q for q in quarter_list(start, end) if q not in have]
    print(f"待取 {len(todo)} 期：{todo}")
    print("\n问句模板（每年替换季度中文）：")
    print('查询{YYYY}年{一/二/三/四}季报，普通股票型基金、偏股混合型基金、'
          '灵活配置型基金中，被最多基金持有的前50只股票，列出股票代码、股票名称、'
          '持股基金数、持仓市值合计')


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="机构持仓数据管道（半自动）")
    ap.add_argument("--ingest", metavar="JSON", help="把 MCP 落盘的 JSON 入库")
    ap.add_argument("--report", action="store_true", help="覆盖统计与缺口")
    ap.add_argument("--plan", action="store_true", help="列出待取季度")
    ap.add_argument("--from", dest="start", default="2016Q1")
    ap.add_argument("--to", dest="end", default="2026Q3")
    a = ap.parse_args()
    if a.ingest:
        ingest(a.ingest)
    elif a.report:
        report()
    elif a.plan:
        plan(a.start, a.end)
    else:
        ap.print_help()
