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


def connect(retries: int = 8):
    """并发安全的连接。

    坑：`PRAGMA journal_mode=WAL` 需要短暂的排他锁，5 个进程同时启动会互相撞，
    直接抛 `database is locked` 让整个分片退出。做法是：
    ① 先由单进程把库切成 WAL（见 `init_wal()`），后续连接不再切换；
    ② 建表与连接整体加重试 + 退避，最坏情况也能等到锁释放。
    """
    os.makedirs(DB_DIR, exist_ok=True)
    last = None
    for i in range(retries):
        try:
            con = sqlite3.connect(DB_PATH, timeout=90)
            con.execute("PRAGMA busy_timeout=90000")
            try:
                con.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                pass  # 已是 WAL 或暂时拿不到锁，不阻塞启动
            con.execute("PRAGMA synchronous=NORMAL")
            # 自动提交：每条语句独立短事务，写完立刻释放写锁。
            # 不用长事务是因为两个连接同时持读锁再升级写锁时，
            # SQLite 的 busy handler 不会等待，会立刻抛 database is locked。
            con.isolation_level = None
            con.executescript(SCHEMA)
            return con
        except sqlite3.OperationalError as e:
            last = e
            time.sleep(1 + i * 1.5)
    raise last


def init_wal():
    """单进程调用一次，把库切到 WAL 模式。"""
    con = sqlite3.connect(DB_PATH, timeout=60)
    mode = con.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    con.close()
    print(f"journal_mode = {mode}")


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


def norm_stock_code(x) -> tuple[str, int]:
    """把东财返回的股票代码规范化，返回 (代码, 是否港股)。

    坑：东财的「股票代码」有时是数值型（920045.0），有时是 5 位港股（700 / 00700）。
    早期一律 `zfill(6)` 会把 00700 变成 000700——正好撞上 A 股模塑科技（真实 000700），
    于是腾讯控股被记成了「汽车」行业的模塑科技。规则改为：
      · 去掉小数点与小数尾（920045.0 → 920045）
      · 只留数字；长度 ≤5 视为港股 → 补足 5 位
      · 长度 6 视为 A 股 → 补足 6 位
    """
    import re
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"\D", "", s)
    if not digits:
        return "", 1
    if len(digits) <= 5:
        return digits.zfill(5), 1
    return digits.zfill(6), 0


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


def fetch(limit: int | None = None, year: str = "2026", codes: str | None = None,
          shard: int = 0, nshards: int = 1, force: bool = False):
    import akshare as ak
    if not os.path.exists(MAP_PATH):
        print("缺少申万映射，先运行 --build-map")
        return
    mp = json.load(open(MAP_PATH, encoding="utf-8"))
    con = connect()
    # 断点续传必须按「年份」判断：一只基金 2026 年抓过，不代表 2024 年也抓过。
    # 早期版本按全库 code 去重，导致换年份后 todo 为空、静默跳过。
    done = set() if force else {r[0] for r in con.execute(
        "SELECT DISTINCT code FROM fund_holdings WHERE quarter LIKE ?", (f"{year}%",))}
    names = _dedup_funds(ak.fund_name_em())
    todo = [r for r in names.itertuples(index=False) if r[0] not in done]
    if codes:
        want = [c.strip() for c in codes.split(",") if c.strip()]
        meta = {r[0]: (r[0], r[1], r[2]) for r in names.itertuples(index=False)}
        todo = [meta.get(c, (c, c, "指定")) for c in want if c not in done]
    todo = sorted(todo, key=lambda x: x[0])
    if nshards > 1:
        # 按代码排序后取模分片：多进程各抓一份，互不重叠、天然可续跑
        todo = [t for i, t in enumerate(todo) if i % nshards == shard]
    if limit:
        todo = todo[:limit]
    print(f"[分片 {shard}/{nshards}] 待抓 {len(todo)} 只（该年已完成 {len(done)} 只），年份 {year}",
          flush=True)
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
        rows = []
        cov_rows = []
        for q, g in d.groupby("q"):
            cov = float(pd.to_numeric(g["占净值比例"], errors="coerce").fillna(0).sum())
            n_hk = int(sum(1 for x in g["股票代码"] if norm_stock_code(x)[1]))
            cov_rows.append((code, q, round(cov, 4), len(g), n_hk))
            for r in g.itertuples(index=False):
                sc, is_hk = norm_stock_code(getattr(r, "股票代码"))
                # 行业只对 6 位 A 股代码查表；港股没有申万一级，留空
                ind = None if is_hk else mp.get(sc)
                rows.append((code, name, q, sc, getattr(r, "股票名称"),
                             float(pd.to_numeric(getattr(r, "占净值比例"), errors="coerce") or 0),
                             float(pd.to_numeric(getattr(r, "持股数"), errors="coerce") or 0),
                             float(pd.to_numeric(getattr(r, "持仓市值"), errors="coerce") or 0),
                             ind, is_hk, "eastmoney_akshare",
                             dt.datetime.now().isoformat(timespec="seconds")))

        def _write():
            for cr in cov_rows:
                con.execute("INSERT OR REPLACE INTO coverage VALUES (?,?,?,?,?)", cr)
            if rows:
                con.executemany(
                    "INSERT OR REPLACE INTO fund_holdings VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                con.execute("INSERT OR REPLACE INTO fund_meta VALUES (?,?,?,?)",
                            (code, name, typ, dt.datetime.now().isoformat(timespec="seconds")))
            con.execute("INSERT INTO fetch_log (code,status,rows,ts,note) VALUES (?,?,?,?,?)",
                        (code, "ok" if rows else "empty", len(rows),
                         dt.datetime.now().isoformat(timespec="seconds"), ""))

        for attempt in range(5):
            try:
                _write()
                if rows:
                    ok += 1
                break
            except sqlite3.OperationalError:
                time.sleep(1 + attempt)
        else:
            # 5 次都写不进去：记失败但不中断分片，下一轮补跑会重新抓这只
            fail += 1
            try:
                con.execute("INSERT INTO fetch_log (code,status,rows,ts,note) "
                            "VALUES (?,?,?,?,?)",
                            (code, "db_locked", len(rows),
                             dt.datetime.now().isoformat(timespec="seconds"), "写入重试失败"))
            except Exception:
                pass
        if i % 20 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  成功 {ok} 失败 {fail}  "
                  f"{el:.0f}s  预计剩余 {el / i * (len(todo) - i) / 60:.0f} 分钟", flush=True)
    _safe_commit(con)
    con.close()
    print(f"完成：成功 {ok}、空/失败 {fail}，耗时 {(time.time()-t0)/60:.1f} 分钟")


def _safe_commit(con, retries: int = 6) -> bool:
    """并发提交：撞锁时退避重试，失败只丢这一批，不让整个分片退出。"""
    for i in range(retries):
        try:
            con.commit()
            return True
        except sqlite3.OperationalError:
            time.sleep(0.5 + i)
    return False


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
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--force", action="store_true",
                    help="忽略断点、强制重抓（补中间缺期用）")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.build_map:
        build_map()
    elif a.fetch:
        fetch(a.limit, a.year, a.codes, a.shard, a.nshards, a.force)
    elif a.report:
        report()
    else:
        ap.print_help()
