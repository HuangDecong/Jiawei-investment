"""补全基金层面的属性：成立日、规模、类型、公司、经理、业绩比较基准。

为什么需要
----------
只看持仓，回答不了三个最基本的问题：
① 这只基金多大了？次新基金的建仓行为不能当观点（建仓期必须买，不是看好）；
② 它管多少钱？50 亿和 2 亿的同一笔加仓，含义完全不同；
③ 它该跟谁比？没有业绩比较基准，就无法判断「涨了」是本事还是大盘涨。

数据来源
--------
- 雪球基金档案（`fund_individual_basic_info_xq`）：成立时间、最新规模、公司、
  经理、类型、托管行、业绩比较基准、投资策略/目标；
- 东方财富经理表（`fund_manager_em`）：基金 → 经理、从业年限、在管规模。

用法
----
    python tools/fetch_fund_profile.py                # 全量（可断点续跑）
    python tools/fetch_fund_profile.py --shard 0 --nshards 5
    python tools/fetch_fund_profile.py --limit 20     # 抽样自检
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
DB = os.path.join(ROOT, "data", "db", "fund_ak.db")
RAW = os.path.join(ROOT, "data", "raw")

SCHEMA = """
CREATE TABLE IF NOT EXISTS fund_profile (
    code        TEXT PRIMARY KEY,
    name        TEXT,
    full_name   TEXT,
    inception   TEXT,      -- 成立日 YYYY-MM-DD
    scale_yi    REAL,      -- 最新规模（亿元）
    company     TEXT,
    managers    TEXT,      -- 现任经理，空格分隔
    custodian   TEXT,
    ftype       TEXT,      -- 混合型-偏股 / 股票型 ...
    benchmark   TEXT,      -- 业绩比较基准原文
    strategy    TEXT,
    target      TEXT,
    raw_json    TEXT,
    ts          TEXT
);
CREATE TABLE IF NOT EXISTS fund_manager (
    code        TEXT,
    manager     TEXT,
    company     TEXT,
    tenure_days INTEGER,
    aum_yi      REAL,      -- 该经理现任基金资产总规模（亿元）
    best_return REAL,
    ts          TEXT,
    PRIMARY KEY (code, manager)
);
"""

NFIELDS = {
    "基金代码": "code", "基金名称": "name", "基金全称": "full_name",
    "成立时间": "inception", "最新规模": "scale_yi", "基金公司": "company",
    "基金经理": "managers", "托管银行": "custodian", "基金类型": "ftype",
    "业绩比较基准": "benchmark", "投资策略": "strategy", "投资目标": "target",
}


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB, timeout=90)
    con.execute("PRAGMA busy_timeout=90000")
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    con.isolation_level = None          # 自动提交：每条语句独立短事务
    return con


def parse_scale(v) -> float | None:
    """'189.33亿' → 189.33；'5,678.90万' → 0.56789；空 → None。"""
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if not s or s in ("nan", "None", "<NA>", "--"):
        return None
    try:
        if s.endswith("亿"):
            return round(float(s[:-1]), 4)
        if s.endswith("万"):
            return round(float(s[:-1]) / 1e4, 4)
        return round(float(s), 4)
    except ValueError:
        return None


def fetch_manager(con: sqlite3.Connection, force: bool = False) -> int:
    """经理表一次全量拉取（约 10 秒），按 code+manager 去重。"""
    n = con.execute("SELECT COUNT(*) FROM fund_manager").fetchone()[0]
    if n > 0 and not force:
        print(f"经理表已有 {n} 行，跳过（--force-manager 可重拉）")
        return n
    import akshare as ak
    d = ak.fund_manager_em()
    d = d.rename(columns={"现任基金代码": "code", "姓名": "manager", "所属公司": "company",
                          "累计从业时间": "tenure_days", "现任基金资产总规模": "aum_yi",
                          "现任基金最佳回报": "best_return"})
    d["code"] = d["code"].astype(str).str.zfill(6)
    d = d.drop_duplicates(["code", "manager"])
    d["ts"] = dt.datetime.now().isoformat(timespec="seconds")
    con.executemany(
        "INSERT OR REPLACE INTO fund_manager VALUES (?,?,?,?,?,?,?)",
        d[["code", "manager", "company", "tenure_days", "aum_yi", "best_return", "ts"]]
        .itertuples(index=False, name=None))
    print(f"经理表：{len(d):,} 行（{d['code'].nunique():,} 只基金、{d['manager'].nunique():,} 位经理）")
    return len(d)


def fetch_profiles(limit: int | None = None, shard: int = 0, nshards: int = 1,
                   sleep: float = 0.05) -> None:
    import akshare as ak
    con = connect()
    codes = [r[0] for r in con.execute(
        "SELECT DISTINCT code FROM fund_holdings ORDER BY code")]
    done = {r[0] for r in con.execute("SELECT code FROM fund_profile")}
    todo = [c for c in codes if c not in done]
    todo = [c for i, c in enumerate(todo) if i % nshards == shard]
    if limit:
        todo = todo[:limit]
    print(f"[分片 {shard}/{nshards}] 待抓 {len(todo):,} 只（已完成 {len(done):,}）", flush=True)

    ok = fail = 0
    t0 = time.time()
    for i, code in enumerate(todo, 1):
        rec = None
        try:
            b = ak.fund_individual_basic_info_xq(symbol=code)
            if b is not None and len(b):
                kv = dict(zip(b["item"].astype(str), b["value"]))
                rec = {v: kv.get(k) for k, v in NFIELDS.items()}
                rec["scale_yi"] = parse_scale(rec.get("scale_yi"))
                # 成立时间统一成 YYYY-MM-DD，便于算基金年龄
                inc = str(rec.get("inception") or "")
                rec["inception"] = inc[:10] if len(inc) >= 10 else None
                if rec.get("code") is None:
                    rec["code"] = code
        except Exception:
            rec = None

        if rec is None:
            fail += 1
            con.execute("INSERT OR REPLACE INTO fund_profile (code, ts) VALUES (?,?)",
                        (code, dt.datetime.now().isoformat(timespec="seconds")))
        else:
            ok += 1
            con.execute(
                "INSERT OR REPLACE INTO fund_profile VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(rec.get("code") or code), rec.get("name"), rec.get("full_name"),
                 rec.get("inception"), rec.get("scale_yi"), rec.get("company"),
                 rec.get("managers"), rec.get("custodian"), rec.get("ftype"),
                 rec.get("benchmark"),
                 (str(rec.get("strategy") or "")[:4000] or None),
                 (str(rec.get("target") or "")[:1000] or None),
                 json.dumps({k: str(v) for k, v in rec.items() if v is not None},
                            ensure_ascii=False),
                 dt.datetime.now().isoformat(timespec="seconds")))
        if sleep:
            time.sleep(sleep)
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(todo)} 成功 {ok} 失败 {fail} {el:.0f}s "
                  f"预计剩余 {el / i * (len(todo) - i) / 60:.0f} 分钟", flush=True)
    con.close()
    print(f"完成：成功 {ok}、失败 {fail}，耗时 {(time.time() - t0) / 60:.1f} 分钟")


def report() -> None:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=30)
    try:
        n = con.execute("SELECT COUNT(*) FROM fund_profile").fetchone()[0]
        have = con.execute("SELECT COUNT(*) FROM fund_profile WHERE inception IS NOT NULL").fetchone()[0]
        print(f"档案 {n:,} 只，其中有成立日 {have:,} 只")
        print(pd.read_sql(
            "SELECT ftype, COUNT(*) n, ROUND(AVG(scale_yi),2) 平均规模亿, "
            "ROUND(SUM(scale_yi),0) 合计规模亿 FROM fund_profile "
            "WHERE ftype IS NOT NULL GROUP BY ftype ORDER BY n DESC LIMIT 10", con)
            .to_string(index=False))
        print("\n规模分布：")
        print(pd.read_sql(
            "SELECT CASE WHEN scale_yi<2 THEN '① 2亿以下' WHEN scale_yi<10 THEN '② 2-10亿' "
            "WHEN scale_yi<50 THEN '③ 10-50亿' WHEN scale_yi<100 THEN '④ 50-100亿' "
            "ELSE '⑤ 100亿以上' END 档, COUNT(*) 只数, ROUND(SUM(scale_yi),0) 合计亿 "
            "FROM fund_profile WHERE scale_yi IS NOT NULL GROUP BY 档 ORDER BY 档", con)
            .to_string(index=False))
    finally:
        con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--force-manager", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        con = connect()
        fetch_manager(con, a.force_manager)
        con.close()
        fetch_profiles(a.limit, a.shard, a.nshards, a.sleep)
        report()
