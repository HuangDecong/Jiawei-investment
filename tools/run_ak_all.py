# -*- coding: utf-8 -*-
"""按年份顺序、多进程分片跑完东财基金持仓抓取（断点续传，可随时中断重跑）。

用法：
    python tools/run_ak_all.py                     # 2026→2023，默认 5 分片
    python tools/run_ak_all.py 2026 2025           # 只跑指定年份
    python tools/run_ak_all.py 2026 --shards 3     # 指定并发数

为什么要多进程：单进程约 25–30 只/分钟，全市场 5000+ 只要 3 小时/年；
东财接口的瓶颈在网络往返，多进程各抓一片能把总时间压到 1/4 左右。
分片按「代码排序后取模」，所以中断重跑不会重复抓同一只。

并发写的安全性：DB 已开 WAL + busy_timeout=60s，每个进程每 50 只 commit 一次。
"""
import datetime
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "tools", "ak_fund_pipeline.py")
LOG_DIR = os.path.join(ROOT, "output")
DB = os.path.join(ROOT, "data", "db", "fund_ak.db")
os.makedirs(LOG_DIR, exist_ok=True)
RUN_LOG = os.path.join(LOG_DIR, "ak_fetch_run.log")


def log(msg: str):
    line = f"[{datetime.datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def db_counts():
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=10)
        n = con.execute("SELECT COUNT(DISTINCT code) FROM fund_holdings").fetchone()[0]
        qs = con.execute("SELECT COUNT(DISTINCT quarter) FROM fund_holdings").fetchone()[0]
        con.close()
        return n, qs
    except Exception:
        return -1, -1


def year_done(year: str) -> int:
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=10)
        n = con.execute("SELECT COUNT(DISTINCT code) FROM fund_holdings "
                        "WHERE quarter LIKE ?", (f"{year}%",)).fetchone()[0]
        con.close()
        return n
    except Exception:
        return -1


def run_year(year: str, nshards: int, rounds: int = 3) -> None:
    """跑一个年份。分片可能因网络/锁异常中途退出，所以补跑最多 3 轮，
    直到该年新增基金数 < 5 为止。"""
    for rd in range(rounds):
        before = year_done(year)
        _run_round(year, nshards, rd)
        after = year_done(year)
        log(f"{year} 第 {rd + 1} 轮：{before} → {after}（新增 {after - before}）")
        if after - before < 5:
            break


def _run_round(year: str, nshards: int, rnd: int) -> None:
    log(f"===== {year} 第 {rnd + 1} 轮开始（{nshards} 分片）=====")
    t0 = time.time()
    handles, procs = [], []
    for i in range(nshards):
        fh = open(os.path.join(LOG_DIR, f"ak_shard_{year}_{rnd}_{i}.log"), "a",
                  encoding="utf-8")
        handles.append(fh)
        procs.append(subprocess.Popen(
            [sys.executable, PIPE, "--fetch", "--year", str(year),
             "--shard", str(i), "--nshards", str(nshards)],
            stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT))
    while any(p.poll() is None for p in procs):
        time.sleep(60)
        n, nq = db_counts()
        alive = sum(1 for p in procs if p.poll() is None)
        log(f"{year} 进行中：存活分片 {alive}/{nshards}，库内 {n} 只基金 / {nq} 期，"
            f"已耗时 {(time.time() - t0) / 60:.0f} 分钟")
    for fh in handles:
        fh.close()
    codes = [p.returncode for p in procs]
    log(f"===== {year} 第 {rnd + 1} 轮结束，耗时 {(time.time() - t0) / 60:.1f} 分钟，"
        f"退出码 {codes}，库内 {db_counts()[0]} 只 =====")


def main(argv):
    nshards = 5
    if "--shards" in argv:
        i = argv.index("--shards")
        nshards = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    years = argv or ["2026", "2025", "2024", "2023"]
    log(f"开始：年份 {years}，{nshards} 分片")
    for y in years:
        try:
            run_year(y, nshards)
        except Exception as e:
            log(f"{y} 异常：{type(e).__name__}: {e}")
    log("全部年份完成")


if __name__ == "__main__":
    main(sys.argv[1:])
