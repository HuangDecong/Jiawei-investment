"""校验 q_hold 与数据源「逐只股票」一致（联网）。

为什么要它
----------
`_ak_hold` 在 r45.1 引入了「按 stock_code 跨季度去重」，把「上季也持有」的股票
当重复删掉：002910 的 2026Q3 因此丢了药明康德 9.86%（真实第一大重仓），
页面把它读成「新进/清仓」；000001 的 2026Q3 整季被删空。
只看行数下限发现不了（Q3 自愿披露 15 行，丢 8 行后仍是 7，勉强能看出；
丢 2 行就完全看不出了），所以必须拿源表逐只对。

两种模式
--------
1) 默认：查最新一期（2026Q2/Q3）全部基金，约 10 分钟。
       python tools/verify_hold_rows.py

2) 历史抽样：按年分层抽样，验证 2015–2024 的持仓
       python tools/verify_hold_rows.py --years 2015-2024 --per-year 20
   每个年份随机抽 N 只「该年有持仓」的基金，把它这一年的四个季度逐只对一遍。
   为什么抽样而不是全量：全量 4,936 只 × 11 年 ≈ 5.4 万次请求（约 5 小时），
   而这类 bug 是系统性的（按股票代码去重），抽样即可发现；真发现问题再定向扩查。

比较口径
--------
- 源表：同一季度内先按股票代码去重（源表会把多个披露日的快照混在一张表里），
  再剔除「占净值比例」为空的行（入库时同样丢掉，因为没有可用信息）。
- 库内：该基金该季度的全部代码。
- 港股代码：旧版把 "00916" 补成 A 股 "000916"，这里允许「源 HK00916 ↔ 库 000916」
  的兼容比较，但会单独计为「旧港股代码」（口径错误，需重抓）。
"""
import argparse
import os
import random
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import akshare as ak  # noqa: E402
import pandas as pd  # noqa: E402
import qreport as Q  # noqa: E402

DB = os.path.join(ROOT, "data", "db", "fund_q.db")
QKEY = {"1季度": "Q1", "2季度": "Q2", "3季度": "Q3", "4季度": "Q4"}


def _db_map(con, years: list[int]) -> dict:
    """(code, quarter) → 代码集合。只载入相关年份，避免一次读全库。"""
    out = {}
    for y in years:
        for c, q, sc in con.execute(
                "SELECT code, quarter, stock_code FROM q_hold "
                "WHERE quarter LIKE ?", (f"{y}%",)):
            out.setdefault((c, q), set()).add(sc)
    return out


def _compare(want: set, have: set):
    """返回 (漏行, 多行, 旧港股代码)。"""
    have_norm, old_hk = set(), set()
    for h in have:
        if h in want:
            have_norm.add(h)
        elif h.isdigit() and ("HK" + str(int(h)).zfill(5)) in want:
            # 旧版把港股补成 6 位 A 股代码：不算漏行，但算口径错误
            have_norm.add("HK" + str(int(h)).zfill(5))
            old_hk.add(h)
        else:
            have_norm.add(h)
    return want - have_norm, have_norm - want, old_hk


def _check_fund(code: str, year: int, db_map: dict):
    try:
        raw = ak.fund_portfolio_hold_em(symbol=code, date=str(year))
    except Exception:  # noqa: BLE001
        # 抓取失败必须计入结果：静默跳过会让「0 异常」变成假象。
        return [(code, f"{year}", -1, -1, -1, "抓取失败", "")]
    if raw is None or raw.empty or "季度" not in raw.columns:
        return [(code, f"{year}", -1, -1, -1, "源表为空", "")]
    out = []
    for qn, suf in QKEY.items():
        r = raw[raw["季度"].astype(str).str.contains(qn)].copy()
        if r.empty:
            continue
        r["sc"] = r["股票代码"].map(Q._norm_code)
        ok = pd.to_numeric(r["占净值比例"], errors="coerce").notna()
        want = set(r.loc[ok, "sc"])
        have = db_map.get((code, f"{year}{suf}"), set())
        miss, extra, old_hk = _compare(want, have)
        if miss or extra or old_hk:
            out.append((code, f"{year}{suf}", len(miss), len(extra), len(old_hk),
                        "、".join(sorted(miss)[:3]), "、".join(sorted(extra)[:3])))
    return out


def _targets(con, args) -> list[tuple[str, int]]:
    if not args.years:
        codes = [r[0] for r in con.execute(
            "SELECT DISTINCT code FROM q_hold WHERE quarter='2026Q3'")]
        return [(c, 2026) for c in sorted(codes)]
    y0, _, y1 = args.years.partition("-")
    years = list(range(int(y0), int(y1 or y0) + 1))
    random.seed(args.seed)
    tg = []
    for y in years:
        cs = [r[0] for r in con.execute(
            "SELECT DISTINCT code FROM q_hold WHERE quarter LIKE ?", (f"{y}%",))]
        cs.sort()
        n = args.per_year or len(cs)
        tg += [(c, y) for c in random.sample(cs, min(n, len(cs)))]
    return tg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="", help="如 2015-2024；留空＝只查 2026Q2/Q3")
    ap.add_argument("--per-year", type=int, default=20, help="每年抽查基金数")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args()

    con = sqlite3.connect(DB)
    tg = _targets(con, args)
    years = sorted({y for _, y in tg})
    db_map = _db_map(con, years)
    scope = (f"{years[0]}–{years[-1]} 抽样 {len(tg)} 个基金×年" if args.years
             else f"2026Q2/Q3 全量 {len(tg)} 只")
    print(f"校验范围：{scope}")

    bad, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_check_fund, c, y, db_map) for c, y in tg]
        for f in as_completed(futs):
            bad += f.result()
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tg)}  异常 {len(bad)}", flush=True)

    df = pd.DataFrame(bad, columns=["code", "quarter", "漏行", "多行",
                                    "旧港股代码", "漏掉的代码", "多出的代码"])
    n_fail = int((df["漏行"] == -1).sum()) if len(df) else 0
    print(f"\n异常 {len(df)} / {len(tg)} 个基金×年（其中抓取失败 {n_fail}）")
    if len(df):
        print(f"其中：真漏行 {int((df['漏行'] > 0).sum())} 个、"
              f"真多行 {int((df['多行'] > 0).sum())} 个、"
              f"旧港股代码 {int((df['旧港股代码'] > 0).sum())} 个")
        print(df.head(30).to_string(index=False))
        fn = "hold_mismatch.csv" if not args.years else "hold_mismatch_hist.csv"
        df.to_csv(os.path.join(ROOT, "output", fn), index=False)
    ok = df.empty or (int((df["漏行"] > 0).sum()) == 0
                      and int((df["多行"] > 0).sum()) == 0
                      and int((df["旧港股代码"] > 0).sum()) == 0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
