# -*- coding: utf-8 -*-
"""全市场个股层持仓变动（东方财富 / AKShare 口径）。

与 Wind 版 `institution.stock_flow` 的区别
-----------------------------------------
Wind 版只能拿到「每期前 50 只」的持股基金数，且行业映射缺失；
这里用的是东财完整持仓明细（每只基金每期最多 100 只股票），
可以直接算全市场每只股票的持股基金数、持仓市值、持股数变化。

最重要的口径约束
----------------
基金季报只披露前十大重仓股，半年报/年报披露全部持仓。所以：

· 2026Q3（季报，前十大）与 2026Q2（半年报，全持仓）直接比较是错的——
  大量股票会从「前十大」掉出而在数据里表现为「清仓」；
· 正确做法是「同类比同类」：
    - 全持仓轨：Q2 vs 上年 Q2（都是半年报/年报口径）
    - 前十大轨：Q1 vs 上年 Q1、Q3 vs 上年 Q3（都是季报口径）

`comparable_pair()` 负责给出可比的一对季度，页面与报告一律走这个函数，
不允许手工指定跨口径的两期。
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "db", "fund_ak.db")


def _con():
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=30)


def conn_ok() -> bool:
    con = _con()
    if con is None:
        return False
    try:
        con.execute("SELECT 1 FROM fund_holdings LIMIT 1").fetchone()
        return True
    except Exception:
        return False
    finally:
        con.close()


def quarters() -> list[str]:
    con = _con()
    qs = []
    if con is not None:
        try:
            qs = [r[0] for r in con.execute(
                "SELECT DISTINCT quarter FROM fund_holdings ORDER BY quarter")]
        except Exception:
            qs = []
        con.close()
    if not qs:
        # 无数据库：用个股流向快照里的两期，保证 comparable_pair 仍能给出同口径对照
        f = _flow_fallback()
        if not f.empty and {"q0", "q1"} <= set(f.columns):
            qs = sorted({str(f["q0"].iloc[0]), str(f["q1"].iloc[0])})
    return qs


def _fmt(q: str) -> str:
    """'2026Q3' → ('2026', 3)。"""
    return q[:4], int(q[-1])


def disclosure_type(q: str) -> str:
    """Q2 = 半年报全持仓；Q1/Q3 = 季报前十大。Q4 若抓到则为年报全持仓。"""
    return "full" if q[-1] in ("2", "4") else "top10"


def comparable_pair(q: str, qs: list[str] | None = None) -> tuple[str, str] | None:
    """返回与 q 同披露口径、上一年同季度的可比对照期。

    Q2 优先对上年 Q2（全持仓 vs 全持仓）；若上年缺失，退回本年 Q4（若同口径）。
    """
    qs = qs or quarters()
    if q not in qs:
        return None
    y, k = _fmt(q)
    cand = [x for x in qs if x[-1] == q[-1] and x != q]
    # 优先上一年同季
    prev = f"{int(y) - 1}Q{k}"
    if prev in cand:
        return q, prev
    # 否则取最近的一个同年同季（向前找）
    older = sorted([x for x in cand if x < q])
    return (q, older[-1]) if older else None


FLOW_FALLBACK = os.path.join(ROOT, "data", "derived", "stock_flow_latest.csv")
_FLOW = None


def _flow_fallback() -> pd.DataFrame:
    """无数据库时的个股层降级数据（最近一组同披露口径两期，约 3000–4000 行）。"""
    global _FLOW
    if _FLOW is None:
        if os.path.exists(FLOW_FALLBACK):
            _FLOW = pd.read_csv(FLOW_FALLBACK, dtype={"stock_code": str})
        else:
            _FLOW = pd.DataFrame()
    return _FLOW


def snapshot(q: str) -> pd.DataFrame:
    """某期个股持仓快照：每只股票被多少基金持有、合计股数、合计市值。

    · mv 单位为万元（东财原始口径），对外展示时除以 1e4 得到亿元；
    · n_funds 是「样本内持有该股的基金数」，样本基金数逐期变化，跨期比较需归一化，
      见 `stock_flow()` 的 `share` 列。
    """
    con = _con()
    if con is None:
        return pd.DataFrame()
    # 只按代码分组：若按「代码+名称」分组，同一代码对应两个名称时，
    # merge(on="stock_code") 会产生笛卡尔重复，Δ持股基金数会翻倍错乱。
    sql = """
        SELECT stock_code, MAX(stock_name) AS stock_name,
               COUNT(DISTINCT code) AS n_funds,
               SUM(shares) AS shares,
               SUM(mv) AS mv,
               MAX(industry) AS industry,
               MAX(is_hk) AS is_hk
        FROM fund_holdings WHERE quarter = ?
        GROUP BY stock_code
    """
    d = pd.read_sql(sql, con, params=(q,))
    con.close()
    if d.empty:
        return d
    # 不能再 zfill(6)：港股是 5 位（00700），补成 6 位就会撞上 A 股 000700（模塑科技）
    d["stock_code"] = d["stock_code"].astype(str).str.strip()
    # 港股没有申万一级分类：单独标「港股」，不要混进「未映射」（后者是 A 股映射失败）
    d["industry"] = np.where(pd.to_numeric(d["is_hk"], errors="coerce").fillna(0) == 1,
                             "港股", d["industry"].fillna("未映射"))
    return d


def sample_funds(q: str) -> int:
    con = _con()
    if con is None:
        return 0
    try:
        n = con.execute("SELECT COUNT(DISTINCT code) FROM fund_holdings WHERE quarter=?",
                        (q,)).fetchone()[0]
    except Exception:
        n = 0
    con.close()
    return int(n)


def stock_flow(q: str, q0: str, min_funds: int = 1) -> pd.DataFrame:
    """两期个股持仓变动。调用方必须先保证 q / q0 披露口径相同。

    返回列：stock_code / stock_name / industry / n0 / n1 / d_n / share /
            mv0 / mv1 / d_mv / shares0 / shares1 / d_shares / state
    """
    a, b = snapshot(q), snapshot(q0)
    if a.empty or b.empty:
        # 部署环境没有 504MB 数据库：回退到随仓库分发的个股流向快照
        f = _flow_fallback()
        if not f.empty and {"q1", "q0"} <= set(f.columns) \
                and str(f["q1"].iloc[0]) == q and str(f["q0"].iloc[0]) == q0:
            cols = ["stock_code", "stock_name", "industry", "n0", "n1", "d_n",
                    "share", "d_share_pp", "mv0", "mv1", "mv1_yi", "d_mv_yi",
                    "shares0", "shares1", "d_shares", "state"]
            out = f[[c for c in cols if c in f.columns]].copy()
            out.attrs["q"], out.attrs["q0"] = q, q0
            out.attrs["n_funds"] = (int(f["n_funds0"].iloc[0]), int(f["n_funds1"].iloc[0])) \
                if {"n_funds0", "n_funds1"} <= set(f.columns) else (0, 0)
            out.attrs["disclosure"] = ("top10", "top10")
            return out
        return pd.DataFrame()
    n1f, n0f = sample_funds(q), sample_funds(q0)
    m = a.merge(b, on="stock_code", how="outer", suffixes=("", "_0"))
    m["stock_name"] = m["stock_name"].fillna(m["stock_name_0"])
    m["industry"] = m["industry"].fillna(m["industry_0"]).fillna("未映射")
    for c in ("n_funds", "shares", "mv"):
        m[c] = pd.to_numeric(m.get(c), errors="coerce")
        m[c + "_0"] = pd.to_numeric(m.get(c + "_0"), errors="coerce")
    m = m.rename(columns={"n_funds": "n1", "n_funds_0": "n0",
                          "shares": "shares1", "shares_0": "shares0",
                          "mv": "mv1", "mv_0": "mv0"})
    m[["n0", "n1", "shares0", "shares1", "mv0", "mv1"]] = \
        m[["n0", "n1", "shares0", "shares1", "mv0", "mv1"]].fillna(0.0)
    m["d_n"] = m["n1"] - m["n0"]
    m["d_mv"] = m["mv1"] - m["mv0"]
    m["d_shares"] = m["shares1"] - m["shares0"]
    m["share"] = (m["n1"] / n1f * 100).round(2) if n1f else np.nan
    m["share0"] = (m["n0"] / n0f * 100).round(2) if n0f else np.nan
    m["d_share_pp"] = (m["share"] - m["share0"]).round(2)

    def _state(r):
        if r["n0"] <= 0 and r["n1"] > 0:
            return "新进"
        if r["n1"] <= 0 and r["n0"] > 0:
            return "退出"
        return "继续持有"

    m["state"] = m.apply(_state, axis=1)
    m = m[m[["n0", "n1"]].max(axis=1) >= min_funds]
    m["mv1_yi"] = (m["mv1"] / 1e4).round(2)
    m["d_mv_yi"] = (m["d_mv"] / 1e4).round(2)
    cols = ["stock_code", "stock_name", "industry", "n0", "n1", "d_n",
            "share", "d_share_pp", "mv0", "mv1", "mv1_yi", "d_mv_yi",
            "shares0", "shares1", "d_shares", "state"]
    out = m[cols].sort_values("d_n", ascending=False).reset_index(drop=True)
    out.attrs["q"], out.attrs["q0"] = q, q0
    out.attrs["n_funds"] = (n0f, n1f)
    out.attrs["disclosure"] = (disclosure_type(q0), disclosure_type(q))
    return out


def stock_rank(q: str, q0: str, top: int = 20, min_n1: int = 10) -> dict:
    """建仓榜 / 撤离榜。min_n1 过滤掉只有 1–2 只基金持有、Δ 噪音极大的小票。"""
    f = stock_flow(q, q0)
    if f.empty:
        return {"add": pd.DataFrame(), "cut": pd.DataFrame(), "meta": {}}
    base = f[(f["n1"] >= min_n1) | (f["n0"] >= min_n1)]
    meta = dict(q=q, q0=q0, n_funds=f.attrs.get("n_funds"),
                disclosure=f.attrs.get("disclosure"), rows=len(f))
    return {
        "add": base.nlargest(top, "d_n").reset_index(drop=True),
        "cut": base.nsmallest(top, "d_n").reset_index(drop=True),
        "meta": meta,
    }


def industry_flow(q: str, q0: str) -> pd.DataFrame:
    """按申万一级汇总的个股东向：净加仓股票数、Δ市值合计、Δ持股基金数合计。"""
    f = stock_flow(q, q0)
    if f.empty:
        return pd.DataFrame()
    f = f[f["industry"] != "未映射"]
    g = f.groupby("industry").agg(
        股票数=("stock_code", "count"),
        净加仓股票数=("d_n", lambda s: int((s > 0).sum())),
        净减仓股票数=("d_n", lambda s: int((s < 0).sum())),
        Δ持股基金数合计=("d_n", "sum"),
        Δ持仓市值亿=("d_mv_yi", "sum"),
    ).round(2)
    g["加仓股票占比"] = (g["净加仓股票数"] /
                       (g["净加仓股票数"] + g["净减仓股票数"]).replace(0, np.nan) * 100).round(1)
    return g.sort_values("Δ持仓市值亿", ascending=False)


if __name__ == "__main__":
    print("可用季度:", quarters())
    p = comparable_pair("2026Q3")
    print("2026Q3 的可比对照:", p)
    if p:
        r = stock_rank(*p, top=5)
        print("样本基金数:", r["meta"].get("n_funds"), "披露口径:", r["meta"].get("disclosure"))
        print("\n建仓榜:\n", r["add"][["stock_name", "industry", "n0", "n1", "d_n", "d_mv_yi"]].to_string(index=False))
        print("\n撤离榜:\n", r["cut"][["stock_name", "industry", "n0", "n1", "d_n", "d_mv_yi"]].to_string(index=False))
