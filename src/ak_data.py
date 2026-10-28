# -*- coding: utf-8 -*-
"""东方财富（AKShare）口径的全市场基金持仓分析。

与 Wind 样本的关系
------------------
· 本模块：全市场主动权益基金（约 4000 只，去重后），来自东方财富公开披露，
  单基金一次返回 2026 年各季度全部股票投资明细；用于分歧度、个股流动的全市场口径。
· institution 模块：50 只样本（Wind MCP 单基金查询），用于**交叉验证**。
两者同口径（季报/半年报披露的占净值比），但覆盖率不同，跨期必须先归一化。

口径
----
· 季报（Q1/Q3）只披露前十大，覆盖率约 44–45%；
  半年报（Q2）披露全部（东财截断 100 只），覆盖率约 80%。
· 行业占比 = Σ(该行业股票占净值比) ÷ 该基金该期 coverage × 100，
  即「占已披露组合的比例」。
· 港股（5 位代码）不在申万一级内，行业汇总时剔除，但仍计入 coverage。
"""
from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pandas as pd

import config as C

DB_PATH = os.path.join(C.DATA, "db", "fund_ak.db")
MAP_PATH = os.path.join(C.DATA_RAW, "sw_industry_map.json")


def _con():
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def available() -> bool:
    return os.path.exists(DB_PATH)


PANEL_PATH = os.path.join(C.DATA, "derived", "fund_industry_panel.csv.gz")
_PANEL = None


def _panel() -> pd.DataFrame:
    """聚合面板（无数据库时的降级数据源，3.5MB，随仓库分发）。"""
    global _PANEL
    if _PANEL is None:
        if not os.path.exists(PANEL_PATH):
            _PANEL = pd.DataFrame()
        else:
            _PANEL = pd.read_csv(PANEL_PATH, compression="gzip")
    return _PANEL


def panel_available() -> bool:
    return os.path.exists(PANEL_PATH)


def coverage_table() -> pd.DataFrame:
    if not available():
        # 降级：从聚合面板重建逐期覆盖
        p = _panel()
        if p.empty:
            return pd.DataFrame()
        g = (p.groupby("quarter")
               .agg(n_funds=("code", "nunique"),
                    avg_cov=("coverage", "mean"),
                    n_rows=("weight_pct", "size"))
               .reset_index())
        return g
    con = _con()
    try:
        return pd.read_sql(
            "SELECT quarter, COUNT(DISTINCT code) n_funds, AVG(coverage) avg_cov,"
            " SUM(n_stock) n_rows FROM coverage GROUP BY quarter ORDER BY quarter", con)
    finally:
        con.close()


def latest_quarters() -> tuple[str, str] | None:
    if not available():
        # 无数据库时用聚合面板的期次（部署环境就是这样，否则会退回 Wind 50 只样本）
        qs = latest_quarters_all()
        return (qs[-1], qs[-2]) if len(qs) >= 2 else None
    con = _con()
    try:
        qs = [r[0] for r in con.execute(
            "SELECT DISTINCT quarter FROM coverage ORDER BY quarter")]
    finally:
        con.close()
    if len(qs) < 2:
        return None
    return qs[-1], qs[-2]


def latest_quarters_all() -> list[str]:
    """库内全部期次（升序）。时序验证需要全部期，而不只是最新两期。"""
    if not available():
        p = _panel()
        return sorted(p["quarter"].unique().tolist()) if not p.empty else []
    con = _con()
    try:
        return [r[0] for r in con.execute(
            "SELECT DISTINCT quarter FROM fund_holdings ORDER BY quarter")]
    finally:
        con.close()


def industry_panel(q: str, normalize: bool = True) -> pd.DataFrame:
    """基金 × 行业 的占比矩阵。index=基金代码，columns=31 行业。

    优先读 SQLite；无数据库时回退到 `data/derived/fund_industry_panel.csv.gz`，
    两者口径相同（都是持仓明细按申万一级聚合后的占净值比 + 覆盖率）。
    """
    if not available():
        p = _panel()
        if p.empty:
            return pd.DataFrame()
        d = p[p["quarter"] == q]
        if d.empty:
            return pd.DataFrame()
        d = d.copy()
        d["coverage"] = d["coverage"].replace(0, np.nan)
        d["share"] = (d["weight_pct"] / d["coverage"] * 100) if normalize else d["weight_pct"]
        return d.pivot_table(index="code", columns="industry", values="share",
                             aggfunc="sum").reindex(columns=C.INDUSTRIES).fillna(0.0)

    con = _con()
    try:
        df = pd.read_sql(
            "SELECT code, industry, weight_pct FROM fund_holdings "
            "WHERE quarter=? AND industry IS NOT NULL", con, params=(q,))
        cov = pd.read_sql(
            "SELECT code, coverage FROM coverage WHERE quarter=?", con, params=(q,))
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(["code", "industry"], as_index=False)["weight_pct"].sum()
    g = g.merge(cov, on="code", how="left")
    # 覆盖率异常剔除：东财个别基金（分级/联接/QDII）的「占净值比」合计会 >105%，
    # 归一化后会把真实行业占比压小。这类（基金×期）不参与分歧度计算，也不填补。
    bad = (g["coverage"] < 1) | (g["coverage"] > 105)
    if bad.any():
        g = g[~bad].copy()
    g["coverage"] = g["coverage"].replace(0, np.nan)
    g["share"] = (g["weight_pct"] / g["coverage"] * 100) if normalize \
        else g["weight_pct"]
    p = g.pivot_table(index="code", columns="industry", values="share", aggfunc="sum")
    return p.reindex(columns=C.INDUSTRIES).fillna(0.0)


def coverage_series(q: str) -> pd.Series:
    if not available():
        return pd.Series(dtype=float)
    con = _con()
    try:
        d = pd.read_sql("SELECT code, coverage FROM coverage WHERE quarter=?",
                        con, params=(q,))
    finally:
        con.close()
    return d.set_index("code")["coverage"] if not d.empty else pd.Series(dtype=float)


def dispersion_ak(q: str, q0: str, min_delta: float = 0.10,
                  theme_cut: float = 80.0, normalize: bool = True) -> pd.DataFrame:
    """全市场分歧度。

    现在**委托给 `canon.breadth`**，不再自己算一遍：
    同一个「加仓占比」以前有两套实现（这里按覆盖率归一化、canon 按前十大口径），
    两处的数字必然对不上，页面上就会出现「同一行业两个加仓占比」。
    对外列名保持不变，调用方无需改动。

    同时修正一个逻辑错误：旧判读要求「加仓占比 ≥70% 且 分歧度 <0.4」，
    但分歧度 = |2×占比−1|，两个条件数学上互斥，所以「共识形成」从来没出现过。
    """
    try:
        import canon as _cn
        _b = _cn.breadth(q0, q, min_move=min_delta)
        if not _b.empty:
            _p = _cn.panel_top10(q, normalize=False)
            _base = (_p.mean() if not _p.empty else pd.Series(dtype=float))
            out = _b.copy()
            out["加减仓比"] = np.nan          # 需要时用 Δ 分布单独算，不在这里凑数
            out["横截面标准差"] = out["离散度pp"]
            out["平均配置"] = out["行业"].map(_base).round(2)
            out["剔除主题基金"] = len(_cn.theme_codes(q))
            out = out[["行业", "样本基金数", "加仓基金数", "减仓基金数", "不变基金数",
                       "加仓占比", "分歧度", "加减仓比", "横截面标准差", "判读",
                       "平均配置", "剔除主题基金"]]
            out.attrs.update(q=q, q0=q0, n_funds=int(_b.attrs.get("n_funds", 0)),
                             normalized=True, basis="前十大（每基金取前 10 只后归一化）",
                             source="东方财富 AKShare（全市场主动权益）")
            return out
    except Exception:
        pass
    a = industry_panel(q, normalize=normalize)
    b = industry_panel(q0, normalize=normalize)
    if a.empty or b.empty:
        return pd.DataFrame()
    codes = a.index.intersection(b.index)
    a, b = a.loc[codes], b.loc[codes]
    rows = []
    for ind in C.INDUSTRIES:
        va = a[ind].to_numpy(float) if ind in a else np.zeros(len(codes))
        vb = b[ind].to_numpy(float) if ind in b else np.zeros(len(codes))
        ok = ~((va == 0) & (vb == 0))
        va, vb = va[ok], vb[ok]
        if len(va) < 5:
            continue
        theme = np.maximum(va, vb) > theme_cut
        va2, vb2 = va[~theme], vb[~theme]
        d = va2 - vb2
        up = int((d > min_delta).sum())
        dn = int((d < -min_delta).sum())
        fl = int(len(d) - up - dn)
        n = len(d)
        if n == 0:
            continue
        share = up / n
        div = 1 - abs(share - 0.5) * 2
        pos = float(d[d > min_delta].sum())
        neg = float(-d[d < -min_delta].sum())
        if share >= 0.70 and div < 0.40:
            tag = "共识形成"
        elif 0.40 <= share <= 0.60 and div > 0.70:
            tag = "多空对撞"
        elif share <= 0.30:
            tag = "系统性撤离"
        else:
            tag = "分歧"
        rows.append(dict(
            行业=ind, 样本基金数=n, 加仓基金数=up, 减仓基金数=dn, 不变基金数=fl,
            加仓占比=round(share * 100, 1), 分歧度=round(div, 2),
            加减仓比=(round(pos / neg, 2) if neg > 1e-9 else np.nan),
            横截面标准差=round(float(pd.Series(va2).std(ddof=1)), 2),
            判读=tag, 平均配置=round(float(pd.Series(va2).mean()), 2),
            剔除主题基金=int(theme.sum()),
        ))
    out = pd.DataFrame(rows)
    if not out.empty:
        out.attrs.update(q=q, q0=q0, n_funds=len(codes), normalized=normalize,
                         source="东方财富 AKShare（全市场主动权益）")
    return out


def top10_bias(q: str = "2026Q2", min_share: float = 3.0) -> pd.DataFrame:
    """前十大口径 vs 全持仓口径的行业偏差（4.5 全持仓校准）。

    做法
    ----
    只对「披露全部持仓」的半年报/年报期（Q2/Q4）可用：拿同一只基金的完整明细，
    先按占净值比排序取前 10 只算一遍行业结构，再用全部明细算一遍，两者相除。

    · 分子分母都先各自归一化到「已披露组合的 100%」——否则分母大的全持仓
      天然占优，比值会被覆盖率而不是集中度决定；
    · 只在全持仓口径下行业占比 ≥ min_share 的基金上取中位数，
      避免小分母（0.2% 对 0.1%）产生 2 倍的伪偏差；
    · 比值 <0.7 表示「前十大系统性低估该行业」，页面必须给出双口径对照。
    """
    if not available():
        return pd.DataFrame()
    con = _con()
    try:
        d = pd.read_sql(
            "SELECT code, industry, weight_pct FROM fund_holdings "
            "WHERE quarter=? AND industry IS NOT NULL AND industry!='未映射'",
            con, params=(q,))
    finally:
        con.close()
    if d.empty:
        return pd.DataFrame()
    rows = []
    for code, g in d.groupby("code"):
        g = g.sort_values("weight_pct", ascending=False)
        full = g.groupby("industry")["weight_pct"].sum()
        top = g.head(10).groupby("industry")["weight_pct"].sum()
        fs, ts = full.sum(), top.sum()
        if fs <= 0 or ts <= 0:
            continue
        fn_ = (full / fs * 100)
        tn_ = (top / ts * 100).reindex(fn_.index).fillna(0.0)
        for ind in fn_.index:
            if fn_[ind] >= min_share:
                rows.append(dict(code=code, industry=ind,
                                 full=fn_[ind], top=tn_[ind],
                                 ratio=tn_[ind] / fn_[ind] if fn_[ind] else np.nan))
    if not rows:
        return pd.DataFrame()
    r = pd.DataFrame(rows)
    out = r.groupby("industry").agg(
        基金数=("code", "nunique"),
        全持仓占比中位=("full", "median"),
        前十大占比中位=("top", "median"),
        前十大除以全持仓=("ratio", "median"),
    ).round(3).reset_index()
    out["低估"] = out["前十大除以全持仓"].apply(
        lambda x: "是（<0.7，前十大系统性低估）" if x < 0.7 else "否")
    out = out.sort_values("前十大除以全持仓")
    out.attrs["q"] = q
    return out


def stock_flow_ak(q: str, q0: str, top: int = 30) -> pd.DataFrame:
    """全市场个股 Δ持有基金数（比 Wind 前 50 只更全）。"""
    if not available():
        return pd.DataFrame()
    con = _con()
    try:
        d = pd.read_sql(
            "SELECT code fund, quarter, stock_code, stock_name, weight_pct, industry"
            " FROM fund_holdings WHERE quarter IN (?,?)", con, params=(q, q0))
    finally:
        con.close()
    if d.empty:
        return pd.DataFrame()
    g = (d.groupby(["stock_code", "quarter"])
           .agg(n_funds=("fund", "nunique"), mv=("weight_pct", "sum"),
                name=("stock_name", "first"), ind=("industry", "first"))
           .reset_index())
    p = g.pivot_table(index="stock_code", columns="quarter",
                      values=["n_funds", "mv"], aggfunc="first")
    out = pd.DataFrame(index=p.index)
    out["name"] = g.groupby("stock_code")["name"].first()
    out["industry"] = g.groupby("stock_code")["ind"].first()
    out["n0"] = p[("n_funds", q0)] if ("n_funds", q0) in p else np.nan
    out["n1"] = p[("n_funds", q)] if ("n_funds", q) in p else np.nan
    out["d_n"] = out["n1"].fillna(0) - out["n0"].fillna(0)
    out = out.dropna(subset=["n0", "n1"], how="all")
    out = out.sort_values("d_n")
    return pd.concat([out.head(top), out.tail(top)]).reset_index()


def cross_validate_wind(q: str, wind_path: str | None = None) -> pd.DataFrame:
    """同一批基金、同一季度：东财口径 vs Wind 口径的行业占比对比。

    只比较两套数据都有的基金与行业，按覆盖率归一化后再比，
    输出每个行业的平均绝对偏差（pp）与相关系数。
    """
    wind_path = wind_path or os.path.join(C.DATA_RAW, "fund_industry_50x4.json")
    if not os.path.exists(wind_path) or not available():
        return pd.DataFrame()
    w = json.load(open(wind_path, encoding="utf-8"))
    wdata = w.get("data", {})
    wcov = w.get("coverage", {})
    codes = []
    for code, per in wdata.items():
        if q not in per or not per[q]:
            continue
        c6 = code.split(".")[0]
        codes.append((code, c6))
    if not codes:
        return pd.DataFrame()
    a = industry_panel(q)          # 东财，已归一化
    rows = []
    for ind in C.INDUSTRIES:
        xs, ys = [], []
        for code, c6 in codes:
            if c6 not in a.index:
                continue
            wv = wdata[code][q].get(ind)
            if wv is None:
                continue
            cov = wcov.get(code, {}).get(q) or 100.0
            xs.append(float(wv) / cov * 100)
            ys.append(float(a.loc[c6, ind]))
        if len(xs) >= 5:
            xs, ys = np.array(xs), np.array(ys)
            rows.append(dict(行业=ind, n=len(xs),
                             平均绝对偏差=round(float(np.abs(xs - ys).mean()), 2),
                             Wind均值=round(float(xs.mean()), 2),
                             东财均值=round(float(ys.mean()), 2),
                             相关=round(float(pd.Series(xs).corr(pd.Series(ys))), 3)))
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["行业", "n", "平均绝对偏差", "Wind均值", "东财均值", "相关"])
    return out.sort_values("平均绝对偏差", ascending=False)
