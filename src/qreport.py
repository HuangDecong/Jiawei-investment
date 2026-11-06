# -*- coding: utf-8 -*-
"""季报数据层：只存「季报原文里披露的数字」，不做任何推算与替代。

为什么单独建库
--------------
旧库 `fund_ak.db` 只有持仓明细（前十/全持仓），要回答「这季季报说了什么」
还缺四块：资产配置（股/债/现金）、按行业分类的投资组合、份额变动、重大买卖。
这些数字在季报里都有原文，只是分散在不同表里。

本模块只做三件事：抓取、落库、读取。任何一处取不到就留空并记进 `q_log`，
绝不用「前十大反推」「按覆盖率归一」之类的替代口径冒充披露值。

数据来源（按取得顺序）
----------------------
1. 东方财富基金档案（转引基金季报原文）
   · 资产配置   fundf10.eastmoney.com/zcpz_{code}.html
     → 股票占净比 / 债券占净比 / 现金占净比 / 净资产
   · 份额变动   fundf10.eastmoney.com/FundArchivesDatas.aspx?type=gmbd
     → 期间申购 / 期间赎回 / 期末总份额 / 期末净资产
   · 财务指标   ...?type=cwzb（仅半年报与年报，季报无此表）
2. akshare（同样转引东财，字段与季报一一对应）
   · 行业配置   fund_portfolio_industry_allocation_em
     → 按《上市公司行业分类指引》门类的公允价值与占净值比（全持仓口径）
   · 持仓明细   fund_portfolio_hold_em（Q1/Q3 前十，Q2/Q4 全部）
   · 重大买卖   fund_portfolio_change_em（累计买入/卖出前 20 名）
   · 债券持仓   fund_portfolio_bond_hold_em

口径提醒
--------
· 行业配置是**证监会门类**（约 19 类），不是申万 31 级；它覆盖全部持仓，
  但粒度粗。申万 31 级只能从前十大里看到，两者不可互相替代。
· 港股通持仓的行业分类在季报里单列，本接口未提供 → 记为缺失，不并入 A 股行业。
· 财务指标（本期利润/已实现收益）只有半年报与年报有，季报期留空。
"""
from __future__ import annotations

import io
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

import config as C

DB_PATH = os.path.join(C.ROOT, "data", "db", "fund_q.db") \
    if hasattr(C, "ROOT") else os.path.join("data", "db", "fund_q.db")

_HDR = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"),
    "Referer": "https://fundf10.eastmoney.com/",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS q_asset(
  code TEXT, period TEXT, quarter TEXT,
  stock_pct REAL, bond_pct REAL, cash_pct REAL, nav_yi REAL,
  src TEXT, fetched_at TEXT, PRIMARY KEY(code, period));
CREATE TABLE IF NOT EXISTS q_industry(
  code TEXT, period TEXT, quarter TEXT, sec_name TEXT,
  ratio_pct REAL, mv_wan REAL, src TEXT, fetched_at TEXT,
  PRIMARY KEY(code, period, sec_name));
CREATE TABLE IF NOT EXISTS q_shares(
  code TEXT, period TEXT, quarter TEXT,
  purchase_yi REAL, redeem_yi REAL, end_share_yi REAL,
  end_nav_yi REAL, nav_chg_pct REAL, src TEXT, fetched_at TEXT,
  PRIMARY KEY(code, period));
CREATE TABLE IF NOT EXISTS q_hold(
  code TEXT, period TEXT, quarter TEXT, rnk INTEGER,
  stock_code TEXT, stock_name TEXT, ratio_pct REAL,
  shares_wan REAL, mv_wan REAL, src TEXT, fetched_at TEXT,
  PRIMARY KEY(code, period, stock_code));
CREATE TABLE IF NOT EXISTS q_change(
  code TEXT, period TEXT, quarter TEXT, direction TEXT,
  stock_code TEXT, stock_name TEXT, amount_wan REAL, pct_nav REAL,
  src TEXT, fetched_at TEXT,
  PRIMARY KEY(code, period, direction, stock_code));
CREATE TABLE IF NOT EXISTS q_bond(
  code TEXT, period TEXT, quarter TEXT, bond_code TEXT, bond_name TEXT,
  ratio_pct REAL, mv_wan REAL, src TEXT, fetched_at TEXT,
  PRIMARY KEY(code, period, bond_code));
CREATE TABLE IF NOT EXISTS q_fin(
  code TEXT, period TEXT, quarter TEXT, item TEXT, value REAL,
  src TEXT, fetched_at TEXT, PRIMARY KEY(code, period, item));
CREATE TABLE IF NOT EXISTS q_log(
  code TEXT PRIMARY KEY, ok INTEGER, msg TEXT, n_rows INTEGER, fetched_at TEXT);
CREATE TABLE IF NOT EXISTS q_ytry(
  code TEXT, year INTEGER, ts TEXT, n_rows INTEGER, PRIMARY KEY(code, year));
"""


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _con() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def ensure_db() -> None:
    with _con() as con:
        con.executescript(_SCHEMA)


def _pct(x) -> float:
    """'84.10%' → 84.10；'---'/'--'/'' → NaN。"""
    if x is None:
        return float("nan")
    s = str(x).strip().replace(",", "").replace("%", "")
    if s in ("", "---", "--", "-", "nan", "None"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _num(x) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip().replace(",", "")
    if s in ("", "---", "--", "-", "nan", "None"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def q_of(period: str) -> str:
    """'2026-09-30' / '2026年3季度股票投资明细' / '2026Q3' → '2026Q3'。"""
    s = str(period)
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return f"{y}Q{max(1, min(4, (mo + 2) // 3))}"
    m = re.match(r"(\d{4})年(\d)季度", s)
    if m:
        return f"{int(m.group(1))}Q{int(m.group(2))}"
    m = re.match(r"(\d{4})Q(\d)", s)
    if m:
        return f"{int(m.group(1))}Q{int(m.group(2))}"
    return s


_Q_RE = re.compile(r"^\d{4}Q[1-4]$")


def valid_q(q) -> bool:
    """季度串是否合法（YYYYQn）。

    为什么必须校验：数据源偶发返回「列错位」的表（截止时间列里放的是基金代码），
    落库后 quarter 会变成 '000126' 这种值。它不会撞上任何真实季度，看起来无害，
    但会让「本基金共 N 期」统计虚高，也会让按年补抓误以为「这年已经抓过」。
    """
    return bool(_Q_RE.match(str(q)))


def _valid_period(s) -> bool:
    s = str(s)
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}", s)
                or re.match(r"^\d{4}年\d季度", s)
                or _Q_RE.match(s))


def _norm_code(x) -> str:
    """股票代码归一：5 位数字＝港股，补 HK 前缀；6 位＝A 股原样。

    为什么必须区分：港股 02517（锅圈）补零成 6 位后正好等于 A 股 002517
    （恺英网络），两行会在同一期撞成一条主键，写库直接 IntegrityError，
    更糟的是行业映射会把港股算进 A 股行业——这正是旧库踩过的坑。
    """
    s = str(x).strip().split(".")[0]
    if re.fullmatch(r"\d{5}", s):
        return "HK" + s
    if re.fullmatch(r"\d{6}", s):
        return s
    if re.fullmatch(r"\d{1,4}", s):
        return "HK" + s.zfill(5)
    return s


# 证监会 19 个门类（A 股行业分类）。不在这个集合里的，都是港股通的恒生行业，
# 两套分类不能混在一张图里比较：制造业（证监会）与工业（恒生）不是同一层级，
# 名称还常带数字前缀（20工业、45信息技术），直接画会让人以为多了几个行业。
CSRC_IND = {"农、林、牧、渔业", "采矿业", "制造业",
            "电力、热力、燃气及水生产和供应业", "建筑业", "批发和零售业",
            "交通运输、仓储和邮政业", "住宿和餐饮业",
            "信息传输、软件和信息技术服务业", "金融业", "房地产业",
            "租赁和商务服务业", "科学研究和技术服务业",
            "水利、环境和公共设施管理业", "居民服务、修理和其他服务业",
            "教育", "卫生和社会工作", "文化、体育和娱乐业", "综合"}


def tag_industry(d: pd.DataFrame) -> pd.DataFrame:
    """给行业行标市场：A股（证监会门类）/ 港股通（恒生行业）。"""
    if d is None or d.empty:
        return pd.DataFrame(columns=["sec_name", "ratio_pct", "market"])
    out = d.copy()
    out["sec_name"] = out["sec_name"].astype(str).str.strip()
    out["market"] = np.where(out["sec_name"].isin(CSRC_IND), "A股", "港股通")
    return out


def industry_view(cur: pd.DataFrame, prev: pd.DataFrame | None = None,
                  top: int = 10) -> pd.DataFrame:
    """把行业表整理成「A股各门类 + 港股通合计」一张可直接画的表。

    为什么把港股通合并成一条：季报里港股通用的是恒生行业分类，与 A 股的
    证监会门类不同源，逐条并列会让读者把「工业」和「制造业」当成两个行业。
    合并成一条既保留了「港股占多少」这个信息，又不制造假的可比性。
    """
    c = tag_industry(cur)
    if c.empty:
        return pd.DataFrame()
    a = (c[(c["market"] == "A股") & (c["ratio_pct"] > 0.005)]
         .nlargest(top, "ratio_pct")[["sec_name", "ratio_pct"]])
    rows = a.to_dict("records")
    hk = float(c[c["market"] == "港股通"]["ratio_pct"].sum())
    if hk > 0.01:
        rows.append({"sec_name": "港股通合计", "ratio_pct": hk})
    out = pd.DataFrame(rows)
    if prev is not None and not prev.empty:
        p = tag_industry(prev)
        pm = p[p["market"] == "A股"].groupby("sec_name")["ratio_pct"].sum()
        out["prev_pct"] = out["sec_name"].map(pm)
        ph = float(p[p["market"] == "港股通"]["ratio_pct"].sum())
        out.loc[out["sec_name"] == "港股通合计", "prev_pct"] = ph if ph > 0 else np.nan
    return out


def _clean(d: pd.DataFrame) -> pd.DataFrame:
    return d.replace({"—": np.nan, "---": np.nan, "--": np.nan, "": np.nan})


# ---------------------------------------------------------------------------
# 抓取：东财 f10（资产配置 / 份额变动 / 财务指标）
# ---------------------------------------------------------------------------
def _http_get(url: str, **kw) -> requests.Response:
    """东财 f10 请求：失败重试 3 次。

    批量抓取时并发一高，FundArchivesDatas 偶尔返回 5xx/空响应；
    不重试就会把「没抓到」记成「没有数据」，页面上就是一片「暂无」。
    """
    last: Exception | None = None
    for i in range(3):
        try:
            r = requests.get(url, headers=_HDR, timeout=20, **kw)
            if r.status_code == 200 and len(r.text) > 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} len={len(r.text)}")
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(0.8 + 0.9 * i)
    raise last if last else RuntimeError("unknown")


def _f10_asset(code: str) -> pd.DataFrame:
    url = f"https://fundf10.eastmoney.com/zcpz_{code}.html"
    r = _http_get(url)
    for t in pd.read_html(io.StringIO(r.text)):
        cols = [str(c) for c in t.columns]
        if "报告期" in cols and any("股票占净比" in c for c in cols):
            d = t.copy()
            d.columns = [str(c).strip() for c in d.columns]
            out = pd.DataFrame({
                "period": d["报告期"].astype(str),
                "stock_pct": d["股票占净比"].map(_pct),
                "bond_pct": d["债券占净比"].map(_pct),
                "cash_pct": d["现金占净比"].map(_pct),
                "nav_yi": d.get("净资产（亿元）", pd.Series(np.nan, index=d.index)).map(_num),
            })
            return out
    return pd.DataFrame()


_GMBD_MEM: dict[str, tuple[pd.DataFrame, list[str]]] = {}
_GMBD_LOCK = threading.Lock()
_GMBD_LAST = [0.0]
_GMBD_GAP = 0.36          # 全局限速：每秒不超过 ~2.8 次，避开东财的频率封禁


def _throttle() -> None:
    """进程级全局限速。

    并发 5 个线程时，实际请求速率约 25 次/秒，东财会返回 514（Frequency Capped），
    每次失败要重试 3 次、累计 sleep 5 秒——反而比限速慢 10 倍。
    限到 2.8 次/秒后，几乎不再触发封禁。
    """
    with _GMBD_LOCK:
        dt = time.time() - _GMBD_LAST[0]
        if dt < _GMBD_GAP:
            time.sleep(_GMBD_GAP - dt)
        _GMBD_LAST[0] = time.time()


def _gmbd_one(code: str, use_cache: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """抓单个份额类别的「份额变动」表，返回 (表, 同一基金的份额类别代码)。

    进程内缓存：修复脚本会成对处理 A/C 两类，两类共享同一 MERGECODE，
    缓存后每个基金只需 2 次请求而不是 4 次（东财有频率限制，省一半时间）。
    """
    if use_cache and code in _GMBD_MEM:
        d, codes = _GMBD_MEM[code]
        return d.copy(), codes
    _throttle()
    r = _http_get("https://fundf10.eastmoney.com/FundArchivesDatas.aspx",
                  params={"type": "gmbd", "code": code, "r": time.time()})
    m = re.search(r'content:"(.*?)"\s*,\s*arryear', r.text, re.S)
    if not m:
        m = re.search(r'content:"(.*?)",', r.text, re.S)
    if not m or len(m.group(1)) < 50:
        return pd.DataFrame(), []
    tabs = pd.read_html(io.StringIO(m.group(1)))
    if not tabs:
        return pd.DataFrame(), []
    d = _clean(tabs[0])
    d.columns = [str(c).strip() for c in d.columns]
    rename = {"日期": "period", "期间申购（亿份）": "purchase_yi",
              "期间赎回（亿份）": "redeem_yi", "期末总份额（亿份）": "end_share_yi",
              "期末净资产（亿元）": "end_nav_yi", "净资产变动率": "nav_chg_pct"}
    for k in rename:
        if k not in d.columns:
            d[k] = np.nan
    out = pd.DataFrame({v: d[k] for k, v in rename.items()})
    out["period"] = out["period"].astype(str)
    for c in ("purchase_yi", "redeem_yi", "end_share_yi", "end_nav_yi",
              "nav_chg_pct"):
        out[c] = out[c].map(_pct if c == "nav_chg_pct" else _num)
    # 表尾脚注（「说明：* 表示该日数据为 A/C 份额之和」）会被 read_html 当成一行数据，
    # 落库后 period 是一整句中文、数值全空，页面上就多出一个叫「说明」的季度。
    out = out[out["period"].map(_valid_period)]
    mc = re.search(r'"MERGECODE"\s*:\s*"([^"]+)"', r.text)
    codes = [c.strip() for c in mc.group(1).split(",")] if mc else []
    out = out.reset_index(drop=True)
    if len(_GMBD_MEM) > 4000:
        _GMBD_MEM.clear()
    _GMBD_MEM[code] = (out, codes)
    return out, codes


def _f10_shares(code: str) -> pd.DataFrame:
    """份额变动（**基金整体**，A/C 等份额类别相加）。

    为什么必须相加
    --------------
    东财的 gmbd 接口按「份额类别」返回：查 015233（C 类）只给 C 类的 0.03 亿份，
    而同一代码的资产配置页（zcpz）给的是**整个基金**的 67.2 亿净资产（A+C 合计）。
    直接把两者放进「规模变化拆解」，会算出 2240 元/份的份额净值——数字全错。
    接口返回里带 MERGECODE（如 "015233,180012"），据此把同一基金的各类份额
    逐类抓下来再求和，才是与资产配置同口径的「基金整体」。

    任何一类抓失败就整表返回空：宁可显示「暂无」，也不入库一份少了一类的
    「看起来完整」的错数（页面会在 20 小时后重试）。
    """
    d0, codes = _gmbd_one(code)
    if d0.empty:
        return d0
    parts = [d0]
    for c in codes:
        if not c or c == code:
            continue
        d1, _ = _gmbd_one(c)
        if d1.empty:
            raise RuntimeError(f"份额类别 {c} 抓取失败，放弃合并")
        parts.append(d1)
    if len(parts) == 1:
        return d0
    allp = pd.concat(parts, ignore_index=True)
    agg = (allp.groupby("period", as_index=False)
           [["purchase_yi", "redeem_yi", "end_share_yi", "end_nav_yi"]]
           .sum(min_count=1))
    # 变动率不能相加，按合并后的净资产重算；缺上期就留空。
    agg = agg.sort_values("period").reset_index(drop=True)
    agg["nav_chg_pct"] = (agg["end_nav_yi"] / agg["end_nav_yi"].shift(1) - 1) * 100
    return agg


def _f10_fin(code: str) -> pd.DataFrame:
    """半年报/年报的主要财务指标（季报没有这张表，故季度页会留空）。"""
    r = _http_get("https://fundf10.eastmoney.com/FundArchivesDatas.aspx",
                  params={"type": "cwzb", "code": code, "r": time.time()})
    m = re.search(r'content:"(.*?)"\s*,\s*arryear', r.text, re.S)
    if not m:
        m = re.search(r'content:"(.*?)",', r.text, re.S)
    if not m or len(m.group(1)) < 50:
        return pd.DataFrame()
    rows = []
    for t in pd.read_html(io.StringIO(m.group(1))):
        t = _clean(t)
        if t.shape[1] < 2:
            continue
        t.columns = [str(c).strip() for c in t.columns]
        key = t.columns[0]
        for c in t.columns[1:]:
            if not re.match(r"\d{4}-\d{2}-\d{2}", str(c)):
                continue
            for _, r0 in t.iterrows():
                rows.append({"period": str(c), "item": str(r0[key]),
                             "value": _num(r0[c])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 抓取：akshare（行业配置 / 持仓 / 重大买卖 / 债券）
# ---------------------------------------------------------------------------
def _ak_industry(code: str, year: int) -> pd.DataFrame:
    import akshare as ak
    try:
        d = ak.fund_portfolio_industry_allocation_em(symbol=code, date=str(year))
    except (KeyError, ValueError):
        # 该期没有行业表时，解析端只有 1 列却按 17 列命名，抛 Length mismatch；
        # 这是「没有」，不是「坏了」。
        return pd.DataFrame()
    if d is None or d.empty or "行业类别" not in d.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "period": d["截止时间"].astype(str),
        "sec_name": d["行业类别"].astype(str).str.strip(),
        "ratio_pct": d["占净值比例"].map(_num),
        "mv_wan": d["市值"].map(_num),
    })
    # 数据源偶发「列错位」：截止时间列返回基金代码，行业类别列返回日期。
    # 这种表整张都是废的，一张都不能落库，否则 quarter 会变成 '000126'。
    out = out[out["period"].map(_valid_period)]
    out = out[~out["sec_name"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
    # 没有「占净值比例」的行无法使用：源表在个别期次只返回占位行
    # （如 025776 的 2025Q4 只有「凯普生物 / 市值 857 万 / 占比空」）。
    # 留着会显示「1 只持仓、合计 0.00%」；去掉后该期显示「无持仓明细」，更诚实。
    out = out[out["ratio_pct"].notna()]
    return out.reset_index(drop=True)


def _asset_navs(code: str, year: int) -> dict[str, float]:
    """该基金该年各报告日的净资产（亿元），用于识别持仓表的「快照」。"""
    try:
        with _con() as con:
            d = pd.read_sql(
                "SELECT period, nav_yi FROM q_asset WHERE code=? AND period LIKE ?",
                con, params=(code, f"{int(year)}-%"))
        return {str(r.period): float(r.nav_yi) for r in d.itertuples()
                if pd.notna(r.nav_yi)}
    except Exception:
        return {}


def _ak_hold(code: str, year: int, nav_map: dict[str, float] | None = None) -> pd.DataFrame:
    import akshare as ak
    try:
        d = ak.fund_portfolio_hold_em(symbol=code, date=str(year))
    except (KeyError, ValueError):
        return pd.DataFrame()
    if d is None or d.empty or "股票代码" not in d.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "period": d["季度"].astype(str),
        "stock_code": d["股票代码"].map(_norm_code),
        "stock_name": d["股票名称"].astype(str),
        "ratio_pct": d["占净值比例"].map(_num),
        "shares_wan": d["持股数"].map(_num),
        "mv_wan": d["持仓市值"].map(_num),
    })
    # 数据源会把同一季度内**多个披露日**的持仓混在一张表里（如 015346 的 2025Q2
    # 同时有 2025-05-07 与 2025-06-30 两套快照，同一只股票出现两行）。
    # 直接按股票代码去重，会得到「A 快照的这只 + B 快照的那只」的混合组合：
    # 015346 因此算出前十大合计 128.87%（股票仓位只有 85.14%）。
    #
    # 判别方法：同一快照内，市值÷占比＝基金净资产，是个常数。
    # 用资产配置表里该季度的候选净资产，把每只重复股票归到「季末那一套」。
    if out["stock_code"].duplicated().any():
        out["_q"] = out["period"].map(q_of)
        for _q, g in out.groupby("_q"):
            if len(g) <= g["stock_code"].nunique():
                continue
            # 隐含净值＝市值÷占比，同一快照内是常数；不同披露日不同。
            _imp = (g["mv_wan"] * 1e4 / (g["ratio_pct"] / 100) / 1e8).round(1)
            # 占比四舍五入成 0.00% 的尾部持仓，隐含净值是 inf/NaN——它们不是
            # 「另一套快照」，必须保留（002910 的 2022Q2 因此才 69 行而不是 23 行）。
            _fin = np.isfinite(_imp)
            _sum = g.loc[_fin].groupby(_imp[_fin])["ratio_pct"].sum()
            if len(_sum) < 2:
                continue
            # 取「合计占比最大」的那一套，而不是行数最多或日期最标准的那套：
            # 转型期/清算期的季末快照可能只有两三行，按日期取会留下一个碎片。
            _keep = float(_sum.idxmax())
            # 容差要随权重变小而放宽：占比只保留两位小数，0.02% 的持仓
            # 光舍入就能让「市值÷占比」偏 ±25%（实测 54.0 vs 63.8 亿），
            # 用固定 ±2% 会把这些真实的小持仓误判成「另一套快照」删掉。
            _tol = (max(0.2, abs(_keep) * 0.02)
                    + abs(_keep) * 0.005 / g["ratio_pct"].clip(lower=0.005))
            _mask = _fin & ((_imp - _keep).abs() <= _tol)
            if _mask.sum() >= 5:
                # 只删「有确定隐含净值、且不属于主快照」的行；
                # 算不出隐含净值的尾行保留，避免把真实持仓当快照冲突删掉。
                out = out.drop(index=g.index[_fin & ~_mask])
        out = out.drop(columns="_q")
    # 占比与市值都为空的「幽灵行」：源表在部分期次只剩一行占位（如 005275 的
    # 2025Q3 只有「锦波生物 / A19121 / 空值」）。它没有信息量，留着会让页面
    # 显示「1 只持仓、合计 0.00%」，看起来像我们的 bug。
    # 没有「占净值比例」的行无法使用：源表在个别期次只返回占位行
    # （如 025776 的 2025Q4 只有「凯普生物 / 市值 857 万 / 占比空」）。
    # 留着会显示「1 只持仓、合计 0.00%」；去掉后该期显示「无持仓明细」，更诚实。
    out = out[out["ratio_pct"].notna()]
    # 去重必须带 period：同一个「年」的返回表里含 Q1–Q4 四期，
    # 只按 stock_code 去重会把「上季已持有、本季继续持有」的股票当成重复删掉，
    # 于是 Q3 的前十里凡是 Q2 也有的全部消失（002910 丢了药明康德 9.86%，
    # 000001 的 Q3 整季被删空），页面就会把这些股票误读成「新进/清仓」。
    out = out.drop_duplicates(["period", "stock_code"], keep="first")
    # 不能用数据源给的「序号」：它在一年的多期之间是连续编号的
    # （Q1 1–17、Q2 18–96、Q3 97–119），直接拿来当名次会出现「第 18 名」
    # 排在第一位的怪象。按占净值比在本期内重排。
    _rk = out.groupby("period")["ratio_pct"].rank(ascending=False, method="first")
    out["rnk"] = _rk.fillna(len(out) + 1).astype(int)   # 权重缺失的排到最后
    return out


def _ak_change(code: str, year: int, direction: str) -> pd.DataFrame:
    import akshare as ak
    try:
        d = ak.fund_portfolio_change_em(symbol=code, date=str(year),
                                        indicator=direction)
    except (KeyError, ValueError):
        return pd.DataFrame()
    if d is None or d.empty or "股票代码" not in d.columns:
        return pd.DataFrame()
    # akshare 的「累计卖出」表沿用了「本期累计买入金额」这个列名，
    # 直接按方向拼列名会 KeyError（本库曾因此丢掉全部卖出数据）。
    amt = next((c for c in d.columns if "累计" in str(c) and "金额" in str(c)), None)
    out = pd.DataFrame({
        "period": d["季度"].astype(str),
        "direction": "买入" if "买入" in direction else "卖出",
        "stock_code": d["股票代码"].map(_norm_code),
        "stock_name": d["股票名称"].astype(str),
        "amount_wan": d[amt].map(_num) if amt else np.nan,
        "pct_nav": d["占期初基金资产净值比例"].map(_num),
    })
    return out


def _ak_bond(code: str, year: int) -> pd.DataFrame:
    import akshare as ak
    try:
        d = ak.fund_portfolio_bond_hold_em(symbol=code, date=str(year))
    except Exception:  # noqa: BLE001
        # 纯股基没有债券持仓时，akshare 会抛 KeyError 或它自己的 APIError
        # （内部按列名取值）。这是「没有」而不是「取失败」，返回空表即可，
        # 否则每只纯股基都会在页头挂一条「部分接口未取到」的假告警。
        return pd.DataFrame()
    if d is None or d.empty or "债券代码" not in d.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "period": d["季度"].astype(str),
        "bond_code": d["债券代码"].astype(str),
        "bond_name": d["债券名称"].astype(str),
        "ratio_pct": d["占净值比例"].map(_num),
        "mv_wan": d["持仓市值"].map(_num),
    })
    return out


# ---------------------------------------------------------------------------
# 落库与读取
# ---------------------------------------------------------------------------
def _write(con, table: str, code: str, d: pd.DataFrame, cols: list[str],
           ts: str, src: str, periods: list[str]) -> int:
    if d is None or d.empty:
        return 0
    d = d.copy()
    d["code"] = code
    d["quarter"] = d["period"].map(q_of)
    # 最后一道闸：任何解析异常（列错位、脚注行）都不许落库。
    d = d[d["quarter"].map(valid_q)]
    if d.empty:
        return 0
    d["fetched_at"] = ts
    d["src"] = src
    _PK = {"q_asset": ["code", "period"],
           "q_industry": ["code", "period", "sec_name"],
           "q_shares": ["code", "period"],
           "q_hold": ["code", "period", "stock_code"],
           "q_change": ["code", "period", "direction", "stock_code"],
           "q_bond": ["code", "period", "bond_code"]}[table]
    d = d.drop_duplicates(subset=_PK, keep="first")
    # 只覆盖本次抓到的期次，避免把更早的历史冲掉
    for p in sorted(set(d["period"])):
        con.execute(f'DELETE FROM {table} WHERE code=? AND period=?', (code, p))
    use = ["code", "period", "quarter"] + cols + ["src", "fetched_at"]
    d[use].to_sql(table, con, if_exists="append", index=False)
    return len(d)


def fetch_fund(code: str, years: tuple[int, ...] | None = None,
               force: bool = False) -> dict:
    """抓一只基金的季报数据并落库。返回 {'tables': {...}, 'errors': [...]}。

    默认抓最近两年（8 个季度）。全部接口失败时也返回空表 + errors，
    由页面显示「暂无数据」，绝不退回推算口径。
    """
    code = str(code).zfill(6)
    ensure_db()
    now = datetime.now()
    years = years or (now.year - 1, now.year)
    ts = now.strftime("%Y-%m-%d %H:%M")
    errors: list[str] = []
    got: dict[str, int] = {}

    def tap(fn, *a, **kw):
        """限速 + 统一异常处理：一个接口坏掉不影响其它接口。"""
        try:
            time.sleep(0.18)   # 两进程并行时的限速，压住东财 f10 的 5xx
            return fn(*a, **kw)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{getattr(fn, '__name__', fn)}({a}) {type(e).__name__}")
            return pd.DataFrame()

    if not force:
        last, was_ok = _last_fetch(code)
        # 只有「上次成功」才走缓存。失败过的（ok=0）下次访问必须重试，
        # 否则一次网络抖动会让这只基金在 20 小时内一直显示「暂无」。
        if last and was_ok and (now - last) < timedelta(hours=20):
            d = load_fund(code)
            d["errors"] = []
            d["cached"] = True
            return d

    asset = tap(_f10_asset, code)
    shares = tap(_f10_shares, code)
    ind = pd.concat([tap(_ak_industry, code, y) for y in years], ignore_index=True) \
        if True else pd.DataFrame()
    # 把已抓到的资产配置传给持仓解析，用来识别同一季度里的多套快照
    _nav_map = ({str(r.period): float(r.nav_yi) for r in asset.itertuples()
                 if pd.notna(r.nav_yi)} if asset is not None and not asset.empty else None)
    hold = pd.concat([tap(_ak_hold, code, y, _nav_map) for y in years],
                     ignore_index=True)
    chg = pd.concat([tap(_ak_change, code, y, "累计买入") for y in years]
                    + [tap(_ak_change, code, y, "累计卖出") for y in years],
                    ignore_index=True)
    bond = pd.concat([tap(_ak_bond, code, y) for y in years], ignore_index=True)
    fin = tap(_f10_fin, code)

    with _con() as con:
        got["q_asset"] = _write(con, "q_asset", code, asset,
                                ["stock_pct", "bond_pct", "cash_pct", "nav_yi"],
                                ts, "东财基金档案·资产配置", [])
        got["q_industry"] = _write(con, "q_industry", code, ind,
                                   ["sec_name", "ratio_pct", "mv_wan"],
                                   ts, "基金季报·按行业分类（证监会门类）", [])
        got["q_shares"] = _write(con, "q_shares", code, shares,
                                 ["purchase_yi", "redeem_yi", "end_share_yi",
                                  "end_nav_yi", "nav_chg_pct"],
                                 ts, "东财基金档案·份额变动", [])
        got["q_hold"] = _write(con, "q_hold", code, hold,
                               ["rnk", "stock_code", "stock_name", "ratio_pct",
                                "shares_wan", "mv_wan"],
                               ts, "基金季报·股票投资明细", [])
        got["q_change"] = _write(con, "q_change", code, chg,
                                 ["direction", "stock_code", "stock_name",
                                  "amount_wan", "pct_nav"],
                                 ts, "基金季报·累计买卖前20", [])
        got["q_bond"] = _write(con, "q_bond", code, bond,
                               ["bond_code", "bond_name", "ratio_pct", "mv_wan"],
                               ts, "基金季报·债券投资明细", [])
        if fin is not None and not fin.empty:
            fin2 = fin.copy()
            fin2["quarter"] = fin2["period"].map(q_of)
            for p in sorted(set(fin2["period"])):
                con.execute("DELETE FROM q_fin WHERE code=? AND period=?", (code, p))
            fin2["code"] = code
            fin2["src"] = "东财基金档案·主要财务指标（半年报/年报）"
            fin2["fetched_at"] = ts
            fin2[["code", "period", "quarter", "item", "value", "src",
                  "fetched_at"]].to_sql("q_fin", con, if_exists="append",
                                        index=False)
            got["q_fin"] = len(fin2)
        con.execute(
            "INSERT OR REPLACE INTO q_log(code, ok, msg, n_rows, fetched_at) "
            "VALUES(?,?,?,?,?)",
            (code, 0 if errors else 1, ";".join(errors[:4])[:300],
             int(sum(got.values())), ts))

    out = load_fund(code)
    out["errors"] = errors
    out["cached"] = False
    return out


def ensure_year(code: str, year: int, force: bool = False) -> dict:
    """按需补抓某一年的「行业 / 持仓 / 买卖 / 债券」，供页面回看早年季度时调用。

    为什么不一上来抓全历史
    ----------------------
    一只 15 年的基金要 5×15＝75 次请求、约 40 秒；而绝大多数时候只看最近两年。
    所以默认只抓两年，用户往回翻时再补那一年；补过的年份记在 q_ytry，
    同一进程内和跨进程都不会重复抓（失败的也记，避免每次点都卡在超时上）。

    只补「明细」四张表：资产配置与份额由 f10 一次返回全历史，不需要按年抓。
    """
    code = str(code).zfill(6)
    ensure_db()
    if not force:
        with _con() as con:
            r = con.execute("SELECT n_rows FROM q_ytry WHERE code=? AND year=?",
                            (code, int(year))).fetchone()
        if r is not None:
            return {"fetched": False, "rows": int(r[0] or 0)}
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    errors: list[str] = []
    got = 0

    def tap(fn, *a):
        try:
            time.sleep(0.15)
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{getattr(fn, '__name__', fn)}({a}) {type(e).__name__}")
            return pd.DataFrame()

    ind = tap(_ak_industry, code, year)
    hold = tap(_ak_hold, code, year)
    chg = pd.concat([tap(_ak_change, code, year, "累计买入"),
                     tap(_ak_change, code, year, "累计卖出")], ignore_index=True)
    bond = tap(_ak_bond, code, year)
    # 按年重抓是「覆盖」而不是「追加」：先清掉这一年的旧行，再写入本次结果。
    # 否则源表后来变成空（如 005275 的 2025Q3 只剩占位行），旧的不完整数据
    # 会一直留在库里，看起来像「抓过但没修」。
    # 只有该接口本次没报错时才清，避免一次网络失败把整年数据删光。
    with _con() as con:
        for _t, _fn in (("q_industry", "_ak_industry"), ("q_hold", "_ak_hold"),
                        ("q_change", "_ak_change"), ("q_bond", "_ak_bond")):
            if not any(_fn in e for e in errors):
                con.execute(f"DELETE FROM {_t} WHERE code=? AND substr(period,1,4)=?",
                            (code, str(int(year))))
        got += _write(con, "q_industry", code, ind,
                      ["sec_name", "ratio_pct", "mv_wan"], ts,
                      "基金季报·按行业分类（证监会门类）", [])
        got += _write(con, "q_hold", code, hold,
                      ["rnk", "stock_code", "stock_name", "ratio_pct",
                       "shares_wan", "mv_wan"], ts,
                      "基金季报·股票投资明细", [])
        got += _write(con, "q_change", code, chg,
                      ["direction", "stock_code", "stock_name",
                       "amount_wan", "pct_nav"], ts,
                      "基金季报·累计买卖前20", [])
        got += _write(con, "q_bond", code, bond,
                      ["bond_code", "bond_name", "ratio_pct", "mv_wan"], ts,
                      "基金季报·债券投资明细", [])
        con.execute("INSERT OR REPLACE INTO q_ytry(code, year, ts, n_rows) "
                    "VALUES(?,?,?,?)", (code, int(year), ts, int(got)))
        # 让「20 小时缓存」以这次补抓为起点重新计时，同时把 lru_cache 的旧快照挤掉
        con.execute("UPDATE q_log SET fetched_at=? WHERE code=?", (ts, code))
    _load_cached.cache_clear()
    return {"fetched": True, "rows": int(got), "errors": errors}


def all_quarters(code: str) -> list[str]:
    """该基金**全部**有披露的季度（升序）。

    以资产配置/份额为主：这两张表一次返回全历史（老基金能到 40 期以上）。
    行业与持仓是「按年补抓」的，缺哪年由页面调 ensure_year 补，
    所以这里的期数才是「这只基金真实可读的季报期数」。
    """
    d = load_fund(code)
    qs: set[str] = set()
    for k in ("q_asset", "q_shares", "q_industry", "q_hold", "q_bond", "q_fin"):
        v = d.get(k)
        if v is not None and not v.empty and "quarter" in v.columns:
            qs |= {str(x) for x in v["quarter"].dropna() if valid_q(x)}
    return sorted(qs)


def tried_years(code: str) -> set[int]:
    """已经补抓过的年份（包括「抓了但确实没有」）。页面用它决定还缺哪几年。"""
    ensure_db()
    with _con() as con:
        rows = con.execute("SELECT year FROM q_ytry WHERE code=?",
                           (str(code).zfill(6),)).fetchall()
    return {int(r[0]) for r in rows}


def coverage_quarters(code: str) -> dict[str, set[str]]:
    """每张表各自有哪些季度——页面用它标注「本期哪几块有数据」。"""
    d = load_fund(code)
    out: dict[str, set[str]] = {}
    for k in ("q_asset", "q_industry", "q_shares", "q_hold", "q_change",
              "q_bond", "q_fin"):
        v = d.get(k)
        out[k] = ({str(x) for x in v["quarter"].dropna() if valid_q(x)}
                  if v is not None and not v.empty else set())
    return out


def _last_fetch(code: str):
    """返回 (上次抓取时间, 是否成功)。"""
    try:
        with _con() as con:
            r = con.execute("SELECT fetched_at, ok FROM q_log WHERE code=?",
                            (code,)).fetchone()
        if not r:
            return None, False
        return datetime.strptime(r[0], "%Y-%m-%d %H:%M"), bool(r[1])
    except Exception:
        return None, False


@lru_cache(maxsize=32)
def _load_cached(code: str, stamp: str) -> dict:
    return _load(code)


def load_fund(code: str) -> dict:
    """读一只基金已落库的季报数据（不联网）。"""
    code = str(code).zfill(6)
    ensure_db()
    with _con() as con:
        st = con.execute("SELECT MAX(fetched_at) FROM q_log WHERE code=?",
                         (code,)).fetchone()
    stamp = st[0] if st and st[0] else "none"
    return dict(_load_cached(code, stamp))


_NUM_COLS = {
    "q_asset": ["stock_pct", "bond_pct", "cash_pct", "nav_yi"],
    "q_industry": ["ratio_pct", "mv_wan"],
    "q_shares": ["purchase_yi", "redeem_yi", "end_share_yi", "end_nav_yi",
                 "nav_chg_pct"],
    "q_hold": ["rnk", "ratio_pct", "shares_wan", "mv_wan"],
    "q_change": ["amount_wan", "pct_nav"],
    "q_bond": ["ratio_pct", "mv_wan"],
    "q_fin": ["value"],
}


def _load(code: str) -> dict:
    ensure_db()
    out: dict[str, pd.DataFrame] = {}
    with _con() as con:
        for t in ("q_asset", "q_industry", "q_shares", "q_hold", "q_change",
                  "q_bond", "q_fin"):
            try:
                d = pd.read_sql(f"SELECT * FROM {t} WHERE code=?",
                                con, params=(code,))
            except Exception:
                d = pd.DataFrame()
            # 必须强制转成 float：SQLite 里的 NULL 读回来是 None，
            # 整列就变成 object，后面任何一个 np.isfinite() 都会抛
            # 「ufunc isfinite not supported」——页面直接白屏。
            for c in _NUM_COLS.get(t, []):
                if c in d.columns:
                    d[c] = pd.to_numeric(d[c], errors="coerce")
            out[t] = d
    return out


def quarters(code: str) -> list[str]:
    d = load_fund(code)
    qs = set()
    for k in ("q_asset", "q_industry", "q_shares", "q_hold"):
        v = d.get(k)
        if v is not None and not v.empty:
            qs |= {str(x) for x in v["quarter"].dropna() if valid_q(x)}
    return sorted(qs)


def coverage() -> dict:
    """本地已缓存多少只基金、多少期（用于页面上的「数据完整度」提示）。"""
    ensure_db()
    with _con() as con:
        n_fund = con.execute("SELECT COUNT(DISTINCT code) FROM q_log").fetchone()[0]
        n_ok = con.execute("SELECT COUNT(*) FROM q_log WHERE ok=1").fetchone()[0]
        n_ind = con.execute("SELECT COUNT(*) FROM q_industry").fetchone()[0]
        n_hold = con.execute("SELECT COUNT(*) FROM q_hold").fetchone()[0]
        last = con.execute("SELECT MAX(fetched_at) FROM q_log").fetchone()[0]
    return {"funds": n_fund, "ok": n_ok, "industry_rows": n_ind,
            "hold_rows": n_hold, "last": last}


def shares_consistent(end_nav: float, nav: float) -> bool:
    """份额表的「期末净资产」与资产配置的「净资产」是否同一口径。

    两者都保留两位小数（亿元），所以小基金的舍入误差会很大：
    0.03 亿的基金写成 0.02 或 0.04，相对偏差 30%，但绝对偏差只有 0.01 亿。
    判定用「绝对容差 0.02 亿 + 相对 0.5%」，而不是单纯看比例。
    """
    try:
        a, b = float(end_nav), float(nav)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return False
    return abs(a - b) <= 0.02 + 0.005 * b


def _one_per_fund(d: pd.DataFrame) -> pd.DataFrame:
    """同一季度保留一条：优先标准季末日（03-31/06-30/09-30/12-31）。

    转型/清盘期会多出一条临时披露日（如 2019-05-23 现金 202.9%），
    不去重会让同一只基金在一个季度里被计两次，临时日的极端值还会带偏中位数。
    """
    if d is None or d.empty:
        return d
    d = d.copy()
    d["_std"] = d["period"].astype(str).str.endswith(
        ("03-31", "06-30", "09-30", "12-31"))
    d = d.sort_values(["_std", "period"], ascending=[False, True])
    d = d.drop_duplicates("code", keep="first")
    return d.drop(columns="_std")


def peer_rank(q: str, val: float, col: str = "stock_pct") -> tuple[float, int]:
    """本基金在同期样本里的分位（0–100）与样本数。"""
    ensure_db()
    with _con() as con:
        try:
            d = pd.read_sql(
                f"SELECT code, period, {col} AS v FROM q_asset WHERE quarter=?",
                con, params=(q,))
        except Exception:
            return float("nan"), 0
    d = _one_per_fund(d)
    v = pd.to_numeric(d["v"], errors="coerce").dropna()
    if len(v) == 0 or not np.isfinite(val):
        return float("nan"), int(len(v))
    return float((v <= val).mean() * 100), int(len(v))


def peer_stat(q: str, col: str = "stock_pct") -> tuple[float, int]:
    """同一季度、已入库基金的某列中位数与样本数（用于「和同行比」）。

    只用已抓到的基金：样本量随批量抓取增长，页面必须显示 N，
    否则会让人误以为这是全市场口径。
    """
    ensure_db()
    with _con() as con:
        try:
            d = pd.read_sql(
                f"SELECT code, period, {col} AS v FROM q_asset WHERE quarter=?",
                con, params=(q,))
        except Exception:
            return float("nan"), 0
    d = _one_per_fund(d)
    v = pd.to_numeric(d["v"], errors="coerce").dropna()
    return (float(v.median()) if len(v) else float("nan")), int(len(v))


def market_series(col: str = "stock_pct") -> pd.DataFrame:
    """已入库基金按季度的中位数与样本数（跨基金汇总，用于「同行在做什么」）。

    样本只含已抓取的基金，且按规模优先抓取，所以是「大基金口径」；
    页面必须同时显示 N，不能用它冒充全市场。
    """
    ensure_db()
    with _con() as con:
        try:
            d = pd.read_sql(f"SELECT code, quarter, period, {col} AS v FROM q_asset",
                            con)
        except Exception:
            return pd.DataFrame(columns=["quarter", "median", "count"])
    d["quarter"] = d["quarter"].astype(str)
    rows = []
    for q, g in d.groupby("quarter"):
        gg = _one_per_fund(g)
        v = pd.to_numeric(gg["v"], errors="coerce").dropna()
        if len(v):
            rows.append({"quarter": q, "median": float(v.median()),
                         "count": int(len(v))})
    if not rows:
        return pd.DataFrame(columns=["quarter", "median", "count"])
    return pd.DataFrame(rows).sort_values("quarter")


def industry_peer(q: str, prev: str, min_n: int = 20) -> pd.DataFrame:
    """全样本行业配置的**配对**环比：同一只基金两期都有，才计算差值。

    为什么必须配对：直接拿「本期中位数 − 上期中位数」会把样本结构变化
    混进来（有的基金只有其中一期），出来的数字不是「基金在加减仓」，
    而是「换了一批基金」。实测同一天同一批数据，未配对得 −8.3pp，
    配对后只有 −3.2pp，差 5pp 全是构成效应。
    """
    ensure_db()
    with _con() as con:
        d = pd.read_sql(
            "SELECT code, quarter, sec_name, ratio_pct FROM q_industry "
            "WHERE quarter IN (?,?)", con, params=(q, prev))
    if d.empty:
        return pd.DataFrame()
    d["sec_name"] = d["sec_name"].astype(str).str.strip()
    d = d[d["sec_name"].isin(CSRC_IND)]          # 只比 A 股门类，港股通是另一套分类
    d["ratio_pct"] = pd.to_numeric(d["ratio_pct"], errors="coerce")
    p = d.pivot_table(index=["code", "sec_name"], columns="quarter",
                      values="ratio_pct", aggfunc="sum")
    if q not in p.columns or prev not in p.columns:
        return pd.DataFrame()
    p = p.dropna(subset=[q, prev])
    if p.empty:
        return pd.DataFrame()
    delta = (p[q] - p[prev]).groupby("sec_name")
    out = pd.DataFrame({"本期": p[q].groupby("sec_name").median(),
                        "上期": p[prev].groupby("sec_name").median(),
                        "差值": delta.median(),
                        "样本": delta.count()})
    # 样本太少的门类中位数没有意义（有几类只有 1 只基金），
    # 直接过滤，避免「居民服务 −0.01pp」这种噪音混进加减仓榜。
    out = out[out["样本"] >= min_n]
    return out.dropna(subset=["差值"]).sort_values("差值")


def latest_quarter(code: str) -> str | None:
    qs = quarters(code)
    return qs[-1] if qs else None


# ---------------------------------------------------------------------------
# 披露值之间的对照：全部由季报原文做差，不做推算
# ---------------------------------------------------------------------------
def top10_diff(d: dict, q: str, prev: str | None) -> pd.DataFrame:
    """前十大逐季进出：新进前十 / 升入前十 / 退出前十 / 清仓 / 增持 / 减持。

    为什么要用「上期全部披露」而不只是上期前十
    ------------------------------------------
    Q2/Q4 披露全部持仓，Q1/Q3 只披露前十。如果只拿两期前十对比，
    一只上期排第 15、本期升到第 8 的股票会被标成「新进前十」，
    而它的股数变化（真实加减仓）明明是知道的——这会让「新进」里混进
    完全不同的两类事：真的新建仓，和从第 11–20 名升上来。
    所以上期若有全持仓，就用它的股数/占比做基准；状态也区分
    「新进前十」（上期未持有）与「升入前十」（上期已持有、未进前十）。

    反过来，本期若是全持仓（Q2/Q4），上期前十里的股票彻底消失，
    就标「清仓」而不是「退出前十」——这两件事的含义完全不同。
    """
    h = d.get("q_hold")
    if h is None or h.empty or not prev:
        return pd.DataFrame()
    c_all = h[h["quarter"] == q].drop_duplicates("stock_code")
    p_all = h[h["quarter"] == prev].drop_duplicates("stock_code")
    if c_all.empty and p_all.empty:
        return pd.DataFrame()
    c_top = c_all[c_all["rnk"] <= 10].set_index("stock_code")
    p_top = p_all[p_all["rnk"] <= 10].set_index("stock_code")
    p_any = p_all.set_index("stock_code")
    c_any = set(c_all["stock_code"])
    # 本期是否「全持仓」：只有半年报/年报（Q2/Q4）才披露全部持仓。
    # 不能只看行数——东财的季报页有时也返回十几二十行（自愿披露/附加明细），
    # 但法定披露范围仍是前十，缺席者不代表清仓。
    c_full = str(q).endswith(("Q2", "Q4")) and len(c_all) > len(c_top)

    rows = []
    for code, r in c_top.iterrows():
        pr = p_any.loc[code] if code in p_any.index else None
        cur_pct, cur_sh = float(r["ratio_pct"]), float(r["shares_wan"])
        prev_pct = float(pr["ratio_pct"]) if pr is not None else np.nan
        prev_sh = float(pr["shares_wan"]) if pr is not None else np.nan
        prev_rk = int(pr["rnk"]) if pr is not None else None
        d_sh = (cur_sh / prev_sh - 1) if (prev_sh and prev_sh > 0
                                          and np.isfinite(cur_sh)) else np.nan
        if pr is None:
            st = "新进前十"
        elif prev_rk is not None and prev_rk > 10:
            st = "升入前十"
        elif np.isfinite(d_sh) and d_sh > 0.005:
            st = "增持"
        elif np.isfinite(d_sh) and d_sh < -0.005:
            st = "减持"
        else:
            st = "不变"
        rows.append({"代码": code, "名称": str(r["stock_name"]), "状态": st,
                     "本期%": cur_pct, "上期%": prev_pct,
                     "本期股数": cur_sh, "上期股数": prev_sh,
                     "d_pct": cur_pct - (prev_pct if np.isfinite(prev_pct) else 0),
                     "d_sh": d_sh})

    for code, r in p_top.iterrows():
        if code in c_top.index:
            continue
        st = "清仓" if (c_full and code not in c_any) else "退出前十"
        prev_pct = float(r["ratio_pct"])
        rows.append({"代码": code, "名称": str(r["stock_name"]), "状态": st,
                     "本期%": np.nan, "上期%": prev_pct,
                     "本期股数": np.nan, "上期股数": float(r["shares_wan"]),
                     "d_pct": -prev_pct, "d_sh": np.nan})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {"新进前十": 0, "升入前十": 1, "增持": 2, "不变": 3, "减持": 4,
             "退出前十": 5, "清仓": 5}
    out["_o"] = out["状态"].map(order)
    out = (out.sort_values(["_o", "本期%"], ascending=[True, False])
           .drop(columns="_o").set_index("代码"))
    return out


def industry_trend(d: dict, n: int = 6) -> pd.DataFrame:
    """行业配置的季度序列（A 股门类，占净值比；取最新一期最大的 n 类 + 其他）。

    季报的行业表覆盖全部股票，所以这条序列的合计约等于股票仓位——
    它回答的是「这只基金的行业押注这三年怎么挪的」，不是前十大的影子。
    """
    ind = d.get("q_industry")
    if ind is None or ind.empty:
        return pd.DataFrame()
    t = tag_industry(ind)
    t = t[t["market"] == "A股"]
    if t.empty:
        return pd.DataFrame()
    t["ratio_pct"] = pd.to_numeric(t["ratio_pct"], errors="coerce")
    p = t.pivot_table(index="quarter", columns="sec_name", values="ratio_pct",
                      aggfunc="sum").sort_index()
    if p.empty:
        return pd.DataFrame()
    # 按「这只基金全部披露季度」重排索引：行业表是按年补抓的，中间会缺几年。
    # 不补成 NaN 的话，折线会把 2018Q4 和 2025Q1 直接连起来，看起来像
    # 「这六年一直保持同一个配置」——那是最容易骗人的一种图。
    a = d.get("q_asset")
    if a is not None and not a.empty:
        qs = sorted({str(x) for x in a["quarter"].dropna() if valid_q(x)})
        p = p.reindex(qs)
    last = p.dropna(how="all").iloc[-1].dropna()
    if last.empty:
        return pd.DataFrame()
    keep = list(last.nlargest(n).index)
    out = p.reindex(columns=keep)
    rest = p.drop(columns=keep, errors="ignore").sum(axis=1, min_count=1)
    out["其他"] = rest
    return out


def industry_hhi(ind: pd.DataFrame) -> float:
    """行业集中度 HHI：按「证监会门类 + 港股通合计」分组，Σ(占净值比)²。

    为什么不能按「A股 / 港股通」两个市场分组：那样算出来是
    (84%)² + (0%)² = 0.71，只是「股票仓位」的平方，跟行业集中毫无关系。
    正确做法是把每个 A 股门类各算一组，港股通因为用了恒生分类、
    与门类不同源，合并成一个「港股通」组，再平方求和。
    """
    t = tag_industry(ind)
    if t.empty:
        return float("nan")
    t["ratio_pct"] = pd.to_numeric(t["ratio_pct"], errors="coerce")
    a = t[t["market"] == "A股"].groupby("sec_name")["ratio_pct"].sum()
    hk = float(t[t["market"] == "港股通"]["ratio_pct"].sum())
    parts = [float(v) for v in a.values]
    if hk > 0.005:
        parts.append(hk)
    if not parts:
        return float("nan")
    return float(sum((v / 100) ** 2 for v in parts))


def hold_conc_series(d: dict) -> pd.DataFrame:
    """前十大集中度与披露只数的季度序列（用于「集中度在升还是在降」）。"""
    h = d.get("q_hold")
    if h is None or h.empty:
        return pd.DataFrame()
    rows = []
    for q, g in h.groupby("quarter"):
        t10 = g[g["rnk"] <= 10]
        rows.append({"quarter": q, "top10": float(t10["ratio_pct"].sum()),
                     "n": int(len(g))})
    out = pd.DataFrame(rows).sort_values("quarter")
    return out


def self_pct(s: pd.Series, value: float | None = None) -> float:
    """某个值在自身历史中的分位（0–100，含当期）。

    默认取序列最后一个值；页面回看历史季度时必须显式传 value，
    否则「2018Q1 的仓位」会被拿去和「2026Q3 的位置」比较，得出完全错误的结论。
    """
    v = pd.to_numeric(s, errors="coerce").dropna()
    if len(v) < 2:
        return float("nan")
    x = float(v.iloc[-1]) if value is None else float(value)
    if not np.isfinite(x):
        return float("nan")
    # 必须带容差：x 与序列里的同一个数可能是两次求和的结果，
    # 二进制约差 1e-14，「v <= x」就会把「等于自己」判成 False，
    # 于是最高/最低那一期被算成「第 0 分位」（实测 007490 的 2026Q3）。
    return float((v <= x + 1e-9).mean() * 100)
