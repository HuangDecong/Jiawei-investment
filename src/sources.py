# -*- coding: utf-8 -*-
"""全站数据来源登记处（Single Source of Truth for citations）。

为什么单独一个模块
------------------
「来源」如果散落在各页的 tiny 文案里，改一次链接要改七个文件，而且必然有页面漏写。
这里统一登记：谁提供数据、哪一天取的、原始披露在哪、用什么代码算的。
页面只调用 `source_line(page)`，方法页与导出文件也读同一份。

口径
----
· 基金持仓/资产/份额：基金定期报告（季报/半年报/年报）。
  原始披露在证监会基金电子披露网站（EID）；本库经东方财富基金档案转引，
  字段级来源写在每行的 src 列（`data/db/fund_q.db`）。
· 行业分类：申万宏源研究《申万行业分类标准》一级行业 31 个。
· 行业指数：申万一级行业指数（日频），由 Wind 终端导出后缓存为
  `data/cache/sw_index_daily_ret.csv`；KD 说明见 glossary「价格分位」。
· 全部数字由 `src/canon.py` 计算，阈值/窗口/分母定义在 `src/caliber.py`。

没有链接的项（Wind 终端、申万分类标准）只写名称，不编造 URL。
"""
from __future__ import annotations

import glob
import os
import sqlite3
from datetime import datetime
from functools import lru_cache

import caliber as CAL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "db", "fund_q.db")
DERIVED = os.path.join(ROOT, "data", "derived")
INDEX_CACHE = os.path.join(ROOT, "data", "cache", "sw_index_daily_ret.csv")

# 只放**已验证可访问**的链接（2026-11-13 逐个 curl 200）。
LINKS = {
    "csrc": ("证监会基金电子披露网站", "http://eid.csrc.gov.cn/fund"),
    "em": ("东方财富基金档案", "https://fundf10.eastmoney.com/"),
    "wind": ("Wind 金融终端", "https://www.wind.com.cn/"),
}
EM_FUND = "https://fundf10.eastmoney.com/jjgg_{code}.html"


def _a(key: str) -> str:
    name, url = LINKS[key]
    return (f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
            f'style="color:#1F6FEB;text-decoration:none">{name}</a>')


@lru_cache(maxsize=1)
def fund_fetch_date() -> str:
    """最近一次抓取基金定期报告的日期（本地库）。无库时返回空串。"""
    if not os.path.exists(DB):
        return ""
    try:
        with sqlite3.connect(DB) as con:
            row = con.execute("SELECT MAX(fetched_at) FROM q_log").fetchone()
        return str(row[0])[:10] if row and row[0] else ""
    except Exception:  # noqa: BLE001
        return ""


@lru_cache(maxsize=1)
def snapshot_date() -> str:
    """派生快照（data/derived/*.csv.gz）的最近构建日期。"""
    files = glob.glob(os.path.join(DERIVED, "canon_*.csv.gz"))
    if not files:
        return ""
    ts = max(os.path.getmtime(f) for f in files)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


@lru_cache(maxsize=1)
def index_date() -> str:
    """行业指数日频缓存的最近更新日期。"""
    if not os.path.exists(INDEX_CACHE):
        return ""
    return datetime.fromtimestamp(os.path.getmtime(INDEX_CACHE)).strftime("%Y-%m-%d")


def _date_txt() -> str:
    f, s = fund_fetch_date(), snapshot_date()
    bits = []
    if f:
        bits.append(f"基金数据访问 {f}")
    if s:
        bits.append(f"快照构建 {s}")
    return "｜".join(bits)


PAGE_EXTRA = {
    "overview": "仓位/主动项＝全持仓半年报；环比/广度/分位＝前十大季报（两套口径不可相加）",
    "report": "六张图全部用季报前十大与半年报全持仓的**披露值**，不含任何推算仓位",
    "map": "CSI ＝（配置分位＋价格分位＋资金分位）÷ 3，三项均为分位、等权",
    "fund": "季报原文披露值：资产配置、份额、前十、行业、买卖、债券、财务指标",
    "factor": "IC 的收益端用申万一级行业指数季度收益；持仓端为前十大归一化",
    "method": "三套口径（全持仓 / 前十大 / 个股只数）分别定义，不混算",
}


def source_line(page: str) -> str:
    """页脚来源行（HTML）。所有页面共用同一份表述，避免各写一遍。"""
    extra = PAGE_EXTRA.get(page, "")
    _idx = index_date()
    return (
        '<div class="src-foot" style="margin:14px 0 4px;padding-top:7px;'
        'border-top:1px solid #E6EBF1;font-size:10.5px;line-height:1.85;color:#6B7A8C">'
        '<b style="color:#40505F">来源与出处</b>　'
        f'基金定期报告：原始披露 {_a("csrc")}，本库转引 {_a("em")}'
        + (f'（{_date_txt()}）' if _date_txt() else "")
        + '｜行业分类：申万宏源研究一级行业（31 个）'
        + f'｜行业指数：申万一级指数，{_a("wind")} 导出'
        + (f'（缓存至 {_idx}）' if _idx else "")
        + f'｜计算与口径：<span class="mono">src/canon.py</span> · '
          f'<span class="mono">src/caliber.py</span>（v{CAL.CALIBER_VERSION}）'
        + (f'<br>{extra}' if extra else "")
        + '</div>')


def rendered_markdown() -> str:
    """README / 导出文件用的纯 Markdown 版本（链接可点）。"""
    f, s, i = fund_fetch_date(), snapshot_date(), index_date()
    return (
        "- 基金定期报告：原始披露 [证监会基金电子披露网站]"
        f"({LINKS['csrc'][1]})；本库转引 [东方财富基金档案]({LINKS['em'][1]})"
        + (f"（基金数据访问 {f}" if f else "")
        + (f"，快照构建 {s}）" if s else ")")
        + "\n- 行业分类：申万宏源研究《申万行业分类标准》一级行业（31 个）\n"
        f"- 行业指数：申万一级指数，[Wind 金融终端]({LINKS['wind'][1]}) 导出"
        + (f"（缓存至 {i}）" if i else "")
        + f"\n- 计算与口径：`src/canon.py`、`src/caliber.py`（v{CAL.CALIBER_VERSION}）"
    )
