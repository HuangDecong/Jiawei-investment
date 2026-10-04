# -*- coding: utf-8 -*-
"""hs300 行业权重落盘工具。

Wind 的 hs300 查询会返回两种形态：
  ① 个股级成份权重明细（按「指数成份代码/简称」逐只列出）——需自行按申万一级行业分组求和；
  ② 行业级总市值占比（列名形如「2025年6月30日沪深300行业总市值占比」，单位 %）
     ——**这是权威口径，直接可用**，也是本工具优先采纳的来源。

设计要点（机构级）：
  - 合并写：同一季度可能分多批抓到，覆盖写会静默丢数据（本项目已踩过坑）。
  - 严格取「沪深300行业总市值占比」列，避免误取「行业总市值」（万亿元）或「A股总市值合计」。
  - 输出格式与既有 hs300__2026Q2.json 完全一致：{行业: 权重百分比}
  - 打印 +N → M 便于审计。

用法：
  python tools/wind_ingest_hs300.py <原始文件.json>
原始文件支持三种结构：
  A) {"block3_行业总市值占比": {"2024Q4": {...}, ...}}   —— 预整理格式
  B) Wind 原始 {"data": {"data": [ {columns, rows}, ... ]}}  —— 自动识别形态
  C) {"blocks": [ {columns, rows}, ... ]}
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402

CACHE = os.path.join(C.DATA, "cache")
os.makedirs(CACHE, exist_ok=True)

# 「2025年6月30日沪深300行业总市值占比」→ 2025-06-30 → 2025Q2
DATE_IN_COL = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")


def _q_from_date(y: int, m: int) -> str:
    return f"{y}Q{(m - 1) // 3 + 1}"


def _norm_ind(name: str) -> str:
    """行业名归一化：'电子(申万)' → '电子'。"""
    s = str(name).strip()
    s = re.sub(r"[（(](申万|SW|中信)[^）)]*[）)]", "", s).strip()
    return s


def _save_quarter(q: str, data: dict) -> int:
    """合并写单季度。返回新增行业数。"""
    dest = os.path.join(CACHE, f"hs300__{q}.json")
    old = {}
    if os.path.exists(dest):
        with open(dest, encoding="utf-8") as f:
            old = json.load(f)
    before = len(old)
    for k, v in data.items():
        old[k] = round(float(v), 6)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=1)
    return len(old) - before, len(old)


def _ingest_blocks(blocks: list) -> dict:
    """从 Wind 返回的 blocks（[{columns, rows}, ...]）中抽取行业级权重占比。

    识别规则：若某 block 同时含「申万一级行业名称」列与列名里带
    「沪深300行业总市值占比」的数值列，则按该 block 落盘。
    """
    out: dict[str, dict] = {}
    for blk in blocks:
        cols = [c.get("name", "") for c in blk.get("columns", [])]
        rows = blk.get("rows", [])
        if not cols or not rows:
            continue
        # 行业列
        i_ind = None
        for i, cn in enumerate(cols):
            if "申万一级行业" in cn or cn in ("行业名称", "所属申万行业"):
                i_ind = i
                break
        if i_ind is None:
            continue
        # 取值列：识别「占沪深300总市值的比例/占比」。
        # Wind 实测出现两种命名与两种量纲：
        #   A) 「2026年3月31日沪深300指数成份权重」            单位 %   → 已是百分数
        #   B) 「2025年9月30日该行业总市值2占沪深300总市值2的比例」 无单位 → 小数，需 ×100
        # 统一换算成「百分数」后落盘。
        value_cols = {}
        for i, cn in enumerate(cols):
            if "A股" in cn:
                continue
            if "沪深300" not in cn:
                continue
            # B 形态优先：行业总市值占沪深300比例（小数）
            if "占" in cn and ("比例" in cn or "占比" in cn):
                scale = 100.0
            elif "占比" in cn or "权重" in cn:
                scale = 1.0
            else:
                continue
            m = DATE_IN_COL.search(cn)
            if m:
                q = _q_from_date(int(m.group(1)), int(m.group(2)))
                value_cols[i] = (q, scale)
        if not value_cols:
            continue
        for r in rows:
            ind = _norm_ind(r[i_ind]) if r[i_ind] is not None else ""
            if not ind or ind not in C.INDUSTRIES:
                continue
            for i, (q, scale) in value_cols.items():
                v = r[i]
                if v is None:
                    continue
                # 形态 B 若已是百分数（>1），说明本次没返回小数，则不重复放大
                sv = float(v)
                out.setdefault(q, {})[ind] = sv * scale if (scale == 100.0 and sv <= 1.5) else sv
    return out


def main(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    # 形态 A：预整理（百分数）
    if isinstance(payload, dict) and "block3_行业总市值占比" in payload:
        per_q = {}
        for q, d in payload["block3_行业总市值占比"].items():
            per_q[q] = {_norm_ind(k): float(v) for k, v in d.items()}
    # 形态 A2：预整理（小数形式，需 ×100）
    elif isinstance(payload, dict) and "block_行业占比_小数" in payload:
        per_q = {}
        for q, d in payload["block_行业占比_小数"].items():
            per_q[q] = {_norm_ind(k): float(v) * 100.0 for k, v in d.items()}
    # 形态 A3：行业总市值（万亿）+ 沪深300总市值（万亿）→ 相除得占比
    elif isinstance(payload, dict) and "行业总市值_万亿" in payload:
        per_q = {}
        den = payload["沪深300总市值_万亿"]
        for q, d in payload["行业总市值_万亿"].items():
            base = float(den[q])
            per_q[q] = {_norm_ind(k): round(float(v) / base * 100.0, 6) for k, v in d.items()}
    else:
        # 形态 B / C：Wind 原始
        blocks = None
        node = payload
        if isinstance(node, dict) and "data" in node:
            node = node["data"]
        if isinstance(node, dict) and "data" in node:
            node = node["data"]
        if isinstance(node, list):
            blocks = node
        elif isinstance(node, dict) and "blocks" in node:
            blocks = node["blocks"]
        if blocks is None:
            raise SystemExit("无法识别返回体结构：既无 block3_行业总市值占比，也无 blocks/data 列表")
        per_q = _ingest_blocks(blocks)

    if not per_q:
        raise SystemExit("未解析出任何行业级权重，请检查返回体是否包含「沪深300行业总市值占比」列")

    for q in sorted(per_q):
        new, tot = _save_quarter(q, per_q[q])
        flag = "" if tot >= 25 else f"  ⚠️{tot}/28"
        print(f"  ✓ hs300     {q}: +{new:>2} → {tot:>2} 行业{flag}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
