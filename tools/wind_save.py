# -*- coding: utf-8 -*-
"""
wind_save.py —— 把一次 Wind 查询结果规范化并落盘到 data/cache/

用法
----
    python tools/wind_save.py <task> <quarter> <json文件>

其中 <json文件> 是 Agent 从 Wind 拿到后原样保存的返回体（形如
{"data": {"data": [{"columns": [...], "rows": [[...], ...]}]}, "error": null}）。
本脚本负责：
  1) 从列名里自动识别「行业名」列与「数值」列（Wind 列名随查询措辞漂移，必须容错）；
  2) 行业名归一（去掉 "(申万)" 后缀、匹配 31 个申万一级行业）;
  3) 落盘为 {task}__{quarter}.json，形如 {"电子": 17704.03, ...}；
  4) 记录 meta（原始列名、行数、抓取时刻），便于审计。
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "data", "cache")
sys.path.insert(0, os.path.join(ROOT, "src"))

import config as C  # noqa: E402

ALIAS = {"名称": "name", "证券简称": "name", "行业名称": "name",
         "申万一级行业名称": "name", "行业": "name"}

# 每类任务要取的数值列应含有的关键字 / 应排除的关键字。
# Wind 返回同一实体多个数值列时（区间涨跌幅、区间成交额、市值…），
# 必须按语义锁定，不能靠列序。
VALUE_HINT = {
    "returns":  (["涨跌幅", "涨幅"], ["成交额", "成交金额", "市值", "占比", "点位"]),
    "turnover": (["成交额", "成交金额"], ["涨跌幅", "涨幅", "市值", "占比", "点位"]),
    # holdings 的列名漂移较大："该申万一级行业总市值"、"基金重仓该行业的股票市值合计"、
    # "该申万一级行业基金重仓股的总市值"…统一特征 = 含"市值"。
    # 注意：**不能**把"合计"列入 avoid —— 列名"…股票市值合计"里的"合计"
    # 只是措辞，不是"总计行"标记。早期误将"合计"列入 avoid，
    # 导致 2025Q2 / 2024Q4 整季解析失败（返回 {}）。
    "holdings": (["市值"], ["占比", "涨跌幅", "成交额", "代码", "总市值合计"]),
    "hs300":    (["权重", "占比"], ["涨跌幅", "成交额", "市值", "代码"]),
}


def _norm_ind(s: str) -> str | None:
    """归一行业名：'电子(申万)' -> '电子'；不匹配返回 None。"""
    if s is None:
        return None
    t = re.sub(r"[（(]申万[)）]", "", str(s)).strip()
    t = t.replace("(申万)", "").replace("（申万）", "").strip()
    for ind in C.INDUSTRIES:
        if t == ind:
            return ind
    # 容忍轻微差异（如 '石油石化' vs '石油石化(申万)'）
    for ind in C.INDUSTRIES:
        if t.startswith(ind):
            return ind
    return None


def extract(payload: dict, task: str = "") -> tuple:
    """从 Wind 返回体中抽取 {行业: 数值}。

    关键：Wind 常在同一实体上返回**多个数值列**（区间涨跌幅 / 区间成交额 /
    总市值…），且列序不稳定。因此必须按 task 的语义锁定取值列，
    否则会把"成交额"当成"涨跌幅"（本项目早期实测踩过）。
    """
    try:
        blocks = payload["data"]["data"]
    except (KeyError, TypeError):
        return {}, [], []
    out: dict[str, float] = {}
    raw_cols = []
    val_names = []
    want, avoid = VALUE_HINT.get(task, ([], []))
    for blk in blocks:
        cols = [c["name"] for c in blk.get("columns", [])]
        raw_cols.append(cols)
        # 找行业列（字符串型）
        i_ind = None
        for i, c in enumerate(cols):
            if c in ALIAS or "名称" in c or ("行业" in c and "占比" not in c):
                i_ind = i
                break
        if i_ind is None:
            continue
        # 找数值列：命中 want 关键字数最多者优先；含 avoid 关键字直接淘汰。
        cands = []
        for i, c in enumerate(blk.get("columns", [])):
            if i == i_ind or c.get("type") != "number":
                continue
            nm = c["name"]
            if "代码" in nm or any(a in nm for a in avoid):
                continue
            hit = sum(1 for w in want if w in nm)
            dated = 1 if ("年" in nm and "月" in nm) else 0   # 精确日期区间优先
            cands.append((hit, dated, i))
        if not cands:
            continue
        cands.sort(key=lambda t: (-t[0], -t[1]))
        i_val = cands[0][2]
        val_names.append(cols[i_val])
        for row in blk.get("rows", []):
            if row is None or len(row) <= max(i_ind, i_val):
                continue
            ind = _norm_ind(row[i_ind])
            v = row[i_val]
            if ind and isinstance(v, (int, float)):
                # 同一行业出现多次时取绝对值较大的（避免空分组覆盖）
                if ind not in out or abs(v) > abs(out[ind]):
                    out[ind] = float(v)
    return out, raw_cols, val_names


def save(task: str, quarter: str, payload_path: str) -> None:
    with open(payload_path, encoding="utf-8") as f:
        payload = json.load(f)
    data, raw_cols, val_names = extract(payload, task)
    dest = os.path.join(CACHE, f"{task}__{quarter}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"{task} {quarter}: 提取 {len(data)} 个行业 -> {os.path.basename(dest)}")
    print(f"  取值列 = {val_names}")
    if len(data) < 31:
        miss = [i for i in C.INDUSTRIES if i not in data]
        print(f"  ⚠️ 缺失 {len(miss)} 个：{miss}")
    print(f"  raw cols = {raw_cols[0] if raw_cols else []}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    save(sys.argv[1], sys.argv[2], sys.argv[3])
