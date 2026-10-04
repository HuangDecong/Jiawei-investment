# -*- coding: utf-8 -*-
"""
wind_multi.py —— 多季度时序批量解析

Wind 的 get_financial_data 在问句里给出**跨季度区间**时，会返回带
「年份」「季度」列的时序表（列名形如"2024年二季度到2026年二季度的每季涨跌幅"）。
一次查询即可覆盖数年 × 31 行业，把 66 次单季查询压缩到几次。

本脚本解析这种时序返回：
    python tools/wind_multi.py <task> <payload.json>
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "data", "cache")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import config as C          # noqa: E402
import wind_save as ws      # noqa: E402

VALUE_HINT = {
    "returns":  (["涨跌幅", "涨幅"], ["成交额", "成交金额", "市值", "点位", "代码"]),
    "turnover": (["成交额", "成交金额"], ["涨跌幅", "涨幅", "市值", "点位", "代码"]),
    "holdings": (["市值"], ["占比", "涨跌幅", "成交额", "代码", "合计"]),
    "hs300":    (["权重", "占比"], ["涨跌幅", "成交额", "市值", "代码"]),
}


def _q_from_span(start: str, end: str) -> str | None:
    """由 Wind 返回的起止日期推断**单季**季度键；若不是单季则返回 None。

    关键陷阱：Wind 时序表里 Q4 行常常是「YYYY-01-01 至 YYYY-12-31」的**全年累计**，
    而不是 Q4 单季。若直接用「季度」列，会把全年涨跌幅当成 Q4 涨跌幅，
    导致 Q4 数据被系统性放大（本项目实测：2010Q4 电子被写成 +39.39%，
    实际单季为 +15.81%）。故必须用 (start, end) 严格校验。
    """
    if not start or not end:
        return None
    m1 = re.match(r"^(\d{4})-(\d{2})-(\d{2})", str(start))
    m2 = re.match(r"^(\d{4})-(\d{2})-(\d{2})", str(end))
    if not (m1 and m2):
        return None
    y1, mo1, d1 = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
    y2, mo2, d2 = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
    if y1 != y2:                       # 跨年 -> 非单季
        return None
    # 起始必须是季初（1/4/7/10 月 1 日）
    if not (mo1 in (1, 4, 7, 10) and d1 == 1):
        return None
    if mo1 == 1 and mo2 not in (3,):   # 1/1 起只有 3/31 才是 Q1 单季
        return None
    if mo1 == 4 and mo2 != 6:
        return None
    if mo1 == 7 and mo2 != 9:
        return None
    if mo1 == 10 and mo2 != 12:
        return None
    if (mo1, mo2) == (10, 12) and d2 != 31:
        return None
    q = {1: 1, 4: 2, 7: 3, 10: 4}[mo1]
    return f"{y1}Q{q}"


def extract_multi(payload: dict, task: str) -> dict:
    """返回 {季度: {行业: 值}}。"""
    want, avoid = VALUE_HINT.get(task, ([], []))
    out: dict[str, dict] = {}
    try:
        blocks = payload["data"]["data"]
    except (KeyError, TypeError):
        return out

    for blk in blocks:
        cols = blk.get("columns", [])
        # 定位列
        i_ind = i_y = i_q = None
        i_val = i_s = i_e = None
        best = -1
        for i, c in enumerate(cols):
            nm = c["name"]
            # 行业列识别：Wind 列名在多次查询间会漂移。
            # 实测出现过：「申万一级行业名称」「证券简称」「行业名称」「Wind代码」旁的中文名。
            # 故用「名称/简称/行业」三类关键字联合判定，并排除「占比」列避免误命中。
            if i_ind is None and (
                nm in ws.ALIAS
                or "名称" in nm
                or "简称" in nm
                or ("行业" in nm and "占比" not in nm)
            ):
                i_ind = i
            if nm == "年份":
                i_y = i
            if nm == "季度":
                i_q = i
            if "起始时间" in nm or nm.endswith("起始"):
                i_s = i
            if "截止时间" in nm or nm.endswith("截止"):
                i_e = i
            # 取值列筛选：
            # 注意——不能用 `"年" not in nm`。Wind 会把指标列命名为
            # 「2022到2024年每季涨跌幅」，含"年"字，粗暴排除会把唯一取值列误杀
            # （本项目实测：returns_2022_2024_d.json 因此整批解析为空）。
            # 正确做法是**精确**排除「年份」「季度」两个真正的维度列。
            if c.get("type") == "number" and "代码" not in nm \
                    and nm != "年份" and nm != "季度" \
                    and not any(a in nm for a in avoid):
                hit = sum(1 for w in want if w in nm)
                if hit > best:
                    best, i_val = hit, i
        if i_ind is None or i_val is None:
            continue
        for row in blk.get("rows", []):
            if row is None or len(row) <= max(i_ind, i_val):
                continue
            ind = ws._norm_ind(row[i_ind])
            v = row[i_val]
            if not ind or not isinstance(v, (int, float)):
                continue
            # 判定季度键。
            # 若返回体带起止时间列，**必须**以起止时间为准：span 判定为 None
            # 说明该行不是单季（典型是 Q4 的全年累计），此时应**丢弃该行**，
            # 绝不能回退到「季度」列 —— 回退会把"全年涨跌幅"写成"Q4 涨跌幅"。
            if i_s is not None and i_e is not None and len(row) > max(i_s, i_e):
                key = _q_from_span(row[i_s], row[i_e])
                if key is None:
                    continue
            elif i_y is not None and i_q is not None \
                    and len(row) > max(i_y, i_q) \
                    and isinstance(row[i_y], (int, float)) \
                    and isinstance(row[i_q], (int, float)):
                key = f"{int(row[i_y])}Q{int(row[i_q])}"
            else:
                continue
            out.setdefault(key, {})[ind] = float(v)
    return out


def main(task: str, path: str) -> None:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    data = extract_multi(payload, task)
    if not data:
        print("!! 未解析到多季度数据，回退单季解析")
        return
    print(f"解析出 {len(data)} 个季度：{sorted(data)[0]} … {sorted(data)[-1]}")
    for q in sorted(data):
        d = data[q]
        dest = os.path.join(CACHE, f"{task}__{q}.json")
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        flag = "" if len(d) >= 31 else f"  ⚠️{len(d)}/31"
        print(f"  ✓ {q}: {len(d):>2} 行业{flag}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
