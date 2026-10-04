# -*- coding: utf-8 -*-
"""
wind_ingest_tsv.py —— 从 TSV 落盘 Wind 数据（合并写）

用途：当 Wind 返回体很大、逐条贴 JSON 容易出错时，改为先把返回整理成
TSV（制表符分隔，无引号陷阱），再由本脚本合并进 cache。

TSV 表头（固定顺序）：
    行业 <TAB> 年份 <TAB> 季度 <TAB> 值
# 开头为注释行，自动跳过。

用法：
    python tools/wind_ingest_tsv.py <文件.tsv> <task>
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import wind_save as ws  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache")


def main(path: str, task: str) -> None:
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#")]

    data: dict[str, dict] = {}
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) < 4:
            continue
        ind_raw, y, q, v = parts[0], parts[1], parts[2], parts[3]
        ind = ws._norm_ind(ind_raw)
        try:
            key = f"{int(y)}Q{int(q)}"
            val = float(v)
        except ValueError:
            continue
        if ind:
            data.setdefault(key, {})[ind] = val

    if not data:
        print(f"!! {os.path.basename(path)} 无有效行")
        return

    for q in sorted(data):
        dest = os.path.join(CACHE, f"{task}__{q}.json")
        # 合并写：同一季度可能来自多个批次（行业分组）
        old = {}
        if os.path.exists(dest):
            with open(dest, encoding="utf-8") as f:
                old = json.load(f)
        before = len(old)
        old.update(data[q])
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(old, f, ensure_ascii=False, indent=1)
        flag = "" if len(old) >= 31 else f"  ⚠️{len(old)}/31"
        print(f"  ✓ {task:<9} {q}: +{len(old) - before:>2} → {len(old):>2} 行业{flag}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
