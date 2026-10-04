# -*- coding: utf-8 -*-
"""
wind_ingest.py —— 原始返回体批量解析落盘（机构级数据管道）

把 data/wind_raw/*.json（Wind 原始返回，一字不改）解析成 cache/<task>__<quarter>.json。
原始体保留不动，保证「可回溯」：任何 cache 数值都能追回原始列。

用法：
    python tools/wind_ingest.py <原始文件.json> <task>
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import wind_multi as wm  # noqa: E402
import wind_save as ws   # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache")


def ingest(path: str, task: str) -> None:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    if task in ("returns", "turnover"):
        data = wm.extract_multi(payload, task)
        if not data:
            print(f"!! {os.path.basename(path)} 解析为空（task={task}）")
            return
        for q in sorted(data):
            dest = os.path.join(CACHE, f"{task}__{q}.json")
            # ------------------------------------------------------------------
            # 关键：**合并写**，不是覆盖写。
            # Wind 单次返回受行数上限约束，同一季度会被拆到多个批次（行业分组）。
            # 若覆盖写，后一批会把前一批已抓到的行业抹掉，造成静默数据丢失。
            # ------------------------------------------------------------------
            old = {}
            if os.path.exists(dest):
                with open(dest, encoding="utf-8") as f:
                    old = json.load(f)
            before = len(old)
            old.update(data[q])
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(old, f, ensure_ascii=False, indent=1)
            new = len(old) - before
            flag = "" if len(old) >= 31 else f"  ⚠️{len(old)}/31"
            print(f"  ✓ {task:<9} {q}: +{new:>2} → {len(old):>2} 行业{flag}")
    else:
        data, cols, vals = ws.extract(payload, task)
        if not data:
            print(f"!! {os.path.basename(path)} 解析为空（task={task}）")
            return
        print(f"  ✓ {task}: {len(data)} 项  取值列={vals[0] if vals else '?'}")


if __name__ == "__main__":
    ingest(sys.argv[1], sys.argv[2])
