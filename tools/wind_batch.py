# -*- coding: utf-8 -*-
"""
wind_batch.py —— 批量落盘工具

把一批「任务+季度+Wind返回值」一次性写入 cache。
Agent 抓取时把 Wind 原始返回存成 json 片段，本脚本统一解析。

用法：
    python tools/wind_batch.py <描述文件.json>
描述文件格式：
    [{"task": "returns", "quarter": "2026Q2", "payload": {...Wind返回...}}, ...]
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import wind_save as ws  # noqa: E402


def main(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    ok = bad = 0
    for it in items:
        task, q, payload = it["task"], it["quarter"], it["payload"]
        data, raw_cols, val_names = ws.extract(payload, task)
        if not data:
            print(f"  ✗ {task} {q}: 无数据")
            bad += 1
            continue
        dest = os.path.join(ws.CACHE, f"{task}__{q}.json")
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        flag = "" if len(data) >= 31 else f"  ⚠️{len(data)}/31"
        print(f"  ✓ {task:<9} {q}: {len(data):>2} 行业  列={val_names[0] if val_names else '?'}{flag}")
        ok += 1
    print(f"\n完成 {ok} 条，失败 {bad} 条")


if __name__ == "__main__":
    main(sys.argv[1])
