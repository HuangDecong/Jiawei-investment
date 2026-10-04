# -*- coding: utf-8 -*-
"""holdings 重抓结果落盘工具（覆盖写）。

与 wind_ingest_tsv.py 的合并写不同，本工具用于**修正口径**：
目标季度需被完整替换，因此采用覆盖写，但会在覆盖前打印旧值供审计比对。

TSV 格式（两列，制表符分隔，`#` 开头为注释）：
    <行业名>\t<市值亿元>

用法：
    python tools/holdings_ingest.py <tsv路径> <季度，如 2024Q2>
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C  # noqa: E402

CACHE = os.path.join(C.DATA, "cache")


def _norm_ind(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"[（(](申万|SW)[^）)]*[）)]", "", s).strip()
    return s


def main(path: str, q: str) -> None:
    data: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip() or ln.startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) < 2:
                continue
            ind = _norm_ind(parts[0])
            try:
                v = float(parts[1])
            except ValueError:
                continue
            if ind in C.INDUSTRIES:
                data[ind] = v

    if not data:
        raise SystemExit(f"未解析出任何行业：{path}")

    dest = os.path.join(CACHE, f"holdings__{q}.json")
    old = {}
    if os.path.exists(dest):
        with open(dest, encoding="utf-8") as f:
            old = json.load(f)

    tot_new = sum(data.values())
    tot_old = sum(old.values()) if old else 0.0
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    miss = [i for i in C.INDUSTRIES if i not in data]
    print(f"  ✓ holdings  {q}: {len(data)}/31 行业  合计 {tot_new:,.0f} 亿"
          f"  (旧 {tot_old:,.0f} 亿, {tot_new/max(tot_old,1e-9)-1:+.1%})")
    if miss:
        print(f"      ⚠️ 缺行业: {','.join(miss)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
