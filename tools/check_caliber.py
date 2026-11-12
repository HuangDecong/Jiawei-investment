# -*- coding: utf-8 -*-
"""口径闸门：核心文件里**不许再出现**硬编码的口径阈值。

查什么
------
`>= 80`、`< 150`、`> 0.5`、`< 8`、`< 12`、`< 4` 这类比较里写死的数字——
它们必须来自 `src/caliber.py` 的常量。只要有第二处写死，改口径就会漏改，
这正是「口径」类问题在 89 次提交里被修 17 次的根因。

范围
----
· **阻断（blocking）**：真正决定页面数字的文件
  canon / report_charts / map_page / factor_engine / institution / fund_page
· **提示（advisory）**：其余 src/*.py（研究脚本、旧 Wind 口径页等），
  本轮不阻断，但会把命中数打出来，作为待办清单（见 README「未修但已确认」）。

豁免
----
行尾写 `# caliber-ok: 理由` 即可放行；必须写理由，不许空豁免。
豁免也要进 code review——它是「我知道这里有个数字，但它不是口径」。

用法
----
    python tools/check_caliber.py           # 阻断文件有命中 → 退出码 1
    python tools/check_caliber.py --all     # 同时打印非阻断文件的命中
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

BLOCKING = ["src/canon.py", "src/report_charts.py", "src/map_page.py",
            "src/factor_engine.py", "src/institution.py", "src/fund_page.py",
            "src/qreport_page.py"]
NUM = r"(?:80|150|0\.5|12|8|4)(?:\.0)?"
PAT = re.compile(rf"(?:>=|<=|>|<|==)\s*{NUM}\b")


def strip_code(line: str, in_triple: str | None) -> tuple[str, str | None]:
    """去掉注释与三引号字符串内容，尽量只看「真代码」。"""
    if in_triple:
        if in_triple in line:
            return line.split(in_triple, 1)[1], None
        return "", in_triple
    for q in ('"""', "'''"):
        if line.count(q) == 1:
            before, after = line.split(q, 1)
            return before, q
    return line.split("#", 1)[0], None


def scan(path: str) -> list[tuple[int, str]]:
    out, triple = [], None
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except Exception:  # noqa: BLE001
        return out
    for i, raw in enumerate(lines, 1):
        code, triple = strip_code(raw, triple)
        if "caliber-ok" in raw:
            continue
        if PAT.search(code):
            out.append((i, raw.strip()[:120]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="连非阻断文件一起打印")
    args = ap.parse_args()

    import caliber as CAL  # noqa: PLC0415
    bad = CAL.check_consistency()
    if bad:
        print("❌ caliber.py 自身不自洽：")
        for b in bad:
            print("   -", b)
        return 1

    block_hits: dict[str, list] = {}
    for rel in BLOCKING:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            h = scan(p)
            if h:
                block_hits[rel] = h

    others: dict[str, list] = {}
    for p in glob.glob(os.path.join(ROOT, "src", "*.py")):
        rel = "src/" + os.path.basename(p)
        if rel in BLOCKING or rel.endswith("caliber.py"):
            continue
        h = scan(p)
        if h:
            others[rel] = h

    print(f"口径表 {CAL.CALIBER_VERSION}｜阻断文件 {len(BLOCKING)} 个"
          f"｜命中 {sum(len(v) for v in block_hits.values())} 处")
    if block_hits:
        print("\n❌ 核心文件里仍有硬编码阈值（应改为 caliber.CONST）：")
        for rel, h in block_hits.items():
            for ln, txt in h:
                print(f"   {rel}:{ln}  {txt}")
    if args.all and others:
        print(f"\n· 提示：非阻断文件仍有 {sum(len(v) for v in others.values())} 处"
              f"（{len(others)} 个文件），登记为待办：")
        for rel, h in sorted(others.items(), key=lambda kv: -len(kv[1])):
            print(f"   {rel}  {len(h)} 处")
    if block_hits:
        return 1
    print("\n✅ 核心文件无硬编码口径阈值")
    return 0


if __name__ == "__main__":
    sys.exit(main())
