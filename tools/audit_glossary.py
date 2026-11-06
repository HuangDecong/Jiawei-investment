# -*- coding: utf-8 -*-
"""指标词典审计：条目完整、术语全覆盖、渲染不报错。

为什么单独审计
--------------
「词典里有 80 条」不等于「页面上的数都解释过」。真正的检查是两件事：
  1. **完整性**：每条都必须有九段（含义/原理/公式/步骤/阈值/来源/经济/投资/误用），
     缺一段就等于没写完——读者查到一半会更困惑；
  2. **覆盖性**：页面文案里出现过的每个指标与术语，都必须能检索到条目。

用法：
    python tools/audit_glossary.py          # 打印问题清单，返回码 0/1
    python tools/audit_glossary.py --list   # 另外打印全部条目名
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import glossary as g  # noqa: E402

FIELDS = ["name", "cat", "unit", "meaning", "principle", "formula", "steps",
          "threshold", "source", "econ", "invest", "misuse"]


def check_fields() -> list[str]:
    bad = []
    for k, v in g.ITEMS.items():
        for f in FIELDS:
            val = v.get(f)
            if val is None or (isinstance(val, str) and not val.strip()) or \
                    (isinstance(val, (list, tuple)) and not val):
                bad.append(f"{k} 缺 {f}")
        if v.get("cat") not in g.CATS:
            bad.append(f"{k} 分类非法：{v.get('cat')}")
    return bad


def check_terms() -> list[str]:
    return [f"检索不到：{t}" for t in g.AUDIT_TERMS if not g.search(t)]


def check_presence() -> list[str]:
    """术语必须真的出现在页面源码里，否则是「词典写了但页面没用」，属于过期条目。"""
    blob = ""
    for p in glob.glob(os.path.join(ROOT, "src", "*.py")) + \
            [os.path.join(ROOT, "app.py")]:
        try:
            blob += open(p, encoding="utf-8").read()
        except Exception:
            pass
    return [f"页面里没出现：{t}" for t in g.AUDIT_TERMS if t not in blob]


def check_render() -> list[str]:
    bad = []
    color = {"price": "#1F6FEB", "accent": "#6E4AB8", "up": "#C62828",
             "threshold": "#E8710A", "line": "#DCE3EA", "grid": "#EEF2F6",
             "text_dim": "#6B7A8C", "panel": "#F7F9FC", "text": "#16283C",
             "bg": "#FFFFFF", "level_拥挤": "#C62828"}
    for k, v in g.ITEMS.items():
        try:
            html = g.card_html(k, color)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{k} 渲染异常 {type(e).__name__}: {e}")
            continue
        if v["name"] not in html:
            bad.append(f"{k} 渲染结果缺少名称")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    probs = []
    for name, fn in (("字段完整性", check_fields), ("术语覆盖", check_terms),
                     ("术语是否真被页面使用", check_presence),
                     ("卡片渲染", check_render)):
        p = fn()
        print(f"{'✅' if not p else '❌'} {name}：{len(p)} 个问题")
        for x in p[:12]:
            print("   -", x)
        probs += p
    ok, all_ = g.coverage()
    print(f"\n词典 {len(g.ITEMS)} 条，{len(g.CATS)} 类；"
          f"页面术语覆盖 {ok}/{all_}")
    if args.list:
        for cat in g.CATS:
            names = [v["name"] for v in g.ITEMS.values() if v["cat"] == cat]
            print(f"  {cat}（{len(names)}）：" + "、".join(names))
    return 0 if not probs else 1


if __name__ == "__main__":
    raise SystemExit(main())
