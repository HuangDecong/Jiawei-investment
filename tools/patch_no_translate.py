# -*- coding: utf-8 -*-
"""禁止浏览器自动翻译的幂等补丁（修复 "Wind"→"风"、"HHI"→"嘻" 的篡改）。

背景
----
浏览器（Chrome / Edge / QQ 浏览器）对中文页面会提供"自动翻译"，
它会把页面里的英文专有名词当成待翻译词：

    "Wind"（数据源）  → 被译成中文名词「风」
    "HHI"（赫芬达尔指数） → 被音译/误译成「嘻」

结果是数据标签被替换成无关汉字，直接造成误读。这不是应用文本的问题——
代码里根本没有"风/嘻"二字，而是浏览器端翻译层注入的。

解决原理
--------
Chrome 系浏览器遵守两项指令，命中其一即停止自动翻译：
    1. <meta name="google" content="notranslate">
    2. 任意元素上的 translate="no"（本补丁打在 <html> 根元素上）

Streamlit 的页面骨架由其自带静态文件 static/index.html 提供，
应用代码无法插入 <head>，因此该补丁直接修改该文件（已备份 .bak）。
升级 / 重装 streamlit 后重新运行本脚本即可。

用法
----
    python tools/patch_no_translate.py          # 打补丁（幂等）
    python tools/patch_no_translate.py --check  # 仅检查
    python tools/patch_no_translate.py --restore   # 还原备份
"""
from __future__ import annotations

import os
import sys

import streamlit

IDX = os.path.join(os.path.dirname(streamlit.__file__), "static", "index.html")
BAK = IDX + ".bak"

META = ('    <!-- 禁止浏览器自动翻译：自动翻译会把页面中的英文专有名词\n'
        '         （如 Wind→“风”、HHI→“嘻”）篡改为无关中文，导致数据误读。\n'
        '         本文件由 tools/patch_no_translate.py 维护，可幂等重打。 -->\n'
        '    <meta name="google" content="notranslate" />\n'
        '    <meta name="translate" content="no" />\n')


def patched(text: str) -> bool:
    return 'content="notranslate"' in text and 'translate="no"' in text


def main() -> int:
    if "--restore" in sys.argv:
        if os.path.exists(BAK):
            open(IDX, "w", encoding="utf-8").write(
                open(BAK, encoding="utf-8").read())
            print("已还原:", IDX)
        else:
            print("无备份文件")
        return 0

    src = open(IDX, encoding="utf-8").read()
    if "--check" in sys.argv:
        print("已打补丁" if patched(src) else "未打补丁")
        return 0 if patched(src) else 1

    if patched(src):
        print("已是最新，无需修改:", IDX)
        return 0

    if not os.path.exists(BAK):
        open(BAK, "w", encoding="utf-8").write(src)
        print("已备份:", BAK)

    # 1) 根元素加 translate="no"，并把语言标为中文（避免浏览器判定"需翻译"）
    if '<html lang="en">' in src:
        src = src.replace('<html lang="en">', '<html lang="zh-CN" translate="no">', 1)
    elif 'translate="no"' not in src:
        src = src.replace("<html", '<html translate="no"', 1)

    # 2) 在 <meta charset> 之后插入 notranslate meta
    anchor = '<meta charset="UTF-8" />'
    if anchor in src and 'content="notranslate"' not in src:
        src = src.replace(anchor, anchor + "\n" + META, 1)

    open(IDX, "w", encoding="utf-8").write(src)
    print("补丁已写入:", IDX)
    print("重启 streamlit 服务后生效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
