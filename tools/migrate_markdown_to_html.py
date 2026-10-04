# -*- coding: utf-8 -*-
"""
migrate_markdown_to_html.py —— 把块级 HTML 从 st.markdown 迁移到 st.html

为什么必须迁移（根因说明）
--------------------------
用户报告的三种前端错误：
    未能在"节点"上执行"removeChild"：被移除的节点不是该节点的子节点
    未能在"节点"上执行"insertBefore"：新节点要插入的节点不是该节点的子节点

这两类都是 React 的 **真实 DOM 与虚拟 DOM 不一致** 的表现。
在 Streamlit 里，触发源几乎只有一个：`st.markdown(..., unsafe_allow_html=True)`。

机制：st.markdown 的内容经过 **两阶段解析**——
   ① react-markdown 先把字符串解析成 markdown 语法树；
   ② rehype-raw 再用 parse5（严格 HTML5 树构造算法）把内联 HTML 解析进来。
第二阶段会按规范**自动改写**树结构：
   · <tr> 直接写在 <table> 下时，parse5 会隐式插入 <tbody>；
   · 块级元素（<div>/<table>）若落在段落上下文，会被 "foster parenting" 提到 <p> 之外；
   · 行内元素（<b>）包含块级元素时会被提前闭合。
React 之后按它自己那棵树去 insertBefore/removeChild，浏览器里却已不是那棵树
→ 切换页面触发重渲染时抛错。

修复：改用 `st.html`。它**只做一次解析**（DOMPurify 消毒后直接挂载），
不存在 markdown 阶段，也就不存在两棵树的分歧。

已实测确认：
  · st.html 保留 class 与 inline style 属性（样式不会丢）；
  · st.html 不经过 markdown，<table>/<tr> 结构原样保留。

迁移规则
--------
  st.markdown(<X>, unsafe_allow_html=True)  →  _h(<X>)

  例外：含 <style> 的块**保持不动**（全局 CSS 注入，st.markdown 已验证可用；
        且 style 标签经 DOMPurify 可能被剥离，不动更安全）。

本脚本用 AST 定位调用、用 ast.get_source_segment 原样搬运实参文本，
因此不会改动任何字符串内容、注释或格式。

用法：
    python tools/migrate_markdown_to_html.py --dry-run   # 只报告
    python tools/migrate_markdown_to_html.py             # 执行迁移（自动备份）
"""
from __future__ import annotations

import ast
import os
import shutil
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
APP = os.path.join(ROOT, "app.py")

HELPER = '''

# ============================================================================
# 原始 HTML 渲染器（本轮新增）
# ============================================================================
# 为什么不用 st.markdown(..., unsafe_allow_html=True)？
# ---------------------------------------------------------------------------
# st.markdown 的内容要经过**两阶段解析**：react-markdown 先解析 markdown，
# rehype-raw 再用 parse5 解析内联 HTML。parse5 严格遵循 HTML5 树构造算法，
# 会按规范自动改写结构（<tr> 隐式包 <tbody>、块级元素被 foster parenting 提出
# 段落、行内元素被提前闭合）。于是浏览器里的真实 DOM 与 React 的虚拟 DOM 不再
# 一致，切换页面触发重渲染时就抛：
#     NotFoundError: removeChild / insertBefore —— 节点不是该节点的子节点
#
# st.html 只解析一次（DOMPurify 消毒后直接挂载），没有 markdown 阶段，
# 因此不存在两棵树的分歧。已实测确认 class 与 inline style 均保留。
def _h(html: str):
    """渲染原始 HTML 片段（替代 st.markdown(..., unsafe_allow_html=True)）。"""
    return st.html(html)
'''


def _offsets(src: str) -> tuple[list[int], list[str]]:
    """返回 (每行起始的绝对字符偏移, 每行文本)。

    ⚠️ 一个必须记住的坑
    -------------------
    Python AST 的 `col_offset` / `end_col_offset` 是 **UTF-8 字节偏移**，不是字符偏移。
    本文件含大量中文（每字 3 字节），若把它们直接当字符偏移用，
    替换区间会**严重右移**，把后面的代码一并吞掉（实测会把
    `_bb = pd.DataFrame({"行业": B.industries,` 的前半段删掉，只剩一个 `s,`）。
    正确做法：先把字节偏移换算回字符偏移。
    """
    lines = src.splitlines(keepends=True)
    offs, pos = [0], 0
    for ln in lines:
        pos += len(ln)
        offs.append(pos)
    return offs, lines


def _byte_to_char(line: str, byte_col: int) -> int:
    """把该行的 UTF-8 字节列号换算为字符列号。"""
    if byte_col <= 0:
        return 0
    b = line.encode("utf-8")
    if byte_col >= len(b):
        return len(line)
    return len(b[:byte_col].decode("utf-8", errors="ignore"))


def main() -> int:
    dry = "--dry-run" in sys.argv
    with open(APP, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    offs, raw_lines = _offsets(src)

    def _abs(lineno: int, byte_col: int) -> int:
        """AST 的 (行号, 字节列) → 绝对字符偏移。"""
        line = raw_lines[lineno - 1] if lineno - 1 < len(raw_lines) else ""
        return offs[lineno - 1] + _byte_to_char(line, byte_col)

    edits: list[tuple[int, int, str]] = []      # (abs_start, abs_end, new_text)
    kept_style = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "markdown"
                and isinstance(fn.value, ast.Name) and fn.value.id == "st"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        ua = kw.get("unsafe_allow_html")
        if not (isinstance(ua, ast.Constant) and ua.value is True):
            continue
        if len(node.args) != 1:
            continue
        seg = ast.get_source_segment(src, node.args[0])
        if seg is None:
            continue
        if "<style>" in seg:            # 全局样式注入：保持原样
            kept_style += 1
            continue
        a = _abs(node.lineno, node.col_offset)
        b = _abs(node.end_lineno, node.end_col_offset)
        edits.append((a, b, f"_h({seg})"))

    print(f"可迁移块：{len(edits)}　｜　保留 <style> 块：{kept_style}")
    if dry:
        for a, b, txt in edits[:8]:
            head = txt.splitlines()[0] if txt.splitlines() else txt
            print(f"  @{a:<7} {head[:88]}")
        return 0

    if not edits:
        print("无需迁移")
        return 0

    shutil.copy2(APP, APP + ".bak_md2html")
    # 逆序（按偏移从大到小）替换，前面的偏移不受影响
    out = src
    for a, b, new in sorted(edits, key=lambda t: t[0], reverse=True):
        out = out[:a] + new + out[b:]

    # 在顶层 import 段之后插入 _h 定义
    tree2 = ast.parse(out)
    anchor = 0
    for n in tree2.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            anchor = max(anchor, getattr(n, "end_lineno", n.lineno))
    olines = out.splitlines(keepends=True)
    out = "".join(olines[:anchor]) + HELPER + "".join(olines[anchor:])

    with open(APP, "w", encoding="utf-8") as f:
        f.write(out)
    ast.parse(out)                                     # 语法自检
    print(f"✅ 已迁移 {len(edits)} 处；备份于 {os.path.basename(APP)}.bak_md2html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
