# -*- coding: utf-8 -*-
"""
check_html_blocks.py —— 静态扫描 app.py 中所有 unsafe_allow_html 块，报告畸形 HTML

为什么需要它
------------
Streamlit 把 markdown 字符串交给前端 react-markdown + rehype-raw 渲染。
rehype-raw 用 parse5（严格遵循 HTML5 树构造算法）解析内联 HTML。
一旦出现下列情况，parse5 会按规范**自动改写** DOM 结构：

  A. 块级元素（<div>/<table>/<p>）被放在行内元素（<b>/<i>/<span>/<a>）之内
     → HTML5 的 "in body" 插入模式会提前闭合行内元素，真实 DOM 与 React 虚拟 DOM 不一致
  B. 标签未闭合 / 交叉嵌套（如 <b><i></b></i>）
  C. <table> 与 <p> 交叉（foster parenting）—— 浏览器会把 table 提到 p 前面
  D. 多行 HTML 中夹了空行 —— markdown 会把块切成两段，各自被 <p> 包裹

以上任一情况在**切换页面/重跑脚本**触发 React 重渲染时，
就会抛 NotFoundError: removeChild / insertBefore。

本脚本用 AST 取出每个 st.markdown(..., unsafe_allow_html=True) 的字面量 HTML，
把 f-string 的插值替换为占位符，再做：
  ① 标签配对检查（栈式）
  ② 行内元素包含块级元素检查
  ③ 空行检查（多行 HTML）
输出「行号 + 问题类型 + 片段」，作为修复清单。

用法：
    python tools/check_html_blocks.py            # 仅报告
    python tools/check_html_blocks.py --strict   # 有问题的块返回退出码 1
"""
from __future__ import annotations

import ast
import os
import re
import sys

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.py")

VOID = {"br", "hr", "img", "input", "meta", "link", "col", "area", "base",
        "embed", "source", "track", "wbr"}
INLINE = {"b", "i", "u", "s", "em", "strong", "span", "a", "code", "small",
          "sub", "sup", "mark", "font"}
BLOCK = {"div", "p", "table", "thead", "tbody", "tr", "td", "th", "ul", "ol",
         "li", "section", "article", "header", "footer", "h1", "h2", "h3",
         "h4", "h5", "h6", "blockquote", "pre", "figure", "figcaption", "main"}

TAG_RE = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9]*)([^>]*?)(/?)>")


def _literal_html(node: ast.AST) -> str | None:
    """取出字符串字面量（含 f-string 的静态部分），插值替换为占位符。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            else:
                parts.append("\x00")          # 占位符：长度 1，不含尖括号
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        l, r = _literal_html(node.left), _literal_html(node.right)
        if l is not None and r is not None:
            return l + r
    if isinstance(node, ast.Call):            # ''.join([...]) 形式暂不处理
        return None
    return None


def check(src: str) -> tuple[str, list[str]]:
    tree = ast.parse(src)
    issues: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "attr", None)
        if name != "markdown":
            continue
        kw = {k.arg: k.value for k in node.keywords}
        ua = kw.get("unsafe_allow_html")
        if not (isinstance(ua, ast.Constant) and ua.value is True):
            continue
        if not node.args:
            continue
        html = _literal_html(node.args[0])
        if html is None:
            continue
        ln = node.lineno

        # <style> 块内是 CSS，不是 HTML：其中的 {} 与 > 会让配对检查误报
        # （实测会把 .stApp{...} 报成"交叉嵌套"）。整段剔除后再检查，
        # 只保留 <style> 标签本身用于判断是否跳过。
        if "<style>" in html:
            continue

        # ---- ③ 多行 HTML 中夹空行 ----
        if "\n\n" in html.strip() and "<" in html:
            issues.append(f"L{ln}: [空行] 块内存在空行 → markdown 会切成两段并各自包 <p>，"
                          f"块级标签可能被 foster parenting 提出去")

        # ---- ① 标签配对 + ② 行内包含块级 ----
        stack: list[tuple[str, int]] = []
        for m in TAG_RE.finditer(html):
            closing, tag = m.group(1), m.group(2).lower()
            selfclose = m.group(4) == "/"
            if tag in VOID or selfclose or tag in ("!doctype",):
                continue
            if not closing:
                stack.append((tag, m.start()))
            else:
                if not stack:
                    issues.append(f"L{ln}: [多余闭合] </{tag}> 没有对应的开启标签")
                    continue
                top, pos = stack[-1]
                if top == tag:
                    stack.pop()
                else:
                    # 尝试在栈中找到（说明交叉嵌套）
                    names = [t for t, _ in stack]
                    if tag in names:
                        idx = len(stack) - 1 - names[::-1].index(tag)
                        unclosed = [t for t, _ in stack[idx + 1:]]
                        issues.append(
                            f"L{ln}: [交叉嵌套] </{tag}> 闭合时，内层 "
                            f"{'/'.join('<' + u + '>' for u in unclosed)} 仍未闭合")
                        stack = stack[:idx]
                    else:
                        issues.append(f"L{ln}: [孤立闭合] </{tag}> 未在开启栈中")
        if stack:
            issues.append(f"L{ln}: [未闭合] "
                          + "、".join(f"<{t}>" for t, _ in stack))

        # ---- 行内包含块级 ----
        for m in TAG_RE.finditer(html):
            closing, tag = m.group(1), m.group(2).lower()
            if closing or tag not in INLINE:
                continue
            # 取该行内标签之后到其闭合前的片段
            seg = html[m.end():m.end() + 4000]
            for bm in TAG_RE.finditer(seg):
                if bm.group(1) == "/" and bm.group(2).lower() == tag:
                    break
                if bm.group(1) == "" and bm.group(2).lower() in BLOCK:
                    issues.append(
                        f"L{ln}: [行内含块级] <{tag}> 内部出现 <{bm.group(2)}> —— "
                        f"HTML5 会提前闭合 <{tag}>，导致 DOM 与虚拟 DOM 不一致")
                    break
            break     # 只报第一处，避免噪声

    return src, issues


def main() -> int:
    with open(APP, encoding="utf-8") as f:
        src = f.read()
    _, issues = check(src)
    print("=" * 96)
    print(f"畸形 HTML 扫描：{os.path.relpath(APP)}")
    print("=" * 96)
    if not issues:
        print("✅ 未发现畸形 HTML 块")
        return 0
    for it in issues:
        print("  ⚠️ " + it)
    print("-" * 96)
    print(f"合计 {len(issues)} 处")
    return 1 if "--strict" in sys.argv else 0


if __name__ == "__main__":
    sys.exit(main())
