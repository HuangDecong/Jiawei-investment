# -*- coding: utf-8 -*-
"""运行版本信息：**只读 `version.json`，运行时不调用 git**。

为什么不在运行时调 git
----------------------
线上容器是「代码目录 + 无 .git」的形态（Streamlit Cloud 的部署包不带仓库元数据），
`subprocess.run(["git", ...])` 会抛 FileNotFoundError 或返回空串，
于是页脚永远是 "dev"——那就等于没有版本。

版本戳怎么来
------------
`tools/make_version.py` 在**发布时**把 git 短 sha、构建时间、依赖版本写进仓库根的
`version.json`，随代码一起提交。运行时只做一次 json 读取（失败就返回 unknown），
零 I/O 依赖、零 git 依赖。

一个已知的「一格偏差」
----------------------
git 的 commit sha 由内容决定，所以**文件里不可能写下包含它自己的那个 commit 的 sha**。
约定：`source_sha` 记的是「生成版本戳时所在的 HEAD」，而这次生成会单独提交一个
stamp commit。因此在干净树上恒有 `source_sha == HEAD^`。
`tools/deploy_check.py` 按 `{HEAD, HEAD^}` 两个候选比对，不要求字面相等。
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "version.json")

_FALLBACK = {
    "app_version": "dev",
    "source_sha": "unknown",
    "build_time": "unknown",
    "branch": "unknown",
    "requirements_sha8": "",
    "caliber_version": "",
    "packages": {},
}


@lru_cache(maxsize=1)
def info() -> dict:
    """读版本戳；文件缺失/损坏时返回占位值，绝不抛异常。"""
    try:
        with open(PATH, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:  # noqa: BLE001
        return dict(_FALLBACK)
    out = dict(_FALLBACK)
    out.update({k: v for k, v in d.items() if v is not None})
    if not isinstance(out.get("packages"), dict):
        out["packages"] = {}
    return out


def sha7() -> str:
    return str(info().get("source_sha") or "unknown")[:7]


def deps_text() -> str:
    """关键依赖版本：季报页真正会缺的三个包 + 计算栈。"""
    pkg = info().get("packages") or {}
    keys = ("akshare", "lxml", "plotly", "pandas", "streamlit")
    return " · ".join(f"{k} {pkg[k]}" for k in keys if k in pkg)


def footer_text(full: bool = False) -> str:
    """一行版式：版本 · sha · 构建时间 · 关键依赖（给页脚/日志用）。"""
    d = info()
    pkg = d.get("packages") or {}
    keys = ("akshare", "lxml", "plotly") if not full else tuple(pkg)
    dep = " / ".join(f"{k} {pkg.get(k, '—')}" for k in keys if k in pkg)
    bits = [f"v{d.get('app_version', 'dev')}", sha7(),
            str(d.get("build_time", ""))[:16], dep]
    return " · ".join([b for b in bits if b and b != "unknown"])


def as_dict() -> dict:
    """给页面「复制诊断信息」用：整份版本戳（不含任何密钥）。"""
    return dict(info())
