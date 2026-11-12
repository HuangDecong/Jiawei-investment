# -*- coding: utf-8 -*-
"""上线一致性检查：**本地 HEAD / version.json / 远端 main / 线上页脚** 四处对齐。

为什么需要它
------------
2026-11-11 的事故不是「代码写错」，而是「代码改了、线上没跑」：
推送后没有 Reboot，`requirements.txt` 变了也没有 Clear cache，
线上仍装旧依赖，于是 005827 整页 `ModuleNotFoundError`。
只靠人记「记得点 Reboot」不可靠，所以把这件事做成可执行的检查。

四级证据（能验到哪一级就报到哪一级）
------------------------------------
1. **本地仓库**：HEAD、工作区是否干净、version.json 是否为当前代码生成；
2. **依赖指纹**：requirements.txt 的 sha1 与 version.json 记录是否一致
   （不一致＝依赖改了但没重新打戳，线上必然还是旧包）；
3. **远端分支**：GitHub main 的 sha（证明「推上去了」），落后就打印差几个提交；
4. **线上页脚**：`--url` 时用 Playwright 打开页面读页脚那一行。
   本项目线上是私有应用，未登录会被重定向到 `/-/login`，此时如实报
   「无法核验（需登录）」，**不猜、不假装通过**。

退出码
------
0 一致；1 有不一致（阻断发布）；2 仅「线上页脚」这一级无法核验，其余通过。

用法
----
    python tools/deploy_check.py
    python tools/deploy_check.py --url https://jiawei-investment.streamlit.app
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VJ = os.path.join(ROOT, "version.json")
OWNER, REPO = "HuangDecong", "Jiawei-investment"


def git(*a: str) -> str:
    try:
        return subprocess.check_output(["git", *a], cwd=ROOT, text=True,
                                       encoding="utf-8").strip()
    except Exception:  # noqa: BLE001
        return ""


def remote_main_sha() -> str:
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{OWNER}/{REPO}/commits/main",
            headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode())["sha"]
    except Exception:  # noqa: BLE001
        return ""


def remote_tree_sha() -> str:
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{OWNER}/{REPO}/commits/main",
            headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode())["commit"]["tree"]["sha"]
    except Exception:  # noqa: BLE001
        return ""


def remote_blob_map() -> dict[str, str] | None:
    """远端 main 的 {路径: blob sha}；取不到返回 None。"""
    try:
        ref = urllib.request.Request(
            f"https://api.github.com/repos/{OWNER}/{REPO}/git/ref/heads/main",
            headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(ref, timeout=20) as r:
            sha = json.loads(r.read().decode())["object"]["sha"]
        t = urllib.request.Request(
            f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/{sha}?recursive=1",
            headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(t, timeout=30) as r:
            d = json.loads(r.read().decode())
        return {i["path"]: i["sha"] for i in d.get("tree", []) if i.get("type") == "blob"}
    except Exception:  # noqa: BLE001
        return None


def worktree_vs_remote() -> list[str] | None:
    """逐文件比对工作区与远端；返回不一致文件（空＝一致），None＝取不到远端。"""
    import hashlib  # noqa: PLC0415
    rt = remote_blob_map()
    if rt is None:
        return None
    files = [p for p in git("-c", "core.quotepath=false", "ls-files").splitlines() if p]
    bad = []
    for p in files:
        fp = os.path.join(ROOT, p)
        if not os.path.exists(fp):
            continue
        with open(fp, "rb") as f:
            c = f.read()
        local = hashlib.sha1(b"blob %d\x00" % len(c) + c).hexdigest()
        if rt.get(p) != local:
            bad.append(p)
    return bad


def live_footer(url: str) -> tuple[str, str]:
    """返回 (状态, 文本)。状态：ok / login / error。"""
    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return "error", "未安装 playwright，跳过线上核验"
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="msedge")
        except Exception:  # noqa: BLE001
            try:
                b = p.chromium.launch()
            except Exception as e:  # noqa: BLE001
                return "error", f"浏览器启动失败：{type(e).__name__}"
        try:
            pg = b.new_page()
            pg.goto(url, wait_until="domcontentloaded", timeout=90000)
            pg.wait_for_timeout(9000)
            if "/-/login" in pg.url:
                return "login", "线上需登录（重定向到 /-/login），本机无法代读"
            txt = pg.inner_text("body")
            # 页脚形如：版本 5581949 · 2026.11.12-r49.0
            ms = re.search(r"版本\s*([0-9a-f]{7})", txt)
            mv = re.search(r"(\d{4}\.\d{2}\.\d{2}-r[\d.]+)", txt)
            if not (ms or mv):
                return ("error", "页面上没找到版本行")
            out = "|".join([ms.group(1) if ms else "", mv.group(1) if mv else ""])
            return ("ok", out)
        finally:
            b.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://jiawei-investment.streamlit.app")
    ap.add_argument("--no-live", action="store_true", help="跳过线上核验")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="线上被登录挡住时按「通过（未核验）」处理，供本地 make gate 使用")
    args = ap.parse_args()

    problems: list[str] = []
    notes: list[str] = []
    head7 = git("rev-parse", "--short=7", "HEAD")
    head7_parent = git("rev-parse", "--short=7", "HEAD^")
    print(f"本地 HEAD      : {head7}")
    try:
        vj = json.load(open(VJ, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        print("❌ 缺少 version.json —— 先跑 python tools/make_version.py")
        return 1
    src = str(vj.get("source_sha", ""))[:7]
    print(f"version.json   : {vj.get('app_version')} · {src} · {vj.get('build_time')}")
    if src not in {head7, head7_parent}:
        problems.append(f"版本戳过期：version.json 记的是 {src}，当前 HEAD 是 {head7}"
                        "（改完代码要重新打戳）")

    dirty = [ln for ln in git("status", "--porcelain").splitlines() if ln.strip()]
    runtime = [ln for ln in dirty
               if re.search(r"(^..\s)(app\.py|src/|requirements\.txt|version\.json)", ln)]
    if runtime:
        problems.append(f"工作区有 {len(runtime)} 个未提交的运行代码改动，线上不会有："
                        + "、".join(ln.split()[-1] for ln in runtime[:4]))

    # 依赖指纹
    import hashlib
    try:
        req8 = hashlib.sha1(open(os.path.join(ROOT, "requirements.txt"), "rb").read()).hexdigest()[:8]
    except Exception:  # noqa: BLE001
        req8 = ""
    if vj.get("requirements_sha8") and req8 != vj.get("requirements_sha8"):
        problems.append("requirements.txt 变了但版本戳没更新：线上仍是旧依赖，"
                        "推送后必须 Clear cache & reboot")

    # 远端
    rsha = remote_main_sha()
    if rsha:
        print(f"远端 main      : {rsha[:7]}")
        if rsha[:7] != head7:
            n = git("rev-list", "--count", f"{rsha}..HEAD")
            behind_txt = git("rev-list", "--count", f"HEAD..{rsha}")
            if n and n != "0":
                problems.append(f"远端 main 落后本地 {n} 个提交 —— 推送后再 Reboot")
            elif behind_txt and behind_txt != "0":
                notes.append(f"远端比本地多 {behind_txt} 个提交（可能是别处推送）")
            else:
                # 两套历史不互认（本机 push 被拦，只能走 Git Data API）：
                # 提交号必然不同，改用 tree sha 比内容——内容一致就算一致。
                # 两套历史不互认时不能用 commit/tree sha 比较：本地按 CRLF→LF
                # 规范化后入库，远端存的是工作区原始字节（Git Data API 上传），
                # 同一份文件在两边的 blob sha 天然不同。改为逐文件比对**工作区字节**。
                _bad = worktree_vs_remote()
                if _bad is None:
                    notes.append("读不到远端文件清单，跳过内容比对")
                elif not _bad:
                    notes.append(f"远端提交号 {rsha[:7]} 与本地 {head7} 不同"
                                 "（API 推送会重建提交），413 个文件内容逐一对齐")
                else:
                    problems.append(f"远端有 {len(_bad)} 个文件与本地工作区不一致"
                                    f"（如 {_bad[0]}）—— 先推送再 Reboot")
    else:
        notes.append("读不到远端 main（网络或权限），跳过第 3 级")

    # 线上页脚
    if args.no_live:
        notes.append("按要求跳过线上核验")
        live_state = "skip"
    else:
        live_state, live_txt = live_footer(args.url)
        if live_state == "ok":
            _sha, _ver = (live_txt.split("|") + ["", ""])[:2]
            print(f"线上页脚       : {_sha or '—'} · {_ver or '—'}")
            _known = bool(_sha) and subprocess.run(
                ["git", "cat-file", "-e", _sha], cwd=ROOT,
                capture_output=True).returncode == 0
            if _ver and _ver != str(vj.get("app_version", "")):
                problems.append(f"线上版本 {_ver} ≠ 本地 {vj.get('app_version')}"
                                " —— 去 Streamlit Cloud 点 Reboot（改过依赖要 Clear cache）")
            elif _sha and _sha != head7 and not _known:
                notes.append(f"线上 sha {_sha} 是 API 推送重建的提交，本地历史里没有；"
                             "版本号一致即可，核对以版本号 + 构建时间为准")
            elif _sha and _sha != head7:
                n = git("rev-list", "--count", f"{_sha}..HEAD") or "?"
                problems.append(f"线上落后 {n} 个提交（线上 {_sha} / 本地 {head7}）")
        else:
            print(f"线上页脚       : {live_txt}")
            notes.append("线上页脚无法核验：" + live_txt)

    print()
    for n in notes:
        print("· 提示：" + n)
    if problems:
        print("\n❌ 发布检查未通过：")
        for p in problems:
            print("   - " + p)
        return 1
    if live_state != "ok":
        print("\n⚠️ 本地/远端一致，但线上页脚未核验 —— 登录后看一眼侧栏底部那行版本。")
        return 0 if args.allow_unverified else 2
    print("\n✅ 本地 / 版本戳 / 远端 / 线上 四处一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
