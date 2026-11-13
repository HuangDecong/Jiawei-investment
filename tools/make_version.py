# -*- coding: utf-8 -*-
"""生成 `version.json`（发布戳）。**只在发布时运行，应用运行时不调用 git。**

写进去的东西
------------
· app_version       —— app.py 里的 APP_VERSION（人可读）
· source_sha        —— 生成时所在的 HEAD 短 sha（7 位）
· branch            —— 当前分支
· build_time        —— 本地时间（+08:00），精确到分钟
· requirements_sha8 —— requirements.txt 的 sha1 前 8 位：依赖变了但没重建，
                        这个值会和线上不一致，`deploy_check` 据此提示
                        「Clear cache & reboot」
· caliber_version   —— 口径表版本（src/caliber.py 的 CALIBER_VERSION），
                        口径改动必须让它变，否则表格与文档会各说各话
· packages          —— 关键运行依赖的实际版本（不是 requirements 里写的，
                        而是本机 import 到的；线上装错版本一眼可见）

用法
----
    python tools/make_version.py          # 写 version.json
    python tools/make_version.py --check  # 只检查是否与当前 HEAD 一致，不写

约定：生成后**单独提交这一次 stamp**，于是干净树上恒有 `source_sha == HEAD^`；
`tools/deploy_check.py` 按 {HEAD, HEAD^} 两个候选比对。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "version.json")
REQ = os.path.join(ROOT, "requirements.txt")
APP = os.path.join(ROOT, "app.py")
sys.path.insert(0, os.path.join(ROOT, "src"))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       encoding="utf-8").strip()
    except Exception:  # noqa: BLE001
        return ""


def _app_version() -> str:
    try:
        src = open(APP, encoding="utf-8").read()
        m = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', src)
        return m.group(1) if m else "dev"
    except Exception:  # noqa: BLE001
        return "dev"


def _req_sha8() -> str:
    try:
        with open(REQ, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:8]
    except Exception:  # noqa: BLE001
        return ""


def _pkg(name: str) -> str:
    try:
        return md.version(name)
    except Exception:  # noqa: BLE001
        return "missing"


def _caliber_version() -> str:
    try:
        import caliber  # noqa: PLC0415
        return str(getattr(caliber, "CALIBER_VERSION", ""))
    except Exception:  # noqa: BLE001
        return ""


def build() -> dict:
    return {
        "app_version": _app_version(),
        "source_sha": _git("rev-parse", "--short=7", "HEAD") or "unknown",
        # 本机分支叫 master，但远端发布分支是 main（API 推送）。
        # 版本戳要写「线上跑的分支」，所以允许用 FCD_BRANCH 覆盖。
        "branch": (os.environ.get("FCD_BRANCH")
                   or _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"),
        "build_time": dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z"),
        "requirements_sha8": _req_sha8(),
        "caliber_version": _caliber_version(),
        "packages": {k: _pkg(k) for k in
                     ("streamlit", "pandas", "numpy", "plotly", "akshare",
                      "lxml", "requests", "scipy")},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只比对，不写文件")
    args = ap.parse_args()
    new = build()
    if args.check:
        try:
            old = json.load(open(OUT, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            print("❌ 没有 version.json，先运行 python tools/make_version.py")
            return 1
        diff = {k: (old.get(k), new.get(k)) for k in new
                if k != "build_time" and old.get(k) != new.get(k)}
        if diff:
            print("❌ version.json 与当前工作区不一致：")
            for k, (a, b) in diff.items():
                print(f"   {k}: 文件 {a} → 现在 {b}")
            print("   运行 python tools/make_version.py 后重新提交")
            return 1
        print(f"✅ 版本戳一致：{new['app_version']} · {new['source_sha']}")
        return 0
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(new, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"✅ 已写 version.json：{new['app_version']} · {new['source_sha']} · "
          f"{new['build_time']}")
    print("   下一步：git add version.json && git commit -m 'chore: 版本戳'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
