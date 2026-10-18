# -*- coding: utf-8 -*-
"""通过 GitHub Git Data API 推送本地提交（用于 git 协议被代理拦截的场景）。

用法：python tools/push_via_api.py <commit_sha>
凭证：从 ~/.git-credentials 读取（不在任何输出中打印）。
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import urllib.request

OWNER = "HuangDecong"
REPO = "Jiawei-investment"
BRANCH = "main"


def _token() -> str:
    p = os.path.join(os.path.expanduser("~"), ".git-credentials")
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "github.com" in line and "@" in line:
                # https://user:token@github.com
                return line.split("@")[0].rsplit(":", 1)[-1]
    raise SystemExit("未找到 GitHub 凭证")


TOKEN = _token()
API = "https://api.github.com"


def call(method: str, path: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(API + path, data=data, method=method)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main() -> None:
    sha = sys.argv[1] if len(sys.argv) > 1 else subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()

    changed = subprocess.check_output(
        ["git", "-c", "core.quotepath=false", "show", "--name-status",
         "--format=", sha],
        text=True, encoding="utf-8").strip().split("\n")
    files = []
    for row in changed:
        parts = row.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0][0], parts[-1]
        files.append((status, path))
    print(f"待推送 {len(files)} 个文件")

    # 1) 远端当前 main
    ref = call("GET", f"/repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}")
    base_sha = ref["object"]["sha"]
    base_commit = call("GET", f"/repos/{OWNER}/{REPO}/git/commits/{base_sha}")
    base_tree = base_commit["tree"]["sha"]
    print(f"远端 HEAD = {base_sha[:8]}")

    # 2) 逐个建 blob
    tree_items = []
    for status, path in files:
        if status == "D":
            tree_items.append(dict(path=path, mode="100644", type="blob", sha=None))
            continue
        with open(path, "rb") as f:
            content = f.read()
        blob = call("POST", f"/repos/{OWNER}/{REPO}/git/blobs", {
            "content": base64.b64encode(content).decode(),
            "encoding": "base64",
        })
        tree_items.append(dict(path=path, mode="100644", type="blob",
                               sha=blob["sha"]))
        print(f"  blob ok  {path}  {len(content)}B")

    # 3) 建 tree / commit / 更新 ref
    tree = call("POST", f"/repos/{OWNER}/{REPO}/git/trees", {
        "base_tree": base_tree, "tree": tree_items})
    msg = subprocess.check_output(
        ["git", "log", "-1", "--format=%s", sha], text=True,
        encoding="utf-8").strip()
    commit = call("POST", f"/repos/{OWNER}/{REPO}/git/commits", {
        "message": msg, "tree": tree["sha"], "parents": [base_sha]})
    call("PATCH", f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
         {"sha": commit["sha"], "force": False})
    print(f"✅ 已推送：{commit['sha'][:8]}  {msg[:40]}")


if __name__ == "__main__":
    main()
