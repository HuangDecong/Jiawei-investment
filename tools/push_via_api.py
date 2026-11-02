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


def _files_of(rev_range: str) -> list[tuple[str, str]]:
    """取一个提交范围的变更文件并集（同名取最后一次的状态）。"""
    out = subprocess.check_output(
        ["git", "-c", "core.quotepath=false", "log", "--name-status",
         "--format=", rev_range],
        text=True, encoding="utf-8").strip().split("\n")
    seen: dict[str, str] = {}
    for row in out:
        parts = row.split("\t")
        if len(parts) < 2:
            continue
        seen[parts[-1]] = parts[0][0]
    return [(st, p) for p, st in seen.items()]


def main() -> None:
    # 不传参数时推「最近 3 个提交」的并集：
    # 只推 HEAD 会漏掉前一次提交——r32.0 就是这样只推了 shot6.py，
    # 线上 canon.py 还是旧的，整页报 AttributeError。
    if len(sys.argv) > 1:
        sha = sys.argv[1]
        files = _files_of(f"{sha}~1..{sha}")
    else:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      text=True).strip()
        files = _files_of("HEAD~3..HEAD")
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

    # 校验：远端每个文件的 blob 必须与本地一致。
    # 不做这一步，「推送成功」可能只是自我安慰（本次事故的根因）。
    tree_after = call("GET", f"/repos/{OWNER}/{REPO}/git/trees/{tree['sha']}?recursive=1")
    remote = {it["path"]: it["sha"] for it in tree_after.get("tree", [])
              if it.get("type") == "blob"}
    bad = []
    for status, path in files:
        if status == "D" or not os.path.exists(path):
            continue
        local = subprocess.check_output(["git", "hash-object", path],
                                        text=True).strip()
        if remote.get(path) != local:
            bad.append(path)
    if bad:
        print(f"❌ 校验失败：{len(bad)} 个文件远端与本地不一致 → " + "、".join(bad[:5]))
        raise SystemExit(1)
    print(f"✅ 校验通过：{len([1 for st, _ in files if st != 'D'])} 个文件与本地一致")


if __name__ == "__main__":
    main()
