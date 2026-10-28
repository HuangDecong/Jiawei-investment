"""并行跑基金档案抓取（分片 + 汇总），供后台一次性执行。

单独写一个 runner 而不是在 shell 里 `&`，是为了让进程留在任务管理里、
失败能拿到退出码，中途中断也能靠 `fund_profile` 表断点续跑。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
LOG = os.path.join(ROOT, "output")
NSHARDS = int(os.environ.get("NSHARDS", "4"))


def main() -> None:
    os.makedirs(LOG, exist_ok=True)
    t0 = time.time()
    procs = []
    for i in range(NSHARDS):
        fh = open(os.path.join(LOG, f"profile_{i}.log"), "a", encoding="utf-8")
        procs.append(subprocess.Popen(
            [PY, os.path.join(ROOT, "tools", "fetch_fund_profile.py"),
             "--shard", str(i), "--nshards", str(NSHARDS)],
            stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT))
    print(f"启动 {NSHARDS} 个分片，PID {[p.pid for p in procs]}", flush=True)
    for p in procs:
        p.wait()
    print(f"全部结束，耗时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    subprocess.run([PY, os.path.join(ROOT, "tools", "fetch_fund_profile.py"), "--report"],
                   cwd=ROOT)


if __name__ == "__main__":
    main()
