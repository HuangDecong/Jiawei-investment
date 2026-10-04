# -*- coding: utf-8 -*-
"""
scheduled_check.py  ——  季度定时数据质量网关
======================================================================
用途
----
给「每季度自动跑一次」的定时任务提供一个**单入口**：
    1. 运行 `src/update_pipeline.py`（一键重算全流程，含 Step 0 自动补动态权重）；
    2. 读取 `output/preflight_gate.csv`，检查 8 项硬约束是否有失败；
    3. 检查关键产物的新鲜度（mtime 是否为本次运行）；
    4. 生成一份**人类可读的状态报告** `output/scheduled_status.md`；
    5. 若硬约束失败或流水线异常 → 退出码 **2**（供调度器识别为"需告警"）；
       正常 → 退出码 **0**。

设计原则
--------
· **不吞异常**：任何失败都要在状态报告里留下完整 traceback 摘要。
· **不静默通过**：闸门文件缺失、行数异常等一律视为失败。
· **可重复执行**：同一季度重复运行不产生副作用（覆盖同名文件）。
· 不做任何"猜测性修复"——发现问题只报告，由人判断。

命令行
------
    python tools/scheduled_check.py            # 正常运行
    python tools/scheduled_check.py --quiet    # 只写报告，不打印全量流水线日志
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
OUT = os.path.join(ROOT, "output")

# 「每次运行都必须刷新」的产物——mtime 必须晚于本次运行开始时间
REQUIRED_ARTIFACTS = [
    "factor_panel.xlsx", "ic_report.xlsx", "signal_report.xlsx",
    "backtest_report.xlsx", "rigor_report.xlsx", "research_report.xlsx",
    "research3_report.xlsx", "strategy_report.xlsx", "fci_report.xlsx",
    "structure_analysis.xlsx", "preflight_gate.csv",
]

# 「一次性生成、只需存在」的产物——由初始化工具生成，流水线不覆盖。
# 若把它们也要求"本次更新"，会稳定产生假告警，反而削弱闸门的可信度。
ONE_TIME_ARTIFACTS = [
    "dynamic_weights_report.xlsx",   # tools/build_dynamic_weights.py（仅在缺失时生成）
    "data_provenance.csv",           # tools/seed_raw_data.py（一次性登记）
]


def _run_pipeline() -> tuple[int, str, float]:
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.join(SRC, "update_pipeline.py")],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
    )
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or ""), time.time() - t0


def _check_gate() -> tuple[bool, list[str], str]:
    """返回 (是否全部通过, 失败项说明, 摘要文本)。"""
    import pandas as pd
    p = os.path.join(OUT, "preflight_gate.csv")
    if not os.path.exists(p):
        return False, ["闸门文件缺失：output/preflight_gate.csv"], "未生成闸门报告"
    df = pd.read_csv(p)
    if "级别" not in df.columns or "结果" not in df.columns:
        return False, [f"闸门报告列异常：{list(df.columns)}"], "闸门报告结构异常"
    hard = df[df["级别"] == "硬"]
    fail = hard[hard["结果"] == "失败"]
    warn = df[df["结果"] == "告警"]
    msgs = [f"{r['编号']} {r['检查项']}：{r['详情']}" for _, r in fail.iterrows()]
    summ = (f"硬约束 {len(hard)} 项，失败 {len(fail)} 项；软约束告警 {len(warn)} 项")
    return len(fail) == 0, msgs, summ


def _check_artifacts(t_start: float) -> tuple[bool, list[str]]:
    missing, stale = [], []
    for f in REQUIRED_ARTIFACTS:
        p = os.path.join(OUT, f)
        if not os.path.exists(p):
            missing.append(f)
        elif os.path.getmtime(p) < t_start - 5:      # 容 5 秒时钟误差
            stale.append(f)
    for f in ONE_TIME_ARTIFACTS:
        if not os.path.exists(os.path.join(OUT, f)):
            missing.append(f)
    msgs = ([f"产物缺失（需运行对应初始化工具）：{f}" for f in missing] +
            [f"产物未在本次运行中更新：{f}" for f in stale])
    return (not msgs), msgs


def _read_key_conclusion() -> dict:
    """从 run_log 中抽取几个关键数字，供报告快速浏览。"""
    import re
    p = os.path.join(OUT, "run_log.txt")
    if not os.path.exists(p):
        return {}
    txt = open(p, encoding="utf-8", errors="ignore").read()
    out: dict = {}
    m = re.search(r"HHI=([\d.]+)（([^）]*)", txt)
    if m:
        out["HHI"] = f"{m.group(1)}（{m.group(2)}）"
    m = re.search(r"CR3=([\d.]+)%\s+CR5=([\d.]+)%\s+CR10=([\d.]+)%", txt)
    if m:
        out["CR3/CR5/CR10"] = f"{m.group(1)}% / {m.group(2)}% / {m.group(3)}%"
    m = re.search(r"第一大重仓行业：(\S+)\s+([\d.]+)%", txt)
    if m:
        out["第一大重仓行业"] = f"{m.group(1)} {m.group(2)}%"
    m = re.search(r"· 极度拥挤行业 \d+ 个：(.+)", txt)
    if m:
        out["极度拥挤行业"] = m.group(1).strip()
    m = re.search(r"· 因子分级裁决（4 项独立检验合成）：(.+)", txt)
    if m:
        out["因子分级裁决"] = m.group(1).strip()
    m = re.search(r"构造依据（F1 与 F2 的 IC 符号相反）最早可得的季度 t\* = \*\*(\S+)\*\*", txt)
    if m:
        out["F15 构造依据最早可得时点"] = m.group(1)
    m = re.search(r"→ \*\*结论：检验「挑设定的流程」而非单个因子时，样本外表现不显著\*\*"
                  r"（p = ([\d.]+)）", txt)
    if m:
        out["Walk-forward 样本外 p 值"] = m.group(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="不在控制台回放流水线全量日志")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"季度数据质量网关　启动于 {stamp}")
    print("=" * 70)

    t_start = time.time()
    code, log, elapsed = _run_pipeline()
    if not args.quiet:
        print(log[-4000:])
    else:
        print(log[-800:])

    os.makedirs(OUT, exist_ok=True)
    # 保留原始日志，便于人工复核
    with open(os.path.join(OUT, "scheduled_run_log.txt"), "w", encoding="utf-8") as f:
        f.write(log)

    pipe_ok = (code == 0)
    gate_ok, gate_msgs, gate_summ = _check_gate()
    art_ok, art_msgs = _check_artifacts(t_start)
    concl = _read_key_conclusion()

    hard_fail = not (pipe_ok and gate_ok and art_ok)
    lines = [
        "# 季度数据质量网关 · 状态报告",
        "",
        f"- **运行时间**：{stamp}",
        f"- **耗时**：{elapsed:.1f} 秒",
        f"- **流水线退出码**：{code}（{'成功' if pipe_ok else '失败'}）",
        f"- **数据质量闸门**：{gate_summ}　→　{'通过' if gate_ok else '**未通过**'}",
        f"- **产物完整性**："
        f"{'全部在本次运行中更新' if art_ok else f'**异常（{len(art_msgs)} 项）**'}",
        "",
        f"## 总体判定：{'✅ 正常' if not hard_fail else '❌ 需人工介入'}",
        "",
    ]
    if gate_msgs or art_msgs:
        lines += ["## 需要处理的问题", ""]
        lines += [f"- {m}" for m in (gate_msgs + art_msgs)]
        lines += [""]
    lines += ["## 本轮关键读数", ""]
    if concl:
        lines += [f"- {k}：{v}" for k, v in concl.items()]
    else:
        lines += ["- （未能从 run_log 抽取到关键读数）"]
    lines += [
        "",
        "## 产物清单（本次运行）", "",
        "| 文件 | 类型 | 大小KB | 更新时间 |", "|---|---|---|---|",
    ]
    for tag, group in (("每次更新", REQUIRED_ARTIFACTS),
                       ("一次性生成", ONE_TIME_ARTIFACTS)):
        for f in group:
            p = os.path.join(OUT, f)
            if os.path.exists(p):
                lines.append(f"| {f} | {tag} | {os.path.getsize(p) / 1024:.1f} | "
                             f"{datetime.fromtimestamp(os.path.getmtime(p)):%Y-%m-%d %H:%M} |")
            else:
                lines.append(f"| {f} | {tag} | — | **缺失** |")
    lines += [
        "",
        "## 说明",
        "",
        "- 本报告由 `tools/scheduled_check.py` 自动生成，**不做任何自动修复**。",
        "- 退出码 0 = 正常；退出码 2 = 需人工介入（调度器可据此告警）。",
        "- 数据更新方式：把新季度文件放入 `data/raw/` 后重新运行本脚本即可。",
        "",
    ]
    rp = os.path.join(OUT, "scheduled_status.md")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print()
    print("-" * 70)
    print(f"流水线：{'成功' if pipe_ok else '失败'}　|　闸门：{'通过' if gate_ok else '未通过'}　"
          f"|　产物：{'完整' if art_ok else '异常'}")
    for m in (gate_msgs + art_msgs):
        print(f"  ！{m}")
    print(f"状态报告：{rp}")
    print(f"总体判定：{'正常' if not hard_fail else '需人工介入'}")
    return 0 if not hard_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
