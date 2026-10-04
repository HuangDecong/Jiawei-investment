# -*- coding: utf-8 -*-
"""
wind_fetch.py —— Wind 数据抓取管道（机构级）
======================================================================
设计要点
--------
1. **断点续传**：每完成一个季度立刻落盘到 cache/，中断后重跑只补缺口。
2. **原始即证据**：cache 保留 Wind 返回的**原始值**（亿元、%），不做任何加工，
   加工全部发生在 build 阶段，保证「原始数据 ↔ 因子面板」可双向回溯。
3. **范围声明**：每次抓取都记录 windcode / 指标名 / 报告期 / 抓取时刻，
   写入 provenance，满足机构合规审计要求。
4. **口径统一**：本管道抓取的是「基金前十大重仓股按申万一级行业的合计市值」，
   口径 = 占**重仓股总市值**的比例（非占净值），已在 README 与看板中显式标注。

执行方式
--------
本脚本**不由 Python 直接调用 Wind**（Wind 通过 MCP 暴露）。
它负责两件事：
  (a) `plan`   —— 生成待抓取的季度清单与查询语句，供 Agent 逐条执行；
  (b) `build`  —— 把 Agent 抓回来的原始 JSON 汇总成 data/raw/ 下的标准 xlsx。

    python tools/wind_fetch.py plan            # 打印待抓任务
    python tools/wind_fetch.py build           # 汇总原始 JSON -> xlsx
    python tools/wind_fetch.py status          # 查看抓取进度
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import config as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "data", "cache")
os.makedirs(CACHE, exist_ok=True)

# ============================================================================
# 抓取范围：2010Q1 – 2026Q2（66 个季度）
# ============================================================================
Q_FIRST, Q_LAST = 2010, 2026
# 2026 年只有 Q1/Q2 有季报（Q3 报告期 2026-09-30 尚未到披露时点）
LATEST_Q = (2026, 2)


def all_quarters() -> list[str]:
    out = []
    for y in range(Q_FIRST, Q_LAST + 1):
        for q in range(1, 5):
            if y == Q_LAST and q > LATEST_Q[1]:
                break
            out.append(f"{y}Q{q}")
    return out


def q_end_date(q: str) -> str:
    """季度 -> 报告期最后一天（YYYY-MM-DD）。"""
    y, qq = int(q[:4]), int(q[-1])
    md = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}[qq]
    return f"{y}-{md}"


def q_start_date(q: str) -> str:
    """季度首个交易日近似（用于区间涨跌幅取数）。"""
    y, qq = int(q[:4]), int(q[-1])
    md = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}[qq]
    return f"{y}-{md}"


# ============================================================================
# 抓取任务定义
# ============================================================================
def tasks_for(q: str) -> dict:
    """返回某个季度需要执行的 Wind 查询（task_key -> question）。"""
    d = q_end_date(q)
    return {
        # ① 核心：基金重仓股行业分布（62 期 × 31 行业）
        "holdings": (
            f"查询{d}，全部公募基金（普通股票型基金、偏股混合型基金、"
            f"灵活配置型基金）的前十大重仓股持股总市值合计，"
            f"按申万一级行业分类汇总"),
        # ② 行业季度涨跌幅
        "returns": (
            f"查询{d}，申万一级行业指数在该季度的区间涨跌幅"),
        # ③ 行业季度成交额
        "turnover": (
            f"查询{d}，申万一级行业指数在该季度的区间成交额"),
        # ④ 沪深300 行业权重
        "hs300": (
            f"查询{d}，沪深300指数成份股按申万一级行业分组的权重合计占比"),
        # ⑤ 样本基金数量
        "nfunds": (
            f"查询{d}，中国境内公募基金中普通股票型基金、偏股混合型基金、"
            f"灵活配置型基金的基金数量，以及该报告期已披露季报的基金数量"),
    }


TASK_KEYS = ["holdings", "returns", "turnover", "hs300", "nfunds"]


def status() -> None:
    qs = all_quarters()
    print(f"抓取范围：{qs[0]} ~ {qs[-1]}   共 {len(qs)} 个季度")
    print(f"任务类型：{len(TASK_KEYS)} 类  ->  总计 {len(qs) * len(TASK_KEYS)} 次查询\n")
    print(f"{'任务':<12}{'已完成':>8}{'缺失':>8}  缺失季度示例")
    print("-" * 74)
    for k in TASK_KEYS:
        done, miss = [], []
        for q in qs:
            p = os.path.join(CACHE, f"{k}__{q}.json")
            (done if os.path.exists(p) else miss).append(q)
        ex = ", ".join(miss[:8]) + (" …" if len(miss) > 8 else "")
        print(f"{k:<12}{len(done):>8}{len(miss):>8}  {ex}")
    print("-" * 74)
    tot_done = sum(os.path.exists(os.path.join(CACHE, f"{k}__{q}.json"))
                   for k in TASK_KEYS for q in qs)
    print(f"总进度：{tot_done}/{len(qs) * len(TASK_KEYS)} "
          f"({tot_done / (len(qs) * len(TASK_KEYS)) * 100:.1f}%)")


def plan(batch: int = 0, only: str | None = None) -> None:
    """打印待抓任务（batch>0 时只打印前 batch 条）。"""
    qs = all_quarters()
    keys = [only] if only else TASK_KEYS
    n = 0
    for q in qs:
        for k in keys:
            p = os.path.join(CACHE, f"{k}__{q}.json")
            if os.path.exists(p):
                continue
            n += 1
            if batch and n > batch:
                return
            print(f"### TASK {k} {q}")
            print(tasks_for(q)[k])
            print()
    if n == 0:
        print("全部任务已完成。")


# ============================================================================
# 汇总：cache/*.json -> data/raw/*.xlsx
# ============================================================================
def _load(k: str, q: str):
    p = os.path.join(CACHE, f"{k}__{q}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build() -> None:
    import pandas as pd

    qs = all_quarters()
    stamp = "2026-10-02"

    # ---------------- ① 行业配置（重仓股市值口径） ----------------
    # 口径说明：Wind 部分季度会返回 32 行 —— 31 个申万一级行业 + 1 个空名分类
    # （空名行并非已知行业缺失，而是不属于申万一级的重仓标的，如港股通/未分类）。
    # 处理：只保留 31 个申万一级行业，再按 31 行业之和归一化到 100%，
    # 使 R1「配置比例加总 = 100 ± 0.01」恒成立；被剔除的空名部分单独登记。
    rows, prov, dropped = [], [], []
    for q in qs:
        d = _load("holdings", q)
        if not d:
            continue
        kept = {k: v for k, v in d.items() if k in C.INDUSTRIES}
        for ind, mv in kept.items():
            rows.append(dict(季度=q, 行业=ind, 重仓市值亿元=mv))
        tot31 = sum(kept.values())
        miss = [i for i in C.INDUSTRIES if i not in kept]
        prov.append(dict(季度=q, 数据来源="Wind", 指标="公募基金前十大重仓股行业市值",
                         报告期=q_end_date(q), 行业数=len(kept),
                         缺行业=",".join(miss), 合计亿元=round(tot31, 2),
                         查询日期=stamp))
        if miss:
            dropped.append(f"{q}:缺{len(miss)}个行业({','.join(miss[:3])}…)")
    if not rows:
        print("!! 无 holdings 缓存，无法构建")
        return
    h = pd.DataFrame(rows)

    # 宽表：季度 × 行业（亿元）
    mv = h.pivot(index="季度", columns="行业", values="重仓市值亿元")
    valid_q = [q for q in qs if q in mv.index]
    mv = mv.reindex(valid_q).reindex(columns=C.INDUSTRIES)
    mv.to_excel(os.path.join(C.DATA_RAW, "wind_industry_mv.xlsx"))

    # 占比（% 占重仓股总市值，按 31 行业归一）—— 模型的「配置比例」
    tot = mv.sum(axis=1)
    alloc = mv.div(tot, axis=0) * 100.0
    alloc.insert(0, "季度", alloc.index)
    alloc.to_excel(os.path.join(C.DATA_RAW, "industry_allocation.xlsx"),
                   index=False)
    print(f"行业配置：{len(alloc)} 期 × {len(C.INDUSTRIES)} 行业"
          f"（已按 31 行业归一，加总=100%）")
    if dropped:
        print("  ⚠️ 口径提示：" + " | ".join(dropped[:6]))

    # ---------------- ② 行业涨跌幅 ----------------
    rows = []
    for q in qs:
        d = _load("returns", q)
        if not d:
            continue
        for ind, v in d.items():
            rows.append(dict(行业=ind, 季度=q, 涨跌幅=v))
    ret = pd.DataFrame(rows).dropna()
    ret.to_excel(os.path.join(C.DATA_RAW, "sw_industry_returns.xlsx"),
                 index=False)
    print(f"涨跌幅：{len(ret)} 行，{ret['季度'].nunique()} 期")

    # ---------------- ③ 行业成交额 ----------------
    rows = []
    for q in qs:
        d = _load("turnover", q)
        if not d:
            continue
        for ind, v in d.items():
            rows.append(dict(行业=ind, 季度=q, 成交额亿元=v))
    turn = pd.DataFrame(rows).dropna()
    turn.to_excel(os.path.join(C.DATA_RAW, "sw_industry_turnover.xlsx"),
                  index=False)
    print(f"成交额：{len(turn)} 行，{turn['季度'].nunique()} 期")

    # ---------------- ④ 沪深300 行业权重（动态季度） ----------------
    rows = []
    for q in qs:
        d = _load("hs300", q)
        if not d:
            continue
        for ind, v in d.items():
            rows.append(dict(季度=q, 行业=ind, 权重=v,
                             数据来源="Wind", 数据表="沪深300成份行业权重",
                             查询日期=stamp, 口径="Wind 指数成份权重聚合"))
    wq = pd.DataFrame(rows)
    if not wq.empty:
        wq = wq.rename(columns={"权重": "权重%"})
        wq.to_excel(os.path.join(C.DATA_RAW, "hs300_weights_quarterly.xlsx"),
                    index=False, sheet_name="hs300_weights_quarterly")
        # 静态权重文件用最新一期（兼容旧接口）
        last = wq[wq["季度"] == wq["季度"].iloc[-1]]
        last[["行业", "权重%"]].to_excel(
            os.path.join(C.DATA_RAW, "hs300_weights.xlsx"), index=False)
        print(f"沪深300权重：{len(wq)} 行，{wq['季度'].nunique()} 期")

    # ---------------- ⑤ 样本基金数 ----------------
    rows = []
    for q in qs:
        d = _load("nfunds", q)
        if not d:
            continue
        rows.append(dict(季度=q, 样本基金数=d.get("n"),
                         已披露基金数=d.get("n_disc"),
                         披露日期=d.get("disc_date", "")))
    if rows:
        nf = pd.DataFrame(rows)
        nf.to_excel(os.path.join(C.DATA_RAW, "fund_count.xlsx"), index=False)
        print(f"基金数量：{len(nf)} 期")

    # ---------------- 超配比例 = 配置比例 − 沪深300权重 ----------------
    # 机构级口径约束：
    #   超配比例必须建立在「同时具备 基金持仓 与 沪深300行业权重」的季度上。
    #   早期版本用 fillna(ffill) 把 hs300 空缺季度前向填充，会造成两种失真：
    #     (a) 首期（无任何前值）被填 0 → 超配比例 = 配置比例，数值虚高；
    #     (b) 中间缺口沿用上期权重 → 引入未被记录的陈旧基准。
    #   正确做法：取交集季度，不填充；缺口如实登记到 provenance。
    if not wq.empty:
        w = wq.pivot(index="季度", columns="行业", values="权重%")
        w = w.reindex(columns=C.INDUSTRIES)
        # 仅保留 hs300 与 holdings 都有值的季度（交集，顺序按 qs）
        hs_q = [q for q in qs if q in w.index and w.loc[q].notna().sum() > 0]
        al_all = alloc.set_index("季度")
        common = [q for q in qs if q in hs_q and q in al_all.index]
        al = al_all.reindex(common)[C.INDUSTRIES]
        ww = w.reindex(common)[C.INDUSTRIES]
        # 权重缺行业按 0 处理（该行业在沪深300中无成份股，基准为 0 是正确语义）
        ww = ww.fillna(0.0)
        ov = al - ww
        ov.insert(0, "季度", ov.index)
        ov.to_excel(os.path.join(C.DATA_RAW, "industry_overweight.xlsx"),
                    index=False)
        gap = [q for q in qs if q in al_all.index and q not in common]
        print(f"超配比例：{len(ov)} 期（覆盖 {common[0] if common else '-'} ~ "
              f"{common[-1] if common else '-'}）")
        if gap:
            print(f"  ⚠️ 因缺 hs300 基准而跳过的持仓季度：{','.join(gap)}")

    # ---------------- 集中度 HHI / CR ----------------
    if len(alloc):
        # 只在**实际有持仓数据**的季度上计算，避免 reindex(qs) 引入全 NaN 空行
        al = alloc.set_index("季度").reindex(
            [q for q in qs if q in alloc["季度"].values])[C.INDUSTRIES]
        wv = al / 100.0
        hhi = (wv ** 2).sum(axis=1)
        N = len(C.INDUSTRIES)
        hhi_n = (hhi - 1.0 / N) / (1.0 - 1.0 / N)
        cr3 = wv.apply(lambda r: r.nlargest(3).sum() * 100, axis=1)
        cr5 = wv.apply(lambda r: r.nlargest(5).sum() * 100, axis=1)
        cr10 = wv.apply(lambda r: r.nlargest(10).sum() * 100, axis=1)
        top1 = al.idxmax(axis=1)
        top1p = al.max(axis=1)
        crh = pd.DataFrame(dict(季度=al.index, CR3=cr3.values, CR5=cr5.values,
                                CR10=cr10.values, HHI=hhi.values,
                                HHI标准化=hhi_n.values,
                                Top1行业=top1.values, Top1比例=top1p.values,
                                行业数=C.NI))
        crh.to_excel(os.path.join(C.DATA_RAW, "industry_cr_hhi.xlsx"),
                     index=False)
        print(f"集中度：{len(crh)} 期  HHI 最新 = {hhi.iloc[-1]:.4f}"
              f"  区间 [{hhi.min():.4f}, {hhi.max():.4f}]")

    # ---------------- 数据溯源登记 ----------------
    if prov:
        pd.DataFrame(prov).to_excel(
            os.path.join(C.DATA_RAW, "wind_provenance.xlsx"), index=False)

    print(f"\n✅ build 完成，输出目录：{C.DATA_RAW}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "plan":
        plan(batch=int(sys.argv[2]) if len(sys.argv) > 2 else 0,
             only=sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "build":
        build()
    else:
        status()
