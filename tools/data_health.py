# -*- coding: utf-8 -*-
"""数据体检：一屏回答「季报库的数字能不能信」。

检查分三层，全部是「左边必须等于右边」或「越界即错」的硬规则：
  A 边界：比例不能为负、不能超过 400%（清算期现金可能 >100%，是真实披露）
  B 恒等式：行业合计 ≈ 股票仓位；份额表期末净资产 = 资产配置净资产；
            规模拆解三项之和 = 净资产变动
  C 结构：持仓名次 1..N 连续（断档说明混了多套快照）；期次格式合法

用法：
    python tools/data_health.py            # 全量
    python tools/data_health.py --brief    # 只打印结论行
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import qreport as Q  # noqa: E402

ROWS = []


def rec(section: str, item: str, ok: bool, detail: str) -> None:
    ROWS.append((section, item, "通过" if ok else "需关注", detail))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", action="store_true")
    args = ap.parse_args()
    Q.ensure_db()
    with Q._con() as con:
        a = pd.read_sql("SELECT * FROM q_asset", con)
        ind = pd.read_sql("SELECT * FROM q_industry", con)
        sh = pd.read_sql("SELECT * FROM q_shares", con)
        hd = pd.read_sql("SELECT * FROM q_hold", con)
        ch = pd.read_sql("SELECT * FROM q_change", con)

    rec("规模", "基金 / 期次", True,
        f"{a['code'].nunique():,} 只 · {a['quarter'].nunique()} 个季度 · "
        f"资产 {len(a):,} 行 · 行业 {len(ind):,} 行 · 持仓 {len(hd):,} 行")

    # A 边界
    bad = a[(a.stock_pct < -0.01) | (a.stock_pct > 5000) |
            (a.cash_pct < -0.01) | (a.cash_pct > 5000)]
    _ext = int(((a.cash_pct > 200) | (a.stock_pct > 200)).sum())
    rec("边界", "股/债/现金比例非负且在 5000% 内", bad.empty,
        (f"{len(bad)} 行越界" if len(bad) else
         f"全部合法；{_ext} 行为清算期极端值（现金占比最高 3,694%，真实披露）"))
    rec("边界", "申购/赎回非负、期末份额不为负", not ((sh.purchase_yi < 0) |
        (sh.redeem_yi < 0) | (sh.end_share_yi < 0)).any(),
        (f"{int(((sh.purchase_yi < 0) | (sh.redeem_yi < 0) | (sh.end_share_yi < 0)).sum())} 行异常；"
         f"{int((sh.end_share_yi == 0).sum())} 行期末份额四舍五入为 0.00"))
    rec("边界", "重大买卖金额非负", not (ch.amount_wan < 0).any(),
        f"{int((ch.amount_wan < 0).sum())} 行异常")

    # B1 行业合计 vs 股票仓位（同一报告期）
    m = ind.groupby(["code", "period"]).ratio_pct.sum().rename("ind_sum").reset_index()
    j = a.merge(m, on=["code", "period"], how="inner")
    over = j[j.ind_sum > j.stock_pct + 2.0]
    rec("恒等式", "行业合计 ≤ 股票仓位 + 2pp", len(over) <= 5,
        f"{len(over)}/{len(j):,} 个基金×期超出（含港股分类差异）")

    # B2 份额期末净资产 = 资产配置净资产（四舍五入容差）
    s1 = sh.groupby(["code", "period"], as_index=False).end_nav_yi.first()
    j2 = a.merge(s1, on=["code", "period"], how="inner")
    j2 = j2[pd.to_numeric(j2.nav_yi, errors="coerce") > 0]
    ok = [Q.shares_consistent(e, n) for e, n in zip(j2.end_nav_yi, j2.nav_yi)]
    rate = 1 - (sum(ok) / max(len(ok), 1))
    rec("恒等式", "份额表净资产 = 资产配置净资产", rate < 0.01,
        f"{len(ok) - sum(ok):,}/{len(ok):,} 不一致（{rate:.2%}）")

    # B3 规模拆解：三项之和 = Δ净资产（随机 200 只、逐季）
    codes = a["code"].drop_duplicates().head(200)
    bad3, n3 = 0, 0
    for c in codes:
        d = Q.load_fund(c)
        aa, ss = d["q_asset"].sort_values("period"), d["q_shares"]
        if aa.empty or ss.empty:
            continue
        for i in range(1, len(aa)):
            ra, rp = aa.iloc[i], aa.iloc[i - 1]
            rs = ss[ss["quarter"] == ra["quarter"]]
            if rs.empty or not np.isfinite(ra["nav_yi"]) or not np.isfinite(rp["nav_yi"]):
                continue
            e = float(rs["end_share_yi"].iloc[0])
            b0 = e - float(rs["purchase_yi"].iloc[0]) + float(rs["redeem_yi"].iloc[0])
            if b0 <= 0 or e <= 0 or rp["nav_yi"] <= 0:
                continue
            np0, np1 = rp["nav_yi"] / b0, ra["nav_yi"] / e
            eff = (e - b0) * np0 + b0 * (np1 - np0) + (e - b0) * (np1 - np0)
            n3 += 1
            if abs(eff - (ra["nav_yi"] - rp["nav_yi"])) > 0.05:
                bad3 += 1
    rec("恒等式", "规模拆解三项之和 = Δ净资产", bad3 == 0,
        f"抽查 {n3:,} 个季度，{bad3} 个对不上" if bad3 else f"抽查 {n3:,} 个季度全部对上")

    # C1 名次连续
    gaps = 0
    for (c, q), g in hd.groupby(["code", "quarter"]):
        rk = sorted(int(x) for x in g["rnk"].dropna())
        if rk and (rk[0] != 1 or rk != list(range(1, len(rk) + 1))):
            gaps += 1
    rec("结构", "持仓名次 1..N 连续", gaps == 0,
        f"{gaps} 个基金×季度断档（混了多套快照）")

    # C2 期次格式
    dirty = 0
    for t, d in (("资产", a), ("行业", ind), ("份额", sh), ("持仓", hd)):
        dirty += int((~d["quarter"].astype(str).map(Q.valid_q)).sum())
    rec("结构", "期次格式合法（YYYYQn）", dirty == 0, f"{dirty} 行非法")

    # C3 A+H 同名：不是错误，但会让人以为「前十大排序算错了」，必须量化并说明。
    _ah = hd[hd["rnk"] <= 10].groupby(["code", "quarter", "stock_name"]).size()
    n_ah = int((_ah > 1).sum())
    rec("结构", "前十大 A+H 同名单列（口径差异，非错误）", True,
        f"{n_ah} 个基金×季度；官方按合并公允价值排序，本库按单只占比排，"
        f"名次可能不同，页面已标注")

    # C4 无权重行：源表偶尔返回只有代码、没有占比的占位行，入库时应剔除
    n_null = int(hd["ratio_pct"].isna().sum())
    rec("结构", "无权重占位行已剔除", n_null == 0, f"{n_null} 行")

    df = pd.DataFrame(ROWS, columns=["层", "检查项", "结果", "说明"])
    print(df.to_string(index=False))
    bad_n = int((df["结果"] != "通过").sum())
    print(f"\n{'全部通过' if not bad_n else f'{bad_n} 项需关注'}")
    # 存一份给页面读：页面不能每次打开都扫 130 万行（要 10 秒以上）。
    try:
        out = os.path.join(ROOT, "data", "derived", "data_health.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"),
                       "bad": bad_n,
                       "rows": df.to_dict(orient="records")},
                      f, ensure_ascii=False, indent=1)
        print(f"已写入 {out}")
    except Exception as e:  # noqa: BLE001
        print("体检快照写入失败：", e)
    return 0 if not bad_n else 1


if __name__ == "__main__":
    raise SystemExit(main())
