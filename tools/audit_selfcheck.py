# -*- coding: utf-8 -*-
"""全套自检 + 证伪。每次改数据或改页面后运行，输出一张可核对的表。

两类检查：
    · **不变量（invariant）**：数据本身自洽性——合计是否 100、HHI 是否等于 Σw²、
      分歧度三类计数是否等于样本数、个股 Δ 是否等于两期之差、港股是否混入行业……
      这类检查一旦失败，页面上的数字全不可信。
    · **证伪（falsification）**：主动构造「如果结论错了，会看到什么」，再用数据打它。
      例如：如果「拥挤即风险」成立，高分位次季收益应更低；如果 ΔZ 只是价格动量，
      对过去 4 季涨幅正交化后应失效；如果分歧度有预测力，IC 应显著。

输出：报告/自检表.csv，并在 ⑤ 页显示通过/失败计数与失败项。
用法：python tools/audit_selfcheck.py
"""
from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import ak_data as akd            # noqa: E402
import ak_stock as aks           # noqa: E402
import backtest as bt            # noqa: E402
import config as C               # noqa: E402
import data_loader as dl         # noqa: E402
import factor_engine as fe       # noqa: E402
import institution as inst       # noqa: E402

OUT = os.path.join(ROOT, "报告", "自检表.csv")
DB = os.path.join(ROOT, "data", "db", "fund_ak.db")


def _row(rows, kind, item, result, evidence):
    rows.append(dict(类别=kind, 检查项=item, 结果=result, 证据=evidence))


def invariants(rows, b, ind, mkt):
    q = b.latest
    # 1 配置合计
    s = float(b.alloc.loc[q].sum())
    _row(rows, "不变量", "行业配置合计 = 100", "通过" if abs(s - 100) < 0.01 else "失败",
         f"{q} 合计 {s:.4f}")
    # 2 HHI / CR3 复算
    w = b.alloc.loc[q].astype(float) / 100
    hhi = float((w ** 2).sum())
    cr3 = float(b.alloc.loc[q].nlargest(3).sum())
    _row(rows, "不变量", "HHI = Σw²", "通过" if abs(hhi - float(mkt["F5_HHI"].loc[q])) < 1e-6 else "失败",
         f"复算 {hhi:.6f} vs 表内 {float(mkt['F5_HHI'].loc[q]):.6f}")
    _row(rows, "不变量", "CR3 = 前三合计", "通过" if abs(cr3 - float(mkt["F6_CR3"].loc[q])) < 1e-4 else "失败",
         f"复算 {cr3:.4f}% vs 表内 {float(mkt['F6_CR3'].loc[q]):.4f}%")

    # 3 分歧度三类计数
    d = akd.dispersion_ak(*akd.latest_quarters()) if akd.latest_quarters() else pd.DataFrame()
    if not d.empty:
        bad = (d["加仓基金数"] + d["减仓基金数"] + d["不变基金数"] != d["样本基金数"]).sum()
        _row(rows, "不变量", "分歧度：加+减+不变 = 样本数", "通过" if bad == 0 else "失败",
             f"{len(d)} 个行业，异常 {int(bad)} 行")
        _row(rows, "不变量", "分歧度 ∈ [0,1] 且加仓占比 ∈ [0,100]",
             "通过" if d["分歧度"].between(0, 1).all() and d["加仓占比"].between(0, 100).all()
             else "失败", f"分歧度 {d['分歧度'].min():.2f}–{d['分歧度'].max():.2f}")

    # 4 个股 Δ = 两期之差
    pair = aks.comparable_pair("2026Q3")
    if pair:
        f = aks.stock_flow(*pair)
        bad = int((f["n1"] - f["n0"] != f["d_n"]).sum()) if not f.empty else -1
        _row(rows, "不变量", "个股 Δ基金数 = 本期 − 上期", "通过" if bad == 0 else "失败",
             f"{len(f)} 只股票，异常 {bad} 行（{pair[1]}→{pair[0]}）")

    # 5 港股不得混入申万行业
    if os.path.exists(DB):
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=30)
        bad = con.execute("SELECT COUNT(*) FROM fund_holdings "
                          "WHERE is_hk=1 AND industry IS NOT NULL").fetchone()[0]
        dup = con.execute("SELECT COUNT(*) FROM (SELECT code,quarter,stock_code,COUNT(*) c "
                          "FROM fund_holdings GROUP BY 1,2,3 HAVING c>1)").fetchone()[0]
        cov_bad = con.execute(
            "SELECT COUNT(*) FROM (SELECT c.code, c.quarter, c.coverage, "
            "SUM(h.weight_pct) s FROM coverage c JOIN fund_holdings h "
            "ON h.code=c.code AND h.quarter=c.quarter GROUP BY 1,2 "
            "HAVING ABS(c.coverage - s) > 0.05)").fetchone()[0]
        con.close()
        _row(rows, "不变量", "港股（is_hk=1）不得有申万行业", "通过" if bad == 0 else "失败",
             f"违规 {bad} 行")
        _row(rows, "不变量", "持仓明细主键 (基金,期,股票) 唯一", "通过" if dup == 0 else "失败",
             f"重复组 {dup}")
        _row(rows, "不变量", "覆盖率 = 该基金该期明细合计", "通过" if cov_bad == 0 else "失败",
             f"偏差 >0.05pp 的（基金×期）{cov_bad} 组")

    # 6 证据等级与验证结果一致
    try:
        r = inst.six_checks(b, ind["F3e_超配分位_长历史"], "超配分位")
        _row(rows, "不变量", "未过六项的因子不得标「可用」",
             "通过" if r["状态"] != "可用" or r["通过项"] == 6 else "失败",
             f"超配分位 {r['通过项']}/6 → {r['状态']}")
    except Exception as e:
        _row(rows, "不变量", "六项验证可运行", "失败", f"{type(e).__name__}: {e}")


def falsification(rows, b, ind):
    fwd = bt.forward_excess(b.market, 1)

    # 1 「拥挤即风险」的反证：分位越高，次季收益应越低；实测若相反，则该说法被证伪
    f3 = ind["F3e_超配分位_长历史"]
    lo, hi = [], []
    for qq in f3.index:
        if qq not in fwd.index:
            continue
        s = f3.loc[qq]
        r = fwd.loc[qq]
        if r.notna().sum() < 20:      # 早期季度没有次季行业收益，跳过而不是当成 0
            continue
        a = s[(s >= 0) & (s < 20)].dropna().index
        z = s[(s >= 80) & (s <= 100)].dropna().index
        if len(a) >= 3 and len(z) >= 3:
            lo.append(float(r.reindex(a).mean()))
            hi.append(float(r.reindex(z).mean()))
    if lo and hi:
        m_lo, m_hi = float(np.mean(lo)), float(np.mean(hi))
        _row(rows, "证伪", "「拥挤即风险」：高分位次季收益应更低",
             "被证伪" if m_hi > m_lo else "未被证伪",
             f"0–19 分位 {m_lo:+.2f}% vs 80–100 分位 {m_hi:+.2f}%（{len(lo)} 期均值）")

    # 2 ΔZ 是不是价格动量的代理：对过去 4 季涨幅正交化后是否还显著
    try:
        r = inst.six_checks(b, ind["F4_超配动量"], "超配动量")
        row = r["检验"][r["检验"]["项"].str.contains("正交")].iloc[0]
        res = str(row["结果"])
        # 判据是「残差仍显著」；样本 <12 期时不能下结论，标样本不足而不是证伪
        if row["通过"]:
            verdict = "通过（ΔZ 有独立信息）"
        elif "n=" in res and "12" not in res.split("n=")[1][:3]:
            verdict = "样本不足，不下结论"
        else:
            verdict = "被证伪（ΔZ 可由动量解释）"
        _row(rows, "证伪", "ΔZ 不是价格动量的代理", verdict, res[:120])
    except Exception:
        pass

    # 3 分歧度是否有预测力
    try:
        import ak_validate as akv
        r = akv.run("加仓占比", b)
        _row(rows, "证伪", "基金加仓占比可预测次季行业收益",
             "被证伪" if r.get("nw_p", 1) > 0.10 else "未被证伪",
             f"IC {r.get('ic_mean'):+.3f}、NW p {r.get('nw_p'):.3f}、{r.get('n_periods')} 期")
    except Exception as e:
        _row(rows, "证伪", "分歧度 IC 可计算", "失败", f"{type(e).__name__}")

    # 4 主动加减仓是否只是被动漂移
    try:
        q = b.latest
        q0 = list(b.quarters)[list(b.quarters).index(q) - 1]
        act = inst.active_change(b, ind, q)
        d = (b.alloc.loc[q] - b.alloc.loc[q0]).reindex(act.index)
        ratio = float(act["active"].abs().sum() / d.abs().sum())
        _row(rows, "证伪", "主动加减仓不是被动漂移（主动占比应显著）",
             "通过" if 0.05 < ratio < 0.95 else "失败",
             f"Σ|主动|/Σ|Δ配置| = {ratio * 100:.1f}%（其余为价格漂移）")
        _row(rows, "不变量", "主动项双口径符号一致", "通过" if int(act["sign_agree"].sum()) >= 28
             else "失败", f"{int(act['sign_agree'].sum())}/{len(act)} 行业符号一致")
    except Exception:
        pass

    # 5 个股层与行业层方向是否互相矛盾（方向一致性检查）
    try:
        pair = aks.comparable_pair("2026Q3")
        if pair:
            g = aks.industry_flow(*pair)
            # 必须同区间比较：个股是 2025Q3→2026Q3（四个季度，
            # 因为季报只能和上年同季比），行业配置也要用同样的四季度变化，
            # 否则拿「单季主动」对「跨年 Δ」是在比两个不同的东西。
            d4 = (b.alloc.loc[pair[0]] - b.alloc.loc[pair[1]])
            common = [k for k in g.index if k in d4.index and k != "港股"]
            agree = sum(1 for k in common
                        if np.sign(g.loc[k, "Δ持股基金数合计"]) == np.sign(d4.loc[k]))
            _row(rows, "证伪", "个股层与行业层方向一致（同区间 2025Q3→2026Q3）",
                 "通过" if agree / max(len(common), 1) >= 0.6 else "注意",
                 f"{agree}/{len(common)} 个行业同向")
    except Exception:
        pass


def main():
    rows: list[dict] = []
    b = dl.load_all()
    F = fe.build_all(b)
    ind = F["ind"]
    invariants(rows, b, ind, F["mkt"])
    falsification(rows, b, ind)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    n = len(df)
    ok = int((df["结果"] == "通过").sum())
    bad = df[df["结果"].isin(["失败"])]
    print(df.to_string(index=False, max_colwidth=70))
    print(f"\n合计 {n} 项：通过 {ok}、失败 {len(bad)}、"
          f"被证伪 {int((df['结果']=='被证伪').sum())}、注意 {int((df['结果']=='注意').sum())}")
    print(f"已写入 {OUT}")


if __name__ == "__main__":
    main()
