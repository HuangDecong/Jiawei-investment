# -*- coding: utf-8 -*-
"""自检：把「口径是否自洽」写成可执行的断言，而不是靠人肉对表。

为什么单独一层
--------------
这一版改了三个会系统性影响结论的口径（价格中性、扩张分位、同行均值），
改口径最容易出的事不是报错，而是**静默不一致**：页面 A 用的分母是
「有变动的基金」，页面 B 用的是「全部样本」，两个数都「看起来对」。
本模块把每条口径写成恒等式，跑一遍就能看出来有没有人改坏了。

每个检查返回 (项目, 检查内容, 实测值, 是否通过, 说明)，页面直接渲染成表。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import canon as cn
import config as C


def _row(name, what, val, ok, note=""):
    return dict(项目=name, 检查内容=what, 实测值=val,
                结果="通过" if ok else "不通过", 说明=note)


def run(q: str | None = None) -> pd.DataFrame:
    q = q or cn.latest()
    rows = []

    # 1) 前十大归一化：每只基金每期合计应为 100
    p = cn.panel_top10(q)
    if not p.empty:
        dev = float((p.sum(axis=1) - 100).abs().max())
        rows.append(_row("前十大归一化", "每只基金行业权重合计 = 100（最大偏差）",
                         f"{dev:.2e}", dev < 1e-6,
                         "归一化写错会让所有行业的水平整体偏移"))

    # 2) 面板覆盖：31 个行业列齐全
    rows.append(_row("行业列", "面板列数 = 31",
                     f"{p.shape[1]}" if not p.empty else "—",
                     (not p.empty) and p.shape[1] == len(C.INDUSTRIES)))

    # 3) 广度：加仓 + 减仓 + 不变 = 样本数
    pq = cn.pair_quarter(q)
    if pq:
        b = cn.breadth(*pq)
        if not b.empty:
            gap = int((b["加仓基金数"] + b["减仓基金数"] + b["不变基金数"]
                       - b["样本基金数"]).abs().max())
            rows.append(_row("广度分母", "加仓 + 减仓 + 不变 = 样本基金数（最大差）",
                             f"{gap}", gap == 0,
                             "分母写错会让「多少基金在加」整体失真"))

    # 4) 价格中性：按行业配置加权，平均主动净额应约为 0
    if not pq:
        pq = None
    if pq:
        d = cn.price_neutral_delta(*pq)
        if not d.empty:
            # 恒等式：每只基金各行业 Δ 之和 = 0（本期与反事实都归一到 100），
            # 所以对基金取平均后，31 个行业的均值之和也必须是 0。
            # 不能按配置权重加权——权重本身与 Δ 相关，加权和不为 0 是正常的。
            agg = float(d.mean().sum())
            rows.append(_row("价格中性", "31 个行业的平均主动净额之和 = 0",
                             f"{agg:+.3f}pp", abs(agg) < 0.02,
                             "反事实权重若算错，整体会凭空多出/少掉一笔钱"))

    # 5) GK 恒等式：主动 + 被动 = 变化
    pf = cn.pair_full()
    if pf:
        dk = cn.decomp(*pf)
        if not dk.empty:
            err = float((dk["主动贡献"] + dk["被动贡献"] - dk["变化"]).abs().max())
            # decomp 输出保留 3 位小数，误差上限就是 0.002pp
            rows.append(_row("GK 分解", "主动 + 被动 − 变化 = 0（最大误差）",
                             f"{err:.1e}", err < 2e-3,
                             "三者对不上说明反事实权重的归一化有问题"))

    # 6) 分位范围
    c = cn.compare_periods(q)
    if not c.empty:
        v = c["历史分位"].dropna()
        ok = bool(v.between(0, 100).all())
        rows.append(_row("配置分位", "全部落在 0–100",
                         f"{v.min():.0f}–{v.max():.0f}", ok))

    # 7) 覆盖率异常
    cov = cn.coverage(q)
    if not cov.empty:
        # 只看样本内：异常记录本来就被 universe() 排除了，
        # 直接对全量 coverage 统计会把这批已剔除的脏数据再算一遍。
        uu = cn.universe(q)
        cv = cov.reindex(list(uu)).dropna() if uu else cov
        bad = int(((cv < 1) | (cv > 105)).sum())
        rows.append(_row("覆盖率", "样本内异常记录（<1% 或 >105%）应为 0",
                         f"{bad}", bad == 0,
                         "异常记录已在样本池阶段剔除，此处复核"))

    # 8) 个股榜一致性：Δ = 本期 − 上期
    if pq:
        sf = cn.stock_flow_top10(*pq)
        if not sf.empty:
            err = float((sf["n_funds"] - sf["n_funds_0"] - sf["d_n"]).abs().max())
            rows.append(_row("个股榜", "Δ基金数 = 本期 − 上期（最大误差）",
                             f"{err:.2e}", err < 1e-9,
                             "个股榜错位通常来自代码补零撞码"))

    # 9) 拥挤度：CSI 落在三个分量之间
    cs = cn.crowding_score(q)
    if not cs.empty:
        _p = cs[["配置分位", "价格分位", "资金分位"]]
        out = ((cs["CSI"] < _p.min(axis=1) - 0.11) |
               (cs["CSI"] > _p.max(axis=1) + 0.11)).sum()
        rows.append(_row("CSI", "CSI 必落在三个分量的最小值与最大值之间",
                         f"越界 {int(out)} 个", int(out) == 0,
                         "等权平均不应超出分量区间"))

    # 10) 样本与文档一致
    u = cn.universe(q)
    rows.append(_row("样本池", f"{q} 主动权益基金数 > 3000", f"{len(u):,}",
                     len(u) > 3000,
                     "样本骤降通常意味着筛选规则被改坏"))

    t = pd.DataFrame(rows)
    t.attrs["n_pass"] = int((t["结果"] == "通过").sum())
    t.attrs["n_all"] = len(t)
    return t


def html(t: pd.DataFrame, color: dict) -> str:
    """渲染成一张紧凑的自检表。"""
    if t.empty:
        return ""
    n_ok, n_all = int(t.attrs.get("n_pass", 0)), int(t.attrs.get("n_all", 0))
    head = (f'<div style="font-size:12px;margin:2px 0 5px">'
            f'<b>{n_ok}/{n_all} 项通过</b>'
            + ('——口径自洽' if n_ok == n_all else
               '——<span style="color:#C62828">有项目不通过，结论暂不可信</span>')
            + '</div>')
    tr = "".join(
        f'<tr><td style="padding:4px 7px;border-top:1px solid {color["grid"]};'
        f'font-size:11px;font-weight:600;white-space:nowrap">{r.项目}</td>'
        f'<td style="padding:4px 7px;border-top:1px solid {color["grid"]};'
        f'font-size:10.5px">{r.检查内容}</td>'
        f'<td style="padding:4px 7px;border-top:1px solid {color["grid"]};'
        f'font-size:10.5px;font-family:ui-monospace,Consolas,monospace">{r.实测值}</td>'
        f'<td style="padding:4px 7px;border-top:1px solid {color["grid"]};'
        f'font-size:10.5px;white-space:nowrap">'
        f'<span style="color:{"#1A7F37" if r.结果 == "通过" else "#C62828"}">'
        f'{"✓" if r.结果 == "通过" else "✗"} {r.结果}</span></td>'
        f'<td style="padding:4px 7px;border-top:1px solid {color["grid"]};'
        f'font-size:10px;color:{color["text_dim"]}">{r.说明}</td></tr>'
        for r in t.itertuples())
    return (head +
            f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr>'
            f'<th style="padding:4px 7px;text-align:left;font-size:10.5px">项目</th>'
            f'<th style="padding:4px 7px;text-align:left;font-size:10.5px">检查内容</th>'
            f'<th style="padding:4px 7px;text-align:left;font-size:10.5px">实测值</th>'
            f'<th style="padding:4px 7px;text-align:left;font-size:10.5px">结果</th>'
            f'<th style="padding:4px 7px;text-align:left;font-size:10.5px">为什么查这条</th>'
            f'</tr></thead><tbody>{tr}</tbody></table>')


if __name__ == "__main__":
    t = run()
    print(t.to_string(index=False))
    print(f"{t.attrs['n_pass']}/{t.attrs['n_all']} 通过")
