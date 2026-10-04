# -*- coding: utf-8 -*-
"""
signal_generator.py  ——  信号生成与判断
======================================================================
三类输出
--------
1. 市场层面信号（按季度）
   · HHI > 0.15            → 极高集中，警告
   · CR3 > 60%             → 高度集中，警告
   · 极度拥挤行业数 ≥ 3    → 结构性风险偏高
   · 显著低配行业数 ≥ 10   → K 型分化极致
   · 附市场状态标语（K型分化极致 / 极高集中 / 高度集中 / 结构性风险偏高 / 分化明显 / 相对均衡）

2. 行业层面明细（按季度 × 行业）
   配置比例、超配、Z、分位、动量、动量Z、成交额占比及其分位、共振得分、综合拥挤度、拥挤等级

3. 历史信号触发记录
   · P90突破（持仓拥挤）     超配首次突破扩窗 P90
   · 成交额拥挤（交易拥挤）   成交额占比分位 ≥90 且环比加速
   · 筹码拥挤                Top20 平均持股基金数 / 市值占比 / ≥100 只数 创扩窗新高
   · 三信号共振              共振得分 = 3
   每条记录附「触发后 3/6/12 个月相对沪深300 超额收益」，便于事后复盘。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from factor_engine import forward_cum_excess

INDUSTRIES = C.INDUSTRIES


# ============================================================================
# 1. 市场层面信号
# ============================================================================
def market_signals(ind: dict, mkt: dict, b) -> pd.DataFrame:
    rows = []
    for q in b.quarters:
        lv = ind["拥挤等级"].loc[q]
        n_ext = int((lv == "极度拥挤").sum())
        n_crowd = int(lv.isin(["极度拥挤", "拥挤"]).sum())
        n_under = int((lv == "显著低配").sum())
        n_data = int((lv == "数据不足").sum())
        hhi = float(mkt["F5_HHI"].loc[q])
        cr3 = float(mkt["F6_CR3"].loc[q])
        rows.append(dict(
            季度=q,
            HHI=hhi, HHI标准化=float(mkt["F5b_HHI标准化"].loc[q]),
            CR3=cr3, CR5=float(mkt["F6_CR5"].loc[q]), CR10=float(mkt["F6_CR10"].loc[q]),
            Top1行业=mkt["F6_Top1行业"].loc[q], Top1比例=float(mkt["F6_Top1比例"].loc[q]),
            极度拥挤数=n_ext, 拥挤及以上数=n_crowd, 显著低配数=n_under, 数据不足数=n_data,
            市场综合拥挤度Z=float(mkt["F11_市场综合拥挤度Z"].loc[q]),
            信号_HHI极高集中=bool(hhi > C.SIGNAL_HHI_EXTREME),
            信号_CR3高度集中=bool(cr3 > C.SIGNAL_CR3_HIGH),
            信号_结构性风险=bool(n_ext >= C.SIGNAL_N_EXTREME),
            信号_K型分化=bool(n_under >= C.SIGNAL_N_UNDER),
            样本基金数=float(mkt["F8_样本基金数"].loc[q]),
        ))
    df = pd.DataFrame(rows).set_index("季度")

    def state(r):
        if r["信号_结构性风险"] and r["信号_K型分化"]:
            return "K型分化极致"
        if r["信号_HHI极高集中"] and r["信号_CR3高度集中"]:
            return "极高集中"
        if r["信号_CR3高度集中"] or r["信号_HHI极高集中"]:
            return "高度集中"
        if r["信号_结构性风险"]:
            return "结构性风险偏高"
        if r["信号_K型分化"]:
            return "分化明显"
        return "相对均衡"

    df["市场状态"] = df.apply(state, axis=1)
    return df


# ============================================================================
# 2. 行业层面明细
# ============================================================================
def industry_detail(ind: dict, b, quarter: str) -> pd.DataFrame:
    q = quarter
    d = pd.DataFrame({
        "行业": INDUSTRIES,
        "配置比例%": b.alloc.loc[q].reindex(INDUSTRIES).to_numpy(),
        "超配比例%": b.overweight.loc[q].reindex(INDUSTRIES).to_numpy(),
        "超配Z": ind["F2_超配Zscore"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "超配分位%": ind["F3_超配历史分位"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "超配动量": ind["F4_超配动量"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "动量Z": ind["F4b_超配动量Z"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "超配P90阈值": ind["F1b_超配P90阈值"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "成交额占比%": ind["F9a_成交额占比%"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "成交额占比分位%": ind["F9b_成交额占比分位"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "成交额占比加速度": ind["F9c_成交额占比加速度"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "交易拥挤信号": ind["F9d_交易拥挤信号"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "S1持仓拥挤": ind["F10_S1_持仓拥挤"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "S2交易拥挤": ind["F10_S2_交易拥挤"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "S3筹码拥挤": ind["F10_S3_筹码拥挤"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "共振得分": ind["F10_共振得分"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "基准权重%": ind["F12b_基准权重%"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "F12配置系数": ind["F12_配置系数"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "F13筹码盈利比例%": ind["F13_筹码盈利比例"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "F14ETF资金流强度%": ind["F14_ETF资金流强度"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "综合拥挤度": ind["F11_综合拥挤度_推荐权重"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "综合拥挤度_IC加权": ind["F11_综合拥挤度_IC加权"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "综合拥挤度_旧口径": ind["F11_综合拥挤度_旧口径对照"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "口径背离": ind["口径背离"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "拥挤等级": ind["拥挤等级"].loc[q].reindex(INDUSTRIES).to_numpy(),
        "行业涨跌幅%": b.market["ret"].loc[q].reindex(INDUSTRIES).to_numpy(),
    })
    return d.sort_values("综合拥挤度", ascending=False).reset_index(drop=True)


def industry_history(ind: dict, b) -> pd.DataFrame:
    """全部季度 × 全部行业 的明细长表。"""
    frames = []
    for q in b.quarters:
        d = industry_detail(ind, b, q)
        d.insert(0, "季度", q)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# 3. 历史信号触发记录
# ============================================================================
_TRIGGERS = [
    ("P90突破", "F10_S1_持仓拥挤", "持仓拥挤：超配首次突破扩窗 P90"),
    ("成交额拥挤", "F10_S2_交易拥挤", "交易拥挤：成交额占比分位≥90 且环比加速"),
    ("筹码拥挤", "F10_S3_筹码拥挤", "筹码拥挤：抱团指标创扩窗新高（市场层面）"),
    ("三信号共振", None, "共振得分=3：三类信号同时触发"),
]


def signal_records(ind: dict, b) -> pd.DataFrame:
    """所有触发记录 + 触发后 3/6/12 个月超额收益。"""
    qs = list(b.quarters)
    fwd = {h: forward_cum_excess(b.market, h) for h in C.HORIZONS_Q}
    recs = []

    for label, col, desc in _TRIGGERS:
        for i in INDUSTRIES:
            if col is None:
                hit = (ind["F10_共振得分"][i] == 3)
            else:
                hit = (ind[col][i] == 1)
            hit = hit.fillna(False).astype(bool)
            for t, q in enumerate(qs):
                if not bool(hit.iloc[t]):
                    continue
                first = (t == 0) or (not bool(hit.iloc[t - 1]))
                row = dict(类型=label, 说明=desc, 季度=q, 行业=i,
                           超配比例=float(b.overweight.loc[q, i]),
                           超配分位=ind["F3_超配历史分位"].loc[q, i],
                           共振得分=ind["F10_共振得分"].loc[q, i],
                           是否首次=bool(first))
                for h in C.HORIZONS_Q:
                    row[f"触发后{h*3}月超额%"] = fwd[h].loc[q, i]
                recs.append(row)
    return pd.DataFrame(recs)


def trigger_summary(records: pd.DataFrame) -> pd.DataFrame:
    """按信号类型 × 持有期汇总。"""
    if records.empty:
        return pd.DataFrame()
    rows = []
    for label in [t[0] for t in _TRIGGERS]:
        sub = records[records["类型"] == label]
        if sub.empty:
            continue
        for h in C.HORIZONS_Q:
            col = f"触发后{h*3}月超额%"
            x = sub[col].dropna()
            if len(x) == 0:
                continue
            rows.append(dict(信号类型=label, 持有时长=f"{h*3}个月", 触发次数=len(x),
                             均值=x.mean(), 中位数=x.median(),
                             负收益概率=x.lt(0).mean() * 100,
                             胜率=x.gt(0).mean() * 100))
    return pd.DataFrame(rows)


def count_summary(records: pd.DataFrame) -> pd.DataFrame:
    """按信号类型统计触发次数与涉及行业。"""
    if records.empty:
        return pd.DataFrame()
    rows = []
    for label in [t[0] for t in _TRIGGERS]:
        sub = records[records["类型"] == label]
        first = sub[sub["是否首次"]]
        rows.append(dict(信号类型=label, 触发次数=len(sub), 首次触发次数=len(first),
                         涉及行业数=sub["行业"].nunique(),
                         涉及行业="、".join(sorted(sub["行业"].unique())[:12])
                         + ("…" if sub["行业"].nunique() > 12 else "")))
    return pd.DataFrame(rows)


# ============================================================================
# 4. 当前预警清单
# ============================================================================
def current_alerts(ind: dict, mkt: dict, b, quarter: str | None = None) -> list:
    q = quarter or b.latest
    out = []
    lv = ind["拥挤等级"].loc[q]
    ext = list(lv.index[lv == "极度拥挤"])
    und = list(lv.index[lv == "显著低配"])
    hhi = float(mkt["F5_HHI"].loc[q])
    cr3 = float(mkt["F6_CR3"].loc[q])
    if hhi > C.SIGNAL_HHI_EXTREME:
        out.append(("高", "市场集中度", f"HHI={hhi:.4f} > {C.SIGNAL_HHI_EXTREME}（极高集中）"))
    if cr3 > C.SIGNAL_CR3_HIGH:
        out.append(("高", "头部集中度", f"CR3={cr3:.2f}% > {C.SIGNAL_CR3_HIGH}%（高度集中）"))
    if len(ext) >= C.SIGNAL_N_EXTREME:
        out.append(("高", "持仓拥挤", f"极度拥挤行业 {len(ext)} 个（≥{C.SIGNAL_N_EXTREME}）："
                                   + "、".join(ext)))
    if len(und) >= C.SIGNAL_N_UNDER:
        out.append(("中", "结构分化", f"显著低配行业 {len(und)} 个（≥{C.SIGNAL_N_UNDER}），K 型分化"))
    s3 = mkt["F10_S3筹码拥挤_时序"].loc[q]
    if s3 == s3 and s3 == 1:
        out.append(("中", "筹码拥挤", "抱团指标创扩窗新高（市场层面）"))
    res3 = ind["F10_共振得分"].loc[q]
    r3 = list(res3.index[res3 == 3])
    if r3:
        out.append(("高", "三信号共振", "共振得分=3 的行业：" + "、".join(r3)))
    # 新增：F12 配置系数（相对基准的极端重配）
    if "F12_配置系数" in ind:
        cf = ind["F12_配置系数"].loc[q]
        r12 = list(cf.index[cf > C.THRESH_CF_EXTREME])
        if r12:
            out.append(("高", "相对基准重配",
                        f"配置系数 > {C.THRESH_CF_EXTREME}（极度拥挤）：" + "、".join(r12)))
    # 新增：F13 获利盘拥挤
    if "F13_筹码盈利比例" in ind:
        pr = ind["F13_筹码盈利比例"].loc[q]
        r13 = list(pr.index[pr >= C.THRESH_PROFIT_EXTREME])
        if r13:
            out.append(("中", "获利盘拥挤",
                        f"筹码盈利比例 ≥{C.THRESH_PROFIT_EXTREME:.0f}%：" + "、".join(r13)))
    # 新增：口径背离
    if "口径背离" in ind:
        dv = ind["口径背离"].loc[q]
        rdv = list(dv.index[dv])
        if rdv:
            out.append(("中", "口径背离",
                        "Z/分位 与 F12 指向相反，不可只用一个标签：" + "、".join(rdv)))
    return out


def dashboard_payload(ind: dict, mkt: dict, b, bt_ic: pd.DataFrame | None = None) -> dict:
    """一次性打包看板所需全部内容。"""
    ms = market_signals(ind, mkt, b)
    det = industry_detail(ind, b, b.latest)
    rec = signal_records(ind, b)
    return dict(market_signals=ms, detail=det, records=rec,
                trigger_summary=trigger_summary(rec),
                count_summary=count_summary(rec),
                alerts=current_alerts(ind, mkt, b),
                industry_history=industry_history(ind, b),
                ic_table=bt_ic)
