# -*- coding: utf-8 -*-
"""
seed_raw_data.py  ——  首次初始化 data/raw/ 目录
======================================================================
用途：把两份原始材料中"已经存在"的数据，一次性拆分成 data/raw/ 下
      规范的、可增量更新的独立文件。之后每季度只需覆盖对应文件即可。

输入：
  1. data/raw/master_workbook.xlsx  —— 《公募基金行业拥挤度与集中度数据.xlsx》副本
  2. tools/market_reference.py      —— 申万一级行业指数季度涨跌幅 / 成交额（东方财富妙想）

输出（全部为 UTF-8 xlsx）：
  data/raw/industry_allocation.xlsx    季度 × 31行业 配置比例(%)
  data/raw/industry_overweight.xlsx    季度 × 31行业 超配比例(pct)
  data/raw/hs300_weights.xlsx          行业, 权重%(沪深300静态权重)
  data/raw/hs300_returns.xlsx          季度, 沪深300涨跌幅%
  data/raw/sw_industry_turnover.xlsx   行业, 季度, 成交额(亿元)
  data/raw/sw_industry_returns.xlsx    行业, 季度, 涨跌幅%
  data/raw/industry_cr_hhi.xlsx        季度, CR3/CR5/CR10/HHI/...
  data/raw/stock_crowd.xlsx            季度, 个股抱团指标时序
  data/raw/position_proxy.xlsx         季度, 前十占净值比 等仓位代理
  data/raw/hot_stocks_top20_2026Q2.xlsx 最新季度抱团股Top20
  data/raw/fund_holdings_TEMPLATE.xlsx 前十大重仓股 格式模板（示例，非真实逐基金数据）

用法：
  python tools/seed_raw_data.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import openpyxl
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from market_reference import (RET, TURN, RET_HS300, QUARTERS_NEWEST_FIRST,  # noqa: E402
                              sanity_check)

ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")

QUARTERS = list(QUARTERS_NEWEST_FIRST)[::-1]          # 旧 -> 新，34 个
INDUSTRIES = [
    "农林牧渔", "基础化工", "钢铁", "有色金属", "电子", "汽车", "家用电器",
    "食品饮料", "纺织服饰", "轻工制造", "医药生物", "公用事业", "交通运输",
    "房地产", "商贸零售", "社会服务", "银行", "非银金融", "综合", "建筑材料",
    "建筑装饰", "电力设备", "机械设备", "国防军工", "计算机", "传媒", "通信",
    "煤炭", "石油石化", "环保", "美容护理",
]
NQ, NI = len(QUARTERS), len(INDUSTRIES)


def _wb() -> openpyxl.Workbook:
    path = os.path.join(RAW, "master_workbook.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 {path}，请先把源工作簿复制为 master_workbook.xlsx")
    return openpyxl.load_workbook(path, data_only=True)


def _block(ws, first_row: int, ncols: int, nrows: int = NQ):
    return [list(r) for r in ws.iter_rows(min_row=first_row, max_row=first_row + nrows - 1,
                                          max_col=ncols, values_only=True)]


def _f(v):
    return float(v) if v is not None else np.nan


# ------------------------------------------------------------------ 各文件
def make_industry_matrices(wb, log):
    ws = wb["2-行业配置比例热力图"]
    hdr = [c for c in next(ws.iter_rows(min_row=4, max_row=4, values_only=True))]
    inds = [h for h in hdr[1:] if h]
    assert inds == INDUSTRIES, f"Sheet2 行业顺序不一致: {inds}"

    alloc = pd.DataFrame([[float(v) for v in r[1:32]] for r in _block(ws, 5, 32)],
                         index=[r[0] for r in _block(ws, 5, 32)], columns=inds).loc[QUARTERS]
    over = pd.DataFrame([[float(v) for v in r[1:32]] for r in _block(wb["4-行业超配比例"], 5, 32)],
                        index=[r[0] for r in _block(wb["4-行业超配比例"], 5, 32)],
                        columns=inds).loc[QUARTERS]

    out_alloc = alloc.reset_index().rename(columns={"index": "季度"})
    out_over = over.reset_index().rename(columns={"index": "季度"})
    out_alloc.to_excel(os.path.join(RAW, "industry_allocation.xlsx"), index=False)
    out_over.to_excel(os.path.join(RAW, "industry_overweight.xlsx"), index=False)
    log.append(f"industry_allocation.xlsx   {out_alloc.shape}  加总={alloc.sum(axis=1).round(6).unique()}")
    log.append(f"industry_overweight.xlsx   {out_over.shape}")

    # 沪深300 静态权重 = 配置比例 - 超配比例（已验证 34 期恒定，std=0，加总=100）
    W = alloc - over
    assert np.allclose(W.std().to_numpy(), 0.0, atol=1e-9), "沪深300权重非静态，口径需重新确认"
    w = W.iloc[-1]
    assert abs(w.sum() - 100.0) < 1e-6, f"权重加总={w.sum()}"
    pd.DataFrame({"行业": INDUSTRIES, "权重%": [round(float(w[i]), 6) for i in INDUSTRIES]}
                 ).to_excel(os.path.join(RAW, "hs300_weights.xlsx"), index=False)
    log.append(f"hs300_weights.xlsx         {NI} 行业  加总={w.sum():.6f}  34期标准差={W.std().max():.2e}")


def make_cr_hhi(wb, log):
    rows = _block(wb["3-行业集中度CR_HHI"], 5, 10)
    df = pd.DataFrame([dict(季度=r[0], CR3=_f(r[1]), CR5=_f(r[2]), CR10=_f(r[3]),
                            HHI=_f(r[4]), HHI标准化=_f(r[5]), Top1行业=r[6],
                            Top1比例=_f(r[7]), Top2行业=r[8], Top3行业=r[9]) for r in rows])
    df = df.set_index("季度").loc[QUARTERS].reset_index()
    df.to_excel(os.path.join(RAW, "industry_cr_hhi.xlsx"), index=False)
    log.append(f"industry_cr_hhi.xlsx       {df.shape}  最新 HHI={df['HHI'].iloc[-1]:.4f}")


def make_stock_crowd(wb, log):
    rows = _block(wb["5-个股拥挤度"], 5, 7)
    df = pd.DataFrame([dict(季度=r[0], Top20平均持股基金数=_f(r[1]), Top20持仓市值占比=_f(r[2]),
                            持股基金数_ge100=_f(r[3]), 持股基金数_ge50=_f(r[4]),
                            持股基金数_ge20=_f(r[5]), 个股总数=_f(r[6])) for r in rows])
    df = df.set_index("季度").loc[QUARTERS].reset_index()
    df.to_excel(os.path.join(RAW, "stock_crowd.xlsx"), index=False)

    hot = pd.DataFrame([dict(股票代码=str(r[0]), 股票名称=r[1], 申万一级=r[2],
                             持股基金数=_f(r[3]), 持仓市值万元=_f(r[4]),
                             持仓市值亿元=_f(r[5]))
                        for r in wb["5b-最新抱团股Top20"].iter_rows(
                            min_row=5, max_row=24, max_col=6, values_only=True)])
    hot.to_excel(os.path.join(RAW, f"hot_stocks_top20_{QUARTERS[-1]}.xlsx"), index=False)
    log.append(f"stock_crowd.xlsx           {df.shape}  最新Top20平均持股基金数="
               f"{df['Top20平均持股基金数'].iloc[-1]:.1f}")
    log.append(f"hot_stocks_top20_{QUARTERS[-1]}.xlsx {hot.shape}  首位={hot['股票名称'].iloc[0]}")


def make_position(wb, log):
    rows = _block(wb["6-基金仓位变化"], 5, 7)
    df = pd.DataFrame([dict(季度=r[0], 样本基金数=_f(r[1]), 前十占净值比均值=_f(r[2]),
                            前十占净值比中位数=_f(r[3]), 重仓股数均值=_f(r[4]),
                            隐含股票仓位估算均值=_f(r[5]), 沪深300季末点位=_f(r[6]))
                       for r in rows])
    df = df.set_index("季度").loc[QUARTERS].reset_index()
    df.to_excel(os.path.join(RAW, "position_proxy.xlsx"), index=False)
    n_bad = int((~df["隐含股票仓位估算均值"].between(0, 100)).sum())
    log.append(f"position_proxy.xlsx        {df.shape}  样本基金数={int(df['样本基金数'].iloc[-1])}"
               f"  隐含股票仓位列异常期数={n_bad}/{NQ}")


def make_market(wb, log):
    turn = pd.DataFrame({k: v[::-1] for k, v in TURN.items()}, index=QUARTERS)[INDUSTRIES]
    ret = pd.DataFrame({k: v[::-1] for k, v in RET.items()}, index=QUARTERS)[INDUSTRIES]

    tl = turn.stack().rename("成交额亿元").reset_index()
    tl.columns = ["季度", "行业", "成交额亿元"]
    tl = tl[["行业", "季度", "成交额亿元"]]
    tl.to_excel(os.path.join(RAW, "sw_industry_turnover.xlsx"), index=False)

    rl = ret.stack().rename("涨跌幅%").reset_index()
    rl.columns = ["季度", "行业", "涨跌幅%"]
    rl = rl[["行业", "季度", "涨跌幅%"]]
    rl.to_excel(os.path.join(RAW, "sw_industry_returns.xlsx"), index=False)

    pd.DataFrame({"季度": QUARTERS, "涨跌幅%": list(RET_HS300)[::-1]}
                 ).to_excel(os.path.join(RAW, "hs300_returns.xlsx"), index=False)

    # 用成交额占比反算的"全市场行业成交额合计"抽样核对
    share = turn.div(turn.sum(axis=1), axis=0) * 100
    log.append(f"sw_industry_turnover.xlsx  {tl.shape}  最新合计成交额={turn.iloc[-1].sum():,.0f}亿")
    log.append(f"sw_industry_returns.xlsx   {rl.shape}")
    log.append(f"hs300_returns.xlsx         {NQ} 期  最新={list(RET_HS300)[::-1][-1]:+.4f}%")
    log.append(f"  成交额占比 最新 Top3: "
               + ", ".join(f"{k}={v:.1f}%" for k, v in share.iloc[-1].nlargest(3).items()))


def make_holdings_template(wb, log):
    """前十大重仓股明细的**格式模板**（示例行，非真实逐基金数据）。"""
    hot = pd.DataFrame([dict(code=str(r[0]), name=r[1], sw=r[2])
                        for r in wb["5b-最新抱团股Top20"].iter_rows(
                            min_row=5, max_row=24, max_col=6, values_only=True)])
    demo = hot.head(6).copy()
    demo["基金代码"] = ["000001", "000001", "000002", "000002", "000003", "000003"]
    demo["占净值比"] = [8.50, 6.20, 7.80, 5.90, 9.10, 4.30]
    demo["持仓市值"] = [round(v * 323000, 0) for v in demo["占净值比"]]   # 示意
    demo["季度"] = QUARTERS[-1]
    out = demo.rename(columns={"code": "股票代码", "name": "股票名称",
                               "sw": "申万一级"})[
        ["基金代码", "股票代码", "股票名称", "申万一级", "持仓市值", "占净值比", "季度"]]
    out.to_excel(os.path.join(RAW, "fund_holdings_TEMPLATE.xlsx"), index=False)

    ph = pd.DataFrame([{
        "基金代码": "000001", "股票代码": "300308", "股票名称": "中际旭创",
        "申万一级": "通信", "持仓市值": 2745500, "占净值比": 8.50, "季度": QUARTERS[-1]},
    ])
    _ = ph  # 示例行仅用于说明
    log.append(f"fund_holdings_TEMPLATE.xlsx {out.shape}  ⚠️ 示例格式模板，"
               "非真实逐基金持仓；真实文件请命名为 fund_holdings_YYYYQX.xlsx")


def main():
    sanity_check()
    os.makedirs(RAW, exist_ok=True)
    log = []
    wb = _wb()
    make_industry_matrices(wb, log)
    make_cr_hhi(wb, log)
    make_stock_crowd(wb, log)
    make_position(wb, log)
    make_market(wb, log)
    make_holdings_template(wb, log)
    wb.close()

    print("=" * 74)
    print("data/raw/ 初始化完成")
    print("=" * 74)
    for line in log:
        print("  " + line)
    print()
    files = sorted(os.listdir(RAW))
    print(f"共 {len(files)} 个文件：")
    for f in files:
        p = os.path.join(RAW, f)
        print(f"  {f:<34} {os.path.getsize(p)/1024:>8.1f} KB")


if __name__ == "__main__":
    main()
