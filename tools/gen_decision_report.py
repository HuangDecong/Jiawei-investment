# -*- coding: utf-8 -*-
"""生成《机构持仓决策报告》（Excel）。

结构（固定）：
    0 决策摘要   结论、证据等级、仓位区间、反证清单
    1 行业配置   31 行：动作 / 主动加减仓 / 加仓占比 / 分歧度 / 分位 / 配置 / 理由 / 失效
    2 个股清单   买入候选 / 观察 / 回避（数据未就绪时只写原因，不留空表）
    3 六项验证   每个度量的六项检验明细与降级结论
    4 数据健康   五类数据的覆盖、缺口、取数日志、清洗剔除
    5 口径说明   每个数字的来源、公式、边界

用法：python tools/gen_decision_report.py [--quarter 2026Q3]
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import change_analysis as ca          # noqa: E402
import config as C                    # noqa: E402
import data_loader as dl              # noqa: E402
import factor_engine as fe            # noqa: E402
import institution as inst            # noqa: E402

HDR_FILL = PatternFill("solid", fgColor="1F2D3D")
HDR_FONT = Font(color="FFFFFF", bold=True, size=10)
TITLE_FONT = Font(bold=True, size=13, color="1F2D3D")
SUB_FONT = Font(size=9, color="6B7A8C")
THIN = Side(style="thin", color="DCE3EA")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _style_sheet(ws, widths: dict | None = None, freeze: str = "A2"):
    for cell in ws[1]:
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.freeze_panes = freeze
    ws.row_dimensions[1].height = 30
    for col, w in (widths or {}).items():
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=1):
        for cell in row:
            cell.border = BORDER
            if cell.row > 1:
                cell.alignment = Alignment(vertical="top", wrap_text=True)


def main(quarter: str | None = None) -> str:
    b = dl.load_all()
    F = fe.build_all(b)
    IND, MKT = F["ind"], F["mkt"]
    q = quarter or b.latest
    qs = list(b.quarters)
    q0 = qs[qs.index(q) - 1]

    t = ca.delta_table(b, IND, q)
    t = ca.credibility_gate(b, IND, t, MKT.get("F8_样本基金数"))
    t0 = ca.credibility_gate(b, IND, ca.delta_table(b, IND, q0), MKT.get("F8_样本基金数"))
    active = inst.active_change(b, IND, q)
    disp = inst.dispersion_proxy(t)
    pos = inst.position_advice(disp)
    checks = [inst.six_checks(b, IND["F3e_超配分位_长历史"], "超配分位（长历史）"),
              inst.six_checks(b, inst.active_panel(b, IND), "主动加减仓"),
              inst.six_checks(b, IND["F1_超配比例"], "超配比例"),
              inst.six_checks(b, IND["F4_超配动量"], "超配动量")]
    ce = inst.counter_evidence(t, checks)
    _dp = inst.latest_fund_industry_period()
    disp_real = inst.dispersion_real(*_dp) if _dp else inst.dispersion_real(q, q0)
    dec = inst.decision_table(t, active, t0=t0, disp=disp_real)
    stock = inst.stock_flow(q, q0)

    out = os.path.join(ROOT, "报告", "机构持仓决策报告.xlsx")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        # ---------- 0 决策摘要 ----------
        n_ok = sum(1 for c in checks if c["状态"] == "可用")
        summary = pd.DataFrame({
            "项": ["数据季度", "上期", "通过六项验证的度量", "证据等级",
                   "仓位建议", "分歧度状态", "个股清单状态", "分歧度数据", "结论"],
            "内容": [
                q, q0, f"{n_ok} / {len(checks)} 个",
                "全部降级为「参考」——无度量同时通过 IC、分组单调、剔Q4、分段、正交化、成本六项",
                f"{pos['low']}%–{pos['high']}%（{pos['stance']}）：{pos['why']}",
                f"代理：{disp['note']}",
                ("可用：每期前 50 只持股基金数，共 "
                 f"{stock.shape[0]} 只（两期并集）") if not stock.empty else "未就绪：需先运行数据管道",
                (f"真实口径：47 只偏股混合（规模前 50 中有两期数据者）；区间 "
                 f"{_dp[1] if _dp else '—'} → {_dp[0] if _dp else '—'}（跨两季）；"
                 "仅前 10 重仓行业、剔除单行业>80% 主题基金，样本偏科技主题")
                if not disp_real.empty else "未就绪：需 fund_industry_50x4.json",
                "本期不给「超配/回避」的可执行指令，只给参考清单；"
                "原因是超配分位剔 Q4 后 p=0.110、2019–2022 段方向相反（IC −0.055）",
            ],
        })
        summary.to_excel(xw, sheet_name="0 决策摘要", index=False)
        ws = xw.sheets["0 决策摘要"]
        _style_sheet(ws, {"A": 22, "B": 110}, freeze="A2")
        for r in range(2, ws.max_row + 1):
            ws.row_dimensions[r].height = 34

        # ---------- 1 行业配置 ----------
        dec.to_excel(xw, sheet_name="1 行业配置", index=False)
        _style_sheet(xw.sheets["1 行业配置"],
                     {"A": 11, "B": 9, "C": 11, "D": 10, "E": 9, "F": 9,
                      "G": 10, "H": 10, "I": 9, "J": 9, "K": 12,
                      "L": 9, "M": 34, "N": 40, "O": 30}, freeze="A2")

        # ---------- 2 个股清单 ----------
        if stock.empty:
            sdf = pd.DataFrame({
                "项": ["状态", "原因", "需要的数据", "当前可用"],
                "内容": ["未就绪",
                         "个股级持股基金数已取到前 50 只/期，但所属行业映射与个股次季收益缺失，"
                         "无法完成 IC 验证，按规则不得进入决策表",
                         "股票→申万一级映射（31 次行业成分查询）、个股季度收益（逐只查询）",
                         "每期 50 只的 Δ持股基金数、Δ持仓市值（仅作观察）"],
            })
        else:
            sdf = stock.rename(columns={
                "name": "股票名称", "n0": f"{q0} 持股基金数", "n1": f"{q} 持股基金数",
                "d_n": "Δ持股基金数", "mv1": "持仓市值(亿)", "mv0": f"{q0} 持仓市值(亿)", "d_mv": "Δ持仓市值(亿)",
                "state": "榜单状态"})
            # 行业映射：现有 hot_stocks_top20 只覆盖 20 只，其余标未映射（不猜测）
            try:
                _pool = ca.load_stock_pool().set_index("股票名称")["行业"]
                sdf["所属行业"] = sdf["股票名称"].map(_pool).fillna("未映射")
            except Exception:
                sdf["所属行业"] = "未映射"
            _act = dec.set_index("行业")["建议动作"]
            sdf["行业动作"] = sdf["所属行业"].map(_act).fillna("未映射")

            def _bucket(r):
                if r["行业动作"] == "超配" and r["Δ持股基金数"] > 0:
                    return "买入候选（行业超配 + Δ基金数>0；未过IC，降级为观察仓）"
                if r["行业动作"] == "回避" or r["Δ持股基金数"] <= -100:
                    return "回避（行业回避或单季减仓≥100家）"
                if r["行业动作"] == "未映射":
                    return "观察（行业未映射，不能分档）"
                return "观察"
            sdf["分档结论"] = sdf.apply(_bucket, axis=1)
            sdf = sdf.sort_values("Δ持股基金数", ascending=False)
        sdf.to_excel(xw, sheet_name="2 个股清单", index=False)
        _style_sheet(xw.sheets["2 个股清单"], {"A": 14, "B": 11, "C": 11, "D": 11,
                                               "E": 11, "F": 12, "G": 12,
                                               "H": 12, "I": 12, "J": 11, "K": 40}, freeze="A2")

        # ---------- 3 六项验证 ----------
        rows = []
        for c in checks:
            for _, r in c["检验"].iterrows():
                rows.append(dict(度量=c["指标"], 检验项=r["项"],
                                 通过="✅" if r["通过"] else "❌",
                                 结果=r["结果"], 总体判定=f"{c['状态']}（{c['通过项']}/{c['总项']}）"))
        pd.DataFrame(rows).to_excel(xw, sheet_name="3 六项验证", index=False)
        _style_sheet(xw.sheets["3 六项验证"], {"A": 18, "B": 14, "C": 7,
                                               "D": 78, "E": 20}, freeze="A2")

        # ---------- 4 数据健康 ----------
        ds = inst.data_status()
        ds.to_excel(xw, sheet_name="4 数据健康", index=False)
        _style_sheet(xw.sheets["4 数据健康"], {"A": 18, "B": 20, "C": 9,
                                               "D": 30, "E": 34, "F": 46}, freeze="A2")

        # ---------- 5 反证清单 ----------
        pd.DataFrame(ce).rename(columns={"点": "可能的错误", "推翻": "什么数据能推翻"}).to_excel(
            xw, sheet_name="5 反证清单", index=False)
        _style_sheet(xw.sheets["5 反证清单"], {"A": 62, "B": 78}, freeze="A2")

        # ---------- 6 口径说明 ----------
        spec = pd.DataFrame({
            "项": ["主动加减仓", "价格漂移", "主动（市值口径）", "超配分位", "分歧度",
                   "个股 Δ 持股基金数", "仓位建议", "样本边界"],
            "口径": [
                "Δ配置pp − 被动漂移pp；被动 = 上季配置 ×(1+行业指数收益) 后按 31 行业归一",
                "用申万一级行业指数季度收益近似个股组合收益",
                "Δ持仓市值 − 上季持仓市值 × 行业收益；总市值为缓存重仓口径",
                "F3e：偏股混合单类型超配序列的扩张窗口分位（无前视）",
                "行业 Δ配置 的横截面离散度（代理，非基金级加仓占比）",
                "两期前 50 只并集，n1−n0；跌出前 50 会被误判为退出",
                "按行业 Δ配置 横截面标准差分档，非基金级分歧度",
                "基金×个股明细 0 期；全持仓 0；日度净值 0；季报文本 0（Wind MCP 仅单基金查询）",
            ],
        })
        spec.to_excel(xw, sheet_name="6 口径说明", index=False)
        _style_sheet(xw.sheets["6 口径说明"], {"A": 22, "B": 110}, freeze="A2")

    print(f"已生成：{out}")
    print(f"  行业表 {len(dec)} 行；动作分布 {dec['建议动作'].value_counts().to_dict()}")
    print(f"  个股清单 {'可用' if not stock.empty else '未就绪'}；"
          f"通过六项的度量 {n_ok}/{len(checks)}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarter", default=None)
    a = ap.parse_args()
    main(a.quarter)
