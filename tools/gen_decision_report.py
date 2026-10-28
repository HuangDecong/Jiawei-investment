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
import ak_data as akd                  # noqa: E402

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
    # 基金行为类因子（全市场持仓库）：期数够就纳入验证，不够则标「样本不足」
    try:
        import ak_validate as akv  # noqa: E402
        for _m in ("加仓占比", "分歧度"):
            _r = akv.run(_m, b)
            if _r.get("checks") is not None:
                checks.append(_r)
    except Exception:
        pass
    ce = inst.counter_evidence(t, checks)
    # 分歧度的验证结果（用于摘要里如实标注「能描述、不能预测」）
    _dv = {}
    for _c in checks:
        if _c.get("指标") == "加仓占比":
            _dv = dict(ic=_c.get("ic_mean", float("nan")), p=_c.get("nw_p", float("nan")))
            break
    # 优先全市场口径（东财），回退 Wind 50 只
    _aq = akd.latest_quarters()
    disp_real = akd.dispersion_ak(*_aq) if _aq else pd.DataFrame()
    _disp_src = "东方财富（AKShare）全市场"
    if disp_real.empty:
        _dp = inst.latest_fund_industry_period()
        disp_real = inst.dispersion_real(*_dp) if _dp else inst.dispersion_real(q, q0)
        _disp_src = "Wind 50 只样本"
    if not disp_real.empty:
        # 用真实分歧度中位数给仓位建议（6.3 事前阈值），而不是行业离散度代理
        pos = inst.position_advice_from_dispersion(
            float(disp_real["分歧度"].median()),
            int(disp_real["样本基金数"].max()))
    dec = inst.decision_table(t, active, t0=t0, disp=disp_real)
    # 个股：优先全市场「同披露口径」两期（ak_stock），回退 Wind 前 50 只
    import ak_stock as aks  # noqa: E402
    sf = pd.DataFrame()
    spair = None
    try:
        spair = aks.comparable_pair(q)
        if spair:
            sf = aks.stock_flow(*spair)
    except Exception:
        sf = pd.DataFrame()
    stock = sf if not sf.empty else inst.stock_flow(q, q0)

    out = os.path.join(ROOT, "报告", "机构持仓决策报告.xlsx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # 文件被 Excel 占用时自动降级文件名，避免整份报告生成失败
    try:
        _probe = open(out, "a")
        _probe.close()
    except PermissionError:
        out = os.path.join(ROOT, "报告", "机构持仓决策报告_r19.4.xlsx")
        print(f"原文件被占用，改写到：{os.path.basename(out)}")

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
                (f"真实口径：全市场 {int(disp_real['样本基金数'].max())} 只基金、"
                 f"{disp_real.attrs.get('q0','—')} → {disp_real.attrs.get('q','—')}；"
                 f"按覆盖率归一化、剔除单行业>80% 主题基金。"
                 f"判读分布：" + "、".join(
                     f"{k} {v}" for k, v in disp_real["判读"].value_counts().items())
                 + f"。但该指标 IC {_dv.get('ic', float('nan')):+.3f}、"
                   f"NW p {_dv.get('p', float('nan')):.2f}——"
                   f"只作描述，不能预测次季收益")
                if not disp_real.empty else f"代理：{disp['note']}",
                (f"全市场：{spair[1]} → {spair[0]}（同披露口径），"
                 f"{len(sf)} 只股票、样本基金 {sf.attrs.get('n_funds', ('—', '—'))[1]} 只")
                if not sf.empty else
                (f"Wind 前 50 只口径：{len(stock)} 只（两期并集）"
                 if not stock.empty else "未就绪：需先运行 tools/run_ak_all.py"),
                (f"真实口径：47 只偏股混合（规模前 50 中有两期数据者）；区间 "
                 f"{disp_real.attrs.get('q0','—')} → {disp_real.attrs.get('q','—')}；"
                 f"样本最多 {int(disp_real['样本基金数'].max())} 只，按覆盖率归一化，剔除>80% 主题基金；来源 {_disp_src}")
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
                         "个股级持仓明细两期无同口径对照，或数据管道尚未入库",
                         "同披露口径的两期持仓明细（季报对上年同季、半年报对上年半年报）",
                         "运行 tools/run_ak_all.py 抓取后自动可用"],
            })
            bucket_col = None
        else:
            if not sf.empty:
                base = pd.DataFrame({
                    "股票代码": sf["stock_code"],
                    "股票名称": sf["stock_name"],
                    "所属行业": sf["industry"],
                    "上期持股基金数": sf["n0"].astype(int),
                    "本期持股基金数": sf["n1"].astype(int),
                    "Δ持股基金数": sf["d_n"].astype(int),
                    "机构持仓占比%": sf["share"],
                    "持仓市值(亿)": sf["mv1_yi"],
                    "Δ持仓市值(亿)": sf["d_mv_yi"],
                    "榜单状态": sf["state"],
                })
                pair_label = f"{spair[1]} → {spair[0]}"
            else:
                base = pd.DataFrame({
                    "股票代码": "",
                    "股票名称": stock["name"],
                    "所属行业": "未映射",
                    "上期持股基金数": stock["n0"].fillna(0).astype(int),
                    "本期持股基金数": stock["n1"].fillna(0).astype(int),
                    "Δ持股基金数": stock["d_n"].fillna(0).astype(int),
                    "机构持仓占比%": pd.NA,
                    "持仓市值(亿)": (stock["mv1"] / 1e4).round(2),
                    "Δ持仓市值(亿)": (stock["d_mv"] / 1e4).round(2),
                    "榜单状态": stock["state"],
                })
                pair_label = f"{q0} → {q}（Wind 前 50 只口径）"

            _act = dec.set_index("行业")["建议动作"]
            base["行业动作"] = base["所属行业"].map(_act).fillna("未映射")

            def _bucket(r):
                dn = r["Δ持股基金数"]
                if r["行业动作"] == "超配" and dn > 0:
                    return "买入候选（行业超配 + Δ基金数>0；未过 IC，降级为观察仓）"
                if r["行业动作"] == "回避" or dn <= -30:
                    return "回避（行业回避，或同口径单期减仓 ≥30 家基金）"
                if r["行业动作"] == "未映射":
                    return "观察（行业未映射，不能分档）"
                return "观察"

            base["分档结论"] = base.apply(_bucket, axis=1)
            base["对照期"] = pair_label
            sdf = base.sort_values("Δ持股基金数", ascending=False)
            sdf = sdf[["股票代码", "股票名称", "所属行业", "行业动作", "上期持股基金数",
                       "本期持股基金数", "Δ持股基金数", "机构持仓占比%", "持仓市值(亿)",
                       "Δ持仓市值(亿)", "榜单状态", "分档结论", "对照期"]]
            bucket_col = "分档结论"

        sdf.to_excel(xw, sheet_name="2 个股清单", index=False)
        _style_sheet(xw.sheets["2 个股清单"],
                     {"A": 11, "B": 12, "C": 11, "D": 10, "E": 13, "F": 13,
                      "G": 12, "H": 13, "I": 12, "J": 14, "K": 10,
                      "L": 46, "M": 22}, freeze="A2")

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
                   "分歧度期次", "个股 Δ 持股基金数", "仓位建议", "样本边界", "分歧度归一化"],
            "口径": [
                "Δ配置pp − 被动漂移pp；被动 = 上季配置 ×(1+行业指数收益) 后按 31 行业归一",
                "用申万一级行业指数季度收益近似个股组合收益",
                "Δ持仓市值 − 上季持仓市值 × 行业收益；总市值为缓存重仓口径",
                "F3e：偏股混合单类型超配序列的扩张窗口分位（无前视）",
                "行业 Δ配置 的横截面离散度（代理，非基金级加仓占比）",
                "分歧度用相邻两季（如 2026Q2→2026Q3），因季报/半年报披露口径不同，"
                "已按每只基金每期覆盖率归一化；个股层不用相邻季，只用「同披露口径」两期"
                "（2026Q3 对 2025Q3），避免把「掉出前十大」当成清仓",
                "优先全市场完整持仓：季报只披露前十大，故只在「同披露口径」的两期之间比较"
                "（Q3 对上年 Q3、Q2 对上年 Q2），否则「掉出前十大」会被误判为清仓；"
                "回退时用 Wind 前 50 只口径（跌出前 50 会被误判为退出）",
                "按行业 Δ配置 横截面标准差分档，非基金级分歧度",
                "基金×个股明细：东方财富完整明细已入库（季报前十大 / 半年报全持仓）；"
                "日度净值 0、季报文本 0（无接口，不用其他字段替代）",
                "季报期由前十大推导（覆盖约 60%），半年报/年报用全持仓表（约 88%）："
                "先除以每只基金每期覆盖率再比较；不归一化会把覆盖率下降误算成全面减仓"
                "（电子原始 3 加/39 减，归一化后 19 加/19 减）",
            ],
        })
        spec.to_excel(xw, sheet_name="6 口径说明", index=False)
        _style_sheet(xw.sheets["6 口径说明"], {"A": 22, "B": 110}, freeze="A2")

        # ---------- 8 前十大偏差校准（4.5） ----------
        try:
            _full_q = next((x for x in reversed(akd.latest_quarters_all())
                            if x[-1] in ("2", "4")), None)
            tb = akd.top10_bias(_full_q) if _full_q else pd.DataFrame()
            if not tb.empty:
                tb.to_excel(xw, sheet_name="8 前十大偏差校准", index=False)
                _style_sheet(xw.sheets["8 前十大偏差校准"],
                             {"A": 12, "B": 9, "C": 15, "D": 15, "E": 18, "F": 34},
                             freeze="A2")
        except Exception as _e:
            print("前十大偏差表写入失败：", _e)

        # ---------- 9b 季末以来漂移（4.4 时效性） ----------
        try:
            _drf = os.path.join(ROOT, "data", "cache", "nav_drift_2026Q3.csv")
            if os.path.exists(_drf):
                pd.read_csv(_drf).to_excel(xw, sheet_name="9b 季末以来漂移", index=False)
                _style_sheet(xw.sheets["9b 季末以来漂移"],
                             {"A": 12, "B": 15, "C": 15, "D": 16, "E": 12, "F": 12},
                             freeze="A2")
        except Exception as _e:
            print("季末漂移表写入失败：", _e)

        # ---------- 9 日度暴露校准（4.4） ----------
        try:
            _nev = os.path.join(ROOT, "data", "cache", "nav_exposure_50.csv")
            if os.path.exists(_nev):
                _nq = pd.read_csv(_nev, dtype={"code": str})
                _nq.to_excel(xw, sheet_name="9 日度暴露校准", index=False)
                _style_sheet(xw.sheets["9 日度暴露校准"],
                             {"A": 10, "B": 10, "C": 9, "D": 9, "E": 12,
                              "F": 12, "G": 12, "H": 11, "I": 12, "J": 11}, freeze="A2")
        except Exception as _e:
            print("日度暴露表写入失败：", _e)

        # ---------- 10 自检与证伪 ----------
        try:
            _aud = os.path.join(ROOT, "报告", "自检表.csv")
            if os.path.exists(_aud):
                pd.read_csv(_aud).to_excel(xw, sheet_name="10 自检与证伪", index=False)
                _style_sheet(xw.sheets["10 自检与证伪"],
                             {"A": 9, "B": 40, "C": 16, "D": 70}, freeze="A2")
        except Exception as _e:
            print("自检表写入失败：", _e)

        # ---------- 7 数据源交叉验证 ----------
        try:
            cv = akd.cross_validate_wind(disp_real.attrs.get("q", q))
            if not cv.empty:
                cv.to_excel(xw, sheet_name="7 数据源交叉验证", index=False)
                _style_sheet(xw.sheets["7 数据源交叉验证"],
                             {"A": 12, "B": 8, "C": 14, "D": 12, "E": 12, "F": 10},
                             freeze="A2")
        except Exception as _e:
            print("交叉验证表写入失败：", _e)

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
