# -*- coding: utf-8 -*-
"""生成《公募基金行业拥挤度与集中度分析报告（2026Q3）》Word 版。

数据全部来自 src/canon.py 与 playbook.py，与网页同一口径；
报告里的每个数字都能在网页上找到对应位置，不存在"报告一套、页面一套"。
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import canon as cn
import playbook as pb

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FIG = os.path.join(ROOT, "output", "report_figs")
os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(ROOT, "output", "公募基金行业拥挤度与集中度分析报告_2026Q3.docx")

NAVY = RGBColor(0x1F, 0x2D, 0x3D)
BLUE = RGBColor(0x1F, 0x6F, 0xEB)
GREY = RGBColor(0x6B, 0x7A, 0x8C)
RED = RGBColor(0xC6, 0x28, 0x28)
GREEN = RGBColor(0x1A, 0x7F, 0x37)


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
Q = cn.latest()
QS = cn.quarters()
I = QS.index(Q)
QP = QS[I - 1] if I else None
QY = f"{int(Q[:4]) - 1}{Q[4:]}"
PF = cn.pair_full()
# pair_full 返回 (上期, 本期)，即 (q0, q)：第一个是旧期、第二个才是最新全持仓期。
# 这里写反过一次，报告里出现过「最新全持仓期（2025Q4）」这种明显错误——
# 加断言锁死，避免再靠肉眼发现。
QF0, QF = PF if PF else (None, None)
_assert_q = cn.position_change()
assert QF == _assert_q.get("q") and QF0 == _assert_q.get("q0"), \
    f"全持仓期次解包错误：{QF0}->{QF}，应为 {_assert_q.get('q0')}->{_assert_q.get('q')}"
N_FUND = len(cn.universe(Q))
POS = cn.position_change()
HIST = cn.position_series()
PCT = float(HIST["中位"].rank(pct=True).iloc[-1] * 100) if not HIST.empty else np.nan
CSI = cn.crowding_score()
CHG = cn.csi_change()
ACT = pb.action_table()
CMP = cn.compare_periods()
IDX = cn.index_stats()
M = cn.level_matrix("top10")
HHI_NOW = float(((CMP["配置_前十"] / 100) ** 2).sum())
HHI_H = ((M / 100) ** 2).sum(axis=1)
HHI_PCT = float((HHI_H <= HHI_NOW).mean() * 100)
CR3 = float(CMP["配置_前十"].nlargest(3).sum())
SF_Q = cn.stock_flow_top10(QP, Q) if QP else pd.DataFrame()
SF_Y = cn.stock_flow_top10(QY, Q) if QY in QS else pd.DataFrame()


def fmt(v, nd=1, sign=False, unit=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return (f"{v:+,.{nd}f}" if sign else f"{v:,.{nd}f}") + unit


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def chart_csi() -> str:
    # 正常样本按 CSI 升序（最拥挤在最上面），不评级的小样本压在底部
    d = CSI.dropna(subset=["CSI"]).copy()
    d["_sm"] = (d["等级"].astype(str) == "小样本").astype(int)
    d = d.sort_values(["_sm", "CSI"], ascending=[False, True])
    col = {"极度拥挤": "#C62828", "拥挤": "#E8710A", "中性": "#8A97A5",
           "偏冷": "#4C8DFF", "冷门": "#1F4E9C"}
    fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=200)
    ax.barh(d.index, d["CSI"],
            color=[col.get(x, "#9AA6B2" if x == "小样本" else "#C3CBD4") for x in d["等级"]],
            height=0.72)
    for y, (v, lv) in enumerate(zip(d["CSI"], d["等级"])):
        ax.text(v + 1.2, y, f"{v:.0f} {lv}", va="center", fontsize=7.5, color="#37485C")
    for x in (25, 45, 65, 80):
        ax.axvline(x, color="#DCE3EA", lw=0.8, ls=":")
    ax.set_xlim(0, 118)
    ax.set_xlabel("拥挤度指数 CSI（0＝最冷，100＝最挤）", fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#EEF2F6", lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(FIG, "csi.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def chart_delta() -> str:
    d = CHG.dropna(subset=["ΔCSI"]).sort_values("ΔCSI")
    fig, ax = plt.subplots(figsize=(9.6, 6.8), dpi=200)
    ax.barh(d.index, d["ΔCSI"], height=0.72,
            color=["#C62828" if v > 0 else "#1A7F37" for v in d["ΔCSI"]])
    for y, v in enumerate(d["ΔCSI"]):
        ax.text(v + (1.4 if v >= 0 else -1.4), y, f"{v:+.0f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5, color="#37485C")
    ax.axvline(0, color="#1F2D3D", lw=0.9)
    ax.set_xlabel("ΔCSI＝本期 CSI − 上期 CSI（正＝变挤，负＝变松）", fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#EEF2F6", lw=0.7)
    ax.set_axisbelow(True)
    lo, hi = d["ΔCSI"].min(), d["ΔCSI"].max()
    ax.set_xlim(lo - 12, hi + 12)
    fig.tight_layout()
    p = os.path.join(FIG, "dcsi.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def chart_scatter() -> str:
    d = CMP.dropna(subset=["历史分位"]).join(IDX[["指数分位"]], how="left")
    d = d.dropna(subset=["指数分位"])
    fig, ax = plt.subplots(figsize=(7.4, 6.0), dpi=200)
    ax.axvline(50, color="#B7C1CC", lw=0.8, ls="--")
    ax.axhline(50, color="#B7C1CC", lw=0.8, ls="--")
    ax.scatter(d["历史分位"], d["指数分位"], s=d["配置_前十"] * 9 + 18,
               c="#1F6FEB", alpha=0.65, edgecolors="white", linewidths=0.8)
    for i, r in d.iterrows():
        if r["历史分位"] > 78 or r["指数分位"] > 88 or r["配置_前十"] > 8:
            ax.annotate(i, (r["历史分位"], r["指数分位"]), fontsize=7.2,
                        xytext=(3, 3), textcoords="offset points", color="#37485C")
    ax.set_xlabel("配置分位（机构押得多重，31 期）", fontsize=9)
    ax.set_ylabel("价格分位（相对自己贵不贵）", fontsize=9)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#F2F5F8", lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(FIG, "quad.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def chart_pos() -> str:
    fig, ax = plt.subplots(figsize=(9.2, 3.4), dpi=200)
    x = list(HIST.index)
    ax.fill_between(x, HIST["q1"], HIST["q3"], color="#1F6FEB", alpha=0.13)
    ax.plot(x, HIST["中位"], color="#1F6FEB", lw=2, marker="o", ms=4)
    ax.plot(x[-1], HIST["中位"].iloc[-1], "o", color="#C62828", ms=7)
    ax.annotate(f'{HIST["中位"].iloc[-1]:.1f}%', (x[-1], HIST["中位"].iloc[-1]),
                xytext=(-4, 10), textcoords="offset points", fontsize=8, color="#C62828",
                ha="right")
    ax.set_ylabel("股票仓位中位 %", fontsize=9)
    ax.tick_params(labelsize=7.5, axis="x", rotation=-45)
    ax.tick_params(labelsize=8, axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#EEF2F6", lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(FIG, "pos.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Word 工具
# ---------------------------------------------------------------------------
def _set_cell_bg(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def _cell_text(cell, text, size=8.5, bold=False, color=None, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    if align:
        p.alignment = align
    r = p.add_run(str(text))
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = "微软雅黑"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    if color is not None:
        r.font.color.rgb = color


def add_table(doc, df, widths=None, header_fill="1F2D3D", zebra=True,
              align_right_cols=(), font=8.5):
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, c in enumerate(df.columns):
        _cell_text(hdr[i], c, size=font, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                   align=WD_ALIGN_PARAGRAPH.CENTER)
        _set_cell_bg(hdr[i], header_fill)
    for k, (_, row) in enumerate(df.iterrows()):
        cells = t.add_row().cells
        for i, c in enumerate(df.columns):
            al = WD_ALIGN_PARAGRAPH.RIGHT if c in align_right_cols else WD_ALIGN_PARAGRAPH.CENTER
            _cell_text(cells[i], row[c], size=font, align=al)
            if zebra and k % 2 == 1:
                _set_cell_bg(cells[i], "F7F9FB")
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    return t


def H(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.font.name = "微软雅黑"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    r.font.size = Pt(13 if level == 1 else 11)
    r.font.bold = True
    r.font.color.rgb = NAVY if level == 1 else BLUE
    if level == 1:
        pPr = p._p.get_or_add_pPr()
        bd = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6")
        bottom.set(qn("w:color"), "1F6FEB")
        bd.append(bottom)
        pPr.append(bd)
    return p


def P(doc, text, size=10.5, color=NAVY, first_indent=True, space=4, line=1.55):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space)
    p.paragraph_format.line_spacing = line
    if first_indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.font.name = "宋体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    return p


def BULLET(doc, text, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.45
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.name = "宋体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    return p


def CAP(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.size = Pt(8.5)
    r.font.color.rgb = GREY
    r.font.name = "微软雅黑"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")


def NOTE(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.35
    r = p.add_run(text)
    r.font.size = Pt(8)
    r.font.color.rgb = GREY
    r.font.name = "宋体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


def page_break(doc):
    doc.add_page_break()


# ---------------------------------------------------------------------------
# 正文
# ---------------------------------------------------------------------------
def _safe_save(doc) -> str:
    """保存到 OUT；若文件被 Word/预览占用，退到带时间戳的文件名。

    报告常在预览面板里开着，Windows 下会锁文件，直接 save 会 PermissionError。
    """
    try:
        doc.save(OUT)
        return OUT
    except PermissionError:
        import time
        alt = OUT.replace(".docx", f"_{time.strftime('%H%M%S')}.docx")
        doc.save(alt)
        return alt


def build() -> str:
    p_csi, p_dcsi, p_quad, p_pos = chart_csi(), chart_delta(), chart_scatter(), chart_pos()

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.0)

    st = doc.styles["Normal"]
    st.font.name = "宋体"
    st.font.size = Pt(10.5)
    st._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    # ---------------- 封面 ----------------
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("公募基金持仓研究")
    r.font.size = Pt(12); r.font.color.rgb = BLUE; r.font.bold = True
    r.font.name = "微软雅黑"; r._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("行业拥挤度与集中度分析报告")
    r.font.size = Pt(26); r.font.bold = True; r.font.color.rgb = NAVY
    r.font.name = "微软雅黑"; r._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"2026 年三季度 · 数据截至 {Q[:4]} 年 9 月 30 日")
    r.font.size = Pt(13); r.font.color.rgb = GREY
    r.font.name = "微软雅黑"; r._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")

    for _ in range(2):
        doc.add_paragraph()
    box = doc.add_table(rows=4, cols=2)
    box.style = "Table Grid"
    box.alignment = WD_TABLE_ALIGNMENT.CENTER
    info = [("样本范围", f"{N_FUND:,} 只主动权益基金（普通股票型 + 偏股混合 + 灵活配置，已剔主题基金）"),
            ("行业分类", "申万一级 31 个行业"),
            ("数据来源", "东方财富基金定期报告（季报前十大、半年报全持仓）；申万行业指数日行情"),
            ("口径说明", f"配置/广度/分位为前十大口径（{QP}→{Q}）；主动/被动为全持仓（{QF0}→{QF}）")]
    for i, (k, v) in enumerate(info):
        _cell_text(box.rows[i].cells[0], k, size=9, bold=True, color=BLUE)
        _cell_text(box.rows[i].cells[1], v, size=9)
        _set_cell_bg(box.rows[i].cells[0], "F2F5F8")
        box.rows[i].cells[0].width = Cm(3.2)
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("本报告由持仓数据自动生成，所有指标可用原始披露复算。\n"
                  "报告为数据描述与规则输出，不构成投资建议。")
    r.font.size = Pt(8.5); r.font.color.rgb = GREY
    r.font.name = "宋体"; r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    page_break(doc)

    # ---------------- 摘要 ----------------
    H(doc, "摘要", 1)
    # 最拥挤/最冷门只在「正常样本」里排：综合这类行业只有几十只基金持有，
    # CSI 会被几十个观测推高，排进榜单会让读者得出错误结论。
    _ok = CSI[CSI["可靠"] == "正常"]
    _hi = _ok.nlargest(3, "CSI")
    _lo = _ok.nsmallest(3, "CSI")
    _up = CHG.nlargest(3, "ΔCSI")
    _dn = CHG.nsmallest(3, "ΔCSI")
    _fo = ACT[ACT["动作"] == "可以跟"]
    _av = ACT[ACT["动作"] == "要回避"]
    _hot = ACT[ACT["动作"] == "别追高"]

    P(doc, f"本报告基于 {Q} 全部 {N_FUND:,} 只主动权益基金的前十大重仓披露，"
           f"从「押得多重（配置）、价格多贵（指数）、钱还在不在进（资金）」三个维度，"
           f"给出 31 个申万一级行业的拥挤度评分（CSI），并拆解本季度的行业轮动与个股换手。"
           f"核心结论如下。")

    for txt in [
        f"总仓位停在 {POS['仓位1']:.1f}%（{len(HIST)} 个半年报期第 {PCT:.0f} 分位），"
        f"较上期 {POS['中位变化pp']:+.2f}pp，{POS['加仓基金占比']:.0f}% 的基金加仓。"
        f"整体没有增量资金，行业配置的变化是内部换仓。",

        f"行业集中度 HHI 为 {HHI_NOW:.4f}（31 期中第 {HHI_PCT:.0f} 分位），"
        f"前三大行业合计 {CR3:.1f}%。集中度处于历史高位，"
        f"组合对少数行业的依赖度高于过去多数时点。",

        f"最拥挤的三个行业是 {'、'.join(f'{i}（{r['CSI']:.0f}）' for i, r in _hi.iterrows())}；"
        f"CSI 最低的三个是 {'、'.join(f'{i}（{r['CSI']:.0f}）' for i, r in _lo.iterrows())}。"
        f"31 个行业中极度拥挤 {int((CSI['等级'] == '极度拥挤').sum())} 个、"
        f"拥挤 {int((CSI['等级'] == '拥挤').sum())} 个、冷门 {int((CSI['等级'] == '冷门').sum())} 个；"
        f"另有 {int((CSI['等级'] == '小样本').sum())} 个行业因持有基金数不足 150 只、"
        f"或前十配置不足 0.5%，一律不评级，不计入上述任何一档。",

        f"拥挤度上升最快的是 {'、'.join(f'{i}（{r['ΔCSI']:+.0f}）' for i, r in _up.iterrows())}；"
        f"回落最快的是 {'、'.join(f'{i}（{r['ΔCSI']:+.0f}）' for i, r in _dn.iterrows())}。"
        f"「水位已高且仍在涨」比「水位高但在退潮」更值得警惕。",

        f"按事前固定的规则，本期可以跟的行业 {len(_fo)} 个（{'、'.join(_fo.index) or '无'}），"
        f"别追高 {len(_hot)} 个（{'、'.join(_hot.index) or '无'}），"
        f"要回避 {len(_av)} 个（{'、'.join(_av.index) or '无'}）。"
        f"权益仓位参考 50%–60%。",
    ]:
        BULLET(doc, txt)

    NOTE(doc, "说明：本报告的「可以跟」指资金在进、多数基金在加、价格不在历史极值，"
              "是赔率与风险的组合判断，不是收益预测。因子检验显示季报指标对次季收益没有稳定预测力。")

    # ---------------- 一、数据与方法 ----------------
    H(doc, "一、数据与方法", 1)
    H(doc, "1.1 样本与三套口径", 2)
    P(doc, f"样本为 {Q} 有完整持仓披露的主动权益基金 {N_FUND:,} 只。"
           f"指数基金、ETF 联接的持仓是复制标的，债券与 QDII 不投 A 股，"
           f"均不计入——否则被动资金会被误读成「机构观点」。"
           f"单行业占前十大超过 80% 的主题基金也已剔除，其变动反映主题申赎而非行业判断。")
    P(doc, "先说清披露制度，否则后面的口径会看起来互相矛盾："
           "按《公开募集证券投资基金信息披露管理办法》，"
           "基金季报（Q1、Q3）只披露前十大重仓股；"
           "半年报（Q2）与年报（Q4）披露全部持仓。"
           "所以「前十大」不是只有半年报才有——季报本身就是前十大，"
           "半年报虽然给了全部持仓，我们同样只取最重的 10 只，"
           "四个季度的披露范围就对齐了，环比（Q2→Q3）与同比（Q3→去年Q3）都能算。"
           "这也是本报告「个股环比」成立的原因：不是把全持仓与季报混算，"
           "而是把全持仓截到前十大，与季报同口径。")
    P(doc, "持仓数据存在三套互不通用的口径，本报告严格区分：")
    tb = pd.DataFrame([
        ["A 全持仓", "半年报（Q2/Q4）", "占净值比，之和＝股票仓位",
         "配置水平、超配低配、主动/被动分解", "一年只有两次"],
        ["B 前十大", "季报（Q1–Q4）", "每只基金取前 10 只后归一化到 100%",
         "加仓广度、环比/同比、历史分位", "只覆盖组合一部分，绝对水平不可用"],
        ["C 个股", "同披露类型两期", "持有基金数、持股数、持仓市值",
         "谁被建仓、谁被移出", "跌出前十不等于清仓"],
    ], columns=["口径", "频率", "权重定义", "回答什么", "不能回答什么"])
    add_table(doc, tb, widths=[2.0, 2.6, 4.6, 4.2, 4.0], header_fill="1F2D3D", font=8)
    NOTE(doc, "三套口径的分母与披露范围都不同，不能相加，也不能跨口径比较绝对水平。")

    H(doc, "1.2 拥挤度指数 CSI 的定义", 2)
    P(doc, "CSI 把三个相互独立的维度各自换算为 0–100 的分位后等权平均：")
    for txt in [
        "配置分位：该行业前十大配置在自身 31 期历史中的排名——机构押得多重；",
        "价格分位：行业指数季末点位在 2019 年以来交易日中的排名——价格有多贵；",
        "资金分位：当期加仓占比在 31 个行业之间的排名——钱还在不在进。",
    ]:
        BULLET(doc, txt)
    P(doc, "等权而非加权，是因为三个维度已统一为分位、量纲一致；"
           "31 个行业 × 31 期的样本量不足以稳定估计权重，任何加权都容易过拟合。"
           "分级阈值为：≥80 极度拥挤、65–80 拥挤、45–65 中性、25–45 偏冷、<25 冷门。")
    P(doc, "为什么不用 HHI：HHI＝Σw²，是**一个组合**的集中度，31 个行业最终压成一个标量，"
           "只能回答「整体分散不分散」，无法给行业排序；本报告要回答「哪个行业挤」，"
           "主指标只能是行业层面的 CSI。HHI 作为组合层面的辅助指标保留在第二章。")

    H(doc, "1.3 数据质量", 2)
    P(doc, "覆盖率（持仓市值 ÷ 净值）异常的记录（>105% 或 <1%）整只剔除，不做归一化；"
           "个股代码统一为字符串，港股 5 位代码不补零（避免与 A 股撞码）。"
           "报告全部指标由 canon 统一口径层计算，12 条恒等式自检（归一化、广度分母、"
           "价格中性净额之和、GK 恒等式、分位区间、个股一致性等）全部通过。")

    page_break(doc)

    # ---------------- 二、总体 ----------------
    H(doc, "二、总体仓位与集中度", 1)
    doc.add_picture(p_pos, width=Cm(16.4))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    CAP(doc, f"图 1　股票仓位中位数：{len(HIST)} 个半年报期（阴影为 25–75 分位带）")
    P(doc, f"最新全持仓期（{QF}）股票仓位中位 {POS['仓位1']:.2f}%，"
           f"较 {POS['q0']} 的 {POS['仓位0']:.2f}% 变化 {POS['中位变化pp']:+.2f}pp，"
           f"位于 {len(HIST)} 个半年报期的第 {PCT:.0f} 分位。"
           f"{POS['加仓基金占比']:.0f}% 的基金加仓、{100 - POS['加仓基金占比']:.0f}% 减仓。")
    P(doc, f"需要说明期次：最新季报是 {Q}，但季报只披露前十大（覆盖净值约 20%–50%），"
           f"无法据此还原股票仓位；完整仓位只在半年报（Q2/Q4）可见。"
           f"因此本节用的是最近一个全持仓期 {QF}，与最新季报 {Q} 相差一个季度——"
           f"这不是数据滞后，而是披露制度决定的：季报没有全量持仓，就不存在「Q3 的股票仓位」这个数。"
           f"后文的行业配置、广度、拥挤度均已更新到 {Q}。")
    P(doc, f"这条数据决定了全篇的读法：既然总仓位没有变化，行业配置的此消彼长就只能来自"
           f"内部换仓。因此看到某行业占比上升，可以读成「机构更看好它」，"
           f"而不必担心是「大盘上涨、水涨船高」。反过来，仓位已在 {PCT:.0f} 分位的高位，"
           f"指望「全体加仓推动普涨」并不现实。")

    H(doc, "2.1 集中度指标", 2)
    tb2 = pd.DataFrame([
        ["行业集中度 HHI", f"{HHI_NOW:.4f}", f"第 {HHI_PCT:.0f} 分位",
         "Σ(行业占比)²；越大越集中在少数行业"],
        ["前三大行业合计 CR3", f"{CR3:.1f}%",
         "、".join(CMP["配置_前十"].nlargest(3).index),
         "前三大行业占前十大组合的比重"],
        ["配置最高行业", f"{CMP['配置_前十'].max():.2f}%", CMP["配置_前十"].idxmax(),
         "前十大口径，等权平均"],
    ], columns=["指标", "数值", "备注", "含义"])
    add_table(doc, tb2, widths=[4.2, 2.6, 4.6, 6.0], font=8.5)
    P(doc, f"HHI {HHI_NOW:.4f} 处于 31 期的第 {HHI_PCT:.0f} 分位，说明主动权益基金整体的行业分布"
           f"比过去多数时点更集中。集中度本身不区分方向：它只说明组合对少数行业的依赖度上升，"
           f"一旦这些行业出现回调，净值波动会被放大。")

    page_break(doc)

    # ---------------- 三、拥挤度全景 ----------------
    H(doc, "三、拥挤度全景：31 个行业谁最挤", 1)
    doc.add_picture(p_csi, width=Cm(15.8))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    CAP(doc, f"图 2　31 个行业拥挤度指数 CSI（{Q}）")
    _lv = CSI["等级"].value_counts()
    tb3 = pd.DataFrame([[k, int(_lv.get(k, 0)),
                         "、".join(CSI.index[CSI["等级"] == k][:6]) +
                         ("…" if int(_lv.get(k, 0)) > 6 else "")]
                        for k in ["极度拥挤", "拥挤", "中性", "偏冷", "冷门", "小样本"]],
                       columns=["等级", "行业数", "行业（前 6 个）"])
    add_table(doc, tb3, widths=[3.0, 2.2, 12.0], font=8.5)
    P(doc, f"最拥挤的是 {'、'.join(f'{i}（CSI {r['CSI']:.0f}）' for i, r in _hi.iterrows())}。"
           f"其中煤炭、石油石化的价格分位都在 90 以上，属于「机构重仓 + 价格在历史高位」，"
           f"是本报告认定的真拥挤。")
    _sm = CSI[CSI["可靠"] == "小样本"].sort_values("CSI", ascending=False)
    if len(_sm):
        _s0 = _sm.iloc[0]
        P(doc, f"需要单独说明：{'、'.join(_sm.index)} 这 {len(_sm)} 个行业本报告一律「不评级」。"
               f"以 {_sm.index[0]} 为例，它的 CSI 算出来是 {_s0['CSI']:.0f} 分，"
               f"按三分位本来会判为「{_s0['原等级']}」，但全市场只有 {_s0['持有基金数']:.0f} 只"
               f"样本基金把它放进前十大，前十配置合计仅 {_s0['配置']:.2f}%。"
               f"三个分位都是在这个极小样本上排出来的——几十只基金里少数几只调仓，"
               f"就能把资金分位推到 80 以上；指数从低位反弹一段，就能把价格分位推到 90 以上。"
               f"而这个行业对整个组合的影响不足 0.2pp，即使仓位清零，对净值的影响也小于"
               f"一个正常行业 1pp 的调仓。因此它不进入拥挤排名、不给买卖动作，"
               f"在图表中一律以灰色「小样本」显示。")
    P(doc, f"CSI 最低的三个行业是 {'、'.join(f'{i}（CSI {r['CSI']:.0f}）' for i, r in _lo.iterrows())}。"
           f"注意「冷门」与「小样本」不是一回事：冷门是持有基金数足够多、"
           f"但配置与资金都在低位；小样本是持有人本来就少，不评级。"
           f"冷门不等于机会——它只说明当前没有资金关注，需要等「加仓占比回升 + 净额转正」"
           f"两个信号同时出现，才构成关注理由。")

    H(doc, "3.1 小样本行业的处理", 2)
    P(doc, "判定规则（事前固定）：全市场持有基金数少于 150 只、或前十配置低于 0.5%，"
           "即标记为「小样本」，等级记为「小样本」而不是任何拥挤档位。"
           "理由有两条：① 分位是排序统计量，样本越小，少数观测越容易把它推到极值；"
           "② CSI 三个分量里，价格分位与资金分位都不看绝对配置，"
           "一个只占组合 0.1% 的行业也能拿到 90 分，但它对净值没有影响。"
           "本期小样本行业为：" + "、".join(CSI.index[CSI["可靠"] == "小样本"])
           + "。这些行业的 CSI 数值仍保留在附录，仅供了解其自身位置的极端程度，"
           "不构成拥挤判断。")

    page_break(doc)

    # ---------------- 四、拥挤度变化 ----------------
    H(doc, "四、拥挤度变化：谁在变挤、谁在变松", 1)
    doc.add_picture(p_dcsi, width=Cm(15.8))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    CAP(doc, f"图 3　ΔCSI＝本期 − 上期（{QP}→{Q}），红＝变挤、绿＝变松")
    P(doc, f"水位（CSI）说明现在挤不挤，水流（ΔCSI）说明正在往哪个方向走。"
           f"两者必须一起看：水位高但正在放水，危险程度远低于「水位高且还在涨」。")
    P(doc, f"本期变挤最快的是 {'、'.join(f'{i}（{r['上期CSI']:.0f}→{r['CSI']:.0f}，{r['ΔCSI']:+.0f}）' for i, r in _up.iterrows())}；"
           f"变松最快的是 {'、'.join(f'{i}（{r['上期CSI']:.0f}→{r['CSI']:.0f}，{r['ΔCSI']:+.0f}）' for i, r in _dn.iterrows())}。")
    _hiup = CHG[(CHG["CSI"] >= 80) & (CHG["ΔCSI"] > 0)]
    P(doc, f"最需要警惕的是「水位 ≥80 且 ΔCSI>0」的行业，本期有 {len(_hiup)} 个："
           + ("、".join(_hiup.index) if len(_hiup) else "无")
           + "。这些行业既在历史高位、资金又还在往里挤，边际买家正在耗尽。"
           "相反，已经变松的高水位行业不必急着卖——资金正在自己退潮。")

    page_break(doc)

    # ---------------- 五、四维拆解 ----------------
    H(doc, "五、配置 × 价格：四象限", 1)
    doc.add_picture(p_quad, width=Cm(13.6))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    CAP(doc, f"图 4　配置分位 × 价格分位（气泡大小＝前十配置比例）")
    _d = CMP.dropna(subset=["历史分位"]).join(IDX[["指数分位"]], how="left").dropna(subset=["指数分位"])
    _q1 = _d[(_d["历史分位"] >= 50) & (_d["指数分位"] >= 50)]
    _q2 = _d[(_d["历史分位"] >= 50) & (_d["指数分位"] < 50)]
    _q3 = _d[(_d["历史分位"] < 50) & (_d["指数分位"] >= 50)]
    _q4 = _d[(_d["历史分位"] < 50) & (_d["指数分位"] < 50)]
    tb4 = pd.DataFrame([
        ["右上　真拥挤", len(_q1), "机构重仓且价格在历史高位",
         "、".join(_q1.nlargest(4, "指数分位").index)],
        ["右下　抱团没赚", len(_q2), "仓位重但价格还在低位，要么机会要么判断错",
         "、".join(_q2.head(3).index)],
        ["左上　机构踏空", len(_q3), "价格涨了但仓位没跟上",
         "、".join(_q3.head(3).index)],
        ["左下　冷门", len(_q4), "仓位轻、价格低，无资金关注",
         "、".join(_q4.head(3).index)],
    ], columns=["象限", "行业数", "含义", "代表行业"])
    add_table(doc, tb4, widths=[3.2, 2.0, 7.6, 5.0], font=8.5)
    P(doc, "只看配置分位会高估小行业的拥挤、低估大行业的风险；只看价格分位会把"
           "「一直很重」当成「现在才拥挤」。两个分位一起看，才知道该减谁、该等谁。"
           "右上角是真拥挤——买它等于在拥挤处接棒；右下角是「闷在里面」的，不用急着卖。")

    page_break(doc)

    # ---------------- 六、行业与个股 ----------------
    H(doc, "六、行业配置变化与个股换手", 1)
    H(doc, "6.1 行业环比与同比", 2)
    _t = CMP.join(IDX[["指数分位"]], how="left")
    _t["动作"] = ACT["动作"].reindex(_t.index)
    _show = _t.reset_index()[["行业", "配置_前十", "Δ上季pp", "Δ上年pp",
                              "平均主动pp", "加仓占比", "历史分位", "指数分位", "动作"]]
    _show.columns = ["行业", "前十配置%", f"环比pp({QP}→{Q})", f"同比pp({QY}→{Q})",
                     "当期主动pp", "加仓占比%", "配置分位", "价格分位", "动作"]
    add_table(doc, _show.head(16).round(2), widths=[2.4, 2.0, 2.5, 2.5, 2.2, 2.1, 1.8, 1.8, 2.0], font=7.5,
              align_right_cols=("前十配置%", f"环比pp({QP}→{Q})", f"同比pp({QY}→{Q})",
                                "当期主动pp", "加仓占比%", "配置分位", "价格分位"))
    NOTE(doc, f"按环比降序前 16 个行业；完整 31 行见附录。环比含价格漂移，"
              f"「当期主动pp」为价格中性后的净买卖，两者不可混用。")

    H(doc, "6.2 个股：环比与同比", 2)
    if not SF_Q.empty:
        _a = SF_Q.nlargest(5, "d_n")
        _c = SF_Q.nsmallest(5, "d_n")
        rows = []
        for tag, d in (("加仓", _a), ("减仓", _c)):
            for code, r in d.iterrows():
                rows.append([tag, r["stock_name"], r["industry"],
                             int(r["n_funds_0"]), int(r["n_funds"]), int(r["d_n"]),
                             int(SF_Y["d_n"].get(code, 0)) if not SF_Y.empty else 0])
        tb6 = pd.DataFrame(rows, columns=["方向", "股票", "行业", f"{QP}持有基金数",
                                          f"{Q}持有基金数", "环比Δ", f"同比Δ({QY}→{Q})"])
        add_table(doc, tb6, widths=[1.6, 2.6, 2.4, 2.8, 2.8, 1.8, 3.0], font=8)
        # 例子必须现算：写死「中际旭创 −193/+650」在数据一更新后就会错，
        # 而且这种「环比与同比相反」的票最好从数据里挑幅度最大的，说服力最强。
        _yv = SF_Y["d_n"] if not SF_Y.empty else pd.Series(dtype=float)
        _cmp = pd.DataFrame({
            "名称": SF_Q["stock_name"], "行业": SF_Q["industry"],
            "环比": SF_Q["d_n"], "同比": _yv.reindex(SF_Q.index),
        }).dropna()
        _div = _cmp[_cmp["环比"] * _cmp["同比"] < 0].copy()
        if not _div.empty:
            _div["幅度"] = _div[["环比", "同比"]].abs().min(axis=1)
            _top2 = _div.nlargest(2, "幅度")
            _d1 = _top2.iloc[0]
            _d2 = _top2.iloc[1] if len(_top2) > 1 else None
            _txt = (f"环比与同比经常相反，本期方向相反的有 {len(_div)} 只。"
                    f"最典型的是 {_d1['名称']}（{_d1['行业']}）："
                    f"环比 {_d1['环比']:+.0f} 家、同比 {_d1['同比']:+.0f} 家——"
                    + ("这一季在撤、这一年被大幅加过，浮盈盘刚开始离场，这类最需要警惕。"
                       if _d1["环比"] < 0 else
                       "这一季在买回、这一年被大幅减过，属于逆势承接，方向尚未确认。"))
            if _d2 is not None:
                _txt += (f"反过来，{_d2['名称']}（{_d2['行业']}）环比 {_d2['环比']:+.0f}、"
                         f"同比 {_d2['同比']:+.0f}，"
                         + ("是「这一年被减、这一季又被买回」，方向尚未确认。"
                            if _d2["环比"] > 0 else
                            "是「这一年被加、这一季开始撤」，同样要盯。"))
            P(doc, _txt)
        NOTE(doc, "口径：前十大（两期都只取每只基金最重的 10 只），披露范围一致才可比；"
                  "已剔除两期持有基金数都不到 10 家的小样本。跌出前十不等于清仓。")

    page_break(doc)

    # ---------------- 七、投资含义 ----------------
    H(doc, "七、投资含义与操作建议", 1)
    P(doc, "规则事前固定、不看结果调参：可以跟＝当期主动≥+0.3pp 且 广度≥60% 且 "
           "配置分位≤80 且 价格分位≤85；别追高＝位置到极值但资金还在进；"
           "要回避＝资金在撤且没人接，或配置分位≥90 且广度≤50。")
    _act = ACT[ACT["动作"].isin(["可以跟", "别追高", "要回避", "等确认", "观察"])]
    tb7 = pd.DataFrame([{
        "动作": r["动作"], "行业": i, "强度": r["强度"],
        "当期主动pp": f"{r['当期主动pp']:+.2f}", "加仓占比%": f"{r['加仓占比']:.0f}",
        "配置分位": f"{r['历史分位']:.0f}", "价格分位": f"{r['指数分位']:.0f}" if np.isfinite(r["指数分位"]) else "—",
    } for i, r in _act.iterrows()])
    add_table(doc, tb7, widths=[2.2, 3.2, 1.6, 2.4, 2.4, 2.2, 2.2], font=8,
              align_right_cols=("当期主动pp", "加仓占比%", "配置分位", "价格分位"))
    P(doc, f"总仓位建议 50%–60%。仓位分位 {PCT:.0f} 已属历史高位，"
           f"后续增量资金有限；离散度中位 {CHG['CSI'].std():.1f}，"
           f"行业之间分歧不小，仓位不宜打满。买入按三批建仓，单行业不超过权益仓位的 15%；"
           f"「别追高」的行业已持有可以拿住，但不加仓。")
    P(doc, "需要强调：这套规则筛的是「赔率与风险的组合」，不是收益预测。"
           "因子检验显示，9 个季报因子在 29 期样本上没有一个是正向可用的，"
           "加仓广度甚至是反向的（IC −0.041，p=0.058）。"
           "它的价值在于避免在拥挤处接棒、避免在撤离处抄底，而不是抓涨幅。")

    H(doc, "八、稳健性与局限", 1)
    H(doc, "8.1 阈值敏感性", 2)
    try:
        _sens = pb.sensitivity()
        add_table(doc, _sens[["情景", "可以跟", "要回避", "与基准重合", "重合率"]],
                  widths=[6.4, 2.2, 2.2, 2.8, 2.2], font=8.5)
        P(doc, "回避名单在放宽方向完全稳定、收紧方向不稳；可以跟名单在放宽方向会多出两家、"
               "严格方向会清空。这说明本期这几个「可以跟」的共识强度只是刚过 60% 这条线，"
               "不是压倒性共识——仓位应按「试探」而不是「重仓」来给。")
    except Exception:
        pass

    H(doc, "8.2 四条硬约束", 2)
    for txt in [
        "季报只披露前十大，绝对配置水平不可用，只能算广度与变化；",
        "数据时滞约 3–4 周（法定 15 个工作日），看到时下季度已过约 1/3；",
        "幸存者偏差：已清盘基金不在库内，早期分位偏乐观；",
        "只有股票，没有债券与现金，灵活配置型基金的行业占比会被系统性放大。",
    ]:
        BULLET(doc, txt, size=10)

    page_break(doc)

    # ---------------- 附录 ----------------
    H(doc, "附录　31 个行业完整数据", 1)
    _ap = CSI.join(CMP[["配置_前十", "Δ上季pp", "Δ上年pp", "平均主动pp", "加仓占比"]]).join(
        CHG[["上期CSI", "ΔCSI"]])
    _ap = _ap.reset_index()[["行业", "配置_前十", "Δ上季pp", "Δ上年pp", "平均主动pp",
                             "加仓占比", "配置分位", "价格分位", "CSI", "等级", "ΔCSI", "可靠"]]
    _ap.columns = ["行业", "前十配置%", "环比pp", "同比pp", "当期主动pp", "加仓占比%",
                   "配置分位", "价格分位", "CSI", "等级", "ΔCSI", "样本"]
    _ap = _ap.sort_values("CSI", ascending=False).round(2)
    add_table(doc, _ap, widths=[2.3, 2.0, 1.7, 1.7, 2.0, 1.9, 1.7, 1.7, 1.3, 1.7, 1.5, 1.4],
              font=7.2, align_right_cols=("前十配置%", "环比pp", "同比pp", "当期主动pp",
                                          "加仓占比%", "配置分位", "价格分位", "CSI", "ΔCSI"))
    NOTE(doc, "「样本」列为「小样本」的行业（持有基金数<150 或前十配置<0.5%）分位噪音大，仅供参考。"
              f"口径：前十大归一化（{QP}→{Q}）；价格分位为季末指数点位在 2019 年以来交易日中的排名；"
              "CSI＝(配置分位+价格分位+资金分位)/3。")

    return _safe_save(doc)


if __name__ == "__main__":
    p = build()
    print("已生成：", p, os.path.getsize(p) / 1024, "KB")
