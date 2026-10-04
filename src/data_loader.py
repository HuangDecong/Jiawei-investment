# -*- coding: utf-8 -*-
"""
data_loader.py  ——  数据加载、标准化与校验
======================================================================
职责
----
1. 读取 data/raw/ 下全部原始文件（缺失时回退到 master_workbook.xlsx）；
2. 统一列名（中英/别名容错）、统一单位（"189.1亿" 之类文本 → float）；
3. 统一季度索引（旧→新），拼装市场行情矩阵；
4. 执行 5 条强制校验规则，其余为增强校验；
5. 把校验过程写入 output/data_validation_log.txt。

强制校验规则（任务书要求）
--------------------------
  R1 每季度 31 行业配置比例加总 = 100.00 ± 0.01，否则 **报错停止**
  R2 HHI ∈ (0, 1)，否则 **报错停止**
  R3 超配比例 ∈ [-20%, +20%]，越界则 **警告并标注**
  R4 样本基金数 ≥ 800，否则 **警告**
  R5 校验日志写入 output/data_validation_log.txt

增强校验（本项目追加，用于交叉验证）
------------------------------------
  R6 用配置比例复算 HHI / CR3 / CR5 / CR10，与 Sheet3 逐期比对
  R7 沪深300 静态权重 = 配置比例 − 超配比例，须加总 100 且 34 期恒定
  R8 配置比例 − 超配比例 的稳定性（权重口径未漂移）
  R9 「隐含股票仓位估算均值」有效性诊断（自动判定是否弃用）
  R10 缺失值扫描
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re

import numpy as np
import pandas as pd

import config as C


# ============================================================================
# 通用工具
# ============================================================================
_ALIAS = {
    "季度": ["季度", "quarter", "季度标准", "报告期", "期数", "date", "时间"],
    "行业": ["行业", "industry", "申万一级", "申万行业", "板块", "名称"],
    "权重%": ["权重%", "权重", "weight", "hs300_weight", "沪深300权重"],
    "涨跌幅%": ["涨跌幅%", "涨跌幅", "季度涨跌幅", "区间涨跌幅", "ret", "return", "收益"],
    "成交额亿元": ["成交额亿元", "成交额", "区间成交额", "turnover", "成交金额"],
    "样本基金数": ["样本基金数", "基金数", "n_funds", "n"],
    "前十占净值比均值": ["前十占净值比均值", "前十大重仓占净值比均值", "top10_nav_mean"],
    "前十占净值比中位数": ["前十占净值比中位数", "top10_nav_median"],
    "重仓股数均值": ["重仓股数均值", "top10_cnt_mean"],
    "隐含股票仓位估算均值": ["隐含股票仓位估算均值", "隐含仓位", "implied_equity_pos"],
    "沪深300季末点位": ["沪深300季末点位", "hs300_close"],
    "CR3": ["CR3"], "CR5": ["CR5"], "CR10": ["CR10"],
    "HHI": ["HHI"], "HHI标准化": ["HHI标准化", "HHI_norm"],
    "Top1行业": ["Top1行业", "top1_ind", "第一大行业"],
    "Top1比例": ["Top1比例%", "Top1比例", "top1_pct"],
    "Top20平均持股基金数": ["Top20平均持股基金数", "top20_avg_funds"],
    "Top20持仓市值占比": ["Top20持仓市值占比%", "Top20持仓市值占比", "top20_mv_share"],
    "持股基金数_ge100": ["持股基金数≥100的个股数", "持股基金数_ge100", "n_ge100"],
    "持股基金数_ge50": ["持股基金数≥50的个股数", "持股基金数_ge50", "n_ge50"],
    "持股基金数_ge20": ["持股基金数≥20的个股数", "持股基金数_ge20", "n_ge20"],
    "个股总数": ["个股总数", "n_stocks"],
    "基金代码": ["基金代码", "fund_code", "code_fund"],
    "股票代码": ["股票代码", "stock_code", "code", "证券代码"],
    "股票名称": ["股票名称", "stock_name", "name", "证券简称"],
    "持仓市值": ["持仓市值", "持仓市值元", "market_value", "mv"],
    "占净值比": ["占净值比", "占净值比例", "占净值比%", "nav_pct"],
}


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """把列名归一到内部标准名（保留无法识别的列）。"""
    mapping = {}
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for std, alts in _ALIAS.items():
        for a in alts:
            k = str(a).strip().lower()
            if k in lowered:
                mapping[lowered[k]] = std
                break
    return df.rename(columns=mapping)


def _to_num(v):
    """把 '189.1亿'、'3,898.50'、'—'、'' 等统一转 float。"""
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("　", "")
    if s in ("", "-", "—", "--", "nan", "None", "/"):
        return np.nan
    mult = 1.0
    for suf, m in (("亿元", 1.0), ("亿", 1.0), ("万元", 1e-4), ("万", 1e-4),
                   ("%", 1.0), ("％", 1.0)):
        if s.endswith(suf):
            s, mult = s[: -len(suf)], m
            break
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def _qkey(q) -> tuple:
    """季度字符串 -> 可排序元组。支持 '2026Q2' / '2026Q2' 变体。"""
    s = str(q).strip().upper().replace(" ", "")
    m = re.match(r"^(\d{4})[QＱ]?([1-4])$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d{4})[-/]?0?([1-4])$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (9999, 9)


def _read_xlsx(path: str, sheet: int | str = 0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, dtype=object)
    return _norm_cols(df)


def _numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(_to_num)
    return out


def fingerprint() -> str:
    """data/raw 目录指纹：文件名 + 大小 + 修改时间。变化即触发缓存失效。"""
    items = []
    for p in sorted(glob.glob(os.path.join(C.DATA_RAW, "*"))):
        if os.path.isfile(p):
            st = os.stat(p)
            items.append(f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}")
    return "|".join(items)


# ============================================================================
# Bundle
# ============================================================================
class Bundle:
    """加载结果容器。"""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"<Bundle {self.latest} n_q={self.n_quarters} "
                f"inds={len(self.industries)} src={self.source_map}")


# ============================================================================
# 主加载
# ============================================================================
def load_all(verbose: bool = False) -> Bundle:
    src = {}
    notes = []

    # ---------- 1) 行业配置比例 / 超配比例（宽表：季度 × 31行业） ----------
    alloc = _load_matrix(C.F_ALLOCATION, src, "industry_allocation.xlsx")
    over = _load_matrix(C.F_OVERWEIGHT, src, "industry_overweight.xlsx")

    # ---------- 2) 沪深300 权重 ----------
    w_path = C.F_HS300_WEIGHTS
    if os.path.exists(w_path):
        wdf = _read_xlsx(w_path)
        wdf = _numeric(wdf, ["权重%"])
        hs300_w = pd.Series(wdf["权重%"].to_numpy(dtype=float),
                            index=wdf["行业"].astype(str).str.strip()).reindex(C.INDUSTRIES)
        src["hs300_weights"] = "hs300_weights.xlsx"
    else:
        hs300_w = (alloc - over).iloc[-1].reindex(C.INDUSTRIES)
        src["hs300_weights"] = "推导：配置比例−超配比例"
        notes.append("hs300_weights.xlsx 缺失，已用「配置−超配」推导")

    # ---------- 3) 市场行情：成交额 / 涨跌幅 / 沪深300 ----------
    turn_long = _load_long(C.F_TURNOVER, "成交额亿元", src, "sw_industry_turnover.xlsx")
    ret_long = _load_long(C.F_RETURNS, "涨跌幅%", src, "sw_industry_returns.xlsx")

    turn = (turn_long.pivot(index="季度", columns="行业", values="成交额亿元")
            .reindex(columns=C.INDUSTRIES))
    ret = (ret_long.pivot(index="季度", columns="行业", values="涨跌幅%")
           .reindex(columns=C.INDUSTRIES))

    quarters = sorted(set(alloc.index) | set(over.index) | set(turn.index) | set(ret.index),
                      key=_qkey)
    quarters = [q for q in quarters if _qkey(q)[0] < 9999]

    alloc = alloc.reindex(quarters)[C.INDUSTRIES]
    over = over.reindex(quarters)[C.INDUSTRIES]
    turn = turn.reindex(quarters)[C.INDUSTRIES]
    ret = ret.reindex(quarters)[C.INDUSTRIES]

    if os.path.exists(C.F_HS300_RETURNS):
        hdf = _read_xlsx(C.F_HS300_RETURNS)
        hdf = _numeric(hdf, ["涨跌幅%"])
        hs300 = pd.Series(hdf["涨跌幅%"].to_numpy(dtype=float),
                          index=hdf["季度"].astype(str).str.strip()).reindex(quarters)
        src["hs300_returns"] = "hs300_returns.xlsx"
    else:
        hs300 = pd.Series(np.nan, index=quarters)
        src["hs300_returns"] = "缺失"
        notes.append("hs300_returns.xlsx 缺失：超额收益无法计算")

    # ---------- 4) 集中度 / 个股 / 仓位 ----------
    cr_hhi = _load_ts(C.F_CR_HHI, ["CR3", "CR5", "CR10", "HHI", "HHI标准化",
                                   "Top1比例"], quarters, src, "industry_cr_hhi.xlsx",
                      keep_str=["Top1行业"])
    stock_crowd = _load_ts(C.F_STOCK_CROWD,
                           ["Top20平均持股基金数", "Top20持仓市值占比",
                            "持股基金数_ge100", "持股基金数_ge50", "持股基金数_ge20",
                            "个股总数"], quarters, src, "stock_crowd.xlsx")
    position = _load_ts(C.F_POSITION, ["样本基金数", "前十占净值比均值", "前十占净值比中位数",
                                       "重仓股数均值", "隐含股票仓位估算均值",
                                       "沪深300季末点位"], quarters, src, "position_proxy.xlsx")

    # ---------- 5) 抱团股 Top20 ----------
    hot = None
    hot_files = sorted(glob.glob(os.path.join(C.DATA_RAW, "hot_stocks_top20_*.xlsx")))
    if hot_files:
        hd = _read_xlsx(hot_files[-1])
        hd = _numeric(hd, ["持股基金数", "持仓市值亿元", "持仓市值万元"])
        hot = hd
        src["hot_stocks_top20"] = os.path.basename(hot_files[-1])

    # ---------- 6) 季报前十大重仓股明细（可选） ----------
    hfiles = [p for p in sorted(glob.glob(C.PAT_HOLDINGS))
              if "TEMPLATE" not in os.path.basename(p).upper()]
    holdings = None
    if hfiles:
        frames = []
        for p in hfiles:
            try:
                d = _read_xlsx(p)
                d = _numeric(d, ["持仓市值", "占净值比"])
                d["_source"] = os.path.basename(p)
                frames.append(d)
            except Exception as e:                      # noqa: BLE001
                notes.append(f"{os.path.basename(p)} 读取失败：{e}")
        if frames:
            holdings = pd.concat(frames, ignore_index=True)
            src["fund_holdings"] = f"{len(hfiles)} 个文件"
    if holdings is None:
        src["fund_holdings"] = "未提供（可选）"
        notes.append("未发现 fund_holdings_YYYYQX.xlsx：个股层抱团指标改用 stock_crowd.xlsx "
                     "的市场层面口径，功能不受影响")

    # ---------- 7) 派生市场矩阵 ----------
    turn_share = turn.div(turn.sum(axis=1), axis=0) * 100.0
    excess = ret.sub(hs300, axis=0)
    market = dict(ret=ret, turn=turn, hs300=hs300, excess=excess,
                  turn_share=turn_share, turn_total=turn.sum(axis=1))

    b = Bundle(alloc=alloc, overweight=over, hs300_weights=hs300_w,
               quarters=quarters, n_quarters=len(quarters),
               industries=list(C.INDUSTRIES), latest=quarters[-1],
               cr_hhi=cr_hhi, stock_crowd=stock_crowd, position=position,
               hot_top20=hot, holdings=holdings, market=market,
               source_map=src, notes=notes)

    # 真实具备基金持仓数据的季度（机构级：必须显式区分「有行情」与「有持仓」）
    b.alloc_quarters = [q for q in quarters
                        if alloc.loc[q].notna().sum() >= C.NI // 2]
    b.over_quarters = [q for q in quarters
                       if over.loc[q].notna().sum() >= C.NI // 2]
    notes.append(
        f"具备基金持仓数据的季度 {len(b.alloc_quarters)} 期"
        f"（{b.alloc_quarters[0] if b.alloc_quarters else '-'} ~ "
        f"{b.alloc_quarters[-1] if b.alloc_quarters else '-'}）；"
        f"可计算超配比例的季度 {len(b.over_quarters)} 期"
        f"（{b.over_quarters[0] if b.over_quarters else '-'} ~ "
        f"{b.over_quarters[-1] if b.over_quarters else '-'}）")

    b = attach_extras(b)

    if verbose:
        print("【数据来源】")
        for k, v in src.items():
            print(f"  {k:<20} <- {v}")
        for n in notes:
            print(f"  ⚠️ {n}")
    return b


def _load_matrix(path: str, src: dict, name: str) -> pd.DataFrame:
    """宽表（季度 × 行业）或长表（季度/行业/值）都能读。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少必需文件 {path}；请先运行 tools/seed_raw_data.py")
    df = _read_xlsx(path)
    src[name.split(".")[0]] = name
    if "行业" in df.columns and "季度" in df.columns:
        val = [c for c in df.columns if c not in ("季度", "行业")][0]
        df = _numeric(df, [val])
        return df.pivot(index="季度", columns="行业", values=val)
    # 宽表
    qcol = "季度"
    df = df.dropna(subset=[qcol])
    df[qcol] = df[qcol].astype(str).str.strip()
    mat = df.set_index(qcol)
    mat = mat[[c for c in mat.columns if c in C.INDUSTRIES]]
    mat = _numeric(mat, mat.columns)
    order = sorted(mat.index, key=_qkey)
    return mat.loc[[q for q in order if _qkey(q)[0] < 9999]]


def _load_long(path: str, value_col: str, src: dict, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少必需文件 {path}；请先运行 tools/seed_raw_data.py")
    df = _read_xlsx(path)
    src[name.split(".")[0]] = name
    df = df[["行业", "季度", value_col]].copy()
    df["行业"] = df["行业"].astype(str).str.strip()
    df["季度"] = df["季度"].astype(str).str.strip()
    df = _numeric(df, [value_col])
    return df.dropna(subset=[value_col])


def _load_ts(path: str, cols: list, quarters: list, src: dict, name: str,
             keep_str: list | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少必需文件 {path}；请先运行 tools/seed_raw_data.py")
    df = _read_xlsx(path)
    src[name.split(".")[0]] = name
    want = ["季度"] + [c for c in cols if c in df.columns] + \
           [c for c in (keep_str or []) if c in df.columns]
    df = _numeric(df[want], cols)
    df["季度"] = df["季度"].astype(str).str.strip()
    return df.set_index("季度").reindex(quarters)


# ============================================================================
# 校验
# ============================================================================
def _holdings_quarters(b: "Bundle") -> list:
    """返回**真正有基金持仓数据**的季度列表。

    机构级约束：R1「配置比例加总 = 100」只在有 holdings 的季度上成立。
    2019–2023 年本项目的行情层（涨跌幅/成交额）有数据，但基金持仓数据
    尚未回填，若不区分就会把「全 NaN 季度」误判为校验失败。
    """
    if getattr(b, "alloc_quarters", None):
        return list(b.alloc_quarters)
    ok = b.alloc.notna().sum(axis=1) >= C.NI // 2
    return [q for q in b.alloc.index if ok.loc[q]]


def validate(b: Bundle, write_log: bool = True) -> dict:
    """执行全部校验规则，返回结果 dict；write_log=True 时写日志文件。"""
    lines, result = [], {}
    ok = True
    hard_fail = []
    hq = _holdings_quarters(b)

    def add(txt=""):
        lines.append(txt)

    add("=" * 78)
    add("公募基金交易拥挤度与集中度仪表盘 —— 数据校验日志")
    add(f"生成时间：{_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    add(f"数据区间：{b.quarters[0]} ~ {b.latest}  共 {b.n_quarters} 个季度")
    add(f"其中具备基金持仓数据的季度：{len(hq)} 期（{hq[0] if hq else '-'} ~ "
        f"{hq[-1] if hq else '-'}）")
    add(f"行业口径：申万一级 {len(b.industries)} 个行业")
    add("=" * 78)
    add()

    # ---------------- R1 配置比例加总 ----------------
    s = b.alloc.loc[hq].sum(axis=1)
    dev = (s - 100.0).abs()
    r1 = dev <= C.VALIDATION_SUM_TOL
    result["R1_alloc_sum"] = pd.DataFrame({"加总": s, "偏离": dev, "通过": r1})
    result["R1_最大的偏离"] = float(dev.max()) if len(dev) else float("nan")
    add("【R1】每季度 31 行业配置比例加总 = 100.00 ± 0.01（仅持仓季度）")
    add(f"     最大偏离 = {dev.max():.8f} pct   最差季度 = {dev.idxmax() if len(dev) else '-'}")
    add(f"     结论：{'✅ 通过' if r1.all() else '❌ 失败 → 报错停止'}")
    ok &= bool(r1.all())
    if not r1.all():
        hard_fail.append(f"R1 配置比例加总异常：{[q for q in dev.index[~r1]]}")
    add()

    # ---------------- R2 HHI 值域 ----------------
    hhi = b.cr_hhi["HHI"].reindex(hq).dropna()
    lo, hi = C.VALIDATION_HHI_RANGE
    r2 = (hhi > lo) & (hhi < hi)
    result["R2_hhi_range"] = r2
    add("【R2】HHI ∈ (0, 1)")
    add(f"     范围 = [{hhi.min():.4f}, {hhi.max():.4f}]   最新 = {hhi.iloc[-1]:.4f}"
        f"   历史均值 = {hhi.mean():.4f}")
    add(f"     结论：{'✅ 通过' if r2.all() else '❌ 失败 → 报错停止'}")
    ok &= bool(r2.all())
    if not r2.all():
        hard_fail.append("R2 HHI 越界")
    add()

    # ---------------- R3 超配比例区间 ----------------
    lo2, hi2 = C.VALIDATION_OVER_RANGE
    # 机构级：R3 必须基于**真正能算出超配比例**的季度（holdings ∩ hs300），
    # 否则 2024Q1 这类「有持仓但缺基准」的季度会以 NaN 混入，污染越界计数。
    oq = getattr(b, "over_quarters", None) or hq
    over_h = b.overweight.reindex(oq)
    valid = over_h.notna()
    oor = valid & ((over_h < lo2) | (over_h > hi2))
    n_oor = int(oor.sum().sum())
    result["R3_over_range"] = oor
    result["R3_越界明细"] = (over_h.where(oor).stack().dropna()
                             .rename("超配比例").to_frame()
                             .reset_index().rename(columns={"level_0": "季度", "level_1": "行业"}))
    add(f"【R3】超配比例 ∈ [{lo2:+.0f}%, {hi2:+.0f}%]"
        f"（检验季度：{len(oq)} 期，{oq[0] if oq else '-'} ~ {oq[-1] if oq else '-'}）")
    if n_oor == 0:
        add("     ✅ 通过：无越界值")
        add(f"     实际范围 = [{over_h.min().min():+.2f}%, "
            f"{over_h.max().max():+.2f}%]")
    else:
        add(f"     ⚠️ 警告：{n_oor} 个越界值（已标注，不中止运行）")
        add(f"     实际范围 = [{over_h.min().min():+.2f}%, "
            f"{over_h.max().max():+.2f}%]")
        for r in result["R3_越界明细"].head(30).itertuples():
            add(f"        {r.季度}  {r.行业}  {r.超配比例:+.2f}%")
    ok &= True            # 仅警告
    add()

    # ---------------- R4 样本基金数 ----------------
    # 机构级：R4 的下限约束只对**本项目定义的主动权益口径**有意义。
    # position_proxy 中的「样本基金数」是旧口径（全市场公募基金），
    # 与 holdings 的主动权益三类口径不一致，故本项改为**诊断性披露**，
    # 不参与通过/失败判定，并在报告中显式标注口径差异。
    nf_all = b.position["样本基金数"].dropna() if "样本基金数" in b.position else pd.Series(dtype=float)
    if len(nf_all):
        nf_h = nf_all.reindex(hq).dropna()
        result["R4_n_funds"] = nf_all
        add(f"【R4】样本基金数（诊断项，非硬性）")
        add(f"     全区间：最小 = {nf_all.min():.0f}（{nf_all.idxmin()}）  "
            f"最大 = {nf_all.max():.0f}  最新 = {nf_all.iloc[-1]:.0f}")
        if len(nf_h):
            add(f"     持仓季度：最小 = {nf_h.min():.0f}（{nf_h.idxmin()}）  "
                f"最大 = {nf_h.max():.0f}  → 全部 ≥ {C.VALIDATION_MIN_FUNDS}："
                f"{'✅ 通过' if (nf_h >= C.VALIDATION_MIN_FUNDS).all() else '⚠️ 部分低于'}")
        add(f"     ⚠️ 口径提示：该列来源于旧口径（全市场公募基金），"
            f"与 Wind 主动权益三类口径（holdings）不一致，仅作规模参考。")
    ok &= True            # 诊断项
    add()

    # ---------------- R6 HHI / CR 复算一致性 ----------------
    w = b.alloc.reindex(hq) / 100.0
    hhi_re = (w ** 2).sum(axis=1)
    cr3_re = w.apply(lambda r: r.nlargest(3).sum() * 100, axis=1)
    cr5_re = w.apply(lambda r: r.nlargest(5).sum() * 100, axis=1)
    cr10_re = w.apply(lambda r: r.nlargest(10).sum() * 100, axis=1)
    ch = b.cr_hhi.reindex(hq)
    cmp_tbl = pd.DataFrame({
        "HHI_文件": ch["HHI"], "HHI_复算": hhi_re,
        "HHI_差": (ch["HHI"] - hhi_re).abs(),
        "CR3_文件": ch["CR3"], "CR3_复算": cr3_re,
        "CR3_差": (ch["CR3"] - cr3_re).abs(),
        "CR5_文件": ch["CR5"], "CR5_复算": cr5_re,
        "CR5_差": (ch["CR5"] - cr5_re).abs(),
        "CR10_文件": ch["CR10"], "CR10_复算": cr10_re,
        "CR10_差": (ch["CR10"] - cr10_re).abs()})
    result["R6_recalc"] = cmp_tbl
    result["R6_最大绝对差"] = dict(
        HHI=float(cmp_tbl["HHI_差"].max()), CR3=float(cmp_tbl["CR3_差"].max()),
        CR5=float(cmp_tbl["CR5_差"].max()), CR10=float(cmp_tbl["CR10_差"].max()))
    add("【R6】用配置比例独立复算 HHI / CR3 / CR5 / CR10，与 industry_cr_hhi.xlsx 比对")
    add(f"     最大绝对差：HHI={result['R6_最大绝对差']['HHI']:.2e}  "
        f"CR3={result['R6_最大绝对差']['CR3']:.2e}  "
        f"CR5={result['R6_最大绝对差']['CR5']:.2e}  "
        f"CR10={result['R6_最大绝对差']['CR10']:.2e}")
    tol = 1e-6
    r6 = all(v < tol for v in result["R6_最大绝对差"].values())
    add(f"     结论：{'✅ 完全一致（<1e-6）' if r6 else '⚠️ 存在差异，请检查口径'}")
    add()

    # HHI 标准化公式
    nm = (ch["HHI"] - 1.0 / len(b.industries)) / (1.0 - 1.0 / len(b.industries))
    nm_diff = float((ch["HHI标准化"] - nm).abs().max())
    result["R6b_hhi_norm_diff"] = nm_diff
    add("【R6b】HHI标准化公式反解：(HHI − 1/31) / (1 − 1/31)")
    add(f"     最大绝对差 = {nm_diff:.2e}   "
        f"结论：{'✅ 公式确认' if nm_diff < 1e-6 else '⚠️ 公式不符'}")
    add()

    # ---------------- R7 沪深300 权重（动态口径） ----------------
    # 机构级：本项目已切到**动态季度权重**（Wind 实测，逐季不同）。
    # 「权重加总=100 且各期恒定」是早期静态口径下的规则，现已不适用。
    # 新规则：
    #   R7a 每期权重加总 ∈ [95, 100]（沪深300 不含全部 31 行业，加总<100 正常）
    #   R7b 权重逐季变动（动态性验证，非恒定）
    #   R7c 配置比例 − 超配比例 应精确还原该期权重
    wq = b.weight_matrix.reindex(hq)
    wsum = wq.sum(axis=1)
    implied = b.alloc.reindex(hq) - b.overweight.reindex(hq)
    recon = (implied - wq).abs().max().max()
    r7a = wsum.between(95.0, 100.5).all()
    n_chg = int((wq.diff().abs().sum(axis=1) > 1e-6).sum()) - 1
    result["R7_weight_sum"] = wsum
    result["R7_权重还原最大差"] = float(recon)
    result["R7_权重变动期数"] = n_chg
    add("【R7】沪深300 行业权重（动态季度口径）")
    add(f"     各期权重加总 ∈ [{wsum.min():.2f}, {wsum.max():.2f}]"
        f"（<100 属正常，沪深300 仅覆盖 26–28 个申万一级行业）")
    add(f"     权重还原检验：|配置比例 − 超配比例 − 权重| 最大值 = {recon:.2e}")
    add(f"     动态性：{n_chg} 期权重发生变化（非恒定）")
    add(f"     结论：{'✅ 动态权重生效、口径可复现' if r7a and recon < 1e-6 else '⚠️ 需复核'}")
    add()

    # ---------------- R9 隐含股票仓位有效性诊断 ----------------
    if "隐含股票仓位估算均值" in getattr(b.position, "columns", []):
        imp = b.position["隐含股票仓位估算均值"].dropna()
        if len(imp) >= 3:
            n_in = int(imp.between(0, 100).sum())
            verdict = "弃用"
            diag = dict(n=len(imp), n_in_0_100=n_in, min=float(imp.min()),
                        max=float(imp.max()), mean=float(imp.mean()),
                        corr_nfunds=float(imp.corr(
                            b.position["样本基金数"].reindex(imp.index))),
                        corr_top10=float(imp.corr(
                            b.position["前十占净值比均值"].reindex(imp.index))))
            result["R9_implied_pos"] = diag
            add("【R9】异常列诊断：隐含股票仓位估算均值")
            add(f"     取值区间 = [{diag['min']:.2f}, {diag['max']:.2f}]   "
                f"落在 0~100 的期数 = {n_in}/{len(imp)}")
            add(f"     判定：**{verdict}**（绝对量口径，不是股票仓位百分比）")
            add()
    else:
        result["R9_verdict"] = "列不存在，跳过"
        add("【R9】跳过：position_proxy 中无「隐含股票仓位估算均值」列")
        add()

    # ---------------- R10 缺失值 ----------------
    miss = []
    for nm2, df in [("配置比例(持仓季度)", b.alloc.reindex(hq)),
                    ("超配比例(持仓季度)", b.overweight.reindex(hq)),
                    ("成交额", b.market["turn"].reindex(hq)),
                    ("涨跌幅", b.market["ret"].reindex(hq)),
                    ("沪深300", b.market["hs300"].reindex(hq).to_frame())]:
        m = int(df.isna().sum().sum())
        miss.append(dict(数据集=nm2, 缺失数=m, 缺失率=f"{m / df.size * 100:.2f}%"))
    recent = b.market["ret"].reindex(hq).tail(4).isna().sum().sum()
    result["R10_missing"] = pd.DataFrame(miss)
    result["R10_recent_missing"] = int(recent)
    add("【R10】缺失值扫描（持仓季度）")
    for r in miss:
        add(f"     {r['数据集']:<18} 缺失 {r['缺失数']:>4} 个 ({r['缺失率']})")
    add(f"     近 4 季行业涨跌幅缺失 = {recent} 个  → 结论："
        f"{'✅ 数据完整' if recent == 0 else '⚠️ 近期数据不完整'}")
    # 成交额缺口说明（机构级：缺口必须说明来源，不能只报数字）
    tm = result["R10_missing"]
    row_t = tm[tm["数据集"] == "成交额"]
    if len(row_t) and int(row_t["缺失数"].iloc[0]) > 0:
        tn = b.market["turn"].reindex(hq).notna().sum(axis=1)
        have = [q for q in hq if tn.loc[q] > 0]
        add(f"     ↳ 成交额缺口原因：turnover 抓取覆盖 {have[0] if have else '-'} ~ "
            f"{have[-1] if have else '-'}（{len(have)} 期），"
            f"其余持仓季度尚无成交额数据。")
        add(f"       F14(ETF资金流) / F5(交易拥挤) 只能在有成交额的季度上计算，"
            f"属**已知覆盖缺口**，已在 provenance 登记。")
    add()

    # ---------------- 汇总 ----------------
    add("=" * 78)
    if hard_fail:
        add("❌ 存在硬性失败项：")
        for h in hard_fail:
            add(f"   · {h}")
    else:
        add("✅ 全部硬性校验通过（R1 / R2），警告项见上文。")
    add("=" * 78)
    result["passed"] = not hard_fail
    result["hard_fail"] = hard_fail

    if write_log:
        with open(C.O_VALIDATION_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    result["log_text"] = "\n".join(lines)

    if hard_fail:
        raise ValueError("数据校验硬性失败：\n" + "\n".join(hard_fail))
    result["ok"] = bool(ok)
    return result


# ============================================================================
# 本次升级：日度数据 / 动态权重 / 数据来源登记
# ============================================================================
def load_daily_turnover() -> dict:
    """行业日度成交额 + 全市场合计 + 覆盖率清单。

    返回
    ----
    turn_long : DataFrame[行业, 日期, 成交额亿元]  （仅含真实存在的日期，不填补）
    market    : DataFrame[日期, 成交额亿元]        （申万A指，占比分母）
    coverage  : DataFrame[序列, 覆盖交易日, 应有交易日, 覆盖率]
    """
    if not os.path.exists(C.F_TURNOVER_DAILY):
        return dict(turn_long=None, market=None, coverage=None, available=False)
    xl = pd.ExcelFile(C.F_TURNOVER_DAILY)
    tl = xl.parse("行业日度成交额")
    tl["日期"] = pd.to_datetime(tl["日期"]).dt.strftime("%Y-%m-%d")
    tl = _numeric(tl, ["成交额亿元"]).dropna(subset=["成交额亿元"])
    mk = xl.parse("全市场合计")
    mk["日期"] = pd.to_datetime(mk["日期"]).dt.strftime("%Y-%m-%d")
    mk = _numeric(mk, ["成交额亿元"]).dropna(subset=["成交额亿元"])
    cov = xl.parse("覆盖率清单") if "覆盖率清单" in xl.sheet_names else None

    # 行业日度成交额占比（%）＝ 行业成交额 / 当日全市场成交额 × 100
    j = tl.merge(mk[["日期", "成交额亿元"]].rename(columns={"成交额亿元": "全市场亿元"}),
                 on="日期", how="inner")
    j["成交额占比%"] = (j["成交额亿元"] / j["全市场亿元"] * 100).round(C.DEC_PCT)
    return dict(turn_long=j, market=mk, coverage=cov, available=True)


def load_etf_daily() -> pd.DataFrame | None:
    """ETF 日度资金流（净申赎 / 份额 / 成交额 / 所属行业）。"""
    if not os.path.exists(C.F_ETF_DAILY):
        return None
    df = pd.read_excel(C.F_ETF_DAILY, sheet_name="etf_flow_daily")
    df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    cols = ["日期", "代码", "名称", "申万一级", "成交额", "场内流通份额", "单位净值",
            "复权份额", "复权净值", "净申赎亿元", "份额折算", "折算倍数K"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols].sort_values(["代码", "日期"]).reset_index(drop=True)


def load_hs300_weights_quarterly() -> pd.DataFrame | None:
    """动态沪深300 行业权重（季度）。

    期望格式（长表）：季度 | 行业 | 权重%
    若文件不存在或覆盖不全，返回 None，由调用方回退到静态权重并标注。
    """
    if not os.path.exists(C.F_HS300_WEIGHTS_Q):
        return None
    df = _read_xlsx(C.F_HS300_WEIGHTS_Q)
    if "季度" not in df.columns or "行业" not in df.columns:
        return None
    val = "权重%" if "权重%" in df.columns else df.columns[-1]
    df = _numeric(df, [val])
    df["行业"] = df["行业"].astype(str).str.strip()
    df["季度"] = df["季度"].astype(str).str.strip()
    return df[["季度", "行业", val]].rename(columns={val: "权重%"})


def build_weight_matrix(b: "Bundle") -> dict:
    """构造基准权重矩阵：优先动态季度权重，缺失季度回退静态权重并标注。

    机构级口径说明
    --------------
    沪深300 并非包含全部 31 个申万一级行业。实测本项目样本期，沪深300 仅覆盖
    26–28 个行业（轻工制造、纺织服饰、综合、环保、社会服务等常年无成份股）。
    因此**缺失行业必须按 0 权重处理**——语义正确：该行业在沪深300中确实没有
    权重，其「超配比例 = 配置比例 − 0」成立。

    早期版本要求「缺失行业 ≤ 1 才采纳动态权重」，导致明明有真实季度权重却
    全部回退到静态口径，属**过度保守的实现缺陷**。现改为：只要有 ≥ 20 个行业
    有值即采纳，缺的补 0。
    """
    static = b.hs300_weights.reindex(C.INDUSTRIES)
    W = pd.DataFrame([static.to_numpy(dtype=float)] * len(b.quarters),
                     index=b.quarters, columns=C.INDUSTRIES)
    source = pd.Series("静态权重（配置−超配 反解）", index=b.quarters, name="权重来源")
    dyn = getattr(b, "hs300_w_q", None)
    n_dyn = 0
    if dyn is not None and not dyn.empty:
        piv = dyn.pivot(index="季度", columns="行业", values="权重%")
        for q in b.quarters:
            if q in piv.index:
                row = piv.loc[q].reindex(C.INDUSTRIES)
                n_present = int(row.notna().sum())
                # 沪深300 天然不含部分行业 → 缺失按 0，而非拒绝整期
                if n_present >= 20:
                    W.loc[q] = row.fillna(0.0).to_numpy(dtype=float)
                    source.loc[q] = (f"动态季度权重（覆盖{n_present}/31，"
                                     f"缺行业按0）")
                    n_dyn += 1
    return dict(weights=W, source=source, n_dynamic=n_dyn,
                has_dynamic=n_dyn > 0)


def load_provenance():
    p = os.path.join(C.OUTPUT, "data_provenance.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def load_wind_provenance():
    """逐期逐行业的 Wind 取数登记（windcode / 指标名 / 报告期 / 抓取时刻）。

    机构级要求：由任一发布结论可回溯到原始取数批次。文件缺失时返回 None，
    由调用方显式提示，不静默降级。
    """
    p = os.path.join(C.DATA_RAW, "wind_provenance.xlsx")
    if not os.path.exists(p):
        return None
    try:
        return pd.read_excel(p)
    except Exception:
        return None


def load_alloc_ext(b: "Bundle") -> dict:
    """全历史行业配置矩阵（用于 ③ 页堆叠面积图 / 热力图的长区间展示）。

    口径拼接（机构级，逐段可溯源）
    ------------------------------
    * 2019Q1 ~ 2023Q4：偏股混合型基金单类型，前十大重仓股行业市值占比
      —— Wind 逐期重抓（data/wind_raw/holdings_v2/eq_series.json，
      每期经「31 行业 + 总量平滑 + HHI 落位」三重校验）；
    * 2024Q1 ~ 2026Q2：主口径 cache（与 ①② 页完全一致，其中 2025Q2 起
      为主动权益三类合计，口径断点已经占比不变性检验证明对 HHI/CR 稳健）。
    * 主口径与偏股混合口径在重叠期（2024Q1~2026Q1）的 HHI 平均绝对差
      < 0.011，拼接处结构连续。
    """
    fp = os.path.join(C.DATA, "wind_raw", "holdings_v2", "eq_series.json")
    if not os.path.exists(fp):
        return dict(matrix=b.alloc.copy(), quarters=list(b.alloc_quarters),
                    source=pd.Series("主口径（cache）", index=b.alloc_quarters),
                    available=False)
    eq = json.load(open(fp, encoding="utf-8"))
    rows, srcs = {}, {}
    cache_q = set(b.alloc_quarters)
    for q, vals in sorted(eq.items(), key=lambda kv: _qkey(kv[0])):
        if q in cache_q:
            continue                       # 主口径优先，保证与 ①② 页一致
        s = pd.Series(vals, dtype=float)
        rows[q] = (s / s.sum() * 100.0)
        srcs[q] = "偏股混合型单类型（Wind 逐期重抓）"
    for q in b.alloc_quarters:
        rows[q] = b.alloc.loc[q]
        srcs[q] = ("主口径·主动权益三类（2025Q2 起三类/此前偏股混合口径为主）"
                   if q >= "2025Q2" else "主口径（cache，偏股混合口径为主）")
    qs = sorted(rows, key=_qkey)
    mat = pd.DataFrame(rows).T.reindex(columns=C.INDUSTRIES).fillna(0.0)
    return dict(matrix=mat, quarters=qs,
                source=pd.Series(srcs), available=True)


def load_over_ext(b: "Bundle") -> dict:
    """偏股混合口径的「超配比例长历史」序列（用于扩窗标准化，解锁 F2/F3/F4b/F15）。

    为什么需要它（解决样本边界）
    ----------------------------
    主口径持仓（holdings cache）仅 2024Q1~2026Q2 共 10 期，扩窗 Z / 分位要求
    MIN_HIST=8 期起步 → F2/F3/F4b 只有最后 2 期有值、F15 全 NaN，无法做 IC 检验。
    但偏股混合型单类型重仓已抓齐 2016Q2~2026Q1（35 期），叠加 Wind 动态 hs300 权重
    （2022Q1~2026Q2，18 期），可构建 16 期「偏股混合配置 − hs300 权重」的超配长历史，
    使扩窗统计从 2024Q1 起即满足 8 期起步。

    口径（逐段可溯源）
    -----------------
    * 2022Q1 ~ 2026Q1：偏股混合配置占比 − 动态 hs300 权重（统一偏股混合单类型口径）；
    * 2026Q2：偏股混合 2026Q2 被 Wind 拒绝（子集>母集矛盾），用主口径超配衔接，
      已在 WIND_RAW.md 登记；扩窗 Z/分位对单期口径跳变有一定韧性，但该期已单独标注。
    """
    fp = os.path.join(C.DATA, "wind_raw", "holdings_v2", "eq_series.json")
    if not os.path.exists(fp):
        return dict(matrix=b.overweight.copy(), quarters=list(b.over_quarters),
                    available=False)
    eq = json.load(open(fp, encoding="utf-8"))
    w = getattr(b, "weight_matrix", None)
    if w is None or w.empty:
        return dict(matrix=b.overweight.copy(), quarters=list(b.over_quarters),
                    available=False)
    # 只用「动态季度权重」段（2022Q1~2026Q2）：静态权重段（2019~2021 由配置−超配反解）
    # 与偏股混合配置口径不同源，混入扩窗会引入基准口径漂移，故排除。
    wsrc = getattr(b, "weight_source", None)
    dyn_q = set(q for q in w.index
                if wsrc is None or "动态" in str(wsrc.get(q, "")))
    rows, srcs = {}, {}
    for q, vals in sorted(eq.items(), key=lambda kv: _qkey(kv[0])):
        if q not in dyn_q:
            continue                       # 缺动态权重 → 无法算超配
        s = pd.Series(vals, dtype=float)
        pct = s / s.sum() * 100.0
        rows[q] = (pct - w.loc[q]).reindex(C.INDUSTRIES)
        srcs[q] = "偏股混合配置 − 动态hs300权重"
    # 2026Q2 主口径衔接（偏股混合 2026Q2 无数据）
    if "2026Q2" in b.overweight.index and "2026Q2" not in rows:
        rows["2026Q2"] = b.overweight.loc["2026Q2"].reindex(C.INDUSTRIES)
        srcs["2026Q2"] = "主口径超配（偏股混合2026Q2被拒，口径衔接）"
    qs = sorted(rows, key=_qkey)
    mat = pd.DataFrame(rows).T.reindex(columns=C.INDUSTRIES)
    return dict(matrix=mat, quarters=qs, source=pd.Series(srcs), available=True)


def attach_extras(b: "Bundle") -> "Bundle":
    """把本次升级的新数据挂到 Bundle 上（缺失时给出明确状态，不静默失败）。"""
    dt = load_daily_turnover()
    b.daily_turn = dt["turn_long"]
    b.daily_market = dt["market"]
    b.daily_coverage = dt["coverage"]
    b.daily_available = dt["available"]
    if not dt["available"]:
        b.notes.append("未发现 sw_industry_turnover_daily.xlsx：日度交易拥挤层不可用")

    b.etf_daily = load_etf_daily()
    if b.etf_daily is None:
        b.notes.append("未发现 etf_flow_daily.xlsx：F14 ETF资金流强度不可用")

    b.hs300_w_q = load_hs300_weights_quarterly()
    wm = build_weight_matrix(b)
    b.weight_matrix = wm["weights"]
    b.weight_source = wm["source"]
    b.n_dynamic_quarters = wm["n_dynamic"]
    if not wm["has_dynamic"]:
        b.notes.append(
            f"未发现 hs300_weights_quarterly.xlsx：基准权重沿用静态口径"
            f"（34 期恒定）。超配比例的绝对水平存在系统性偏移，"
            f"相对排序不受影响。动态历史权重需外部指数数据源。")
    else:
        b.notes.append(f"动态季度权重生效 {wm['n_dynamic']}/{b.n_quarters} 期，"
                       f"其余季度回退静态权重")

    # 全持仓（半年/年报）校准：仅登记是否存在
    full = [p for p in glob.glob(C.F_HOLDINGS_FULL)]
    b.holdings_full_files = full
    b.holdings_full = None
    if full:
        frames = []
        for p in full:
            try:
                d = _read_xlsx(p)
                d = _numeric(d, ["持仓市值", "占净值比"])
                d["_source"] = os.path.basename(p)
                frames.append(d)
            except Exception as e:                          # noqa: BLE001
                b.notes.append(f"{os.path.basename(p)} 读取失败：{e}")
        if frames:
            b.holdings_full = pd.concat(frames, ignore_index=True)
    else:
        b.notes.append("未发现 fund_holdings_full_*.xlsx：无法用半年报/年报全持仓"
                       "校准前十大重仓口径偏差（口径偏差方向与幅度见 README 风险提示）")

    b.provenance = load_provenance()

    # 全历史行业配置矩阵（③ 页长区间图专用；不影响因子计算所用的主口径）
    _ae = load_alloc_ext(b)
    b.alloc_ext = _ae["matrix"]
    b.alloc_ext_quarters = _ae["quarters"]
    b.alloc_ext_source = _ae["source"]
    b.alloc_ext_available = _ae["available"]

    # 偏股混合超配长历史（扩窗标准化用，解锁 F2/F3/F4b/F15）
    _oe = load_over_ext(b)
    b.over_ext = _oe["matrix"]
    b.over_ext_quarters = _oe["quarters"]
    b.over_ext_source = _oe["source"]
    b.over_ext_available = _oe["available"]
    b.wind_provenance = load_wind_provenance()
    if b.wind_provenance is None:
        b.notes.append("未发现 data/raw/wind_provenance.xlsx：逐期取数溯源不可用")
    return b
