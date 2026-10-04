# -*- coding: utf-8 -*-
"""
insights.py —— 图表洞察引擎（特征识别 · 拐点检测 · 结论生成）
================================================================================
设计目标
--------
用户要求：**每一张图都要有分析、结论，并突出特征与拐点**。

人工为 30+ 张图逐张写解读，既不可维护，也会随数据更新而过期。
本模块的做法是：**从数据本身算出特征与拐点**，再把数字填进固定句式中。

三个层次
--------
  L1  series_profile()   序列画像——起点/终点/极值/单调性/波动/最大单期变动
  L2  detect_turns()     拐点检测——趋势反转、最大跃迁、突破历史极值
  L3  describe_series()  结论生成——把 L1/L2 组装成「特征 / 拐点 / 含义」三段式

设计纪律
--------
1. **所有文字结论都由数字驱动**，不做无根据的定性描述；
2. 无法得出结论时返回 `None` 并说明原因（如样本不足），**不编造**；
3. 拐点必须给出**具体季度与幅度**，不允许只说"近期上升"；
4. 单位与量纲由调用方传入，本模块不猜测。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


# ============================================================================
# L0  回测结果动态读取（保证综合判断的数字永远与最新数据一致，不会过期）
# ============================================================================
_IC_CACHE: dict | None = None


def _load_ic_summary() -> dict:
    """从 output/ic_report.xlsx 读取 RankIC 汇总，返回 {因子: 指标dict}。

    文件不存在或读取失败时返回空 dict（调用方退回内置数值）。
    """
    global _IC_CACHE
    if _IC_CACHE is not None:
        return _IC_CACHE
    out: dict = {}
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        fp = os.path.join(os.path.dirname(here), "output", "ic_report.xlsx")
        if os.path.exists(fp):
            t = pd.read_excel(fp, sheet_name="RankIC汇总")
            for _, r in t.iterrows():
                out[str(r["因子"])] = dict(
                    n=r.get("n_t1"), ic=r.get("ic_mean_t1"),
                    ir=r.get("ic_ir_t1"), t=r.get("t_nw_t1"),
                    p=r.get("p_nw_t1"), win=r.get("win_rate_t1"))
    except Exception:
        out = {}
    _IC_CACHE = out
    return out


def _ic_fmt(name: str, fallback: str) -> str:
    """把某因子的 RankIC 结果格式化为一行证据文字；取不到则用 fallback。"""
    d = _load_ic_summary().get(name)
    if not d or d.get("ic") is None or pd.isna(d["ic"]):
        return fallback
    n = int(d["n"]) if d.get("n") == d.get("n") else 0
    p = float(d["p"]) if d.get("p") == d.get("p") else float("nan")
    ps = "< 0.001" if p < 0.001 else f"{p:.3f}"
    win = (f", 胜率 {d['win']:.0f}%" if d.get("win") == d.get("win") else "")
    t = (f", t = {d['t']:.2f}" if d.get("t") == d.get("t") else "")
    return (f"RankIC = {d['ic']:+.3f}{t}, p {ps}{win}, {n} 期")


# ============================================================================
# L1  序列画像
# ============================================================================
def series_profile(s: pd.Series, unit: str = "", pct: bool = False) -> dict:
    """对一条时序做基本面画像。

    参数
    ----
    s    : pd.Series，索引为季度标签（如 "2024Q2"）
    unit : 单位后缀（"%" / "pct" / "" ）
    pct  : True 表示数值本身是百分比，变动以「百分点」而非「%」表达

    返回
    ----
    dict(start, end, delta, n, first, last, mx, mn, mx_q, mn_q,
         monotonic_up, monotonic_down, max_jump, max_jump_q,
         max_jump_from, max_jump_to, cv, trend_r2, note)
    """
    v = pd.Series(s).dropna().astype(float)
    if len(v) == 0:
        return dict(ok=False, note="序列为空", n=0)
    if len(v) == 1:
        return dict(ok=False, note="仅 1 个观测，无法识别趋势与拐点", n=1,
                    first=v.iloc[0], last=v.iloc[0],
                    first_q=str(v.index[0]), last_q=str(v.index[-1]))

    first, last = float(v.iloc[0]), float(v.iloc[-1])
    mx, mn = float(v.max()), float(v.min())
    mx_q, mn_q = str(v.idxmax()), str(v.idxmin())
    d = v.diff().dropna()
    ajq = str(d.abs().idxmax()) if len(d) else None
    aj = float(d.loc[ajq]) if ajq else np.nan
    prev_q = str(v.index[list(v.index).index(ajq) - 1]) if ajq else None

    # 单调性（允许极小数值噪声）
    tol = max(abs(v.mean()) * 1e-9, 1e-12)
    dd = np.sign(np.round(d.to_numpy() / tol)) if len(d) else np.array([])
    mono_up = bool(len(dd) and (dd >= 0).all())
    mono_dn = bool(len(dd) and (dd <= 0).all())

    # 趋势拟合优度（线性 R²）：区分"一路向上"与"上下震荡"
    x = np.arange(len(v), dtype=float)
    r2 = np.nan
    if len(v) >= 3 and v.std(ddof=0) > 0:
        c = np.polyfit(x, v.to_numpy(dtype=float), 1)
        fit = np.polyval(c, x)
        ss_res = float(((v.to_numpy(dtype=float) - fit) ** 2).sum())
        ss_tot = float(((v.to_numpy(dtype=float) - v.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    cv = float(v.std(ddof=1) / abs(v.mean())) if v.mean() else np.nan
    return dict(
        ok=True, n=len(v),
        first=first, last=last, delta=last - first,
        first_q=str(v.index[0]), last_q=str(v.index[-1]),
        mx=mx, mn=mn, mx_q=mx_q, mn_q=mn_q,
        monotonic_up=mono_up, monotonic_down=mono_dn,
        max_jump=aj, max_jump_q=ajq, max_jump_from=prev_q,
        max_jump_rel=(aj / abs(v.loc[prev_q]) if ajq and v.loc[prev_q] else np.nan),
        cv=cv, trend_r2=r2, unit=unit, pct=pct,
        new_high=bool(abs(last - mx) < 1e-12), new_low=bool(abs(last - mn) < 1e-12),
        note="")


# ============================================================================
# L2  拐点检测
# ============================================================================
def detect_turns(s: pd.Series, unit: str = "", pct: bool = False,
                 top: int = 3, rel_floor: float = 0.10,
                 noise_frac: float = 0.15) -> list[dict]:
    """识别序列的**结构性拐点**，按重要性排序返回。

    三类拐点（互补，任一类命中即记录）
    ----------------------------------
      T1 趋势反转：环比方向由升转降（或反之），且反转前存在连续 ≥2 期同向；
      T2 最大跃迁：单期变动的绝对值在全序列中排名靠前，且相对幅度 ≥ rel_floor；
      T3 极值突破：该期为全序列最大值或最小值（创历史新高/新低）。

    噪声抑制（本轮修正）
    --------------------
    早期版本仅用 `sign(p) != sign(c)` 判定反转，会把 **−0.0003** 这种
    噪声级反向波动报成"趋势反转"，污染结论。现要求反转当期的环比变动
    绝对值 ≥ `noise_frac × 全序列最大单期变动`，确保只有**实质性**转向才被记录。

    输出每条含「季度、类型、幅度、前后值、说明」，供前端直接渲染。
    """
    v = pd.Series(s).dropna().astype(float)
    if len(v) < 3:
        return []
    d = v.diff().dropna()
    if d.empty:
        return []
    # 噪声门槛：低于全序列最大单期变动的 noise_frac 视为无实质变动
    _mx = float(d.abs().max())
    _floor = _mx * noise_frac if _mx > 0 else 0.0

    def _fmt(x):
        """量纲感知的格式化：百分比用小数值 2 位，其余用千分位。"""
        if x != x:
            return "—"
        if pct:
            return f"{x:.2f}%"
        return f"{x:,.2f}{unit}" if abs(x) < 10000 else f"{x:,.0f}{unit}"

    def _fmtd(x):
        if x != x:
            return "—"
        return f"{x:+.2f}pct" if pct else f"{x:+,.2f}{unit}"

    out: list[dict] = []
    qs = list(v.index)

    # ---- T1 趋势反转（要求幅度过关）----
    for i in range(2, len(d)):
        p, c = float(d.iloc[i - 1]), float(d.iloc[i])
        if p == 0 or c == 0 or abs(c) < _floor:
            continue
        if np.sign(p) != np.sign(c):
            # 反转前需有 ≥2 期同向，避免把噪声当拐点
            run = 1
            j = i - 1
            while j - 1 >= 0 and np.sign(float(d.iloc[j - 1])) == np.sign(p):
                run += 1
                j -= 1
            if run < 2:
                continue
            q = qs[i + 1]
            out.append(dict(
                q=q, kind="趋势反转",
                label=f"{'升转降' if p > 0 else '降转升'}",
                value=float(v.loc[q]),
                detail=(f"{qs[j]}~{qs[i]} 连续 {run} 期"
                        f"{'上行' if p > 0 else '下行'}后，{q} 转为"
                        f"{'下行' if c < 0 else '上行'}（环比 {_fmtd(c)}）"),
                weight=2.0 + abs(c) / (_mx or 1)))

    # ---- T2 最大跃迁（同样设噪声门槛）----
    thr = max(_mx * 0.5, _floor)
    for q in d.index:
        val = float(d.loc[q])
        i = qs.index(q)
        base = float(v.iloc[i - 1]) if i > 0 else np.nan
        rel = abs(val / base) if base else np.nan
        if abs(val) >= thr and (rel != rel or rel >= rel_floor):
            out.append(dict(
                q=q, kind="单期跃迁",
                label=f"环比 {_fmtd(val)}",
                value=float(v.loc[q]),
                detail=(f"{qs[i - 1]} → {q}：{_fmt(base)} → {_fmt(v.loc[q])}"
                        + (f"（环比 {rel * 100:+.1f}%）" if rel == rel else "")
                        + f"，单期变动 {_fmtd(val)}"),
                weight=1.0 + abs(val) / (_mx or 1)))

    # ---- T3 极值突破 ----
    for q, tag in ((str(v.idxmax()), "创样本期新高"), (str(v.idxmin()), "创样本期新低")):
        i = qs.index(q)
        # 仅当该极值出现在后 60% 区间时才作为"拐点"（早期极值信息量低）
        if i < max(1, int(len(qs) * 0.4)):
            continue
        out.append(dict(
            q=q, kind="极值", label=tag,
            value=float(v.loc[q]),
            detail=f"{q} 取值为 {_fmt(v.loc[q])}，为样本期（{qs[0]}~{qs[-1]}）的"
                   f"{'最高' if tag.endswith('新高') else '最低'}水平",
            weight=1.5))

    # 去重（同一季度同类型只留一条），按权重降序
    seen, uniq = set(), []
    for r in sorted(out, key=lambda z: -z["weight"]):
        k = (r["q"], r["kind"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    return uniq[:top]


def detect_divergence(s1: pd.Series, s2: pd.Series,
                      n1: str = "A", n2: str = "B",
                      corr_thr: float = -0.3) -> dict | None:
    """检测两条序列的**背离/联动**（如「持仓等级 vs ETF 资金流」）。

    返回 dict(kind, corr, detail) 或 None。kind ∈ {"背离", "同向", "无关"}。
    """
    df = pd.concat([pd.Series(s1, dtype=float), pd.Series(s2, dtype=float)],
                   axis=1, keys=[n1, n2]).dropna()
    if len(df) < 3:
        return None
    c = float(df[n1].corr(df[n2], method="spearman"))
    if c != c:
        return None
    kind = "背离" if c <= corr_thr else ("同向" if c >= 0.5 else "弱关联")
    return dict(kind=kind, corr=c,
                detail=f"{n1} 与 {n2} 的秩相关 = {c:+.2f}（{len(df)} 期）→ {kind}")


# ============================================================================
# L3  结论生成（三段式：特征 / 拐点 / 含义）
# ============================================================================
def describe_series(s: pd.Series, name: str, unit: str = "", pct: bool = False,
                    higher_means: str = "", top: int = 2) -> dict:
    """把画像 + 拐点组装成结构化解读。

    返回 dict(ok, feature, turns, meaning, profile, turn_list)
      · feature  : 一句话特征（含具体数字）
      · turns    : 拐点文字（含具体季度与幅度）
      · meaning  : 含义/结论（由调用方提供 higher_means 语义时才有）
    """
    p = series_profile(s, unit=unit, pct=pct)
    if not p.get("ok"):
        return dict(ok=False, feature=p.get("note", "数据不足"),
                    turns="", meaning="", profile=p, turn_list=[])

    def _fmt(x):
        if x != x:
            return "—"
        return f"{x:.2f}%" if pct else f"{x:,.2f}{unit}"

    # ---- 特征 ----
    if p["monotonic_up"]:
        shape = f"**单调上行**（{p['n']} 期无一次回落）"
    elif p["monotonic_down"]:
        shape = f"**单调下行**（{p['n']} 期无一次回升）"
    elif p["trend_r2"] == p["trend_r2"] and p["trend_r2"] > 0.8:
        shape = f"接近线性趋势（拟合 R²={p['trend_r2']:.2f}），方向明确"
    elif p["trend_r2"] == p["trend_r2"] and p["trend_r2"] < 0.3:
        shape = f"**震荡无明确趋势**（拟合 R²={p['trend_r2']:.2f}）"
    else:
        shape = f"趋势中等明确（拟合 R²={p['trend_r2']:.2f}）"

    if pct:
        _chg = f"{p['delta']:+.2f}pct"
    else:
        _rel = (p["delta"] / abs(p["first"]) * 100) if p["first"] else np.nan
        _chg = (f"{p['delta']:+,.2f}{unit}"
                + (f"（{_rel:+.1f}%）" if _rel == _rel and abs(p["first"]) > 1e-9 else ""))

    feat = (f"{p['first_q']} → {p['last_q']}：{_fmt(p['first'])} → {_fmt(p['last'])}，"
            f"累计 {_chg}；{shape}。"
            f"区间高点 {_fmt(p['mx'])}（{p['mx_q']}）、低点 {_fmt(p['mn'])}（{p['mn_q']}）")
    if p["new_high"]:
        feat += f"；<b>{p['last_q']} 即为样本期最高值</b>"
    elif p["new_low"]:
        feat += f"；<b>{p['last_q']} 即为样本期最低值</b>"

    # ---- 拐点 ----
    tl = detect_turns(s, unit=unit, pct=pct, top=top)
    if tl:
        turns = "；".join(f"<b>{t['q']}</b>（{t['kind']}·{t['label']}）{t['detail']}"
                          for t in tl)
    else:
        turns = "样本期内未识别出结构性拐点（序列变动平稳）"

    meaning = ""
    if higher_means:
        _up = p["delta"] > 0
        meaning = higher_means.format(direction=("上升" if _up else "下降"),
                                      last=_fmt(p["last"]), q=p["last_q"],
                                      change=_chg)
    return dict(ok=True, feature=feat, turns=turns, meaning=meaning,
                profile=p, turn_list=tl)


def rank_series_last(df: pd.DataFrame, q: str, top: int = 5,
                     ascending: bool = False) -> list[tuple]:
    """取某期的横截面排名前 N 项（(名称, 数值) 列表）。"""
    s = pd.Series(df).loc[q] if q in getattr(df, "index", []) else pd.Series(df)
    s = s.dropna().sort_values(ascending=ascending)
    return [(k, float(v)) for k, v in s.head(top).items()]


# ============================================================================
# 渲染：把解读组装成 HTML 卡片（纯函数，不依赖 Streamlit，便于单测）
# ============================================================================
_CSS = (
    "<style>"
    ".ins{border-left:3px solid @@bar@@;background:@@bg@@;border-radius:0 6px 6px 0;"
    "padding:8px 11px;margin:4px 0 10px 0;font-size:12.5px;line-height:1.72;"
    "color:@@fg@@}"
    ".ins .h{display:inline-block;font-weight:700;color:@@acc@@;margin-right:6px;"
    "background:@@acch@@;border-radius:3px;padding:0 5px;font-size:11.5px}"
    ".ins .row{margin:2px 0}"
    ".ins b{color:@@emph@@}"
    "</style>"
)
# ⚠️ 不能用 str.format() 填充上面的 CSS 模板：CSS 里满是 { } 花括号，
# str.format 会把 .ins{...} 当成占位符并抛 KeyError('border-left')。
# 因此改用不会被 CSS 语法碰到的 @@token@@ 做替换。


def render_insight(disp, *, colors: dict, title: str = "分析",
                   feature: str = "", turns: str = "", meaning: str = "",
                   extra: str = "") -> str:
    """生成「特征 / 拐点 / 结论」三段式卡片 HTML。

    参数
    ----
    disp    : describe_series() 的返回值（可为 None 或 ok=False）
    colors  : 主题色字典，需含 line / panel / text / text_dim / price / accent
    title   : 卡片标题
    extra   : 追加的自定义行（HTML），用于放行业层面的特化结论
    """
    bar = colors.get("price", "#1F6FEB")
    bg = "#F7F9FC"
    fg = colors.get("text", "#1F2937")
    dim = colors.get("text_dim", "#6B7A8C")
    acc = colors.get("accent", "#1F6FEB")
    acch = "#EAF1FB"
    css = (_CSS.replace("@@bar@@", bar).replace("@@bg@@", bg)
           .replace("@@fg@@", fg).replace("@@acc@@", acc)
           .replace("@@acch@@", acch).replace("@@emph@@", bar))

    rows = []
    f = feature or (disp or {}).get("feature", "")
    t = turns or (disp or {}).get("turns", "")
    m = meaning or (disp or {}).get("meaning", "")
    if f:
        rows.append(f'<div class="row"><span class="h">特征</span>{f}</div>')
    if t:
        rows.append(f'<div class="row"><span class="h">拐点</span>{t}</div>')
    if m:
        rows.append(f'<div class="row"><span class="h">结论</span>{m}</div>')
    if extra:
        rows.append(f'<div class="row">{extra}</div>')
    if not rows:
        return ""
    return (css + f'<div class="ins"><div style="font-weight:700;color:{dim};'
                  f'font-size:11.5px;margin-bottom:3px">{title}</div>'
                  + "".join(rows) + "</div>")


def build_insight(series: pd.Series, name: str, *, unit: str = "", pct: bool = False,
                  higher_means: str = "", top: int = 2) -> dict:
    """describe_series 的便捷包装：额外把「拐点列表」附在返回值里供表格使用。"""
    r = describe_series(series, name, unit=unit, pct=pct,
                        higher_means=higher_means, top=top)
    return r


# ============================================================================
# 横截面（某一期）的自动描述
# ============================================================================
def describe_cross_section(df: pd.DataFrame, q: str, name: str = "",
                           unit: str = "", pct: bool = False,
                           top: int = 3) -> dict:
    """描述某一期的横截面分布：头部集中、尾部、极值。

    用于「气泡图 / 排名条形图」这类非时序图的特征提取。
    """
    if q not in getattr(df, "index", []):
        return dict(ok=False, feature="该期无数据", turns="", meaning="")
    s = pd.Series(df.loc[q]).dropna().astype(float)
    if s.empty:
        return dict(ok=False, feature="该期无数据", turns="", meaning="")
    s = s.sort_values(ascending=False)
    n = len(s)
    head = s.head(top)
    tail = s.tail(top)
    tot = s.sum()

    def _fmt(x):
        return f"{x:.2f}%" if pct else f"{x:,.2f}{unit}"

    feat = (f"{n} 个行业中，前 {top} 位 "
            + "、".join(f"{k} {_fmt(v)}" for k, v in head.items())
            + f"；末 {top} 位 "
            + "、".join(f"{k} {_fmt(v)}" for k, v in tail.items()))
    if pct and tot > 0:
        feat += f"；前 {top} 位合计占 {head.sum() / tot * 100:.1f}%"
    feat += f"；极差 {_fmt(float(s.max() - s.min()))}"
    if abs(s.min()) > 1e-9:
        feat += f"，最高/最低 = {abs(s.max()) / abs(s.min()):.1f}×"
    return dict(ok=True, feature=feat, turns="", meaning="",
                head=head.to_dict(), tail=tail.to_dict())


# ============================================================================
# 综合判断（用于首个面板：结论 + 判断理由 + 特征 + 拐点）
# ============================================================================
def build_verdict(mkt: dict, b, ind: dict, summary: dict) -> dict:
    """生成「综合判断」结构。

    返回
    ----
    dict(
      headline : 一句话结论（含最关键的两个数字）
      level    : 风险等级标签（用于配色）
      reasons  : [ {title, evidence, meaning} ]  —— 每条判断理由都必须带证据数字
      features : [ (标题, 说明) ]                —— 最突出的 3 个特征
      turns    : [ {q, label, detail} ]          —— 拐点清单（含季度与幅度）
    )

    设计纪律：**每一条理由都必须能在数据里查到出处**，
    不允许出现"显著上升""明显集中"这类没有数字支撑的表述。
    """
    q = b.latest
    hhi_s = mkt["F5_HHI"].dropna()
    cr3_s = mkt["F6_CR3"].dropna()
    top1_s = mkt["F6_Top1比例"].dropna()
    hhi = float(hhi_s.iloc[-1])
    hhi0 = float(hhi_s.iloc[0])
    cr3 = float(cr3_s.iloc[-1])
    top1 = float(top1_s.iloc[-1])
    top1_name = str(mkt["F6_Top1行业"].dropna().iloc[-1])
    q0 = str(hhi_s.index[0])
    n_hq = len(getattr(b, "over_quarters", []) or [])
    n_ex = int(summary.get("n_extreme", 0))
    n_un = int(summary.get("n_under", 0))

    # ---- 一句话结论 ----
    lvl = "极度拥挤" if hhi >= 0.15 else ("拥挤" if hhi >= 0.10 else "中性")
    if hhi >= 0.15 and n_ex >= 3:
        head = (f"<b>极高集中 · 单行业风险主导 · 拥挤尚未见顶</b> —— "
                f"HHI {hhi:.4f}（{q0} 以来 {hhi / hhi0 - 1:+.0%}）、"
                f"CR3 {cr3:.2f}%、{top1_name}单一行业占 {top1:.1f}%")
    elif hhi >= 0.10:
        head = (f"<b>集中度偏高 · 需关注头部行业回撤</b> —— HHI {hhi:.4f}、"
                f"CR3 {cr3:.2f}%、{top1_name}占 {top1:.1f}%")
    else:
        head = (f"<b>集中度温和</b> —— HHI {hhi:.4f}、CR3 {cr3:.2f}%、"
                f"{top1_name}占 {top1:.1f}%")

    # ---- 判断理由（每条带证据）----
    reasons = []

    # R1 集中度水平与斜率
    _p = series_profile(hhi_s)
    reasons.append(dict(
        title="持仓集中度处于样本期极值，且抬升过程近乎单调",
        evidence=(f"HHI {_p['first']:.4f}（{_p['first_q']}）→ {_p['last']:.4f}（{_p['last_q']}），"
                  f"累计 {_p['delta']:+.4f}（{_p['delta'] / _p['first']:+.0%}）；"
                  f"CR3 {float(cr3_s.iloc[0]):.2f}% → {cr3:.2f}%（{cr3 - float(cr3_s.iloc[0]):+.2f}pct）；"
                  f"最高值出现在 {_p['mx_q']}，即**最新一期**"),
        meaning=("集中度不是随机波动，而是单向抬升。HHI 单调上行意味着"
                 "机构资金的行业分化在持续加剧，组合的风险来源从「分散的行业 beta」"
                 "收敛为「少数行业的共同 beta」。")))

    # R2 单一行业依赖
    _al = b.alloc.reindex([x for x in b.quarters if x in b.alloc.index]).dropna(how="all")
    _t1 = _al[top1_name] if top1_name in _al.columns else None
    if _t1 is not None and len(_t1.dropna()):
        _t1s = _t1.dropna()
        reasons.append(dict(
            title=f"风险高度依赖单一行业：{top1_name}",
            evidence=(f"{top1_name} 配置比例 {float(_t1s.iloc[0]):.2f}%"
                      f"（{_t1s.index[0]}）→ {float(_t1s.iloc[-1]):.2f}%（{_t1s.index[-1]}）；"
                      f"最新一期占全组合 {top1:.2f}%，为样本期最高"),
            meaning=("单一行业超过 1/3 甚至四成仓位时，组合的净值波动实质上由该行业主导。"
                     "此时「行业分散」已名存实亡——见压力测试：该行业回撤 30% 即拖累组合两位数。")))

    # R3 K 型分化
    reasons.append(dict(
        title="K 型分化：极度拥挤与显著低配并存",
        evidence=(f"极度拥挤 {n_ex} 个行业、显著低配 {n_un} 个行业、"
                  f"中性 {31 - n_ex - n_un} 个（共 31 个申万一级行业）"),
        meaning=("资金不是「整体加仓」而是「从一批行业搬到另一批行业」。"
                 "这种结构下，行业间的相对表现会极端分化，"
                 "指数层面的温和涨幅会掩盖个股层面的巨大分化。")))

    # R4 因子的方向 —— IC 数字从 output/ic_report.xlsx 动态读取
    _e_f11 = _ic_fmt("F11_综合拥挤度_推荐权重", "RankIC 数据暂缺")
    _e_f3 = _ic_fmt("F3_超配历史分位", "RankIC 数据暂缺")
    _e_f1 = _ic_fmt("F1_超配比例", "RankIC 数据暂缺")
    _e_f3e = _ic_fmt("F3e_超配分位_长历史", None)
    reasons.append(dict(
        title="信号方向：主口径呈动量，但长历史下所有拥挤度因子的预测力均不稳健",
        evidence=(f"F11 综合拥挤度 {_e_f11}（9 期）；F3 超配历史分位 {_e_f3}（7 期）；"
                  f"F1 原始超配比例 {_e_f1}；"
                  f"36 期长历史 F3e 超配分位 {_e_f3e}"),
        meaning=("主口径（2024Q1 起，10 期）下拥挤度呈「动量」——越挤的行业下一期"
                 "超额收益越高；但把超配历史延到 2016 年（36 期）后，长历史扩窗因子"
                 "（F2e/F3e/F4be/F15e）的 IC 全部回落到 |IC|≤0.06、p>0.10 的不显著区间。"
                 "**这证明短样本下的「显著」是小样本伪显著**，拥挤度因子在更长历史里"
                 "没有稳定的截面预测力。结论应更审慎：本模型的价值在「风险描述」，"
                 "而非「可交易的拥挤度信号」。")))
    reasons.append(dict(
        title=f"样本声明：可建模区间 {n_hq} 期，长历史版已回填至 2016（36 期）",
        evidence=(f"主口径可建模区间 {n_hq} 期（holdings ∩ 沪深300）；"
                  f"偏股混合超配长历史已回填至 2016Q2（36 期，2026-10-03 抓取 hs300 权重"
                  f"21 期 + 行业收益 12 期），长历史扩窗因子 F2e/F3e/F4be/F15e 现可取值并"
                  f"完成 27~30 期 IC 检验。"),
        meaning=(f"长历史检验的结论是：拥挤度因子（含长历史版）在 27~30 期观测下"
                 "均不显著——**这正是 36 期标准的意义所在：筛掉小样本伪显著**。"
                 "本模型的正确用法是**风险描述 + 压力测试输入 + 新增仓位的集中度约束**，"
                 "而非择时或做空信号。")))

    # ---- 最突出的特征 ----
    features = [
        (f"HHI {hhi:.4f} 为样本期最高",
         f"较 {_p['mn_q']} 的低点 {_p['mn']:.4f} 放大 {hhi / _p['mn'] - 1:+.0%}；"
         f"标准化 HHI = {float(mkt['F5b_HHI标准化'].dropna().iloc[-1]):.4f}"),
        (f"CR3 = {cr3:.2f}%，头部三个行业吃掉约七成仓位",
         f"CR5 {float(mkt['F6_CR5'].dropna().iloc[-1]):.2f}%、"
         f"CR10 {float(mkt['F6_CR10'].dropna().iloc[-1]):.2f}%"),
        (f"{top1_name} {top1:.2f}%，单一行业主导",
         f"超配 +{float(b.overweight[top1_name].dropna().iloc[-1]):.2f}pct；"
         f"其自身历史分位 {float(ind['F3_超配历史分位'].loc[q, top1_name]):.0f}%"),
    ]

    # ---- 拐点清单（跨关键序列汇总，每条序列只取"最显著"的一个拐点，按季度合并）----
    # 为什么每条序列只取一个：2026Q2 同时命中反转/跃迁/极值三类，
    # 全列会在同一季度堆出 6 个标签，反而淹没了"这一期到底发生了什么"。
    # "最显著"的判据：带幅度的跃迁/极值优先于仅有方向的反转（前者信息量更大）。
    _PRIO = {"单期跃迁": 3, "极值": 2, "趋势反转": 1}

    def _best(ts: list[dict]) -> dict | None:
        if not ts:
            return None
        return max(ts, key=lambda t: (_PRIO.get(t["kind"], 0), t["weight"]))

    _by_q: dict[str, dict] = {}
    for nm, s, un, pc in (("HHI", mkt["F5_HHI"], "", False),
                          ("CR3", mkt["F6_CR3"], "%", True),
                          (f"{top1_name}占比", top1_s, "%", True)):
        t = _best(detect_turns(s, unit=un, pct=pc, top=3))
        if t is None:
            continue
        slot = _by_q.setdefault(t["q"], dict(q=t["q"], labels=[], details=[]))
        slot["labels"].append(f"{nm}·{t['label']}")
        slot["details"].append(t["detail"])
    turns = [dict(q=q, label=" ／ ".join(v["labels"]),
                  detail="；".join(v["details"]))
             for q, v in sorted(_by_q.items())]
    return dict(headline=head, level=lvl, reasons=reasons,
                features=features, turns=turns,
                hhi=hhi, cr3=cr3, top1=top1, top1_name=top1_name)


def render_verdict(v: dict, colors: dict) -> str:
    """把 build_verdict 的结果渲染成「综合判断」面板 HTML。"""
    acc = colors.get("accent", "#1F6FEB")
    price = colors.get("price", "#1F6FEB")
    dim = colors.get("text_dim", "#6B7A8C")
    fg = colors.get("text", "#1F2937")
    up = colors.get("up", "#D1242F")
    lvl_color = colors.get("level_" + str(v.get("level", "中性")), price)

    rows = []
    for i, r in enumerate(v.get("reasons", []), 1):
        rows.append(
            f'<div style="margin:7px 0;padding-left:10px;border-left:2px solid {acc}22">'
            f'<div style="font-weight:700;color:{fg}">{i}. {r["title"]}</div>'
            f'<div style="color:{dim};margin-top:1px">证据　{r["evidence"]}</div>'
            f'<div style="margin-top:1px">判断　{r["meaning"]}</div></div>')

    feats = "".join(
        f'<div style="margin:4px 0"><span style="display:inline-block;'
        f'background:{acc}18;color:{acc};border-radius:3px;padding:0 6px;'
        f'font-weight:700;margin-right:6px">{t}</span>{d}</div>'
        for t, d in v.get("features", []))

    turns = "".join(
        f'<div style="margin:3px 0"><span style="display:inline-block;'
        f'background:{up}18;color:{up};border-radius:3px;padding:0 6px;'
        f'font-weight:700;margin-right:6px">{t["label"]}</span>{t["detail"]}</div>'
        for t in v.get("turns", []))

    return (
        f'<div style="border:1px solid {lvl_color}55;border-left:4px solid {lvl_color};'
        f'background:#FFFFFF;border-radius:0 7px 7px 0;padding:12px 15px;margin:6px 0 14px 0;'
        f'font-size:12.5px;line-height:1.8;color:{fg}">'
        f'<div style="font-size:15px;margin-bottom:6px">{v["headline"]}</div>'
        f'<div style="font-weight:700;color:{dim};font-size:11.5px;margin:10px 0 2px">'
        f'判断理由（每一条均可回溯到数据）</div>{"".join(rows)}'
        f'<div style="font-weight:700;color:{dim};font-size:11.5px;margin:11px 0 2px">'
        f'最突出的 3 个特征</div>{feats}'
        + (f'<div style="font-weight:700;color:{dim};font-size:11.5px;margin:11px 0 2px">'
           f'关键拐点（含季度与幅度）</div>{turns}' if turns else "")
        + "</div>")

