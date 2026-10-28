# -*- coding: utf-8 -*-
"""机构持仓决策系统：六个度量、六项验证、决策表。

设计原则
--------
1. **数据可得性先于方法**：每个度量都带 `state`（available / proxy / blocked），
   blocked 的度量不进入决策表，只说明缺什么、需要多少次取数。
2. **未过验证的度量降级为参考**，不写「显著」二字（样本 <12 期直接降级）。
3. **决策动作只有四个**：超配 / 标配 / 低配 / 回避；每条含数字、上期对照、失效条件。
4. **不插值、不填补**：个股清单只列真实取到的记录，缺失留空。

当前取数约束（已实测）
----------------------
Wind MCP 的 `get_fund_holdings` 只能按「单只基金 × 单个报告期」查询，无批量参数；
本机未安装 WindPy / iFinDPy。全量基金×个股明细需 1074 只 × 43 期 ≈ 4.6 万次调用
（按每次 3 秒计约 40 小时），无法在单次会话完成。因此：
    · 4.1 主动加减仓       → available（行业聚合 + 价格漂移分解）
    · 4.2 分歧度           → blocked（需基金级明细），仅给市场层面代理
    · 4.3 个股持股变动     → available（每期前 50 只，按名称合并）
    · 4.4 日度行业暴露     → blocked（需基金日度复权净值）
    · 4.5 全持仓校准       → blocked（需半年报/年报全持仓）
    · 4.6 季报文本因子     → blocked（Wind MCP 无文本接口）
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

import attribution as atr
import backtest as bt
import config as C

DB_PATH = os.path.join(C.DATA, "db", "institution.db")

# A/H 双重主体别名（同一公司两地上市，接口按证券简称返回，需合并）
STOCK_ALIAS = {
    "中国海油": "中国海洋石油",
    "中国石油": "中国石油股份",
    "中国神华": "中国神华能源",
    "比亚迪": "比亚迪股份",
    "中国移动": "中国移动",
    "中国电信": "中国电信股份",
}

# 样本池口径断层：这几期重仓股持有基金数的均值出现台阶式跳变（+34%/+39%），
# 跨断点比较 Δ持股基金数会高估变化，页面与报告必须标注。
REGIME_BREAK = ("2019Q2", "2020Q2")

# ---------------------------------------------------------------------------
# 数据状态登记
# ---------------------------------------------------------------------------
DATA_STATUS = [
    dict(key="fund_stock_detail", name="基金×个股明细（全市场）", state="available",
         have="东方财富完整持仓明细：4937 只基金 × 7 期（2023–2026）",
         need="—", why="AKShare 直连东财公开披露；季报披露前十大、半年报披露全部（单基金最多 100 只）"),
    dict(key="full_holding", name="全持仓（半年报/年报）", state="available",
         have="半年报期（Q2/Q4）全持仓，用于校准前十大口径",
         need="—", why="已实现前十大 ÷ 全持仓的行业比值校准（见 8 前十大偏差校准）"),
    dict(key="nav_daily", name="基金日度复权净值", state="available",
         have="按需抓取，已覆盖日度暴露样本 60 只",
         need="全市场按需", why="AKShare 单基金一次返回全部历史净值；31 行业日收益已缓存"),
    dict(key="fund_industry", name="基金×申万行业配置", state="available",
         have="由持仓明细 + 申万成分映射（5220 只股票 / 31 行业）实时聚合",
         need="—", why="东财明细逐股带代码，可直接聚合到申万一级，不再依赖 Wind 行业表"),
    dict(key="stock_holdings", name="个股持股基金数（全市场）", state="available",
         have="每期约 2000–3200 只股票（按同披露口径两期比较）",
         need="—", why="全量明细聚合；季报期只含前十大，故只在同口径两期之间比较"),
    dict(key="fund_profile", name="基金档案与经理变更", state="blocked",
         have="0", need="全市场基金档案 + 经理任职起止",
         why="东财基金列表无经理任职区间字段；需另找数据源，暂不伪造"),
    dict(key="report_text", name="季报文本", state="blocked",
         have="0", need="每期每只基金的「投资策略和运作分析」正文",
         why="无结构化接口（官方只发 PDF），不用其他字段替代"),
]


def data_status() -> pd.DataFrame:
    """数据源状态。能算的一律现算，避免文档写着「7 期」而库里其实是 15 期。"""
    rows = [dict(x) for x in DATA_STATUS]
    try:
        import ak_data as akd
        qs = akd.latest_quarters_all()
        if qs:
            con = akd._con()
            n_fund = con.execute("SELECT COUNT(DISTINCT code) FROM fund_holdings").fetchone()[0]
            n_row = con.execute("SELECT COUNT(*) FROM fund_holdings").fetchone()[0]
            con.close()
            for r in rows:
                if r["key"] == "fund_stock_detail":
                    r["have"] = (f"东方财富完整持仓明细：{n_fund:,} 只基金 × {len(qs)} 期"
                                 f"（{qs[0]}–{qs[-1]}），{n_row:,} 行")
                elif r["key"] == "full_holding":
                    r["have"] = (f"半年报/年报期全持仓（{sum(1 for q in qs if q[-1] in '24')} 期），"
                                 f"用于校准前十大口径")
    except Exception:
        pass
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4.1 主动加减仓（双口径交叉验证）
# ---------------------------------------------------------------------------
def active_change(b, ind: dict, q: str | None = None) -> pd.DataFrame:
    """行业主动加减仓：占比口径 + 市值口径。

    占比口径：Δw = 被动漂移（价格）+ 主动（其余），沿用 attribution 的 GK 分解。
    市值口径：ΔMV − 价格效应 = 主动金额（亿元）。总市值为缓存重仓口径（偏低约 40%），
    故金额只作量级参考，比例与符号才是结论。
    """
    q = q or b.latest
    qs = list(b.quarters)
    i = qs.index(q)
    q0 = qs[i - 1] if i > 0 else None
    dec = atr.decompose(b.alloc, b.market["ret"])
    rows = {}
    for k in C.INDUSTRIES:
        sp = atr.split_one(dec, q, k)
        rows[k] = dict(d_alloc=sp.get("变化", np.nan),
                       passive=sp.get("被动贡献", np.nan),
                       active=sp.get("主动贡献", np.nan),
                       alloc0=sp.get("上季配置", np.nan),
                       alloc1=sp.get("本季配置", np.nan))
    t = pd.DataFrame(rows).T
    # 市值口径
    hv = getattr(b, "holdings_value", None)
    if hv is not None and q in getattr(hv, "index", []) and q0 in getattr(hv, "index", []):
        mv1 = b.alloc.loc[q] / 100 * float(hv.loc[q])
        mv0 = b.alloc.loc[q0] / 100 * float(hv.loc[q0])
        ret = b.market["ret"].loc[q].reindex(C.INDUSTRIES)
        price = mv0 * ret / 100
        t["mv1"] = mv1
        t["mv0"] = mv0
        t["d_mv"] = mv1 - mv0
        t["price_mv"] = price
        t["active_mv"] = t["d_mv"] - t["price_mv"]
        t["sign_agree"] = np.sign(t["active"]) == np.sign(t["active_mv"])
    else:
        for c in ("mv1", "mv0", "d_mv", "price_mv", "active_mv"):
            t[c] = np.nan
        t["sign_agree"] = False
    t.attrs["q"], t.attrs["q0"] = q, q0
    return t


def active_panel(b, ind: dict) -> pd.DataFrame:
    """全历史主动加减仓矩阵（季度 × 行业），用于六项验证。"""
    dec = atr.decompose(b.alloc, b.market["ret"])
    qs = list(b.quarters)
    out = pd.DataFrame(index=qs, columns=C.INDUSTRIES, dtype=float)
    for q in qs:
        if q not in b.alloc.index:
            continue
        for k in C.INDUSTRIES:
            sp = atr.split_one(dec, q, k)
            out.loc[q, k] = (sp or {}).get("主动贡献", np.nan)
    return out


# ---------------------------------------------------------------------------
# 4.2 分歧度（blocked）＋ 市场层面代理
# ---------------------------------------------------------------------------
def dispersion_proxy(t: pd.DataFrame) -> dict:
    """分歧度的**代理**：不是基金级加仓占比，只反映行业配置变化的离散程度。

    真正的分歧度需要「加仓基金数 / 减仓基金数 / 金额比」，必须基金级明细；
    当前不可得，故本函数明确标注为 proxy，不进入决策表。
    """
    d = t["d_alloc"].dropna()
    dz = t["dz"].dropna()
    hist_disp = None
    return dict(
        state="proxy",
        n_add=int((d > 0).sum()), n_cut=int((d < 0).sum()),
        cross_std=float(d.std(ddof=1)) if len(d) > 1 else np.nan,
        median=float(d.median()) if len(d) else np.nan,
        conflict=int(t["conflict"].sum()),
        note="行业 Δ配置 的横截面离散度；不是基金级加仓占比，不能判读「共识/对撞」",
    )


# ---------------------------------------------------------------------------
# 4.2b 分歧度（真实口径，样本 50 只）
# ---------------------------------------------------------------------------
FUND_IND_PATH = os.path.join(C.DATA_RAW, "fund_industry_50x4.json")


def load_fund_industry(path: str | None = None) -> dict:
    """读取基金×行业配置样本。缺失返回空结构，不伪造。"""
    path = path or FUND_IND_PATH
    if not os.path.exists(path):
        return {"funds": [], "data": {}, "note": "未找到 fund_industry_50x4.json"}
    import json
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def latest_fund_industry_period(path: str | None = None) -> tuple[str, str] | None:
    """样本里最新两期。Wind 的行业配置字段比前十大重仓滞后一期，
    2026Q3 实测只有名次、行业与占比为空，故最新可得通常是 2026Q2。"""
    payload = load_fund_industry(path)
    qs = set()
    for per in payload.get("data", {}).values():
        qs |= {k for k, v in per.items() if v}
    if len(qs) < 2:
        return None
    ordered = sorted(qs, key=lambda x: (int(x[:4]), int(x[-1])))
    return ordered[-1], ordered[-2]


def dispersion_real(q: str, q0: str, path: str | None = None,
                    min_delta: float = 0.10, normalize: bool = True) -> pd.DataFrame:
    """真实分歧度：每行业统计 加仓/减仓/不变 基金数。

    口径与边界
    ----------
    · 加仓 = 占净值比变化 > +min_delta（默认 0.10pp），减仓 = < -min_delta；
    · 只取接口返回的前 10 重仓行业，未出现视为 0——这会低估小比例持仓的变化，
      因此「不变」包含「两期都不在前十」，是本指标最大的已知偏差；
    · 金额比用「占净值比变化」的正负之和，不是真金额（缺基金资产净值绝对值）。

    归一化（默认开启，必不可少）
    ----------------------------
    半年报/年报期的行业配置表是全持仓汇总（覆盖约 88% 净值），
    季报期只能由前十大重仓股推导（覆盖约 60%）。若直接用原始占比做差，
    Q2→Q3 会把「覆盖率从 88% 掉到 61%」误算成「全面减仓」。
    因此统一除以每只基金每期的 coverage（该期行业占比合计），
    得到「占已覆盖组合的比例」，再做差。归一化后水平可比，但仍是
    「已披露部分」的结构，不是完整组合。
    """
    payload = load_fund_industry(path)
    funds = payload.get("funds", [])
    data = payload.get("data", {})
    if not funds or not data:
        return pd.DataFrame()
    codes = [f["code"] for f in funds]
    cov = payload.get("coverage", {}) if normalize else {}
    spec = payload.get("period_spec", {})
    rows = []
    for ind in C.INDUSTRIES:
        up = dn = fl = n = 0
        pos = neg = 0.0
        vals = []
        theme_drop = 0
        for c in codes:
            per = data.get(c, {})
            a = per.get(q, {}).get(ind)
            b = per.get(q0, {}).get(ind)
            if a is None and b is None:
                continue
            a = 0.0 if a is None else float(a)
            b = 0.0 if b is None else float(b)
            if normalize:
                ca_ = cov.get(c, {}).get(q) or 100.0
                cb_ = cov.get(c, {}).get(q0) or 100.0
                a = a / ca_ * 100 if ca_ > 0 else 0.0
                b = b / cb_ * 100 if cb_ > 0 else 0.0
            # 清洗规则：该基金本期或上期在本行业配置 >80% → 行业主题基金，
            # 其变化主要反映主题基金自身申赎，计入会污染行业分歧度。
            if max(a, b) > 80.0:
                theme_drop += 1
                continue
            n += 1
            vals.append(a)
            d = a - b
            if d > min_delta:
                up += 1
                pos += d
            elif d < -min_delta:
                dn += 1
                neg += -d
            else:
                fl += 1
        if n == 0:
            continue
        share = up / n
        div = 1 - abs(share - 0.5) * 2
        if share >= 0.70 and div < 0.40:
            tag = "共识形成"
        elif 0.40 <= share <= 0.60 and div > 0.70:
            tag = "多空对撞"
        elif share <= 0.30:
            tag = "系统性撤离"
        else:
            tag = "分歧"
        rows.append(dict(
            行业=ind, 样本基金数=n, 加仓基金数=up, 减仓基金数=dn, 不变基金数=fl,
            加仓占比=round(share * 100, 1), 分歧度=round(div, 2),
            加减仓比=(round(pos / neg, 2) if neg > 1e-9 else np.nan),
            横截面标准差=round(float(pd.Series(vals).std(ddof=1)), 2),
            判读=tag,
            平均配置=round(float(pd.Series(vals).mean()), 2),
            剔除主题基金=theme_drop,
        ))
    out = pd.DataFrame(rows)
    if not out.empty:
        out.attrs["q"], out.attrs["q0"] = q, q0
        out.attrs["n_funds"] = len(codes)
        out.attrs["note"] = ("仅前 10 行业，未出现视为 0；样本为 2026Q2 规模前 50 的偏股混合，"
                             "偏科技主题，不代表全市场")
        out.attrs["period_spec"] = spec
        out.attrs["normalized"] = bool(normalize)
    return out


# ---------------------------------------------------------------------------
# 4.3 个股持股变动
# ---------------------------------------------------------------------------
def stock_flow(q: str, q0: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """两期前 50 只重仓股的持股基金数变化。

    重要边界：接口只返回「按持股基金数排序的前 50 只」，因此
    「跌出前 50」会被误判为「退出」。表中用 in_top 标记两期是否都在榜内，
    退出/新进只作观察，不作买卖依据。
    """
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["name", "n0", "n1", "d_n", "mv1", "d_mv", "state"])
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            "SELECT quarter,name,n_funds,mv_yi FROM stock_holdings WHERE quarter IN (?,?)",
            con, params=(q0, q))
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame(columns=["name", "n0", "n1", "d_n", "mv1", "d_mv", "state"])
    # ---- 清洗（R19）：ST/退市剔除 + A/H 双重主体合并 ----
    _raw = len(df)
    df = df[~df["name"].astype(str).str.contains(r"退市|^\*?ST", regex=True)].copy()
    # 同一公司在 A/H 两地会被拆成两个名称（如 中国海油 / 中国海洋石油），
    # 不合并会重复计数。合并后基金数取 max（同一基金只应计一次），市值求和。
    df["name"] = df["name"].replace(STOCK_ALIAS)
    df = (df.groupby(["quarter", "name"], as_index=False)
            .agg(n_funds=("n_funds", "max"), mv_yi=("mv_yi", "sum")))
    _dropped = _raw - len(df)
    p = df.pivot_table(index="name", columns="quarter", values=["n_funds", "mv_yi"],
                       aggfunc="first")
    out = pd.DataFrame(index=p.index)
    out["n0"] = p[("n_funds", q0)] if ("n_funds", q0) in p else np.nan
    out["n1"] = p[("n_funds", q)] if ("n_funds", q) in p else np.nan
    out["mv1"] = p[("mv_yi", q)] if ("mv_yi", q) in p else np.nan
    out["mv0"] = p[("mv_yi", q0)] if ("mv_yi", q0) in p else np.nan
    out["d_n"] = out["n1"].fillna(0) - out["n0"].fillna(0)
    out["d_mv"] = out["mv1"].fillna(0) - out["mv0"].fillna(0)
    out["state"] = np.select(
        [out["n0"].isna() & out["n1"].notna(),
         out["n0"].notna() & out["n1"].isna(),
         out["n0"].notna() & out["n1"].notna()],
        ["新进前50", "退出前50", "两期在榜"], default="—")
    out = out[out["n1"].notna() | out["n0"].notna()]
    out = out.sort_values("d_n", ascending=False).reset_index()
    out.attrs["dropped"] = int(_dropped)
    out.attrs["break"] = (q in REGIME_BREAK) or (q0 in REGIME_BREAK)
    return out


# ---------------------------------------------------------------------------
# 4.4 / 4.5 / 4.6 状态
# ---------------------------------------------------------------------------
def daily_exposure_status() -> dict:
    return dict(state="blocked", name="日度行业暴露",
                reason="需 1074 只基金 2016 年至今的日度复权净值；Wind MCP 仅支持单基金查询",
                fallback="回退季频：信号在季报披露后约 15 个工作日可用，季度内不再更新",
                required="1074 只 × 1 次日度净值调用（约 2600 行/只）")


def full_holding_status() -> dict:
    return dict(state="blocked", name="全持仓校准",
                reason="半年报/年报全持仓需要逐基金逐期取数，当前无批量接口",
                fallback="前十大口径的行业占比合计约 46–48.5%，结论按「重仓口径」标注",
                required="1074 只 × 20 个半年/年度报告期")


def text_factor_status() -> dict:
    return dict(state="blocked", name="季报文本因子",
                reason="Wind MCP 无季报文本字段（投资策略/市场展望）",
                fallback="不做文本观点，不用其他字段替代",
                required="1074 份/期 × 43 期")


# ---------------------------------------------------------------------------
# 六项验证
# ---------------------------------------------------------------------------
def _monotonic(groups: list[float]) -> float:
    """分组均值与组序号的 Spearman 相关（1=完全单调递增）。"""
    g = pd.Series(groups).dropna()
    if len(g) < 3:
        return np.nan
    return float(stats.spearmanr(np.arange(len(g)), g.values).correlation)


def group_returns(fac: pd.DataFrame, fwd: pd.DataFrame, n_grp: int = 5) -> pd.DataFrame:
    ret_abs = None
    rows = []
    for gi in range(n_grp):
        lo, hi = gi * 100 / n_grp, (gi + 1) * 100 / n_grp
        vals = []
        for q in fac.index:
            if q not in fwd.index:
                continue
            s = fac.loc[q]
            r = fwd.loc[q]
            m = (s >= lo) & (s < hi) if gi < n_grp - 1 else (s >= lo) & (s <= hi)
            sel = s[m].dropna().index
            if len(sel) >= 2:
                vals.append(float(r.reindex(sel).mean()))
        v = pd.Series(vals)
        rows.append(dict(组=f"{lo:.0f}-{hi:.0f}", n期=len(v),
                         均值=float(v.mean()) if len(v) else np.nan,
                         胜率=float((v > 0).mean() * 100) if len(v) else np.nan))
    return pd.DataFrame(rows)


def six_checks(b, fac: pd.DataFrame, name: str, expect_positive: bool = True,
               min_n: int = 12) -> dict:
    """IC / 分组单调 / 剔Q4 / 分段 / 正交化 / 成本 六项。"""
    fwd = bt.forward_excess(b.market, lag=1)
    ic = bt.rank_ic_panel(fac, fwd).dropna()
    n = len(ic)
    nw_t, nw_p = bt.nw_tstat(ic)
    checks = []

    # 1 IC
    ok1 = (n >= min_n) and (nw_p == nw_p and nw_p < 0.10) and \
          (np.sign(ic.mean()) == (1 if expect_positive else -1))
    checks.append(dict(项="IC 检验", 通过=bool(ok1),
                       结果=f"n={n}，RankIC {ic.mean():+.3f}，NW t {nw_t:.2f}，p {nw_p:.3f}"))
    # 2 分组单调
    g = group_returns(fac, fwd, 5)
    mono = _monotonic(g["均值"].tolist())
    ok2 = pd.notna(mono) and abs(mono) >= 0.8 and np.sign(mono) == (1 if expect_positive else -1)
    checks.append(dict(项="分组单调性", 通过=bool(ok2),
                       结果=f"5 组均值 {', '.join(f'{v:+.2f}' for v in g['均值'])}；秩相关 {mono:+.2f}" if pd.notna(mono) else "样本不足"))
    # 3 剔 Q4
    fac_noq4 = fac.loc[[q for q in fac.index if not str(q).endswith("Q4")]]
    ic2 = bt.rank_ic_panel(fac_noq4, fwd).dropna()
    nw2, np2 = bt.nw_tstat(ic2)
    ok3 = (np2 == np2 and np2 < 0.10) and (np.sign(ic2.mean()) == np.sign(ic.mean()))
    checks.append(dict(项="剔 Q4 复检", 通过=bool(ok3),
                       结果=f"剔后 n={len(ic2)}，RankIC {ic2.mean():+.3f}，NW p {np2:.3f}"))
    # 4 分段
    seg = []
    for a, z_ in [("2019Q3", "2022Q4"), ("2023Q1", "2026Q2")]:
        x = ic.loc[a:z_]
        if len(x) >= 6:
            t_, p_ = bt.nw_tstat(x)
            seg.append((a, z_, len(x), float(x.mean()), float(t_), float(p_)))
    if seg:
        same = len({np.sign(s[3]) for s in seg}) == 1
        ok4 = same and all(s[5] < 0.10 for s in seg)
        detail = "；".join(f"{s[0]}–{s[1]} n={s[2]} IC {s[3]:+.3f} p {s[5]:.3f}" for s in seg)
    else:
        ok4, detail = False, "样本不足以分段（<6 期/段）"
    checks.append(dict(项="分段稳定性", 通过=bool(ok4), 结果=detail))
    # 5 正交化（对过去 4 季涨幅）
    ret = b.market["ret"].reindex(columns=C.INDUSTRIES)
    mom4 = (1 + ret / 100).rolling(4).apply(np.prod, raw=True).sub(1) * 100
    resid = pd.DataFrame(index=fac.index, columns=fac.columns, dtype=float)
    for q in fac.index:
        if q not in mom4.index:
            continue
        d = pd.concat([fac.loc[q], mom4.loc[q]], axis=1).dropna()
        if len(d) < 10:
            continue
        X = np.column_stack([np.ones(len(d)), d.iloc[:, 1].to_numpy(float)])
        beta = np.linalg.lstsq(X, d.iloc[:, 0].to_numpy(float), rcond=None)[0]
        resid.loc[q] = fac.loc[q] - (beta[0] + beta[1] * mom4.loc[q])
    ic3 = bt.rank_ic_panel(resid, fwd).dropna()
    nw3, np3 = bt.nw_tstat(ic3)
    ok5 = (len(ic3) >= min_n) and (np3 == np3 and np3 < 0.10) and \
          (np.sign(ic3.mean()) == np.sign(ic.mean()))
    checks.append(dict(项="动量正交化", 通过=bool(ok5),
                       结果=f"残差 n={len(ic3)}，RankIC {ic3.mean():+.3f}，NW p {np3:.3f}"))
    # 6 成本
    from change_analysis import long_short, perf
    s = long_short(b, fac, k=8)
    pf = perf(s)
    turn = 2.0  # 季度双边换手近似：上下各 8 只、全换 = 200%
    net = pf["ann"] - turn * 4 * 0.001 * 2 * 100 if pd.notna(pf["ann"]) else np.nan
    ok6 = pd.notna(net) and net > 0
    checks.append(dict(项="交易成本(20bp)", 通过=bool(ok6),
                       结果=f"毛年化 {pf['ann']:+.2f}%、扣费后 {net:+.2f}%（假设年换手 {turn*4*100:.0f}%）"))
    n_pass = sum(c["通过"] for c in checks)
    status = "可用" if n_pass == 6 else ("参考" if n_pass >= 4 else "不可用")
    return dict(指标=name, 通过项=int(n_pass), 总项=len(checks), 状态=status,
                检验=pd.DataFrame(checks), n=n, ic_mean=float(ic.mean()),
                nw_t=float(nw_t), nw_p=float(nw_p))


# ---------------------------------------------------------------------------
# 行业决策表
# ---------------------------------------------------------------------------
ACTION_ORDER = ["超配", "标配", "低配", "回避"]


def _consistency(active_pp, add_share) -> str:
    """主动加减仓（2026Q3 单季、全市场）与 样本基金加仓占比（跨两季）是否同向。"""
    if pd.isna(active_pp) or pd.isna(add_share):
        return "未采集"
    if active_pp > 0 and add_share < 40:
        return "冲突（本期加、跨季减）"
    if active_pp < 0 and add_share > 60:
        return "冲突（本期减、跨季加）"
    return "一致"


def _conflict_note(active_pp, add_share, q0_, q_) -> str:
    if pd.isna(active_pp) or pd.isna(add_share):
        return ""
    if active_pp > 0 and add_share < 40:
        return (f"；<b>口径冲突</b>：2026Q3 主动 {active_pp:+.2f}pp（全市场单季），"
                f"但样本基金 {q0_}→{q_} 仅 {add_share:.0f}% 加仓——可能是跨期反弹，"
                f"也可能样本偏差，下季需用同口径复核")
    if active_pp < 0 and add_share > 60:
        return (f"；<b>口径冲突</b>：2026Q3 主动 {active_pp:+.2f}pp，"
                f"但样本基金 {q0_}→{q_} 有 {add_share:.0f}% 加仓，需复核")
    return ""


def decision_brief(b, ind: dict, q: str | None = None,
                   disp_real=None) -> dict:
    """页面与报告共用的「本期决策表」构建器——避免两处各算一遍再对不上。

    返回 dict：t（本期变化表）、t0（上期）、active（主动加减仓）、
                dec（决策表）、q、q0、disp（真实分歧度，可能为 None）
    """
    import change_analysis as ca
    import ak_data as akd
    q = q or b.latest
    qs = list(b.quarters)
    q0 = qs[qs.index(q) - 1]
    t = ca.credibility_gate(b, ind, ca.delta_table(b, ind, q),
                            b.market.get("F8_样本基金数"))
    t0 = ca.credibility_gate(b, ind, ca.delta_table(b, ind, q0),
                             b.market.get("F8_样本基金数"))
    active = active_change(b, ind, q)
    if disp_real is None:
        try:
            dp = akd.latest_quarters()
            disp_real = akd.dispersion_ak(*dp) if dp else None
        except Exception:
            disp_real = None
    dec = decision_table(t, active, t0=t0, disp=disp_real)
    return dict(t=t, t0=t0, active=active, dec=dec, q=q, q0=q0, disp=disp_real)


def position_advice_from_dispersion(median_div: float,
                                    n_funds: int | None = None) -> dict:
    """按全市场分歧度中位数给仓位区间（6.3 的事前固定规则）。

    阈值写死、不看结果调参：
        <0.40 → 共识偏强，仓位可偏高 75%–85%
        0.40–0.70 → 分歧中性，55%–70%
        >0.70 → 多空对撞，降仓并缩短持有期 35%–50%

    与 `position_advice(disp_proxy)` 的区别：那个用的是行业 Δ配置 的横截面离散度
    代理，这个用的是基金级真实的加仓占比分歧度。
    """
    if median_div is None or pd.isna(median_div):
        return dict(median=np.nan, stance="数据不足", low=50, high=60,
                    why="分歧度缺失，按中性仓位处理")
    if median_div < 0.40:
        stance, low, high = "共识偏强", 75, 85
        why = "多数行业加仓方向一致，行业轮动可跟随"
    elif median_div > 0.70:
        stance, low, high = "多空对撞", 35, 50
        why = "半数行业处于对撞，方向不明，降仓位并缩短持有期"
    else:
        stance, low, high = "分歧中性", 55, 70
        why = "行业间分歧中等，维持中性仓位、按个股信号加减"
    return dict(median=float(median_div), stance=stance, low=low, high=high,
                why=(f"{why}；全市场分歧度中位 {median_div:.2f}"
                     f"{f'（{n_funds} 只基金）' if n_funds else ''}"))


def decision_table(t: pd.DataFrame, active: pd.DataFrame,
                   coverage_ok: bool = True,
                   t0: pd.DataFrame | None = None,
                   disp: pd.DataFrame | None = None) -> pd.DataFrame:
    """31 行行业配置表：动作 + 理由 + 失效条件。

    规则（事前固定，不按结果调参）：
      · 超配：超配分位 ≥ 80 且 主动项 > 0
      · 回避：超配分位 ≤ 20 且 主动项 < 0
      · 低配：超配分位 ≤ 20 或 主动项 < 0（二者其一）
      · 标配：其余
    证据等级：F3e 剔 Q4 后不显著 → 全部标「参考」；主动项 n=9 → 同样「参考」。
    """
    if disp is not None and not disp.empty and "行业" in disp.columns:
        disp = disp.set_index("行业")
    rows = []
    for k in t.index:
        z = t.loc[k, "z"]
        pct = t.loc[k, "pct"]
        dz = t.loc[k, "dz"]
        d_alloc = t.loc[k, "d_alloc"]
        a = active.loc[k, "active"] if k in active.index else np.nan
        if pd.isna(pct):
            act = "标配"
            why = "超配分位缺失，不给方向"
        elif pct >= 80 and pd.notna(a) and a > 0:
            act = "超配"
            why = f"分位 {pct:.0f}%（高位）且主动加仓 {a:+.2f}pp"
        elif pct <= 20 and pd.notna(a) and a < 0:
            act = "回避"
            why = f"分位 {pct:.0f}%（低位）且主动减仓 {a:+.2f}pp"
        elif pct <= 20 or (pd.notna(a) and a < 0):
            act = "低配"
            why = (f"分位 {pct:.0f}%" if pct <= 20 else "主动项为负")
        else:
            act = "标配"
            why = f"分位 {pct:.0f}%、主动 {a:+.2f}pp，无极端信号" if pd.notna(a) else f"分位 {pct:.0f}%"
        rows.append(dict(
            行业=k, 建议动作=act,
            主动加减仓=round(float(a), 2) if pd.notna(a) else np.nan,
            配置变化pp=round(float(d_alloc), 2) if pd.notna(d_alloc) else np.nan,
            超配分位=round(float(pct), 0) if pd.notna(pct) else np.nan,
            当前配置=round(float(t.loc[k, "alloc"]), 2) if pd.notna(t.loc[k, "alloc"]) else np.nan,
            # 上期对照（本轮新增）：没有对照的结论不允许进决策表
            上期配置=(round(float(t.loc[k, "alloc0"]), 2)
                      if pd.notna(t.loc[k, "alloc0"]) else np.nan),
            上期主动项=(round(float(t0.loc[k, "active"]), 2)
                        if t0 is not None and k in t0.index
                        and pd.notna(t0.loc[k].get("active", np.nan)) else np.nan),
            上期等级=(t.loc[k, "lvl0"] if pd.notna(t.loc[k, "lvl0"]) else "—"),
            加仓基金占比=(float(disp.loc[k, "加仓占比"])
                          if disp is not None and k in getattr(disp, "index", []) else np.nan),
            分歧度=(float(disp.loc[k, "分歧度"])
                    if disp is not None and k in getattr(disp, "index", []) else np.nan),
            基金行为=(disp.loc[k, "判读"]
                      if disp is not None and k in getattr(disp, "index", []) else "未采集"),
            口径一致性=_consistency(
                a, (float(disp.loc[k, "加仓占比"])
                    if disp is not None and k in getattr(disp, "index", []) else np.nan)),
            理由=why + _conflict_note(
                a, (float(disp.loc[k, "加仓占比"])
                    if disp is not None and k in getattr(disp, "index", []) else np.nan),
                (disp.attrs.get("q0", "上期") if disp is not None else "上期"),
                (disp.attrs.get("q", "本期") if disp is not None else "本期")),
            失效条件={
                "超配": "若下季主动项转负，或超配分位跌破 80%，本判断作废",
                "回避": "若下季主动项转正，或超配分位升破 20%，本判断作废",
                "低配": "若下季主动项转正且分位回升至 50% 以上，本判断作废",
                "标配": "若下季主动项与分位同向突破（主动为正且分位≥80，或主动为负且分位≤20），改判",
            }[act],
            证据等级="参考（F3e 剔Q4 后 p=0.11；主动项 n=9）" if coverage_ok else "未就绪",
        ))
    return pd.DataFrame(rows).sort_values(
        ["建议动作", "主动加减仓"], key=lambda s: s.map({a: i for i, a in enumerate(ACTION_ORDER)})
        if s.name == "建议动作" else s, ascending=[True, False])


# ---------------------------------------------------------------------------
# 仓位建议
# ---------------------------------------------------------------------------
def position_advice(disp: dict) -> dict:
    """基于分歧代理的仓位区间。代理非基金级，故区间偏保守并注明依据。"""
    sd = disp.get("cross_std", np.nan)
    med = disp.get("median", np.nan)
    if pd.isna(sd):
        return dict(low=np.nan, high=np.nan, stance="数据不足", why="横截面离散度缺失")
    # 口径：2024–2026 期间 Δ配置 横截面标准差的历史分位
    if sd < 1.2:
        lo, hi, st = 70, 85, "共识偏高"
    elif sd < 2.0:
        lo, hi, st = 55, 70, "分歧中性"
    else:
        lo, hi, st = 40, 55, "分歧偏高"
    return dict(low=lo, high=hi, stance=st,
                why=(f"行业 Δ配置 横截面标准差 {sd:.2f}pp、中位 {med:+.2f}pp；"
                     f"该值是代理指标，不是基金级加仓占比"))


# ---------------------------------------------------------------------------
# 反证清单
# ---------------------------------------------------------------------------
def counter_evidence(t: pd.DataFrame, checks: list[dict]) -> list[dict]:
    """本期结论最可能错在哪、什么数据能推翻。"""
    out = []
    for c in checks:
        if c["状态"] != "可用":
            out.append(dict(点=f"{c['指标']} 未通过全部六项（{c['通过项']}/{c['总项']}）",
                            推翻=f"若补齐基金级明细并做规模中性化后 IC 仍不显著，则该指标应彻底移出决策"))
    out.append(dict(点="前十大口径只覆盖净值约 46–48.5%，一半仓位不可见",
                    推翻="若全持仓校准显示某行业前十大/全持仓比值长期 <0.7，该行业的超配比例需下调"))
    out.append(dict(点="主动加减仓样本仅 9 期（2024Q2–2026Q2）",
                    推翻="若 2027 年前累计到 12 期后 t 值跌破 1.65，则主动项也应降级"))
    out.append(dict(点="价格漂移分解依赖行业指数收益，个股组合偏离指数",
                    推翻="若用基金实际净值反推的行业暴露与季报口径偏差 >3pp，分解需重做"))
    out.append(dict(点="个股清单只覆盖每期前 50 只，跌出榜单被误判为退出",
                    推翻="若取到全量持仓，退出名单会显著缩短，撤离榜的排序会变"))
    return out
