"""唯一口径层（canonical layer）：所有页面、图表、结论都必须从这里取数。

为什么要有这个模块
------------------
之前的页面里同时存在多种「行业配置比例」：有的用占净值比、有的按覆盖率归一化、
有的把半年报全持仓和季报前十大放在一起比较。三者混用会直接造出假信号：

    腾讯控股曾被算成「汽车行业、撤离榜 −855 家」
    —— 根因是港股 5 位代码被补零撞上 A 股代码，但同类问题还有：

    ① 季报只披露前十大、半年报披露全部。把 Q2（全持仓）和 Q3（前十大）直接相减，
       得到的是「披露范围变小」，不是「机构减仓」；
    ② 按覆盖率归一化能对齐「占已披露组合的比例」，但对不齐「选股范围」：
       前十大天然集中在重仓行业，归一化后仍会系统性放大集中行业；
    ③ 行业占比之和 = 股票仓位，季报期这个和只有 20–40%，直接当「配置比例」用，
       会把「仓位低」误读成「不看好所有行业」。

因此本模块只提供三套口径，每套只回答一个问题，调用方必须显式选择：

    A. `panel_full(q)`   全持仓口径（仅半年报 Q2/Q4 有）
       → 回答「行业配置占净值多少、超配低配多少、主动还是被动」
       → 权重是占净值比，之和 = 股票仓位，可分解仓位与行业两个维度

    B. `panel_top10(q)`  前十大口径（每季都有）
       → 回答「多少基金在加、分歧有多大」
       → 每只基金取权重最大的 10 只，再归一化到 100%，跨季同构可比

    C. `stock_flow(q0,q)` 个股口径（只在同披露类型的两期之间）
       → 回答「哪只股票被更多基金持有」

所有函数的 index 都是 6 位基金代码，columns 都是 31 个申万一级行业。
港股不属于申万一级，一律排除在行业聚合之外（见 `industry_panel` 的 hk 处理）。
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
import pandas as pd

import config as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "db", "fund_ak.db")

# 主动权益：只要这三类。指数/ETF 联接/债券/QDII 的持仓是复制指数或做债，
# 拿它们算「机构观点」等于把被动资金当成主动判断。
ACTIVE_TYPES = ("股票型", "混合型-偏股", "混合型-灵活")

# 单行业占披露组合超过这个比例，判为主题基金：它的加减仓是申赎与主题轮动，
# 不是行业观点，会把行业分歧度整体拉高。
THEME_MAX = 80.0

# 覆盖率（占净值比合计）低于/高于这个区间，说明该基金该期的披露数据失真
# （东财个别分级、联接、QDII 会算出 181%），必须剔除而不是归一化。
COV_MIN, COV_MAX = 1.0, 105.0


# ---------------------------------------------------------------------------
# 连接与缓存
# ---------------------------------------------------------------------------
def available() -> bool:
    """数据库是否可用。

    设 `FCD_FORCE_FALLBACK=1` 可强制走压缩快照，用来在本地验证
    「线上没有数据库」这条路径——否则部署后才发现页面空白就太晚了。
    """
    if os.environ.get("FCD_FORCE_FALLBACK"):
        return False
    return os.path.exists(DB)


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)


@contextmanager
def _con():
    """每次查询新建连接。

    坑：早期把 Connection 用 lru_cache 缓存下来复用，在 Streamlit 下必炸——
    每次 rerun 跑在新线程里，sqlite3 默认禁止跨线程使用，
    而所有查询都包在 try/except 里，异常被吞掉就返回空表，
    表现是「页面什么都没有、但也不报错」。读多写少的场景，
    每次开连接的成本远小于这个坑。
    """
    con = _connect()
    try:
        yield con
    finally:
        con.close()


@lru_cache(maxsize=8)
def quarters() -> tuple[str, ...]:
    """库内全部期次（升序）。"""
    if not available():
        d = _fb_all()
        return tuple(sorted(d["quarter"].unique())) if not d.empty else ()
    try:
        with _con() as con:
            qs = [r[0] for r in con.execute(
                "SELECT DISTINCT quarter FROM fund_holdings ORDER BY quarter")]
    except Exception:
        return ()
    return tuple(qs)


def latest(n: int = 1) -> str | None:
    qs = quarters()
    return qs[-n] if len(qs) >= n else None


def is_full(q: str) -> bool:
    """是否全持仓披露期（半年报 Q2、年报 Q4）。"""
    return q[-1] in ("2", "4")


def full_quarters() -> list[str]:
    return [q for q in quarters() if is_full(q)]


def prev_full(q: str) -> str | None:
    """上一期「全持仓」季度（用于全持仓口径的相邻比较）。"""
    fs = [x for x in full_quarters() if x < q]
    return fs[-1] if fs else None


# ---------------------------------------------------------------------------
# 基金属性
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def profiles() -> pd.DataFrame:
    """基金档案：成立日、规模、类型、公司、经理、比较基准。缺表时返回空表。"""
    if not available():
        d = _fb("profile")
        if d.empty:
            return d
        d["inception"] = pd.to_datetime(d["inception"], errors="coerce")
        return d
    try:
        with _con() as con:
            d = pd.read_sql("SELECT * FROM fund_profile", con, dtype={"code": str})
    except Exception:
        return pd.DataFrame()
    if not d.empty:
        d["code"] = d["code"].astype(str).str.zfill(6)
        d["inception"] = pd.to_datetime(d["inception"], errors="coerce")
    return d


def benchmark_style(code: str) -> str:
    """从业绩比较基准里粗分风格：沪深300 / 中证500 / 中证1000 / 港股 / 均衡。

    只用于给「这只基金该跟谁比」一个可读标签，不做精确归因——
    精确归因需要基准指数日收益，本库没有。
    """
    p = profiles()
    if p.empty or code not in set(p["code"]):
        return "未知"
    b = str(p.loc[p["code"] == code, "benchmark"].iloc[0])
    if b in ("nan", "None", ""):
        return "未知"
    tags = []
    if "沪深300" in b:
        tags.append("沪深300")
    if "中证500" in b:
        tags.append("中证500")
    if "中证1000" in b:
        tags.append("中证1000")
    if "港股" in b or "恒生" in b:
        tags.append("港股")
    if "创业板" in b:
        tags.append("创业板")
    if "中债" in b or "国债" in b:
        tags.append("含债")
    return "+".join(tags) if tags else "其他"


# ---------------------------------------------------------------------------
# 样本池
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def equity_codes() -> frozenset[str]:
    """主动权益基金全集。

    主口径用 `fund_meta.type`（抓取时由东财基金列表写入，4941 只全覆盖），
    再用雪球档案的 `ftype` 做两件事：剔除偏债/平衡/QDII 等非纯权益，
    以及补上 fund_meta 缺失但档案能确认的基金。

    两条路径（有库 / 读快照）必须走同一段筛选逻辑。早期降级分支直接
    返回「type 命中就留」，漏掉了剔除偏债/平衡型这一步，
    导致线上样本比本地多 63 只——同一句话在两个环境给出两个数。
    """
    p = profiles()
    if available():
        with _con() as con:
            d = pd.read_sql("SELECT code, type FROM fund_meta", con,
                            dtype={"code": str})
        keep = set(d.loc[d["type"].isin(ACTIVE_TYPES), "code"].str.zfill(6))
    else:
        if p.empty or "type" not in p.columns:
            return frozenset()
        keep = set(p.loc[p["type"].isin(ACTIVE_TYPES), "code"])
    if not p.empty and "ftype" in p.columns:
        ft = p["ftype"].astype(str)
        keep -= set(p.loc[ft.str.contains("偏债|货币|理财|债券|平衡|保本|QDII",
                                          na=False), "code"])
        keep |= set(p.loc[ft.str.startswith(("股票型", "混合型-偏股", "混合型-灵活")),
                           "code"])
    return frozenset(keep)


@lru_cache(maxsize=16)
def theme_codes(q: str) -> frozenset[str]:
    """该季被判为「主题基金」的代码集合（单行业 >80%，按前十大口径算）。

    用前十大口径而不是全持仓：季报期只有前十大，判定标准必须跨季统一，
    否则 Q2 判得出、Q3 判不出，样本会随披露类型跳变。
    """
    p = panel_top10(q)
    if p.empty:
        return frozenset()
    return frozenset(p.index[p.max(axis=1) > THEME_MAX])


@lru_cache(maxsize=16)
def universe(q: str) -> frozenset[str]:
    """该季的样本基金：主动权益 + 数据可用 + 非主题基金。"""
    cov = coverage(q)
    if cov.empty:
        return frozenset()
    eq = equity_codes()
    keep = cov.index.intersection(list(eq)) if eq else cov.index
    good = cov.loc[keep]
    good = good[(good >= COV_MIN) & (good <= COV_MAX)]
    out = set(good.index) - set(theme_codes(q))
    return frozenset(out)


# ---------------------------------------------------------------------------
# 覆盖率 = 股票仓位（仅全持仓期可解释为仓位）
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def coverage(q: str) -> pd.Series:
    """每只基金该期「占净值比」合计。

    全持仓期（Q2/Q4）它就是股票仓位；
    季报期（Q1/Q3）只有前十大，这个值只能当「披露充分度」，不是仓位。
    """
    if not available():
        d = _fb("coverage")
        if d.empty:
            return pd.Series(dtype=float)
        d = d[d["quarter"] == q][["code", "cov"]]
        return d.set_index("code")["cov"].astype(float) if not d.empty \
            else pd.Series(dtype=float)
    try:
        with _con() as con:
            d = pd.read_sql(
                "SELECT code, SUM(weight_pct) AS cov FROM fund_holdings "
                "WHERE quarter=? GROUP BY code", con, params=(q,), dtype={"code": str})
    except Exception:
        return pd.Series(dtype=float)
    if d.empty:
        return pd.Series(dtype=float)
    d["code"] = d["code"].astype(str).str.zfill(6)
    return d.set_index("code")["cov"].astype(float)


# ---------------------------------------------------------------------------
# 降级数据源：线上部署没有 500MB 数据库，读压缩快照
# ---------------------------------------------------------------------------
DERIVED = os.path.join(ROOT, "data", "derived")
_FB_FILES = {
    "top10": "canon_top10.csv.gz", "full": "canon_full.csv.gz",
    "coverage": "canon_coverage.csv.gz", "stocks": "canon_stocks.csv.gz",
    "profile": "canon_profile.csv.gz",
}


@lru_cache(maxsize=8)
def _fb(name: str) -> pd.DataFrame:
    """读压缩快照（由 tools/export_canon.py 生成）。"""
    p = os.path.join(DERIVED, _FB_FILES[name])
    if not os.path.exists(p):
        return pd.DataFrame()
    d = pd.read_csv(p, compression="gzip",
                    dtype={"code": str, "stock_code": str})
    if "code" in d.columns:
        d["code"] = d["code"].astype(str).str.zfill(6)
    return d


@lru_cache(maxsize=1)
def _fb_all() -> pd.DataFrame:
    """只取 quarter 列，用于无数据库时枚举期次。"""
    p = os.path.join(DERIVED, _FB_FILES["top10"])
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p, compression="gzip", usecols=["quarter"])


def fallback_available() -> bool:
    return all(os.path.exists(os.path.join(DERIVED, f)) for f in _FB_FILES.values())


def _to_panel(d: pd.DataFrame, wcol: str, normalize: bool) -> pd.DataFrame:
    """长表 → 基金 × 行业 矩阵（可选归一化到 100%）。"""
    if d.empty:
        return pd.DataFrame()
    p = d.pivot_table(index="code", columns="industry", values=wcol,
                      aggfunc="sum")
    p = p.reindex(columns=list(C.INDUSTRIES)).fillna(0.0)
    if normalize:
        s = p.sum(axis=1)
        p = p.div(s.where(s > 0), axis=0) * 100.0
    return p


# ---------------------------------------------------------------------------
# A. 全持仓口径
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def panel_full(q: str, normalize: bool = True) -> pd.DataFrame:
    """基金 × 行业，占净值比（%）。normalize=True 时归一化到合计 100%。

    只在 `is_full(q)` 为真时有意义。归一化后的含义是
    「这只基金的股票市值里，行业 k 占多少」——回答行业配置结构；
    不归一化（normalize=False）时之和 = 股票仓位，回答「买了多少股票」。
    """
    if not available():
        d = _fb("full")
        return _to_panel(d[d["quarter"] == q] if not d.empty else d, "w", normalize)
    try:
        with _con() as con:
            d = pd.read_sql(
                "SELECT code, industry, weight_pct FROM fund_holdings "
                "WHERE quarter=? AND industry IS NOT NULL AND industry!=''",
                con, params=(q,), dtype={"code": str})
    except Exception:
        return pd.DataFrame()
    return _to_panel(d, "weight_pct", normalize)


# ---------------------------------------------------------------------------
# B. 前十大口径
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def panel_top10(q: str, normalize: bool = True) -> pd.DataFrame:
    """基金 × 行业，只取每只基金权重最大的 10 只股票，再按行业加总。

    这是唯一能跨季比较「多少基金在加」的口径：
    Q1/Q3 交易所只要求披露前十大，Q2/Q4 虽然披露全部，
    但同样截断到前十大之后，四个季度的「选股范围」就一致了。
    """
    if not available():
        d = _fb("top10")
        return _to_panel(d[d["quarter"] == q] if not d.empty else d, "w", normalize)
    # 关键：先按「全部持仓（含港股）」取前 10 只，再只把 A 股行业汇总。
    # 若一开始就过滤掉港股，含 3 只港股重仓的基金只会剩下 7 只 A 股参与归一化，
    # 行业占比被整体放大，和「取满 10 只」的基金不可比。
    sql = """
        SELECT code, industry, SUM(weight_pct) AS w
        FROM (
            SELECT code, industry, weight_pct,
                   ROW_NUMBER() OVER (PARTITION BY code ORDER BY weight_pct DESC) rn
            FROM fund_holdings
            WHERE quarter=?
        )
        WHERE rn <= 10 AND industry IS NOT NULL AND industry!=''
        GROUP BY code, industry
    """
    try:
        with _con() as con:
            d = pd.read_sql(sql, con, params=(q,), dtype={"code": str})
    except Exception:
        return pd.DataFrame()
    return _to_panel(d, "w", normalize)



# ---------------------------------------------------------------------------
# 相邻期对
# ---------------------------------------------------------------------------
def pair_quarter(q: str | None = None) -> tuple[str, str] | None:
    """相邻两季（前十大口径可比）：返回 (上期, 本期)。"""
    qs = quarters()
    q = q or (qs[-1] if qs else None)
    if not q or q not in qs:
        return None
    i = qs.index(q)
    return (qs[i - 1], q) if i > 0 else None


def pair_full(q: str | None = None) -> tuple[str, str] | None:
    """相邻两个全持仓期（半年报口径可比）：返回 (上期, 本期)。"""
    fs = full_quarters()
    if not fs:
        return None
    q = q or fs[-1]
    earlier = [x for x in fs if x < q]
    return (earlier[-1], q) if earlier else None


def pair_same(q: str | None = None) -> tuple[str, str] | None:
    """同一披露类型、相隔一年的两期：返回 (上年同期, 本期)。

    个股层面的唯一合法口径：季报只能和上年同季比，
    否则「从第 11 名掉出前十大」会被当成「清仓」。
    """
    qs = quarters()
    q = q or (qs[-1] if qs else None)
    if not q:
        return None
    cand = f"{int(q[:4]) - 1}{q[4:]}"
    return (cand, q) if cand in qs else None


# ---------------------------------------------------------------------------
# 广度与分歧：前十大口径
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def breadth(q0: str, q: str, min_move: float = 0.10) -> pd.DataFrame:
    """行业层面的「加仓广度」与「分歧度」，前十大口径。

    定义（事前固定，不看结果调参）：
        加仓基金数 = Δ行业占比 > +min_move pp 的基金数
        减仓基金数 = Δ行业占比 < −min_move pp 的基金数
        加仓占比   = 加仓 / (加仓 + 减仓)
        分歧度     = 1 − |2×加仓占比 − 1|，取值 [0,1]
                     0＝所有基金同向；1＝多空各半

    注意：分歧度与加仓占比不是两个独立指标（分歧度由占比完全决定），
    所以任何图都不能把两者当两个轴画散点——那是一根线。
    """
    p0, p1 = panel_top10(q0), panel_top10(q)
    if p0.empty or p1.empty:
        return pd.DataFrame()
    common = p0.index.intersection(p1.index)
    u = universe(q) & universe(q0)
    common = common.intersection(list(u)) if u else common
    if len(common) < 30:
        return pd.DataFrame()
    d = p1.loc[common] - p0.loc[common]
    rows = []
    for k in C.INDUSTRIES:
        if k not in d.columns:
            continue
        v = d[k].dropna()
        if len(v) < 30:
            continue
        up = int((v > min_move).sum())
        dn = int((v < -min_move).sum())
        tot = up + dn
        share = up / tot * 100 if tot else 50.0
        rows.append(dict(
            行业=k, 样本基金数=int(len(v)), 加仓基金数=up, 减仓基金数=dn,
            不变基金数=int(len(v) - tot),
            平均变化pp=round(float(v.mean()), 3),
            中位变化pp=round(float(v.median()), 3),
            加仓占比=round(share, 1),
            # 离散度 = Δ 的标准差，衡量「基金之间差多少」，与广度相互独立。
            # 老版本用「分歧度 = 1−|2×占比−1|」，它完全由加仓占比决定，
            # 与广度是一根线上的两个刻度，不能当第二个指标用。
            离散度pp=round(float(v.std(ddof=0)), 3),
            分歧度=round(1 - abs(2 * share / 100 - 1), 3),
        ))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["判读"] = t.apply(_label, axis=1)
    t.attrs.update(q0=q0, q=q, n_funds=len(common), kind="top10")
    return t.sort_values("加仓占比", ascending=False).reset_index(drop=True)


def _label(r) -> str:
    """只看加仓占比，不用分歧度。

    旧规则写的是「加仓占比 ≥70% 且 分歧度 <0.4」，但分歧度 = |2×占比−1|，
    占比 ≥70% 时分歧度必然 ≥0.4，两个条件互斥——这条规则永远不成立，
    所以页面上从来没出现过「共识形成」。标签只用一个维度即可：

        ≥65% 基金在加 → 共识形成
        ≤35% 基金在加 → 系统性撤离
        其余            → 多空对撞（强度看离散度）
    """
    s = r["加仓占比"]
    if s >= 65:
        return "共识形成"
    if s <= 35:
        return "系统性撤离"
    return "多空对撞"


# ---------------------------------------------------------------------------
# 主动 / 被动拆解：全持仓口径
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def decomp(q0: str, q: str) -> pd.DataFrame:
    """行业配置变化的 GK 分解（全持仓口径，权重归一化到 100%）。

        Δw_k = 被动_k + 主动_k
        被动_k = w0_k(1+r_k)/Σ w0_j(1+r_j) − w0_k     ← 什么都不做，价格自己走出来的
        主动_k = Δw_k − 被动_k                        ← 调仓真正贡献的

    这是回答「机构到底动没动」的唯一正确方式；把 Δw 直接叫「减仓」是错的。
    """
    import attribution as atr

    p0, p1 = panel_full(q0), panel_full(q)
    if p0.empty or p1.empty:
        return pd.DataFrame()
    import data_loader as dl
    try:
        ret = dl.load_all().market["ret"]
    except Exception:
        return pd.DataFrame()
    if q not in ret.index:
        return pd.DataFrame()
    common = p0.index.intersection(p1.index)
    u = universe(q) & universe(q0)
    common = common.intersection(list(u)) if u else common
    if len(common) < 30:
        return pd.DataFrame()
    # 用样本均值当「代表性组合」，逐基金分解太噪，行业层面的结论靠均值更稳
    w0 = p0.loc[common].mean()
    w1 = p1.loc[common].mean()
    r = ret.loc[q].reindex(w0.index).astype(float) / 100.0
    ok = w0.notna() & w1.notna() & r.notna()
    g = w0[ok] * (1 + r[ok])
    drift = g / g.sum() * 100.0
    out = pd.DataFrame({
        "上期配置": w0[ok], "本期配置": w1[ok],
        "被动贡献": (drift - w0[ok]), "主动贡献": (w1[ok] - drift),
    })
    out["变化"] = out["本期配置"] - out["上期配置"]
    out["被动占比"] = (out["被动贡献"].abs() /
                     (out["被动贡献"].abs() + out["主动贡献"].abs()).replace(0, np.nan))
    out.index.name = "行业"
    out.attrs.update(q0=q0, q=q, n_funds=len(common), kind="full")
    return out.sort_values("变化").round(3)


def position_change(q: str | None = None) -> dict:
    """股票仓位变化（全持仓口径）：整体加仓，还是只在行业间挪。

    这是季报里最容易被忽略、但最能说明「机构态度」的一行：
    行业配置比例上升可能只是「股票仓位提高」，不代表看好某个行业。
    """
    pf = pair_full(q)
    if not pf:
        return {}
    q0, q1 = pf
    c0, c1 = coverage(q0), coverage(q1)
    u = universe(q0) & universe(q1)
    idx = c0.index.intersection(c1.index).intersection(list(u))
    if len(idx) < 30:
        return {}
    d = (c1.loc[idx] - c0.loc[idx]).dropna()
    return dict(q0=q0, q=q1, n=len(d),
                仓位0=float(c0.loc[idx].median()),
                仓位1=float(c1.loc[idx].median()),
                中位变化pp=float(d.median()),
                加仓基金占比=float((d > 0.5).mean() * 100))


def industry_returns() -> pd.DataFrame:
    """31 个申万一级行业的季度收益率（%），来自 Wind 缓存。"""
    import data_loader as dl
    return dl.load_all().market["ret"]


def industry_flow(q0: str, q: str) -> pd.DataFrame:
    """行业层面的资金/持仓迁徙（全持仓口径）：Δ市值 + Δ持股基金数代理。

    Δ市值 = 本期持仓市值 − 上期持仓市值，按行业加总（亿元）。
    「持股基金数」在全持仓口径下用「持有该行业的基金数」，可反映广度。
    """
    if not available():
        return pd.DataFrame()
    def _one(qq):
        with _con() as con:
            d = pd.read_sql(
                "SELECT industry, SUM(mv) mv, COUNT(DISTINCT code) n FROM fund_holdings "
                "WHERE quarter=? AND industry IS NOT NULL AND industry!='' "
                "GROUP BY industry", con, params=(qq,))
        return d.set_index("industry")
    a, b = _one(q0), _one(q)
    j = b.join(a, rsuffix="_0", how="outer").reindex(list(C.INDUSTRIES))
    out = pd.DataFrame({
        "本期市值亿": j["mv"] / 1e4, "上期市值亿": j["mv_0"] / 1e4,
        "Δ市值亿": (j["mv"] - j["mv_0"]) / 1e4,
        "本期基金数": j["n"], "上期基金数": j["n_0"],
        "Δ基金数": j["n"] - j["n_0"],
    }).round(1)
    out.index.name = "行业"
    out.attrs.update(q0=q0, q=q)
    return out.sort_values("Δ市值亿", ascending=False)


# ---------------------------------------------------------------------------
# C. 个股口径
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def stock_snapshot(q: str) -> pd.DataFrame:
    """单季个股快照：持股基金数、持股数、持仓市值、行业、是否港股。"""
    if not available():
        d = _fb("stocks")
        if d.empty:
            return pd.DataFrame()
        d = d[d["quarter"] == q].copy()
        d["stock_code"] = d["stock_code"].astype(str).str.strip()
        d["industry"] = np.where(pd.to_numeric(d["is_hk"], errors="coerce").fillna(0) == 1,
                                 "港股", d["industry"].fillna("未映射"))
        return d.set_index("stock_code")
    sql = """
        SELECT stock_code, MAX(stock_name) stock_name, MAX(industry) industry,
               MAX(is_hk) is_hk, COUNT(DISTINCT code) n_funds,
               SUM(shares) shares, SUM(mv) mv
        FROM fund_holdings WHERE quarter=?
        GROUP BY stock_code
    """
    with _con() as con:
        d = pd.read_sql(sql, con, params=(q,), dtype={"stock_code": str})
    if d.empty:
        return d
    d["stock_code"] = d["stock_code"].str.strip()
    d["industry"] = np.where(d["is_hk"].fillna(0) == 1, "港股",
                             d["industry"].fillna("未映射"))
    return d.set_index("stock_code")


@lru_cache(maxsize=4)
def stock_flow(q0: str, q: str) -> pd.DataFrame:
    """个股 Δ持股基金数（只在同披露类型的两期之间调用）。"""
    a, b = stock_snapshot(q0), stock_snapshot(q)
    if a.empty or b.empty:
        return pd.DataFrame()
    j = b.join(a, rsuffix="_0", how="outer")
    for c in ("n_funds", "shares", "mv"):
        j[c] = j[c].fillna(0)
        j[c + "_0"] = j[c + "_0"].fillna(0)
    j["d_n"] = j["n_funds"] - j["n_funds_0"]
    j["d_mv_yi"] = (j["mv"] - j["mv_0"]) / 1e4
    j["mv_yi"] = j["mv"] / 1e4
    n0, n1 = len(universe(q0)) or 1, len(universe(q)) or 1
    j["share"] = j["n_funds"] / n1 * 100
    j["share_0"] = j["n_funds_0"] / n0 * 100
    j["d_share_pp"] = j["share"] - j["share_0"]
    j.attrs.update(q0=q0, q=q, n_funds=(n0, n1))
    return j


# ---------------------------------------------------------------------------
# 自检：口径差异有多大（改口径前必须看这个数）
# ---------------------------------------------------------------------------
def compare_definitions(q: str | None = None) -> pd.DataFrame:
    """同一行业在三种口径下的「加仓占比」，用来证明口径不是无关紧要的细节。"""
    pq = pair_quarter(q)
    ps = pair_same(q)
    pf = pair_full(q)
    rows = []
    if pq:
        b1 = breadth(*pq)
        for _, r in b1.iterrows():
            rows.append(dict(口径="相邻季·前十大归一化（季频）", 期次=f"{pq[0]}→{pq[1]}",
                             行业=r["行业"], 加仓占比=r["加仓占比"], 分歧度=r["分歧度"],
                             样本=r["样本基金数"]))
    if ps:
        b2 = breadth(*ps)
        for _, r in b2.iterrows():
            rows.append(dict(口径="上年同季·前十大归一化（年频）", 期次=f"{ps[0]}→{ps[1]}",
                             行业=r["行业"], 加仓占比=r["加仓占比"], 分歧度=r["分歧度"],
                             样本=r["样本基金数"]))
    if pf:
        b3 = breadth(*pf)
        for _, r in b3.iterrows():
            rows.append(dict(口径="相邻半年报·前十大归一化", 期次=f"{pf[0]}→{pf[1]}",
                             行业=r["行业"], 加仓占比=r["加仓占比"], 分歧度=r["分歧度"],
                             样本=r["样本基金数"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("期次:", len(quarters()), quarters()[:2], "→", quarters()[-1:])
    print("全持仓期:", len(full_quarters()))
    p = profiles()
    print("档案:", len(p), "有成立日", int(p["inception"].notna().sum()) if not p.empty else 0)
    q = latest()
    print(f"\n{latest()} 样本基金:", len(universe(q)))
    print("主题基金剔除:", len(theme_codes(q)))
    c = coverage(q)
    print("覆盖率分位:", c.quantile([0.05, 0.5, 0.95]).round(1).to_dict())
    pf = pair_full()
    if pf:
        print("\n全持仓口径:", pf, "样本", len(universe(pf[1]) & universe(pf[0])))
        print(position_change())
    pq = pair_quarter()
    if pq:
        b = breadth(*pq)
        print(f"\n前十大口径 {pq[0]}→{pq[1]}：样本 {b.attrs['n_funds']}")
        print(b.head(5)[["行业", "样本基金数", "加仓占比", "分歧度", "判读"]].to_string(index=False))
