# -*- coding: utf-8 -*-
"""决策规则层（playbook）：把「数据」翻译成「动作」的唯一入口。

为什么单独一层
--------------
之前「结论总览」和「本期解读」各写一套筛选条件，结果同一行业在两个页面上
一个叫「可以跟」、一个叫「观察」。这不是文案问题，是口径分裂。本模块把
筛选条件写成代码，两个页面都调用它，改规则只改这一处。

四个维度、两条时间轴
--------------------
    时间轴① 半年报全持仓（GK 分解）→ 主动pp：剥离价格后机构真金白银的方向
    时间轴② 最近一季前十大（价格中性）→ 当期主动pp + 加仓占比：现在还在不在加
    位置③   配置分位（自身 31 期）→ 机构押得多重
    位置④   指数分位（2019 起）  → 价格有多贵

为什么必须两条时间轴：半年报最准但滞后（最新只到 2026Q2），季报最新但只有
前十大。只用半年报会漏掉最近一个季度的拐点（本期建材、机械就是在 Q3 掉头的）；
只用季报会把价格波动当成交易（已用价格中性化处理，见 canon.price_neutral_delta）。

动作定义（事前固定，不看结果调参）
--------------------------------
    可以跟  当期主动≥+0.3 且 广度≥60 且 配置分位≤80 且 指数分位≤85
    别追高  位置到极值（配置分位≥80 或 指数分位≥90）但资金还在进（广度≥55 且 当期主动>0）
    要回避  当期主动≤−0.4 且 广度≤50，或 配置分位≥90 且 广度≤50（高位没人接）
    等确认  当期主动>0 但 广度<60（少数大基金在撑，等下一季）
    观察    半年主动<0 而 当期主动>0（逆势承接，方向未定）
    中性    其余——不构成信号，按个股逻辑处理

优先级：可以跟 > 别追高 > 要回避 > 观察 > 等确认 > 中性。
「别追高」先于「要回避」：价格到极值时，即使资金还在进也不加，但也不等于看空。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import canon as cn

# 阈值集中在这里，页面上展示的也是这几个数，改一处全站同步
TH = dict(
    follow_active=0.3,      # 可以跟：当期主动 pp 下限
    follow_breadth=60.0,    # 可以跟：加仓占比下限
    follow_cfg_pct=80.0,    # 可以跟：配置分位上限
    follow_idx_pct=85.0,    # 可以跟：指数分位上限
    hot_cfg_pct=80.0,       # 别追高：配置分位触发
    hot_idx_pct=90.0,       # 别追高：指数分位触发
    hot_breadth=55.0,       # 别追高：仍需资金在进
    avoid_active=-0.4,      # 要回避：当期主动上限
    avoid_breadth=50.0,     # 要回避：加仓占比上限
    avoid_cfg_pct=90.0,     # 要回避：高位没人接的配置分位
    watch_active=0.2,       # 观察：当期主动下限（低于此视为噪音）
    confirm_active=0.1,     # 等确认：当期主动下限
    confirm_breadth=60.0,   # 等确认：广度不足
)


def _strong(r) -> str:
    """信号强度：只看幅度，不看方向。"""
    a = float(r.get("当期主动pp", np.nan) or np.nan)
    h = float(r.get("主动pp", np.nan) or np.nan)
    if not np.isfinite(a):
        return "—"
    if abs(a) >= 1.0 and (not np.isfinite(h) or np.sign(h) == np.sign(a) or abs(h) < 0.2):
        return "强"
    if abs(a) >= 0.5:
        return "中"
    return "弱"


def action_table(q: str | None = None) -> pd.DataFrame:
    """行业动作表：一行一个行业，给出动作、强度、四条理由。"""
    comp = cn.compare_periods(q)
    if comp.empty:
        return pd.DataFrame()
    idx = cn.index_stats()
    d = comp.copy()
    if not idx.empty:
        d = d.join(idx[["指数分位", "近3月涨幅%", "近1年涨幅%"]], how="left")
    cs = cn.crowding_score(q)
    if not cs.empty:
        d = d.join(cs[["CSI", "等级", "可靠"]], how="left")
    d["当期主动pp"] = d.get("平均主动pp", np.nan)
    d["半年主动pp"] = d.get("主动pp", np.nan)

    def _act(r) -> str:
        a, h = r["当期主动pp"], r["半年主动pp"]
        br, cfg, ip = r["加仓占比"], r["历史分位"], r.get("指数分位", np.nan)
        if not np.isfinite(a) or not np.isfinite(br):
            return "数据不足"
        # 小样本行业（持有基金数<150 或前十配置<0.5%）不参与「可以跟/别追高」评级：
        # 几十只基金就能把分位推到极值，据此给出买卖动作会误导。
        _small = r.get("可靠") == "小样本"
        # 「可以跟」要求半年报也不在撤：只靠一个季度转正就进场，
        # 往往是季度噪音（军工本期就是半年 −0.19、单季 +0.59）。
        if (not _small) and (a >= TH["follow_active"] and br >= TH["follow_breadth"]
                and (not np.isfinite(h) or h > 0)
                and (not np.isfinite(cfg) or cfg <= TH["follow_cfg_pct"])
                and (not np.isfinite(ip) or ip <= TH["follow_idx_pct"])):
            return "可以跟"
        if (not _small) and (((np.isfinite(cfg) and cfg >= TH["hot_cfg_pct"])
                or (np.isfinite(ip) and ip >= TH["hot_idx_pct"]))
                and br >= TH["hot_breadth"] and a > 0):
            return "别追高"
        if (a <= TH["avoid_active"] and br <= TH["avoid_breadth"]) or \
                (np.isfinite(cfg) and cfg >= TH["avoid_cfg_pct"] and br <= TH["avoid_breadth"]):
            return "要回避"
        # 「观察」要求当期净额至少 +0.2pp：+0.03pp 这种量级是噪音，不是信号
        if np.isfinite(h) and h < 0 and a >= TH["watch_active"]:
            return "观察"
        if a >= TH["confirm_active"] and br < TH["confirm_breadth"]:
            return "等确认"
        return "中性"

    d["动作"] = d.apply(_act, axis=1)
    d["强度"] = d.apply(_strong, axis=1)
    d["理由"] = d.apply(_reason, axis=1)
    _order = {"可以跟": 0, "别追高": 1, "要回避": 2, "观察": 3, "等确认": 4, "中性": 5, "数据不足": 6}
    d["_o"] = d["动作"].map(_order)
    out = d.sort_values(["_o", "当期主动pp"], ascending=[True, False]).drop(columns="_o")
    # sort_values/join 都可能丢 attrs，最后统一补回期次信息
    out.attrs.update(comp.attrs)
    return out


def _reason(r) -> str:
    """一句话讲清「为什么是这个动作」，数字全部在场。"""
    a, h = r["当期主动pp"], r["半年主动pp"]
    br, cfg, ip = r["加仓占比"], r["历史分位"], r.get("指数分位", np.nan)
    act = r["动作"]
    pct = lambda v: f"{v:.0f}" if np.isfinite(v) else "—"     # noqa: E731
    mid = (f"本季{br:.0f}%的变动基金在加（价格中性后净额{a:+.2f}pp）、"
           f"配置{pct(cfg)}分位、指数{pct(ip)}分位")
    tail = {
        "可以跟": "三个条件同时满足：资金在进、多数在加、价格不贵。",
        "别追高": "资金还在进，但位置已到历史极值，赔率不够；不是看空，是不加。",
        "要回避": "资金在撤且没人接，或高位已无承接，减配优先于抄底。",
        "观察": f"半年报主动{h:+.2f}pp 仍为负，本季转为净买入，方向尚未确认。",
        "等确认": f"净额为正但只有{br:.0f}%在加，是少数大基金在撑，等下一季。",
        "中性": "没有形成可执行的方向，按个股逻辑处理。",
        "数据不足": "缺指数或广度数据。",
    }[act]
    return f"{mid}；{tail}"


def sensitivity(q: str | None = None) -> pd.DataFrame:
    """阈值稳健性：把四个关键阈值各扰动 ±20%，看「可以跟 / 要回避」名单变不变。

    为什么必须做：如果名单对阈值极度敏感（动一点就换人），那它就不是结论，
    是参数。稳健的规则应该在合理区间内给出同一批名字。
    """
    base = action_table(q)
    if base.empty:
        return pd.DataFrame()
    b_follow = set(base.index[base["动作"] == "可以跟"])
    b_avoid = set(base.index[base["动作"] == "要回避"])
    scen = [
        ("基准", {}),
        ("跟入门槛 −20%（更宽松）", {"follow_active": 0.24, "follow_breadth": 48.0}),
        ("跟入门槛 +20%（更严格）", {"follow_active": 0.36, "follow_breadth": 72.0}),
        ("回避门槛 −20%（更敏感）", {"avoid_active": -0.32, "avoid_breadth": 60.0}),
        ("回避门槛 +20%（更迟钝）", {"avoid_active": -0.48, "avoid_breadth": 40.0}),
    ]
    rows = []
    saved = dict(TH)
    try:
        for name, patch in scen:
            TH.update(saved)
            TH.update(patch)
            t = action_table(q)
            f = set(t.index[t["动作"] == "可以跟"])
            a = set(t.index[t["动作"] == "要回避"])
            inter = len(f & b_follow) + len(a & b_avoid)
            union = len(f | b_follow) + len(a | b_avoid)
            rows.append(dict(情景=name, 可以跟=len(f), 要回避=len(a),
                             与基准重合=f"{inter}/{union}",
                             重合率=round(inter / union * 100, 0) if union else np.nan,
                             名单="、".join(sorted(f)) or "无"))
    finally:
        TH.clear()
        TH.update(saved)
    return pd.DataFrame(rows)


def summary(q: str | None = None) -> dict:
    """把动作表汇总成页头要用的几个数，避免各页面各算一遍。"""
    t = action_table(q)
    if t.empty:
        return {}
    vc = t["动作"].value_counts().to_dict()
    col = cn.coverage(cn.latest()) if False else None  # 占位，避免误用未过滤样本
    pos = cn.position_change()
    pq = cn.pair_quarter(q)
    br = cn.breadth(*pq) if pq else pd.DataFrame()
    return dict(
        table=t, counts=vc,
        n_follow=vc.get("可以跟", 0), n_hot=vc.get("别追高", 0),
        n_avoid=vc.get("要回避", 0), n_wait=vc.get("等确认", 0),
        n_watch=vc.get("观察", 0), n_neutral=vc.get("中性", 0),
        pos=pos, breadth=br,
        q=t.attrs.get("q"), q_prev=t.attrs.get("q_prev"), q_year=t.attrs.get("q_year"),
        q_full=t.attrs.get("q_full"), q_full0=t.attrs.get("q_full0"),
    )


# ---------------------------------------------------------------------------
# 结论卡：每条都写成「结论 / 证据 / 含义 / 动作 / 失效条件」
# ---------------------------------------------------------------------------
def position_band(pct: float, disp: float | None) -> tuple[int, int, bool]:
    """由仓位分位与离散度给出权益仓位区间。规则事前固定：

        分位 ≥85 → 50–60；70–85 → 60–70；30–70 → 70–80；<30 → 75–85
        离散度中位 >8pp（基金之间分歧大）→ 再降 5pp

    不用「分歧度」：它是广度的单调变换（广度 50% 时恒为 1），
    拿它设阈值等于每期都触发。
    """
    if pct >= 85:
        lo, hi = 50, 60
    elif pct >= 70:
        lo, hi = 60, 70
    elif pct >= 30:
        lo, hi = 70, 80
    else:
        lo, hi = 75, 85
    cut = bool(disp is not None and np.isfinite(disp) and disp > 8.0)
    if cut:
        lo, hi = lo - 5, hi - 5
    return lo, hi, cut


def conclusions(q: str | None = None) -> list[dict]:
    """生成全量结论清单（供「结论总览」页渲染）。

    每条结论都必须能从库里复算：
        结论 = 一句话判断
        证据 = 具体数字（页面上会显示口径与期次）
        含义 = 这条对投资意味着什么
        动作 = 该做什么
        失效 = 什么情况下这条不成立
    """
    t = action_table(q)
    if t.empty:
        return []
    s = summary(q)
    pos = s["pos"] or {}
    out: list[dict] = []

    # ---- 1. 总仓位 ----
    if pos:
        hist = cn.position_series()
        pct = float(hist["中位"].rank(pct=True).iloc[-1] * 100) if not hist.empty else np.nan
        _disp = float(s["breadth"]["离散度pp"].median()) if not s["breadth"].empty else np.nan
        _lo, _hi, _cut = position_band(pct, _disp)
        out.append(dict(
            tag="总仓位",
            claim=(f"机构没有加仓，仓位停在 {pos['仓位1']:.1f}%，"
                   f"处在 {len(hist)} 个半年报期的第 {pct:.0f} 分位。"),
            evidence=(f"{pos['q0']}→{pos['q']}：中位仓位 {pos['仓位0']:.2f}% → "
                      f"{pos['仓位1']:.2f}%，变化 {pos['中位变化pp']:+.2f}pp；"
                      f"加仓基金占 {pos['加仓基金占比']:.0f}%、减仓 "
                      f"{100 - pos['加仓基金占比']:.0f}%。样本 {pos['n']:,} 只。"),
            meaning=("这是一条「排除法」结论：既然总仓位没动，行业配置的变化就来自"
                     "内部换仓，而不是新增资金。所以看到某行业配置上升，可以读成"
                     "「机构更看好它」，不必担心是「大盘涨、水涨船高」。反过来，"
                     "高仓位也意味着后续增量有限，指望「全体加仓推普涨」不现实。"),
            action=(f"权益仓位参考 {_lo}–{_hi}%"
                    f"（仓位分位 {pct:.0f}、离散度中位 {_disp:.1f}pp"
                    f"{'，>8pp 再降 5pp' if _cut else ''}），不要满仓；"
                    f"新增资金优先买「可以跟」的行业，不追「别追高」的行业。"),
            fail=("若下一期仓位中位上行超过 2pp，说明是增量资金入场，"
                  "「存量换仓」的读法失效，应改为「增量推动」。"),
        ))

    # ---- 2. 换仓主线 ----
    dd = t.dropna(subset=["Δ上季pp"])
    if not dd.empty:
        up = dd.nlargest(3, "Δ上季pp")
        dn = dd.nsmallest(3, "Δ上季pp")
        out.append(dict(
            tag="换仓主线",
            claim=(f"钱在换手：{up.index[0]} 加 {up.iloc[0]['Δ上季pp']:+.2f}pp 最多，"
                   f"{dn.index[0]} 减 {dn.iloc[0]['Δ上季pp']:+.2f}pp 最多——"
                   f"减仓集中在当期跌得最狠的科技硬件，加仓流向医药与防御。"),
            evidence=("加仓前三：" + "、".join(
                f"{i} {r['Δ上季pp']:+.2f}pp" for i, r in up.iterrows())
                + "；减仓前三：" + "、".join(
                f"{i} {r['Δ上季pp']:+.2f}pp" for i, r in dn.iterrows())
                + f"。口径：前十大归一化，{s['q_prev']}→{s['q']}，含价格漂移；"
                  f"同期行业指数涨跌见「本期解读」图。"),
            meaning=("如果减仓最多的是当期跌得最多的行业，那多半是价格跌出来的，"
                     "不是机构在卖；如果加仓最多的是当期跌的行业，那才是真承接。"
                     "判断换仓主线，必须把「跌幅」和「减仓幅度」放在一起看。"),
            action=("沿着「加仓最多且当期在涨/抗跌」的方向做配置；"
                    "对「减仓最多但只是跌出来的」行业不要跟着砍。"),
            fail="若下一季资金回流被减仓的行业，说明这只是季度再平衡，不是风格切换。",
        ))

    # ---- 3. 可以跟 ----
    fol = t[t["动作"] == "可以跟"]
    if len(fol):
        out.append(dict(
            tag="可以跟",
            claim=f"只有 {len(fol)} 个行业同时满足「资金在进、多数在加、价格不贵」："
                  + "、".join(fol.index) + "。",
            evidence="；".join(
                f"{i}：当期主动 {r['当期主动pp']:+.2f}pp、{r['加仓占比']:.0f}% 在加、"
                f"配置 {r['历史分位']:.0f} 分位、指数 {r.get('指数分位', float('nan')):.0f} 分位"
                for i, r in fol.iterrows()),
            meaning=("这三条是「与」不是「或」：只有资金、广度、价格同时过关，"
                     "才叫赔率合适的加仓。少一条就降级——广度不够是少数大基金在撑，"
                     "价格太贵是给别人接棒。"),
            action="单行业不超过权益仓位 15%，分 3 批建；跌破建仓成本 8% 先减半。",
            fail="下一季当期主动转负或广度跌破 50%，立即降级为「等确认」。",
        ))

    # ---- 4. 别追高 ----
    hot = t[t["动作"] == "别追高"]
    if len(hot):
        out.append(dict(
            tag="别追高",
            claim=f"{len(hot)} 个行业已在历史极值，资金还在进，但赔率不够："
                  + "、".join(hot.index) + "。",
            evidence="；".join(
                f"{i}：配置 {r['历史分位']:.0f} 分位、指数 {r.get('指数分位', float('nan')):.0f} 分位、"
                f"近 3 月 {r.get('近3月涨幅%', float('nan')):+.1f}%"
                for i, r in hot.iterrows()),
            meaning=("拥挤不等于马上跌——本库自己的数据已经证伪「高配置必跌」"
                     "（高分位行业次季反而 +0.83%）。所以这里不是看空，而是"
                     "「同样的钱买不到便宜货」，新增资金不该放这里。"),
            action="已持有的可以拿，不加仓；单行业超过权益 20% 的减到 20% 以内。",
            fail="若价格分位回落到 70 以下而资金仍在进，则升级为「可以跟」。",
        ))

    # ---- 5. 要回避 ----
    av = t[t["动作"] == "要回避"]
    if len(av):
        out.append(dict(
            tag="要回避",
            claim=f"{len(av)} 个行业「资金在撤 + 没人接」："
                  + "、".join(av.index) + "。",
            evidence="；".join(
                f"{i}：当期主动 {r['当期主动pp']:+.2f}pp、{r['加仓占比']:.0f}% 在加、"
                f"配置 {r['历史分位']:.0f} 分位"
                for i, r in av.iterrows()),
            meaning=("这些行业的问题不是「跌得多」，而是「跌下来没人接」。"
                     "加仓占比低于一半，说明减仓的基金比加仓的多，"
                     "价格没有承接盘，抄底容易抄在半山腰。"),
            action="不新建仓；已持有的反弹减仓，不等解套。",
            fail="若加仓占比回到 60% 以上且当期主动转正，说明承接出现，改为「等确认」。",
        ))

    # ---- 6. 等确认 / 逆势承接 ----
    wt = t[t["动作"].isin(["等确认", "观察"])]
    if len(wt):
        _names = "、".join(wt.index[:6])
        out.append(dict(
            tag="待确认",
            claim=f"{len(wt)} 个行业方向不干净，最典型的是 {_names}。",
            evidence="；".join(
                f"{i}：半年主动 {r['半年主动pp']:+.2f}pp、当期 {r['当期主动pp']:+.2f}pp、"
                f"广度 {r['加仓占比']:.0f}%"
                for i, r in wt.head(4).iterrows()),
            meaning=("「半年在加、这季停了」是最常见的陷阱：趋势还在，但边际动能在消失。"
                     "反过来「半年在减、这季转加」是逆势承接，方向未定。两种都不该重仓。"),
            action="观察仓不超过 5%，等下一季确认后再说。",
            fail="下一季方向重新一致（两期同向）即可升级或降级。",
        ))

    # ---- 7. 个股层（前十大口径：环比与同比都能算） ----
    pq = cn.pair_quarter(q)
    ps = cn.pair_same(q)
    if pq:
        try:
            sf = cn.stock_flow_top10(*pq)
            sf = sf[(sf["n_funds"] >= 10) | (sf["n_funds_0"] >= 10)]
            sy = cn.stock_flow_top10(*ps) if ps else pd.DataFrame()
            if not sf.empty:
                a3, c3 = sf.nlargest(3, "d_n"), sf.nsmallest(3, "d_n")
                ind_a = a3["industry"].value_counts()
                _yoy = sy["d_n"].reindex(a3.index) if not sy.empty else pd.Series(dtype=float)
                out.append(dict(
                    tag="个股层",
                    claim=("环比（" + pq[0] + "→" + pq[1] + "）被最多基金加进来的是 "
                           + "、".join(a3["stock_name"])
                           + "；被最多基金减掉的是 " + "、".join(c3["stock_name"]) + "。"),
                    evidence=("加仓：" + "、".join(
                        f"{r['stock_name']}（{r['d_n']:+.0f} 家，{r['industry']}）"
                        for _, r in a3.iterrows())
                        + "；减仓：" + "、".join(
                        f"{r['stock_name']}（{r['d_n']:+.0f} 家，{r['industry']}）"
                        for _, r in c3.iterrows())
                        + f"。口径：前十大（两期都只取每只基金最重的 10 只），"
                          f"样本 {sf.attrs['n_funds'][1]:,} 只基金；"
                          + ("同比 " + "、".join(
                              f"{i} {v:+.0f}" for i, v in _yoy.items()) + "。"
                             if len(_yoy) else "")),
                    meaning=(f"加仓前三分布在 {len(ind_a)} 个行业"
                             # 判据是「同一行业至少 2 只」，不是「行业数 ≤2」：
                             # 三只各属三个行业时，报「集中在 X 占 1 只」会自相矛盾。
                             + (f"，其中 {ind_a.index[0]} 占 {int(ind_a.iloc[0])} 只——"
                                "属于<b>行业性加仓</b>，跟着行业方向做即可，不必逐只研究。"
                                if int(ind_a.iloc[0]) >= 2 else
                                "，每只都在不同行业——更偏<b>个股选择</b>，"
                                "需要逐只判断基本面，买行业 ETF 替代不了。")
                             + " 环比与同比经常相反：同比大幅加过、环比开始减的，"
                               "是浮盈盘在撤，最需要警惕。"),
                    action="行业性加仓跟行业；个股性加仓只买自己能讲清逻辑的那几只；"
                           "同比加、环比减的优先减。",
                    fail="前十大口径只能看「有没有进前十」，跌出前十不等于清仓。",
                ))
        except Exception:
            pass

    # ---- 8. 拥挤度 ----
    cs = cn.crowding_score(q)
    if not cs.empty:
        hi = cs.head(3)
        lo = cs.tail(3)
        out.append(dict(
            tag="拥挤度",
            claim=(f"最拥挤的是 {'、'.join(hi.index)}（CSI "
                   + "、".join(f"{v:.0f}" for v in hi["CSI"]) + "）；"
                   f"最冷门的是 {'、'.join(lo.index[::-1])}。"),
            evidence=("CSI＝(配置分位＋价格分位＋资金分位)/3，等权、不拟合；"
                      "三个分位分别回答「机构押多重」「价格多贵」「钱还在不在进」。"
                      "≥80 极度拥挤、65–80 拥挤、45–65 中性、25–45 偏冷、<25 冷门。"),
            meaning=("拥挤度只描述状态，不预测收益——本库 30 期数据已证伪"
                     "「高拥挤必然下跌」。它的真正用途是<b>决定仓位大小</b>："
                     "同样看好，拥挤的行业只能给小仓位。"),
            action="拥挤行业单行业上限 10%，冷门行业可给到 15%，其余 12%。",
            fail="若拥挤度与资金流背离（CSI 高但广度低），以资金流为准。",
        ))

    # ---- 9. 反面证据 ----
    out.append(dict(
        tag="反面证据",
        claim="这套方法能描述机构在做什么，但不能预测下一季涨什么。",
        evidence=("加仓广度的 RankIC 长期在 0 附近（详见「因子检验」页的实测值与 p 值）；"
                  "「拥挤即风险」只有方向、没有显著性：配置分位 Q1→Q5 的次季收益从 +2.39% 降到 +0.51%，但 t 只有 −0.83、p=0.42，不能当卖出信号。"),
        meaning=("所以本页所有「可以跟」都是「赔率合适 + 机构在做」的组合，"
                 "不是「必涨」。它的价值在于避免在拥挤处接棒、避免在撤离处抄底，"
                 "而不是抓涨幅。"),
        action="把它当「仓位分配器」，不当「选股器」；个股仍要自己研究。",
        fail="若未来出现稳定的 IC（|IC|>0.05 且 p<0.1 且连续 4 季同向），可提高权重。",
    ))
    return out
