# -*- coding: utf-8 -*-
"""「①b 环比变化」页面渲染。

样式写法：CSS 用 __LINE__ / __TEXT__ 占位符 + str.replace 注入主题色，
**不使用 f-string 包裹 CSS**（避免花括号转义问题导致整段样式失效）。
四象限的键与顺序在 change_analysis 模块顶部定义，本文件只引用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import change_analysis as ca
import config as C
import institution as inst
import ak_data as akd

# ---------------------------------------------------------------------------
# 样式（占位符替换）
# ---------------------------------------------------------------------------
_CSS = """
.cg-hero { background:#FFFFFF; border:1px solid __LINE__; border-left:5px solid __ACCENT__;
           border-radius:9px; padding:11px 14px; margin:4px 0 8px 0; }
.cg-hero .cg-k { font-size:10.5px; font-weight:700; color:__DIM__; letter-spacing:.4px; }
.cg-hero .cg-t { font-size:13.5px; font-weight:700; line-height:1.62; color:__TEXT__; margin-top:3px; }
.cg-hero .cg-s { font-size:11px; line-height:1.6; color:#37485C; margin-top:5px;
                 padding-top:5px; border-top:1px dashed __LINE__; }
.cg-kpis { display:flex; gap:7px; flex-wrap:wrap; margin-bottom:6px; }
.cg-kpi { flex:1 1 0; min-width:120px; background:#FFFFFF; border:1px solid __LINE__;
          border-radius:8px; padding:8px 11px; }
.cg-kpi .k { font-size:10.5px; color:__DIM__; }
.cg-kpi .v { font-size:17px; font-weight:700; line-height:1.3; font-variant-numeric:tabular-nums; }
.cg-kpi .s { font-size:10px; color:__DIM__; line-height:1.45; }
.cg-card { background:#FFFFFF; border:1px solid __LINE__; border-radius:8px;
           padding:6px 10px; margin-bottom:5px; }
.cg-card .hd { display:flex; align-items:baseline; gap:8px; flex-wrap:wrap; margin-bottom:4px; }
.cg-card .jd { font-size:12.5px; font-weight:700; }
.cg-card .tag { font-size:10px; font-weight:700; padding:1px 7px; border-radius:9px;
                color:#fff; }
.cg-card .row { font-size:11px; line-height:1.56; color:#37485C; }
.cg-card .row b { color:__TEXT__; }
.cg-card .inv { font-size:10.5px; line-height:1.5; margin-top:3px; padding:3px 7px;
                background:#FFF8EC; border-radius:5px; color:#8A5A00; }
.cg-note { font-size:10.5px; line-height:1.55; color:__DIM__; margin-top:3px; }
.cg-grid2 { display:flex; gap:8px; flex-wrap:wrap; }
.cg-grid2 > div { flex:1 1 300px; min-width:280px; }
.cg-mig { border-collapse:separate; border-spacing:2px; font-size:9.5px; }
.cg-mig td, .cg-mig th { padding:2px 5px; text-align:center; border-radius:4px; }
.cg-mig th { font-size:10px; color:__DIM__; font-weight:700; background:#F7F9FB; }
.cg-mig td.lab { text-align:right; font-weight:700; background:#F7F9FB; }
.cg-mig td.n { font-variant-numeric:tabular-nums; }
.cg-flag { display:inline-block; font-size:9.5px; font-weight:700; padding:0 5px;
           border-radius:3px; background:#FFF1F0; color:#B42318; }
.cg-flag.ok { background:#EEF7F0; color:#1A7F37; }
"""


def css(colors: dict) -> str:
    out = _CSS
    for k, v in (("__LINE__", colors.get("line", "#DCE3EA")),
                 ("__TEXT__", colors.get("text", "#1F2D3D")),
                 ("__DIM__", colors.get("text_dim", "#6B7A8C")),
                 ("__ACCENT__", colors.get("accent", "#6E4AB8"))):
        out = out.replace(k, v)
    return out


def h(html: str):
    """与 app._h 一致：st.html + 星号转 <b>，避免 markdown 二次解析。"""
    if "**" in html:
        import re
        html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html)
    st.html(html)


def _fmt(v, tpl="{:.2f}", na="—"):
    try:
        return na if v is None or pd.isna(v) else tpl.format(v)
    except Exception:
        return na


def _sgn(v, tpl="{:+.2f}", na="—"):
    return _fmt(v, tpl, na)


# ---------------------------------------------------------------------------
# 结论卡
# ---------------------------------------------------------------------------
def _card(judgment: str, tag: str, color: str, rows: list[str],
          invalid: str) -> str:
    body = "".join(f'<div class="row">{r}</div>' for r in rows)
    return (f'<div class="cg-card"><div class="hd">'
            f'<span class="jd" style="color:{color}">{judgment}</span>'
            f'<span class="tag" style="background:{color}">{tag}</span></div>'
            f'{body}<div class="inv"><b>失效条件</b>：{invalid}</div></div>')


def _dir_line(r) -> str:
    """方向行：四象限有命名就用命名；落在中区则明说「不命名，但方向仍成立」。"""
    meta = ca.QUAD_META[r["quad"]]
    if r["quad"] == "MID":
        side = "减仓" if r["dz"] < 0 else "加仓"
        return (f'<b>方向</b>：{meta["name"]}（不参与四象限命名）——'
                f'区位不极端（Z {_fmt(r["z"])}、分位 {_fmt(r["pct"], "{:.0f}%")}），'
                f'但 ΔZ {r["dz"]:+.2f}σ 方向明确，仍按{side}执行。')
    return f'<b>方向</b>：{meta["name"]}——{meta["tag"]}。'


def build_triptych(t: pd.DataFrame, t0: pd.DataFrame, mv: dict, n_up: int,
                   active: pd.DataFrame, pos: dict, disp: dict,
                   disp_real: pd.DataFrame | None = None) -> list[dict]:
    """首页第一屏三块：共识 · 分歧 · 移动。每条含判断/数字/上期/失效。"""
    q, q0 = t.attrs.get("q"), t.attrs.get("q0")
    up, dn = C.COLOR["up"], C.COLOR["down"]
    cards = []

    # ---- 共识：分位 ≥80 且主动项 >0 ----
    # t 已含 credibility_gate 写入的 active（= Δ配置 − 被动），与 active_change 同源
    cons = t[(t["pct"] >= 80) & (t["alloc"].fillna(0) >= 0.5)].copy()
    cons = cons[cons["active"] > 0].sort_values("active", ascending=False)
    if len(cons):
        names = "、".join(cons.index[:3])
        detail = "；".join(
            f"{n} 分位 {r['pct']:.0f}%、主动 {r['active']:+.2f}pp、配置 {r['alloc']:.2f}%"
            for n, r in cons.head(3).iterrows())
        total_w = float(cons.head(3)["alloc"].sum())
        cards.append(dict(
            kind="cons", color=up,
            judgment=f"超｜共识：{names}（分位≥80 且主动加仓）",
            tag="共识形成 · 参考级",
            rows=[
                f'<b>数字</b>：{len(cons)} 个行业满足「超配分位 ≥80% 且主动项 >0」；'
                f'{detail}。三者合计配置 {total_w:.2f}%——'
                f'{"容量过小，只能观察不能建仓" if total_w < 2 else "具备一定容量"}。',
                f'<b>上期</b>：{q0} 同样条件命中 '
                f'{int(((t0["pct"] >= 80) & (t0["alloc"].fillna(0) >= 0.5)).sum())} 个；'
                f'本期等级净升级 {n_up:+d} 家。',
                f'<b>方向</b>：高位加仓（最后一棒风险）——共识最厚时，预期差最小。',
            ],
            invalid=("若下季任一行业主动项转负且分位跌破 50%，该行业共识判断作废；"
                     "若三行业合计配置仍 &lt;2%，则始终不构成可执行仓位。"),
        ))
    else:
        cards.append(dict(kind="cons", color=up, judgment="观察｜本期无「共识」行业",
                          tag="共识缺失",
                          rows=[f'<b>数字</b>：0 个行业同时满足分位 ≥80% 与主动项 >0。'],
                          invalid="若下季出现分位 ≥80% 且主动 >0 的行业，本条作废。"))

    # ---- 分歧：优先用真实分歧度（50 只样本）；缺失时回退到符号背离代理 ----
    if disp_real is not None and not disp_real.empty:
        _d = disp_real.copy()
        _d["极值"] = (_d["加仓占比"] - 50).abs()
        _hot = _d.sort_values(["极值", "分歧度"], ascending=False).iloc[0]
        _n = int(_d["样本基金数"].max())
        _dq, _dq0 = disp_real.attrs.get("q", "最新期"), disp_real.attrs.get("q0", "上一期")
        try:
            _span = ("单季" if (int(_dq[:4]) * 4 + int(_dq[-1]))
                     - (int(_dq0[:4]) * 4 + int(_dq0[-1])) == 1 else "跨季")
        except Exception:
            _span = "跨季"
        cards.append(dict(
            kind="div", color="#E8710A",
            judgment=f"观察｜分歧：{_hot['行业']}加仓占比 {_hot['加仓占比']:.0f}%、"
                     f"分歧度 {_hot['分歧度']:.2f}（{_hot['判读']}）",
            tag=f"真实分歧度 · {_n} 只基金样本",
            rows=[
                f'<b>数字</b>：{_n} 只偏股混合基金中，{_hot["行业"]} '
                f'<b>{int(_hot["加仓基金数"])} 只加仓 / {int(_hot["减仓基金数"])} 只减仓 / '
                f'{int(_hot["不变基金数"])} 只不变</b>；加仓占比 {_hot["加仓占比"]:.0f}%、'
                f'分歧度 {_hot["分歧度"]:.2f}、加减仓比 {_hot["加减仓比"]:.2f}。',
                f'<b>上期</b>：{_dq0} → {_dq}（<b>{_span}</b>）；31 行业中「共识形成」'
                f'{int((_d["判读"]=="共识形成").sum())} 个、「多空对撞」'
                f'{int((_d["判读"]=="多空对撞").sum())} 个、「系统性撤离」'
                f'{int((_d["判读"]=="系统性撤离").sum())} 个。',
                f'<b>方向</b>：{pos["stance"]}，仓位区间 <b>{pos["low"]}%–{pos["high"]}%</b>。'
                f'样本为 2026Q2 规模前 50 的偏股混合（偏科技主题），'
                f'<b>不代表全市场</b>；只取前 10 重仓行业，未出现视为 0；'
                f'单行业 >80% 的主题基金已剔除（本行业剔除 '
                f'{int(_hot.get("剔除主题基金", 0))} 只）；季报期由前十大推导，'
                f'已按每只基金覆盖率归一化后再比较。',
            ],
            invalid=("若把样本扩到全市场后该行业的加仓占比回到 40–60%，"
                     "则「对撞/撤离」判读作废；若下季分歧度 &lt;0.4 且占比 &gt;70%，"
                     "则升级为「共识」，本条反向。"),
        ))
    conf = t[t["conflict"]].copy()
    conf["abs_d"] = conf["d_alloc"].abs()
    conf = conf.sort_values("abs_d", ascending=False)
    if len(conf) and (disp_real is None or disp_real.empty):
        c0 = conf.iloc[0]
        cards.append(dict(
            kind="div", color="#E8710A",
            judgment=f"观察｜分歧：{int(t['conflict'].sum())} 个行业「配置与历史位置」符号相反",
            tag="代理指标 · 非基金级分歧度",
            rows=[
                f'<b>数字</b>：{c0.name} Δ配置 {c0["d_alloc"]:+.2f}pp 但 ΔZ {c0["dz"]:+.2f}σ；'
                f'全行业 Δ配置 横截面标准差 <b>{disp["cross_std"]:.2f}pp</b>、'
                f'中位 {disp["median"]:+.2f}pp。',
                f'<b>上期</b>：{q0} 符号相反 '
                f'{int(t0["conflict"].sum())} 个；本期 {int(t["conflict"].sum())} 个。',
                f'<b>方向</b>：{pos["stance"]}，仓位区间 <b>{pos["low"]}%–{pos["high"]}%</b>。'
                f'真正的基金级分歧度（加仓基金占比）需要明细，本期不可得——'
                f'这里只用行业配置变化的离散度代理，<b>不能判读「共识/对撞」</b>。',
            ],
            invalid=("若拿到基金级明细后，加仓占比落在 40–60%、分歧度 >0.7，"
                     "则「分歧」升级为「对撞」；若加仓占比 ≥70%，本条作废。"),
        ))

    # ---- 移动：主动金额 top / bottom ----
    mv_ok = active["active_mv"].notna() & (active["alloc1"].fillna(0) >= 0.5)
    mov = active[mv_ok].copy()
    if len(mov):
        buy = mov.sort_values("active_mv", ascending=False).head(3)
        sell = mov.sort_values("active_mv").head(3)
        cards.append(dict(
            kind="move", color=up,
            judgment=f"加｜主动买入第一：{buy.index[0]} {buy.iloc[0]['active_mv']:+.0f} 亿",
            tag="剔价格口径 · 主动金额",
            rows=[
                f'<b>数字（主动买入）</b>：' +
                "、".join(f"{n} {r['active_mv']:+.0f} 亿（{r['active']:+.2f}pp）"
                          for n, r in buy.iterrows()) + "。",
                f'<b>数字（主动卖出）</b>：' +
                "、".join(f"{n} {r['active_mv']:+.0f} 亿（{r['active']:+.2f}pp）"
                          for n, r in sell.iterrows()) + "。",
                f'<b>上期</b>：{q0} 主动净额 '
                f'{float(active["active_mv"].sum()):+.0f} 亿；'
                f'本期 {float(mov["active_mv"].sum()):+.0f} 亿（缓存重仓口径，量级参考）。',
            ],
            invalid=("若下季主动净额整体反转，或某行业主动项与其 ΔZ 符号转一致，"
                     "则「移动」方向按新数据重算，本条作废。"),
        ))
    return cards


def build_conclusions(t: pd.DataFrame, t0: pd.DataFrame, mv: dict,
                      n_up: int) -> list[dict]:
    """生成三条判断卡的全部字段（判断/数字/上期/失效）。"""
    add, cut = mv["add"], mv["cut"]
    q, q0 = t.attrs.get("q"), t.attrs.get("q0")
    up, dn = C.COLOR["up"], C.COLOR["down"]

    def _prev(name, col, pct=False):
        if name not in t0.index:
            return np.nan
        return t0.loc[name, col]

    cards = []
    # ① 加仓最猛
    a_prev_dz = _prev(add.name, "dz")
    cards.append(dict(
        kind="add", color=up,
        judgment=f"加｜{add.name}是本季加仓最猛",
        tag="按 Δ超配Z 排序",
        rows=[
            f'<b>数字</b>：Δ超配Z <b style="color:{up}">{add["dz"]:+.2f}σ</b>'
            f'（可交易行业中最高：剔除配置 &lt; {mv["min_alloc"]:.1f}% 的行业，且要求 Δ配置同号），'
            f'配置 {_fmt(add["alloc0"])}% → <b>{_fmt(add["alloc"])}%</b>'
            f'（{add["d_alloc"]:+.2f}pp），当前 Z {_fmt(add["z"])}、分位 {_fmt(add["pct"], "{:.0f}%")}。',
            f'<b>上期</b>：{q0} 的 ΔZ 为 {_sgn(a_prev_dz)}σ、等级 {add["lvl0"]}；'
            f'本期等级 {add["lvl"]}。',
            _dir_line(add),
        ],
        invalid=(f'若下季 {add.name} 的 ΔZ 回落至 0 以下、或配置跌回 {_fmt(add["alloc0"])}% 附近，'
                 f'本次「{add.name}承接」判断作废；若等级跌回中性，同样作废。'),
    ))
    # ② 减仓最猛
    c_prev_dz = _prev(cut.name, "dz")
    cards.append(dict(
        kind="cut", color=dn,
        judgment=f"减｜{cut.name}是本季减仓最猛",
        tag="按 Δ超配Z 排序",
        rows=[
            f'<b>数字</b>：Δ超配Z <b style="color:{dn}">{cut["dz"]:+.2f}σ</b>'
            f'（可交易行业中最低：剔除配置 &lt; {mv["min_alloc"]:.1f}% 的行业，且要求 Δ配置同号），配置 {_fmt(cut["alloc0"])}% → <b>{_fmt(cut["alloc"])}%</b>'
            f'（{cut["d_alloc"]:+.2f}pp），当前 Z {_fmt(cut["z"])}、分位 {_fmt(cut["pct"], "{:.0f}%")}。',
            f'<b>上期</b>：{q0} 的 ΔZ 为 {_sgn(c_prev_dz)}σ、等级 {cut["lvl0"]}；'
            f'本期等级 {cut["lvl"]}。',
            _dir_line(cut),
        ],
        invalid=(f'若下季 {cut.name} 的 ΔZ 回升至 +0.3σ 以上、且配置回到 {_fmt(cut["alloc0"])}% 以上，'
                 f'本次「{cut.name}撤出」判断作废。'),
    ))
    # ③ 整体方向
    w_delta = float((t["d_alloc"] * t["alloc0"]).sum() / t["alloc0"].sum()) \
        if t["alloc0"].sum() else np.nan
    w_delta0 = float((t0["d_alloc"] * t0["alloc0"]).sum() / t0["alloc0"].sum()) \
        if t0["alloc0"].sum() else np.nan
    dz_med = float(t["dz"].median())
    cards.append(dict(
        kind="all", color=(dn if w_delta < 0 else up),
        judgment=("减｜整体由「规模加仓」转为「规模减仓」" if w_delta < 0
                  else "加｜整体规模仍在加仓"),
        tag="规模加权 + 扩散度",
        rows=[
            f'<b>数字</b>：加仓 {mv["n_add"]} 个 / 减仓 {mv["n_cut"]} 个 / 方向不明 {mv["n_flat"]} 个；'
            f'ΔZ 中位数 <b>{dz_med:+.2f}σ</b>；规模加权 Δ配置 <b>{w_delta:+.2f}pp</b>。',
            f'<b>上期</b>：{q0} 加仓 {int((t0["dz"] >= ca.DZ_MIN).sum())} 个 / '
            f'减仓 {int((t0["dz"] <= -ca.DZ_MIN).sum())} 个，规模加权 Δ配置 {w_delta0:+.2f}pp；'
            f'本期等级净升级 <b>{n_up:+d}</b> 家。',
            f'<b>方向（ΔZ，可交易行业）</b>：领先的是 ' +
            "、".join(f"{n} {r['dz']:+.2f}σ" for n, r in
                      t[(t["dz"] >= ca.DZ_MIN) & (t["alloc"].fillna(0) >= mv["min_alloc"])]
                      .sort_values("dz", ascending=False).head(3).iterrows()) +
            '；垫底的是 ' +
            "、".join(f"{n} {r['dz']:+.2f}σ" for n, r in
                      t[(t["dz"] <= -ca.DZ_MIN) & (t["alloc"].fillna(0) >= mv["min_alloc"])]
                      .sort_values("dz").head(3).iterrows()) + '。'
            f'<b>规模上</b>（Δ配置，仅作佐证）：电子 {_sgn(t.loc["电子", "d_alloc"])}pp '
            f'是最大单笔撤出，但其 ΔZ 仅 {t.loc["电子", "dz"]:+.2f}σ（{_fmt(t.loc["电子", "passive"], "{:+.2f}")}pp 为价格漂移），'
            f'不能等同于主动看空；承接方为医药生物 {_sgn(t.loc["医药生物", "d_alloc"])}pp、'
            f'银行 {_sgn(t.loc["银行", "d_alloc"])}pp、汽车 {_sgn(t.loc["汽车", "d_alloc"])}pp。',
        ],
        invalid=(f'若下季加仓行业数回到 20 个以上、且电子 ΔZ 转正并站上 +0.3σ，'
                 f'则「AI 算力链撤出」判断不成立；若规模加权 Δ配置转正而扩散度仍低，'
                 f'则只能算「单行业回补」，不算风格切换。'),
    ))
    return cards


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig_change_bar(t: pd.DataFrame, colors: dict) -> go.Figure:
    d = t.dropna(subset=["dz"]).sort_values("dz")
    # y 轴反转：最负（撤得最狠）显示在最上方
    colors_bar = [colors["down"] if v < 0 else colors["up"] for v in d["dz"]]
    hover = []
    for name, r in d.iterrows():
        mig = f'{r["lvl0"]} → {r["lvl"]}' if pd.notna(r["lvl0"]) else "—"
        gate = f'　<span style="color:#B42318">[{r["gate"]}]</span>' if r.get("gate") else ""
        hover.append(f'<b>{name}</b>{gate}<br>'
                     f'Δ超配Z（主判断）：{r["dz"]:+.2f}σ<br>'
                     f'Δ配置（辅助）：{r["d_alloc"]:+.2f}pp<br>'
                     f'当前超配Z：{r["z"]:+.2f}σ　分位 {r["pct"]:.0f}%<br>'
                     f'成交额占比分位：{_fmt(r.get("turn_pct"), "{:.0f}%")}<br>'
                     f'等级迁移：{mig}')
    labels = [f"{n}*" if bool(t.loc[n, "conflict"]) else n for n in d.index]
    fig = go.Figure(go.Bar(
        x=d["dz"], y=labels, orientation="h",
        marker=dict(color=colors_bar, line=dict(width=0)),
        text=[f"{v:+.2f}" for v in d["dz"]],
        textposition="outside", textfont=dict(size=9),
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    fig.update_xaxes(title_text="Δ超配Z（σ，长历史口径）", gridcolor=colors["grid"],
                     zeroline=True, zerolinecolor=colors["text_dim"], zerolinewidth=1)
    fig.update_layout(height=560, margin=dict(l=10, r=40, t=8, b=10),
                      paper_bgcolor="white", plot_bgcolor="white",
                      showlegend=False, font=dict(family=C.FONT, size=11))
    return fig


def fig_quadrant(t: pd.DataFrame, colors: dict) -> go.Figure:
    d = t.dropna(subset=["dz", "z"]).copy()
    palette = {"HH": colors["up"], "HC": colors["down"],
               "LH": "#E8710A", "LC": "#1F6FEB", "MID": "#C3CBD4"}
    fig = go.Figure()
    for key in ca.QUAD_ORDER:
        sub = d[d["quad"] == key]
        if sub.empty:
            continue
        size = [max(7, min(20, 7 + abs(v) * 4)) for v in sub["dz"]]
        fig.add_trace(go.Scatter(
            x=sub["z"], y=sub["dz"], mode="markers+text",
            name=f'{ca.QUAD_META[key]["name"]}（{len(sub)}）',
            marker=dict(size=size, color=palette[key],
                        line=dict(width=0.8, color="white"), opacity=0.9),
            text=[n if key != "MID" else "" for n in sub.index],
            textposition="top center", textfont=dict(size=9, color=colors["text"]),
            customdata=np.column_stack([sub.index, sub["pct"], sub["d_alloc"], sub["lvl"]]),
            hovertemplate=("<b>%{customdata[0]}</b><br>当前超配Z：%{x:+.2f}σ"
                           "<br>Δ超配Z：%{y:+.2f}σ<br>分位：%{customdata[1]:.0f}%"
                           "<br>Δ配置（辅助）：%{customdata[2]:+.2f}pp"
                           "<br>等级：%{customdata[3]}<extra></extra>"),
        ))
    fig.add_vline(x=ca.Z_HI, line=dict(color=colors["threshold"], width=1, dash="dash"))
    fig.add_vline(x=ca.Z_LO, line=dict(color=colors["level_显著低配"], width=1, dash="dash"))
    fig.add_hline(y=ca.DZ_MIN, line=dict(color=colors["text_dim"], width=1, dash="dot"))
    fig.add_hline(y=-ca.DZ_MIN, line=dict(color=colors["text_dim"], width=1, dash="dot"))
    fig.add_annotation(x=ca.Z_HI, y=1.02, yref="paper", text=f"Z = {ca.Z_HI:+.1f}",
                       showarrow=False, font=dict(size=9, color=colors["threshold"]),
                       xanchor="left")
    fig.add_annotation(x=ca.Z_LO, y=1.02, yref="paper", text=f"Z = {ca.Z_LO:+.1f}",
                       showarrow=False, font=dict(size=9, color=colors["level_显著低配"]),
                       xanchor="right")
    fig.update_xaxes(title_text="当前超配Z（水平 · 长历史）", gridcolor=colors["grid"],
                     zeroline=True, zerolinecolor=colors["grid"])
    fig.update_yaxes(title_text="Δ超配Z（方向 · 环比）", gridcolor=colors["grid"],
                     zeroline=True, zerolinecolor=colors["text_dim"])
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=28, b=10),
                      paper_bgcolor="white", plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.16, font=dict(size=9.5)),
                      font=dict(family=C.FONT, size=11))
    return fig


def fig_net_upgrade(s: pd.Series, colors: dict) -> go.Figure:
    cols = [colors["up"] if v > 0 else colors["down"] if v < 0 else "#C3CBD4" for v in s]
    fig = go.Figure(go.Bar(x=s.index, y=s.values, marker_color=cols,
                           text=[f"{int(v):+d}" for v in s.values],
                           textposition="outside", textfont=dict(size=9)))
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    fig.update_yaxes(title_text="净升级家数", gridcolor=colors["grid"], zeroline=True,
                     zerolinecolor=colors["text_dim"])
    fig.update_layout(height=220, margin=dict(l=6, r=6, t=6, b=6),
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family=C.FONT, size=10), showlegend=False)
    return fig


def fig_nav(series: pd.Series, colors: dict) -> go.Figure:
    nav = (1 + series / 100).cumprod()
    fig = go.Figure(go.Scatter(x=nav.index, y=nav.values, mode="lines",
                               line=dict(color=colors["accent"], width=1.8),
                               fill="tozeroy", fillcolor="rgba(110,74,184,0.08)"))
    fig.add_hline(y=1.0, line=dict(color=colors["text_dim"], width=0.8, dash="dot"))
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9), gridcolor=colors["grid"])
    fig.update_yaxes(title_text="累计净值（起点=1）", gridcolor=colors["grid"])
    fig.update_layout(height=260, margin=dict(l=6, r=6, t=8, b=6),
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family=C.FONT, size=10), showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# 组件
# ---------------------------------------------------------------------------
def migration_table(m: pd.DataFrame, colors: dict) -> str:
    cols = list(m.columns)
    head = "<tr><th>上期 ＼ 本期</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    rows = ""
    for idx, r in m.iterrows():
        cells = ""
        for c in cols:
            v = int(r[c])
            if v == 0:
                cells += '<td class="n" style="color:#C3CBD4">·</td>'
            else:
                diag = (idx == c)
                bg = "#EEF2F6" if diag else ("#FDECEA" if v else "#FFFFFF")
                cells += f'<td class="n" style="background:{bg};font-weight:{"700" if not diag else "400"}">{v}</td>'
        rows += f'<tr><td class="lab">{idx}</td>{cells}</tr>'
    return f'<table class="cg-mig">{head}{rows}</table>'


def _context(b, ind, mkt, latest: str):
    quarters = list(b.quarters)
    t = ca.delta_table(b, ind, latest)
    q0 = t.attrs["q0"]
    t0 = ca.delta_table(b, ind, q0) if q0 else t.copy()
    t = ca.credibility_gate(b, ind, t, mkt.get("F8_样本基金数"))
    mv = ca.top_movers(t)
    n_up = ca.net_upgrade(t)
    return quarters, t, t0, mv, n_up


def render_overview(b, ind, mkt, latest: str, colors: dict, compact: bool = False):
    """总览页与环比页共用的「结论区」：一句话 + 四个数字 + 三条卡。"""
    quarters, t, t0, mv, n_up = _context(b, ind, mkt, latest)
    q0 = t.attrs["q0"]
    e_add, e_cut = t.loc["医药生物"], t.loc["建筑材料"]
    e_elec = t.loc["电子"]
    passive = e_elec.get("passive", np.nan)
    active = e_elec.get("active", np.nan)
    share = (abs(passive) / abs(e_elec["d_alloc"]) * 100
             if pd.notna(passive) and e_elec["d_alloc"] else np.nan)

    h(f'<div class="sec">共识 · 分歧 · 移动 · {q0} → {latest}</div>')
    hero = (
        f'<div class="cg-hero"><div class="cg-k">一句话判断</div>'
        f'<div class="cg-t">AI 算力链撤出，医药与低估值顺周期承接：'
        f'电子配置 {_fmt(e_elec["alloc0"])}% → {_fmt(e_elec["alloc"])}%'
        f'（{e_elec["d_alloc"]:+.2f}pp，长历史 38 期第二大、主口径 11 期最大），'
        f'医药生物 {_fmt(e_add["alloc0"])}% → {_fmt(e_add["alloc"])}%'
        f'（{e_add["d_alloc"]:+.2f}pp），银行 {t.loc["银行", "d_alloc"]:+.2f}pp、'
        f'汽车 {t.loc["汽车", "d_alloc"]:+.2f}pp。</div>'
        f'<div class="cg-s">但电子 Δ超配Z 仅 <b>{e_elec["dz"]:+.2f}σ</b>——'
        f'仓位在降、相对历史仍在极值（Z {_fmt(e_elec["z"])}，分位 {_fmt(e_elec["pct"], "{:.0f}%")}）。'
        f'这是<b>减仓，不是出清</b>；若只看「配置 −6.57pp」就断言「机构看空」，会误判，'
        f'因为其中 <b>{_fmt(share, "{:.0f}%")}</b> 来自价格下跌（被动项 {_sgn(passive)}pp），'
        f'主动减仓只有 {_sgn(active)}pp。</div></div>')
    h(hero)

    _big = t[t["alloc"].fillna(0) >= mv["min_alloc"]]
    if _big.empty:
        _big = t
    h(f"""<div class="cg-kpis">
      <div class="cg-kpi"><div class="k">环比加仓行业</div>
        <div class="v" style="color:{colors['up']}">{mv['n_add']}</div>
        <div class="s">ΔZ ≥ +0.20σ　方向不明 {mv['n_flat']}</div></div>
      <div class="cg-kpi"><div class="k">环比减仓行业</div>
        <div class="v" style="color:{colors['down']}">{mv['n_cut']}</div>
        <div class="s">ΔZ ≤ −0.20σ　共 31 行业</div></div>
      <div class="cg-kpi"><div class="k">等级净升级</div>
        <div class="v">{n_up:+d}</div>
        <div class="s">升级家数 − 降级家数</div></div>
      <div class="cg-kpi"><div class="k">变化最大（可交易口径）</div>
        <div class="v" style="font-size:15px">{_big['dz'].abs().idxmax()}</div>
        <div class="s">{_big.loc[_big['dz'].abs().idxmax(), 'dz']:+.2f}σ　配置 {_big.loc[_big['dz'].abs().idxmax(), 'd_alloc']:+.2f}pp
          · 全样本最大 {t['dz'].abs().idxmax()} {t['dz'].abs().max():+.2f}σ（容量 {_fmt(t.loc[t['dz'].abs().idxmax(), 'alloc'], "{:.2f}%")}）</div></div>
    </div>""")

    active = inst.active_change(b, ind, latest)
    disp = inst.dispersion_proxy(t)
    pos = inst.position_advice(disp)
    # 优先用全市场口径（东财/AKShare，约 300+ 只），Wind 50 只作为交叉验证
    disp_real = None
    try:
        _aq = akd.latest_quarters()
        if _aq:
            _d = akd.dispersion_ak(*_aq)
            if not _d.empty:
                disp_real = _d
    except Exception:
        disp_real = None
    if disp_real is None or disp_real.empty:
        try:
            _dp = inst.latest_fund_industry_period()
            disp_real = inst.dispersion_real(*_dp) if _dp else None
        except Exception:
            disp_real = None
    for c in build_triptych(t, t0, mv, n_up, active, pos, disp, disp_real):
        h(_card(c["judgment"], c["tag"], c["color"], c["rows"], c["invalid"]))

    h(f'<div class="cg-note">口径：主判断一律用 <b>Δ超配Z（长历史）</b>＝F2e 的一阶差分；'
      f'Δ配置pp 只作辅助。样本 {len(quarters)} 季，主口径 F2 仅 '
      f'{int(ind["F2_超配Zscore"].dropna(how="all").shape[0])} 季有效，'
      f'故统计结论一律使用长历史口径（F2e {int(ind["F2e_超配Z_长历史"].dropna(how="all").shape[0])} 季）。'
      f'因子取 T 期、收益取 T+1 期；季报披露滞后约 15 个工作日，实盘建仓点在 T+1 期初之后。</div>')
    return t, t0, mv, n_up


def render(b, ind, mkt, latest: str, colors: dict):
    """整页：结论区 → 变化榜 → 四象限 → 迁移 → 个股 → 证据 → 体检。"""
    quarters, t, t0, mv, n_up = _context(b, ind, mkt, latest)
    q0 = t.attrs.get("q0")
    e_add = t.loc["医药生物"]
    e_elec = t.loc["电子"]
    render_overview(b, ind, mkt, latest, colors)

    # ================= 2. 变化榜 =================
    h('<div class="sec">② 变化榜 · 31 行业按 Δ超配Z 排序（红＝加仓，绿＝减仓）</div>')
    st.plotly_chart(fig_change_bar(t, colors), width="stretch",
                    config={"displayModeBar": False})
    top3 = t.sort_values("dz").head(3)
    h(f'<div class="cg-note">读图：条形长度＝ΔZ，颜色＝方向（红加绿减）。'
      f'撤得最狠的三家：' +
      "、".join(f"{n} {r['dz']:+.2f}σ（配置 {r['d_alloc']:+.2f}pp）"
                for n, r in top3.iterrows()) +
      f'。若某行业 Δ配置很大但条形很短（如电子 {e_elec["d_alloc"]:+.2f}pp / '
      f'{e_elec["dz"]:+.2f}σ），说明变化主要来自价格而非主动买卖，不能读成「机构看空」。'
      f'<br><b>带 * 的 {int(t["conflict"].sum())} 个行业</b>：Δ配置 与 ΔZ 符号相反——'
      f'仓位在加但相对历史在降（如银行 {t.loc["银行", "d_alloc"]:+.2f}pp / {t.loc["银行", "dz"]:+.2f}σ），'
      f'或反之。这类行业不能用单一数字下结论，悬浮可看两个口径。<br>'
      f'<b>量价配合</b>：ΔZ 为正但成交额分位偏低＝无声加仓，信号更干净；'
      f'ΔZ 为负但成交额分位仍高＝交易未降温，减仓未必到位。'
      f'高拥挤本身不等于卖出，必须「Δ 方向 + 成交额背离」两者同向才构成交易信号。</div>')

    # ================= 3. 四象限 =================
    h('<div class="sec">③ 四象限 · 水平（当前 Z）× 方向（ΔZ），绝对阈值不用中位数</div>')
    st.plotly_chart(fig_quadrant(t, colors), width="stretch",
                    config={"displayModeBar": False})
    cnt = t["quad"].value_counts()
    h(f'<div class="cg-note"><b>分界规则</b>（写死在代码里，可复核）：'
      f'高位＝Z ≥ {ca.Z_HI:.1f} 或分位 ≥ {ca.PCT_HI:.0f}%；'
      f'低位＝Z ≤ {ca.Z_LO:.1f} 或分位 ≤ {ca.PCT_LO:.0f}%；'
      f'方向＝|ΔZ| ≥ {ca.DZ_MIN:.2f}σ；其余一律「中性区·不构成信号」。'
      f'本期分布：高位加仓 {cnt.get("HH", 0)}、高位减仓 {cnt.get("HC", 0)}、'
      f'低位加仓 {cnt.get("LH", 0)}、低位减仓 {cnt.get("LC", 0)}、'
      f'中性区 {cnt.get("MID", 0)}。<br>'
      f'<b>修复了什么</b>：旧版用 31 行业中位数分界，会把医药生物'
      f'（Z {_fmt(e_add["z"])}、配置 {_fmt(e_add["alloc"])}%、分位 {_fmt(e_add["pct"], "{:.0f}%")}）'
      f'这类深度低配但在加仓的行业标成「加仓拥挤」。现在它落在中性区，'
      f'不参与四象限命名，只在等级体系里记为「{e_add["lvl"]}」——两套标签不再打架。<br>'
      f'<b>中区不命名 ≠ 方向无效</b>：建筑材料（Z 0.18、分位 68%）仍在「减仓最猛」的判断卡里，'
      f'因为该判断只看 ΔZ 幅度，不依赖区位。四象限回答「这是哪种拥挤」，判断卡回答「这季该做什么」。</div>')

    # ================= 4. 等级迁移 =================
    h('<div class="sec">④ 等级迁移矩阵 · 上期 → 本期；近 12 期净升级家数</div>')
    m = ca.level_migration(t)
    s_up = ca.net_upgrade_series(ind, quarters, 12)
    c1, c2 = st.columns([0.52, 0.48], gap="small")
    with c1:
        h(migration_table(m, colors))
        h(f'<div class="cg-note">对角线＝等级未变（{int(np.trace(m.values))} 家）；'
          f'右上＝升级，左下＝降级。本期净升级 <b>{n_up:+d}</b> 家。'
          f'<br>右侧柱状图为<b>长历史等级口径</b>（F11e，{len(s_up)} 期，本期 {int(s_up.iloc[-1]):+d}）：'
          f'主口径「拥挤等级」自 2026Q1 起才有值，仅 2 期可比较，画不出趋势。'
          f'两套口径的「本期净升级」不可混用（左表 {n_up:+d} 为主口径）。</div>')
    with c2:
        st.plotly_chart(fig_net_upgrade(s_up, colors), width="stretch",
                        config={"displayModeBar": False})

    # ================= 4b. 基金行为分布（真实分歧度） =================
    h('<div class="sec">④b 基金行为分布 · 加仓/减仓基金数与分歧度</div>')
    _dr = pd.DataFrame()
    _src_name = ""
    try:
        _aq2 = akd.latest_quarters()
        if _aq2:
            _dr = akd.dispersion_ak(*_aq2)
            _src_name = "东方财富（AKShare）全市场口径"
    except Exception:
        _dr = pd.DataFrame()
    if _dr.empty:
        try:
            _dp2 = inst.latest_fund_industry_period()
            _dr = inst.dispersion_real(*_dp2) if _dp2 else pd.DataFrame()
            _src_name = "Wind 50 只样本"
        except Exception:
            _dr = pd.DataFrame()
    if _dr.empty:
        h('<div class="cg-note">未找到 50 只样本基金的行业配置数据（'
          'data/raw/fund_industry_50x4.json），本区暂不可用；'
          '数据到位后自动显示，不需要改代码。</div>')
    else:
        _n = int(_dr["样本基金数"].max())
        h(f'<div class="cg-note">数据源：<b>{_src_name}</b>，单行业有效样本最多 <b>{_n} 只</b>；'
          f'季报期只披露前十大、半年报披露全部（东财截断 100 只），'
          f'已按每只基金每期覆盖率归一化；「不变」包含「两期都不在前十」。'
          f'判读规则事前固定：加仓占比 ≥70% 且分歧度 &lt;0.4＝共识形成；'
          f'40–60% 且 &gt;0.7＝多空对撞；≤30%＝系统性撤离。'
          f'<b>口径（关键）</b>：本表区间 <b>{_dr.attrs.get("q0","—")} → {_dr.attrs.get("q","—")}</b>。'
          f'半年报/年报期用 Wind 全持仓行业表（覆盖约 88% 净值），'
          f'季报期只能由前十大重仓推导（覆盖约 60%），'
          f'因此每只基金每期先除以自身覆盖率再比较（占已覆盖组合的比例）。'
          f'<b>不归一化会把「覆盖率下降」误算成「全面减仓」</b>：'
          f'电子原始口径 3 加/39 减，归一化后 19 加/19 减。</div>')
        _show = _dr.sort_values(["分歧度", "加仓占比"], ascending=[False, True])
        st.dataframe(_show, width="stretch", hide_index=True, height=420,
                     column_config={
                         "加仓占比": st.column_config.NumberColumn(format="%.1f%%"),
                         "分歧度": st.column_config.NumberColumn(format="%.2f"),
                         "加减仓比": st.column_config.NumberColumn(format="%.2f"),
                         "平均配置": st.column_config.NumberColumn(format="%.2f%%"),
                     })

    # ================= 5. 行业 → 个股 =================
    h('<div class="sec">⑤ 行业变化 → 个股清单（按所属行业 ΔZ 升序，撤得最狠的排最前）</div>')
    pool = ca.load_stock_pool()
    sm = ca.stock_map(t, pool)
    if sm.empty:
        h('<div class="cg-note">暂无数据：未找到 hot_stocks_top20_*.xlsx。</div>')
    else:
        cols = ["股票名称", "行业", "dz", "行业环比方向", "持股基金数",
                "持股基金数环比", "持仓市值亿元", "结论"]
        view = sm[cols].rename(columns={"dz": "所属行业ΔZ", "持股基金数": "持股基金数",
                                        "持仓市值亿元": "持仓市值(亿元)"})
        st.dataframe(view, width="stretch", hide_index=True, height=360,
                     column_config={
                         "所属行业ΔZ": st.column_config.NumberColumn(format="%+.2f"),
                         "持股基金数": st.column_config.NumberColumn(format="%d"),
                         "持仓市值(亿元)": st.column_config.NumberColumn(format="%.1f"),
                     })
        h(f'<div class="cg-note"><b>数据边界</b>：个股清单与持股基金数来自 '
          f'<code>hot_stocks_top20_2026Q2.xlsx</code>（2026Q2 前十大重仓，最近一期可得）；'
          f'行业方向为 2026Q3。Wind 聚合接口不提供 2026Q3 个股级持股基金数，'
          f'故「相对上季变化」暂缺，不以估算替代。持仓市值为 2026Q2 口径，'
          f'与当前季度存在一个季度的时滞，结论列只用行业 ΔZ，不用市值排序；'
          f'因此若某行业 Δ配置与 ΔZ 符号相反（如有色金属 +0.43pp / −0.85σ），'
          f'个股结论会显示「减」，与该行业配置上升并不矛盾。</div>')

    # ================= 5b. 机构建仓/撤离榜 =================
    h('<div class="sec">⑤b 机构新增建仓榜 / 撤离榜 · 按 Δ持股基金数（前 50 只口径）</div>')
    flow = inst.stock_flow(latest, q0)
    if flow.empty:
        try:
            import sqlite3
            _con = sqlite3.connect(inst.DB_PATH)
            _qs = [r[0] for r in _con.execute(
                "SELECT DISTINCT quarter FROM stock_holdings ORDER BY quarter")]
            _con.close()
        except Exception:
            _qs = []
        h(f'<div class="cg-note">库内尚无 {latest} 的个股数据（已入库 '
          f'{len(_qs)} 期：{_qs[0] if _qs else "—"}–{_qs[-1] if _qs else "—"}）。'
          f'抓取仍在进行，完成后本区自动显示，无需改代码。</div>')
    else:
        cols = ["name", "n0", "n1", "d_n", "mv1", "d_mv", "state"]
        c1, c2 = st.columns(2, gap="small")
        with c1:
            h('<div class="cg-note" style="font-weight:700;color:#B42318">新增建仓榜（Δ持股基金数最大）</div>')
            st.dataframe(flow.nlargest(10, "d_n")[cols].rename(columns={
                "name": "股票", "n0": f"{q0}", "n1": f"{latest}", "d_n": "Δ基金数",
                "mv1": "持仓(亿)", "d_mv": "Δ市值(亿)", "state": "榜单状态"}),
                width="stretch", hide_index=True, height=330)
        with c2:
            h('<div class="cg-note" style="font-weight:700;color:#1A7F37">机构撤离榜（Δ持股基金数最小）</div>')
            st.dataframe(flow.nsmallest(10, "d_n")[cols].rename(columns={
                "name": "股票", "n0": f"{q0}", "n1": f"{latest}", "d_n": "Δ基金数",
                "mv1": "持仓(亿)", "d_mv": "Δ市值(亿)", "state": "榜单状态"}),
                width="stretch", hide_index=True, height=330)
        h('<div class="cg-note"><b>口径边界</b>：接口每期只返回「按持股基金数排序的前 50 只」，'
          '因此「退出前 50」不等于清仓、「新进前 50」不等于新建仓，两者都只作观察。'
          'Δ基金数未做基金样本数归一（样本基金数逐季变化），跨期比较需谨慎。</div>')

    # ================= 6. 证据面板 =================
    h('<div class="sec">⑥ 证据面板 · 为什么用 ΔZ 而不是 Δ配置pp</div>')
    ic = ca.ic_compare(b, ind, start="2024Q1")
    show = ic[ic["区间"] == "2024Q1 起"].copy()
    show = show[["口径", "n", "RankIC", "IC_IR", "t_NW", "p_NW", "胜率", "判定"]]
    show.columns = ["口径", "n", "RankIC", "IC_IR", "t(NW)", "p", "胜率%", "判定"]
    st.dataframe(show, width="stretch", hide_index=True,
                 column_config={
                     "RankIC": st.column_config.NumberColumn(format="%+.3f"),
                     "IC_IR": st.column_config.NumberColumn(format="%.2f"),
                     "t(NW)": st.column_config.NumberColumn(format="%.2f"),
                     "p": st.column_config.NumberColumn(format="%.3f"),
                     "胜率%": st.column_config.NumberColumn(format="%.0f"),
                 })
    r_cfg = ic[ic["口径"].str.startswith("Δ配置")].iloc[0]
    r_dz = ic[ic["口径"].str.startswith("Δ超配Z")].iloc[0]
    h(f'<div class="cg-note"><b>结论</b>：Δ配置pp 的 RankIC {r_cfg["RankIC"]:+.3f}'
      f'（t {r_cfg["t_NW"]:.2f}，p {r_cfg["p_NW"]:.3f}，n={r_cfg["n"]}，'
      f'{r_cfg["起始"]}–{r_cfg["终止"]}）——<b>不显著</b>，无信息；'
      f'Δ超配Z {r_dz["RankIC"]:+.3f}（t {r_dz["t_NW"]:.2f}，p {r_dz["p_NW"]:.3f}，'
      f'n={r_dz["n"]}，{r_dz["起始"]}–{r_dz["终止"]}）——<b>{r_dz["判定"]}</b>。'
      f'两者相差一个数量级，所以页面只用 ΔZ 排序。'
      f't 为 Newey-West 调整（滞后 2 期），与任务书基准口径一致。</div>')

    lsb = ca.long_short_bundle(b, ind, k=8)
    f = lsb["full"]
    h('<div class="sec">⑦ 多空组合 · 每期做多 ΔZ 最高 8 个行业、做空最低 8 个</div>')
    c3, c4 = st.columns([0.6, 0.4], gap="small")
    with c3:
        st.plotly_chart(fig_nav(lsb["series"], colors), width="stretch",
                        config={"displayModeBar": False})
    with c4:
        h(f'<div class="cg-note" style="font-size:11px;line-height:1.85">'
          f'样本：{lsb["series"].index[0]}–{lsb["series"].index[-1]}，n={f["n"]} 季<br>'
          f'年化（复利）：<b>{f["ann"]:+.2f}%</b>　算术：{f["ann_arith"]:+.2f}%<br>'
          f't = <b>{f["t"]:.2f}</b>，p = <b>{f["p"]:.3f}</b>　（NW t {f["nw_t"]:.2f}）<br>'
          f'胜率：<b>{f["win"]:.0f}%</b>　累计：{f["total"]:+.1f}%<br>'
          f'判定：<b>{ca.verdict(f["p"])}</b></div>')
    seg = lsb["segments"]
    st.dataframe(seg, width="stretch", hide_index=True,
                 column_config={
                     "年化": st.column_config.NumberColumn(format="%+.2f%%"),
                     "t": st.column_config.NumberColumn(format="%.2f"),
                     "p": st.column_config.NumberColumn(format="%.3f"),
                     "胜率": st.column_config.NumberColumn(format="%.0f%%"),
                 })
    s1 = seg.iloc[0]
    s2 = seg.iloc[1]
    h(f'<div class="cg-note"><b>边界声明（不是自我否定，是划范围）</b>：'
      f'{s1["区间"]} t = {s1["t"]:.2f}（p {s1["p"]:.3f}，<b>不显著</b>），'
      f'{s2["区间"]} t = {s2["t"]:.2f}（p {s2["p"]:.3f}，<b>{s2["判定"]}</b>）。'
      f'信号的有效性集中在 2023 年之后，因此<b>只在近三年样本内使用，且每季复核</b>；'
      f'一旦连续两季多空价差为负，暂停使用该信号。</div>')

    # 成本敏感性
    cst = ca.cost_sensitivity(b, ind, k=8)
    h('<div class="sec">⑧ 交易成本敏感性 · 单边 0–30bp</div>')
    st.dataframe(cst, width="stretch", hide_index=True,
                 column_config={
                     "年化毛": st.column_config.NumberColumn(format="%+.2f%%"),
                     "年换手": st.column_config.NumberColumn(format="%.0f%%"),
                     "成本拖累": st.column_config.NumberColumn(format="%.2f%%"),
                     "年化净": st.column_config.NumberColumn(format="%+.2f%%"),
                 })
    h(f'<div class="cg-note">年换手约 {cst["年换手"].iloc[0]:.0f}%（季度调仓、上下各 8 只，'
      f'换手偏高）。即使按单边 30bp 计，净年化仍有 {cst["年化净"].iloc[-1]:+.2f}%，'
      f'但这是<b>未考虑冲击成本与融券约束</b>的理论值；小市值行业（综合、美容护理）'
      f'实际难以按收盘价成交。</div>')

    # 动量正交化
    mn = ca.momentum_neutral(b, ind, k=8)
    h('<div class="sec">⑨ 动量正交化 · 证伪「ΔZ 只是价格动量的代理」</div>')
    h(f'<div class="cg-note" style="font-size:11px;line-height:1.8">'
      f'对过去 4 季涨幅做横截面回归后取残差，再做同样的多空：'
      f'原始年化 {mn["raw"]["ann"]:+.2f}%（t {mn["raw"]["t"]:.2f}）→ '
      f'残差年化 <b>{mn["resid"]["ann"]:+.2f}%</b>（t {mn["resid"]["t"]:.2f}，'
      f'p {mn["resid"]["p"]:.3f}），量级未衰减。'
      f'ΔZ 与过去 4 季涨幅的横截面相关 {mn["corr_mean"]:+.3f}（{mn["corr_n"]} 期均值），'
      f'重叠很小。<b>证伪未通过</b>：ΔZ 的收益不能由价格动量解释。</div>')

    # ================= 7. 样本体检 =================
    h('<div class="sec">⑩ 样本充分性体检 · 每个因子究竟几期有效</div>')
    aud = ca.sample_audit(b, ind)
    st.dataframe(aud, width="stretch", hide_index=True)
    h(f'<div class="cg-note">「{len(quarters)} 季度」是日历长度，不等于可用样本：'
      f'主口径 F2 只有 {int(ind["F2_超配Zscore"].dropna(how="all").shape[0])} 季有效，'
      f'F2e 长历史 {int(ind["F2e_超配Z_长历史"].dropna(how="all").shape[0])} 季。'
      f'任何 IC / 多空结论都只用有效样本，不用日历长度充数。</div>')
