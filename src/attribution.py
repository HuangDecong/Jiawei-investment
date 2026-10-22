# -*- coding: utf-8 -*-
"""
attribution.py —— 配置变化归因与持仓压力测算
================================================================================
为什么需要它
------------
「本季电子配置从 22.13% 升到 43.24%」这句话本身没有信息量：它既可能是基金经理
主动加仓，也可能只是重仓股涨出来的（被动漂移）。两者的投资含义完全相反——

  * 主动加仓 +18pp：新钱进场，共识在扩张，后续需要更多增量资金维持；
  * 被动漂移 +18pp：没人买，只是价格涨了，一旦回调占比自动回落。

本节把 Δw 拆成两项，公式与口径如下（Grinold & Kahn 的权重漂移分解，逐季精确）：

    w_t = w_{t-1} · (1 + r_t) / Σ_j [ w_{t-1,j} · (1 + r_{t,j}) ]
    Δw  = (w_t − w_{t-1}) = (drift_t − w_{t-1}) + (w_t − drift_t)
          └──── 被动漂移 ────┘   └──── 主动加减仓 ────┘

口径声明（不可省略）
--------------------
* r_t 用**申万一级行业指数季度涨跌幅**近似「该行业持仓的市值涨幅」。
  真实持仓是个股组合，与行业指数有跟踪差异（个股选择、权重偏离），
  因此分解结果是**方向性归因**，不是精确会计；偏差主要落在「主动」项上。
* 分母按 31 个行业同时归一，保证 Σw_t = 100 与 ΣΔw = 0（可加性校验通过）。
* 只在相邻两期都有持仓数据的季度上计算（主口径 2024Q1 起），不做跨缺口插值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def decompose(w: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    """逐季分解配置变化。返回长表，索引为 (季度, 行业)。"""
    qs = [q for q in w.index if q in ret.index]
    rows = []
    for i in range(1, len(qs)):
        q0, q1 = qs[i - 1], qs[i]
        w0 = w.loc[q0].astype(float)
        w1 = w.loc[q1].astype(float)
        r = ret.loc[q1].reindex(w0.index).astype(float) / 100.0
        ok = w0.notna() & w1.notna() & r.notna()
        if ok.sum() < 20:
            continue
        g = w0[ok] * (1 + r[ok])
        tot = g.sum()
        if not np.isfinite(tot) or tot <= 0:
            continue
        drift = g / tot * 100.0
        # 未参与归一的行（缺收益）保持 NaN，避免伪装成 0
        full = pd.Series(np.nan, index=w0.index)
        full[ok] = drift
        rows.append(pd.DataFrame({
            "季度": q1, "行业": w0.index,
            "上季配置": w0.values, "本季配置": w1.values,
            "被动漂移后": full.values,
            "变化": (w1 - w0).values,
            "被动贡献": (full - w0).values,
            "主动贡献": (w1 - full).values,
        }))
    if not rows:
        return pd.DataFrame(columns=["季度", "行业", "上季配置", "本季配置",
                                     "被动漂移后", "变化", "被动贡献", "主动贡献"])
    out = pd.concat(rows, ignore_index=True)
    return out.set_index(["季度", "行业"])


def split_one(dec: pd.DataFrame, q: str, ind: str) -> dict | None:
    """取某季某行业的分解结果；缺数据返回 None。"""
    try:
        r = dec.loc[(q, ind)]
    except KeyError:
        return None
    d = {k: (float(r[k]) if pd.notna(r[k]) else np.nan)
         for k in ("上季配置", "本季配置", "被动漂移后", "变化", "被动贡献", "主动贡献")}
    d["ok"] = bool(pd.notna(d["变化"]))
    return d


def verdict(d: dict) -> dict:
    """把分解结果翻译成一句可判断的结论。

    判定规则（事前固定，不随数据调参）：
      |主动| ≥ 2×|被动| 且 |主动| ≥ 1.0pp  → 主动主导
      |被动| ≥ 2×|主动| 且 |被动| ≥ 1.0pp  → 被动主导（涨跌出来的）
      其余                                  → 混合
    """
    a, p = d.get("主动贡献", np.nan), d.get("被动贡献", np.nan)
    if not np.isfinite(a) or not np.isfinite(p):
        return dict(tag="无法分解", text="缺收益或持仓数据。", cls="")
    if abs(a) >= 2 * abs(p) and abs(a) >= 1.0:
        return dict(tag="主动主导", cls="up" if a > 0 else "dn",
                    text=("配置变化主要由<b>主动加减仓</b>驱动："
                          f"主动{'加仓' if a > 0 else '减仓'} {abs(a):.2f} pct，"
                          f"价格漂移仅 {p:+.2f} pct。这是真实的新增/撤出资金，"
                          "会改变该行业的边际供需。"))
    if abs(p) >= 2 * abs(a) and abs(p) >= 1.0:
        return dict(tag="被动主导", cls="",
                    text=("配置变化主要由<b>价格漂移</b>驱动："
                          f"被动 {p:+.2f} pct，主动仅 {a:+.2f} pct。"
                          "占比上升不等于加仓——回调时这部分会同样自动回落。"))
    return dict(tag="混合驱动", cls="",
                text=(f"主动 {a:+.2f} pct 与被动 {p:+.2f} pct 量级相当，"
                      "需要结合成交与筹码判断哪一方在主导。"))


def pressure(over_now: float, over_target: float, alloc_now: float,
             total_yi: float | None = None,
             turnover_yi: float | None = None,
             days: int = 61) -> dict:
    """持仓压力测算：超配回归目标水平时，需要卖出的规模与流动性含义。

    输出全部为**可直接引用的量**，并显式标注口径：
      need_pct_industry : 需卖出占该行业当前持仓的百分比（口径无关，最稳健）
      need_pct_total    : 需卖出占全部重仓市值的百分比（＝超配回落幅度）
      need_yi           : 折算金额（仅当传入 total_yi）
      days_of_turnover  : 相当于该行业多少个交易日的成交额（仅当传入 turnover_yi）
    """
    d_over = float(over_now) - float(over_target)
    out = dict(d_over=d_over, need_pct_industry=np.nan, need_pct_total=d_over,
               need_yi=np.nan, days_of_turnover=np.nan)
    if not np.isfinite(d_over) or d_over <= 0:
        return out
    if np.isfinite(alloc_now) and alloc_now > 0:
        out["need_pct_industry"] = d_over / alloc_now * 100.0
    if total_yi and np.isfinite(total_yi) and total_yi > 0:
        out["need_yi"] = d_over / 100.0 * total_yi
        if turnover_yi and np.isfinite(turnover_yi) and turnover_yi > 0 and days > 0:
            out["days_of_turnover"] = out["need_yi"] / (turnover_yi / days)
    return out
