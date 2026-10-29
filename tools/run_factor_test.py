# -*- coding: utf-8 -*-
"""把因子检验结果算好存成 JSON，供页面秒开。

为什么不在页面里现算
--------------------
全流程（9 个因子 × 31 期 × 31 行业，含价格中性化）约 80 秒。
Streamlit 每次冷启动都要等这么久，体验不可接受。所以：
    · 本脚本离线算一次，写 data/derived/factor_test.json；
    · 页面直接读 JSON（<10ms），并显示生成时间与样本期；
    · 数据更新后重跑本脚本；页面上也有「重算」按钮可临时现算。

用法：python tools/run_factor_test.py
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

import canon as cn          # noqa: E402
import factor_test as ft    # noqa: E402


def build() -> dict:
    t0 = time.time()
    tab = ft.run_all()
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "elapsed_s": round(time.time() - t0, 1),
        "sample": {
            "quarters": len(cn.quarters()),
            "q0": cn.quarters()[0], "q1": cn.quarters()[-1],
            "industries": len(cn.INDUSTRIES) if hasattr(cn, "INDUSTRIES") else 31,
            "n_funds": len(cn.universe(cn.latest())),
        },
        "thresholds": ft.TH,
        "factors": [],
    }
    for k, meta in ft.FACTORS.items():
        r = ft.test_factor(k)
        lab, why = ft.verdict(r)
        d = dict(key=k, name=meta["name"], direction=meta["direction"],
                 freq=meta["freq"], formula=meta["formula"], expect=meta["expect"],
                 verdict=lab, why=why,
                 ic_mean=r.get("ic_mean"), ic_ir=r.get("ic_ir"),
                 t=r.get("t"), p=r.get("p"), win=r.get("win"), n=r.get("n"),
                 ls_mean=r.get("ls_mean"), ls_win=r.get("ls_win"),
                 ann=r.get("ann"), mono=r.get("mono"))
        if r.get("ok"):
            d["ic_series"] = {str(i): (None if pd_isna(v) else round(float(v), 4))
                              for i, v in r["ic"].items()}
            d["layers"] = {str(i): round(float(v), 3) for i, v in r["layers"].items()}
        out["factors"].append(d)
    return out


def pd_isna(v) -> bool:
    try:
        return v != v
    except Exception:
        return True


def main() -> int:
    out = build()
    p = os.path.join(ROOT, "data", "derived", "factor_test.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"写入 {p}　耗时 {out['elapsed_s']}s")
    for f in out["factors"]:
        print(f"  {f['name']:<8} IC={f['ic_mean']:+.3f} t={f['t']:+.2f} "
              f"p={f['p']:.3f} n={f['n']} → {f['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
