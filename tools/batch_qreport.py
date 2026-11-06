# -*- coding: utf-8 -*-
"""批量预抓季报数据（资产配置 / 行业分类 / 份额 / 持仓 / 买卖 / 债券）。

为什么需要
----------
④ 单只基金·季报全解是「按需抓取 + 20 小时缓存」：第一次点开某只基金要 3–6 秒。
把规模靠前的基金先跑一遍，常用基金就能秒开；同时给审计提供足够样本。

用法
----
    python tools/batch_qreport.py --n 80          # 按规模取前 80 只
    python tools/batch_qreport.py --all           # 全部（约 5 小时，可断点续跑）
    python tools/batch_qreport.py --n 80 --force # 忽略 20 小时缓存

断点续跑：已缓存的基金会跳过（除非 --force），中断后重跑即可。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import qreport as Q  # noqa: E402


def _codes(n: int | None) -> list[str]:
    import canon as cn
    p = cn.profiles()
    if p is None or p.empty:
        return []
    p = p.copy()
    p["code"] = p["code"].astype(str).str.zfill(6)
    p = p.sort_values("scale_yi", ascending=False, na_position="last")
    codes = p["code"].drop_duplicates().tolist()
    return codes if n is None else codes[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only-failed", action="store_true",
                    help="只重抓 q_log 里 ok=0 的基金")
    ap.add_argument("--shard", type=int, default=0, help="分片序号（从 0 开始）")
    ap.add_argument("--of", type=int, default=1, help="分片总数")
    args = ap.parse_args()

    if args.only_failed:
        with Q._con() as con:
            codes = [r[0] for r in con.execute(
                "SELECT code FROM q_log WHERE ok=0 ORDER BY code")]
        print(f"待重抓（上次有接口失败）：{len(codes)} 只", flush=True)
    else:
        codes = _codes(None if args.all else args.n)
    if args.of > 1:
        # 多进程并行抓取：每个进程只处理自己那一片，避免重复请求。
        codes = [c for i, c in enumerate(codes) if i % args.of == args.shard]
    print(f"目标 {len(codes)} 只基金（分片 {args.shard}/{args.of}）", flush=True)
    ok = skip = fail = 0
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        try:
            d = Q.fetch_fund(code, force=args.force)
            if d.get("cached"):
                skip += 1
            else:
                ok += 1
            if i % 10 == 0 or i == len(codes):
                print(f"  {i}/{len(codes)}  新增 {ok}｜跳过 {skip}｜失败 {fail}"
                      f"｜{time.time() - t0:.0f}s", flush=True)
        except KeyboardInterrupt:
            print("中断，已保存", ok)
            return 0
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  {code} 失败：{type(e).__name__}: {str(e)[:80]}", flush=True)
    cov = Q.coverage()
    print(f"完成：新增 {ok}、跳过 {skip}、失败 {fail}；"
          f"库内 {cov['funds']} 只、行业 {cov['industry_rows']} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
