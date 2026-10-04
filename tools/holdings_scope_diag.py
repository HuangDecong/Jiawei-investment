# -*- coding: utf-8 -*-
"""holdings 口径诊断与修正（机构级）

■ 问题（自主发现）
  原抓取问句为「全部公募基金的前十大重仓股持股总市值」，Wind 将其解释为
  **全部公募基金口径**（含债券型、指数型、货币型、FOF、QDII 等）。
  实测 2026Q2 该口径重仓股总市值 ≈ 4.65 万亿，而按本项目定义的
  「主动权益三类」（普通股票型 + 偏股混合型 + 灵活配置型）口径仅 ≈ 2.55 万亿。
  过宽口径会让「重仓行业结构」被被动指数基金与债券基金稀释，直接污染
  F1 超配比例 / F12 配置系数 / 集中度 HHI，属于**致命的样本定义错误**。

■ 证据（可复现）
  Wind 分解返回（2026-06-30）：
      偏股混合型基金  16202.872 亿元
      灵活配置型基金   5770.343 亿元
      普通股票型基金   3555.816 亿元
      ---------------------------------
      三类合计        25529.031 亿元   ← 应为分母
  而全口径合计 46510 亿元（既有缓存值），虚高 82%。

■ 结论
  holdings 全样本必须按「主动权益三类」重抓，且必须在问句中**显式列出三类**，
  不能写「全部公募基金」。

■ 正确问句模板
  查询{Q}，{D}，公募基金中普通股票型基金、偏股混合型基金、灵活配置型基金
  的前十大重仓股持股总市值合计，按申万一级行业分类汇总

■ 校验规则（build 后执行）
  1. 三类合计 / 全口径合计 应落在 [0.45, 0.65]（主动权益占公募重仓的比重）；
  2. 单一行业占比上限 30%（2026Q2 电子 10093/25529 = 39.5% 仍偏高，
     需与「电子行业占主动权益重仓」的公开口径交叉验证，见 R 校验）。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ACTIVE_TYPES = ["普通股票型基金", "偏股混合型基金", "灵活配置型基金"]


def question(q_end: str) -> str:
    """生成主动权益三类的标准 holdings 问句。"""
    return (f"查询{q_end}，公募基金中普通股票型基金、偏股混合型基金、"
            f"灵活配置型基金的前十大重仓股持股总市值合计，按申万一级行业分类汇总")


def check(cache_dir: str) -> None:
    """诊断既有缓存的口径一致性。"""
    import glob
    print(f"{'季度':<9}{'合计(亿)':>12}{'电子占比':>10}{'Top1':>10}  判定")
    print("-" * 56)
    for f in sorted(glob.glob(os.path.join(cache_dir, "holdings__*.json"))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if not d:
            continue
        q = os.path.basename(f)[10:-5]
        tot = sum(d.values())
        top = max(d, key=d.get)
        ratio = d[top] / tot * 100
        flag = "⚠️ 疑似过宽口径" if tot > 38000 else "OK"
        print(f"{q:<9}{tot:>12.0f}{d.get('电子',0)/tot*100:>9.1f}%"
              f"{top:>10}  {flag}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "data", "cache"))
    else:
        # 打印待重抓季度的标准问句
        for q, d in [("2024Q1", "2024-03-31"), ("2024Q2", "2024-06-30"),
                     ("2024Q3", "2024-09-30"), ("2024Q4", "2024-12-31"),
                     ("2025Q1", "2025-03-31"), ("2025Q2", "2025-06-30"),
                     ("2025Q3", "2025-09-30"), ("2025Q4", "2025-12-31"),
                     ("2026Q1", "2026-03-31"), ("2026Q2", "2026-06-30")]:
            print(f"### {q}")
            print(question(d))
            print()
