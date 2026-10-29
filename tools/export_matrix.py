# -*- coding: utf-8 -*-
"""把「行业 × 期次」的配置矩阵预先算好，导出成一个小文件。

为什么需要
----------
`level_matrix()` 要遍历 31 个季度，每期跑一次「前十大」SQL 并构造成
4000×31 的矩阵，首次调用约 22 秒。这个矩阵是首页、拥挤度页、单只基金页
都要用的，于是**服务重启后的第一次访问会白屏 20 秒以上**（Streamlit Cloud
休眠唤醒后尤其明显）。

矩阵本身是纯派生数据：只由 fund_holdings 决定，不含任何随机或外部输入。
所以把它预计算成 CSV，运行时直接读（约 0.05 秒）。为保证不过期，
`canon.level_matrix()` 会校验两件事，任一不满足就回退现场计算：

    1) 文件里的期次列表与数据库当前期次完全一致；
    2) 文件比数据库新（导出后没有重新抓数）。

用法
----
    python tools/export_matrix.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import canon as cn  # noqa: E402

OUT = os.path.join(ROOT, "data", "derived", "canon_level_top10.csv.gz")
OUT_POS = os.path.join(ROOT, "data", "derived", "canon_position.csv.gz")


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    m = cn.level_matrix("top10")
    if m.empty:
        print("矩阵为空，未导出")
        return
    m.to_csv(OUT, index=True, compression="gzip", encoding="utf-8")
    print(f"canon_level_top10.csv.gz  {m.shape[0]} 期 × {m.shape[1]} 行业  "
          f"{os.path.getsize(OUT) / 1e3:.1f} KB")
    print("期次：", m.index[0], "→", m.index[-1])

    # 仓位分布：15 行 × 4 列。现场算每期要一次全表聚合，合计 12 秒，
    # 是首页首屏的主要耗时；预计算后读文件不到 1 毫秒。
    ps = cn.position_series()
    if not ps.empty:
        ps.to_csv(OUT_POS, index=True, compression="gzip", encoding="utf-8")
        print(f"canon_position.csv.gz     {ps.shape[0]} 期 × {ps.shape[1]} 列  "
              f"{os.path.getsize(OUT_POS) / 1e3:.1f} KB")
        print("全持仓期：", ps.index[0], "→", ps.index[-1])


if __name__ == "__main__":
    main()
