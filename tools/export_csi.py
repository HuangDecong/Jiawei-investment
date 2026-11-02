# -*- coding: utf-8 -*-
"""把 31 期的 CSI（拥挤度指数）面板预计算出来。

为什么需要
----------
画「拥挤度变化折线」和做「阈值 → 次季收益」的实证，都需要**每一期**的 CSI。
现场算一期要 2–4 秒（资金分位依赖逐期加仓广度），31 期就是 1–2 分钟，
页面每次打开都等这么久不可接受。CSI 是纯派生数据（只由持仓 + 指数决定），
所以预计算成 31×31 的小矩阵（约 10KB），运行时直接读。

口径
----
三个分位全部是**扩张窗口**（只用当期及以前的数据），最后一期与全样本相等：

    配置分位 = 该行业当期配置在 [2019Q1..q] 中的排名
    价格分位 = 该行业指数季末点位在 [2019-01-01..季末日] 中的排名
    资金分位 = 当期加仓占比在 31 个行业之间的排名（横截面，本身无前视）

用法
----
    python tools/export_csi.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import canon as cn  # noqa: E402

OUT = os.path.join(ROOT, "data", "derived", "canon_csi_history.csv.gz")


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    qs = list(cn.quarters())
    rows = {}
    for q in qs:
        s = cn.crowding_score(q)
        if s.empty:
            continue
        rows[q] = s["CSI"]
    import pandas as pd  # noqa: E402

    d = pd.DataFrame(rows).T
    d = d.reindex(index=qs, columns=list(cn.C.INDUSTRIES))
    if d.dropna(how="all").empty:
        print("CSI 面板为空，未导出")
        return
    d.to_csv(OUT, index=True, compression="gzip", encoding="utf-8")
    ok = int(d.notna().sum().sum())
    print(f"canon_csi_history.csv.gz  {d.shape[0]} 期 × {d.shape[1]} 行业  "
          f"有效 {ok} 格  {os.path.getsize(OUT) / 1e3:.1f} KB")
    print("首期有效：", d.dropna(how="all").index[0], " 末期：", d.index[-1])
    print("末期前三：", d.iloc[-1].nlargest(3).round(1).to_dict())


if __name__ == "__main__":
    main()
