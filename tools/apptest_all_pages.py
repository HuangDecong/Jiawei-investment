# -*- coding: utf-8 -*-
"""AppTest 逐页异常检测：覆盖 9 个页面，输出每页 exception 数与详情。"""
import os

from streamlit.testing.v1 import AppTest

# 主入口已统一为项目根目录 app.py（部署口径）；用 __file__ 相对定位，避免硬编码
APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.py")

PAGES = ["① 总览仪表盘", "② 拥挤度总览", "③ 行业占比变迁", "④ ETF 资金流",
         "⑤ 因子研究", "⑥ 轮动策略与因子裁决", "⑦ 稳健性检验",
         "⑧ 数据质量与溯源", "⑨ 方法说明与数据来源"]

total_err = 0
for p in PAGES:
    at = AppTest.from_file(APP, default_timeout=1200)
    at.run()
    at.radio(key="page").set_value(p)
    at.run()
    n = len(at.exception)
    total_err += n
    print(f"{p}  exceptions={n}")
    for e in at.exception:
        print("   ERR:", str(e.value)[:250].replace("\n", " | "))
print("TOTAL exceptions =", total_err)
