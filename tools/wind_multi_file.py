# -*- coding: utf-8 -*-
"""把 Wind 多季度时序返回落盘到 cache（供 wind_multi 解析）。

由于 Wind 返回体较大，Agent 分批查询（每批约 4 年），
每批结果存为一个 json，本脚本逐批解析。

用法：python tools/wind_multi_file.py <task> <payload.json>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import wind_multi as wm  # noqa: E402

if __name__ == "__main__":
    wm.main(sys.argv[1], sys.argv[2])
