# -*- coding: utf-8 -*-
"""本地预览启动器：绕开 Windows 上 uvicorn 的已知崩溃。

现象
----
浏览器标签页被强制关闭（或 Playwright 断开）时，uvicorn 的
`_ProactorBasePipeTransport._call_connection_lost` 在回调里抛
`ConnectionResetError: [WinError 10054]`，异常发生在 asyncio 回调栈里，
整个 Streamlit 进程直接退出（本地演示被打断，线上 Linux 不受影响）。

做法
----
Windows 的 Proactor 事件循环有这个问题，Selector 没有。
必须在 import streamlit **之前**设置策略——服务端的 loop 是在
`cli.main()` 里创建的，进了 app.py 再设已经晚了。

用法
----
    python tools/serve_local.py [端口]      # 默认 8509

线上（Streamlit Cloud）仍用 `streamlit run app.py`，不经过这个包装。
"""
from __future__ import annotations

import asyncio
import os
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from streamlit.web import cli as stcli  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = sys.argv[1] if len(sys.argv) > 1 else "8509"

sys.argv = [
    "streamlit", "run", os.path.join(ROOT, "app.py"),
    "--server.port", PORT,
    "--server.headless", "true",
    "--server.fileWatcherType", "none",
]
sys.exit(stcli.main())
