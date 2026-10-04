# -*- coding: utf-8 -*-
"""抓取 9 个页面的可见文本，定位异常字符（如单独出现的 风/嘻）。"""
import sys, time, re
sys.path.insert(0, "src")
from playwright.sync_api import sync_playwright

PAGES = ["① 总览仪表盘", "② 拥挤度总览", "③ 行业占比变迁", "④ ETF 资金流",
         "⑤ 因子研究", "⑥ 轮动策略与因子裁决", "⑦ 稳健性检验",
         "⑧ 数据质量与溯源", "⑨ 方法说明与数据来源"]

out = open("output/_pagetext.txt", "w", encoding="utf-8")
errs = []
with sync_playwright() as pw:
    br = pw.chromium.launch(headless=True, args=["--no-sandbox"])
    pg = br.new_page(viewport={"width": 1600, "height": 1000})
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8509", wait_until="commit", timeout=90000)
    for _ in range(60):
        try:
            if pg.get_by_text("综合判断", exact=False).count() > 0:
                break
        except Exception:
            pass
        pg.wait_for_timeout(4000)
    time.sleep(2)
    for i, name in enumerate(PAGES):
        try:
            pg.get_by_text(name, exact=False).first.click(timeout=20000)
            time.sleep(6 if i else 3)
            txt = pg.inner_text("body")
            out.write(f"\n{'='*30} PAGE {name} {'='*30}\n")
            out.write(txt)
            # 找孤立的单字
            sus = re.findall(r"[\u4e00-\u9fff]{1}(?=[，。、；：\s\n「」（）]|$)", txt)
            print(f"[{i}] {name}  len={len(txt)}")
        except Exception as e:
            print(f"[{i}] {name} FAIL {type(e).__name__}")
    br.close()
out.close()
print("console errors:", len(errs))
full = open("output/_pagetext.txt", encoding="utf-8").read()
print("嘻 count:", full.count("嘻"))
# 打印嘻上下文
for m in re.finditer("嘻", full):
    print("CTX:", full[max(0, m.start()-60):m.start()+60].replace("\n", "⏎"))
