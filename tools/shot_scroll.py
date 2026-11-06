import os
import sys

from playwright.sync_api import sync_playwright

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots")
EXE = r"C:\Users\21204\AppData\Local\ms-playwright\chromium-1243\chrome-win64\chrome.exe"
os.makedirs(OUT, exist_ok=True)

page_name = sys.argv[1] if len(sys.argv) > 1 else "④ 单只基金"
tag = sys.argv[2] if len(sys.argv) > 2 else "fund"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
switch = sys.argv[4] if len(sys.argv) > 4 else ""

with sync_playwright() as pw:
    br = pw.chromium.launch(headless=True, executable_path=EXE,
                            args=["--no-sandbox", "--disable-gpu"])
    pg = br.new_page(viewport={"width": 1500, "height": 1100})
    pg.goto("http://127.0.0.1:8509", wait_until="commit", timeout=90000)
    pg.wait_for_timeout(15000)
    pg.get_by_text(page_name, exact=False).first.click(timeout=15000)
    pg.wait_for_timeout(15000)
    if switch:
        pg.get_by_text(switch, exact=False).first.click(timeout=15000)
        pg.wait_for_timeout(15000)
    pg.screenshot(path=os.path.join(OUT, f"{tag}_00.png"))
    for i in range(1, n):
        pg.mouse.move(760, 600)
        pg.mouse.wheel(0, 900)
        pg.wait_for_timeout(1300)
        pg.screenshot(path=os.path.join(OUT, f"{tag}_{i:02d}.png"))
    br.close()
print("ok", n)
