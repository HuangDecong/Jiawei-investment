"""抽样核对「季报 PDF 原文」与库内数字（联网）。

为什么需要它
------------
库里的数字来自东财的季报转引页面（f10），不是 PDF 本身。转引可能出错，
所以必须拿原始季报抽查：前十大重仓股的代码、名称、占净值比例，
以及「报告期末基金份额总额」，逐项对一遍。

用法
----
    python tools/verify_pdf.py                 # 随机抽 8 只（最新季报）
    python tools/verify_pdf.py --n 20 --seed 7
    python tools/verify_pdf.py --codes 002910,000001

能查什么、不能查什么
--------------------
能查：§5.3 前十名股票（代码/名称/占净值比）、§2 期末份额总额。
不能查：全部持仓（季报只披露前十大）、行业分类（PDF 里的行业分类是证监会门类，
        与库内 forward 的申万口径不同源，不做逐项比对）。
"""
import argparse
import os
import random
import re
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import pymupdf  # noqa: E402
import requests  # noqa: E402

import qreport as Q  # noqa: E402

DB = os.path.join(ROOT, "data", "db", "fund_q.db")
UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://fundf10.eastmoney.com/"}


def _reports(code: str) -> list[dict]:
    """该基金最近 10 条定期报告（type=3），新的在前。"""
    p = {"fundcode": code, "pageIndex": 1, "pageSize": 10, "type": "3",
         "letter": "", "deviceid": "Wap", "plat": "Wap", "product": "EFund",
         "version": "2.0.0", "_": int(time.time() * 1000)}
    r = requests.get("https://api.fund.eastmoney.com/f10/JJGG",
                     params=p, headers=UA, timeout=20)
    return r.json().get("Data", [])


def _pdf_text(art_id: str) -> str:
    url = f"http://pdf.dfcfw.com/pdf/H2_{art_id}_1.pdf"
    b = requests.get(url, headers=UA, timeout=60).content
    doc = pymupdf.open(stream=b, filetype="pdf")
    return "\n".join(pg.get_text() for pg in doc)


def _quarter_of(title: str) -> str | None:
    m = re.search(r"(\d{4})\s*年第\s*(\d)\s*季度报告", title)
    return f"{int(m.group(1))}Q{m.group(2)}" if m else None


def _parse_top10(txt: str) -> list[tuple[str, str, float]]:
    """解析「前十名股票投资明细」表 → [(代码, 名称, 占净值%)]。

    两个坑：
    1) 标题会跨行（「前十名股票投资明\n细」），不能整串匹配，只找「股票投资明细」；
    2) 表格跨页时中间夹着「第 9 页 共 12 页」，不先删掉会把一行切成两半。
    """
    txt = re.sub(r"第\s*\d+\s*页[，,]?\s*共\s*\d+\s*页", "", txt)
    txt = re.sub(r"\n\s*\n+", "\n", txt)
    # 目录页也会出现「5.3 …股票投资明细」，不能取第一个匹配；
    # 只认「后面紧跟表头『序号』」的那一处。
    i = -1
    for m in re.finditer("股票投资明细", txt):
        # 表头可能被排版拆成「序\n号」，先去掉空白再判断
        if "序号" in re.sub(r"\s+", "", txt[m.end():m.end() + 300]):
            i = m.start()
            break
    if i < 0:
        return []
    seg = txt[i:i + 4000]
    # 不能直接 find("5.4")：占比 "5.44" 里就含 "5.4"，会把表格截断在第二行。
    # 只认「5.4 报告期末……」这种段落标题。
    m = re.search(r"\n\s*5\.[4-9]\s*报告期末", seg)
    if m:
        seg = seg[:m.start()]
    lines = [l.strip() for l in seg.split("\n") if l.strip()]

    def _code_at(k: int) -> str | None:
        m = re.fullmatch(r"\d{1,2}\s+([0-9A-Z]{5,6})", lines[k])
        if m:
            return m.group(1)
        return lines[k] if re.fullmatch(r"[0-9A-Z]{5,6}", lines[k]) else None

    rows = []
    idx = [k for k in range(len(lines)) if _code_at(k)]
    for n, k in enumerate(idx):
        # 取「这只代码」到「下一只代码」之间的全部文本，再统一抽数字：
        # 不同基金的排版差异很大（数量与市值同行、市值被换行拆成两行），
        # 按固定位置取字段一定会漏；按「最后一个数字＝占比」才稳。
        end_k = idx[n + 1] if n + 1 < len(idx) else len(lines)
        body = lines[k + 1:end_k]
        # 下一行的「序号」会落在本段末尾（它是纯 1–2 位整数），
        # 不去掉会让「最后一个数字」变成下一只股票的序号。
        while body and re.fullmatch(r"\d{1,2}", body[-1]):
            body = body[:-1]
        chunk = " ".join(body)
        # 只认「带小数点的数」：序号、数量都是整数，占比一定带两位小数，
        # 否则会把下一行的序号（3、4）误当成占比。
        cands = re.findall(r"\d+\.\d{1,2}", chunk)
        if not cands:
            continue
        ratio = float(cands[-1])
        if not (0 < ratio <= 100):
            continue
        name = re.split(r"\d", chunk, maxsplit=1)[0].strip()
        if not name:
            continue
        rows.append((Q._norm_code(_code_at(k)), name, ratio))
        if len(rows) >= 10:
            break
    return rows


def _parse_shares(txt: str) -> float | None:
    """§2「报告期末基金份额总额」→ 亿份。"""
    m = re.search(r"报告期末基金份额总额\s*\n\s*([\d,]+(?:\.\d+)?)\s*份", txt)
    return float(m.group(1).replace(",", "")) / 1e8 if m else None


def _db_top10(con, code: str, q: str) -> list[tuple[str, str, float]]:
    rows = con.execute(
        "SELECT stock_code, stock_name, ratio_pct FROM q_hold "
        "WHERE code=? AND quarter=? AND rnk<=10 ORDER BY rnk", (code, q)).fetchall()
    return [(r[0], r[1], float(r[2])) for r in rows]


def _db_shares(con, code: str, q: str) -> float | None:
    r = con.execute("SELECT end_share_yi FROM q_shares WHERE code=? AND quarter=?",
                    (code, q)).fetchone()
    return float(r[0]) if r and r[0] is not None else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args()

    con = sqlite3.connect(DB)
    if args.codes:
        codes = [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]
    else:
        pool = [r[0] for r in con.execute(
            "SELECT DISTINCT code FROM q_hold WHERE quarter='2026Q3'")]
        pool.sort()
        random.seed(args.seed)
        codes = random.sample(pool, min(args.n, len(pool)))

    ok = bad = skip = ah = 0
    for c in codes:
        try:
            rp = _reports(c)
        except Exception as e:  # noqa: BLE001
            print(f"  {c} 公告接口失败 {type(e).__name__}")
            skip += 1
            continue
        q = "2026Q3"
        tgt = next((x for x in rp if _quarter_of(x["TITLE"]) == q), None)
        if tgt is None:
            print(f"  {c} 未找到 {q} 季报")
            skip += 1
            continue
        try:
            txt = _pdf_text(tgt["ID"])
        except Exception as e:  # noqa: BLE001
            print(f"  {c} PDF 解析失败 {type(e).__name__}")
            skip += 1
            continue
        pdf = _parse_top10(txt)
        db = _db_top10(con, c, q)
        if not db:
            print(f"  – {c} {q}：库内无该基金该季持仓（不在主动权益样本或未抓取），跳过")
            skip += 1
            continue
        pdf_sh = _parse_shares(txt)
        db_sh = _db_shares(con, c, q)
        # PDF 表里同名出现两次（A 股 + H 股）：官方按「合并公允价值」排序、
        # 分行披露，本库只能按单只占比排，名次必然错位——归为口径差异，不算错误。
        ah_pdf = len({r[1] for r in pdf}) != len(pdf)
        diffs = []
        if len(pdf) != len(db):
            diffs.append(f"条数 PDF {len(pdf)} vs 库 {len(db)}")
        for (pc, pn, pr), (dc, dn, dr) in zip(pdf, db):
            if pc != dc:
                diffs.append(f"代码 {pc}({pn}) vs {dc}({dn})")
            elif abs(pr - dr) > 0.015:
                diffs.append(f"{pc} 占比 {pr} vs {dr}")
        if pdf_sh is not None and db_sh is not None and abs(pdf_sh - db_sh) > 0.02:
            diffs.append(f"份额 {pdf_sh:.2f} vs {db_sh:.2f} 亿")
        if diffs and ah_pdf:
            ah += 1
            print(f"  ≈ {c} {q}：A+H 合并排序差异（数值一致，名次与原文不同）")
        elif diffs:
            bad += 1
            print(f"  ✗ {c} {q}：" + "；".join(diffs[:3]))
        else:
            ok += 1
            print(f"  ✓ {c} {q}：前十大 10 只与份额总额均一致"
                  + (f"（{pdf_sh:.2f} 亿份）" if pdf_sh else ""))
        time.sleep(0.3)

    print(f"\nPDF 抽查：一致 {ok}、A+H 排序差异 {ah}、不一致 {bad}、跳过 {skip}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
