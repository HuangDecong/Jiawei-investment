# -*- coding: utf-8 -*-
"""修复股票代码错乱：港股被 zfill 成 6 位后与 A 股撞码。

问题
----
东财返回的港股代码是 5 位（00700），而 A 股是 6 位（000700 = 模塑科技）。
早期管道一律 `zfill(6)`，于是：
    腾讯控股 00700 → 000700 → 撞上模塑科技 → 行业被标成「汽车」
更糟的是部分行的「股票代码」是数值型（920045.0），长度判断失效，
北交所/存托凭证被误标成港股。

修复思路
--------
不信任已入库的代码，改用**名称 → 申万成分股代码**重建：
· 申万 31 个一级行业的成分股表带「证券代码 + 证券名称」，覆盖全部 A 股；
· 名称能对上 → 用申万给的 6 位代码，行业也用申万的一级分类；
· 名称对不上 → 判为港股/海外，代码按「去前导零后补足 5 位」，行业置 NULL。

用法：
    python tools/repair_stock_codes.py            # 只报告，不改库
    python tools/repair_stock_codes.py --apply    # 实际写入
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import unicodedata

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "db", "fund_ak.db")
MAP = os.path.join(ROOT, "data", "raw", "sw_industry_map.json")
NAME_MAP = os.path.join(ROOT, "data", "raw", "sw_name_code.json")


def norm(x) -> str:
    s = unicodedata.normalize("NFKC", str(x))
    return s.replace(" ", "").replace("　", "").upper()


def build_name_map(force: bool = False) -> dict:
    """名称 → {code, industry}，来源申万一级成分股。"""
    if os.path.exists(NAME_MAP) and not force:
        return json.load(open(NAME_MAP, encoding="utf-8"))
    import akshare as ak
    sw = ak.sw_index_first_info()
    out, code2ind = {}, {}
    for _, r in sw.iterrows():
        code = str(r["行业代码"]).split(".")[0]
        name = str(r["行业名称"])
        try:
            c = ak.index_component_sw(symbol=code)
        except Exception as e:
            print(f"  成分失败 {name}: {type(e).__name__}")
            continue
        for _, x in c.iterrows():
            sc = str(x["证券代码"]).zfill(6)
            sn = str(x["证券名称"])
            out[norm(sn)] = dict(code=sc, industry=name, raw=sn)
            code2ind[sc] = name
    json.dump(out, open(NAME_MAP, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(code2ind, open(MAP, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"申万名称映射：{len(out)} 只 → {NAME_MAP}")
    return out


HK_PATH = os.path.join(ROOT, "data", "raw", "hk_names.json")


def build_hk_names(force: bool = False) -> dict:
    """港股名称 → 5 位代码。用于把港股与 A 股彻底分开。"""
    if os.path.exists(HK_PATH) and not force:
        return json.load(open(HK_PATH, encoding="utf-8"))
    import akshare as ak
    d = ak.stock_hk_spot()
    out = {}
    for _, r in d.iterrows():
        out[norm(r["中文名称"])] = str(r["代码"]).zfill(5)
    json.dump(out, open(HK_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"港股名单：{len(out)} 只 → {HK_PATH}")
    return out


def strip_st(name: str) -> str:
    """去掉 ST / *ST / SST / 退市 等前后缀，回到基础简称。"""
    s = norm(name)
    s = re.sub(r"^S?\*?ST", "", s)
    s = re.sub(r"^\*", "", s)
    s = re.sub(r"（退市）|\(退市\)|退$", "", s)
    return s


def a_share_names() -> dict:
    """备用 A 股名称表（含当前 ST 简称）。"""
    p = os.path.join(ROOT, "data", "raw", "a_share_code_name.csv")
    if not os.path.exists(p):
        return {}
    d = pd.read_csv(p, dtype={"code": str})
    return {norm(r["name"]): str(r["code"]).zfill(6) for _, r in d.iterrows()}


def hk_code(stored) -> str:
    """把错乱的存储代码还原成 5 位港股代码。"""
    s = str(stored).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"\D", "", s)
    return digits[-5:].zfill(5) if digits else "00000"


def main(apply: bool):
    nm = build_name_map()
    con = sqlite3.connect(DB, timeout=90)
    con.execute("PRAGMA busy_timeout=90000")
    con.isolation_level = None

    hk = build_hk_names()
    a2c = a_share_names()
    d = pd.read_sql("SELECT stock_name, stock_code, COUNT(*) n FROM fund_holdings "
                    "GROUP BY stock_name, stock_code", con)
    stat = {"申万名称": 0, "A股名称表": 0, "港股名单": 0, "代码兜底": 0, "判为港股": 0}
    plan = []
    for name, g in d.groupby("stock_name"):
        k = norm(name)
        ks = strip_st(name)
        top = g.sort_values("n", ascending=False).iloc[0]
        info = nm.get(ks) or nm.get(k)
        if info:
            plan.append((info["code"], 0, info["industry"], name))
            stat["申万名称"] += int(g["n"].sum())
            continue
        if ks in a2c or k in a2c:
            code = a2c.get(ks) or a2c.get(k)
            ind = None
            for v in nm.values():
                if v["code"] == code:
                    ind = v["industry"]
                    break
            plan.append((code, 0, ind, name))
            stat["A股名称表"] += int(g["n"].sum())
            continue
        # 只用「全名」匹配港股：去掉 ST 前缀后的简称会误撞港股（如 *ST中润 → 中润）
        if k in hk:
            plan.append((hk[k], 1, None, name))
            stat["港股名单"] += int(g["n"].sum())
            continue
        # 兜底：ST/退市 一定是 A 股（这类主体已被剔除出申万成分，名称对不上，
        # 但它们不是港股）；其余按代码长度判断，短代码视为港股。
        digits = re.sub(r"\D", "", str(top["stock_code"]))
        is_st = ("ST" in k) or ("退" in k)
        if len(digits) == 6 or is_st:
            plan.append((digits.zfill(6) if digits else "000000", 0, None, name))
            stat["代码兜底"] += int(g["n"].sum())
        else:
            plan.append((hk_code(top["stock_code"]), 1, None, name))
            stat["判为港股"] += int(g["n"].sum())

    tot = sum(stat.values())
    print(f"名称 {d['stock_name'].nunique()} 个、{tot} 行，判定分布：")
    for k_, v_ in stat.items():
        print(f"  {k_}: {v_} 行（{v_ / tot * 100:.1f}%）")
    print("港股样例：", [p[3] for p in plan if p[1] == 1][:8])
    print("A股样例：", [(p[3], p[0], p[2]) for p in plan if p[1] == 0][:5])

    if not apply:
        print("\n未加 --apply，仅报告。")
        con.close()
        return

    before = con.execute("SELECT COUNT(*) FROM fund_holdings").fetchone()[0]
    skipped = 0
    for code, is_hk, ind, name in plan:
        try:
            con.execute("UPDATE fund_holdings SET stock_code=?, is_hk=?, industry=? "
                        "WHERE stock_name=?", (code, is_hk, ind, name))
        except sqlite3.IntegrityError:
            # 两只不同简称落到同一个代码（改名/同一主体两地上市），
            # 同一基金同季已存在该代码行 → 跳过，避免破坏主键
            skipped += 1
    print(f"因主键冲突跳过 {skipped} 个名称")
    after = con.execute("SELECT COUNT(*) FROM fund_holdings").fetchone()[0]
    dup = con.execute(
        "SELECT COUNT(*) FROM (SELECT code, quarter, stock_code, COUNT(*) c "
        "FROM fund_holdings GROUP BY 1,2,3 HAVING c>1)").fetchone()[0]
    con.close()
    print(f"已更新：行数 {before} → {after}，主键冲突组 {dup}")

    # 复核：港股不再有行业，A 股不再有 5 位代码
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    chk = pd.read_sql("SELECT is_hk, LENGTH(stock_code) L, COUNT(*) n "
                      "FROM fund_holdings GROUP BY is_hk, L", con)
    con.close()
    print("\n修复后代码形态：\n", chk.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="强制重建申万名称映射")
    a = ap.parse_args()
    if a.refresh and os.path.exists(NAME_MAP):
        os.remove(NAME_MAP)
    main(a.apply)
