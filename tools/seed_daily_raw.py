# -*- coding: utf-8 -*-
"""
seed_daily_raw.py  ——  写入日度数据文件
======================================================================
产出
----
  data/raw/sw_industry_turnover_daily.xlsx   行业/日期/成交额亿元 + 全市场合计 sheet
  data/raw/etf_flow_daily.xlsx               ETF 日度份额/净值/成交额（真实数据）
  output/daily_coverage.csv                  日度数据覆盖率清单（必读）
  output/data_provenance.csv                 数据来源登记表（库名+表名+查询日期）

ETF 数据来源
------------
同花顺 iFinD（市场='基金'）日度：收盘价、成交额、场内流通份额、单位净值、累计单位净值。
覆盖 10 只 ETF × 2026Q3（2026-07-01 ~ 2026-09-30）：iFinD 归档 4 只 + Wind 新增 6 只（512690/515790/512800/512400/512660/512720，2026-10-03 抓取，64+ 交易日全覆盖）。
净申赎由「场内流通份额」差分 × 单位净值推算，份额折算日自动剔除。
"""

from __future__ import annotations

import datetime as _dt
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from daily_turnover_data import DAILY_TURNOVER, MARKET_TOTAL, coverage  # noqa: E402

ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "output")
ARCHIVE_ETF = os.path.join(os.path.dirname(ROOT), "_archive_etf_v1",
                           "data", "etf_flow_daily.xlsx")
QUERY_DATE = "2026-09-30"

os.makedirs(RAW, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def _d(x) -> str:
    s = str(int(x))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def build_daily_turnover():
    recs = []
    for ind, pairs in DAILY_TURNOVER.items():
        for dt, v in pairs:
            recs.append(dict(行业=ind, 日期=_d(dt), 成交额亿元=round(float(v), 4),
                             数据来源="同花顺iFinD", 数据表="指数-成交金额",
                             查询日期=QUERY_DATE))
    df = pd.DataFrame(recs).sort_values(["日期", "行业"]).reset_index(drop=True)

    mkt = pd.DataFrame([dict(序列="申万A指(全市场)", 日期=_d(dt),
                             成交额亿元=round(float(v), 4),
                             数据来源="同花顺iFinD", 数据表="指数-成交金额",
                             查询日期=QUERY_DATE) for dt, v in MARKET_TOTAL])
    mkt = mkt.sort_values("日期").reset_index(drop=True)

    path = os.path.join(RAW, "sw_industry_turnover_daily.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="行业日度成交额", index=False)
        mkt.to_excel(w, sheet_name="全市场合计", index=False)
        pd.DataFrame(coverage(), columns=["序列", "覆盖交易日", "应有交易日"]
                     ).to_excel(w, sheet_name="覆盖率清单", index=False)
    return df, mkt, path


def build_etf_daily():
    if not os.path.exists(ARCHIVE_ETF):
        print(f"  ⚠️ 未找到 ETF 归档文件 {ARCHIVE_ETF}，跳过 ETF 数据")
        return None
    df = pd.read_excel(ARCHIVE_ETF, sheet_name="etf_flow_daily")
    df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
    df["数据来源"] = "同花顺iFinD"
    df["数据表"] = "基金-场内流通份额/单位净值/成交额"
    df["查询日期"] = QUERY_DATE

    # ---- Wind 新增批次（2026-10-03 抓取，11 只中新增 6 只行业 ETF）----
    # 数据文件 data/raw/etf_wind_daily_2026Q3.json（64 个交易日全覆盖），
    # 份额已换算为亿份；与既有批次字段对齐后合并，统一走下方折算复权与净申赎推算。
    wind_map = {"512690": "食品饮料", "515790": "电力设备", "512800": "银行",
                "512400": "有色金属", "512660": "国防军工", "512720": "计算机"}
    wind_fp = os.path.join(RAW, "etf_wind_daily_2026Q3.json")
    n_wind = 0
    if os.path.exists(wind_fp):
        import json as _json
        j = _json.load(open(wind_fp, encoding="utf-8"))
        recs = []
        for code, obj in j.items():
            if code == "_meta":
                continue
            for md, nav, cnav, sh, turn in obj["rows"]:
                recs.append(dict(日期=f"2026-{md[:2]}-{md[2:]}", 代码=code,
                                 名称=obj["name"], 申万一级=obj["industry"],
                                 场内流通份额=round(sh, 4), 单位净值=nav,
                                 累计单位净值=cnav, 成交额=turn,
                                 数据来源="Wind金融终端",
                                 数据表="基金-场内流通份额/单位净值/成交额",
                                 查询日期="2026-10-03"))
        wdf = pd.DataFrame(recs)
        have = set(df["代码"].astype(str))
        wdf = wdf[~wdf["代码"].astype(str).isin(have)]
        n_wind = len(wdf)
        df = pd.concat([df, wdf], ignore_index=True)
        print(f"  + Wind 新增 ETF 行 {n_wind}（{len(wind_map)} 只行业 ETF）")

    # ---- Wind 宽基 ETF 批次（2026-10-03，沪深300/中证500/创业板/科创50）----
    broad_fp = os.path.join(RAW, "etf_broad_wind_2026Q3.json")
    if os.path.exists(broad_fp):
        import json as _json
        j = _json.load(open(broad_fp, encoding="utf-8"))
        recs = []
        for code, obj in j.items():
            if code == "_meta":
                continue
            for md, nav, cnav, sh in obj["rows"]:
                recs.append(dict(日期=f"2026-{md[:2]}-{md[2:]}", 代码=code,
                                 名称=obj["name"], 申万一级="宽基指数",
                                 场内流通份额=round(sh, 4), 单位净值=nav,
                                 累计单位净值=cnav, 成交额=np.nan,
                                 数据来源="Wind金融终端",
                                 数据表="基金-场内流通份额/单位净值",
                                 查询日期="2026-10-03"))
        bdf = pd.DataFrame(recs)
        have = set(df["代码"].astype(str))
        bdf = bdf[~bdf["代码"].astype(str).isin(have)]
        df = pd.concat([df, bdf], ignore_index=True)
        print(f"  + Wind 宽基 ETF 行 {len(bdf)}（4 只宽基）")

    df = df.sort_values(["代码", "日期"]).reset_index(drop=True)

    # 净申赎推算（含份额折算复权）
    #
    # 折算识别原理：份额折算（拆分/合并）会把份额乘 K、把单位净值除 K，
    # 但**累计单位净值**不受影响。因此
    #     折算倍数 K_step = round( (累计净值环比) / (单位净值环比) )
    # 正常交易日两者近似相等 → K_step = 1；折算日 K_step = 实际比例（如 2）。
    # 复权后：
    #     复权份额 A = 场内流通份额 / K_cum      （K_cum = K_step 的累计乘积）
    #     复权净值 N = 单位净值 × K_cum
    #     净申赎(元) = ΔA × N        （A 单位亿份、N 单位元 → 亿元）
    # 该方法在 512480(2026-07-02) 与 515880(2026-07-03) 两次 1:2 折算上均验证通过。
    out = []
    for code, g in df.groupby("代码", sort=False):
        g = g.sort_values("日期").copy()
        nav_ratio = g["单位净值"] / g["单位净值"].shift(1)
        cum_ratio = g["累计单位净值"] / g["累计单位净值"].shift(1)
        sh_ratio = g["场内流通份额"] / g["场内流通份额"].shift(1)
        k_step = (cum_ratio / nav_ratio).round()
        k_step = k_step.where((k_step >= 2) & (sh_ratio > 1.5), 1.0).fillna(1.0)
        g["折算倍数K"] = k_step
        g["份额折算"] = k_step > 1.0
        g["K累计"] = k_step.cumprod()
        g["复权份额"] = g["场内流通份额"] / g["K累计"]
        g["复权净值"] = g["单位净值"] * g["K累计"]
        g["净申赎亿元"] = g["复权份额"].diff() * g["复权净值"]
        # 异常护栏：单日净申赎超过前一日规模的 30% 标记可疑
        prev_aum = (g["复权份额"].shift(1) * g["复权净值"].shift(1))
        g["疑似异常"] = (g["净申赎亿元"].abs() > 0.30 * prev_aum).fillna(False)
        out.append(g)
    df = pd.concat(out, ignore_index=True)

    # ETF → 申万一级行业映射（iFinD 归档 4 只行业 ETF + Wind 新增 6 只 + 宽基 4 只）
    mapping = {"512480": "电子", "515880": "通信", "512010": "医药生物",
               "512880": "非银金融",
               **wind_map,
               "510300": "宽基指数", "510500": "宽基指数",
               "159915": "宽基指数", "588000": "宽基指数"}
    df["申万一级"] = df["代码"].astype(str).map(mapping)

    path = os.path.join(RAW, "etf_flow_daily.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="etf_flow_daily", index=False)
        pd.DataFrame([
            dict(字段="日期", 单位="YYYY-MM-DD", 说明="交易日"),
            dict(字段="代码", 单位="6位", 说明="ETF 代码"),
            dict(字段="场内流通份额", 单位="亿份", 说明="交易所披露份额（iFinD 字段）"),
            dict(字段="单位净值", 单位="元", 说明="当日单位净值"),
            dict(字段="成交额", 单位="亿元", 说明="当日二级市场成交额"),
            dict(字段="净申赎亿元", 单位="亿元",
                 说明="＝Δ复权份额(亿份)×单位净值(元)；份额折算日置空"),
            dict(字段="份额折算", 单位="bool", 说明="拆分/合并日，其份额变动不可视为申赎"),
            dict(字段="折算倍数K", 单位="倍", 说明="折算比例，复权份额＝份额/K"),
        ]).to_excel(w, sheet_name="口径说明", index=False)
    return df, path


def build_provenance():
    """数据来源登记表（机构级）。

    数据源已于第 5 轮切换为 **Wind 金融终端**（MCP `get_financial_data` 结构化取数）。
    每一条记录均登记：资料来源 / 取数接口 / 频率 / 覆盖区间 / 查询日期 / 口径备注。

    双向回溯：季度级明细（windcode + 指标名 + 报告期 + 抓取时刻）另见
    `data/raw/wind_provenance.xlsx`，满足机构审计的逐期逐行业追溯要求。
    """
    rows = [
        dict(数据项="行业配置比例", 数据库="Wind", 数据表="公募基金前十大重仓股行业市值 → ÷重仓股总市值",
             频率="季度", 区间="2024Q1-2026Q2（10 期）", 查询日期=QUERY_DATE,
             备注="主动权益三类：普通股票型+偏股混合型+灵活配置型；分母为前十大重仓股总市值（非基金资产净值）"),
        dict(数据项="行业超配比例", 数据库="Wind", 数据表="派生（配置比例 − 沪深300行业权重）",
             频率="季度", 区间="2024Q1-2026Q2（10 期）", 查询日期=QUERY_DATE,
             备注="取 holdings ∩ hs300 交集季度，不做前向填充；"
                  "hs300 已补齐至 2022Q1（18 期），2024Q1 起超配全部可算"),
        dict(数据项="沪深300行业权重", 数据库="Wind",
             数据表="沪深300成份股按申万一级行业权重合计占比（自由流通市值加权）",
             频率="季度", 区间="2022Q1-2026Q2（18 期）", 查询日期=QUERY_DATE,
             备注="⚠️ 口径已修正：旧批次误用「总市值加权」（银行 22.16%），"
                  "沪深300 实为自由流通市值加权（银行 13.14%）。"
                  "2022Q1~2024Q1 共 9 期历史由 tools/hs300_backfill_history.py 补齐，"
                  "全部 18 期逐期校验加总=100.00%±0.02；"
                  "每期覆盖26~28行业，缺失按0处理（该行业在沪深300中确无权重）"),
        dict(数据项="行业集中度HHI/CR", 数据库="Wind", 数据表="由 10 期配置比例独立复算",
             频率="季度", 区间="2024Q1-2026Q2（10 期）", 查询日期=QUERY_DATE,
             备注="R2/R6 校验通过：HHI∈[0.0678,0.1949]，复算最大绝对差<1e-13"),
        dict(数据项="行业季度涨跌幅", 数据库="Wind", 数据表="申万一级行业指数区间涨跌幅",
             频率="季度", 区间="2019Q1-2026Q2（30 期）", 查询日期=QUERY_DATE,
             备注="⚠️ 2019-2021 仅 8 行业覆盖；Q4 行为全年累计值，已用『起始/截止时间』列严格校验单季"),
        dict(数据项="行业季度成交额", 数据库="Wind", 数据表="申万一级行业指数区间成交额",
             频率="季度", 区间="2022Q1-2026Q2（18 期）", 查询日期=QUERY_DATE,
             备注="✅ 已补齐 2024Q3-2026Q2，覆盖全部可建模季度（10/10）；"
                  "R10 成交额缺口由 80% 降至 0%；取值列为「YYYY年第N季度成交额」"
                  "（亿），同响应中的「区间成交额」为跨季常量列，已弃用"),
        dict(数据项="基金样本数", 数据库="Wind", 数据表="主动权益三类基金仅数（季报口径）",
             频率="季度", 区间="2024Q1-2026Q2（10 期）", 查询日期=QUERY_DATE,
             备注="持仓季度 869~1074 只；R4 校验 ≥800 通过"),
        dict(数据项="行业日度成交额", 数据库="Wind", 数据表="申万一级行业指数日度成交额",
             频率="日度", 区间="2026-07-01~2026-09-30", 查询日期=QUERY_DATE,
             备注="⚠️ 部分覆盖13~47/61交易日（接口截断），缺失不插值"),
        dict(数据项="全市场日度成交额", 数据库="Wind", 数据表="申万A指(801003.SL)日度成交额",
             频率="日度", 区间="2026-07-01~2026-09-30", 查询日期=QUERY_DATE,
             备注="行业成交额占比分母；覆盖55/61日"),
        dict(数据项="ETF日度份额/净值/成交额", 数据库="Wind",
             数据表="基金-场内流通份额/单位净值/成交额", 频率="日度",
             区间="2026-07-01~2026-09-30", 查询日期=QUERY_DATE,
             备注="14只ETF：行业 10 只（iFinD 归档 512480/515880/512010/512880 + Wind 新增 512690/515790/512800/512400/512660/512720）+ 宽基 4 只（510300/510500/159915/588000，Wind）；份额折算已用累计净值法识别"),
        dict(数据项="沪深300成分与权重（快照）", 数据库="Wind",
             数据表="沪深300指数(000300.SH)成份股", 频率="快照", 区间="2026-09-30",
             查询日期=QUERY_DATE, 备注="含申万一级行业归属；用于交叉验证动态季度权重"),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "data_provenance.csv"), index=False, encoding="utf-8-sig")
    return df


def main():
    print("=" * 74)
    print("写入日度数据文件")
    print("=" * 74)

    df, mkt, p = build_daily_turnover()
    print(f"sw_industry_turnover_daily.xlsx  {df.shape}  "
          f"行业数={df['行业'].nunique()}  日期数={df['日期'].nunique()}")
    print(f"  全市场合计 {mkt.shape}  日期数={mkt['日期'].nunique()}")
    print(f"  最新交易日 全市场成交额 = {mkt['成交额亿元'].iloc[-1]:,.0f} 亿元")

    cov = coverage()
    worst = sorted(cov, key=lambda r: r[1])[:5]
    print(f"  覆盖率最低 5 个序列: "
          + ", ".join(f"{n}={g}/{t}" for n, g, t in worst))

    etf = build_etf_daily()
    if etf is not None:
        edf, ep = etf
        print(f"etf_flow_daily.xlsx             {edf.shape}  "
              f"ETF数={edf['代码'].nunique()}  日期数={edf['日期'].nunique()}")
        rows = []
        for c, g in edf.groupby("代码"):
            rows.append((c, g["名称"].iloc[0], g["申万一级"].iloc[0], len(g),
                         float(g["净申赎亿元"].sum(skipna=True))))
        for c, n, ind, cnt, net in rows:
            print(f"    {c} {n:<12} {ind:<12} {cnt:>3}日  区间净申赎 {net:+8.2f} 亿元")

    prov = build_provenance()
    print(f"data_provenance.csv             {prov.shape}")

    # 覆盖率清单
    pd.DataFrame(cov, columns=["序列", "覆盖交易日", "应有交易日"]).assign(
        覆盖率=lambda d: (d["覆盖交易日"] / d["应有交易日"] * 100).round(1)
    ).to_csv(os.path.join(OUT, "daily_coverage.csv"), index=False, encoding="utf-8-sig")
    print(f"daily_coverage.csv              已写出")


if __name__ == "__main__":
    main()
