# -*- coding: utf-8 -*-
"""显示契约测试：**把「读者能看到什么」写成断言**。

为什么不是普通的单元测试
------------------------
这个项目 89 次提交里 25 次带「修」，同一类问题反复出现——因为过去只验证
「代码跑不跑」，没验证「页面上会不会出现异常名、0.00%、越界名次」。
这组用例全部从**渲染结果**取样：表格的单元格、图的 trace、CSV 的文本。

约定
----
· 断言只依赖公开函数（canon / report_charts / caliber / version_info）；
· 不迁就实现：用例写的是「应该是什么样」，实现错了就让它红。
· `make gate` = pytest + check_caliber + deploy_check，任一失败阻断发布。
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import caliber as CAL          # noqa: E402
import canon as cn            # noqa: E402
import report_charts as rc    # noqa: E402
import version_info as vinfo  # noqa: E402


@pytest.fixture(scope="module")
def H() -> pd.DataFrame:
    return cn.csi_history()


@pytest.fixture(scope="module")
def small_map() -> dict[str, str]:
    cs = cn.crowding_score()
    return {i: f"持有 {int(cs.loc[i, '持有基金数'])} 只" for i in cs.index
            if cs.loc[i, "等级"] in ("小样本", "数据不足")}


@pytest.fixture(scope="module")
def H2(H) -> pd.DataFrame:
    """在真实面板上追加一个只有 2 期的新行业，覆盖时序小样本分支。"""
    d = H.copy()
    d["测试新行业"] = [np.nan] * (len(d) - 2) + [55.0, 61.0]
    return d


# ---------------------------------------------------------------------------
# A01 页面不许出现函数名 / 异常类名 / 元组参数
# ---------------------------------------------------------------------------
def test_A01_no_internal_names_in_errors():
    import qreport as Q  # noqa: PLC0415

    Q._DEP_PROBE = ["lxml", "akshare"]
    try:
        d = Q.fetch_fund("005827", force=True)
    finally:
        Q._DEP_PROBE = None
    assert d.get("errors"), "缺依赖时必须给出结构化错误"
    for e in d["errors"]:
        assert e["kind"] in ("missing_dep", "upstream_fail", "no_data", "partial")
        ui_text = " ".join(str(e.get(k, "")) for k in ("label", "reason", "action"))
        for bad in ("_f10", "_ak_", "ImportError", "ModuleNotFoundError", "(('"):
            assert bad not in ui_text, f"UI 文案泄漏内部细节：{bad} → {ui_text}"


# ---------------------------------------------------------------------------
# A02 行业结构图：图例 ≤12 项；没有「0.00%」的段；配色逐对可辨
# ---------------------------------------------------------------------------
def test_A02_structure_legend_and_zero_rows():
    cols = ["食品饮料", "医药生物", "汽车", "电子", "计算机", "通信", "公用事业",
            "交通运输", "银行", "非银金融", "房地产", "建筑装饰", "传媒"]
    M = pd.DataFrame(0.0, index=["2024Q3", "2024Q4", "2025Q1"], columns=cols)
    M.loc["2024Q3"] = [30, 20, 15, 15, 10, 5, 5, 0, 0, 0, 0, 0, 0]
    M.loc["2024Q4"] = [50, 20, 10, 10, 5, 5, 0, 0, 0, 0, 0, 0, 0]
    M.loc["2025Q1"] = [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fig = rc.fig_ind_structure(M, "测试")
    meta = fig.layout.meta
    assert len(meta["active"]) <= CAL.LEGEND_MAX_ITEMS
    # 0 值必须是 NaN（不进悬浮），不能是 0.00
    for tr in fig.data:
        ys = [y for y in tr.y if y is not None]
        assert not any((isinstance(y, float) and abs(y) < 1e-12) for y in ys), \
            f"{tr.name} 里仍有 0 值，会在悬浮里显示 0.00%"
    assert rc.validate_palette(rc._STACK12) == [], "调色板相邻色不可辨（含色盲模拟）"


# ---------------------------------------------------------------------------
# A03 n<8 的行业：z / P10–P90 / 排名 一律「—」
# ---------------------------------------------------------------------------
def test_A03_small_n_gets_dash(H2, small_map):
    t = rc.csi_stat_table(H2, window=None, small=small_map)
    row = t[t["行业"] == "测试新行业"].iloc[0]
    for col in ("z值", "P10", "P50", "P90", "排名", "全行业分位"):
        assert row[col] == "—" or row[col] == "—（不评级）", f"{col}={row[col]}"
    assert row["期数"].endswith("*"), "期数应带 * 标注样本不足"
    assert "仅 2 期" in row["评级"]


# ---------------------------------------------------------------------------
# A04 排名不越界：最大名次 == 可评级行业数
# ---------------------------------------------------------------------------
def test_A04_rank_bound(H, small_map):
    t = rc.csi_stat_table(H, window=None, small=small_map)
    ranks = pd.to_numeric(t["排名"], errors="coerce").dropna()
    assert ranks.max() == t.attrs["n_pool"], "排名最大值应等于可评级行业数"
    assert ranks.min() == 1
    assert len(ranks) == t.attrs["n_pool"]


# ---------------------------------------------------------------------------
# A05 全行业分位的分母 == 可评级行业数（不是 31）
# ---------------------------------------------------------------------------
def test_A05_cross_section_denominator(H, small_map):
    t = rc.csi_stat_table(H, window=None, small=small_map)
    numeric = pd.to_numeric(t["全行业分位"], errors="coerce").dropna()
    assert len(numeric) == t.attrs["n_pool"]
    assert t.attrs["n_pool"] == len(H.columns) - t.attrs["n_small"]
    assert numeric.max() <= 100 and numeric.min() > 0


# ---------------------------------------------------------------------------
# A06 同一指标在表 / CSV / 图 三处一致
# ---------------------------------------------------------------------------
def test_A06_table_csv_chart_agree(H, small_map):
    t = rc.csi_stat_table(H, window=None, small=small_map)
    csv = pd.read_csv(pd.io.common.StringIO(t.to_csv(index=False)), dtype=str)
    assert list(csv.columns) == list(t.columns)
    assert (csv["排名"].fillna("") == t["排名"].fillna("")).all(), "CSV 与表格排名不一致"
    # 图上标题的「全行业第 X/N」必须与表里的 X、N 相同
    sel = t.iloc[0]["行业"]
    fig = rc.fig_csi_trend(H, sel, True,
                           rank_universe=[c for c in H.columns
                                          if c not in small_map])
    title = fig.layout.title.text
    assert f"全行业第 {t.iloc[0]['排名']}/{t.attrs['n_pool']}" in title, title


# ---------------------------------------------------------------------------
# A07 阈值只在 caliber.py 里出现
# ---------------------------------------------------------------------------
def test_A07_thresholds_centralized():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_caliber.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-800:]


# ---------------------------------------------------------------------------
# A08 CSV 带同样的小样本标注
# ---------------------------------------------------------------------------
def test_A08_csv_keeps_small_sample_marks(H, small_map):
    t = rc.csi_stat_table(H, window=None, small=small_map)
    csv_text = t.to_csv(index=False)
    assert "小样本" in csv_text
    assert "—（不评级）" in csv_text
    # 期数带 * 的样本不足行也要在 CSV 里保留（真实数据若没有这类行，跳过）
    stars = [x for x in t["期数"].astype(str) if x.endswith("*")]
    if stars:
        assert stars[0] in csv_text


# ---------------------------------------------------------------------------
# A09 「全历史触及」不是总期数
# ---------------------------------------------------------------------------
def test_A09_touch_column_not_total_periods(H, small_map):
    t = rc.csi_stat_table(H, window=None, small=small_map)
    touch = pd.to_numeric(t["全历史触及≥80"], errors="coerce").dropna()
    n_col = pd.to_numeric(t.loc[touch.index, "期数"].str.rstrip("*"), errors="coerce")
    assert touch.max() < n_col.max(), "「触及次数」不可能等于总期数——疑似取错列"
    assert (touch <= n_col).all()


# ---------------------------------------------------------------------------
# A10 侧栏页脚版本行存在且 sha 与 version.json 一致
# ---------------------------------------------------------------------------
def test_A10_footer_version_line():
    src = open(os.path.join(ROOT, "src", "sidebar_panel.py"), encoding="utf-8").read()
    assert "vinfo.sha7()" in src and "build_time" in src, "侧栏缺版本行"
    vi = vinfo.info()
    assert vi["source_sha"] and len(vinfo.sha7()) == 7, "version.json 未生成"
    assert "akshare" in vinfo.deps_text()


# ---------------------------------------------------------------------------
# A11 每页都有来源页脚：原始披露 + 转引 + 可点链接 + 日期
# ---------------------------------------------------------------------------
def test_A11_source_footer_present_and_linked():
    import sources  # noqa: PLC0415

    for page in ("overview", "report", "map", "fund", "factor", "method"):
        html = sources.source_line(page)
        assert "来源与出处" in html, f"{page} 缺来源页脚"
        # 原始披露（官方）与转引（东财）都必须出现，且是 <a href> 可点的
        assert "eid.csrc.gov.cn" in html, f"{page} 缺官方原始披露链接"
        assert "fundf10.eastmoney.com" in html, f"{page} 缺东财转引链接"
        assert html.count("<a href=") >= 3, f"{page} 链接不足（应含官方/东财/Wind）"
        assert 'target="_blank"' in html, f"{page} 链接不是新窗口打开"
        # 必须有日期：抓取日或快照日，二者至少一个
        assert ("访问" in html) or ("快照构建" in html), f"{page} 缺访问日期"


def test_A12_source_dates_are_iso_or_empty():
    import sources  # noqa: PLC0415
    import re  # noqa: PLC0415

    for d in (sources.fund_fetch_date(), sources.snapshot_date(), sources.index_date()):
        assert d == "" or re.fullmatch(r"\d{4}-\d{2}-\d{2}", d), f"日期格式不对：{d}"


# ---------------------------------------------------------------------------
# A13 份额表同一季有多条披露日时，必须取季末日（2026-11-13 真实 bug）
# ---------------------------------------------------------------------------
def test_A13_share_row_picks_quarter_end():
    import qreport_page as QP  # noqa: PLC0415

    df = pd.DataFrame({
        "quarter": ["2017Q1", "2017Q1", "2017Q1"],
        "period": ["2017-01-25", "2017-02-24", "2017-03-31"],
        "purchase_yi": [None, None, 0.03],
        "redeem_yi": [None, None, 0.15],
        "end_share_yi": [2.54, None, 2.42],
    })
    row = QP._pick_row(df, "2017Q1")
    assert row is not None
    assert row["period"] == "2017-03-31", f"取到了周度行：{row['period']}"
    assert float(row["purchase_yi"]) == 0.03


# ---------------------------------------------------------------------------
# A14 季报速览：净资产环比与单位净值涨跌必须是两列，不能混成一个「净值涨跌」
# ---------------------------------------------------------------------------
def test_A14_nav_vs_unit_nav_are_separate_columns():
    src = open(os.path.join(ROOT, "src", "qreport_page.py"), encoding="utf-8").read()
    assert "净资产环比%" in src
    assert "单位净值涨跌%" in src
    assert '"净值涨跌%"' not in src, "旧的「净值涨跌%」标签又回来了（源站列名是净资产变动率）"


# ---------------------------------------------------------------------------
# A15 因子页必须解释「无效」是什么意思，且热力图必须常驻
# ---------------------------------------------------------------------------
def test_A15_verdict_legend_and_heatmap_visible():
    fs = open(os.path.join(ROOT, "src", "factor_page.py"), encoding="utf-8").read()
    for lab in ("与 0 无法区分", "不是「算错」", "样本不足"):
        assert lab in fs, f"因子页缺少结论解释：{lab}"
    ms = open(os.path.join(ROOT, "src", "map_page.py"), encoding="utf-8").read()
    assert "31 个行业全景" in ms
    assert "看历史全景：" not in ms, "热力图又被折叠了"
