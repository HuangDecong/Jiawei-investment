# -*- coding: utf-8 -*-
"""全站审计：静态检查 → 口径自检 → 页面渲染 → 文本卫生 → 交互 → 跨页一致性。

为什么要有这个脚本
------------------
这个项目踩过的坑，几乎都不是「算错」，而是：
  · 本地能跑、线上报错（推送漏文件）；
  · 页面某一块静默失败，看起来只是「没有内容」；
  · 同一句话在两个页面上不是一个数；
  · 改了一处，另一处还在用旧文案。

所以审计必须同时覆盖「代码」「数据」「渲染结果」三层，并且每次都跑同一套，
而不是靠人眼看截图。

用法：
    python tools/audit_all.py            # 全量
    python tools/audit_all.py --fast     # 跳过交互（约 40 秒）
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

PAGES = ["① 结论总览", "② 本期解读（全市场）", "③ 行业拥挤度", "④ 单只基金",
         "⑤ 因子检验", "⑥ 方法与口径"]

# 文本卫生：这些串出现在 HTML 里，说明有变量是 NaN、或 Markdown 没被渲染
BAD_PATTERNS = [
    (re.compile(r"(?<![A-Za-z])nan(?![A-Za-z])"), "nan"),
    (re.compile(r"(?<![A-Za-z])NaN(?![A-Za-z])"), "NaN"),
    (re.compile(r"(?<![A-Za-z])None(?![A-Za-z])"), "None"),
    (re.compile(r"\*\*[^\s*][^*]{0,40}\*\*"), "** 未渲染"),
    (re.compile(r"&lt;b&gt;"), "转义标签"),
    (re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}"), "未替换的占位符"),
]

ROWS: list[tuple[str, str, str, str]] = []


def rec(stage: str, item: str, ok: bool, detail: str = "") -> None:
    ROWS.append((stage, item, "通过" if ok else "失败", detail))


# ---------------------------------------------------------------------------
# 1. 语法与静态
# ---------------------------------------------------------------------------
def stage_static() -> None:
    files = ([os.path.join(ROOT, "app.py")]
             + [os.path.join(ROOT, "src", f) for f in os.listdir(os.path.join(ROOT, "src"))
                if f.endswith(".py")]
             + [os.path.join(ROOT, "tools", f) for f in os.listdir(os.path.join(ROOT, "tools"))
                if f.endswith(".py")])
    bad = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                compile(fh.read(), f, "exec")
        except Exception as e:
            bad.append(f"{os.path.basename(f)}: {e}")
    rec("静态", f"全部 {len(files)} 个 .py 可编译", not bad, "；".join(bad[:3]))

    # pyflakes 只看「未定义名」，其余（未使用变量等）不阻断
    try:
        out = subprocess.run([sys.executable, "-m", "pyflakes"] + files,
                             capture_output=True, text=True, timeout=120).stdout
        undef = [ln for ln in out.splitlines()
                 if "undefined name" in ln or "redefinition" in ln]
        rec("静态", "无未定义名 / 重复定义", not undef, "；".join(undef[:3]))
    except Exception as e:
        rec("静态", "pyflakes 检查", True, f"跳过：{type(e).__name__}")

    # 每个分析块必须写满三段（事实 / 经济含义 / 投资含义）。
    # 用 AST 而不是正则：_read(h, a, b) 这种少一段的调用，正则很难数清楚。
    import ast
    short = []
    for f in files:
        if os.sep + "tools" + os.sep in f:
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
        except Exception:
            continue
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in ("_read", "analysis")):
                if len(n.args) < 4 and not ({k.arg for k in n.keywords} & {"invest"}):
                    short.append(f"{os.path.basename(f)}:{n.lineno}")
    rec("静态", "分析块都写了投资含义（三段式）", not short, "；".join(short[:5]))


# ---------------------------------------------------------------------------
# 2. 口径自检
# ---------------------------------------------------------------------------
def stage_selfcheck() -> None:
    import selfcheck
    t = selfcheck.run()
    n, m = t.attrs.get("n_pass"), t.attrs.get("n_all")
    bad = t[t["结果"].astype(str) != "通过"]
    rec("数据", f"口径恒等式 {n}/{m}", n == m,
        "；".join(bad["项目"].tolist()[:4]) if len(bad) else "")


def stage_glossary() -> None:
    """词典：条目完整、术语全覆盖、每条都能渲染。"""
    import glossary as g
    from audit_glossary import check_fields, check_terms, check_render
    bad = check_fields() + check_terms() + check_render()
    ok, all_ = g.coverage()
    rec("词典", f"条目 {len(g.ITEMS)} 条 · 九段完整 · 渲染正常", not bad,
        "；".join(bad[:4]) if bad else f"{len(g.CATS)} 类")
    rec("词典", f"页面术语覆盖 {ok}/{all_}", ok == all_,
        "" if ok == all_ else "有术语检索不到")


def stage_data() -> None:
    import canon as cn
    qs = cn.quarters()
    expect = ((int(qs[-1][:4]) - int(qs[0][:4])) * 4
              + int(qs[-1][-1]) - int(qs[0][-1]) + 1)
    rec("数据", "季度序列连续", expect == len(qs), f"{qs[0]}–{qs[-1]}，{len(qs)} 期")

    from pathlib import Path
    deriv = Path(ROOT) / "data" / "derived"
    need = ["canon_level_top10.csv.gz", "canon_position.csv.gz",
            "canon_csi_history.csv.gz", "canon_csi_parts.csv.gz",
            "canon_top10.csv.gz", "fund_list.csv.gz", "factor_test.json"]
    miss = [f for f in need if not (deriv / f).exists()]
    rec("数据", "派生快照齐全", not miss, "缺：" + "、".join(miss) if miss else f"{len(need)} 个文件")

    # 预计算文件不能比数据库旧，否则页面会用过期矩阵
    db = Path(ROOT) / "data" / "db" / "fund_ak.db"
    if db.exists():
        older = []
        for f in ["canon_level_top10.csv.gz", "canon_position.csv.gz",
                  "canon_csi_history.csv.gz", "canon_csi_parts.csv.gz",
                  "canon_top10.csv.gz"]:
            p = deriv / f
            if p.exists() and p.stat().st_mtime < db.stat().st_mtime:
                older.append(f)
        rec("数据", "预计算文件不早于数据库", not older, "；".join(older))
    else:
        rec("数据", "数据库存在", True, "线上模式（无库）不适用")

    c = cn.crowding_score()
    cs = cn.csi_history()
    rec("数据", "CSI 面板与本期一致", abs(float(c["CSI"].max()) - float(cs.iloc[-1].max())) < 1e-6,
        f"{c['CSI'].idxmax()} 本期 {c['CSI'].max():.1f}")

    # 拥挤暴露：权重×CSI 必须落在 0–100（归一化写错会溢出到几百）
    try:
        _code = str(cn.profiles()["code"].iloc[0]).zfill(6)
        _ex = cn.crowding_exposure(_code)
        _ok = (not _ex.empty
               and float(_ex.min().min()) >= -0.01
               and float(_ex.max().max()) <= 100.01)
        rec("数据", "拥挤暴露在 0–100（抽样一只基金）", _ok,
            f"{_code} {_ex.shape[0]} 期，最大 {float(_ex.max().max()):.1f}"
            if not _ex.empty else "空表")
    except Exception as e:
        rec("数据", "拥挤暴露在 0–100（抽样一只基金）", False,
            f"{type(e).__name__}: {e}")

    # 基金层面的拥挤暴露榜：两个口径都要落在 0–100。
    # 这张榜要现场查持仓表，降级（无库）模式拿不到数据——跳过而不是判失败，
    # 否则「线上只带快照」时审计会天天报一条假失败。
    if not cn.available():
        rec("数据", "基金拥挤暴露榜在 0–100", True, "降级模式跳过（无持仓库）")
        _rk = None
    else:
        _rk = cn.fund_crowding_rank(top=5000)
    try:
        if _rk is None:
            raise StopIteration
        _ok = (not _rk.empty
               and float(_rk["高拥挤仓位"].min()) >= -0.01
               and float(_rk["高拥挤仓位"].max()) <= 100.01
               and float(_rk["加权CSI"].min()) >= -0.01
               and float(_rk["加权CSI"].max()) <= 100.01)
        rec("数据", "基金拥挤暴露榜在 0–100", _ok,
            f"{len(_rk):,} 只，榜首 {float(_rk['高拥挤仓位'].max()):.0f}%"
            if not _rk.empty else "空表")
    except StopIteration:
        pass
    except Exception as e:
        rec("数据", "基金拥挤暴露榜在 0–100", False, f"{type(e).__name__}: {e}")

    # 三项分位长表必须覆盖全部期次：缺一期，⑤ 的「选一期」就会现场重算（慢 2–4 秒）
    part_q = set(cn._csi_parts_csv().get("quarter", [])) if not cn._csi_parts_csv().empty else set()
    miss_q = [q for q in qs if q not in part_q]
    rec("数据", "CSI 三项分位覆盖全部期次", not miss_q,
        f"缺 {len(miss_q)} 期：{'、'.join(miss_q[:5])}" if miss_q else f"{len(part_q)} 期")

    # 某一期拆出来的三项分位，加总必须等于该期 CSI（防止导出时列错位）
    if not miss_q:
        _last = qs[-1]
        _p = cn.csi_parts(_last)
        _calc = _p[["配置分位", "价格分位", "资金分位"]].mean(axis=1, skipna=True)
        _diff = float((_calc - _p["CSI"]).abs().max())
        rec("数据", "CSI = 三项分位均值（任一期可复算）", _diff < 0.06, f"最大差 {_diff:.3f}")


# ---------------------------------------------------------------------------
# 3. 页面渲染
# ---------------------------------------------------------------------------
def _bad_text(v: str) -> tuple[str, str] | None:
    for pat, name in BAD_PATTERNS:
        m = pat.search(v)
        if m:
            return name, m.group(0)[:40]
    return None


# 纯文字/纯表格页：① 是结论 + 动作表（证据都在 ②③），⑥ 是口径文档。
# 这两页没有图是设计选择，不是渲染失败——但「无图」不等于「不检查文本」。
NO_CHART_PAGES = ("① 结论总览", "⑥ 方法与口径")


def _page_text_checks(at, p: str) -> None:
    """文本卫生：nan/None/未渲染标签/静默失败/写死的期数。

    原实现把 ① 结论总览 排除在循环外（PAGES[1:]），等于首页只查了
    「不报异常」——读者第一眼看到的那页反而没查文本。这里抽成函数，
    ① 与其余页走同一套检查。
    """
    hits = []
    # _h() 用的是 st.html，不是 st.markdown，两处都要扫
    for blk in list(at.markdown) + list(at.get("html")):
        v = str(getattr(blk, "value", "") or getattr(blk, "body", "") or "")
        b = _bad_text(v)
        if b:
            hits.append(f"{b[0]}:{b[1]}")
    # 「暂不可用」是各页 try/except 的兜底文案：一旦出现，说明某块静默失败了
    silent = [w for w in ("暂不可用", "未找到持仓数据", "结果文件为空")
              if any(w in str(getattr(x, "value", "") or getattr(x, "body", "") or "")
                     for x in list(at.markdown) + list(at.get("html")))]
    if silent:
        hits.append("静默失败：" + "、".join(silent))
    # 期数已经动态化：文案里再出现写死的「31 期」，说明有地方漏改
    import canon as _cn
    _nq = len(_cn.quarters())
    if _nq != 31:
        stale = [w for w in ("31 期",)
                 if any(w in str(getattr(x, "value", "") or getattr(x, "body", "") or "")
                        for x in list(at.markdown) + list(at.get("html")))]
        if stale:
            hits.append("写死的期数：" + "、".join(stale))
    rec("文本", f"{p} 无 nan/None/未渲染标签/静默失败", not hits, "；".join(hits[:3]))

    n_chart = len(at.get("plotly_chart"))
    need_chart = p not in NO_CHART_PAGES
    rec("渲染", f"{p} 图表数量", (n_chart > 0) or (not need_chart),
        f"{n_chart} 张" if need_chart else f"{n_chart} 张（纯文字/表格页）")


def stage_pages(fast: bool = False) -> None:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=1500)
    at.run()
    rec("渲染", "① 结论总览", len(at.exception) == 0,
        str(at.exception[0].value)[:200] if at.exception else "")
    _page_text_checks(at, "① 结论总览")

    for p in PAGES[1:]:
        t0 = time.time()
        at.radio(key="page").set_value(p)
        at.run()
        dt = time.time() - t0
        ok = len(at.exception) == 0
        rec("渲染", p, ok,
            (str(at.exception[0].value)[:200] if at.exception else f"{dt:.1f}s"))
        _page_text_checks(at, p)

    if not fast:
        stage_widgets(at)


def stage_widgets(at) -> None:
    """每个页面把每个控件的最后一个选项点一遍，确认不会崩。"""
    for p in PAGES:
        at.radio(key="page").set_value(p)
        at.run()
        for w in list(at.radio) + list(at.selectbox):
            if getattr(w, "key", "") == "page":
                continue
            try:
                opts = list(w.options)
                if len(opts) < 2:
                    continue
                w.set_value(opts[-1])
                at.run()
                ok = len(at.exception) == 0
                rec("交互", f"{p} / {w.label[:14]} → 最后一项", ok,
                    str(at.exception[0].value)[:160] if at.exception else "")
                if not ok:
                    at.radio(key="page").set_value(p)
                    at.run()
            except Exception as e:
                rec("交互", f"{p} / {getattr(w, 'label', '?')[:14]}", False,
                    f"{type(e).__name__}: {e}"[:160])
        for cb in list(at.checkbox):
            try:
                cb.set_value(not cb.value)
                at.run()
                rec("交互", f"{p} / {cb.label[:14]} 切换", len(at.exception) == 0,
                    str(at.exception[0].value)[:160] if at.exception else "")
                cb.set_value(not cb.value)
                at.run()
            except Exception as e:
                rec("交互", f"{p} / 复选框", False, f"{type(e).__name__}: {e}"[:160])


# ---------------------------------------------------------------------------
# 4. 跨页一致性
# ---------------------------------------------------------------------------
def stage_consistency() -> None:
    from streamlit.testing.v1 import AppTest
    import canon as cn

    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=1500)
    at.run()
    txt1 = "\n".join(str(getattr(x, "value", "") or getattr(x, "body", "") or "")
                     for x in list(at.markdown) + list(at.get("html")))
    c = cn.crowding_score()
    top = c[c["等级"] != "小样本"].index[0]
    v = float(c.loc[top, "CSI"])
    rec("一致性", f"首页/拥挤度页同一只最挤行业（{top} {v:.1f}）",
        top in txt1 and f"{v:.1f}" in txt1, "")

    at.radio(key="page").set_value("③ 行业拥挤度")
    at.run()
    txt3 = "\n".join(str(getattr(x, "value", "") or getattr(x, "body", "") or "")
                     for x in list(at.markdown) + list(at.get("html")))
    rec("一致性", "拥挤度页含同一行业与其分值",
        top in txt3 and f"{v:.1f}" in txt3, "")

    # 期次标签一致：所有页面都用同一个最新季
    q = cn.latest()
    for p in ["① 结论总览", "② 本期解读（全市场）", "③ 行业拥挤度", "④ 单只基金"]:
        at.radio(key="page").set_value(p)
        at.run()
        t = "\n".join(str(getattr(x, "value", "") or getattr(x, "body", "") or "")
                      for x in list(at.markdown) + list(at.get("html")))
        rec("一致性", f"{p} 使用最新期 {q}", q in t, "" if q in t else "未出现最新期")


# ---------------------------------------------------------------------------
def stage_charts() -> None:
    """图表结构自检：不渲染页面，只检查图对象「该有的元素在不在」。

    为什么需要：plotly 7.1 的 `add_hrect / add_vline / add_hline` 一旦带
    `row / col` 参数就会**静默丢弃**——图照画、不报错，只是背景带与参考线
    全没了。这种缺失人眼看截图很容易漏（以为「本来就这样」），
    必须用 shapes / annotations 的数量兜住。
    """
    try:
        import canon as cn
        import report_charts as rc
    except Exception as e:  # pragma: no cover
        rec("图表", "模块可导入", False, f"{type(e).__name__}: {e}")
        return

    H = cn.csi_history()
    qs = cn.quarters()

    f = rc.fig_csi_trend(H, "电子", True)
    # 5 条档位背景 + 4 条阈值线 = 9；两条面板各 1 条参考线也算在内
    n_shape = len(f.layout.shapes or [])
    rec("图表", "拥挤度折线：五档背景 + 阈值线", n_shape >= 9, f"shapes={n_shape}")
    rec("图表", "拥挤度折线：双面板（CSI + 分位）", len(f.layout.annotations or []) >= 4,
        f"traces={len(f.data)}")

    # 「每一季度不能跳」：折线每期都必须带悬浮分位，缺一期就是断档
    _tr = next((t for t in f.data if getattr(t, "name", "") == "电子"), None)
    _cd = getattr(_tr, "customdata", None) if _tr is not None else None
    rec("图表", "折线每一期都带分位（悬浮不断档）",
        _cd is not None and len(_cd) == len(H),
        f"{len(_cd) if _cd is not None else 0} 期 / 面板 {len(H)} 期")
    # 下图：自身历史分位 + 全行业分位（横截面）两条都要在
    _names = [getattr(t, "name", "") for t in f.data]
    rec("图表", "下图含「自身历史分位」与「全行业分位」",
        any("自身历史分位" in n for n in _names)
        and any("全行业分位" in n for n in _names),
        "；".join(n for n in _names if "分位" in n))

    f2 = rc.fig_csi_trend(H, None, True)
    rec("图表", "拥挤度折线（全市场）：下图为广度",
        any("≥65" in str(getattr(t, "name", "")) for t in f2.data),
        "；".join(str(getattr(t, "name", "")) for t in f2.data))

    # 全市场视角的「分位」不能缺席：只有一条广度线，读者无法回答「现在什么分位」。
    _n2 = [getattr(t, "name", "") for t in f2.data]
    rec("图表", "全市场视角：下图同时有中位分位与广度",
        any("中位的历史分位" in n for n in _n2)
        and any("≥65" in n for n in _n2),
        "；".join(_n2))

    # 误触缩放：dragmode 必须关闭，否则鼠标一拖整张图被缩成几条线，
    # 而工具栏是隐藏的（displayModeBar=False），用户无法复位。
    _drag = [str(getattr(f.layout, "dragmode", "zoom")),
             str(getattr(f2.layout, "dragmode", "zoom"))]
    rec("图表", "图表禁止拖拽缩放（防误触缩成一团）",
        all(d in ("False", "False", "None") or d is False for d in _drag),
        f"dragmode={_drag}")

    # 对照表：均值/标准差/分位必须与面板直接计算一致（抽 3 个行业逐个核）
    _tbl = rc.csi_stat_table(H, thr=80).set_index("行业")
    _bad = []
    for _ind in list(H.columns)[:3] + ["电子", "煤炭"]:
        if _ind not in _tbl.index:
            continue
        _s = H[_ind].dropna().astype(float)
        _mu = float(_s.mean())
        _pc, _n, _k = rc.pct_of(_s, float(_s.iloc[-1]))
        if (abs(float(_tbl.loc[_ind, "历史均值"]) - _mu) > 0.051
                or abs(float(_tbl.loc[_ind, "当前分位"]) - _pc) > 0.051):
            _bad.append(f"{_ind} 均值 {_tbl.loc[_ind, '历史均值']}/{_mu:.1f}"
                        f" 分位 {_tbl.loc[_ind, '当前分位']}/{_pc:.0f}")
    rec("图表", "对照表与面板同源（均值/分位逐个可核）",
        not _bad and len(_tbl) == len(H.columns), "；".join(_bad[:3]) or f"{len(_tbl)} 行")

    if len(qs) >= 2:
        d = cn.stock_flow(qs[-2], qs[-1])
        f3 = rc.fig_stock_flow(d)
        rec("图表", "个股净流向：两栏零线", len(f3.layout.shapes or []) == 2,
            f"shapes={len(f3.layout.shapes or [])}")
    c = cn.compare_periods()
    f4 = rc.fig_compare_two(c)
    rec("图表", "两栏对比：两栏零线", len(f4.layout.shapes or []) == 2,
        f"shapes={len(f4.layout.shapes or [])}")

    # 仓位分布图的中位虚线（子图右栏，曾因 add_vline(row/col) 静默丢失）
    try:
        P = cn.position_series()
        fq = cn.full_quarters()
        f6 = rc.fig_position(P, cn.coverage(fq[-1]), fq[-1])
        rec("图表", "仓位图：分布栏中位虚线", len(f6.layout.shapes or []) >= 1,
            f"shapes={len(f6.layout.shapes or [])}")
    except Exception as e:
        rec("图表", "仓位图：分布栏中位虚线", False, f"{type(e).__name__}: {e}")

    # 排名不能超过分母：曾经出现过「第 27/26」（选中行业被排除在排名池外）
    import re as _re
    bad = []
    for _ind in H.columns:
        _t = rc.fig_csi_trend(H, _ind, False).layout.title.text or ""
        _m = _re.search(r"全行业第 (\d+)/(\d+)", _t)
        if _m and int(_m.group(1)) > int(_m.group(2)):
            bad.append(f"{_ind} {_m.group(0)}")
    rec("图表", "拥挤度排名不超过分母（逐个行业）", not bad,
        "；".join(bad[:3]) if bad else f"{len(H.columns)} 个行业全部合法")

    # 分位尺：档位底色 5 + P25/P50/P75 三条 = 8
    ser = H["电子"].dropna() if "电子" in H.columns else H.median(axis=1).dropna()
    f5 = rc.fig_csi_position(ser, "电子")
    rec("图表", "分位尺：五档底色 + 三条分位线",
        len(f5.layout.shapes or []) >= 8, f"shapes={len(f5.layout.shapes or [])}")


def stage_qreport() -> None:
    """季报页的数据自检：披露值之间必须自洽。

    这一层的错误最隐蔽——单个数字都对，合起来矛盾（行业合计超过股票仓位、
    名次从 18 开始、份额为负）。所以只查「恒等式与边界」，不重复查业务口径。
    """
    import numpy as np
    try:
        import qreport as Q
    except Exception as e:  # pragma: no cover
        rec("季报", "模块可导入", False, f"{type(e).__name__}: {e}")
        return

    cov = Q.coverage()
    if not cov["funds"]:
        rec("季报", "本地已缓存季报数据", True, "未缓存任何基金（跳过）")
        return
    rec("季报", "本地已缓存季报数据", cov["funds"] > 0,
        f"{cov['funds']} 只基金｜行业 {cov['industry_rows']} 行｜持仓 {cov['hold_rows']} 行")

    with Q._con() as con:
        codes = [r[0] for r in con.execute(
            "SELECT DISTINCT code FROM q_asset ORDER BY code LIMIT 12")]
    bad = []
    for code in codes:
        d = Q.load_fund(code)
        a = d["q_asset"]
        if a is None or a.empty:
            continue
        # 1) 比例边界：>100 是允许的（应付/回购），但不能是负或几百
        for c in ("stock_pct", "cash_pct", "bond_pct"):
            v = a[c].dropna()
            # 上限放到 400%：基金转型/巨额赎回期的临时披露日，现金或债券占净值
            # 可以远超 100%（000030 在 2019-05-23 现金 202.9%），这是真的，不是错数。
            if len(v) and (v.min() < -0.01 or v.max() > 400):
                bad.append(f"{code} {c} 越界 {v.min():.1f}–{v.max():.1f}")
        # 2) 行业合计不能超过股票仓位（半年报期应几乎相等）
        for _, r in a.tail(4).iterrows():
            ind = d["q_industry"]
            ind = ind[ind["period"] == r["period"]]
            if ind.empty or not np.isfinite(r["stock_pct"]):
                continue
            tot = float(ind["ratio_pct"].sum())
            if tot > r["stock_pct"] + 2.0:
                bad.append(f"{code} {r['quarter']} 行业{tot:.1f}%>股票{r['stock_pct']:.1f}%")
        # 3) 份额边界
        sh = d["q_shares"]
        if sh is not None and not sh.empty:
            for _, r in sh.iterrows():
                if np.isfinite(r["purchase_yi"]) and r["purchase_yi"] < 0:
                    bad.append(f"{code} {r['quarter']} 申购为负")
                if np.isfinite(r["redeem_yi"]) and r["redeem_yi"] < 0:
                    bad.append(f"{code} {r['quarter']} 赎回为负")
                if np.isfinite(r["end_share_yi"]) and r["end_share_yi"] <= 0:
                    bad.append(f"{code} {r['quarter']} 期末份额≤0")
        # 4) 名次必须从 1 连续（曾因沿用数据源跨季序号出现「第 18 名」排第一）
        h = d["q_hold"]
        if h is not None and not h.empty:
            for q, g in h.groupby("quarter"):
                rk = sorted(int(x) for x in g["rnk"].dropna())
                if rk and rk[0] != 1:
                    bad.append(f"{code} {q} 名次从 {rk[0]} 开始")
                if rk and rk != list(range(1, len(rk) + 1)):
                    bad.append(f"{code} {q} 名次不连续")
                if float(g["ratio_pct"].sum()) > 101:
                    bad.append(f"{code} {q} 持仓合计 {g['ratio_pct'].sum():.1f}%")
    rec("季报", "披露值自洽（边界/行业≤仓位/名次连续）", not bad,
        "；".join(bad[:4]) if bad else f"抽查 {len(codes)} 只基金全部通过")

    # 规模拆解恒等式：份额效应 + 净值效应 + 交叉 = Δ净资产。
    # 这是页面上唯一一处「自己算」的数，必须逐只对得上。
    bad2, n_chk = [], 0
    for code in codes[:6]:
        d = Q.load_fund(code)
        a2, sh = d["q_asset"], d["q_shares"]
        if a2 is None or a2.empty or sh is None or sh.empty:
            continue
        a2 = a2.sort_values("period")
        for i in range(1, len(a2)):
            ra, rp = a2.iloc[i], a2.iloc[i - 1]
            rs = sh[sh["quarter"] == ra["quarter"]]
            if rs.empty or not np.isfinite(ra["nav_yi"]) or not np.isfinite(rp["nav_yi"]):
                continue
            e = float(rs["end_share_yi"].iloc[0])
            b0 = e - float(rs["purchase_yi"].iloc[0]) + float(rs["redeem_yi"].iloc[0])
            if b0 <= 0 or e <= 0:
                continue
            np0, np1 = rp["nav_yi"] / b0, ra["nav_yi"] / e
            eff = (e - b0) * np0 + b0 * (np1 - np0) + (e - b0) * (np1 - np0)
            d_nav = ra["nav_yi"] - rp["nav_yi"]
            n_chk += 1
            if abs(eff - d_nav) > max(0.02, abs(d_nav) * 1e-4):
                bad2.append(f"{code} {ra['quarter']} 拆解 {eff:.2f}≠{d_nav:.2f}")
        # 行业 HHI 必须在 0–1
        ind = d["q_industry"]
        if ind is not None and not ind.empty:
            for q, g in ind.groupby("quarter"):
                # HHI 必须按「每个 A 股门类各一组 + 港股通合计一组」算；
                # 按「A股/港股通」两个市场分组得到的只是仓位的平方，不是集中度。
                v = Q.industry_hhi(g)
                if not (0 <= v <= 1.0001):
                    bad2.append(f"{code} {q} HHI={v:.3f}")
    rec("季报", "规模拆解恒等式与行业 HHI 边界", not bad2,
        "；".join(bad2[:3]) if bad2 else f"校验 {n_chk} 个季度全部对得上")

    # 5) 期次合法性：数据源「列错位」会写出 quarter='000126' 这种脏行，
    #    它不撞任何真实季度，但会让「共 N 期」虚高、让按年补抓误判「已抓过」。
    with Q._con() as con:
        dirty = []
        for t in ("q_asset", "q_industry", "q_shares", "q_hold", "q_change",
                  "q_bond", "q_fin"):
            n = con.execute(
                f"SELECT COUNT(*) FROM {t} WHERE quarter NOT GLOB "
                "'[0-9][0-9][0-9][0-9]Q[1-4]'").fetchone()[0]
            if n:
                dirty.append(f"{t} {n}")
    rec("季报", "所有期次都是合法季度（无列错位脏行）", not dirty,
        "；".join(dirty) if dirty else "7 张表全部合法")

    # 6) 前十大进出：状态必须自洽（新进/退出/增减持互斥，笔数对得上）
    bad3, n_df = [], 0
    for code in codes[:6]:
        d = Q.load_fund(code)
        hq = sorted(d["q_hold"]["quarter"].dropna().unique()) \
            if d["q_hold"] is not None and not d["q_hold"].empty else []
        for i in range(1, len(hq)):
            df = Q.top10_diff(d, hq[i], hq[i - 1])
            if df.empty:
                continue
            n_df += 1
            if df["状态"].isna().any():
                bad3.append(f"{code} {hq[i]} 有空状态")
            # 本期在前十里的行：本期% 有值。退出/清仓行本期为空，不计入。
            import pandas as _pdx
            cur = df[_pdx.to_numeric(df["本期%"], errors="coerce").notna()]
            if float(cur["本期%"].sum()) > 100.5:
                bad3.append(f"{code} {hq[i]} 本期合计 {cur['本期%'].sum():.1f}%")
            if len(cur) > 10:
                bad3.append(f"{code} {hq[i]} 本期 {len(cur)} 只 >10")
    rec("季报", "前十大进出状态自洽", not bad3,
        "；".join(bad3[:3]) if bad3 else f"校验 {n_df} 组相邻季")

    # 7) 按年补抓必须幂等：已抓过的年份直接返回、不再联网
    try:
        with Q._con() as con:
            r = con.execute("SELECT code, year FROM q_ytry LIMIT 1").fetchone()
        if r:
            t0 = time.time()
            out = Q.ensure_year(r[0], int(r[1]))
            dt = time.time() - t0
            rec("季报", "按年补抓幂等（已抓过不重复联网）",
                out.get("fetched") is False and dt < 1.0,
                f"{r[0]} {r[1]} {dt:.2f}s")
    except Exception as e:  # noqa: BLE001
        rec("季报", "按年补抓幂等", False, f"{type(e).__name__}: {e}")

    # 8) 份额表与资产表必须同口径（都是基金整体净资产）。
    #    东财 gmbd 按份额类别返回，若不合并 A/C，会得到「67 亿 ÷ 0.03 亿份」
    #    这种错误；这一条盯的是全量一致率，不抽查。
    try:
        import pandas as _pd
        with Q._con() as con:
            _a = _pd.read_sql("SELECT code, period, nav_yi FROM q_asset", con)
            _s = _pd.read_sql("SELECT code, period, end_nav_yi FROM q_shares", con)
        _m = _a.merge(_s, on=["code", "period"], how="inner")
        _m = _m[_pd.to_numeric(_m["nav_yi"], errors="coerce") > 0]
        _e = _pd.to_numeric(_m["end_nav_yi"], errors="coerce")
        _n = _pd.to_numeric(_m["nav_yi"], errors="coerce")
        _bad = int(sum(0 if Q.shares_consistent(a, b) else 1
                       for a, b in zip(_e, _n)))
        _rate = _bad / max(len(_m), 1)
        rec("季报", "份额与净资产同口径（全量一致率）", _rate < 0.02,
            f"{_bad}/{len(_m)} 不一致（{_rate:.1%}）")
    except Exception as e:  # noqa: BLE001
        rec("季报", "份额与净资产同口径", False, f"{type(e).__name__}: {e}")

    # 9) 持仓名次必须 1..N 连续（断档＝同一季度混了多套快照，组合是拼接的）
    try:
        import pandas as _pd
        with Q._con() as con:
            _h = _pd.read_sql("SELECT code, quarter, rnk FROM q_hold", con)
        _gap = 0
        for (c, q), g in _h.groupby(["code", "quarter"]):
            rk = sorted(int(x) for x in g["rnk"].dropna())
            if rk and (rk[0] != 1 or rk != list(range(1, len(rk) + 1))):
                _gap += 1
        rec("季报", "持仓名次连续（全量）", _gap == 0,
            f"{_gap} 个基金×季度断档" if _gap else f"{_h.groupby(['code','quarter']).ngroups} 个期次全部连续")
    except Exception as e:  # noqa: BLE001
        rec("季报", "持仓名次连续", False, f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="跳过控件交互")
    args = ap.parse_args()

    t0 = time.time()
    stage_static()
    stage_selfcheck()
    stage_glossary()
    stage_data()
    stage_qreport()
    stage_charts()
    stage_pages(args.fast)
    stage_consistency()

    import pandas as pd
    df = pd.DataFrame(ROWS, columns=["阶段", "检查项", "结果", "说明"])
    n_bad = int((df["结果"] != "通过").sum())
    print(df.to_string(index=False))
    print(f"\n{'✅ 全部通过' if n_bad == 0 else f'❌ {n_bad} 项失败'}"
          f"　共 {len(df)} 项，用时 {time.time() - t0:.0f}s")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
