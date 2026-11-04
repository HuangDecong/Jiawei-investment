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

PAGES = ["① 结论总览", "② 本期解读", "③ 行业拥挤度",
         "④ 单只基金", "⑤ 因子检验", "⑥ 方法与口径"]

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

    # 基金层面的拥挤暴露榜：两个口径都要落在 0–100
    try:
        _rk = cn.fund_crowding_rank(top=5000)
        _ok = (not _rk.empty
               and float(_rk["高拥挤仓位"].min()) >= -0.01
               and float(_rk["高拥挤仓位"].max()) <= 100.01
               and float(_rk["加权CSI"].min()) >= -0.01
               and float(_rk["加权CSI"].max()) <= 100.01)
        rec("数据", "基金拥挤暴露榜在 0–100", _ok,
            f"{len(_rk):,} 只，榜首 {float(_rk['高拥挤仓位'].max()):.0f}%"
            if not _rk.empty else "空表")
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


def stage_pages(fast: bool = False) -> None:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=1500)
    at.run()
    rec("渲染", "① 结论总览", len(at.exception) == 0,
        str(at.exception[0].value)[:200] if at.exception else "")

    for p in PAGES[1:]:
        t0 = time.time()
        at.radio(key="page").set_value(p)
        at.run()
        dt = time.time() - t0
        ok = len(at.exception) == 0
        rec("渲染", p, ok,
            (str(at.exception[0].value)[:200] if at.exception else f"{dt:.1f}s"))

        # 文本卫生：只看 HTML 块，不看 dataframe
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
        # ⑥ 方法与口径 是纯文字页，没有图表是正常的
        need_chart = p not in ("⑥ 方法与口径",)
        rec("渲染", f"{p} 图表数量", (n_chart > 0) or (not need_chart),
            f"{n_chart} 张" if need_chart else f"{n_chart} 张（纯文字页）")

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
    rec("一致性", f"首页/拥挤度页同一只最挤行业（{top} {v:.0f}）",
        top in txt1 and f"{v:.0f}" in txt1, "")

    at.radio(key="page").set_value("③ 行业拥挤度")
    at.run()
    txt3 = "\n".join(str(getattr(x, "value", "") or getattr(x, "body", "") or "")
                     for x in list(at.markdown) + list(at.get("html")))
    rec("一致性", "拥挤度页含同一行业与其分值",
        top in txt3 and f"{v:.0f}" in txt3, "")

    # 期次标签一致：所有页面都用同一个最新季
    q = cn.latest()
    for p in ["① 结论总览", "② 本期解读", "③ 行业拥挤度", "④ 单只基金"]:
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

    f2 = rc.fig_csi_trend(H, None, True)
    rec("图表", "拥挤度折线（全市场）：下图为广度",
        any("≥65" in str(getattr(t, "name", "")) for t in f2.data),
        "；".join(str(getattr(t, "name", "")) for t in f2.data))

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="跳过控件交互")
    args = ap.parse_args()

    t0 = time.time()
    stage_static()
    stage_selfcheck()
    stage_data()
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
