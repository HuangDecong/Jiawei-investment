# -*- coding: utf-8 -*-
"""
add_chart_insights.py —— 按**行号**在每张图后插入洞察卡片（避免字符串锚点撞车）

为什么不用简单字符串替换
------------------------
页面里 `_f2` / `_f3` / `_f4` 等变量名在多个页面被复用，
`str.replace(old, new, 1)` 会命中**第一次出现**的位置，
结果洞察卡被插到别的页面（本轮确实踩过这个坑，导致 IndentationError）。
本脚本改为：先用 AST 取出所有 `st.plotly_chart(...)` 调用的**行号**，
再按行号从下往上插入，插入位置由「图变量名 + 出现次序」唯一确定。

用法：
  python tools/add_chart_insights.py --list          # 列出每张图的位置与标题
  python tools/add_chart_insights.py --apply         # 按内置映射插入
"""
from __future__ import annotations

import ast
import os
import textwrap
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
APP = os.path.join(ROOT, "app.py")

# 图变量名 -> [(出现次序, 洞察代码)]；次序从 0 开始
# 洞察代码在 _INSIGHTS 中定义（生成要插入的 Python 语句文本）
PLAN: dict[int, str] = {
    # 行号 -> 洞察代码（行号取自 tools/add_chart_insights.py --list）
    1089: "etf_cum",      # 页面④ ETF 累计资金流
    1110: "etf_day",      # 页面④ 单日净申赎
    1194: "ic_ts",        # 页面⑤ 因子 IC 时序
    1213: "corr",         # 页面⑤ 因子相关性热力图
    1239: "decay",        # 页面⑤ IC 衰减
    1279: "bubble",       # 页面⑤ 拥挤度气泡图
    1298: "stress",       # 页面⑤ 压力测试
    1495: "rot_net",      # 页面⑥ 策略净值
    1521: "layer",        # 页面⑥ 分层回测
    1624: "wf",           # 页面⑦ walk-forward
    1656: "win",          # 页面⑦ 设计窗与状态
    1696: "dual",         # 页面⑦ 双因子
    1767: "attr",         # 页面⑦ 超额归因
    1913: "holdings",     # 页面⑧ 持仓序列体检
}

_INSIGHTS = {
    "etf_cum": """
    _es = _edf.groupby("代码")["累计净申赎亿元"].last().dropna()
    _ins(_es, "ETF 累计净申赎", unit="亿元", pct=False,
         title="① 分析与结论 · ETF 资金流",
         higher_means=("各只 ETF 的累计净申赎在区间内总体{direction}（最大 {last}，{q}）。"
                       "**这是「快层」信息**：ETF 资金流是日频增量，季报持仓是季频存量，"
                       "两者相差一个季度；方向不一致时不是矛盾，而是时间维度不同。"))
""",
    "etf_day": """
    _ins(None, title="② 分析与结论 · 单日净申赎",
         feature=("柱状图给出日度申赎的方向与量级。份额折算日（拆分/合并）已用"
                  "「累计单位净值 ÷ 单位净值」法识别并剔除——"
                  "**否则折算会被误计为巨额申购**"
                  "（实测 512480 折算日会凭空多出 +116.68 亿元、515880 多出 +248.19 亿元）。"))
""",
    "ic_ts": """
    _icm = _BTR.get("ic")
    if isinstance(_icm, pd.DataFrame) and not _icm.empty and "因子" in _icm.columns:
        _bk = _icm["ic_ir_t1"].abs().sort_values(ascending=False).index[0]
        _bo = _icm.loc[_bk]
        _ins(None, title="① 分析与结论 · IC 时序与滚动 IC_IR",
             feature=(f"候选因子中 |IC_IR| 最高的是 <b>{_bo['因子']}</b>"
                      f"（IC_IR {_bo['ic_ir_t1']:+.3f}、RankIC {_bo['ic_mean_t1']:+.4f}、"
                      f"NW p {_bo['p_nw_t1']:.4f}）。"
                      f"曲线穿越 0 轴的频次即「因子有效性时变」的直接证据；"
                      f"若某段长期为负，说明该因子在该时期反向。"))
""",
    "corr": """
    _ins(None, title="③ 分析与结论 · 因子相关性",
         feature=("热力图给出因子两两的逐期截面 Spearman 平均相关。"
                  "**同族因子相关性越高，合成时的分散化收益越小**；"
                  "相关性抬升还意味着「大家都在用同一类信号」——"
                  "这正是 FCI 因子拥挤指数中「因子相关性」分量的来源。"))
""",
    "decay": """
    _ins(None, title="④ 分析与结论 · IC 衰减",
         feature=("对比滚动 4 / 8 / 12 季窗口的 IC 均值：短窗（4 季）灵敏但噪声大，"
                  "长窗（12 季）稳定但滞后。**若长窗均值明显低于短窗，"
                  "说明因子近期在增强；反之则在衰减。**"
                  "本样本仅 9 期，三个窗口的差异会明显小于长历史下的差异。"))
""",
    "bubble": """
    _ins(None, title="② 分析与结论 · 拥挤度气泡图",
         feature=("气泡大小＝沪深300 基准权重。右上象限（高配置 + 高 Z）＝真拥挤；"
                  "**左侧大泡＝「基准权重大但机构低配」**，是潜在的低配风险敞口；"
                  "右下＝相对自身历史极重但绝对配置仍低，属口径背离区。"))
""",
    "stress": """
    _ins(None, title="③ 分析与结论 · 压力测试",
         feature=("本表直接给出「单一行业回撤对组合的拖累」——"
                  "**它不依赖任何因子有效性**，是本模型最可落地的产出。"
                  "组合冲击 ≈ 该行业回撤幅度 × 其配置占比。"))
""",
    "rot_net": """
    _ins(_rt_tbl["超额_净"] / 100.0 if "超额_净" in _rt_tbl.columns else None,
         "策略净超额", pct=False, unit="",
         title="分析与结论 · 策略净值",
         higher_means=("策略净超额累计 {change}（最新 {last}，{q}）。"
                       "必须并列引用：对**等权行业**基准的正超额，"
                       "与对**市值加权沪深300**基准的结论可能相反，只报有利口径即为误导。"))
""",
    "layer": """
    _ins(None, title="分析与结论 · 分层回测",
         feature=("五层累积净值的单调性是因子截面区分度的直观检验。"
                  "**若层1→层5 不单调，说明因子与收益的关系非单调**，"
                  "此时「多空价差」的显著性不能直接解读为单调预测力。"))
""",
    "wf": """
    _ins(None, title="① 分析与结论 · Walk-forward",
         feature=(f"滚动样本外 IC 均值 {_ws['留出期均值']:+.4f}，"
                  f"NW t = {_ws['留出期NW_t']:+.3f}，p = {_ws['留出期p']:.4f}"
                  f"（{int(_ws['留出期数'])} 期）。"
                  f"**它检验的是「挑设定的流程」而非单个因子**："
                  f"若 p 不显著，则「流程可复制」这一点并未被证明——"
                  f"这与「某个事先说明的假设通过检验」是两件不同的事。"))
""",
    "win": """
    _ins(None, title="② 分析与结论 · 设计窗与状态分解",
         feature=("若样本外表现集中在某个**时间区块**或某种**市场状态**，"
                  "则结论的普适性必须打折。分红＝A 股习惯（红涨绿跌）。"
                  "**关键读法：看显著性是否只出现在单一区块。**"))
""",
    "dual": """
    _ins(None, title="④ 分析与结论 · 双因子组合",
         feature=("正交双因子的价值取决于两因子的**截面相关性**：相关越低，"
                  "分散化收益越大、合成 IC_IR 提升越明显；"
                  "**若合成后 IC_IR 反而低于单因子，说明正交化付出的代价"
                  "（增量信息被剔除）超过了分散化收益，此时不应合成。**"))
""",
    "attr": """
    _ins(None, title="⑥ 分析与结论 · 超额归因",
         feature=("逐期精确分解 Σ(w−bm)·r。**关键检验：剔除最大负贡献行业后"
                  "超额是否归零**——若归零，则超额本质是「低配某大牛行业」的 beta；"
                  "若仍显著为正，则更接近真正的行业选择能力。"))
""",
    "holdings": """
    _ins(_chk.set_index("季度")["重仓合计(亿元)"].dropna(), "重仓股总市值",
         unit="亿元", pct=False,
         title="③ 分析与结论 · 持仓序列体检",
         higher_means=("重仓股总市值的{direction}至 {last}（{q}）。"
                       "**必须分清「总市值变化」与「占比变化」**：总市值受基金规模与"
                       "股价双重影响，而配置比例（占比）才是本模型的输入；"
                       "占比对样本覆盖面跳升具免疫性。"))
""",
}


def _charts(src: str) -> list[tuple[int, str]]:
    """返回 [(行号, 图变量名)]，按行号升序。"""
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", None) != "plotly_chart":
            continue
        if not node.args:
            continue
        a = node.args[0]
        name = a.id if isinstance(a, ast.Name) else ast.unparse(a)
        out.append((node.lineno, name))
    return sorted(out)


def main() -> int:
    with open(APP, encoding="utf-8") as f:
        s = f.read()
    ch = _charts(s)
    if "--list" in sys.argv:
        for i, (ln, name) in enumerate(ch, 1):
            line = s.splitlines()[ln - 1].strip()
            print(f"{i:>2}. L{ln:<5} {name:<8} {line[:70]}")
        return 0

    valid = {ln for ln, _ in ch}
    # PLAN 的值是 _INSIGHTS 的键名，必须取出真正的代码文本
    plan_hits = [(ln, _INSIGHTS[k]) for ln, k in PLAN.items()
                 if ln in valid and k in _INSIGHTS]
    missing = [ln for ln in PLAN if ln not in valid]
    if missing:
        print(f"⚠️ 以下行号已不是 st.plotly_chart 调用（文件改动过？）：{missing}")
    print(f"匹配到 {len(plan_hits)} 处待插入（共 {len(PLAN)} 处计划）")

    lines = s.splitlines(keepends=True)
    # 从下往上插入，避免行号漂移
    for ln, code in sorted(plan_hits, key=lambda t: -t[0]):
        # 缩进必须跟随目标行（若目标行在 if/with 内，需 8 空格而非 4）
        raw = lines[ln - 1]
        indent = raw[:len(raw) - len(raw.lstrip())]
        # 洞察代码在 _INSIGHTS 里本身就带 4 空格缩进，必须先 dedent，
        # 否则会叠加成 8 空格 → IndentationError。
        body = textwrap.dedent(code.strip("\n"))
        block = "".join((indent + x if x.strip() else "") + "\n"
                        for x in body.split("\n"))
        lines.insert(ln, block)
    out = "".join(lines)
    ast.parse(out)
    if "--apply" in sys.argv:
        with open(APP, "w", encoding="utf-8") as f:
            f.write(out)
        print("✅ 已写入")
    else:
        print("（预演，未写入；加 --apply 生效）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
