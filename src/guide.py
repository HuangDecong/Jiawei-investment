# -*- coding: utf-8 -*-
"""
guide.py —— 投资机会解读 · 操作指南 · 数据持续更新指南
================================================================================
定位：在"拥挤度模型"的既有结论（风险描述，而非收益预测）之上，给出
「如何据此判断投资机会」的可操作框架，以及看板的使用与数据更新方法。

写作纪律：
  1. 与 R8 的诚实结论严格一致——拥挤度因子在 36 期长历史下无稳定截面预测力，
     因此"投资机会"的判断是**风险回避 + 结构性机会 + 组合约束**，不是择时买卖；
  2. 所有阈值取自 src/config.py 真实实现；
  3. 所有"机会/风险"提示都附带"为什么"与"这只是一个方向性判断"的免责边界。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 内部小工具
# ---------------------------------------------------------------------------
def _fmt_pct(x) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "—"


def _lv_color(lv: str, C) -> str:
    key = f"level_{lv}"
    return getattr(C.COLOR, "get", None) and C.COLOR.get(key, C.COLOR["text"])


# ---------------------------------------------------------------------------
# 一、当前状态的投资机会解读（用于 ① 总览仪表盘，最醒目）
# ---------------------------------------------------------------------------
def opportunity_card(M: dict, IND: dict | None, C, LATEST: str) -> str:
    """基于当前市场汇总 M_SUM 生成「投资机会解读」卡（HTML）。"""
    extreme = M.get("extreme_industries", [])
    under = M.get("under_industries", [])
    top1 = M.get("top1", "—")
    top1_pct = M.get("top1_pct", 0.0)
    hhi = M.get("HHI", 0.0)
    hhi_pct = M.get("HHI_pct", 0.0)
    hhi_level = M.get("HHI_level", "—")
    state = M.get("state", "—")
    n_extreme = M.get("n_extreme", 0)
    n_under = M.get("n_under", 0)

    # 极值/低配行业标签
    extreme_txt = ("、".join(extreme[:4]) + ("等" if len(extreme) > 4 else "")) if extreme else "无"
    under_txt = ("、".join(under[:4]) + ("等" if len(under) > 4 else "")) if under else "无"

    # 判断要点（风险侧 + 机会侧）
    risk_lines = []
    if extreme:
        risk_lines.append(
            f"<b>回避/减配</b>：{extreme_txt} 已达「极度拥挤」（历史分位 ≥90 或配置倍数 &gt;2.5），"
            "尾部踩踏脆弱性最高，新增仓位应主动规避、存量仓位设好回撤止损。")
    else:
        risk_lines.append("<b>暂无极度拥挤行业</b>：当前无行业触发「极度拥挤」，系统性踩踏风险处于低位。")
    if n_extreme >= C.SIGNAL_N_EXTREME:
        risk_lines.append(
            f"<b>结构性警示</b>：极度拥挤行业达 {n_extreme} 个（≥{C.SIGNAL_N_EXTREME}），"
            "定价权集中度处于极值区，属「越涨越挤、越挤越脆」的正反馈状态。")
    if top1_pct >= 40:
        risk_lines.append(
            f"<b>单点风险</b>：{top1} 单一行业占 {top1_pct:.2f}%，其回撤 30% 即拖累组合约 "
            f"{top1_pct * 0.3:.1f} pct——这是最直接、可直接量化的压力测试结论。")

    opp_lines = []
    if under:
        opp_lines.append(
            f"<b>关注名单（低拥挤）</b>：{under_txt} 当前「显著低配」（历史分位 ≤10）。"
            "<b>注意</b>：低拥挤 ≠ 超卖反转（历史事件研究：超配首次突破 P90 后 12 个月超额收益"
            "中位数为正，低拥挤同样不等于必然反弹），需叠加「动量转正 + ETF 资金净流入」才升级为机会。")
    else:
        opp_lines.append("<b>暂无显著低配行业</b>：当前资金分布相对均衡，缺乏明显的「逆向布局」候选。")

    # 组合层面
    port_lines = []
    if hhi_level in ("高", "极高"):
        port_lines.append(
            f"<b>组合分散化</b>：HHI={hhi:.4f}（历史分位 {hhi_pct:.0f}%，等级「{hhi_level}」），"
            "已达高集中区间——组合层应主动再平衡、避免在单一赛道追加风险敞口。")
    else:
        port_lines.append(
            f"<b>组合层面</b>：HHI={hhi:.4f}（历史分位 {hhi_pct:.0f}%，等级「{hhi_level}」），"
            "集中度尚在可控区间，但需跟踪边际变化。")

    # 关于"机会"的本质说明（诚实边界）
    caveat = (
        "<b>关于「投资机会」的诚实边界</b>：本模型在 36 期长历史下，拥挤度因子"
        "无稳定的截面收益预测力（IC≈0）。因此上面的「机会/风险」是<b>风险维度的排序</b>"
        "——它回答的是「哪里脆弱、哪里相对安全、哪里定价权过度集中」，"
        "而<b>不是</b>「哪里会涨、哪里会跌」。把它当作<b>组合风险预算与集中度约束</b>的输入，"
        "而不是择时信号。")

    risk_html = "".join(f"<li>{x}</li>" for x in risk_lines)
    opp_html = "".join(f"<li>{x}</li>" for x in opp_lines)
    port_html = "".join(f"<li>{x}</li>" for x in port_lines)

    return f"""<div class="opp">
<div class="opp-h"><span class="opp-badge">投资机会解读</span>　当前状态 · {LATEST}</div>
<div class="opp-cols">
  <div class="opp-col opp-risk">
    <div class="opp-t">① 风险侧（回避 / 减配）</div>
    <ul>{risk_html}</ul>
  </div>
  <div class="opp-col opp-opp">
    <div class="opp-t">② 机会侧（关注 / 逆向布局）</div>
    <ul>{opp_html}</ul>
  </div>
  <div class="opp-col opp-port">
    <div class="opp-t">③ 组合侧（分散化 / 压力测试）</div>
    <ul>{port_html}</ul>
  </div>
</div>
<div class="opp-caveat">{caveat}</div>
</div>"""


# ---------------------------------------------------------------------------
# 二、投资机会判断框架（用于 ⑤ 因子研究页，与因子手册同页）
# ---------------------------------------------------------------------------
def render_investment_framework(C) -> str:
    """完整的「投资机会判断框架」——三层法，纯 HTML。"""
    return """
<div class="sec" style="margin-top:14px">投资机会判断框架（三层法 · 基于拥挤度模型）</div>
<div class="alert" style="border-color:#E8A13A">
<b>前提声明（决定框架的用法）</b>：36 期长历史 IC 检验显示，拥挤度因子<b>没有稳定的截面收益预测力</b>
（F2e +0.024 / F15e +0.003，均不显著）。因此本框架<b>不回答「会涨会跌」</b>，只回答
<b>「哪里脆弱、哪里相对安全、哪里需要约束」</b>——它是<b>风险预算工具</b>，不是择时工具。
</div>
<div class="opp-cols" style="margin-top:8px">
  <div class="opp-col opp-risk">
    <div class="opp-t">第一层 · 风险回避（减配 / 回避）</div>
    <ul>
      <li><b>极度拥挤</b>：历史分位 ≥90（F3）或配置倍数 &gt;2.5（F12）→ 回避，存量设止损</li>
      <li><b>三信号共振</b>：F10=3/3（持仓 + 交易 + 筹码同时极端）→ 强回避</li>
      <li><b>新增拥挤</b>：F15e 高（绝对超配高、但相对历史尚不极端）→ 最危险，警惕快速反转</li>
    </ul>
    <div class="opp-why">依据：这三类状态对应「定价权过度集中 + 踩踏脆弱性最高」，
    是尾部风险（而非收益）的直接来源。</div>
  </div>
  <div class="opp-col opp-opp">
    <div class="opp-t">第二层 · 机会识别（关注 / 试探）</div>
    <ul>
      <li><b>低拥挤</b>：显著低配（分位 ≤10）→ 列入关注名单</li>
      <li><b>拐点确认</b>：动量 Z（F4b）由负转正 → 资金边际转向</li>
      <li><b>快层佐证</b>：ETF 资金净流入（F14）→ 日频增量确认</li>
      <li>满足 <b>≥2/3</b>（低拥挤 + 动量转正 + 资金流入）→ 升级为「试探性机会」</li>
    </ul>
    <div class="opp-why">依据：低拥挤只说明「不脆弱」，不等于「会涨」；必须叠加
    资金边际转向的确认，避免把「无人问津」误读为「即将反转」。</div>
  </div>
  <div class="opp-col opp-port">
    <div class="opp-t">第三层 · 组合约束（分散化 / 压力测试）</div>
    <ul>
      <li><b>集中度红线</b>：HHI &gt;0.15 或 CR3 &gt;60% → 主动再平衡</li>
      <li><b>单点压测</b>：单一行业占比 ≥40% → 强制做回撤 30% 的拖累测算</li>
      <li><b>K 型分化</b>：极度拥挤 ≥3 且显著低配 ≥10 → 回避极端拥挤、向低拥挤分散</li>
    </ul>
    <div class="opp-why">依据：集中度是「系统性风险」的直接读数，与因子是否有效无关，
    这一步不依赖任何预测力、永远可执行。</div>
  </div>
</div>
<div class="alert" style="border-color:#DCE3EA">
<b>怎么落地成决策</b>：三层按顺序执行——先做「风险回避」（剔除极度拥挤/三信号共振的行业），
再在「机会识别」里挑出低拥挤且资金边际转向的行业做<b>小仓位试探</b>，最后用「组合约束」
控制单一赛道敞口。任何一层都不输出「满仓/清仓」信号，只输出<b>风险预算的加减</b>。
</div>"""


# ---------------------------------------------------------------------------
# 三、操作指南（怎么看这个看板）
# ---------------------------------------------------------------------------
def render_operation_guide() -> str:
    """看板操作指南：9 页各自看什么、结论怎么读。"""
    return """
<div class="sec" style="margin-top:14px">操作指南 · 9 页看板怎么看</div>
<div class="opp-cols">
  <div class="opp-col">
    <div class="opp-t">读结论（先看这三处）</div>
    <ul>
      <li><b>① 综合判断</b>：3 分钟把握全貌——市场状态、HHI 分位、极度拥挤行业数、FCI、以及 5 条带证据的判断理由。</li>
      <li><b>① 投资机会解读</b>：风险侧（回避谁）/ 机会侧（关注谁）/ 组合侧（怎么约束）三层结论。</li>
      <li><b>⑥ 轮动策略与因子裁决</b>：压力测试结论（可直接用于组合风控）。</li>
    </ul>
  </div>
  <div class="opp-col">
    <div class="opp-t">读数据（想深挖时看）</div>
    <ul>
      <li><b>② 拥挤度总览</b>：单行业钻取 + 31 行业拥挤等级明细 + 信号触发记录。</li>
      <li><b>③ 行业占比变迁</b>：堆叠面积图 / 热力图 / 瀑布图，看资金在行业间的「搬家路径」。</li>
      <li><b>④ ETF 资金流</b>：宽基 + 行业 ETF 的日度资金动向（快层信号）。</li>
    </ul>
  </div>
  <div class="opp-col">
    <div class="opp-t">读方法（想审计时看）</div>
    <ul>
      <li><b>⑤ 因子研究</b>：因子手册（六要素）+ 投资机会判断框架 + 全部 IC/分层/稳健性检验表。</li>
      <li><b>⑦ 稳健性检验</b>：样本内外、牛熊、参数敏感性、正交化、Bootstrap、多重检验。</li>
      <li><b>⑧ 数据质量与溯源</b>：口径断点检验、独立口径交叉验证、缺陷登记、数据溯源表。</li>
      <li><b>⑨ 方法说明与数据来源</b>：扩窗（无前视）口径、风险提示、更新指南。</li>
    </ul>
  </div>
</div>
<div class="alert" style="border-color:#DCE3EA">
<b>读图三原则</b>：① 先看「拐点」再看「水平」——拐点（环比转向）比绝对高低更有信息；② 占比类指标
（HHI/CRn/超配）跨期可比，绝对规模类指标只在同口径区间内比较；③ 任何单期极值都可能由口径或单期
驱动，结论要回到 ⑤⑦ 页的检验表与 ⑧ 页的溯源表交叉确认。
</div>"""


# ---------------------------------------------------------------------------
# 四、数据持续更新指南（怎么把数据更新到最新季度）
# ---------------------------------------------------------------------------
def render_update_guide() -> str:
    """数据持续更新指南：更新顺序、命令、校验要点。"""
    return """
<div class="sec" style="margin-top:14px">数据持续更新指南 · 每个季报期怎么做</div>
<div class="alert" style="border-color:#DCE3EA">
<b>更新频率</b>：公募基金季报在每季末后约 15 个工作日披露完毕（4/7/10/次年1 月中下旬）。
建议每个季报披露完成后更新一次，全流程约需 30~60 分钟。
</div>
<div class="opp-cols">
  <div class="opp-col">
    <div class="opp-t">第 1 步 · 抓取新季报（Wind）</div>
    <ul>
      <li><b>持仓</b>：主动权益三类（普通股票型/偏股混合型/灵活配置型）前十大重仓股，逐类型查询后相加；<b>必须</b>与「类型总额」交叉验证（gap 约 5~11%）。</li>
      <li><b>基准权重</b>：沪深300 成份股按申万一级行业权重，逐期校验<b>加总 = 100%</b>。</li>
      <li><b>行业收益 / 成交额</b>：申万一级 31 行业季度涨跌幅与成交额，校验 31 行业齐全。</li>
      <li>写入 <code>data/cache/holdings__{Q}.json</code>、<code>hs300__{Q}.json</code> 等。</li>
    </ul>
  </div>
  <div class="opp-col">
    <div class="opp-t">第 2 步 · 重建下游（命令）</div>
    <ul>
      <li><code>python tools/wind_fetch.py build</code> —— 重建超配比例、HHI/CR、因子面板。</li>
      <li><code>python src/update_pipeline.py</code> —— 重跑 IC/分层/稳健性检验，重建 <code>output/ic_report.xlsx</code> 等审计产物。</li>
      <li><code>python tools/seed_daily_raw.py</code> —— 刷新 ETF 日度资金流（如需）。</li>
      <li><code>python tools/data_quality_report.py</code> —— 刷新数据质量报告。</li>
    </ul>
  </div>
  <div class="opp-col">
    <div class="opp-t">第 3 步 · 校验与验收（必做）</div>
    <ul>
      <li><code>python tools/apptest_all_pages.py</code> —— 9 页 0 异常。</li>
      <li>preflight 硬约束 H1–H8 全通过、S1 告警为 0。</li>
      <li>④ 检查「口径断点」：占比类指标（HHI/CR）跨期可比，绝对规模只在同口径区间内比较。</li>
      <li>重启 streamlit 服务，浏览器 Ctrl+F5 强刷确认。</li>
    </ul>
  </div>
</div>
<div class="alert" style="border-color:#E8A13A">
<b>数据治理红线（宁可样本短、不可口径污染）</b>：① 任何一期数据都要过「三重校验」——
31 行业集合完整、总量环比平滑、HHI 落位合理，否则不摄入；② Wind 自然语言接口对
<b>年报期（Q4）与早年 Q1/Q3</b> 的持仓问句不稳定（会返回全市场市值或万元量纲），
这类响应一律不采信并登记根因；③ 数据源优先顺序：Wind（主力）→ 东方财富妙想 / 同花顺 iFinD
（交叉验证与补缺，需另行授权）。
</div>"""


# ---------------------------------------------------------------------------
# 五、各页通用「投资机会」短提示（用于 ②③④⑥ 页，轻量）
# ---------------------------------------------------------------------------
def opportunity_hint(M: dict, C, focus: str) -> str:
    """生成一行的投资机会提示（轻量卡片），focus 为该页的上下文行业/主题。"""
    extreme = M.get("extreme_industries", [])
    under = M.get("under_industries", [])
    extreme_txt = "、".join(extreme[:3]) if extreme else "无"
    under_txt = "、".join(under[:3]) if under else "无"
    return f"""<div class="opp opp-mini">
<span class="opp-badge">投资机会</span>　{focus}　|　
<b>回避</b>：{extreme_txt}（极度拥挤）　·　<b>关注</b>：{under_txt}（显著低配，需动量+资金确认）　·　
<b>原则</b>：本模型只做风险排序、不做收益预测，机会须叠加资金边际转向验证。</div>"""
