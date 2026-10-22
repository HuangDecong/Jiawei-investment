# -*- coding: utf-8 -*-
"""
metric_doc.py —— 指标释义库（定义 · 公式 · 经济内容 · 投资含义 · 读法 · 误读）
================================================================================
为什么单独成模块
----------------
页面上每个数字都会被追问三件事：**这个数是怎么算出来的**、**它对应现实世界的什么行为**、
**看到它应该怎么用**。过去这些解释散落在图注与因子手册里，读者要来回跳；
本模块把它们收敛成一套结构化条目，由页面在**原位**渲染，并强制要求：

  1) 公式必须与 src/factor_engine.py 的实现逐字对应，禁止"文档一套代码一套"；
  2) 本期数值一律由调用方传入（`values`），模块内**不写死任何数字**，
     数据更新后释义卡自动跟着变，不会过期；
  3) 每条必须同时给出「经济内容」与「投资含义」——只讲公式是数学，不讲含义是废话。

三条写作纪律
------------
* 经济内容回答：这个数在真实世界里**对应谁的行为**（基金经理、指数编制、散户、做市商）；
* 投资含义回答：它对**仓位与风险**意味着什么（不是涨跌预测，本项目不做择时承诺）；
* 误读回答：这个指标最常见的**错误用法**是什么（这是本项目反复修正口径的产物）。
"""
from __future__ import annotations

import pandas as pd


# ============================================================================
# 一、行业层指标（② 行业拥挤度页）
# ============================================================================
INDUSTRY: list[dict] = [
    dict(
        key="alloc", name="配置比例", unit="%",
        formula="Σ(该行业重仓股市值) ÷ Σ(全部重仓股市值)",
        econ="公募把多少筹码押在这个行业。它是<b>存量</b>：由过去若干季度的买入累积而成，"
             "记录的是「机构愿意把多少资产负债表放在这里」，不是当下的边际意愿。",
        invest="配置越高，该行业的边际定价权越在公募手里——申赎、排名考核、季末调仓"
               "会直接变成这个行业的买卖盘；配置低的行业对公募资金不敏感，"
               "涨跌更多由产业资本、外资与游资决定。",
        read="三比：比自身历史（分位）、比基准（超配/配置系数）、比成交（持仓热度 vs 交易热度）。",
        pitfall="高配置 ≠ 高估；行业市值暴涨也会被动推高占比，被动上升不含主动加仓的信息。",
    ),
    dict(
        key="bench", name="基准权重", unit="%",
        formula="沪深300 成份中该行业自由流通市值 ÷ 指数总自由流通市值（逐季实测）",
        econ="被动资金的「默认答案」，代表不做主动选择也能拿到的行业暴露。"
             "权重由<b>存量市值</b>决定，对产业变迁的反应天然滞后——这正是主动偏离的来源。",
        invest="基准权重低的行业，主动基金一旦看多就必须大幅超配，超配幅度天然大；"
               "基准权重高的行业（银行、非银），低配是常态，不能读成看空。",
        read="与配置比例并排看倍数（F12）；基准权重 &lt;1% 的行业，超配几个点就会让倍数失真。",
        pitfall="把「低配银行」当看空银行——这是指数编制与公募考核共同造成的结构性结果，"
                "不是每季度都有的观点。",
    ),
    dict(
        key="over", name="超配比例", unit="pct",
        formula="配置比例 − 基准权重",
        econ="主动偏离的绝对幅度：正＝集体押注，负＝系统性回避。"
             "超配的那部分资金，同时就是<b>未来的同向卖压</b>——它和对冲盘不是同一批人。",
        invest="超配越厚，该行业越像「公募的共识仓位」：逻辑被证伪时，卖出来的是同一批人、"
               "同一方向、同一时点，流动性会在最需要的时候消失。",
        read="必须双坐标：相对基准（倍数）与相对自身历史（分位/Z）。只报 pct 无法跨行业比较。",
        pitfall="跨行业直接比大小。银行超配 5pct 是极端，电子超配 5pct 只是常态，基数不同。",
    ),
    dict(
        key="Z", name="超配 Z（扩窗）", unit="",
        formula="(本期超配 − 扩窗均值) ÷ 扩窗标准差；最少 8 期，只用 ≤t 的数据",
        econ="把 31 个行业放到「自身历史」这把尺子上，度量仓位的罕见程度。"
             "扩窗（无前视）保证每个时点只用当时已知的信息。",
        invest="Z 高＝仓位已到历史少见的位置，继续加仓需要<b>新资金</b>而非存量调仓；"
               "边际买盘一旦衰竭，同等幅度的利空会被放大。",
        read="|Z|&gt;1.5 关注、&gt;2 极端；与分位交叉验证（厚尾分布下 Z 会夸大罕见度）。",
        pitfall="把 Z 当涨跌预测。Z 是位置不是方向，Z=4 只说明仓位罕见，不说明要跌。",
    ),
    dict(
        key="pct", name="超配历史分位", unit="%",
        formula="扩窗内「历史超配 &lt; 当前超配」的期数占比 ×100",
        econ="与 Z 同源但更稳健：分位数不受离群值绑架，语义直白——"
             "「有数据以来 100% 的时间都没这么重过」。",
        invest="分位越高，剩余的<b>仓位空间</b>越小。这是仓位维度的约束，不是价格维度的约束。",
        read="≥90 极度拥挤；75–90 偏拥挤；≤10 显著低配。",
        pitfall="分位 100% 不等于到顶：只要净值继续跑赢，高仓位可以靠上涨自我维持很久。",
    ),
    dict(
        key="F12", name="配置系数 F12", unit="×",
        formula="配置比例 ÷ 基准权重",
        econ="「相对指数下了几倍的注」。剔除行业体量差异后，31 个行业第一次可以横向比较。",
        invest="&gt;2.5＝公募仓位是指数权重的 2.5 倍以上，属极端共识；"
               "&lt;0.5＝机构基本放弃该行业，边际定价权已让渡给其他资金。",
        read="与分位互补：F12 相对市场，分位相对自身历史；两者同高才是「双高」。",
        pitfall="基准权重接近 0 的行业（综合、纺织服饰）倍数会被除零放大，排序必须设权重门槛。",
    ),
    dict(
        key="F13", name="筹码盈利比例 F13", unit="%",
        formula="现价高于成交量加权成本的市值 ÷ 行业总市值",
        econ="浮盈盘的厚度。它同时是持有者的心理安全垫，和随时可以兑现的潜在筹码。",
        invest="高浮盈 + 高配置＝双脆弱：回撤把浮盈打掉后，减仓会自我强化。"
               "但高浮盈本身也是趋势确认——套牢盘少，上行阻力小。",
        read="&gt;70% 获利盘极重；与超配分位同高时风险叠加；与低配同高时是「轻装」。",
        pitfall="当成卖出信号。它只改变回撤时的抛压弹性，不决定方向。",
    ),
    dict(
        key="turn", name="成交额占比 / 分位", unit="%",
        formula="行业成交额 ÷ 全市场成交额；分位为扩窗历史分位",
        econ="注意力与换手的分布，是<b>流量</b>而非存量。持仓告诉你谁拿着，成交告诉你谁在交易。",
        invest="持仓拥挤 + 交易拥挤＝真拥挤（机构与散户共振）；只有交易热而持仓不重＝主题行情，"
               "通常来得快去得也快。",
        read="成交分位 &gt;90 且配置分位 &lt;50 → 游资/主题；两者同高 → 共振；"
             "成交分位 &lt;30 而配置分位 &gt;90 → 存量抱团、增量已撤。",
        pitfall="成交占比高可能只是行业扩容（大市值、新股上市），必须同时看历史分位。",
    ),
    dict(
        key="F11", name="综合拥挤度 F11 / F11e", unit="",
        formula="超配、超配 Z、超配分位、超配动量 四项横截面 Z 加权（0.20/0.25/0.15/0.10 归一）",
        econ="把「仓位水平、历史位置、边际方向」压成一个可排序的分数，回答唯一的问题：<b>谁最挤</b>。"
             "F11e 是同一算法在偏股混合长历史（2016 起）上的版本，跨期可比。",
        invest="排序工具，不是买卖信号。前 3 名是共识最厚、也最依赖增量资金维持的位置。",
        read="三看：绝对值（&gt;1.5）、排名（前 5）、变化（环比变挤/变松，见页面底部变化榜）。",
        pitfall="合成会平均掉单维极值：F11 中性但分位 100% 的行业（如建筑材料），必须回看分项。",
    ),
    dict(
        key="level", name="拥挤等级", unit="",
        formula="由超配分位与配置系数双阈值触发，取最高级（5 档）",
        econ="把连续指标离散成可扫描的预警档位，服务于「31 个行业一屏看完」的实务需求。",
        invest="等级越高，「共识—脆弱」的组合越极端；但等级本身<b>不区分</b>"
               "「景气驱动」与「炒作驱动」，这两种拥挤的持续性完全不同。",
        read="等级 + 口径背离标记 + 四象限（配置分位 × 成交分位）三者合看。",
        pitfall="把等级当估值。等级是相对历史与相对基准的位置，不含绝对估值信息。",
    ),
]


# ============================================================================
# 二、市场层指标（① 总览 / ③ 集中度与结构）
# ============================================================================
MARKET: list[dict] = [
    dict(
        key="HHI", name="HHI 行业集中度", unit="",
        formula="Σ(各行业配置占比)²，占比取小数；31 行业等权时 = 1/31 ≈ 0.0323",
        econ="行业维度的分散度。HHI 越高，组合越像「少数行业的组合」，"
             "行业分散这道风控防线越失效。",
        invest="HHI 高＝净值由少数行业共同驱动，回撤时同向、无对冲；"
               "也是基金同质化的度量——同质化越高，赎回冲击越集中。",
        read="与等权基准（0.0323）比倍数、与自身历史分位比位置；0.23 ≈ 等权的 7 倍。",
        pitfall="HHI 上升可能只是行业市值分化（涨出来的权重），不等于主动加仓。",
    ),
    dict(
        key="CR3", name="CR3 / CR5 / CR10", unit="%",
        formula="前 3 / 5 / 10 大行业的配置占比之和",
        econ="头部集中度。CR3 直接回答「多少筹码压在几个行业上」。",
        invest="CR3 超过 2/3 时，「选行业」比「选个股」更决定净值；"
               "任一头部行业的政策或景气变化，都会主导整个主动权益群体的表现。",
        read="CR3 与 CR10 的差额＝第 4~10 名的厚度；差额小说明是「三强鼎立」，差额大说明是「一超多强」。",
        pitfall="把 CR 上升一律解读为抱团——行业市值涨跌本身会改变 CR。",
    ),
    dict(
        key="TOP1", name="第一大重仓行业", unit="%",
        formula="配置占比最高的行业及其占比",
        econ="单点暴露。它标出当前主动权益群体「最重要的一笔共识」。",
        invest="占比 &gt;40% 时，组合已近似行业基金：风格标签失真、赎回压力集中在同一赛道、"
               "与基准的偏离无法靠其他行业对冲。",
        read="看三点：占比、领先第二名多少、以及这个行业自身处在什么拥挤等级。",
        pitfall="把「第一大重仓」当成推荐。它描述的是群体行为，不是投资建议。",
    ),
    dict(
        key="POS", name="个股集中度 / 前十占净值比", unit="%",
        formula="前十大重仓股合计市值 ÷ 基金净值（样本均值）；另统计重仓股只数",
        econ="选股集中度，同时是<b>仓位代理</b>（不是真实股票仓位）。"
             "只数越少、占比越高，说明基金经理越依赖少数标的表达观点。",
        invest="集中度高＝个股风险大、调仓冲击成本高，但也说明有确信度；"
               "对拥挤度而言，重仓股即组合，卖出时没有缓冲。",
        read="与样本基金数、重仓股只数一起看：只数下降而占比上升＝共识收敛。",
        pitfall="把「前十占净值比」当股票仓位。它是下限性质的代理，真实仓位还包含非重仓持股。",
    ),
    dict(
        key="DELTA", name="配置占比的环比变化", unit="pct",
        formula="本期行业配置占比 − 上一期行业配置占比",
        econ="存量分布<b>逐季的差分</b>，也就是「这个季度资金从哪搬到哪」。"
             "占比变化把「行业涨跌」与「主动加减仓」混在一起，是流量与价格效应的合成。",
        invest="单季 ±5pct 以上的搬家属于结构性调仓，通常对应一次主线切换；"
               "±1pct 以内多为净值涨跌带来的被动漂移，不能读成主动决策。",
        read="看「谁减、谁增」的配对关系：加仓行业的资金来源，往往是同一批被减仓的行业。",
        pitfall="把占比上升全算作主动加仓。行业自身大涨也会让占比上升，需结合超配变化拆分。",
    ),
    dict(
        key="FCI", name="FCI 因子拥挤指数", unit="",
        formula="行业拥挤度的市场层合成（含扩窗分位标准化）",
        econ="全市场拥挤的<b>总量</b>度量：不问「哪里挤」，只问「整体有多挤」。",
        invest="分位到 100% 说明增量资金的需求处在历史峰值；总量指标不能指出方向，"
               "必须回到行业层面找结构。",
        read="FCI 高位 + HHI 高位＝「总量拥挤 + 结构集中」，是最脆弱的组合。",
        pitfall="用总量指标做择时。市场可以在极拥挤状态下继续上涨数月。",
    ),
]


# ============================================================================
# 三、统计检验指标（④ 因子与检验）
# ============================================================================
STAT: list[dict] = [
    dict(
        key="rankic", name="RankIC", unit="",
        formula="每期：因子值与下期行业收益的 Spearman 秩相关系数",
        econ="因子的<b>排序能力</b>：它能不能把「未来涨得多的行业」排到前面。"
             "取秩相关而非 Pearson，是为了不被极端值支配。",
        invest="|IC|≈0.1 已具备经济意义（31 个行业的排序，0.1 对应可观的多空差）；"
               "符号决定用法：正号＝高值行业后续占优，负号＝高值行业后续落后。",
        read="看均值、看胜率、看显著性，三者同向才算有效；单期 IC 几乎全是噪声。",
        pitfall="拿单期 IC 说事。一期 IC=0.3 可能只是某个行业的一次事件驱动。",
    ),
    dict(
        key="icir", name="IC_IR", unit="",
        formula="IC 均值 ÷ IC 标准差",
        econ="因子有效性的<b>稳定性</b>：同样的均值，波动越小越可信。等价于「单位风险的超额信息」。",
        invest="&gt;0.3 可用、&gt;0.5 较好；IR 低的因子即使均值漂亮，也会在实盘中反复失效。",
        read="与样本期数一起看：8 期的 IR 0.5 远不如 30 期的 IR 0.3 可信。",
        pitfall="忽略期数直接比 IR。短样本的 IR 极不稳定。",
    ),
    dict(
        key="t", name="NW t 值 / p 值", unit="",
        formula="IC 序列均值的新息 t 检验，标准误经 Newey-West 自相关调整",
        econ="考虑序列相关后的<b>统计显著性</b>。季度 IC 存在自相关，普通 t 检验会高估显著性。",
        invest="|t|&gt;2 才算在 5% 水平显著；不显著意味着「现有样本无法排除运气」。",
        read="t 与 p 配套看；同时看 Bootstrap 置信区间是否含 0（本页已给）。",
        pitfall="t 值大就下结论。多重比较（试了很多因子）会系统性抬高假阳性。",
    ),
    dict(
        key="win", name="胜率", unit="%",
        formula="IC 与预期方向一致的期数 ÷ 有效期数",
        econ="方向的一致性。高胜率说明因子不是靠某一期的极值撑起均值。",
        invest="&gt;60% 说明方向稳定；均值高但胜率 50% 的因子，本质是「押注少数极端期」。",
        read="与 IC 均值、最大单期贡献一起看：剔除最好一期后是否仍有效。",
        pitfall="把小样本胜率当概率。8 期里 7 期为正，置信区间仍然很宽。",
    ),
    dict(
        key="layer", name="分层收益 / 单调性", unit="",
        formula="按因子值把 31 个行业分 5 层，统计各层下期平均收益",
        econ="因子与收益的关系<b>形状</b>：是单调（线性可用），还是只在尾部有效（阈值效应）。",
        invest="单调才适合做加权与打分；非单调说明因子只在极端层有信息，"
               "中间层是噪声，不能用线性模型解释。",
        read="看第 1 层与第 5 层的差（多空）、看是否逐层递增、看中间层是否混乱。",
        pitfall="只看多空差、不看单调性。多空差可能全部来自最极端的一层。",
    ),
    dict(
        key="event", name="事件研究超额", unit="%",
        formula="信号触发后 3 / 6 / 12 个月，相对基准的累计超额收益",
        econ="信号的<b>实际经济价值</b>：不只是统计相关，而是真金白银能拿到的差额。",
        invest="均值显著为负，才支持「拥挤＝风险」；不显著则说明拥挤度只是状态描述，"
               "不构成择时依据——本项目实测正是后一种情形（样本期短、触发次数少）。",
        read="看均值、中位数、负收益概率、t 值四件套；触发次数 &lt;10 时结论只能是「待观察」。",
        pitfall="用 3 个月的结果推断长期。拥挤的代价往往在 6~12 个月才体现。",
    ),
]


# ============================================================================
# 渲染
# ============================================================================
def _fmt_value(v) -> tuple[str, str, str]:
    """兼容三种传值：字符串 / (值, 小字) / dict(v=, sub=, cls=)。"""
    if isinstance(v, dict):
        return (str(v.get("v", "—")), str(v.get("sub", "")), str(v.get("cls", "")))
    if isinstance(v, (tuple, list)) and len(v) >= 2:
        return (str(v[0]), str(v[1]), str(v[2]) if len(v) > 2 else "")
    if v is None:
        return ("—", "", "")
    return (str(v), "", "")


def render_grid(rows: list[dict], values: dict | None = None, colors: dict | None = None,
                cols: int = 2, *, show_formula: bool = True) -> str:
    """指标释义卡网格：左「名称 + 本期值 + 公式」，右「经济 / 投资 / 读法 / 误读」。

    为什么用卡片而不是表格：每格都有 4 段中文长文本，表格在 1400px 宽度下
    会把每列压到 200px 以内，读者只会看到一堆折行；卡片可以按 2 列自适应。
    """
    values = values or {}
    colors = colors or {}
    line = colors.get("line", "#E3E8EF")
    dim = colors.get("text_dim", "#6B7A8F")
    accent = colors.get("price", "#1F6FEB")
    cards = []
    for r in rows:
        raw = values.get(r["key"])
        v, sub, cls = _fmt_value(raw)
        color = colors.get("up" if cls == "up" else "down" if cls == "dn" else "text",
                           "#1A2433")
        f = (f'<div class="mx-f">= {r["formula"]}</div>' if show_formula else "")
        cards.append(
            f'<div class="mx-card">'
            f'<div class="mx-h"><span class="mx-n">{r["name"]}</span>'
            f'<span class="mx-v" style="color:{color}">{v}</span></div>'
            f'{f}'
            f'<div class="mx-l"><span class="mx-k">经济</span>{r["econ"]}</div>'
            f'<div class="mx-l"><span class="mx-k">投资</span>{r["invest"]}</div>'
            f'<div class="mx-l"><span class="mx-k">怎么看</span>{r["read"]}</div>'
            f'<div class="mx-l mx-p"><span class="mx-k">误读</span>{r["pitfall"]}</div>'
            f'</div>')
    style = (f'display:grid;grid-template-columns:repeat({cols},minmax(0,1fr));'
             f'gap:7px;')
    head = (f'<div style="font-size:10px;color:{dim};margin:-2px 0 4px 0">'
            f'经济＝这个数对应现实世界的什么行为；投资＝对仓位与风险意味着什么；'
            f'误读＝最常见的错误用法。数值随数据自动更新。</div>')
    return f'<div style="{style}">' + "".join(cards) + "</div>" + head


def render_chain(steps: list[tuple[str, str, str]], colors: dict | None = None) -> str:
    """三步算术链：① 分子 ② 分母 ③ 差额。每步给出「标签 / 数 / 说明」。"""
    colors = colors or {}
    line = colors.get("line", "#E3E8EF")
    dim = colors.get("text_dim", "#6B7A8F")
    cells = []
    for i, (lab, val, txt) in enumerate(steps, 1):
        cells.append(
            f'<div class="ch-cell">'
            f'<div class="ch-t"><span class="ch-i">{i}</span>{lab}</div>'
            f'<div class="ch-v">{val}</div>'
            f'<div class="ch-x">{txt}</div></div>')
        if i < len(steps):
            cells.append(f'<div class="ch-op">→</div>')
    return f'<div class="ch-wrap" style="border-color:{line}">' + "".join(cells) + "</div>"


def crowding_regime(alloc_pct: float, turn_pct: float) -> dict:
    """四象限：机构仓位高度（超配分位）× 交易热度（成交额占比分位）。

    这是本页「能看出来什么」的核心判定：同样是高拥挤，由产业景气驱动的
    与由主题炒作驱动的，持续性与风险完全不同，而单一指标无法区分。
    """
    a, t = alloc_pct, turn_pct
    if pd.isna(a) or pd.isna(t):
        return dict(tag="数据不足", cls="mut",
                    reading="分位数据不足，无法判定象限。",
                    meaning="等待至少 8 期扩窗历史后可判定。")
    if a >= 75 and t >= 75:
        return dict(
            tag="双热 · 产业趋势型", cls="up",
            reading=f"机构仓位分位 {a:.0f}% 与交易分位 {t:.0f}% 同时处于历史高位——"
                    f"持仓与交易共振，是典型的<b>产业趋势</b>结构，而非短期主题。",
            meaning="趋势最强，脆弱性也最高：定价已包含乐观预期，边际利好会钝化，"
                    "对盈利不及预期极度敏感。风险不在方向，在「预期差的空间」。")
    if a >= 75 and t < 50:
        return dict(
            tag="存量抱团 · 增量退潮", cls="",
            reading=f"机构仓位分位 {a:.0f}% 仍高，但交易分位只有 {t:.0f}%——"
                    f"筹码还在，增量资金已经撤了。",
            meaning="最典型的高位滞涨结构：价格靠存量维持，一旦出现替代赛道，"
                    "减仓会先于价格发生。此时成交萎缩比下跌更值得警惕。")
    if a < 50 and t >= 75:
        return dict(
            tag="主题交易 · 机构缺席", cls="",
            reading=f"交易分位 {t:.0f}% 很热，但机构仓位分位只有 {a:.0f}%——"
                    f"涨的是情绪与游资，不是公募的配置盘。",
            meaning="持续性取决于赚钱效应：成交萎缩即结束。因为机构没上车，"
                    "回调时也缺少配置盘的承接，波动会更大。")
    if a <= 25 and t <= 25:
        return dict(
            tag="双冷 · 无人问津", cls="dn",
            reading=f"仓位分位 {a:.0f}%、交易分位 {t:.0f}%，两项都在历史低位。",
            meaning="拥挤风险最低，但也意味着缺少催化剂与增量关注；"
                    "低配可以持续很久，「便宜」本身不构成买入理由。")
    return dict(
        tag="中性区间", cls="",
        reading=f"仓位分位 {a:.0f}%、交易分位 {t:.0f}%，两项都未进入极值区。",
        meaning="没有明显的拥挤或出清特征，方向取决于景气与政策，"
                "拥挤度指标在此区间的解释力最弱。")


def render_regime(reg: dict, colors: dict | None = None) -> str:
    colors = colors or {}
    c = colors.get({"up": "up", "dn": "down"}.get(reg.get("cls", ""), "text"), "#1A2433")
    bg = "#FFF6F4" if reg.get("cls") == "up" else ("#F3F7FF" if reg.get("cls") == "dn" else "#F8FAFC")
    return (f'<div class="rg" style="background:{bg}">'
            f'<div class="rg-t">结构判定：<b style="color:{c}">{reg["tag"]}</b></div>'
            f'<div class="rg-r">{reg["reading"]}</div>'
            f'<div class="rg-m"><span class="mx-k">投资含义</span>{reg["meaning"]}</div>'
            f'</div>')


def _n(v, spec="{:.2f}", na="—") -> str:
    """数值安全格式化：NaN / None / 非数值一律返回占位符，绝不输出 nan。"""
    try:
        if v is None or pd.isna(v):
            return na
    except Exception:
        pass
    try:
        return spec.format(v)
    except Exception:
        return na


def decompose_overweight(ctx: dict, colors: dict | None = None) -> str:
    """「为什么是超配 X pct」的三步拆解。

    ctx 需要的字段（全部由调用方从当期数据算出）：
        sel, q, alloc, bench, over, f12, alloc_rank, bench_rank, n_ind,
        n_funds, n_stocks, cr3, top3_bench, top3_alloc, f13, turn, turn_pct,
        pct, z, bench_name
    """
    colors = colors or {}
    c_up = colors.get("up", "#D1242F")
    c_dn = colors.get("down", "#2EA043")
    dim = colors.get("text_dim", "#6B7A8F")
    sel = ctx.get("sel", "")
    over = ctx.get("over")
    if over is None or pd.isna(over):
        return (f'<div class="dz"><div class="dz-h">为什么是这个数？——{sel}（{ctx.get("q", "")}）'
                f'</div><div class="r3-x">该行业当期超配数据不足（基准权重缺失、或扩窗历史不足 8 期），'
                f'无法拆解。这不是「数值为 0」，而是样本尚未满足计算前提。</div></div>')

    alloc, bench = ctx.get("alloc"), ctx.get("bench")
    f12, f13 = ctx.get("f12"), ctx.get("f13")
    c = c_up if over >= 0 else c_dn
    sign = "+" if over >= 0 else ""
    rank_txt = (f"配置排名 {ctx.get('alloc_rank', '—')}/{ctx.get('n_ind', 31)}、"
                f"基准排名 {ctx.get('bench_rank', '—')}/{ctx.get('n_ind', 31)}")
    f12_txt = (f"<b>{f12:.2f}×</b>" if pd.notna(f12)
               else "<b>无定义</b>（基准权重≈0，倍数不可用）")
    steps = [
        ("分子 · 机构实际配置", _n(alloc, "{:.2f}%"),
         f"{_n(ctx.get('n_funds'), '{:.0f}')} 只主动权益基金、"
         f"{_n(ctx.get('n_stocks'), '{:.0f}')} 只重仓股合计："
         f"每 100 元重仓市值有 <b>{_n(alloc, '{:.2f}')} 元</b>押在{sel}。{rank_txt}。"),
        ("分母 · 指数给的权重", _n(bench, "{:.2f}%"),
         f"{ctx.get('bench_name', '沪深300')} 按自由流通市值加权，{sel}只占 "
         f"{_n(bench, '{:.2f}%')}；权重前三为 {ctx.get('top3_bench', '—')}——"
         f"指数仍以存量市值为锚。"),
        ("差额 · 主动偏离", f"{sign}{over:.2f} pct",
         f"{_n(alloc, '{:.2f}')} − {_n(bench, '{:.2f}')} = {sign}{over:.2f} pct，"
         f"即配置系数 {f12_txt}。"),
    ]
    body = render_chain(steps, colors)
    # 四股力量：把「差额」归因到可验证的机制，而不是讲故事
    forces = [
        ("基准的构造", "沪深300 的权重由<b>存量市值</b>决定，产业变迁要经过股价上涨才进入指数——"
                   "所以指数对新兴产业的暴露天然滞后于主动基金的研究结论。"),
        ("主动的定价", f"公募按<b>边际盈利变化</b>配置：{sel}的盈利上修若在 31 个行业中靠前，"
                   f"资金就会持续从低景气行业搬过来，{ctx.get('top3_alloc', '—')} 合计已占"
                   f" {_n(ctx.get('cr3'), '{:.2f}')}%（CR3）。"),
        ("正反馈", f"重仓行业上涨 → 净值排名靠前 → 申购流入 → 继续买。{sel}当前筹码盈利比例"
                 f" {_n(f13, '{:.1f}')}%，浮盈厚既支撑持有，也积累兑现压力。"),
        ("容量约束", "能容纳千亿级资金的行业只有少数几个。小行业即使被看好也「配不上去」，"
                   "这解释了为什么极端超配总是出现在大市值成长行业。"),
    ]
    fl = "".join(f'<div class="fc"><div class="fc-t">{k}</div><div class="fc-x">{v}</div></div>'
                 for k, v in forces)
    # 三条读法：把数字放回三个坐标系
    pct_v, z_v = ctx.get("pct"), ctx.get("z")
    pct_txt = ("有数据以来最重" if pd.notna(pct_v) and pct_v >= 99.5
               else "处于历史高位" if pd.notna(pct_v) and pct_v >= 90
               else "处于历史中位附近" if pd.notna(pct_v) else "分位数据不足")
    reads = [
        ("相对基准", f"{_n(f12, '{:.2f}×')} —— 公募把指数里的 {_n(bench, '{:.2f}%')} 配成了 "
                 f"{_n(alloc, '{:.2f}%')}，偏离幅度 {sign}{over:.2f} pct。"),
        ("相对自身历史", f"超配分位 {_n(pct_v, '{:.0f}%')}（{_n(z_v, '{:+.2f}')}σ，扩窗无前视）——"
                    f"{pct_txt}，继续加仓需要新资金。"),
        ("相对交易热度", f"成交额占比 {_n(ctx.get('turn'), '{:.2f}%')}"
                    f"（分位 {_n(ctx.get('turn_pct'), '{:.0f}%')}）——"
                    f"{'持仓与交易同步极值，双重拥挤' if (pd.notna(ctx.get('turn_pct')) and pd.notna(pct_v) and ctx['turn_pct'] >= 90 and pct_v >= 90) else '交易热度与持仓位置需分开判断'}。"),
    ]
    rl = "".join(f'<div class="r3"><div class="r3-t">{k}</div><div class="r3-x">{v}</div></div>'
                 for k, v in reads)
    return (f'<div class="dz"><div class="dz-h">为什么是 <b style="color:{c}">{sign}{over:.2f} pct</b>？'
            f'——{sel}的超配拆解（{ctx.get("q", "")}）</div>'
            f'{body}'
            f'<div class="dz-sub">差额从哪来：四股力量</div><div class="fc-wrap">{fl}</div>'
            f'<div class="dz-sub">这组数字能看出什么</div><div class="r3-wrap">{rl}</div>'
            f'<div class="dz-note" style="color:{dim}">拆解全部由当期数据实时计算：'
            f'分子＝该行业重仓市值占比，分母＝{ctx.get("bench_name", "沪深300")}行业权重，'
            f'差额＝两者之差。修改口径（股票池、基准、季度）后这三个数会同步变化，'
            f'不会出现「文字写死、数据已变」的情况。</div></div>')


# 页面 CSS（由 app.py 注入到全局 <style>；占位符替换而非 format，避免 CSS 花括号转义）
def css(colors: dict | None = None) -> str:
    colors = colors or {}
    out = _CSS
    for k, v in (("__LINE__", colors.get("line", "#E3E8EF")),
                 ("__TEXT__", colors.get("text", "#1A2433")),
                 ("__DIM__", colors.get("text_dim", "#6B7A8F")),
                 ("__ACCENT__", colors.get("price", "#1F6FEB"))):
        out = out.replace(k, v)
    return out


_CSS = """
  .mx-card { background:#FFFFFF; border:1px solid __LINE__; border-radius:7px;
             padding:6px 9px; }
  .mx-h { display:flex; align-items:baseline; justify-content:space-between;
          gap:8px; margin-bottom:1px; }
  .mx-n { font-size:11.5px; font-weight:700; color:__TEXT__; }
  .mx-v { font-size:13px; font-weight:700; font-variant-numeric:tabular-nums; }
  .mx-f { font-size:9.5px; color:__DIM__; line-height:1.45; margin-bottom:3px;
          padding-bottom:3px; border-bottom:1px dashed __LINE__; }
  .mx-l { font-size:10.5px; line-height:1.58; color:#37485C; margin-top:2px; }
  .mx-k { display:inline-block; font-size:9px; font-weight:700; color:#FFFFFF;
          background:__DIM__; border-radius:3px; padding:0 4px; margin-right:4px;
          vertical-align:1px; }
  .mx-p { color:#8A5A00; background:#FFFBF2; border-radius:4px; padding:2px 5px; }

  .ch-wrap { display:flex; align-items:stretch; gap:4px; flex-wrap:wrap;
             border:1px solid __LINE__; border-radius:7px; padding:6px; background:#FFFFFF; }
  .ch-cell { flex:1 1 210px; min-width:190px; }
  .ch-t { font-size:10px; color:__DIM__; font-weight:700; }
  .ch-i { display:inline-block; width:14px; height:14px; line-height:14px; text-align:center;
          border-radius:50%; background:__ACCENT__; color:#fff; font-size:9px;
          margin-right:4px; }
  .ch-v { font-size:17px; font-weight:700; line-height:1.3; font-variant-numeric:tabular-nums; }
  .ch-x { font-size:10.5px; line-height:1.55; color:#37485C; }
  .ch-op { align-self:center; color:__DIM__; font-size:14px; padding:0 1px; }

  .dz { border:1px solid __LINE__; border-radius:8px; padding:8px 10px; background:#FDFEFE; }
  .dz-h { font-size:12px; font-weight:700; margin-bottom:6px; }
  .dz-sub { font-size:11px; font-weight:700; margin:8px 0 4px 0; color:__TEXT__;
            padding-left:6px; border-left:3px solid __ACCENT__; }
  .dz-note { font-size:9.5px; line-height:1.5; margin-top:6px; }
  .fc-wrap { display:flex; gap:6px; flex-wrap:wrap; }
  .fc { flex:1 1 240px; min-width:210px; background:#FFFFFF; border:1px solid __LINE__;
        border-radius:6px; padding:5px 8px; }
  .fc-t { font-size:10.5px; font-weight:700; color:__ACCENT__; margin-bottom:2px; }
  .fc-x { font-size:10.5px; line-height:1.6; color:#37485C; }
  .r3-wrap { display:flex; gap:6px; flex-wrap:wrap; }
  .r3 { flex:1 1 240px; min-width:210px; border-left:3px solid __LINE__; padding-left:8px; }
  .r3-t { font-size:10.5px; font-weight:700; color:__TEXT__; }
  .r3-x { font-size:10.5px; line-height:1.6; color:#37485C; }

  .rg { border:1px solid __LINE__; border-radius:7px; padding:7px 10px; margin-top:6px; }
  .rg-t { font-size:11.5px; font-weight:700; }
  .rg-r { font-size:11px; line-height:1.6; color:#37485C; margin-top:2px; }
  .rg-m { font-size:10.5px; line-height:1.6; color:#37485C; margin-top:3px;
          padding-top:3px; border-top:1px dashed __LINE__; }
"""
