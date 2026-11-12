# -*- coding: utf-8 -*-
"""**全站唯一口径源**：阈值、窗口、分母、最小样本、缺失处理都从这里取。

为什么要有这个文件
------------------
git 历史里 25 次提交带「修」，主题频次：口径 17、分位 14、小样本 4、阈值 4、排名 3。
同一类问题反复出现，根因不是改错，而是**每次只改当下这一处**：
「80 分算拥挤」在 map_page、canon、report_charts、glossary 各写一遍，
改了一处、其余六处还是旧值；读者看到的就是「图说 80、表说 85、方法页说 75」。

从 2026-11-12 起：
  · 所有**数值型口径**（阈值/最小样本/分母规则）只在本文件定义；
  · 图表与表格 `import caliber as CAL` 后引用，不再写字面量；
  · `tools/check_caliber.py` 在核心文件里 grep 硬编码，命中即失败；
  · README 的口径表由 `caliber.table_markdown()` 生成，禁止文档另写一套。

字段含义（每条指标一行）
------------------------
metric_id   机器名（代码里引用）
name        中文名（页面/文档展示）
definition  一句话口径（含「算什么、不算什么」）
sample      样本区间
window      窗口规则（谁决定、默认值）
denominator 分母（**最容易出错的一栏**）
min_n       低于这个期数不出统计量
missing     缺失怎么处理（留空/不评级/剔除）
version     该条口径最后一次实质变更的日期
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

# 口径表自身的版本：任何一条 definition/denominator 变了都要改这里，
# 它会进 version.json，页面底部与 deploy_check 都能看到。
CALIBER_VERSION = "2026-11-12"

# ---------------------------------------------------------------------------
# 一、阈值常量（**全站唯一**；改这里，图、表、文案一起变）
# ---------------------------------------------------------------------------
THR_CROWD = 80          # 拥挤阈值：CSI ≥ 80 视为极度拥挤（可在页面滑块 60–95 调整）
THR_WARM = 65           # 偏挤：广度统计的「已挤」口径（≥65 分的行业占比）
THR_NEUTRAL = 45        # 中性下沿
THR_COLD = 25           # 冷门下沿
BAND_EDGES = (0, THR_COLD, THR_NEUTRAL, THR_WARM, THR_CROWD, 101)  # 五档底色

SMALL_HOLD_N = 150      # 小样本（横截面）：持有该行业的基金数下限
SMALL_SHARE_PCT = 0.5   # 小样本（横截面）：前十配置合计下限（%）
MIN_N_PCT = 8           # 分位/排名/均值/标准差 的最小期数（时间序列维度）
MIN_N_STAT = 4          # 任何统计量的绝对最小期数；低于它只报「当前值 + 期数」
MIN_IC_N = 12           # 做 IC / t 检验的最小期数
TOUCH_THR = THR_CROWD   # 「全历史触及 ≥X」默认用的阈值

# 其他「长得像 80」的阈值，必须与拥挤阈值分开定义：
# 它们数值相同、含义不同，混用一个常量会导致改拥挤线时主题基金也被改掉。
THEME_FUND_PCT = 80.0    # 单行业配置占净值 >80% → 判为行业主题基金（分歧度清洗用）
THR_EXTREME_PCT = 90.0   # 「指数分位」极值线（与 CSI 的 80 不是一回事）

# 最小样本：横截面相关至少要有 8 个行业，否则 Spearman 没有意义
MIN_OBS_XSEC = 8

# 图表排版阈值（不是统计口径，但也只在这里出现，避免各图各写）
LEGEND_MAX_ITEMS = 12   # 图例/堆叠段上限：超过就合并
MIN_LABEL_PCT = 4.0     # 段内写数值的最小占比（低于它写字会压到边界）
RESID_MIN_PCT = 0.5     # 残差带/合并的最小占比
MERGE_CUR_PCT = 1.0     # 合并阈值①：当期占比 < 1%
MERGE_PEAK_PCT = 3.0    # 合并阈值②：历史峰值 < 3%（两条同时满足才并入「其他」）


@dataclass(frozen=True)
class Caliber:
    metric_id: str
    name: str
    definition: str
    sample: str
    window: str
    denominator: str
    min_n: str
    missing: str
    version: str


def _c(**kw) -> Caliber:
    return Caliber(**kw)


# ---------------------------------------------------------------------------
# 二、指标口径表（首批 7 条：本轮争议全部落在这里）
# ---------------------------------------------------------------------------
ITEMS: dict[str, Caliber] = {k: _c(**v) for k, v in {
    "csi_pct": dict(
        metric_id="csi_pct", name="CSI 分位（自身历史）",
        definition="当期 CSI 在该行业自身历史序列中由低到高的百分位；不是「配置分位」，"
                   "也不是横截面排名。",
        sample="该行业全部有 CSI 的期次（2016Q4 起）",
        window="页面「分位窗口」控件：全部历史 / 近 20 期 / 近 12 期；"
               "均值、标准差、P10/P50/P90 与分位共用同一窗口",
        denominator="该行业在该窗口内的可比期数",
        min_n=f"≥{MIN_N_PCT} 期才给分位；<{MIN_N_STAT} 期只报当前值与期数",
        missing="期数不足→留空并注明「样本不足」，不插值、不外推",
        version="2026-11-12"),

    "cross_pct": dict(
        metric_id="cross_pct", name="全行业分位（截面）",
        definition="当期 CSI 在**可评级行业**中的百分位；小样本行业不参与，"
                   "因此分母不是 31。",
        sample="当期有 CSI 且可评级的行业",
        window="当期横截面（无时间窗口）",
        denominator=f"可评级行业数（31 − 小样本数），当前为 26",
        min_n="单期截面，无需最小期数",
        missing="小样本→不给分位（显示「—（不评级）」）",
        version="2026-11-12"),

    "rank": dict(
        metric_id="rank", name="排名",
        definition="当期 CSI 在可评级行业中的名次，1＝最挤；并列取最小名次（method=min）。",
        sample="当期有 CSI 且可评级的行业",
        window="当期",
        denominator="可评级行业数",
        min_n=f"该行业自身期数 <{MIN_N_PCT} 时不给排名",
        missing="小样本/期数不足→「—（不评级）」",
        version="2026-11-12"),

    "z": dict(
        metric_id="z", name="z 值",
        definition="(当前 CSI − 同窗口均值) / 同窗口样本标准差（ddof=1）。",
        sample="同 csi_pct 的窗口",
        window="与 csi_pct 完全一致（同一窗口，不另设）",
        denominator="该窗口内可比期数",
        min_n=f"≥{MIN_N_PCT} 期且标准差 > 0",
        missing="期数不足/标准差为 0→留空",
        version="2026-11-12"),

    "touch": dict(
        metric_id="touch", name="全历史触及 ≥X",
        definition=f"该行业全历史中 CSI ≥ {TOUCH_THR} 的期数（分子），分母是该行业的"
                   "可比期数——**不是**总期数 40。",
        sample="该行业全部有 CSI 的期次",
        window="全历史（不受「分位窗口」影响）",
        denominator="该行业可比期数",
        min_n=f"<{MIN_N_PCT} 期→显示「—」",
        missing="无值期不参与分子分母",
        version="2026-11-12"),

    "hold_share": dict(
        metric_id="hold_share", name="持仓占比",
        definition="基金持有该股票的市值占基金净值的比例。**前十大口径**只含季报前十大，"
                   "**全持仓口径**含半年报/年报全部持仓，两者不可混用、不可相加。",
        sample="季报（前十大）/ 半年报与年报（全持仓）",
        window="单期",
        denominator="基金净资产（不是股票市值、不是前十大合计）",
        min_n="单期，无最小期数",
        missing="未披露→留空；港股行业分类缺失单列，不并入 A 股",
        version="2026-11-12"),

    "fund_shares": dict(
        metric_id="fund_shares", name="份额（A/C 合并）",
        definition="同一基金各份额类别的份额相加，得到「基金整体」份额；"
                   "与资产配置表的净资产同口径。",
        sample="该基金全部份额类别",
        window="报告期",
        denominator="不适用（绝对量，亿元）",
        min_n="单期",
        missing="任一类别抓取失败→整表留空（宁可暂无，也不入库缺一类的错数）",
        version="2026-11-12"),
}.items()}


def get(metric_id: str) -> Caliber:
    """取一条口径；取不到就抛 KeyError（不许静默用默认值）。"""
    return ITEMS[metric_id]


def table() -> list[dict]:
    return [asdict(c) for c in ITEMS.values()]


def table_markdown() -> str:
    """README / 方法页用的口径表（与代码同源，禁止手写第二份）。"""
    head = ("| metric_id | 中文名 | 口径定义 | 分母 | 最小样本 | 缺失处理 | 版本 |\n"
            "|---|---|---|---|---|---|---|")
    rows = [f"| `{c.metric_id}` | {c.name} | {c.definition} | {c.denominator} | "
            f"{c.min_n} | {c.missing} | {c.version} |" for c in ITEMS.values()]
    return head + "\n" + "\n".join(rows)


def check_consistency() -> list[str]:
    """自洽性检查：阈值顺序、区间闭合、常量未被写反。"""
    bad = []
    if not (BAND_EDGES[0] < BAND_EDGES[1] < BAND_EDGES[2] < BAND_EDGES[3]
            < BAND_EDGES[4] < BAND_EDGES[5]):
        bad.append(f"BAND_EDGES 不是严格递增：{BAND_EDGES}")
    if not (THR_COLD < THR_NEUTRAL < THR_WARM < THR_CROWD):
        bad.append("阈值顺序应为 冷门 < 中性 < 偏挤 < 拥挤")
    if not (MIN_N_STAT < MIN_N_PCT <= MIN_IC_N):
        bad.append("最小样本应满足 MIN_N_STAT < MIN_N_PCT ≤ MIN_IC_N")
    if not (MERGE_CUR_PCT < MERGE_PEAK_PCT):
        bad.append("合并阈值应满足 当期阈值 < 历史峰值阈值")
    for k, c in ITEMS.items():
        if c.metric_id != k:
            bad.append(f"键与 metric_id 不一致：{k} != {c.metric_id}")
        if not c.denominator.strip():
            bad.append(f"{k} 缺分母定义")
    return bad
