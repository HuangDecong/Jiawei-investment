# -*- coding: utf-8 -*-
"""
config.py  ——  路径、行业口径、阈值、配色、因子权重
======================================================================
本项目唯一需要按需修改的参数集中地。

设计原则
--------
1. 路径全部由 ROOT 派生，换机器只改 ROOT（或不动，用 __file__ 自动定位）。
2. 阈值分三类：
   * 数据校验阈值  VALIDATION_*   —— 越界即报错/告警
   * 因子判断阈值  THRESH_*       —— 决定因子"拥挤/低配"标签
   * 市场信号阈值  SIGNAL_*       —— 决定市场层面的整体状态标语
3. 配色遵循 A 股习惯：**红涨绿跌**（与欧美相反）。
"""

from __future__ import annotations

import os

# ============================================================================
# 路径
# ============================================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "data")
DATA_RAW = os.path.join(DATA, "raw")
DATA_PROCESSED = os.path.join(DATA, "processed")
OUTPUT = os.path.join(ROOT, "output")
ASSETS = os.path.join(ROOT, "assets")

for _d in (DATA_RAW, DATA_PROCESSED, OUTPUT, ASSETS):
    os.makedirs(_d, exist_ok=True)

# 原始文件（全部可在 data/raw/ 下按同名覆盖以实现增量更新）
F_ALLOCATION = os.path.join(DATA_RAW, "industry_allocation.xlsx")
F_OVERWEIGHT = os.path.join(DATA_RAW, "industry_overweight.xlsx")
F_HS300_WEIGHTS = os.path.join(DATA_RAW, "hs300_weights.xlsx")
F_HS300_RETURNS = os.path.join(DATA_RAW, "hs300_returns.xlsx")
F_TURNOVER = os.path.join(DATA_RAW, "sw_industry_turnover.xlsx")
F_RETURNS = os.path.join(DATA_RAW, "sw_industry_returns.xlsx")
F_CR_HHI = os.path.join(DATA_RAW, "industry_cr_hhi.xlsx")
F_STOCK_CROWD = os.path.join(DATA_RAW, "stock_crowd.xlsx")
F_POSITION = os.path.join(DATA_RAW, "position_proxy.xlsx")
F_MASTER = os.path.join(DATA_RAW, "master_workbook.xlsx")

# ---- 本次升级新增的数据文件 ----
F_HS300_WEIGHTS_Q = os.path.join(DATA_RAW, "hs300_weights_quarterly.xlsx")  # 动态季度权重
F_TURNOVER_DAILY = os.path.join(DATA_RAW, "sw_industry_turnover_daily.xlsx")  # 行业日度成交额
F_ETF_DAILY = os.path.join(DATA_RAW, "etf_flow_daily.xlsx")                   # ETF日度资金流
F_HOLDINGS_FULL = os.path.join(DATA_RAW, "fund_holdings_full_*.xlsx")         # 全持仓(半年/年报)

# 自动合并多个同名系列的 glob 模式（如 fund_holdings_2026Q2.xlsx / _2026Q3.xlsx）
PAT_HOLDINGS = os.path.join(DATA_RAW, "fund_holdings_*.xlsx")

# 输出文件
O_VALIDATION_LOG = os.path.join(OUTPUT, "data_validation_log.txt")
O_FACTOR_PANEL = os.path.join(OUTPUT, "factor_panel.xlsx")
O_IC_REPORT = os.path.join(OUTPUT, "ic_report.xlsx")
O_SIGNAL_REPORT = os.path.join(OUTPUT, "signal_report.xlsx")
O_BACKTEST_REPORT = os.path.join(OUTPUT, "backtest_report.xlsx")
O_STRUCTURE = os.path.join(OUTPUT, "structure_analysis.xlsx")   # 结构变迁可视化数据
O_FCI_REPORT = os.path.join(OUTPUT, "fci_report.xlsx")           # FCI 因子拥挤指数
O_COVERAGE = os.path.join(OUTPUT, "daily_coverage.csv")          # 日度数据覆盖率
O_PROVENANCE = os.path.join(OUTPUT, "data_provenance.csv")       # 数据来源登记表
O_RESEARCH_REPORT = os.path.join(OUTPUT, "research_report.xlsx")  # 第2轮：研究级检验
O_RESEARCH3_REPORT = os.path.join(OUTPUT, "research3_report.xlsx")  # 第3轮：walk-forward 等
O_RESEARCH4_REPORT = os.path.join(OUTPUT, "research4_report.xlsx")  # 第4轮：成本/容量/因子动量
O_DYNAMIC_WEIGHTS = os.path.join(OUTPUT, "dynamic_weights_report.xlsx")
O_PREFLIGHT = os.path.join(OUTPUT, "preflight_gate.csv")          # 数据质量前置闸门
O_STRATEGY = os.path.join(OUTPUT, "strategy_report.xlsx")         # 组合层面检验

P_FACTOR_PANEL = os.path.join(DATA_PROCESSED, "factor_panel.csv")
P_SIGNAL_REPORT = os.path.join(DATA_PROCESSED, "signal_report.csv")
P_MARKET_INDICATORS = os.path.join(DATA_PROCESSED, "market_indicators.csv")

# ============================================================================
# 行业口径
# ============================================================================
INDUSTRIES = [
    "农林牧渔", "基础化工", "钢铁", "有色金属", "电子", "汽车", "家用电器",
    "食品饮料", "纺织服饰", "轻工制造", "医药生物", "公用事业", "交通运输",
    "房地产", "商贸零售", "社会服务", "银行", "非银金融", "综合", "建筑材料",
    "建筑装饰", "电力设备", "机械设备", "国防军工", "计算机", "传媒", "通信",
    "煤炭", "石油石化", "环保", "美容护理",
]
NI = len(INDUSTRIES)                    # 31

# 行业分组（仅用于展示归类，不参与计算）
INDUSTRY_GROUP = {
    "上游资源": ["煤炭", "石油石化", "有色金属", "钢铁", "基础化工"],
    "中游制造": ["电力设备", "机械设备", "国防军工", "建筑材料", "建筑装饰"],
    "科技成长": ["电子", "通信", "计算机", "传媒"],
    "消费": ["食品饮料", "家用电器", "汽车", "纺织服饰", "轻工制造",
             "商贸零售", "社会服务", "美容护理", "农林牧渔", "医药生物"],
    "金融地产": ["银行", "非银金融", "房地产"],
    "公用与其他": ["公用事业", "交通运输", "环保", "综合"],
}

# ============================================================================
# 口径与窗口
# ============================================================================
MIN_HIST = 8                 # 扩窗统计最小历史季度数（Z / 分位 / P90 均需满足）

# ---------------------------------------------------------------------------
# 样本内 / 样本外划分（数据源切换 Wind 后重设）
# ---------------------------------------------------------------------------
# 背景：本项目当前的真实可建模区间为 holdings ∩ hs300 = 2024Q2 ~ 2026Q2（9 期）。
# 若沿用旧的 IS_LAST="2023Q4"，样本内将为空集，所有样本外检验失效。
# 因此改为**自适应划分**：取可建模区间的前 6 期作样本内（因子形式设计），
# 后 3 期作样本外（形式冻结后的验证）。设计期末随数据增长自动右移。
#
# 注意：当历史数据补齐至 2010 年后，应改回固定划分（如 IS_LAST="2023Q4"）。
IS_LAST = "2025Q3"           # 样本内末期（可建模区间前 6 期）
OOS_FIRST = "2025Q4"         # 样本外首期（后 3 期）
IS_ADAPTIVE = True           # True = 由数据自动推导；补齐长历史后改 False


def derive_split(quarters) -> dict:
    """由**实际可用的季度序列**推导 样本内/样本外 与 设计期/留出期 切点。

    为什么必须自适应
    ----------------
    数据源切换为 Wind 后，可建模区间收缩为 2024Q2 ~ 2026Q2（9 期）。
    而研究脚本 research.py / research3.py 里原先写死的
    `DEV_END = "2022Q4"` / `HOLDOUT_START = "2023Q1"` 直接导致：

      · 设计期（≤2022Q4）**为空集**；
      · `design_oos_test` 对全部候选因子都因 `len(ic_dev) < 6` 跳过；
      · 返回 `table` 为空 DataFrame（0 行 0 列）；
      · 下游 `factor_verdict_table` 执行 `.set_index("因子")` 抛 KeyError，
        整个看板崩溃。

    这不是"因子无效"，而是**样本切点与数据区间不匹配的实现缺陷**。
    机构级做法：切点必须由数据推导，并在日志中显式登记实际使用的切点，
    使「样本外检验在什么区间上做的」可回溯。

    切分规则（事前写死，不随结果调整）
    ----------------------------------
      可建模期数 n
        n >= 14 : 设计期取前 60%，留出期取后 40%（至少留 4 期）
        9 <= n < 14 : 设计期取前 ceil(n·2/3)，留出期取剩余（至少留 3 期）
        n < 9  : 不做切分，返回 None（样本不足，样本外检验不成立——如实报告）

    返回
    ----
    dict(dev_end, holdout_start, n, ok, note)
    """
    qs = sorted(str(q) for q in quarters)
    n = len(qs)
    if n == 0:
        return dict(dev_end=None, holdout_start=None, n=0, ok=False,
                    note="无可建模季度：样本外检验不成立")
    if IS_ADAPTIVE and n >= 9:
        if n >= 14:
            k = max(int(round(n * 0.60)), n - max(int(round(n * 0.40)), 4))
        else:
            k = int(np.ceil(n * 2.0 / 3.0))
            k = min(k, n - 3)          # 至少留 3 期留出期
        k = max(min(k, n - 3), 4)      # 设计期至少 4 期、留出期至少 3 期
        dev_end = qs[k - 1]
        holdout_start = qs[k]
        return dict(dev_end=dev_end, holdout_start=holdout_start, n=n, ok=True,
                    note=(f"自适应切分（可建模 {n} 期）：设计期 {qs[0]}~{dev_end}"
                          f"（{k} 期）｜留出期 {holdout_start}~{qs[-1]}（{n - k} 期）"))
    if not IS_ADAPTIVE and n:
        return dict(dev_end=IS_LAST, holdout_start=OOS_FIRST, n=n, ok=True,
                    note=f"固定切分：设计期 ≤{IS_LAST}｜留出期 ≥{OOS_FIRST}")
    return dict(dev_end=None, holdout_start=None, n=n, ok=False,
                note=(f"可建模区间仅 {n} 期（<9），样本量不足以支持样本外切分；"
                      "样本外检验结论不成立——如实报告，不做切分。"))

# numpy 仅在本模块的 derive_split 中使用，函数内导入以避免与调用方同名冲突
import numpy as np  # noqa: E402
LAYER_N = 5                  # 分层回测层数
NW_LAGS = 2                  # Newey-West 滞后阶数
HORIZONS_Q = (1, 2, 4)       # 事件研究持有期（季度）= 3 / 6 / 12 个月
ROBUST_WINDOWS = (8, 12, 16)  # 参数敏感性：滚动窗口长度

# 季报披露滞后说明（写入文档，不参与计算）
DISCLOSURE_LAG_NOTE = (
    "季报法定披露：Q1 报 4/22 前、半年报 8/31 前、Q3 报 10/22 前、年报次年 4/30 前，"
    "约 15 个工作日。故第 T 期持仓在第 T+1 季度首月即可获得，"
    "不构成对 T+1 季度收益的前视；保守口径另测 T→T+2。"
)

# ============================================================================
# 数据校验阈值
# ============================================================================
VALIDATION_SUM_TOL = 0.01        # 每季度 31 行业配置比例加总与 100 的允许偏离(pct)
VALIDATION_OVER_RANGE = (-20.0, 20.0)   # 超配比例合理区间(pct)，越界=警告+标注
VALIDATION_MIN_FUNDS = 800       # 样本基金数下限
VALIDATION_HHI_RANGE = (0.0, 1.0)        # HHI 值域（开区间）

# ============================================================================
# 因子判断阈值（行业层面）
# ============================================================================
THRESH_Z_EXTREME = 2.0      # Z > 2.0 极度拥挤
THRESH_Z_CROWDED = 1.5      # Z > 1.5 拥挤
THRESH_Z_UNDER = -1.5       # Z < -1.5 显著低配
THRESH_PCT_EXTREME = 90.0   # 超配分位 >= 90 极度拥挤
THRESH_PCT_CROWDED = 75.0   # 分位 75~90 偏拥挤
THRESH_PCT_UNDER = 10.0     # 分位 <= 10 显著低配
THRESH_MOM_Z_ACCEL = 1.0    # 动量Z > 1.0 资金加速流入

# HHI 分级
HHI_LEVELS = [(0.05, "低"), (0.10, "中"), (0.15, "高"), (float("inf"), "极高")]
CR3_HIGH = 60.0             # CR3 > 60% 高度集中

# ============================================================================
# 市场层面信号阈值
# ============================================================================
SIGNAL_HHI_EXTREME = 0.15       # HHI > 0.15 极高集中
SIGNAL_CR3_HIGH = 60.0          # CR3 > 60% 高度集中
SIGNAL_N_EXTREME = 3            # 极度拥挤行业数 >= 3 结构性风险偏高
SIGNAL_N_UNDER = 10             # 显著低配行业数 >= 10 K型分化极致

# 拥挤等级标签（顺序即展示顺序）
LEVELS = ["极度拥挤", "拥挤", "偏拥挤", "中性", "显著低配"]

# ============================================================================
# F11 综合拥挤度权重
# ============================================================================
# 原定权重（含 F7）：F2=0.35, F3=0.30, F4b=0.10, F7=0.15, F9=0.10
# 说明：F7(个股集中度) 是**市场层面时序**指标，同一季度对 31 个行业取同一值，
#       进入行业横截面 z-score 后恒为 0（常数列标准化后无信息）。
#       故行业综合拥挤度只用 4 个"行业内可区分"的分量，权重按原比例重新归一化；
#       F7 的 0.15 权重移至**市场层面**综合拥挤度（见 market_crowding_score）。
W_INDUSTRY_RAW = {
    "F1_超配比例": 0.20,
    "F12_配置系数": 0.25,
    "F3_超配历史分位": 0.15,
    "F13_筹码盈利比例": 0.15,
    "F14_ETF资金流强度": 0.15,
    "F4b_超配动量Z": 0.10,
}
_S = sum(W_INDUSTRY_RAW.values())
W_INDUSTRY = {k: v / _S for k, v in W_INDUSTRY_RAW.items()}
MIN_COMPONENTS = 2          # 合成时至少可用的分量数

W_MARKET_RAW = {"F5_HHI": 0.30, "F6_CR3": 0.25, "F7_个股集中度Z": 0.30,
                "F8_仓位代理Z": 0.15}
_SM = sum(W_MARKET_RAW.values())
W_MARKET = {k: v / _SM for k, v in W_MARKET_RAW.items()}

IC_WEIGHT_LOOKBACK = 8      # IC 加权合成的滚动窗口（季度）

# ============================================================================
# 新增因子阈值（本次升级）
# ============================================================================
THRESH_CF_EXTREME = 2.5       # F12 配置系数 > 2.5 极度拥挤
THRESH_CF_CROWDED = 1.8       # F12 > 1.8 拥挤
THRESH_CF_UNDER = 0.5         # F12 < 0.5 显著低配
THRESH_PROFIT_EXTREME = 80.0  # F13 筹码盈利比例 >= 80% 获利盘拥挤
THRESH_PROFIT_UNDER = 20.0    # F13 <= 20% 深度套牢
THRESH_FLOW_Z = 1.5           # F14 ETF资金流强度 Z > 1.5 显著净申购

# ============================================================================
# 统计方法升级
# ============================================================================
BOOTSTRAP_N = 1000            # Bootstrap 重抽样次数
BOOTSTRAP_BLOCK = 4           # 块大小（季度），保留样本内自相关
BOOTSTRAP_SEED = 20260930     # 固定随机种子 -> 结果可复现
DECAY_WINDOW_DAYS = 252       # 因子衰减滚动窗口（交易日）= 1 年
DECAY_WINDOW_Q = 4            # 等价季度数
DECAY_WINDOWS_Q = (4, 8, 12)  # 衰减分析比较的窗口（季度）
FCI_COMPONENTS = ["z_空头集中度", "z_ETF资金流", "z_因子相关性"]

# ============================================================================
# 数值精度（本次升级要求：数值 2 位小数，百分比 0.01%）
# ============================================================================
# ---------------------------------------------------------------------------
# 精度规则（第 4 轮：解除 L2「一律两位小数」的不合理约束）
# ---------------------------------------------------------------------------
# 原约束「所有数值精确到小数点后两位」对**金额/比例**合理，但对**小量纲比率**
# 不合理：HHI 取值区间只有 0.07~0.19，两位小数会退化成 0.07/0.19 两个值，
# 时间序列信息完全丢失；IC_IR、p 值、相关系数同理。
# 故改为**按列名分层**：默认 2 位，命中「小量纲关键字」的列保留 4 位。
DEC_VALUE = 2
DEC_PCT = 2
DEC_Z = 2
DEC_HHI = 4
DEC_RATIO = 2
DEC_SMALL = 4                      # 小量纲比率的位数

# 命中以下任一关键字的列 → 保留 DEC_SMALL 位
KEEP_SMALL_KEYWORDS = (
    "HHI", "IC", "IR", "p值", "p_NW", "q值", "相关系数", "相关性", "Spearman",
    "Z值", "Z_", "Zscore", "分位", "t值", "NW_t", "momIC", "参与率", "冲击bp",
    "成本bp", "折算倍数", "标准化", "得分", "离散度", "强度",
)
# 命中以下任一关键字的列 → 保持整数
KEEP_INT_KEYWORDS = ("数", "期数", "n", "个数", "只数", "行数", "交易日数", "ETF数")

PRECISION_NOTE = (
    "金额/比例/权重保留 2 位小数；HHI/IC/IR/p 值/相关系数/Z 值/分位等小量纲"
    "保留 4 位；计数类保持整数。")
PROVENANCE_COLS = ["数据来源", "数据表", "查询日期"]

PROVENANCE_COLS = ["数据来源", "数据表", "查询日期"]


# ============================================================================
# 配色（浅色主题，红涨绿跌，符合 A 股习惯）
# ============================================================================
COLOR = {
    "bg": "#FFFFFF",
    "panel": "#F7F9FB",
    "line": "#DCE3EA",
    "grid": "#EEF2F6",
    "text": "#1F2D3D",
    "text_dim": "#6B7A8C",
    "price": "#1F6FEB",        # 主蓝（超配比例）
    "alloc": "#2EA043",        # 绿（配置比例）
    "threshold": "#D1242F",    # 红（P90 阈值）

    # 拥挤等级：极度拥挤=红 → 中性=灰 → 显著低配=蓝
    "level_极度拥挤": "#C62828",
    "level_拥挤": "#E8710A",
    "level_偏拥挤": "#D9A406",
    "level_中性": "#8A97A5",
    "level_显著低配": "#1F6FEB",
    "level_数据不足": "#C3CBD4",

    # 涨跌（A 股习惯：红涨绿跌）
    "up": "#D1242F",
    "down": "#1A7F37",
    "flow_pos": "#D1242F",
    "flow_neg": "#1A7F37",

    "turnover": "#CFE0F5",
    "pct_line": "#E8710A",
    "accent": "#6E4AB8",
    "signal1": "#C62828",
    "signal2": "#E8710A",
    "signal3": "#6E4AB8",
}

PLOTLY_TEMPLATE = "plotly_white"
FONT = "Microsoft YaHei, PingFang SC, Source Han Sans SC, sans-serif"
CHART_H_MAIN = 430
CHART_H_SMALL = 260

# 市场状态标语
MARKET_STATE_RULES = [
    ("K型分化极致", lambda m: m["n_extreme"] >= SIGNAL_N_EXTREME and m["n_under"] >= SIGNAL_N_UNDER),
    ("极高集中", lambda m: m["HHI"] > SIGNAL_HHI_EXTREME and m["CR3"] > SIGNAL_CR3_HIGH),
    ("高度集中", lambda m: m["CR3"] > SIGNAL_CR3_HIGH or m["HHI"] > SIGNAL_HHI_EXTREME),
    ("结构性风险偏高", lambda m: m["n_extreme"] >= SIGNAL_N_EXTREME),
    ("分化明显", lambda m: m["n_under"] >= SIGNAL_N_UNDER),
    ("相对均衡", lambda m: True),
]
