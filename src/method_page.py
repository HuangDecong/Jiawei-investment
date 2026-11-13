# -*- coding: utf-8 -*-
"""⑥ 方法与口径：只保留「会影响结论」的内容。

为什么重写
----------
上一版把数据源探查记录、Wind 接口日志、覆盖率矩阵、逐期溯源明细全部堆在一页，
接近一千行，读者翻三屏还看不到「口径到底是什么」。这一版按重要性重排：

    1. 三套口径（决定一切结论）
    2. 数据从哪来（样本漏斗，回答「为什么只有 4,270 只」）
    3. 五个关键口径（价格中性、广度、分位、CSI、GK）
    4. 自检（现场跑 12 条恒等式）
    5. 局限（六条硬约束）
    6. 复现命令

被删掉的内容不是不重要，而是属于「审计留痕」，需要时看 tools/ 下的脚本输出即可。
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

import canon as cn
import config as C
import ui

FS = 13.5


def _card(title: str, rows: list[tuple[str, str]], color: str) -> str:
    body = "".join(
        f'<div style="display:flex;gap:10px;padding:6px 0;'
        f'border-top:1px solid {C.COLOR["grid"]}">'
        f'<div style="flex:0 0 76px;font-size:12.5px;font-weight:700;color:{color};'
        f'line-height:1.85">{k}</div>'
        f'<div style="flex:1;font-size:{FS}px;line-height:1.9;color:#26374A">{v}</div>'
        f'</div>' for k, v in rows)
    return (f'<div style="background:#FFFFFF;border:1px solid {C.COLOR["line"]};'
            f'border-top:4px solid {color};border-radius:9px;padding:10px 14px;'
            f'height:100%"><div style="font-size:14.5px;font-weight:700;'
            f'color:#16283C">{title}</div>{body}</div>')


def render(h, B=None, IND=None) -> None:
    if not (cn.available() or cn.fallback_available()):
        h('<div class="cg-note">未找到持仓数据。</div>')
        return
    q = cn.latest()
    qs = cn.quarters()
    fq = cn.full_quarters()
    n_funds = len(cn.universe(q))

    # 价格漂移贡献：现算，避免写死「ρ=0.73→0.18」这种换个样本就复现不了的数
    _pe = cn.price_effect()
    # 这里是**逐基金配对平均**（两期都可得才计入），与 ① 的「全样本合计」略有差异；
    # 两个数都对，但口径不同，必须写清楚，否则同一只电子会出现 -4.93 与 -5.51。
    _pe_txt = (f"最近一期 {_pe['q0']}→{_pe['q']}（逐基金配对平均）：行业平均 "
               f"|价格漂移| 占 |原始变化| 的 {_pe['share']:.0f}%；"
               f"{_pe['top']} {_pe['raw']:+.2f}pp 里 {_pe['mech']:+.2f}pp 是价格，"
               f"占 {_pe['pct']:.0f}%") if _pe else ""

    # =====================================================================
    # 结论
    # =====================================================================
    h('<div class="sec">一页看懂：这套数字能信到什么程度</div>')
    h(ui.concl_board([
        '<b>三套口径不能混用，也不能相加。</b>全持仓（半年报）回答「配了多少、'
        '主动还是被动」；前十大（季报）回答「多少基金在加」；个股（同披露两期）'
        '回答「谁被建仓」。页面上每个数字都标了口径与期次。',
        '<b>两个会系统性骗人的坑已经修掉</b>：加仓占比改成<b>价格中性</b>'
        + (f'（{_pe_txt}）；' if _pe_txt else '（行业涨跌会直接改写权重）；') +
        '分位改成<b>扩张窗口</b>（每期只排到当期），否则做检验时会用到未来数据。',
        '<b>它是描述工具，不是预测模型。</b>9 个季报因子在 '
        f'{len(qs)} 期可得样本上没有一个是正向可用的（各因子有效 n 见「⑤ 因子检验」）。'
        '这套系统的价值是读懂机构在做什么、避免在拥挤处接棒、避免在撤离处抄底。',
    ]))
    h(ui.stat_strip([
        ("样本基金", f"{n_funds:,}", f"{q} 主动权益"),
        ("期数", f"{len(qs)}", f"{qs[0]}–{qs[-1]}"),
        ("全持仓期", f"{len(fq)}", "半年报，可做 GK 分解"),
        ("行业", "31", "申万一级"),
    ]))

    # =====================================================================
    # 三套口径
    # =====================================================================
    h('<div class="sec">二、三套口径：每套只回答一个问题</div>')
    _cl = pd.DataFrame([
        ["A 全持仓", "半年报（Q2/Q4）", "占净值比，各行业之和＝股票仓位",
         "配置水平、超配低配、主动/被动分解", "一年只有两次，看不到季度变化"],
        ["B 前十大", "季报（Q1–Q4）", "每只基金取最重的 10 只后归一化到 100%",
         "加仓广度、环比/同比、历史分位、CSI", "覆盖净值 20%–50%，绝对水平不可用"],
        ["C 个股", "同披露类型两期", "持有基金数、持股数、持仓市值",
         "谁被建仓、谁被移出", "跌出前十 ≠ 清仓"],
    ], columns=["口径", "频率", "权重定义", "回答什么", "不能回答什么"])
    st.dataframe(_cl, width="stretch", hide_index=True, height=190,
                 column_config={
                     "口径": st.column_config.TextColumn(width="small"),
                     "权重定义": st.column_config.TextColumn(width="medium"),
                     "回答什么": st.column_config.TextColumn(width="medium"),
                     "不能回答什么": st.column_config.TextColumn(width="medium")})
    h('<div class="cg-note" style="font-size:12.5px">'
      '<b>为什么不能相加</b>：A 的分母是基金净值，B 的分母是前十大市值，'
      'C 的单位是「只数」。把「全持仓 3.2%」和「前十配置 5.1%」相加没有任何含义。'
      '<br><b>为什么季报也能算环比</b>：季报（Q1/Q3）法定只披露前十大，'
      '半年报/年报披露全部持仓。对四个季度统一只取「每只基金最重的 10 只」，'
      '披露范围就一致了——这不是跨口径拼接，而是把全持仓截到前十大。</div>')

    # =====================================================================
    # 口径表（代码同源）
    # =====================================================================
    import caliber as CAL  # noqa: PLC0415
    import version_info as vinfo  # noqa: PLC0415
    h(f'<div class="sec">口径表：每一条都只有一个定义（v{CAL.CALIBER_VERSION}）</div>')
    h('<div class="tiny">这张表由 <span class="mono">src/caliber.py</span> 直接生成，'
      '页面上所有阈值、窗口、分母都从同一个地方取——'
      '过去「口径」类问题被修了 17 次，就是因为同一句话在七个文件里各写一遍。'
      '改口径只能改 caliber.py，改完 <span class="mono">tools/check_caliber.py</span> '
      '会在核心文件里 grep 硬编码，命中即失败。</div>')
    # 七条口径改成卡片，而不是一张七列的宽表：
    # 「口径定义」一列在 1200px 里只能分到 200px，长句被压成一列竖排的字，
    # 读者反馈「看不清」。卡片按两列排，定义占满卡片宽度，分母/最小样本/
    # 缺失处理放同一行元信息，七条一眼能扫完。
    _cal = CAL.table()
    _cards = []
    for _r in _cal:
        _cards.append(
            f'<div style="flex:1 1 48%;min-width:420px;background:#FFFFFF;'
            f'border:1px solid {C.COLOR["line"]};border-radius:9px;padding:10px 13px">'
            f'<div style="display:flex;justify-content:space-between;gap:8px;'
            f'align-items:baseline">'
            f'<div style="font-size:13.5px;font-weight:700;color:#1B2A3A">'
            f'{_r["name"]}'
            f'<span style="font-family:ui-monospace,Consolas,monospace;font-size:10.5px;'
            f'color:{C.COLOR["text_dim"]};margin-left:6px">{_r["metric_id"]}</span></div>'
            f'<div style="font-size:10.5px;color:{C.COLOR["text_dim"]};'
            f'white-space:nowrap">{_r["version"]}</div></div>'
            f'<div style="font-size:12px;line-height:1.75;color:#37485C;'
            f'margin:4px 0 5px">{_r["definition"]}</div>'
            f'<div style="font-size:11px;line-height:1.7;color:{C.COLOR["text_dim"]};'
            f'border-top:1px dashed {C.COLOR["grid"]};padding-top:4px">'
            f'分母：{_r["denominator"]}　·　最小样本：{_r["min_n"]}'
            f'<br>缺失处理：{_r["missing"]}</div></div>')
    h('<div style="display:flex;flex-wrap:wrap;gap:9px;margin:4px 0 8px">'
      + "".join(_cards) + '</div>')

    # =====================================================================
    # 样本漏斗
    # =====================================================================
    try:
        _mk = cn.market_fund_counts()
        h(f'<div class="sec">三、数据从哪来：从 {_mk.get("全市场基金", 0):,} 只到 '
          f'{len(cn.universe(q)):,} 只</div>')
        _pool = cn.profiles()
        _cov = cn.coverage(q)
        _eq = cn.equity_codes()
        _keep = _cov.index.intersection(list(_eq)) if _eq else _cov.index
        _good = _cov.loc[_keep]
        _good = _good[(_good >= cn.COV_MIN) & (_good <= cn.COV_MAX)]
        _uni = cn.universe(q)
        _tot = _mk.get("主动权益基金") or 0
        _rows = [
            ("全市场公募基金", _mk.get("全市场基金"), 0,
             f"东财名录 {_mk.get('全市场份额', 0):,} 条份额，按名称合并 A/C/后端；"
             f"含债、货币、指数、QDII、FOF、REITs、商品"),
            ("其中主动权益", _tot, 1,
             "普通股票 + 偏股混合 + 灵活配置 + 平衡混合；其余类型要么不投 A 股，"
             "要么是复制指数（被动资金不算「机构观点」）"),
            ("本库抓取", len(_pool), 1,
             f"覆盖主动权益的 {len(_pool) / _tot * 100:.1f}%；缺口主要是「后端」收费模式"
             f"（与 A 类同一只基金）与新发未建仓"),
            ("当期有持仓", len(_cov), 2,
             f"{q} 无披露的 {len(_pool) - len(_cov)} 只：新发未建仓、已清盘、停牌"),
            ("主动权益且覆盖率合规", len(_good), 3,
             f"再剔 {len(_cov) - len(_good)} 只：非权益类型、持仓市值÷净值 &gt;105% 或 &lt;1%"),
            ("最终样本", len(_uni), 4,
             f"再剔 {len(_good) - len(_uni)} 只主题基金（单行业占前十大 &gt;80%，"
             f"仓位变动反映申赎而非行业判断）"),
        ]

        def _row_html(idx, nm, n, lv, note):
            _bg = ' style="background:#FAFBFC"' if idx % 2 else ""
            _ind = "　" * lv + ("└ " if lv else "")
            return (f"<tr{_bg}>"
                    f'<td style="padding:6px 9px;font-size:13px">{_ind}{nm}</td>'
                    f'<td style="padding:6px 9px;font-size:13.5px;font-weight:700;'
                    f'text-align:right;font-variant-numeric:tabular-nums">{n:,}</td>'
                    f'<td style="padding:6px 9px;font-size:12px;color:#5A6B7D">{note}</td>'
                    f"</tr>")

        _body = "".join(_row_html(i, nm, n, lv, note)
                        for i, (nm, n, lv, note) in enumerate(_rows) if n)
        h(f'<div style="border:1px solid {C.COLOR["line"]};border-radius:9px;'
          f'overflow:hidden"><table style="width:100%;border-collapse:collapse">'
          f'<tr style="background:{C.COLOR["panel"]}">'
          f'<th align="left" style="padding:7px 9px;font-size:12.5px">层级</th>'
          f'<th align="right" style="padding:7px 9px;font-size:12.5px">只数</th>'
          f'<th align="left" style="padding:7px 9px;font-size:12.5px">为什么少了</th></tr>'
          f'{_body}</table></div>')
        h(f'<div class="tiny">名录口径：东财基金列表，截至 {_mk.get("截至", "—")}'
          f'（新发与清盘会让这个数每季变动几十只）。'
          f'「只数」是<b>基金</b>不是<b>份额</b>——同一只基金的 A/C/后端持仓完全相同，'
          f'按份额统计会让广度被同一只基金重复计数。</div>')
    except Exception as _e:
        h(f'<div class="tiny">样本漏斗暂不可用：{type(_e).__name__}</div>')

    # =====================================================================
    # 数据来源清单：字段级
    # =====================================================================
    h('<div class="sec">四、数据来源与字段对照：每个数字从哪个接口、哪张表来</div>')
    # 名录条数用实时统计，不写死：数据更新后这里跟着变
    try:
        _mkc = cn.market_fund_counts()
    except Exception:  # noqa: BLE001
        _mkc = {}
    _n_share = _mkc.get("全市场份额")
    _n_fund = _mkc.get("全市场基金")
    _mk_txt = (f"全市场 {_n_share:,} 条份额记录，按名称合并 A/C/后端"
               f"（{_n_fund:,} 只基金）" if _n_share and _n_fund
               else "全市场基金名录，按名称合并 A/C/后端")
    _src_rows = [
        ("基金名录 / 类型 / 成立日 / 管理人",
         "东方财富基金列表（天天基金）",
         "基金合同、招募说明书",
         _mk_txt,
         "每日", "无档案 → 不给结论"),
        ("资产配置（股票 / 债券 / 现金 / 净资产）",
         "东财 F10 · 资产配置",
         "季报 §5 资产组合",
         "4,941 只 × 39 期（2017Q1–2026Q3）",
         "每季，披露后 15 个工作日内", "未列示写「—」，不补 0"),
        ("前十大持仓（代码 / 名称 / 占比 / 股数）",
         "东财 F10 · 持仓明细",
         "季报 §5.3 前十名股票",
         "4,937 只，合计 100 万行",
         "每季", "该期未取到 → 提示并可补抓"),
        ("行业配置",
         "东财 F10 · 行业配置",
         "季报 §5.2 按行业分类（证监会门类，港股通单列）",
         "26.7 万行；每期 6–8 个门类",
         "每季", "同上"),
        ("份额变动（申购 / 赎回 / 期末份额 / 净值增长率）",
         "东财 F10 · 规模份额",
         "季报 §6 份额变动",
         "含季末披露日；早期有周度披露日，取季末日",
         "每季", "无季末行 → 「—」"),
        ("主要财务指标（本期利润 / 已实现收益）",
         "东财 F10 · 财务指标",
         "半年报 §3 / 年报",
         "17.9 万行，仅半年报与年报有",
         "每半年", "季报期「未披露」（法规不要求）"),
        ("行业指数日收益（申万一级）",
         "Wind 金融终端导出",
         "申万宏源研究 · 申万一级行业指数",
         "31 个行业，2015-01 至 2026-10-28",
         "手动更新（缓存文件）", "缺行业 → 该行业不评级"),
        ("行业分类标准",
         "申万宏源研究 · 申万一级（31 个）",
         "《申万行业分类标准》2021 版",
         "美容护理 / 石油石化 / 环保为 2021 新设",
         "随指数更新", "新设行业历史不足 → 不入排名"),
    ]
    def _src_row(i, r):
        # 注意：不在 f-string 的表达式里写反斜杠——那是 Python 3.12+ 才允许的语法，
        # 云端若跑 3.11 会直接 SyntaxError。这里先算好属性串再拼。
        _bg = ' style="background:#FAFBFC"' if i % 2 else ""
        _cells = "".join(
            f'<td style="padding:6px 9px;font-size:12px;vertical-align:top">{v}</td>'
            for v in r)
        return f"<tr{_bg}>{_cells}</tr>"

    _src_html = "".join(_src_row(i, r) for i, r in enumerate(_src_rows))
    h(f'<div style="border:1px solid {C.COLOR["line"]};border-radius:9px;overflow-x:auto">'
      f'<table style="width:100%;border-collapse:collapse;min-width:900px">'
      f'<tr style="background:{C.COLOR["panel"]}">'
      + "".join(f'<th align="left" style="padding:7px 9px;font-size:12px;'
                f'white-space:nowrap">{t}</th>'
                for t in ("数据项", "来源接口", "原始披露位置", "覆盖范围", "更新频率", "缺失处理"))
      + f'</tr>{_src_html}</table></div>')
    h('<div class="tiny" style="margin:4px 0 6px">'
      '可点核对的入口：'
      '<a href="http://eid.csrc.gov.cn/fund" target="_blank" rel="noopener">'
      '证监会基金电子披露网站</a>（原始季报 PDF）· '
      '<a href="https://fundf10.eastmoney.com/" target="_blank" rel="noopener">'
      '东方财富基金档案 F10</a>（本库转引来源）· '
      '<a href="https://www.wind.com.cn/" target="_blank" rel="noopener">Wind 金融终端</a>'
      '（行业指数）· '
      '申万宏源研究（行业分类标准；官网当前返回 5xx，<b>不做死链</b>，'
      '分类口径见上表「行业分类标准」一行）。<br>'
      '<b>本库不使用任何代理口径</b>：不用前十大反推全持仓、不用行业指数替代个股收益、'
      '不对缺失值做插值或补 0；每一格要么是披露值，要么是「—/未披露」。</div>')

    # =====================================================================
    # 五个关键口径
    # =====================================================================
    h('<div class="sec">五、五个关键口径：所有结论都建立在它们上面</div>')
    c1, c2 = st.columns(2, gap="small")
    with c1:
        h(_card("价格中性 Δ（最重要）", [
            ("公式", "反事实权重 = 上期权重 ×(1+行业收益) ÷ 组合收益；"
                     "主动 Δ = 本期权重 − 反事实权重"),
            ("解决什么", "行业涨了 20%、没人交易，权重也会自动升 2pp。不剥离，"
                        "暴跌季会把「没交易」算成「集体减仓」。"),
            ("实测", _pe_txt or "行业涨跌会直接改写权重，需用反事实权重剥离。"),
        ], C.COLOR["price"]))
    with c2:
        h(_card("加仓广度", [
            ("公式", "加仓基金数 ÷ (加仓 + 减仓) × 100%"),
            ("分母", "只含有实质变动（±0.1pp）的基金，不含不变与不持有"),
            ("为什么", "数人头而不是数钱：一只百亿基金加 3pp 与两百只小基金各加 3pp，"
                      "含义完全相反，广度对规模免疫。"),
        ], C.COLOR["up"]))
    c3, c4 = st.columns(2, gap="small")
    with c3:
        h(_card("分位（扩张窗口）", [
            ("定义", "当前值在「首期至当期」中的排名 ×100"),
            ("为什么扩张", "用全样本排名会让早期的分位用上后来的数据，"
                          "做检验时 IC 被系统性高估。"),
            ("注意", "最后一期与全样本完全相等，所以当期数字不变，只有历史序列会变。"),
        ], C.COLOR["accent"]))
    with c4:
        h(_card("CSI 拥挤度", [
            ("公式", "(配置分位 + 价格分位 + 资金分位) ÷ 3，等权"),
            ("为什么等权", "三维已统一为分位；31 行业 × 全样本期不足以稳定估计权重。"),
            ("小样本", "持有基金数 &lt;150 或前十配置 &lt;0.5% 的行业不评级，"
                      "只显示灰色。"),
        ], C.COLOR["threshold"]))
    c5, c6 = st.columns(2, gap="small")
    with c5:
        h(_card("GK 主动/被动分解", [
            ("公式", "被动贡献 = 反事实权重 − 上期权重；主动贡献 = 本期 − 反事实权重"),
            ("频率", "只在半年报（全持仓）做，季报做噪声太大"),
            ("用途", "回答「配置变化里有多少是机构真的在买卖」"),
        ], C.COLOR["text_dim"]))
    with c6:
        h(_card("口径边界", [
            ("季报", "只有前十大，覆盖净值 20%–50%，绝对水平不可用"),
            ("时滞", "法定 15 个工作日，本库实测约 28 天"),
            ("不含", "债券、现金明细、季报正文；只有股票"),
        ], C.COLOR["level_拥挤"]))

    # =====================================================================
    # 指标词典（全量，可检索）
    # =====================================================================
    import glossary as g
    _ok, _all = g.coverage()
    h(f'<div class="sec">六、指标词典：页面里出现的每个数，这里都有定义</div>')
    h(ui.stat_strip([
        ("词典条目", f"{len(g.ITEMS)}", "按 10 类组织，每条九段"),
        ("术语覆盖", f"{_ok}/{_all}", "页面文案里出现过的指标与术语"),
        ("每条的字段", "9", "含义/原理/公式/步骤/阈值/来源/经济/投资/误用"),
        ("检索", "实时", "输入关键词即可，侧边栏同一份"),
    ]))
    if _ok < _all:
        _miss = [t for t in g.AUDIT_TERMS if not g.search(t)]
        h(f'<div class="cg-note">以下术语尚未收录：{"、".join(_miss[:12])}</div>')
    h('<div class="tiny" style="margin:-2px 0 8px">'
      '这份词典是<b>唯一事实源</b>：侧边栏速查、图上的悬浮说明、本页表格都从这里取，'
      '改一处全站同步。术语覆盖数是<b>审计结果</b>——页面上写到的每个指标和术语，'
      '都必须能在下面检索到；缺一条，审计脚本就会报错。</div>')
    _kw = st.text_input("搜索指标或术语", value="", key="dict_kw",
                        placeholder="如：净申购 / 规模拆解 / CSI / 分位 / 前十大 / 兑现比例")
    _keys = g.search(_kw) or list(g.ITEMS)
    if _kw and not g.search(_kw):
        h(f'<div class="cg-note">没有匹配「{_kw}」的条目。'
          f'试试更短的关键词，或直接翻下面的全量清单。</div>')
    _pick = st.selectbox("选择条目，展开完整说明（九段）", _keys,
                         format_func=g.label, key="dict_pick")
    st.markdown(g.card_html(_pick, C.COLOR), unsafe_allow_html=True)
    with st.expander(f"全量清单：{len(g.ITEMS)} 条，按分类（一屏看完都有哪些）",
                     expanded=False):
        st.markdown(g.compact_table(C.COLOR), unsafe_allow_html=True)

    # =====================================================================
    # 自检
    # =====================================================================
    h('<div class="sec">七、自检：12 条口径恒等式，现场跑一遍</div>')
    try:
        import selfcheck as _sc
        _t = _sc.run()
        _pass = int(_t["结果"].astype(str).str.contains("通过").sum())
        h(f'<div style="font-size:14px;font-weight:700;margin:2px 0 6px;'
          f'color:{C.COLOR["up"] if _pass == len(_t) else C.COLOR["threshold"]}">'
          f'{_pass}/{len(_t)} 通过</div>')
        st.dataframe(_t, width="stretch", hide_index=True, height=420,
                     column_config={
                         "实测值": st.column_config.TextColumn(width="small"),
                         "结果": st.column_config.TextColumn(width="small"),
                         "为什么查这条": st.column_config.TextColumn(width="large"),
                     })
        h('<div class="tiny">这些检查全部是「左边必须等于右边」的恒等式，'
          '不是「看起来差不多」。任何一处口径改动，这里会立刻变红。'
          '除这 12 条外，<span class="mono">python tools/audit_all.py</span> 还会跑'
          '「代码可编译、无未定义名、六页渲染无异常、页面无空值与未渲染标签、'
          '控件逐项点击、词典完整性与术语覆盖、跨页数字一致」等全部检查。</div>')
    except Exception as _e:
        h(f'<div class="tiny">自检脚本未通过：{type(_e).__name__}: {_e}</div>')

    # ---- 数据体检（源数据层）：计算对，不代表上游披露没有瑕疵 ----
    _dh_fn = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "derived", "data_health.json")
    if os.path.exists(_dh_fn):
        try:
            with open(_dh_fn, encoding="utf-8") as f:
                _dh = json.load(f)
            _bad = int(_dh.get("bad", 0))
            with st.expander(f"数据体检：源数据层 {len(_dh.get('rows', []))} 条硬规则"
                             f"（{_bad} 项需关注）", expanded=False):
                _dd = pd.DataFrame(_dh["rows"])
                st.dataframe(_dd, hide_index=True, width="stretch", height=330,
                             column_config={
                                 "层": st.column_config.TextColumn(width="small"),
                                 "结果": st.column_config.TextColumn(width="small"),
                                 "说明": st.column_config.TextColumn(width="large"),
                             })
                h(f'<div class="tiny">这是「计算对不对」之外的「源数据有没有瑕疵」：'
                  f'生成于 {_dh.get("generated", "—")}。'
                  f'「需关注」不是代码算错，而是上游披露本身的问题（如 1 行负申购、'
                  f'港股分类的分母差异）——本库原样保留、不篡改，并在对应页面标注口径。</div>')
        except Exception:
            pass

    # =====================================================================
    # 局限
    # =====================================================================
    h('<div class="sec">八、六条硬约束（不写清楚就等于骗人）</div>')
    # 样本数写死会在下一季变成错数；这里按当期实际 universe 现算。
    _q_now = cn.latest()
    _n_q = len(cn.universe(_q_now))
    _n_15 = len(cn.universe("2015Q1"))
    _n_19 = len(cn.universe("2019Q1"))
    h(ui.concl_board([
        '<b>季报只有前十大。</b>Q1/Q3 每只基金只披露 10 只股票，覆盖净值约 20%–50%。'
        '所以季报期的配置一律用来算广度与变化，不用来判断绝对水平。',
        '<b>披露有时滞。</b>季末到数据可得约 3–4 周，你看到时下季度已经过了约 1/3。'
        '所有「本期」都是「上一个已披露期」，不是「今天」。',
        '<b>幸存者偏差 + 样本量递减。</b>已清盘基金不在库内，早期分位会偏乐观——'
        f'活下来的基金天然仓位更高、更集中。样本数逐期变化：2015Q1 只有 {_n_15:,} 只、'
        f'2019Q1 {_n_19:,} 只、{_q_now} {_n_q:,} 只；早期每期的标准误更大，'
        '看方向可以，看小数点后一位没有意义。',
        '<b>只有股票，没有债券与现金。</b>灵活配置型基金的真实股票仓位可能远低于'
        '「各行业之和」，行业占比会被系统性放大。',
        '<b>A+H 两地上市按「单只占比」排序。</b>官方季报对同一家公司的 A 股与 H 股'
        '是「合并公允价值参与排序、按不同股票分别披露」，本库只能按单只占比排；'
        '这类公司的名次会与原文错位（全库 473 个基金×季度），数值本身不受影响。',
        '<b>仓位中位数剔除打新基金。</b>2016–2018 年有大量灵活配置型「打新基金」，'
        '股票只放 20–30%；若一并统计，会把当年中位数压到 76%，'
        '让今天的 89% 显得「史上最高」。本库只统计股票仓位 ≥40% 的基金，'
        '口径写在仓位图下方。',
    ], numbered=False))

    with st.expander("复现命令与数据文件", expanded=False):
        st.code("""python tools/run_ak_all.py        # 抓全市场持仓（东财，可断点续传）
python tools/export_canon.py      # 导出压缩快照（无数据库时线上可用）
python tools/export_matrix.py     # 预计算行业矩阵与仓位分布
python tools/export_csi.py        # 预计算 CSI 面板（全期 × 31 行业）+ 三项分位长表
python tools/run_factor_test.py   # 生成因子检验结果 factor_test.json
python tools/batch_qreport.py     # 预抓季报（资产配置/行业分类/份额/买卖/债券）
python tools/repair_qreport.py    # 修复「有仓位却没明细」的基金（列错位脏数据）
python tools/repair_shares.py     # 份额按 A/C 合并（否则规模拆解会算错份额净值）
python tools/data_health.py       # 数据体检：边界/恒等式/结构 三层硬规则
python tools/verify_hold_rows.py  # 深度校验：逐只股票对比源表（联网约 10 分钟）
python tools/verify_pdf.py        # 原文核对：下载季报 PDF，比对前十大与份额总额
python tools/audit_glossary.py    # 指标词典：九段完整性 + 页面术语覆盖
python tools/audit_all.py         # 全站审计：静态/口径/渲染/文本/交互/一致性
python tools/test_deploy_paths.py # 部署路径：没有数据库 + 缺依赖时，六页是否仍然可用
python -c "import sys; sys.path.insert(0,'src'); import selfcheck; print(selfcheck.run())"
""", language="bash")
        h('<div class="tiny">窗口：<b>2015Q1 起</b>——东财持仓明细最早覆盖到这里。'
          '2015 年的行业季度收益由申万日行情累乘得到，与 Wind 在 2016 年后的重叠期逐季核对'
          '（28 个行业完全相同）；<b>美容护理 / 石油石化 / 环保</b> 是 2021 年新设的一级行业，'
          '2022 年前没有指数历史，这三年只按「配置 + 资金」两项展示、不评级。'
          '样本是「当前存续的主动权益基金」，已清盘基金无法取得历史，见上文局限。<br>'
          '数据文件：<span class="mono">data/db/fund_ak.db</span>'
          '（SQLite，约 1.0GB，不入库）；派生快照在 '
          '<span class="mono">data/derived/*.csv.gz</span>（约 23MB，随仓库发布）。'
          '线上环境没有数据库时自动读快照，两条路径的计算结果已逐格核对一致。<br>'
          '季报页另有 <span class="mono">data/db/fund_q.db</span>：只存季报披露值'
          '（股票/债券/现金占净比、证监会门类行业配置、份额变动、前十名股票、'
          '累计买卖前 20、债券明细），<b>不做任何推算</b>；'
          '每只基金抓取后缓存 20 小时，字段级来源记在每行的 src 列。'
          '资产配置与份额一次返回全历史；行业/持仓按年补抓（页面选到早年季度会自动抓，'
          '也可以点「补齐缺失年份」一次抓完），抓到即写库。</div>')

    h('<div class="tiny" style="margin-top:10px">'
      f'版本口径：季度 {qs[0]}–{qs[-1]}（{len(qs)} 期）、CSI 可算 '
      f'{max(len(qs) - 7, 0)} 期（前 7 期历史不足 8 个观测；不在这里现场算 CSI，'
      f'免得首次打开要等两分钟）、全持仓 {len(fq)} 期、'
      f'指数 2015-01 起；样本 {n_funds:,} 只主动权益基金。'
      '更细的审计留痕（数据源探查记录、逐期溯源）见仓库 tools/ 下的脚本输出。</div>')

    ui.source_footer(h, "method", "csi")
