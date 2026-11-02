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
    5. 局限（四条硬约束）
    6. 复现命令

被删掉的内容不是不重要，而是属于「审计留痕」，需要时看 tools/ 下的脚本输出即可。
"""
from __future__ import annotations

import numpy as np
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

    # =====================================================================
    # 结论
    # =====================================================================
    h('<div class="sec">一页看懂：这套数字能信到什么程度</div>')
    h(ui.concl_board([
        '<b>三套口径不能混用，也不能相加。</b>全持仓（半年报）回答「配了多少、'
        '主动还是被动」；前十大（季报）回答「多少基金在加」；个股（同披露两期）'
        '回答「谁被建仓」。页面上每个数字都标了口径与期次。',
        '<b>两个会系统性骗人的坑已经修掉</b>：加仓占比改成<b>价格中性</b>'
        '（原始变化与当期行业收益的秩相关 ρ=0.73，中性化后降到 0.18）；'
        '分位改成<b>扩张窗口</b>（每期只排到当期），否则做检验时会用到未来数据。',
        '<b>它是描述工具，不是预测模型。</b>9 个季报因子在 '
        f'{len(qs)} 期样本上没有一个是正向可用的（见「⑤ 因子检验」）。'
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
    # 样本漏斗
    # =====================================================================
    h('<div class="sec">三、数据从哪来：从 1.5 万只到 4,270 只</div>')
    try:
        _mk = cn.market_fund_counts()
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
    # 五个关键口径
    # =====================================================================
    h('<div class="sec">四、五个关键口径：所有结论都建立在它们上面</div>')
    c1, c2 = st.columns(2, gap="small")
    with c1:
        h(_card("价格中性 Δ（最重要）", [
            ("公式", "反事实权重 = 上期权重 ×(1+行业收益) ÷ 组合收益；"
                     "主动 Δ = 本期权重 − 反事实权重"),
            ("解决什么", "行业涨了 20%、没人交易，权重也会自动升 2pp。不剥离，"
                        "暴跌季会把「没交易」算成「集体减仓」。"),
            ("实测", "原始 Δ 与当期行业收益的秩相关 ρ=0.73；中性化后 0.18。"),
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
            ("定义", "当前值在「2019Q1 至当期」中的排名 ×100"),
            ("为什么扩张", "用全样本排名会让 2019 年的分位用上 2026 年的数据，"
                          "做检验时 IC 被系统性高估。"),
            ("注意", "最后一期与全样本完全相等，所以当期数字不变，只有历史序列会变。"),
        ], C.COLOR["accent"]))
    with c4:
        h(_card("CSI 拥挤度", [
            ("公式", "(配置分位 + 价格分位 + 资金分位) ÷ 3，等权"),
            ("为什么等权", "三维已统一为分位；31 行业 × 31 期不足以稳定估计权重。"),
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
    # 自检
    # =====================================================================
    h('<div class="sec">五、自检：12 条口径恒等式，现场跑一遍</div>')
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
          '不是「看起来差不多」。任何一处口径改动，这里会立刻变红。</div>')
    except Exception as _e:
        h(f'<div class="tiny">自检脚本未通过：{type(_e).__name__}: {_e}</div>')

    # =====================================================================
    # 局限
    # =====================================================================
    h('<div class="sec">六、四条硬约束（不写清楚就等于骗人）</div>')
    h(ui.concl_board([
        '<b>季报只有前十大。</b>Q1/Q3 每只基金只披露 10 只股票，覆盖净值约 20%–50%。'
        '所以季报期的配置一律用来算广度与变化，不用来判断绝对水平。',
        '<b>披露有时滞。</b>季末到数据可得约 3–4 周，你看到时下季度已经过了约 1/3。'
        '所有「本期」都是「上一个已披露期」，不是「今天」。',
        '<b>幸存者偏差。</b>已清盘基金不在库内，早期分位会偏乐观——'
        '活下来的基金天然仓位更高、更集中。',
        '<b>只有股票，没有债券与现金。</b>灵活配置型基金的真实股票仓位可能远低于'
        '「各行业之和」，行业占比会被系统性放大。',
    ], numbered=False))

    with st.expander("复现命令与数据文件", expanded=False):
        st.code("""python tools/run_ak_all.py        # 抓全市场持仓（东财，可断点续传）
python tools/export_canon.py      # 导出压缩快照（无数据库时线上可用）
python tools/export_matrix.py     # 预计算行业矩阵与仓位分布
python tools/export_csi.py        # 预计算 CSI 面板（31×31）
python tools/run_factor_test.py   # 生成因子检验结果 factor_test.json
python -c "import sys; sys.path.insert(0,'src'); import selfcheck; print(selfcheck.run())"
""", language="bash")
        h('<div class="tiny">数据文件：<span class="mono">data/db/fund_ak.db</span>'
          '（SQLite，约 900MB，不入库）；派生快照在 '
          '<span class="mono">data/derived/*.csv.gz</span>（约 12MB，随仓库发布）。'
          '线上环境没有数据库时自动读快照，两条路径的计算结果已逐格核对一致。</div>')

    h('<div class="tiny" style="margin-top:10px">'
      f'版本口径：季度 {qs[0]}–{qs[-1]}（{len(qs)} 期）、全持仓 {len(fq)} 期、'
      f'指数 2019-01 起；样本 {n_funds:,} 只主动权益基金。'
      '更细的审计留痕（数据源探查记录、逐期溯源）见仓库 tools/ 下的脚本输出。</div>')
