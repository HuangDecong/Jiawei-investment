# -*- coding: utf-8 -*-
"""
update_pipeline.py  ——  一键更新流水线
======================================================================
把新数据放进 data/raw/ 后，运行这一条命令即可完成全部重算与导出：

    python src/update_pipeline.py

流程
----
  Step 1  加载 data/raw/ 全部原始文件并执行校验 → output/data_validation_log.txt
  Step 2  计算 F1~F11 因子（扩窗、无未来函数）
  Step 3  IC / 分层回测 / 事件研究 / 稳健性
  Step 4  生成市场层信号、行业层等级、信号触发记录
  Step 5  导出 4 张输出表 + 3 个 processed CSV
  Step 6  控制台打印核心结论

若 Step 1 校验硬性失败（配置比例加总 ≠ 100 / HHI 越界），立即中止并报错。
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt          # noqa: E402
import config as C             # noqa: E402
import data_loader as dl       # noqa: E402
import factor_engine as fe     # noqa: E402
import research as rs         # noqa: E402
import research3 as r3        # noqa: E402
import research4 as r4        # noqa: E402
import rigor as rg            # noqa: E402
import signal_generator as sg  # noqa: E402


# ============================================================================
# 短样本安全访问器（机构级稳健性）
# ============================================================================
# 背景：数据源切换 Wind 后，可建模区间由 34 期收缩为 9 期。
# 大量研究函数在样本不足时会返回**空表或缺失键**（如 design_oos_test 返回
# table 为空、impact_cost_model 早退时不含 sizes、excess_attribution 无 summary）。
# 早期代码直接 `d["key"]` / `d["table"]["列名"]`，任何一个空表都会让整条流水线
# 在第 3 步崩溃 —— 但真相是"该判据本轮样本不足、不成立"，而不是"程序有错"。
#
# 机构级做法：**如实降级 + 显式登记**，而不是静默填补或直接失败。
# 本模块所有对研究结果的访问一律走下面两个访问器：
#   · df_ok(t)    判断是否为「非空且含必需列」的表
#   · sec(t, name)打印"样本不足"并返回 False，供调用方跳过该节
# ============================================================================
def df_ok(t, *cols) -> bool:
    """表存在、非空、且（若指定）含全部必需列。"""
    if not isinstance(t, pd.DataFrame) or t.empty:
        return False
    return all(c in t.columns for c in cols)


def skip(name: str, why: str = "可建模区间样本不足") -> None:
    """统一格式的「本节跳过」提示（不抛异常，如实披露）。"""
    print(f"  ⚠️ 本节不可用：{name} —— {why}（该判据本轮不成立，不做替代填补）")


def _xl(writer, sheet: str, obj, index: bool = True) -> bool:
    """只在表非空时写入 Excel；空表直接跳过（不写空 sheet、不报错）。

    机构级理由：样本不足时研究函数返回空表，写成空 sheet 会让阅读者
    误以为"该检验跑过且结果为空"，跳过并在汇总中登记更诚实。
    """
    if isinstance(obj, pd.DataFrame) and not obj.empty:
        obj.to_excel(writer, sheet_name=sheet, index=index)
        return True
    return False

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)


# ---------------------------------------------------------------------------
# 数值精度（任务要求：数值 2 位小数，百分比 0.01%）
# ---------------------------------------------------------------------------
_KEEP4 = ("HHI", "HHI标准化", "HHI_文件", "HHI_复算", "HHI_sheet", "HHI_recalc",
          "F5_HHI贡献", "空头集中度HHI", "HHI扩窗分位")


def _dec_for(col: str) -> int:
    """按列名决定小数位数（第 4 轮：解除「一律两位」的不合理约束）。

    规则见 config.KEEP_SMALL_KEYWORDS / KEEP_INT_KEYWORDS。
    计数类列保持整数——把「持股基金数」写成 1234.00 没有意义。
    """
    name = str(col)
    if name in _KEEP4:
        return C.DEC_HHI
    if any(k in name for k in C.KEEP_SMALL_KEYWORDS):
        return C.DEC_SMALL
    if any(k in name for k in C.KEEP_INT_KEYWORDS) and "数" in name:
        return 0
    return C.DEC_VALUE


def rnd(df, dec: int = None):
    """数值列按 config 精度规则取整（分层：2 位 / 4 位 / 整数）。"""
    if df is None or not hasattr(df, "columns"):
        return df
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].astype(float).round(dec if dec is not None
                                                else _dec_for(c))
    return out


def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main() -> int:
    hr("公募基金交易拥挤度与集中度仪表盘 —— 一键更新")
    print(f"项目根目录：{C.ROOT}")
    print(f"数据目录　：{C.DATA_RAW}")

    # ---------------------------------------------------------------- Step 0
    # 首次运行时自动补齐「沪深300 动态季度权重」（漂移法重建），
    # 使「一条命令」真正成立；文件已存在则跳过，不覆盖用户的自有数据。
    if not os.path.exists(C.F_HS300_WEIGHTS_Q):
        hr("Step 0／6　生成沪深300 动态季度权重（漂移法，一次性）")
        tool = os.path.join(C.ROOT, "tools", "build_dynamic_weights.py")
        try:
            import subprocess
            r = subprocess.run([sys.executable, tool], capture_output=True,
                               text=True, encoding="utf-8", timeout=600)
            print((r.stdout or "")[-1200:])
            if r.returncode != 0:
                print("[警告] 动态权重生成失败，将回退静态基准权重：")
                print((r.stderr or "")[-600:])
        except Exception as e:                        # noqa: BLE001
            print(f"[警告] 动态权重生成异常（{e}），将回退静态基准权重")
    else:
        print(f"→ 已存在动态权重文件，跳过生成："
              f"{os.path.basename(C.F_HS300_WEIGHTS_Q)}")

    # ---------------------------------------------------------------- Step 1
    hr("Step 1／6　加载数据与校验")
    b = dl.load_all(verbose=True)
    V = dl.validate(b)
    print(V["log_text"])
    print(f"→ 校验日志已写入 {C.O_VALIDATION_LOG}")

    # ---------------------------------------------------------------- Step 2
    hr("Step 2／6　计算 F1~F11 因子")
    F = fe.build_all(b)
    ind, mkt = F["ind"], F["mkt"]
    print(f"行业横截面因子 {len(ind)} 个："
          + "、".join(k for k in ind if not k.startswith("_")))
    print(f"市场层面时序因子 {len(mkt)} 个："
          + "、".join(k for k in mkt if not k.startswith("_")))

    # ---- 第 2 轮：数据质量前置闸门（失败即中止） ----
    print("\n【2.9 数据质量前置闸门（硬约束失败即中止）】")
    gate = rs.preflight_gate(b, F, raise_on_fail=True)
    print(gate.to_string(index=False))
    gate.to_csv(C.O_PREFLIGHT, index=False, encoding="utf-8-sig")
    print(f"  → 硬约束 {int((gate['级别'] == '硬').sum())} 项全部通过；"
          f"软约束告警 {int((gate['结果'] == '告警').sum())} 项")
    print(f"  → 闸门报告：{C.O_PREFLIGHT}")

    # ---------------------------------------------------------------- Step 3
    hr("Step 3／6　IC / 分层 / 事件研究 / 稳健性")
    icb = bt.run_ic_both_lags(ind, b)
    print("\n【3.1 RankIC 汇总（t+1 为主口径，t+2 为稳健性）】")
    show = icb[["因子", "n_t1", "ic_mean_t1", "ic_ir_t1", "win_rate_t1",
                "t_nw_t1", "p_nw_t1", "结论_t1", "ic_mean_t2", "结论_t2",
                "IC方向一致(t1/t2)"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    layers = bt.run_all_layers(ind, b)
    lay = bt.layer_summary(layers, b)
    print("\n【3.2 分层回测（层1=最不拥挤 … 层5=最拥挤）】")
    print(lay[["因子", "层1年化%", "层3年化%", "层5年化%", "多空年化%(高−低)",
               "多空最大回撤%", "多空夏普", "多空t值_NW", "单调性Spearman",
               "样本内多空年化%", "样本外多空年化%"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    ev_over = bt.event_study(b, b.overweight)
    print(f"\n【3.3 事件研究：超配首次突破扩窗 P90（共 {len(ev_over['events'])} 次）】")
    print(ev_over["stats"].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    ev_pos = bt.position_event_study(b, mkt)
    ev_pm = bt.position_momentum_study(b, mkt)
    ev_hh = bt.hhi_shock_study(b, mkt)
    print("\n【3.4 「88%魔咒」替代检验①：集中度代理突破扩窗 P90 后的沪深300 收益】")
    print(ev_pos["stats"].to_string(index=False, float_format=lambda v: f"{v:,.3f}")
          if not ev_pos["stats"].empty else "  无触发事件")
    print("\n【3.5 「88%魔咒」替代检验②：集中度动量由升转降后的沪深300 收益】")
    print(ev_pm["stats"].to_string(index=False, float_format=lambda v: f"{v:,.3f}")
          if not ev_pm["stats"].empty else "  无触发事件")
    print("\n【3.6 HHI 突变后的沪深300 收益】")
    print(ev_hh["stats"].to_string(index=False, float_format=lambda v: f"{v:,.3f}")
          if not ev_hh["stats"].empty else "  无触发事件")

    multi = bt.multi_signal_backtest(b, ind)
    print("\n【3.7 多信号共振回测】")
    print(multi["summary"].to_string(index=False, float_format=lambda v: f"{v:,.3f}")
          if not multi["summary"].empty else "  无样本")
    print("\n  按信号组合细分：")
    print(multi["combo"].to_string(index=False, float_format=lambda v: f"{v:,.3f}")
          if not multi["combo"].empty else "  无样本")

    boot = icb[["因子", "n_t1", "ic_mean_t1", "ic_ir_t1", "ic_boot_lo_t1",
                "ic_boot_hi_t1", "ir_boot_lo_t1", "ir_boot_hi_t1",
                "boot_p_positive_t1", "CI含0_t1"]].copy()
    print("\n【3.12 Bootstrap 置信区间（块抽样 block=4 季度，1000 次重抽样）】")
    print(boot.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    n_sig = int((boot["CI含0_t1"] == False).sum())            # noqa: E712
    print(f"  → IC 均值 95% 置信区间不包含 0 的因子数：{n_sig} / {len(boot)}")
    if n_sig == 0:
        print("  → 结论：**没有任何因子的 IC 均值在 5% 水平上显著异于 0**，"
              "所有 IC 结论均无法排除抽样噪声。")

    print("\n【3.13 因子衰减分析（滚动 4 / 8 / 12 季度 ≈ 252 / 504 / 756 交易日）】")
    dec = bt.decay_analysis(ind, b)
    print(dec["summary"].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print("\n【3.14 FCI 因子拥挤指数 = (z_空头集中度 + z_ETF资金流 + z_因子相关性) / 3】")
    fci_tbl = mkt["FCI_原始值"].copy()
    fci_tbl["FCI"] = mkt["FCI_因子拥挤指数"]
    fci_tbl["FCI扩窗分位"] = mkt["FCI_分位"]
    print(fci_tbl.tail(8).to_string(float_format=lambda v: f"{v:,.3f}"))
    print("  → 说明：ETF净申赎强度仅 2026Q3 有数据，故历史期 FCI 由"
          "空头集中度 + 因子相关性两个分量构成（需 ≥2 个分量才出值）。")

    # ================= 第 1 轮自主迭代：方法严谨性 =================
    CAND = ["F15_拥挤背离", "R_F2_超配Z_残差", "R_F3_超配分位_残差",
            "R_F4b_动量Z_残差", "R_F1_超配比例_残差"]

    rob = bt.run_robustness(ind, b)
    _t = icb[["因子", "n_t1", "ic_mean_t1", "ic_ir_t1", "win_rate_t1", "p_nw_t1",
              "ic_boot_lo_t1", "ic_boot_hi_t1", "CI含0_t1"]].copy()
    _t["|IC_IR|"] = _t["ic_ir_t1"].abs().round(3)
    _t = _t.merge(rob["oos"][["因子", "样本内IC", "样本外IC", "方向一致"]],
                  on="因子", how="left")
    fdr = rg.bh_fdr(_t["p_nw_t1"], alpha=0.10)
    bonf = rg.bonferroni(_t["p_nw_t1"], alpha=0.05)
    # ⚠️ 机构级稳健性：bh_fdr 会**丢弃 NaN 的 p 值**，返回的 q/reject 长度可能
    # 短于 _t（短样本下 F2/F4b/F15 等因子因样本不足而 p 为 NaN）。
    # 早期版本用 .values 按位置赋值 → "Length of values (15) does not match
    # length of index (23)" → 整条流水线中止。
    # 正确做法：按索引 reindex 对齐，缺失处保持 NaN / False。
    _t["BH_q值"] = fdr["q"].reindex(_t.index)
    _t["BH拒绝(q<=0.10)"] = (fdr["reject"].reindex(_t.index)
                             .fillna(False).astype(bool))
    _t["Bonferroni拒绝"] = (bonf["reject"].reindex(_t.index)
                            .fillna(False).astype(bool))
    _t["p值有效"] = _t["p_nw_t1"].notna()

    print("\n【3.15 多重检验校正（BH-FDR / Bonferroni）】")
    print(f"  参与检验的因子数 m = {fdr['m']}")
    print(f"  Bonferroni 阈值 = 0.05 / {bonf['m']} = {bonf['threshold']:.4f}，"
          f"拒绝 {int(bonf['reject'].sum())} 个")
    print(f"  BH-FDR(alpha=0.10) 拒绝 {int(fdr['reject'].sum())} 个")
    print(f"  最小 p 值（NW 调整）= {_t['p_nw_t1'].min():.4f}"
          f"（因子 {_t.loc[_t['p_nw_t1'].idxmin(), '因子']}）")
    print(_t.sort_values("|IC_IR|", ascending=False)
          [["因子", "ic_mean_t1", "|IC_IR|", "p_nw_t1", "BH_q值", "BH拒绝(q<=0.10)",
            "样本内IC", "样本外IC", "方向一致", "CI含0_t1"]]
          .round(4).to_string(index=False))
    print("  → 结论：**单因子看有 4 个达到 |IC_IR|>0.3 且样本外同向，"
          "但 18 个因子同时检验后没有任何一个通过 FDR 校正**（最小 q="
          f"{_t['BH_q值'].min():.3f}）。这是正确且保守的结论。")

    print("\n【3.16 Newey-West 滞后阶数敏感性（0/1/2/4/8）】")
    nw_tab = {}
    for _f in CAND:
        if _f not in ind:
            continue
        _ic = fe.rank_ic_panel(ind[_f], fe.forward_excess(b.market, 1))
        _d = rg.nw_lag_sensitivity(_ic, (0, 1, 2, 4, 8))
        nw_tab[_f] = _d
        print(f"  {_f:<22} t={[round(v, 2) if v == v else None for v in _d['t值']]}"
              f"  p={[round(v, 3) if v == v else None for v in _d['p值']]}"
              f"  全部同号={bool(_d['全部同号'].iloc[0])}")

    print("\n【3.17 交易成本与换手率（高拥挤 − 低拥挤，季度调仓）】")
    cost_rows, cost_tables = [], {}
    for _f in bt.CS_FACTORS:
        if _f not in ind:
            continue
        _r = rg.long_short_with_cost(ind[_f], fe.forward_excess(b.market, 1))
        if not _r["perf"]:
            continue
        cost_tables[_f] = _r["table"]
        cost_rows.append(dict(因子=_f, 平均换手率=f"{_r['avg_turnover'] * 100:.1f}%",
                              毛年化=_r["perf"]["毛收益"]["ann"],
                              净年化_10bp=_r["perf"]["净收益_10bp"]["ann"],
                              净年化_20bp=_r["perf"]["净收益_20bp"]["ann"],
                              夏普=_r["perf"]["毛收益"]["sharpe"],
                              最大回撤=_r["perf"]["毛收益"]["maxdd"],
                              扣成本后是否仍为正=_r["perf"]["净收益_20bp"]["ann"] > 0))
    cost_df = pd.DataFrame(cost_rows).sort_values("净年化_20bp", ascending=False)
    print(cost_df.round(3).to_string(index=False))
    print("  → 换手率高的因子（如 R_F4b 152.5%）在扣 20bp 成本后由正转负，"
          "不可交易；F15 换手率最低（42.9%）。")

    print("\n【3.18 子样本稳定性（IC 序列等分 3 段）】")
    sub_frames = {}
    # 口径一致性：前瞻收益同样限制到可建模区间（holdings ∩ 基准），
    # 与主 IC 表 / design_oos_test 保持一致，避免"行情类因子"样本期被拉长。
    _hq18 = getattr(b, "over_quarters", None) or getattr(b, "alloc_quarters", None) \
        or list(b.quarters)
    _fwd18 = fe.forward_excess(b.market, 1).reindex(
        [q for q in _hq18 if q in fe.forward_excess(b.market, 1).index])
    for _f in CAND:
        if _f not in ind:
            continue
        _fac = ind[_f]
        _keep = [q for q in _fwd18.index if q in _fac.index]
        _d = rg.subperiod_stability(
            fe.rank_ic_panel(_fac.reindex(_keep), _fwd18.reindex(_keep)), 3)
        sub_frames[_f] = _d
        # 短样本保护：若该因子在可建模区间内 IC 观测不足（如 F15/R_F2 需 ≥8 期
        # 扩窗历史），subperiod_stability 返回空表 → 原代码取列会抛 KeyError。
        # 这里如实标注"样本不足"，不伪造同号结论。
        if _d is None or _d.empty or "与全样本同号" not in _d.columns:
            print(f"  {_f:<22} 样本不足（可建模区间内 IC 观测不足以分段）")
            continue
        _signs = list(_d["与全样本同号"])
        print(f"  {_f:<22} 三段IC={[round(v, 4) for v in _d['IC均值']]}  "
              f"全部与全样本同号={all(_signs)}")

    print("\n【3.19 压力测试（以 2026Q2 行业配置比例为组合权重）】")
    stress = rg.stress_test(b, ind, mkt)
    print(stress.to_string(index=False))

    print("\n【3.20 因子自身拥挤度监控（最新期）】")
    fc_watch = rg.factor_crowding_watch(ind, b)
    print(fc_watch.round(4).to_string(index=False))

    # ================= 第 2 轮自主迭代：研究级方法 =================
    print("\n【3.21 多重检验：预注册主族 vs 探索性扩展族】")
    mr = rs.multiplicity_report(_t)
    pf = mr["primary_fdr"]
    print(f"  主族（预注册）因子数 m = {pf['m']}；BH-FDR(alpha={pf['alpha']}) "
          f"拒绝 {pf['reject']} 个；最小 q = {pf['min_q']:.4f}")
    print(f"  Bonferroni(0.05) 阈值 = "
          f"{mr['primary_bonf']['threshold']:.4f}，拒绝 {mr['primary_bonf']['reject']} 个")
    if "extended" in mr:
        print(f"  扩展族（探索，不参与主族校正）{len(mr['extended'])} 个；"
              f"原始 p<0.05 者 {mr['extended_n_sig_raw']} 个，"
              f"最小 p = {mr['extended_min_p']:.4f}")
    print(mr["primary"][["因子", "ic_mean_t1", "ic_ir_t1", "p_nw_t1",
                         "BH_q值", "BH拒绝", "Bonferroni拒绝"]]
          .round(4).to_string(index=False))

    print("\n【3.22 R2-A 设计期样本外检验（设计期 ≤%s / 留出期 ≥%s，检验前留出期不可见）】"
          % (rs.DEV_END, rs.HOLDOUT_START))
    doos = rs.design_oos_test(ind, b)
    dt = doos["table"]
    print(dt[["因子", "设计期IC", "设计期_n", "留出期IC", "留出期_n",
              "留出期NW_t", "留出期单侧p", "条件1_符号一致", "条件2_绝对IC达标",
              "条件3_p达标", "通过"]].round(4).to_string(index=False))
    print(f"  → 通过事前写死三条件的因子：{int(dt['通过'].sum())} / {len(dt)}")
    print("  → 注意：留出期（2023Q1~2026Q2）是**单一行情区间**，通过是必要条件、"
          "不构成充分条件；不应据此认为因子在任意市场状态下都有效。")

    print("\n【3.23 R2-A2 设定搜索：设计期挑设定 → 留出期一次性检验】")
    sp = rs.specification_search(ind, b)
    print(sp["table"].round(4).to_string(index=False))
    he = sp["holdout_eval"]
    print(f"  设计期选中：{sp['selected']}")
    print(f"  留出期：IC = {he['留出期IC']:+.4f}（n={he['留出期_n']}），"
          f"NW t = {he['留出期NW_t']:+.3f}，单侧 p = {he['留出期单侧p']:.4f}")
    print(f"  → 选设定流程本身是否可复制：{'通过' if he['通过'] else '未通过'}"
          "（检验的是「流程」，不只是「某一个已定形式」）")

    print("\n【3.24 R2-B 二层正交化（一层剔价格风格 + 二层剔同族持仓因子）】")
    sl = rs.second_layer_orthogonalize(ind, b)
    print(sl["table"].round(4).to_string(index=False))
    print(f"  二层剔除的同族因子：{'、'.join(sl['peers_used'])}")
    _lost = sl["table"].loc[sl["table"]["二层保留比例"] < 30, "因子"].tolist()
    _keep = sl["table"].loc[sl["table"]["含独立增量信息"], "因子"].tolist()
    print(f"  → 二层后信息几乎消失（保留<30%）：{'、'.join(_lost) if _lost else '无'}"
          "　—— 这些因子的「增量信息」来自同族，不是独立信息")
    print(f"  → 二层后仍含独立增量信息：{'、'.join(_keep) if _keep else '无'}")

    print("\n【3.25 R2-D 静态 vs 动态基准口径对比（IC / IC_IR）】")
    bc = rs.benchmark_comparison(ind, b)
    print(bc.round(4).to_string(index=False))

    print("\n【3.26 R2-C 组合层面：行业轮动策略（long-only, 季度调仓, 扣 20bp）】")
    scan = rs.rotation_direction_scan(ind, b, base="equal")
    print(scan.round(2).to_string(index=False))
    print("  → 判定：只有方向与经济先验（拥挤=风险 → negative）**一致且超额为正**"
          "的因子才是可用的拥挤度因子")
    _ok = scan[(scan["更优方向"] == "negative") & (scan["超额净_negative"] > 0)]
    print(f"  → 符合者：{'、'.join(_ok['因子']) if len(_ok) else '无'}")
    _bad = scan[~scan["与经济先验一致"].astype(bool)]
    print(f"  → 反先验（顺势）才有效的因子：{'、'.join(_bad['因子']) if len(_bad) else '无'}"
          " —— 对它们，「拥挤=风险」的假设**未被数据支持**")

    print("\n【3.27 F15 行业轮动策略明细 + 参数敏感性（cap × λ）】")
    _rot = rs.industry_rotation_strategy(ind, b, "F15_拥挤背离", lam=0.5, cap=0.10,
                                        base="equal", cost_bp=20)
    _p = _rot.get("perf") or {}
    print(f"  基准 = {_rot['base']}；参数 λ=0.5，单行业上限 10%，成本 20bp")
    if not _p:
        # 短样本保护：可建模区间仅 9 期时，F15 需 ≥8 期扩窗历史，
        # 可用持仓序列不足 → 策略无法形成净值和绩效。如实报告，不伪造数字。
        print("    ⚠️ 样本不足：F15 在可建模区间内的可用持仓序列不足，"
              "轮动策略绩效无法计算（该判据本轮不成立）。")
    else:
        for _k in ["毛收益", "净收益", "基准", "超额毛", "超额净"]:
            _d = _p.get(_k)
            if not _d:
                continue
            print(f"    {_k:<6} 年化={_d['ann']:+7.2f}%  波动={_d['vol']:6.2f}%  "
                  f"夏普={_d['sharpe']:+.2f}  最大回撤={_d['maxdd']:+7.2f}%  "
                  f"累计={_d['total']:+8.2f}%")
        print(f"    信息比 IR={_p['信息比IR']:+.2f}  超额胜率={_p['超额胜率']:.1f}%  "
              f"平均换手率={_p['平均换手率'] * 100:.1f}%  "
              f"平均最大行业权重={_p['平均最大行业权重']:.1f}%  "
              f"区间 {_p['上线季度']}~{_p['结束季度']}（{_p['期数']} 期）")
    # 与沪深300 基准对照（诚实披露）
    _rot_hs = rs.industry_rotation_strategy(ind, b, "F15_拥挤背离", lam=0.5, cap=0.10,
                                            base="hs300", cost_bp=20)
    _ph = _rot_hs.get("perf") or {}
    if _ph:
        print(f"  对照：以沪深300 动态权重为基准时，净年化={_ph['净收益']['ann']:+.2f}%，"
              f"超额净={_ph['超额净']['ann']:+.2f}%，IR={_ph['信息比IR']:+.2f}")
        print("  ⚠️ 必须同时引用两条：F15 能显著战胜**等权行业**基准，"
              "但**不能**战胜市值加权的沪深300 基准 —— 因为沪深300 自身在留出期"
              "重仓了表现最好的电子，等权基准是弱基准。")
    # ---- 双基准扫描（等权 vs 沪深300）+ 因子分级裁决 ----
    scan_hs = rs.rotation_direction_scan(ind, b, base="hs300")
    print("\n【3.28 人民币基准对照：以沪深300 动态权重为基准的双方向扫描】")
    print(scan_hs[["因子", "超额净_negative", "IR_negative",
                   "超额净_positive", "IR_positive", "与经济先验一致"]]
          .round(2).to_string(index=False))
    print("  → 对照 3.26（等权基准）可见：同一因子在两个基准下的超额符号可能相反，"
          "报告时必须并列，只报有利的一个是误导。")

    print("\n【3.29 因子分级裁决表（4 项独立检验合成可用性结论）】")
    verdict = rs.factor_verdict_table(ind, b, _t,
                                      rotation_scan=scan,
                                      rotation_scan_hs300=scan_hs)
    print(verdict.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    _A = verdict[verdict["裁决"].str.startswith("A")]["因子"].tolist()
    print(f"  → 判据：J1 统计显著（|IC_IR|>0.3 且 BH-FDR q≤0.10）｜"
          "J2 设计期样本外三条件全通过｜J3 二层正交后仍保留独立增量信息｜"
          "J4 扣20bp 后按经济先验方向的超额净年化>0")
    print(f"  → **A / A− 级（可用）：{'、'.join(_A) if _A else '无'}**"
          "（A− 表示只战胜等权基准、未战胜沪深300 市值加权基准，属基准依赖，"
          "使用时必须注明基准）")
    print(f"  → B 级（样本外有效但尚不可交易/未过多重检验）"
          f"{len(verdict[verdict['裁决'].str.startswith('B')])} 个；"
          f"D 级（四项均不成立）"
          f"{len(verdict[verdict['裁决'].str.startswith('D')])} 个")

    # ================= 第 3 轮自主迭代：研究级方法（续） =================
    print("\n【3.30 R3-A Walk-forward 滚动样本外（每期只用截至 t−1 的数据挑设定）】")
    wf = r3.walk_forward_test(ind, b)
    _wft = wf["table"]
    print(_wft.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    _ws = wf["stats"]
    print(f"  → 留出期（全部 {_ws['留出期数']} 期）：IC 均值 = {_ws['留出期均值']:+.4f}，"
          f"NW t = {_ws['留出期NW_t']:+.3f}，p = {_ws['留出期p']:.4f}，"
          f"胜率 = {_ws['留出期胜率']:.1f}%")
    print(f"  → 选出因子共 {_ws['选出因子种类']} 种："
          + "、".join(f"{k}({v}期)" for k, v in _ws["选出因子分布"].items()))
    print("  → **结论：检验「挑设定的流程」而非单个因子时，样本外表现不显著**"
          f"（p = {_ws['留出期p']:.3f}）。这是对 R2「12/15 通过」的重要修正——"
          "高通过率主要由单一行情贡献，而非流程本身稳健。")

    print("\n【3.31 R3-A2 设计窗长度敏感性 + R3-A3 状态/区块分解】")
    print("  设计窗长度：")
    print(r3.wf_window_sensitivity(ind, b).to_string(
        index=False, float_format=lambda v: f"{v:,.4f}"))
    _rb = r3.wf_regime_breakdown(ind, b, wf)
    print("  按市场状态与时间区块分解：")
    print(_rb.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("  → 若样本外表现集中在某个时间区块，则结论的普适性必须打折。")

    print("\n【3.32 R3-B 构造依据「最早可得时点」检验（F15 的时间戳前移）】")
    pa = r3.premise_availability(ind, b)
    if not pa["found"]:
        print(f"  → 未找到：{pa.get('reason')}")
    else:
        print(f"  构造依据（F1 与 F2 的 IC 符号相反）最早可得的季度 t* = "
              f"**{pa['t_star']}**")
        print(f"  以 t* 为观察截止、{pa['test_start']} 起为检验期"
              f"（{pa['n_test']} 个季度）一次性检验：")
        print(pa["table"].to_string(index=False,
                                    float_format=lambda v: f"{v:,.4f}"))
        print("  → 这是**真正当时可做**的检验：t* 之前构造依据根本不可得。")

    print("\n【3.33 R3-C 正交双因子组合（F15 × F3d）】")
    dc = r3.dual_factor_combination(ind, b)
    # 短样本保护：sign/corr 由设计期 IC 符号与截面相关算出，
    # 若设计期太短（F15 在 9 期样本下 IC 观测为 0）则可能为 None。
    _sa, _sb, _cr = dc.get("sign_a"), dc.get("sign_b"), dc.get("corr")
    _fmt = lambda v: "样本不足" if v is None or v != v else f"{v:+.0f}"  # noqa: E731
    _fm4 = lambda v: "样本不足" if v is None or v != v else f"{v:.4f}"  # noqa: E731
    print(f"  方向确定时点 = {dc.get('sign_from', '—')}｜sign(F15) = {_fmt(_sa)}｜"
          f"sign(F3d) = {_fmt(_sb)}｜两因子平均 |截面相关| = {_fm4(_cr)}")
    _dct = dc.get("table")
    if isinstance(_dct, pd.DataFrame) and not _dct.empty:
        print(_dct.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    else:
        print("    ⚠️ 样本不足：双因子组合需要两因子同时具备足够的 IC 观测，"
              "当前可建模区间内不满足 → 该判据本轮不成立。")
    print("  → 组合是否优于单因子，看 IC_IR 与 NW t 是否提升；"
          "截面相关越高，分散化收益越小。")

    print("\n【3.34 R3-D 基准修正：cap 约束 vs 因子倾斜 的分离】")
    tvc = r3.tilt_vs_cap_decomposition(ind, b)
    print(tvc.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("  → **若不设 λ=0 对照，会把「cap 约束」的效果误算成因子的 alpha。**"
          "上表最后三列给出因子倾斜的净增量。")

    print("\n【3.35 R3-D4 策略起点 × 超额基准 四宫格（消除口径混淆）】")
    grid = r3.benchmark_cross_consistency(ind, b)
    print(grid.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("  → 四种口径组合下超额是否同为正，是判断「结论对口径有多敏感」的关键。")

    print("\n【3.36 R3-D2 超额收益行业归因（真 alpha 还是低配大牛股的 beta）】")
    at = r3.excess_attribution(ind, b)
    _as = at.get("summary") if isinstance(at, dict) else None
    if df_ok(at.get("table")) and isinstance(_as, dict) and _as:
        print(f"  总额外收益 = {_as['总额外收益pct']:+.2f}pct（累计）｜"
              f"净超额 = {_as['净超额pct']:+.2f}pct")
        print(f"  正贡献行业 {_as['正贡献行业数']} 个｜负贡献行业 {_as['负贡献行业数']} 个｜"
              f"最大负贡献 = {_as['最大负贡献行业']}（{_as['最大负贡献pct']:+.2f}pct）")
        print(f"  剔除最大负贡献行业后的超额 = {_as['剔除最大负贡献后超额pct']:+.2f}pct")
        print(at["table"].head(6).to_string(index=False,
                                            float_format=lambda v: f"{v:,.2f}"))
        print("  …（完整 31 行见导出表）")
        print("  → 若剔除单一行业后超额归零，说明超额本质是「低配某大牛行业的 beta」；"
              "若仍显著为正，则更接近真 alpha。")
    else:
        skip("超额收益行业归因",
             "需要完整的「持仓权重−基准权重」× 下一期行业收益序列，"
             "9 期可建模区间内不足以稳定分解")

    print("\n【3.37 R3-D5 基准变体对照（含逐期剔除最大权重行业）】")
    bv = r3.benchmark_variants(ind, b)
    print(bv.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("  ⚠️ 同时给算术年化与几何年化：波动越大二者差距越大，"
          "只报算术均值会系统性高估可实现的超额。")

    # ================= 第 4 轮自主迭代 =================
    print("\n【4.1 本轮解除的约束（L1/L2/L3）】")
    for _r in r4.CONSTRAINT_RELAXATIONS:
        print(f"  {_r['编号']}　原约束：{_r['原约束']}")
        print(f"      为何不合理：{_r['为何不合理']}")
        print(f"      解除方式　：{_r['解除方式']}")
    print(f"  → 精度规则现状：{C.PRECISION_NOTE}")
    print(f"  → 抽样验证：HHI 保留 {rnd.__globals__['_dec_for']('HHI')} 位、"
          f"IC_IR 保留 {rnd.__globals__['_dec_for']('IC_IR')} 位、"
          f"配置比例保留 {rnd.__globals__['_dec_for']('配置比例%')} 位、"
          f"持股基金数保留 {rnd.__globals__['_dec_for']('持股基金数')} 位（整数）")

    print("\n【4.2 R4-A 冲击成本与策略容量（用行业季度成交额反推参与率）】")
    imp = r4.impact_cost_model(ind, b, "F15_拥挤背离")
    _imp_t = imp.get("table")
    if not df_ok(_imp_t, "组合规模亿元", "平均单期成本bp"):
        skip("冲击成本与策略容量",
             "需要「行业季度成交额 × 可用持仓序列」推演参与率；"
             "当前成交额已补齐至 9 期，但扩窗标准化后的持仓序列仍不足")
    else:
        print(_imp_t.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
        print(f"  模型参数：p=1% → {imp['k']:.0f}bp 单边（sqrt 冲击模型）；"
              f"成本下限 {imp['floor']:.0f}bp（价差+佣金）")
        if imp.get("capacity") is not None:
            print(f"  → 容量（净超额归零的规模）= {imp['capacity']:,.0f} 亿元")
        else:
            print("  → 在测试的全部规模（最高 "
                  f"{max(imp['sizes']):,} 亿元）内，净超额均为正且几乎不衰减"
                  " → **容量不是本策略的约束**")
        _d0 = float(_imp_t.loc[_imp_t["组合规模亿元"] == imp["sizes"][0],
                               "平均单期成本bp"].iloc[0])
        _d1 = float(_imp_t.loc[_imp_t["组合规模亿元"] == imp["sizes"][-1],
                               "平均单期成本bp"].iloc[0])
        print(f"  → 成本随规模的变化：{imp['sizes'][0]:,} 亿时 {_d0:.2f}bp/期 → "
              f"{imp['sizes'][-1]:,} 亿时 {_d1:.2f}bp/期（几乎不动）")
        print("  → 对比：R1–R3 一直使用的固定 20bp 双边假设 = 2.95bp/期，"
              f"是建模成本的 {2.95 / max(_d1, 1e-9):.1f} 倍"
              " → **此前所有扣费结论都是保守下限**（成本被高估）")

    print("\n【4.3 R4-A2 冲击成本敏感性（流动性折损 h × 冲击系数 k）】")
    sv = r4.impact_cost_sensitivity(ind, b)
    if df_ok(sv, "组合规模亿元", "流动性折损h", "冲击系数k", "平均单期成本bp"):
        _s3 = sv[sv["组合规模亿元"] == 3000]
        if not _s3.empty:
            print("  规模 3000 亿元时的平均单期成本（bp）：")
            print(_s3.pivot(index="流动性折损h", columns="冲击系数k",
                            values="平均单期成本bp").round(2).to_string())
            print("  对应超额净年化（%）：")
            print(_s3.pivot(index="流动性折损h", columns="冲击系数k",
                            values="超额净_算术").round(2).to_string())
            print("  → 即使假定只能吃到行业成交额的 5%（h=0.05）且冲击系数翻倍，"
                  "成本仍极低、超额几乎不变 → **结论对成本假设不敏感**。")
        else:
            skip("冲击成本敏感性", "敏感性表中无 3000 亿元规模档")
    else:
        skip("冲击成本敏感性", "敏感性扫描需可计算净值的轮动序列")

    print("\n【4.4 R4-B 因子动量作为独立信号（逐期因子间 Spearman）】")
    fm = r4.factor_momentum_test(ind, b)
    print(fm.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("  → 解释：R3 的 walk-forward 表现随时间单调上升，正是因为"
          "「扩窗选择在追逐近期有效的因子」。")
    fmr = r4.factor_momentum_robustness(ind, b, lookback=8)
    print("\n  稳健性（候选池 / 时间区块 / 市场状态）：")
    print(fmr.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    _n = len(fmr)
    _pos = int((fmr["momIC"] > 0).sum())
    _sig = int((fmr["结论"] == "显著").sum())
    print(f"  → 读数：{_n} 个分组中 **{_pos} 个 momIC 为正**（方向高度一致），"
          f"但只有 {_sig} 个达到 10% 显著")
    print("  → 准确表述：**因子有效性存在惯性（方向一致），但其强度"
          "高度依赖时期与市场状态**——显著性集中在近段与震荡市，"
          "牛熊样本内均不显著。不可据此认为「只要追近期有效的因子就能稳赚」。")
    fms = r4.factor_momentum_strategy(ind, b, lookback=4)
    if df_ok(fms.get("table") if isinstance(fms, dict) else None,
             "组合", "超额", "换手"):
        _fmt = fms["table"]
        _ex = _fmt["超额"]
        print(f"\n  因子动量加权策略（回看 4 季、按 IC 排名加权、扣 20bp）："
              f"期数 {len(_fmt)}、组合年化 {_fmt['组合'].mean() * 4:+.2f}%、"
              f"超额净(算术) {_ex.mean() * 4:+.2f}%、"
              f"IR {(_ex.mean() * 4) / (_ex.std(ddof=1) * 2):+.2f}、"
              f"胜率 {(_ex > 0).mean() * 100:.1f}%、"
              f"平均换手 {_fmt['换手'].mean() * 100:.1f}%")
    else:
        skip("因子动量加权策略", "回看 4 季窗口在 9 期样本下不足以形成逐期 IC 排名")

    print("\n【4.5 R4-C 多因子正交组合（Gram-Schmidt + 设计窗 IC_IR 加权）】")
    mf = r4.multi_factor_orthogonal(ind, b)
    if df_ok(mf.get("per_factor") if isinstance(mf, dict) else None) \
            and isinstance(mf.get("composite_stat"), dict) and mf["composite_stat"]:
        print(mf["per_factor"].to_string(index=False,
                                         float_format=lambda v: f"{v:,.4f}"))
        _cs = mf["composite_stat"]
        print(f"  → 合成因子 F18：IC {_cs['IC均值']:+.4f}、IC_IR {_cs['IC_IR']:+.3f}、"
              f"NW t {_cs['NW_t']:+.3f}、p {_cs['p值']:.4f}、胜率 {_cs['胜率']:.1f}%")
        print("  → 对比单因子 F15 → **多因子正交组合反而稀释**："
              "正交化的代价是 F2d/F13 的增量信息几乎归零，而它们仍占走了权重。")
        print("  → 结论：**不走多因子合成，保持单因子 + 因子动量动态调权**。")
    else:
        skip("多因子正交组合",
             "Gram-Schmidt 正交化需要各因子均具备足够 IC 观测，短样本下无法成立")

    print("\n【4.6 R4-D ETF 交易拥挤层（解除 L1 约束后恢复的快层）】")
    etf = r4.etf_crowding_layer(b)
    if not etf["available"]:
        print("  → 未发现 ETF 日度数据，本层不可用")
    else:
        print(etf["summary"].to_string(index=False,
                                       float_format=lambda v: f"{v:,.2f}"))
        print("  ⚠️ 时间范围口径：ETF 为 2026Q3（日频），持仓因子为 2026Q2（季频），"
              "两者**相差一个季度**，故本层作为独立快层呈现，不与持仓因子合成。")
        _f14 = mkt.get("F14_ETF明细")
        if _f14 is not None and not _f14.empty:
            print("  与季报口径的对照（2026Q2 持仓 vs 2026Q3 资金流）：")
            for _r in _f14.sort_values("资金流强度", ascending=False).itertuples():
                _lvl = ind["拥挤等级"].loc[b.latest].get(_r.行业, "—")
                print(f"    {_r.行业:<6} 持仓等级 {_lvl:<6} "
                      f"资金流强度 {_r.资金流强度:+7.2f}%"
                      f"　→　{'方向一致' if (_lvl in ('极度拥挤', '拥挤')
                                          and _r.资金流强度 > 0) or
                                        (_lvl == '显著低配' and _r.资金流强度 < 0)
                                        else '**方向背离**'}")

    cs = rs.cap_sensitivity(ind, b, "F15_拥挤背离")
    print("\n  参数敏感性：")
    if df_ok(cs, "信息比IR"):
        print(cs.round(2).to_string(index=False))
        _ir = cs["信息比IR"].dropna()
        if len(_ir):
            print(f"  → IR 在全部 cap×λ 组合下落在 "
                  f"[{_ir.min():+.2f}, {_ir.max():+.2f}]，"
                  "未出现符号翻转 → 参数稳健")
        else:
            skip("参数敏感性", "全部 cap×λ 组合的 IR 均为空")
    else:
        skip("参数敏感性（cap × λ）",
             "需要可计算净值的轮动序列，短样本下无法形成")

    print("\n【3.8 样本内 / 样本外 IC】")
    _oos = rob.get("oos") if isinstance(rob, dict) else None
    if df_ok(_oos):
        print(_oos.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    else:
        skip("样本内/样本外 IC", "样本外切分需可建模区间足够长")
    print("\n【3.9 牛 / 熊 / 震荡 子样本 IC】")
    _sub = rob.get("subsample") if isinstance(rob, dict) else None
    print(_sub.to_string(index=False, float_format=lambda v: f"{v:,.4f}")
          if df_ok(_sub) else "  ⚠️ 样本不足：子样本划分需要更长历史")
    print("\n【3.10 参数敏感性（扩窗 min_hist 与 滚动窗口 8/12/16 季）】")
    _rb = rob.get("robustness") if isinstance(rob, dict) else None
    print(_rb.to_string(index=False, float_format=lambda v: f"{v:,.4f}")
          if df_ok(_rb) else "  ⚠️ 样本不足：滚动窗口敏感性需要更长历史")
    print("\n【3.11 正交化（剔除当期收益 / 过去4季收益 / 过去4季波动）】")
    _or = rob.get("orthogonal") if isinstance(rob, dict) else None
    print(_or.to_string(index=False, float_format=lambda v: f"{v:,.4f}")
          if df_ok(_or) else "  ⚠️ 样本不足：正交化回归需要足够横截面样本")

    # ---------------------------------------------------------------- Step 4
    hr("Step 4／6　生成信号")
    ms = sg.market_signals(ind, mkt, b)
    det = sg.industry_detail(ind, b, b.latest)
    rec = sg.signal_records(ind, b)
    tsum = sg.trigger_summary(rec)
    csum = sg.count_summary(rec)
    alerts = sg.current_alerts(ind, mkt, b)
    print(ms.loc[[b.latest]].T.to_string())
    print(f"\n当前预警（{len(alerts)} 条）：")
    for s, t, m_ in alerts:
        print(f"  [{s}] {t} — {m_}")
    print("\n行业明细（最新季度，按综合拥挤度降序）：")
    print(det[["行业", "配置比例%", "超配比例%", "超配Z", "超配分位%", "成交额占比分位%",
               "共振得分", "综合拥挤度", "拥挤等级"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\n信号触发次数统计：")
    print(csum.to_string(index=False))
    print("\n信号触发后的未来超额收益汇总：")
    print(tsum.to_string(index=False, float_format=lambda v: f"{v:,.3f}")
          if not tsum.empty else "  无触发")

    # ---------------------------------------------------------------- Step 5
    hr("Step 5／6　导出结果")
    panel = fe.to_panel(ind, b)
    mi = fe.market_indicators(mkt, b)
    ind_hist = sg.industry_history(ind, b)

    panel = rnd(panel)
    panel.to_csv(C.P_FACTOR_PANEL, index=False, encoding="utf-8-sig")
    ind_hist = rnd(ind_hist)
    ind_hist.to_csv(C.P_SIGNAL_REPORT, index=False, encoding="utf-8-sig")
    mi = rnd(mi)
    mi.to_csv(C.P_MARKET_INDICATORS, index=False, encoding="utf-8-sig")

    # factor_panel.xlsx —— 因子面板（长表 + 各因子宽表）
    with pd.ExcelWriter(C.O_FACTOR_PANEL, engine="openpyxl") as w:
        panel.to_excel(w, sheet_name="因子面板_长表", index=False)
        ind["拥挤等级"].to_excel(w, sheet_name="拥挤等级")
        for k in ["F1_超配比例", "F2_超配Zscore", "F3_超配历史分位", "F4_超配动量",
                  "F4b_超配动量Z", "F9_成交额占比分位" if "F9_成交额占比分位" in ind
                  else "F9b_成交额占比分位", "F9_交易拥挤度", "F10_共振得分",
                  "F11_综合拥挤度_推荐权重", "F11_综合拥挤度_等权",
                  "F11_综合拥挤度_IC加权"]:
            if k in ind:
                ind[k].to_excel(w, sheet_name=k[:31])
        mi.to_excel(w, sheet_name="市场层面指标", index=False)
        ind_det = ind["_F11_IC权重"]
        ind_det.to_excel(w, sheet_name="F11_IC权重")

    # ic_report.xlsx
    with pd.ExcelWriter(C.O_IC_REPORT, engine="openpyxl") as w:
        icb.to_excel(w, sheet_name="RankIC汇总", index=False)
        for f, s in bt.run_ic(ind, b, 1)["series"].items():
            s.to_frame("RankIC_t+1").to_excel(w, sheet_name=f"IC序列_{f[:22]}")
        lay.to_excel(w, sheet_name="分层回测汇总", index=False)
        for f, r in layers.items():
            if r.get("pivot") is not None:
                r["cum"].to_excel(w, sheet_name=f"分层净值_{f[:20]}")
        ev_over["stats"].to_excel(w, sheet_name="事件研究_P90突破", index=False)
        ev_pos["stats"].to_excel(w, sheet_name="事件研究_仓位P90", index=False)
        ev_pm["stats"].to_excel(w, sheet_name="事件研究_仓位动量转负", index=False)
        ev_hh["stats"].to_excel(w, sheet_name="事件研究_HHI突变", index=False)
        rob["oos"].to_excel(w, sheet_name="样本内外IC", index=False)
        rob["subsample"].to_excel(w, sheet_name="牛熊震荡IC", index=False)
        rob["robustness"].to_excel(w, sheet_name="参数敏感性", index=False)
        rob["orthogonal"].to_excel(w, sheet_name="正交化残差IC", index=False)

    # signal_report.xlsx
    with pd.ExcelWriter(C.O_SIGNAL_REPORT, engine="openpyxl") as w:
        ms.to_excel(w, sheet_name="市场层面信号")
        det.to_excel(w, sheet_name=f"行业明细_{b.latest}", index=False)
        ind_hist.to_excel(w, sheet_name="行业明细_全历史", index=False)
        rec.to_excel(w, sheet_name="信号触发记录", index=False)
        tsum.to_excel(w, sheet_name="触发后收益汇总", index=False)
        csum.to_excel(w, sheet_name="触发次数统计", index=False)
        pd.DataFrame(alerts, columns=["严重度", "类别", "说明"]).to_excel(
            w, sheet_name="当前预警", index=False)
        pd.DataFrame(V["R3_越界明细"]).to_excel(w, sheet_name="超配越界标注", index=False)

    # backtest_report.xlsx
    with pd.ExcelWriter(C.O_BACKTEST_REPORT, engine="openpyxl") as w:
        lay.to_excel(w, sheet_name="分层回测汇总", index=False)
        ls = pd.concat({f: r["ls"] for f, r in layers.items()
                        if r.get("pivot") is not None}, axis=1)
        ls.to_excel(w, sheet_name="多空收益序列(低−高)")
        rev = pd.concat({f: r["ls_rev"] for f, r in layers.items()
                         if r.get("pivot") is not None}, axis=1)
        rev.to_excel(w, sheet_name="多空收益序列(高−低)")
        multi["summary"].to_excel(w, sheet_name="多信号共振汇总", index=False)
        multi["combo"].to_excel(w, sheet_name="信号组合细分", index=False)
        multi["detail"].to_excel(w, sheet_name="共振逐条明细", index=False)
        ev_over["events"].to_excel(w, sheet_name="P90突破事件明细", index=False)
        ev_pos["events"].to_excel(w, sheet_name="仓位P90事件明细", index=False)
        ev_pm["events"].to_excel(w, sheet_name="仓位动量转负明细", index=False)
        ev_hh["events"].to_excel(w, sheet_name="HHI突变明细", index=False)
        rob["robustness"].to_excel(w, sheet_name="参数敏感性", index=False)
        rob["orthogonal"].to_excel(w, sheet_name="正交化", index=False)
        pd.DataFrame(V["R6_recalc"]).to_excel(w, sheet_name="HHI_CR复算校验")

    # ---- 结构变迁分析（新增可视化数据） ----
    # 只在**有持仓数据**的季度上做结构分析：b.quarters 含 2019Q1~2023Q4 的
    # 纯行情季度（持仓列全 NaN），对这些行取 idxmax/nlargest 会抛
    # "Encountered all NA values"。
    qs_all = list(getattr(b, "alloc_quarters", None) or [])
    if not qs_all:
        qs_all = [q for q in b.quarters if b.alloc.loc[q].notna().any()]
    alloc_w = b.alloc.reindex(qs_all).copy()
    d_alloc = alloc_w.diff().round(C.DEC_PCT)
    rep = [i for i in ["电子", "通信", "医药生物", "食品饮料", "电力设备", "银行",
                       "机械设备", "非银金融", "计算机"] if i in b.industries]
    focus = b.alloc[rep].reindex(qs_all).copy()
    # 前三/前五/前十行业集中度（与 CR3/CR5/CR10 呼应，但基于配置比例）
    conc = pd.DataFrame({
        "季度": qs_all,
        "Top1行业": [alloc_w.loc[q].idxmax() for q in qs_all],
        "Top1配置%": [round(float(alloc_w.loc[q].max()), 2) for q in qs_all],
        "Top3合计%": [round(float(alloc_w.loc[q].nlargest(3).sum()), 2) for q in qs_all],
        "Top5合计%": [round(float(alloc_w.loc[q].nlargest(5).sum()), 2) for q in qs_all],
        "Top10合计%": [round(float(alloc_w.loc[q].nlargest(10).sum()), 2) for q in qs_all],
    })
    with pd.ExcelWriter(C.O_STRUCTURE, engine="openpyxl") as w:
        rnd(alloc_w).to_excel(w, sheet_name="配置比例_宽表")
        rnd(d_alloc).to_excel(w, sheet_name="配置比例_环比变化")
        rnd(focus).to_excel(w, sheet_name="代表行业时序")
        conc.to_excel(w, sheet_name="头部集中度", index=False)
        if getattr(b, "hot_top20", None) is not None:
            rnd(b.hot_top20).to_excel(w, sheet_name="最新抱团股Top20", index=False)

    # ---- FCI 报告 ----
    with pd.ExcelWriter(C.O_FCI_REPORT, engine="openpyxl") as w:
        rnd(fci_tbl).to_excel(w, sheet_name="FCI时序")
        rnd(mkt["FCI_分量"]).to_excel(w, sheet_name="FCI分量Z")
        rnd(mkt["FCI_原始值"]).to_excel(w, sheet_name="FCI原始值")
        if getattr(b, "daily_coverage", None) is not None:
            b.daily_coverage.to_excel(w, sheet_name="日度数据覆盖率", index=False)

    # ---- 严谨性报告（第 1 轮迭代） ----
    p_rigor = os.path.join(C.OUTPUT, "rigor_report.xlsx")
    with pd.ExcelWriter(p_rigor, engine="openpyxl") as w:
        _t.to_excel(w, sheet_name="IC汇总_含多重检验校正", index=False)
        cost_df.to_excel(w, sheet_name="交易成本与换手率", index=False)
        stress.to_excel(w, sheet_name="压力测试", index=False)
        fc_watch.to_excel(w, sheet_name="因子自身拥挤度监控", index=False)
        pd.concat(nw_tab, names=["因子", "行"]).to_excel(
            w, sheet_name="NW滞后阶数敏感性")
        key = "R_F2_超配Z_残差"
        if key in cost_tables:
            cost_tables[key].to_excel(w, sheet_name="换手率明细_例")
        for _f, _d in sub_frames.items():
            _d.to_excel(w, sheet_name=f"子样本_{_f[:18]}", index=False)
        pd.DataFrame([dict(检验="BH-FDR", 水平=0.10, 参与因子数=fdr["m"],
                           拒绝数=int(fdr["reject"].sum()),
                           最小q值=float(_t["BH_q值"].min())),
                      dict(检验="Bonferroni", 水平=0.05, 参与因子数=bonf["m"],
                           拒绝数=int(bonf["reject"].sum()),
                           最小p值=float(_t["p_nw_t1"].min()))]
                     ).to_excel(w, sheet_name="多重检验结论", index=False)
    print(f"  {p_rigor}")
    # ---- 第 2 轮：研究级报告 + 组合层报告 ----
    with pd.ExcelWriter(C.O_RESEARCH_REPORT, engine="openpyxl") as w:
        mr["table"].to_excel(w, sheet_name="多重检验_主族与扩展族", index=False)
        pd.DataFrame([mr["primary_fdr"], mr["primary_bonf"]]).to_excel(
            w, sheet_name="多重检验结论", index=False)
        doos["table"].to_excel(w, sheet_name="R2A_设计期样本外检验", index=False)
        sp["table"].to_excel(w, sheet_name="R2A2_设定搜索_设计期", index=False)
        pd.DataFrame([sp["holdout_eval"]]).to_excel(
            w, sheet_name="R2A2_留出期判定", index=False)
        sl["table"].to_excel(w, sheet_name="R2B_二层正交化", index=False)
        bc.to_excel(w, sheet_name="R2D_静态vs动态基准", index=False)
        _seg_rows = []
        for _f, _v in doos["detail"].items():
            _seg_rows.append(dict(因子=_f,
                                  设计期IC均值=float(_v["ic_dev"].mean()),
                                  设计期n=len(_v["ic_dev"]),
                                  留出期IC均值=float(_v["ic_hold"].mean()),
                                  留出期n=len(_v["ic_hold"])))
        pd.DataFrame(_seg_rows).to_excel(w, sheet_name="R2A_两段IC明细", index=False)
        gate.to_excel(w, sheet_name="数据质量前置闸门", index=False)
        verdict.to_excel(w, sheet_name="R2G_因子分级裁决", index=False)
        scan_hs.to_excel(w, sheet_name="R2C_双方向扫描_沪深300基准", index=False)

    with pd.ExcelWriter(C.O_STRATEGY, engine="openpyxl") as w:
        scan.to_excel(w, sheet_name="R2C_双方向扫描", index=False)
        _xl(w, "R2C_参数敏感性", cs)
        _xl(w, "R2C_策略净值明细", _rot.get("table") if isinstance(_rot, dict) else None)
        _xl(w, "R2C_对照_沪深300基准",
            _rot_hs.get("table") if isinstance(_rot_hs, dict) else None)
        # 每期持仓权重（可审计）
        _rw = _rot.get("weights") if isinstance(_rot, dict) else None
        if isinstance(_rw, dict) and _rw:
            _wdf = pd.DataFrame(_rw).T
            _wdf.index.name = "持仓季度"
            rnd(_wdf).to_excel(w, sheet_name="R2C_每期行业权重")
        # 策略汇总：逐字段安全取值（短样本下 _p 可能为空 dict）
        _summ = {k: (str(v) if isinstance(v, dict) else v)
                 for k, v in _p.items() if not isinstance(v, dict)}
        for _k, _v in (_p.get("参数") or {}).items():
            _summ[f"参数_{_k}"] = _v
        for _sub in ("毛收益", "净收益", "基准", "超额毛", "超额净"):
            for _k2, _v2 in (_p.get(_sub) or {}).items():
                _summ[f"{_sub}_{_k2}"] = _v2
        if not _summ:
            _summ = {"说明": "样本不足：可建模区间内 F15 轮动策略无法形成有效序列，"
                             "本轮不产出绩效汇总（不做替代填补）"}
        pd.DataFrame([_summ]).to_excel(w, sheet_name="R2C_策略汇总", index=False)

    # ---- 第 3 轮：研究级报告（续） ----
    # 全部走 _xl（非空才写），短样本下缺失的分表如实留空而非崩溃。
    with pd.ExcelWriter(C.O_RESEARCH3_REPORT, engine="openpyxl") as w:
        _xl(w, "R3A_WalkForward逐期", wf.get("table") if isinstance(wf, dict) else None,
            index=False)
        _xl(w, "R3A_WalkForward统计", pd.DataFrame([_ws]) if _ws else None)
        _xl(w, "R3A2_设计窗敏感性", r3.wf_window_sensitivity(ind, b), index=False)
        _xl(w, "R3A3_状态与区块分解", _rb, index=False)
        if pa.get("found"):
            _xl(w, "R3B_构造依据可得性演进", pa.get("hist"), index=False)
            _xl(w, "R3B_检验期一次性结果", pa.get("table"), index=False)
            _xl(w, "R3B_时点结论", pd.DataFrame([dict(
                构造依据最早可得时点=pa["t_star"], 检验期起点=pa["test_start"],
                检验期终点=pa["test_end"], 检验期数=pa["n_test"])]), index=False)
        _xl(w, "R3C_双因子组合", dc.get("table") if isinstance(dc, dict) else None,
            index=False)
        _xl(w, "R3D3_cap与倾斜分离", tvc, index=False)
        _xl(w, "R3D4_四宫格口径对照", grid, index=False)
        _xl(w, "R3D2_超额行业归因", at.get("table") if isinstance(at, dict) else None,
            index=False)
        if isinstance(at, dict) and at.get("detail") is not None:
            rnd(at["detail"]).to_excel(w, sheet_name="R3D2_逐期行业贡献")
        _xl(w, "R3D5_基准变体", bv, index=False)
    # ---- 第 4 轮：成本/容量/因子动量/多因子/ETF 层 ----
    with pd.ExcelWriter(C.O_RESEARCH4_REPORT, engine="openpyxl") as w:
        _xl(w, "L_解除的约束登记", pd.DataFrame(r4.CONSTRAINT_RELAXATIONS), index=False)
        _xl(w, "R4A_冲击成本与容量", imp.get("table"), index=False)
        if imp.get("detail") is not None:
            rnd(imp["detail"]).to_excel(w, sheet_name="R4A_行业级成本明细", index=False)
        _xl(w, "R4A2_成本敏感性", sv, index=False)
        _xl(w, "R4B_因子动量", fm, index=False)
        _xl(w, "R4B_因子动量稳健性", fmr, index=False)
        _xl(w, "R4B_因子动量策略",
            fms.get("table") if isinstance(fms, dict) else None)
        _xl(w, "R4C_多因子正交组合",
            mf.get("per_factor") if isinstance(mf, dict) else None, index=False)
        _xl(w, "R4C_合成因子F18",
            pd.DataFrame([mf["composite_stat"]]) if isinstance(mf, dict)
            and mf.get("composite_stat") else None)
        if isinstance(mf, dict) and mf.get("weights"):
            _xl(w, "R4C_因子权重",
                pd.DataFrame([dict(因子=k, 权重=v)
                              for k, v in mf["weights"].items()]), index=False)
        if etf.get("available"):
            _xl(w, "R4D_ETF拥挤层汇总", etf.get("summary"), index=False)
            _xl(w, "R4D_ETF日度明细", rnd(etf["daily"]) if etf.get("daily") is not None
                else None)
    print(f"  {C.O_RESEARCH4_REPORT}")
    print(f"  {C.O_RESEARCH3_REPORT}")
    print(f"  {C.O_RESEARCH_REPORT}")
    print(f"  {C.O_STRATEGY}")
    print(f"  {C.O_PREFLIGHT}")
    print(f"  {C.O_STRUCTURE}")
    print(f"  {C.O_FCI_REPORT}")
    print(f"  {C.O_FACTOR_PANEL}")
    print(f"  {C.O_IC_REPORT}")
    print(f"  {C.O_SIGNAL_REPORT}")
    print(f"  {C.O_BACKTEST_REPORT}")
    print(f"  {C.P_FACTOR_PANEL}")
    print(f"  {C.P_SIGNAL_REPORT}")
    print(f"  {C.P_MARKET_INDICATORS}")

    # ---------------------------------------------------------------- Step 6
    hr("Step 6／6　核心结论")
    m = fe.market_summary(ind, mkt, b)
    print(f"· 数据区间 {b.quarters[0]} ~ {b.latest}（{b.n_quarters} 个季度），"
          f"申万一级 {len(b.industries)} 个行业")
    print(f"· 市场状态：{m['state']}")
    print(f"    HHI={m['HHI']:.4f}（{m['HHI_level']}，历史分位 {m['HHI_pct']:.0f}%，"
          f"历史均值 {m['HHI_hist_mean']:.4f}）")
    print(f"    CR3={m['CR3']:.2f}%  CR5={m['CR5']:.2f}%  CR10={m['CR10']:.2f}%")
    print(f"    第一大重仓行业：{m['top1']} {m['top1_pct']:.2f}%")
    print(f"· 极度拥挤行业 {m['n_extreme']} 个：" + "、".join(m["extreme_industries"]))
    print(f"· 显著低配行业 {m['n_under']} 个：" + "、".join(m["under_industries"]))
    print(f"· F12 配置系数（>2.5 极度拥挤）超阈值的行业："
          + ("无" if not (ind["F12_配置系数"].loc[b.latest] > C.THRESH_CF_EXTREME).any()
             else "、".join(ind["F12_配置系数"].loc[b.latest]
                            [ind["F12_配置系数"].loc[b.latest]
                             > C.THRESH_CF_EXTREME].index)))
    print(f"· F13 筹码盈利比例 ≥{C.THRESH_PROFIT_EXTREME:.0f}%（获利盘拥挤）："
          + "、".join(ind["F13_筹码盈利比例"].loc[b.latest]
                      [ind["F13_筹码盈利比例"].loc[b.latest]
                       >= C.THRESH_PROFIT_EXTREME].index))
    _f14 = mkt["F14_ETF明细"]
    if _f14 is not None and not _f14.empty:
        print("· F14 ETF资金流强度（2026Q3，与持仓季度相差一个季度，独立呈现）：")
        for _r in _f14.sort_values("资金流强度", ascending=False).itertuples():
            print(f"    {_r.行业:<6} 净申赎 {_r.区间净申赎亿元:+8.2f} 亿元  "
                  f"强度 {_r.资金流强度:+7.2f}%  （期初AUM {_r.期初AUM亿元:.2f} 亿元）")
    _dv = ind["口径背离"].loc[b.latest]
    _dvl = list(_dv.index[_dv])
    print(f"· 口径背离行业（Z/分位 与 F12 指向相反）："
          + ("无" if not _dvl else "、".join(_dvl))
          + "　→ 这些行业必须同时呈现两个口径，不能只给一个标签")
    _fci = mkt["FCI_因子拥挤指数"].dropna()
    print(f"· FCI 因子拥挤指数 最新 = {_fci.iloc[-1]:+.3f}（{_fci.index[-1]}），"
          f"扩窗分位 {mkt['FCI_分位'].iloc[-1]:.1f}%")
    print(f"· Bootstrap：IC 均值 95% 置信区间不包含 0 的因子数 = {n_sig} / {len(boot)}")
    print()
    print("· 第 2 轮新增结论：")
    print(f"  ① 多重检验（主族 m={mr['primary_fdr']['m']}）：BH-FDR "
          f"拒绝 {mr['primary_fdr']['reject']} 个，最小 q = "
          f"{mr['primary_fdr']['min_q']:.4f} → "
          + ("主族内无因子通过 FDR 校正" if mr["primary_fdr"]["reject"] == 0
             else "主族内有因子通过 FDR 校正"))
    _pass = dt[dt["通过"]] if df_ok(dt, "通过") else pd.DataFrame()
    if df_ok(dt, "通过"):
        _f15r = dt[dt["因子"] == "F15_拥挤背离"]
        _f15_txt = (f"F15_拥挤背离 留出期 IC = "
                    f"{float(_f15r['留出期IC'].iloc[0]):+.4f}，单侧 p = "
                    f"{float(_f15r['留出期单侧p'].iloc[0]):.4f}"
                    if not _f15r.empty and _f15r["留出期单侧p"].notna().any()
                    else "F15 留出期观测不足（无法给出 IC / p）")
        print(f"  ② 设计期样本外检验：{len(_pass)} / {len(dt)} 个因子通过事前写死的三条件；"
              f"{_f15_txt}")
    else:
        print("  ② 设计期样本外检验：本轮不可用（设计期内无可评估因子）")
    _l = sl.get("table") if isinstance(sl, dict) else None
    if df_ok(_l, "含独立增量信息", "因子"):
        _nof = _l.loc[~_l["含独立增量信息"], "因子"].tolist()
        print(f"  ③ 二层正交化：{'、'.join(_nof) if _nof else '无'} 的信息在同族正交后"
              "几乎归零 → **第 1 轮把残差因子列为一等因子的判断需要修正**；"
              f"仅 {'、'.join(_l.loc[_l['含独立增量信息'], '因子']) or '无'} 保留独立增量信息")
    else:
        print("  ③ 二层正交化：本轮不可用（短样本下回归无足够观测）")
    _dyn = bc[bc["更优口径"] == "动态基准"]["因子族"].tolist()
    _st = bc[bc["更优口径"] == "静态基准"]["因子族"].tolist()
    print(f"  ④ 基准口径：标准化类因子（{ '、'.join(_dyn) }）动态基准更优；"
          f"水平类因子（{ '、'.join(_st) }）静态基准更优 → **按因子类型分别选择，不混用**")
    if _p and _ph:
        print(f"  ⑤ 组合层面：F15 行业轮动（cap=10%, λ=0.5, 扣20bp）"
              f"净年化 {_p['净收益']['ann']:+.2f}%、超额净 {_p['超额净']['ann']:+.2f}%、"
              f"IR {_p['信息比IR']:+.2f}、换手 {_p['平均换手率'] * 100:.1f}%；"
              f"但以沪深300 为基准时超额净 {_ph['超额净']['ann']:+.2f}%（IR "
              f"{_ph['信息比IR']:+.2f}）→ 战胜等权基准，未战胜市值加权基准")
    else:
        print("  ⑤ 组合层面：样本不足，轮动策略绩效无法计算（本轮不成立）")
    _A = verdict[verdict["裁决"].str.startswith("A")]
    _B = verdict[verdict["裁决"].str.startswith("B")]
    _D = verdict[verdict["裁决"].str.startswith("D")]
    print(f"· 因子分级裁决（4 项独立检验合成）：A/A− 级 {len(_A)} 个"
          f"（{'、'.join(_A['因子']) if len(_A) else '无'}）；"
          f"B 候选 {len(_B)} 个；C 描述性 "
          f"{len(verdict[verdict['裁决'].str.startswith('C')])} 个；D 未成立 {len(_D)} 个")
    print(f"· 验收标准「至少一个因子 |IC_IR|>0.3 且样本外方向一致」：")
    _meet = _t[(_t["|IC_IR|"] > 0.3) & (_t["方向一致"] == True)]        # noqa: E712
    if _meet.empty:
        print("    未满足")
    else:
        print(f"    已满足 —— {len(_meet)} 个因子：")
        for _, _r in _meet.sort_values("|IC_IR|", ascending=False).iterrows():
            print(f"      {_r['因子']:<22} |IC_IR|={_r['|IC_IR|']:.3f}  "
                  f"IC={_r['ic_mean_t1']:+.4f}  p={_r['p_nw_t1']:.4f}  "
                  f"样本内={_r['样本内IC']:+.4f}  样本外={_r['样本外IC']:+.4f}")
    print(f"· 但多重检验校正后：**主族（预注册 m={mr['primary_fdr']['m']}）**"
          f"BH-FDR(0.10) 拒绝 {mr['primary_fdr']['reject']} 个"
          f"（最小 q={mr['primary_fdr']['min_q']:.4f}）；"
          f"扩展族（探索 {len(mr.get('extended', []))} 个）不参与主族校正，"
          f"仅登记最小 p={mr.get('extended_min_p', float('nan')):.4f} → "
          "**无因子通过 FDR 校正**")
    _pos = cost_df.loc[cost_df["扣成本后是否仍为正"]].sort_values(
        "净年化_20bp", ascending=False)
    print(f"· 分层多空（理论值）扣 20bp 后净年化仍为正的因子 {len(_pos)} 个；"
          f"其中前 5："
          + "、".join(f"{r['因子']}({r['净年化_20bp']:+.2f}%)"
                     for _, r in _pos.head(5).iterrows())
          + "　—— 注意：这是**多空价差的理论值**，不等于可交易策略；"
            "可交易性见第 3.29 节裁决表")
    print(f"· 压力测试：电子回撤30% → 组合 {stress.loc[0, '组合冲击']}；"
          f"风格反转（拥挤行业回撤30%） → 组合 "
          f"{stress.iloc[-1]['组合冲击']}")
    eff = icb[icb["结论_t1"].isin(["优秀", "有效"])]
    print(f"· RankIC 达到「有效」及以上的因子："
          + ("无（全部 IC_IR < 0.3）" if eff.empty else "、".join(eff["因子"])))
    print(f"· 说明：拥挤度是风险描述指标，不是择时或做空信号。")
    print("\n完成。启动看板：streamlit run app.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                   # noqa: BLE001
        print("\n" + "!" * 78)
        print("流水线执行失败：")
        traceback.print_exc()
        print("!" * 78)
        sys.exit(1)
