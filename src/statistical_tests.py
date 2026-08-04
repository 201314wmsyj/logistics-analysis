__test__ = False  # Prevent pytest from collecting test_ functions directly

"""
统计检验模块
============
功能：对供应链数据进行严格的统计假设检验，量化结论的可信度。

检验列表：
  1. Welch's t 检验 -- 中断 vs 正常货运的时效差异（含 Mann-Whitney U 非参数补充）
  2. 单因素 ANOVA + Tukey HSD 事后检验 -- 四种运输模式的时效差异
  3. 卡方检验 -- 中断率 x 运输模式 / 产品类别的独立性
  4. 卡方检验 -- 天气条件 x 中断率
  5. 卡方检验 -- 地缘政治风险等级 x 中断率

方法论保障：
  - 所有检验均报告效应量（Cohen's d / eta^2 / Cramer's V），而非仅凭 p 值
  - 汇总阶段使用 Bonferroni 校正控制族系误差率（FWER）
  - ANOVA 显著时自动运行 Tukey HSD 事后检验，识别具体哪些组间存在差异
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def _significance_label(p_value: float, corrected: bool = False) -> str:
    """将 p 值转为可读的显著性标签"""
    threshold = 0.05
    if p_value < 0.001:
        return f"*** (p < 0.001{' corr.' if corrected else ''})"
    elif p_value < 0.01:
        return f"** (p < 0.01{' corr.' if corrected else ''})"
    elif p_value < threshold:
        return f"* (p < 0.05{' corr.' if corrected else ''})"
    else:
        return f"n.s. (p = {p_value:.3f})"


def _bonferroni_corrected_significance(p_value: float, n_tests: int) -> tuple:
    """Bonferroni 校正：p_adj = min(p * n_tests, 1.0)"""
    p_adj = min(p_value * n_tests, 1.0)
    return p_adj, p_adj < 0.05


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """计算 Cohen's d 效应量"""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def _cramers_v(contingency_table: pd.DataFrame) -> float:
    """计算 Cramer's V（类别变量关联强度）"""
    chi2 = stats.chi2_contingency(contingency_table.values)[0]
    n = contingency_table.values.sum()
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


# ============================================================
# 1. Welch's t 检验 -- 中断 vs 正常的时效差异
# ============================================================

def test_lead_time_by_disruption(df: pd.DataFrame) -> dict:
    """
    H0: 中断和正常货运的时效均值无差异
    H1: 中断和正常货运的时效均值有差异

    使用 Welch's t-test（不假设方差齐性），辅以 Mann-Whitney U。
    """
    print("\n" + "-" * 45)
    print("[检验 1] 中断 vs 正常 的时效差异 (Welch's t-test)")
    print("-" * 45)

    disrupted = df.loc[df["Disruption_Occurred"] == 1, "Lead_Time_Days"]
    normal = df.loc[df["Disruption_Occurred"] == 0, "Lead_Time_Days"]

    t_stat, p_value = stats.ttest_ind(disrupted, normal, equal_var=False)
    d = _cohens_d(disrupted.values, normal.values)
    label = _significance_label(p_value)

    print(f"  中断组均值: {disrupted.mean():.2f} +/- {disrupted.std():.2f} 天 (n={len(disrupted):,})")
    print(f"  正常组均值: {normal.mean():.2f} +/- {normal.std():.2f} 天 (n={len(normal):,})")
    print(f"  t = {t_stat:.4f}, Cohen's d = {d:.4f}, {label}")

    # 非参数补充（Welch 已经对非正态稳健，Mann-Whitney 作为敏感性分析）
    u_stat, u_p = stats.mannwhitneyu(disrupted, normal, alternative="two-sided")
    u_label = _significance_label(u_p)
    print(f"  Mann-Whitney U: U = {u_stat:,.0f}, {u_label} (non-parametric sensitivity)")

    return {
        "test": "Welch's t-test: Lead Time by Disruption",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "effect_size_cohens_d": round(d, 4),
        "effect_magnitude": (
            "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
        ),
        "mann_whitney_p": float(u_p),
    }


# ============================================================
# 2. 单因素 ANOVA + Tukey HSD 事后检验
# ============================================================

def test_lead_time_by_mode(df: pd.DataFrame) -> dict:
    """
    H0: 四种运输模式的时效均值无差异
    H1: 至少一种运输模式的时效均值与其他不同

    若 ANOVA 显著，自动运行 Tukey HSD 事后检验以识别具体差异组。
    """
    print("\n" + "-" * 45)
    print("[检验 2] 运输模式间时效差异 (One-Way ANOVA)")
    print("-" * 45)

    groups = [
        df.loc[df["Transport_Mode"] == mode, "Lead_Time_Days"].values
        for mode in df["Transport_Mode"].unique()
    ]
    mode_names = list(df["Transport_Mode"].unique())

    f_stat, p_value = stats.f_oneway(*groups)
    label = _significance_label(p_value)

    for name, group in zip(mode_names, groups):
        print(f"  {name:<8s}: mu = {np.mean(group):.2f}, sigma = {np.std(group):.2f}, n = {len(group):,}")

    # Eta-squared (效应量)
    grand_mean = df["Lead_Time_Days"].mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = ((df["Lead_Time_Days"] - grand_mean) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    print(f"  F = {f_stat:.4f}, eta^2 = {eta_sq:.4f}, {label}")

    # ---- Tukey HSD 事后检验 ----
    tukey_results = []
    if p_value < 0.05:
        print("\n  [Tukey HSD 事后检验] 显著组对:")
        n_total = sum(len(g) for g in groups)
        k = len(groups)

        # 手工实现 Tukey HSD（避免依赖 statsmodels）
        mse = sum(
            np.sum((g - np.mean(g)) ** 2) for g in groups
        ) / (n_total - k)

        all_pairs = list(combinations(range(k), 2))
        for i, j in all_pairs:
            diff = np.mean(groups[i]) - np.mean(groups[j])
            se = np.sqrt(mse * (1 / len(groups[i]) + 1 / len(groups[j])))

            # Tukey's q statistic
            q_stat = abs(diff) / se if se > 0 else 0

            # Studentized range critical value for alpha=0.05
            # 使用保守近似 q_crit = 3.63 for k=4, df large
            q_crit = 3.63  # q(0.05, 4, inf)
            is_sig = q_stat > q_crit

            # Cohen's d for this pair
            d_pair = _cohens_d(groups[i], groups[j])

            if is_sig:
                print(f"    {mode_names[i]:<8s} vs {mode_names[j]:<8s}: "
                      f"delta = {diff:+.1f}d, d = {d_pair:+.3f}, q = {q_stat:.2f} [SIG]")

            tukey_results.append({
                "group1": mode_names[i],
                "group2": mode_names[j],
                "mean_diff": round(float(diff), 2),
                "cohens_d": round(d_pair, 4),
                "q_statistic": round(float(q_stat), 2),
                "significant": is_sig,
            })

    return {
        "test": "One-Way ANOVA: Lead Time by Transport Mode",
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "effect_size_eta_squared": round(eta_sq, 4),
        "effect_magnitude": (
            "large" if eta_sq > 0.14 else "medium" if eta_sq > 0.06 else "small"
        ),
        "tukey_posthoc": tukey_results,
    }


# ============================================================
# 3. 卡方检验 -- 中断 x 运输模式 / 产品类别
# ============================================================

def test_disruption_independence(df: pd.DataFrame) -> dict:
    """
    H0: 中断率与运输模式/产品类别独立
    H1: 中断率与运输模式/产品类别有关联
    """
    print("\n" + "-" * 45)
    print("[检验 3] 中断率与类别变量的独立性 (Chi-Square)")
    print("-" * 45)

    results = {}

    for var in ["Transport_Mode", "Product_Category"]:
        contingency = pd.crosstab(df[var], df["Disruption_Occurred"])
        chi2, p_value, dof, _expected = stats.chi2_contingency(contingency.values)
        v = _cramers_v(contingency)
        label = _significance_label(p_value)

        print(f"\n  [{var}]")
        print(f"    chi^2 = {chi2:.4f}, df = {dof}, Cramer's V = {v:.4f}, {label}")

        results[var] = {
            "chi2": float(chi2),
            "p_value": float(p_value),
            "dof": dof,
            "significant": p_value < 0.05,
            "cramers_v": round(v, 4),
            "effect_magnitude": (
                "large" if v > 0.3 else "medium" if v > 0.1 else "small"
            ),
        }

    return results


# ============================================================
# 4. 天气条件 x 中断率的卡方检验
# ============================================================

def test_weather_disruption(df: pd.DataFrame) -> dict:
    """
    H0: 中断率与天气条件独立
    H1: 中断率与天气条件有关联
    """
    print("\n" + "-" * 45)
    print("[检验 4] 天气条件与中断率的独立性 (Chi-Square)")
    print("-" * 45)

    contingency = pd.crosstab(df["Weather_Condition"], df["Disruption_Occurred"])
    chi2, p_value, dof, _expected = stats.chi2_contingency(contingency.values)
    v = _cramers_v(contingency)
    label = _significance_label(p_value)

    for cond in contingency.index:
        total = contingency.loc[cond].sum()
        disrupted = contingency.loc[cond, 1] if 1 in contingency.columns else 0
        print(f"  {cond:<15s}: {disrupted}/{total} = {disrupted/total*100:.1f}%")

    print(f"\n  chi^2 = {chi2:.4f}, df = {dof}, Cramer's V = {v:.4f}, {label}")

    return {
        "test": "Chi-Square: Disruption by Weather",
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": dof,
        "significant": p_value < 0.05,
        "cramers_v": round(v, 4),
        "effect_magnitude": (
            "large" if v > 0.3 else "medium" if v > 0.1 else "small"
        ),
    }


# ============================================================
# 5. 地缘政治风险等级 x 中断率的卡方检验
# ============================================================

def test_risk_level_disruption(df: pd.DataFrame) -> dict:
    """
    H0: 中断率与风险等级独立
    H1: 中断率与风险等级有关联
    """
    print("\n" + "-" * 45)
    print("[检验 5] 地缘政治风险等级与中断率 (Chi-Square)")
    print("-" * 45)

    contingency = pd.crosstab(df["Risk_Level"], df["Disruption_Occurred"])
    chi2, p_value, dof, _expected = stats.chi2_contingency(contingency.values)
    v = _cramers_v(contingency)
    label = _significance_label(p_value)

    for lvl in ["Low", "Medium", "High"]:
        if lvl in contingency.index:
            total = contingency.loc[lvl].sum()
            disrupted = contingency.loc[lvl, 1] if 1 in contingency.columns else 0
            print(f"  {lvl:<8s}: {disrupted}/{total} = {disrupted/total*100:.1f}%")

    print(f"\n  chi^2 = {chi2:.4f}, df = {dof}, Cramer's V = {v:.4f}, {label}")

    return {
        "test": "Chi-Square: Disruption by Risk Level",
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": dof,
        "significant": p_value < 0.05,
        "cramers_v": round(v, 4),
        "effect_magnitude": (
            "large" if v > 0.3 else "medium" if v > 0.1 else "small"
        ),
    }


# ============================================================
# 汇总运行（含 Bonferroni 校正）
# ============================================================

def run_statistical_tests(df: pd.DataFrame) -> dict:
    """
    运行所有统计检验，并在汇总时应用 Bonferroni 校正。

    Returns
    -------
    dict
        所有检验结果，包含原始 p 值和 Bonferroni 校正后的显著性判断
    """
    print("\n" + "=" * 55)
    print("[统计假设检验]")
    print("=" * 55)

    results = {}
    results["ttest_lt_disruption"] = test_lead_time_by_disruption(df)
    results["anova_lt_mode"] = test_lead_time_by_mode(df)
    results["chi2_disruption_independence"] = test_disruption_independence(df)
    results["chi2_weather"] = test_weather_disruption(df)
    results["chi2_risk"] = test_risk_level_disruption(df)

    # ---- 汇总（含 Bonferroni 校正） ----
    print("\n" + "-" * 45)
    print("[检验结果汇总] 含 Bonferroni 校正")
    print("-" * 45)

    summaries = [
        ("中断 vs 正常时效 (Welch t)", results["ttest_lt_disruption"]),
        ("运输模式时效差异 (ANOVA)", results["anova_lt_mode"]),
        ("中断 x 运输模式 (chi^2)", results["chi2_disruption_independence"]["Transport_Mode"]),
        ("中断 x 产品类别 (chi^2)", results["chi2_disruption_independence"]["Product_Category"]),
        ("中断 x 天气条件 (chi^2)", results["chi2_weather"]),
        ("中断 x 风险等级 (chi^2)", results["chi2_risk"]),
    ]

    n_tests = len(summaries)

    for name, r in summaries:
        if "p_value" in r:
            raw_p = r["p_value"]
            p_adj, is_sig_adj = _bonferroni_corrected_significance(raw_p, n_tests)
            status = "[SIG]" if is_sig_adj else "[NS]"
            es = r.get("effect_magnitude", "")
            print(f"  {name:<35s} p_raw={raw_p:.4f}  p_adj={p_adj:.4f}  {status:<8s} effect: {es}")

    # ---- 关键发现摘要 ----
    print("\n" + "-" * 45)
    print("[关键统计发现]")
    print("-" * 45)

    # ANOVA Tukey 结果
    tukey = results["anova_lt_mode"].get("tukey_posthoc", [])
    sig_pairs = [t for t in tukey if t.get("significant")]
    if sig_pairs:
        print(f"  运输模式事后检验: {len(sig_pairs)}/{len(tukey)} 组对显著差异")
        for t in sig_pairs:
            print(f"      {t['group1']} vs {t['group2']}: delta = {t['mean_diff']:+.1f}d, d = {t['cohens_d']:+.3f}")

    # 最大效应量检验
    max_es = max(
        summaries, key=lambda x: abs(x[1].get("effect_size_cohens_d", x[1].get("effect_size_eta_squared", x[1].get("cramers_v", 0))))
    )
    print(f"  最大效应量: {max_es[0]} (effect: {max_es[1].get('effect_magnitude','')})")

    return results


if __name__ == "__main__":
    from data_loader import prepare_data

    df = prepare_data()
    results = run_statistical_tests(df)
