"""
蒙特卡洛模拟模块
================
功能：基于历史数据分布，通过随机抽样模拟供应链中断场景，量化风险敞口。

模拟情景：
  1. 中断率分布模拟 -- Bootstrap 重抽样，估计中断率置信区间
  2. 场景压力测试 -- 地缘政治风险升高 / 恶劣天气概率增加后的中断率变化
  3. 路线级 VaR -- 各路线时效的 95% / 99% 风险价值 (Value-at-Risk)
  4. 总成本模拟 -- 基于中断损失假设，模拟整体供应链风险成本
  5. ROI 模拟 -- 模拟"提升承运商可靠性"的关联性影响估计

方法论说明：
  - ROI 模拟使用 train/test 分离，并在保留集上诚实地评估模型性能
  - 所有结论表述为关联性估计（非因果推断），README 中列出了未观测混杂因素的局限性
"""

import numpy as np
import pandas as pd


# ================================================================
# 共享工具函数
# ================================================================

def _bootstrap_means(rng: np.random.Generator, values: np.ndarray,
                     n_samples: int, sample_size: int = None) -> np.ndarray:
    """
    通用 Bootstrap 均值重抽样。

    Parameters
    ----------
    rng : np.random.Generator
    values : np.ndarray
        原始数据
    n_samples : int
        Bootstrap 迭代次数
    sample_size : int, optional
        每次抽样大小，默认与原始数据相同

    Returns
    -------
    np.ndarray of shape (n_samples,)
    """
    if sample_size is None:
        sample_size = len(values)
    return np.array([
        np.mean(rng.choice(values, size=sample_size, replace=True))
        for _ in range(n_samples)
    ])


def _weighted_bootstrap_means(rng: np.random.Generator, values: np.ndarray,
                              weights: np.ndarray, n_samples: int,
                              sample_size: int = None) -> np.ndarray:
    """
    加权 Bootstrap 均值重抽样（用于压力测试）。

    Parameters
    ----------
    rng : np.random.Generator
    values : np.ndarray
    weights : np.ndarray
        抽样概率（自动归一化）
    n_samples : int
    sample_size : int, optional

    Returns
    -------
    np.ndarray of shape (n_samples,)
    """
    if sample_size is None:
        sample_size = len(values)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return np.array([
        np.mean(rng.choice(values, size=sample_size, replace=True, p=weights))
        for _ in range(n_samples)
    ])


# ================================================================
# MonteCarloEngine
# ================================================================

class MonteCarloEngine:
    """蒙特卡洛模拟引擎 -- 基于历史供应链数据"""

    def __init__(self, df: pd.DataFrame, random_seed: int = 42):
        self.df = df.copy()
        self.rng = np.random.default_rng(random_seed)

        self._disruption_rate = df["Disruption_Occurred"].mean()
        self._routes = df["Route"].unique()
        self._modes = df["Transport_Mode"].unique()

    # ================================================================
    # 1. 中断率分布 (Bootstrap)
    # ================================================================

    def simulate_disruption_rate(self, n_samples: int = 10000) -> dict:
        """Bootstrap 重抽样：估计全局中断率的抽样分布与置信区间。"""
        disruptions = self.df["Disruption_Occurred"].values
        boot_means = _bootstrap_means(self.rng, disruptions, n_samples)

        return {
            "metric": "Global Disruption Rate",
            "observed": self._disruption_rate,
            "simulated_mean": float(np.mean(boot_means)),
            "simulated_std": float(np.std(boot_means)),
            "ci_95_lower": float(np.percentile(boot_means, 2.5)),
            "ci_95_upper": float(np.percentile(boot_means, 97.5)),
            "ci_99_lower": float(np.percentile(boot_means, 0.5)),
            "ci_99_upper": float(np.percentile(boot_means, 99.5)),
            "samples": boot_means,
        }

    def simulate_disruption_by_mode(self, n_samples: int = 10000) -> pd.DataFrame:
        """Bootstrap：各运输模式中断率的抽样分布。"""
        results = []
        for mode in self._modes:
            mask = self.df["Transport_Mode"] == mode
            disruptions = self.df.loc[mask, "Disruption_Occurred"].values
            n = len(disruptions)
            if n == 0:
                continue

            boot_means = _bootstrap_means(self.rng, disruptions, n_samples)

            results.append({
                "Transport_Mode": mode,
                "Observed_Rate": np.mean(disruptions),
                "Simulated_Mean": float(np.mean(boot_means)),
                "CI_95_Lower": float(np.percentile(boot_means, 2.5)),
                "CI_95_Upper": float(np.percentile(boot_means, 97.5)),
                "CV": float(np.std(boot_means) / np.mean(boot_means)) if np.mean(boot_means) > 0 else 0,
            })

        return pd.DataFrame(results).sort_values("Observed_Rate", ascending=False)

    # ================================================================
    # 2. 场景压力测试
    # ================================================================

    def stress_test_geopolitical_risk(
        self, risk_shift: float = 2.0, n_samples: int = 10000
    ) -> dict:
        """
        压力测试：地缘政治风险评分上移 -> 加权 Bootstrap -> 中断率变化。

        权重 = Geopolitical_Risk_Score + risk_shift，模拟高评分事件多发场景。
        """
        df = self.df.copy()
        y = df["Disruption_Occurred"].values

        baseline_rates = _bootstrap_means(self.rng, y, n_samples)

        weights = np.clip(df["Geopolitical_Risk_Score"].values + risk_shift, 0.1, None)
        stressed_rates = _weighted_bootstrap_means(self.rng, y, weights, n_samples)

        return {
            "scenario": f"Geopolitical Risk +{risk_shift:.1f}",
            "baseline_mean": float(np.mean(baseline_rates)),
            "stressed_mean": float(np.mean(stressed_rates)),
            "delta": float(np.mean(stressed_rates) - np.mean(baseline_rates)),
            "pct_change": float(
                (np.mean(stressed_rates) - np.mean(baseline_rates))
                / np.mean(baseline_rates) * 100
            ),
            "prob_worse": float(np.mean(stressed_rates > baseline_rates)),
            "baseline_samples": baseline_rates,
            "stressed_samples": stressed_rates,
        }

    def stress_test_weather(self, n_samples: int = 10000) -> dict:
        """压力测试：恶劣天气（暴风雨+飓风）概率 x3，中断率变化？"""
        df = self.df.copy()
        y = df["Disruption_Occurred"].values

        baseline_rates = _bootstrap_means(self.rng, y, n_samples)

        severe_mask = df["Weather_Condition"].isin(["Storm", "Hurricane"])
        n = len(y)
        weights = np.ones(n)
        weights[severe_mask.values] *= 3.0
        stressed_rates = _weighted_bootstrap_means(self.rng, y, weights, n_samples)

        return {
            "scenario": "Severe Weather Probability x3",
            "baseline_mean": float(np.mean(baseline_rates)),
            "stressed_mean": float(np.mean(stressed_rates)),
            "delta": float(np.mean(stressed_rates) - np.mean(baseline_rates)),
            "pct_change": float(
                (np.mean(stressed_rates) - np.mean(baseline_rates))
                / np.mean(baseline_rates) * 100
            ),
            "prob_worse": float(np.mean(stressed_rates > baseline_rates)),
            "baseline_samples": baseline_rates,
            "stressed_samples": stressed_rates,
        }

    # ================================================================
    # 3. 路线级 VaR (Value-at-Risk)
    # ================================================================

    def route_var_simulation(self, top_n: int = 10, n_samples: int = 10000) -> pd.DataFrame:
        """对 Top N 路线 Bootstrap 估计时效，计算 95% / 99% VaR 和 CVaR。"""
        top_routes = self.df["Route"].value_counts().head(top_n).index
        results = []

        for route in top_routes:
            lt = self.df.loc[self.df["Route"] == route, "Lead_Time_Days"].values
            n_route = len(lt)
            if n_route < 5:
                continue

            boot_means = _bootstrap_means(self.rng, lt, n_samples)

            results.append({
                "Route": route,
                "Shipments": n_route,
                "Observed_Mean_LT": np.mean(lt),
                "Observed_Std_LT": np.std(lt),
                "Simulated_Mean": float(np.mean(boot_means)),
                "VaR_95": float(np.percentile(boot_means, 95)),
                "VaR_99": float(np.percentile(boot_means, 99)),
                "CVaR_95": float(np.mean(boot_means[boot_means >= np.percentile(boot_means, 95)])),
            })

        return pd.DataFrame(results).sort_values("VaR_95", ascending=False)

    # ================================================================
    # 4. 总风险成本模拟
    # ================================================================

    def simulate_total_cost(
        self,
        cost_per_disruption: float = 5000.0,
        cost_per_day_overrun: float = 200.0,
        n_samples: int = 10000,
    ) -> dict:
        """Bootstrap 模拟总供应链风险成本 = 中断成本 + 时效超限成本。"""
        df = self.df.copy()
        n = len(df)
        baseline_lt = df["Lead_Time_Days"].median()

        total_costs = []
        disruption_counts = []

        for _ in range(n_samples):
            sample_idx = self.rng.choice(n, size=n, replace=True)
            sample = df.iloc[sample_idx]

            n_disruptions = sample["Disruption_Occurred"].sum()
            disruption_cost = n_disruptions * cost_per_disruption
            overrun_days = np.maximum(0, sample["Lead_Time_Days"] - baseline_lt)
            overrun_cost = overrun_days.sum() * cost_per_day_overrun

            total_costs.append(disruption_cost + overrun_cost)
            disruption_counts.append(n_disruptions)

        total_costs = np.array(total_costs)

        return {
            "scenario": "Total Supply Chain Risk Cost",
            "cost_per_disruption": cost_per_disruption,
            "cost_per_day_overrun": cost_per_day_overrun,
            "mean_cost": float(np.mean(total_costs)),
            "std_cost": float(np.std(total_costs)),
            "median_cost": float(np.median(total_costs)),
            "VaR_95_cost": float(np.percentile(total_costs, 95)),
            "VaR_99_cost": float(np.percentile(total_costs, 99)),
            "CVaR_95_cost": float(np.mean(total_costs[total_costs >= np.percentile(total_costs, 95)])),
            "mean_disruptions": float(np.mean(disruption_counts)),
            "cost_samples": total_costs,
        }

    # ================================================================
    # 5. 提升可靠性的 ROI 模拟（含 train/test 分离）
    # ================================================================

    def simulate_reliability_improvement(
        self,
        improvement_pct: float = 0.10,
        n_samples: int = 10000,
    ) -> dict:
        """
        模拟"承运商可靠性提升 X%" 对中断率的关联性影响。

        WARNING: 此为关联性估计，非因果推断。
        未观测混杂因素（如承运商规模、路线特征）可能同时影响可靠性评分和中断率。

        改进点：
          - 使用 train/test split 诚实评估模型
          - 报告保留集 AUC 作为模型可靠性指标
          - Bootstrap 使用保留集的预测概率分布，避免 overfitting bias
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        df = self.df.copy()

        # ---- Step 1: Train/test split ----
        X = df[["Carrier_Reliability_Score", "Geopolitical_Risk_Score"]].values
        y = df["Disruption_Occurred"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # ---- Step 2: Fit on training set only ----
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)

        # Evaluate honestly on test set
        y_proba_test = lr.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba_test)

        # ---- Step 3: Use the model to estimate improvement ----
        # 在全部测试集上计算"改进后"的预测概率
        X_test_improved = X_test.copy()
        X_test_improved[:, 0] = np.clip(X_test_improved[:, 0] * (1 + improvement_pct), 0, 1.0)
        prob_improved = lr.predict_proba(X_test_improved)[:, 1]
        prob_baseline = y_proba_test

        # ---- Step 4: Bootstrap the difference ----
        n_test = len(X_test)
        baseline_rates = np.array([
            np.mean(self.rng.binomial(1, self.rng.choice(prob_baseline, size=n_test, replace=True)))
            for _ in range(n_samples)
        ])

        improved_rates = np.array([
            np.mean(self.rng.binomial(1, self.rng.choice(prob_improved, size=n_test, replace=True)))
            for _ in range(n_samples)
        ])

        # ---- Step 5: Also bootstrap observed rate for reference ----
        _ = _bootstrap_means(self.rng, y_test, n_samples)

        return {
            "scenario": f"Carrier Reliability +{improvement_pct:.0%}",
            "method": "associational (not causal)",
            "model_auc_test": round(float(auc), 4),
            "baseline_mean": float(np.mean(baseline_rates)),
            "improved_mean": float(np.mean(improved_rates)),
            "observed_test_rate": float(np.mean(y_test)),
            "delta": float(np.mean(baseline_rates) - np.mean(improved_rates)),
            "pct_reduction": float(
                (np.mean(baseline_rates) - np.mean(improved_rates))
                / np.mean(baseline_rates) * 100
            ),
            "prob_improvement": float(np.mean(improved_rates < baseline_rates)),
            "baseline_samples": baseline_rates,
            "improved_samples": improved_rates,
        }


# ================================================================
# 高层运行函数
# ================================================================

def run_monte_carlo_analysis(engine: MonteCarloEngine, n_samples: int = 10000) -> dict:
    """运行完整的蒙特卡洛分析流程。"""
    print("\n" + "=" * 55)
    print("[蒙特卡洛模拟分析]")
    print(f"   Simulation Samples: {n_samples:,}")
    print("=" * 55)

    results = {}

    # 1. 中断率 Bootstrap
    print("\n  [1/5] 全局中断率分布 (Bootstrap)...")
    results["disruption_rate"] = engine.simulate_disruption_rate(n_samples=n_samples)
    print(f"        观测值: {results['disruption_rate']['observed']:.4f}")
    print(f"        95% CI: [{results['disruption_rate']['ci_95_lower']:.4f}, "
          f"{results['disruption_rate']['ci_95_upper']:.4f}]")

    # 2. 各运输模式中断率
    print("\n  [2/5] 按运输模式中断率 Bootstrap...")
    results["disruption_by_mode"] = engine.simulate_disruption_by_mode(n_samples=n_samples)
    for _, row in results["disruption_by_mode"].iterrows():
        print(f"        {row['Transport_Mode']:<8s}: {row['Observed_Rate']:.4f} "
              f"[{row['CI_95_Lower']:.4f}, {row['CI_95_Upper']:.4f}]")

    # 3. 压力测试
    print("\n  [3/5] 场景压力测试...")
    results["stress_geo"] = engine.stress_test_geopolitical_risk(risk_shift=2.0, n_samples=n_samples)
    print(f"        地缘风险 +2.0: 中断率 "
          f"{results['stress_geo']['baseline_mean']:.4f} -> "
          f"{results['stress_geo']['stressed_mean']:.4f} "
          f"(+{results['stress_geo']['pct_change']:.1f}%)")

    results["stress_weather"] = engine.stress_test_weather(n_samples=n_samples)
    print(f"        恶劣天气 x3: 中断率 "
          f"{results['stress_weather']['baseline_mean']:.4f} -> "
          f"{results['stress_weather']['stressed_mean']:.4f} "
          f"(+{results['stress_weather']['pct_change']:.1f}%)")

    # 4. 路线 VaR
    print("\n  [4/5] 路线级时效 VaR...")
    results["route_var"] = engine.route_var_simulation(top_n=10, n_samples=n_samples)
    if len(results["route_var"]) > 0:
        worst = results["route_var"].iloc[0]
        print(f"        最差路线: {worst['Route']}, VaR_95 = {worst['VaR_95']:.1f} days")

    # 5. 总成本模拟
    print("\n  [5/5] 总风险成本模拟...")
    results["total_cost"] = engine.simulate_total_cost(n_samples=n_samples)
    tc = results["total_cost"]
    print(f"        日均成本: ${tc['mean_cost']:,.0f}")
    print(f"        95% VaR:  ${tc['VaR_95_cost']:,.0f}")
    print(f"        CVaR_95:  ${tc['CVaR_95_cost']:,.0f}")

    # 6. ROI 模拟
    print("\n  [Bonus] 可靠性提升 ROI 模拟...")
    results["roi"] = engine.simulate_reliability_improvement(improvement_pct=0.10, n_samples=n_samples)
    print(f"        模型 AUC (test): {results['roi']['model_auc_test']:.4f}")
    print(f"        可靠性 +10%: 中断率降低 {results['roi']['pct_reduction']:.1f}%")
    print("        [关联性估计，非因果推断。详见 README Limitations]")

    print("\n  [蒙特卡洛模拟完成]")
    return results


if __name__ == "__main__":
    from data_loader import prepare_data

    df = prepare_data()
    engine = MonteCarloEngine(df, random_seed=42)
    results = run_monte_carlo_analysis(engine, n_samples=5000)
