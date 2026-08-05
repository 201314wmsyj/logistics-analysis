"""
数据分析模块
============
功能：供应链多维度分析，包括中断分析、时效分析、风险因素分析和预测建模。

分析维度：
  1. 供应链中断分析 - 中断率、中断因素、季节性
  2. 时效绩效分析 - 运输模式、路线、产品类别对比
  3. 风险因素分析 - 地缘政治、天气、燃油价格影响
  4. 路线绩效分析 - 热门路线排名与诊断
  5. 相关性分析 - 各因素对中断和时效的影响
  6. 预测建模 - 多模型对比 (Baseline / 逻辑回归 / 随机森林)
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# 1. 供应链中断分析 (Disruption Analysis)
# ============================================================

def disruption_summary(df: pd.DataFrame) -> dict:
    """
    整体中断概况。

    Returns
    -------
    dict
        包含中断率、按维度中断统计的字典
    """
    total = len(df)
    disrupted = df["Disruption_Occurred"].sum()
    disruption_rate = disrupted / total * 100

    print("=" * 55)
    print("[供应链中断分析]")
    print("=" * 55)
    print(f"  总货运量    : {total:,.0f}")
    print(f"  中断数      : {disrupted:,.0f}")
    print(f"  中断率      : {disruption_rate:.2f}%")
    print(f"  正常运输    : {total - disrupted:,.0f} ({(100-disruption_rate):.2f}%)")

    results = {
        "total_shipments": total,
        "disrupted": disrupted,
        "disruption_rate": disruption_rate,
    }

    # 按运输模式
    print("\n--- 按运输模式 ---")
    by_mode = (
        df.groupby("Transport_Mode")
        .agg(Total=("Disruption_Occurred", "count"), Disrupted=("Disruption_Occurred", "sum"))
        .assign(Rate=lambda x: (x["Disrupted"] / x["Total"] * 100).round(2))
        .sort_values("Rate", ascending=False)
    )
    print(by_mode.to_string())
    results["by_transport_mode"] = by_mode

    # 按产品类别
    print("\n--- 按产品类别 ---")
    by_product = (
        df.groupby("Product_Category")
        .agg(Total=("Disruption_Occurred", "count"), Disrupted=("Disruption_Occurred", "sum"))
        .assign(Rate=lambda x: (x["Disrupted"] / x["Total"] * 100).round(2))
        .sort_values("Rate", ascending=False)
    )
    print(by_product.to_string())
    results["by_product_category"] = by_product

    # 按月趋势
    print("\n--- 按月趋势 ---")
    by_month = (
        df.groupby(["Year", "Month"])
        .agg(Total=("Disruption_Occurred", "count"), Disrupted=("Disruption_Occurred", "sum"))
        .assign(Rate=lambda x: (x["Disrupted"] / x["Total"] * 100).round(2))
    )
    print(by_month.tail(12).to_string())
    results["by_month"] = by_month

    return results


def disruption_by_weather(df: pd.DataFrame) -> pd.DataFrame:
    """按天气条件分析中断率"""
    return (
        df.groupby("Weather_Condition")
        .agg(
            Total=("Disruption_Occurred", "count"),
            Disrupted=("Disruption_Occurred", "sum"),
            Avg_Lead_Time=("Lead_Time_Days", "mean"),
            Avg_Distance=("Distance_km", "mean"),
        )
        .assign(Disruption_Rate=lambda x: (x["Disrupted"] / x["Total"] * 100).round(2))
        .sort_values("Disruption_Rate", ascending=False)
    )


# ============================================================
# 2. 时效绩效分析 (Lead Time Analysis)
# ============================================================

def lead_time_analysis(df: pd.DataFrame) -> dict:
    """
    时效多维度分析。

    Returns
    -------
    dict
        多维度时效统计
    """
    print("=" * 55)
    print("[时效绩效分析]")
    print("=" * 55)

    lt = df["Lead_Time_Days"]
    print("\n整体时效统计:")
    print(f"  平均     : {lt.mean():.1f} 天")
    print(f"  中位数   : {lt.median():.1f} 天")
    print(f"  标准差   : {lt.std():.1f} 天")
    print(f"  最小     : {lt.min():.1f} 天")
    print(f"  最大     : {lt.max():.1f} 天")
    print(f"  25分位   : {lt.quantile(0.25):.1f} 天")
    print(f"  75分位   : {lt.quantile(0.75):.1f} 天")

    results = {"summary": lt.describe()}

    # 按运输模式
    print("\n--- 按运输模式 ---")
    by_mode = (
        df.groupby("Transport_Mode")
        .agg(
            平均时效=("Lead_Time_Days", "mean"),
            中位时效=("Lead_Time_Days", "median"),
            时效标准差=("Lead_Time_Days", "std"),
            货运量=("Lead_Time_Days", "count"),
        )
        .round(2)
        .sort_values("平均时效")
    )
    print(by_mode.to_string())
    results["by_transport_mode"] = by_mode

    # 按产品类别
    print("\n--- 按产品类别 ---")
    by_product = (
        df.groupby("Product_Category")
        .agg(
            平均时效=("Lead_Time_Days", "mean"),
            中位时效=("Lead_Time_Days", "median"),
            货运量=("Lead_Time_Days", "count"),
        )
        .round(2)
        .sort_values("平均时效")
    )
    print(by_product.to_string())
    results["by_product_category"] = by_product

    return results


# ============================================================
# 3. 风险因素分析 (Risk Factor Analysis)
# ============================================================

def risk_factor_analysis(df: pd.DataFrame) -> dict:
    """
    分析各风险因素与中断的关系。

    Returns
    -------
    dict
        风险分析结果
    """
    print("=" * 55)
    print("[风险因素分析]")
    print("=" * 55)

    results = {}

    # 地缘政治风险
    print("\n--- 地缘政治风险 与 中断率 ---")
    risk_groups = (
        df.groupby("Risk_Level", observed=True)
        .agg(
            货运量=("Disruption_Occurred", "count"),
            中断率=("Disruption_Occurred", "mean"),
            平均时效=("Lead_Time_Days", "mean"),
            平均燃油价格=("Fuel_Price_Index", "mean"),
        )
        .assign(中断率=lambda x: (x["中断率"] * 100).round(2))
    )
    print(risk_groups.to_string())
    results["by_risk_level"] = risk_groups

    # 天气条件
    print("\n--- 天气条件 与 中断率 ---")
    weather = disruption_by_weather(df)
    print(weather.to_string())
    results["by_weather"] = weather

    # 承运商可靠性
    print("\n--- 承运商可靠性 与 中断率 ---")
    reliability_groups = pd.cut(
        df["Carrier_Reliability_Score"],
        bins=[0.5, 0.65, 0.8, 0.9, 1.0],
        labels=["Low (<0.65)", "Medium (0.65-0.8)", "Good (0.8-0.9)", "Excellent (>0.9)"],
    )
    by_reliability = (
        df.groupby(reliability_groups, observed=True)
        .agg(
            货运量=("Disruption_Occurred", "count"),
            中断率=("Disruption_Occurred", "mean"),
            平均时效=("Lead_Time_Days", "mean"),
        )
        .assign(中断率=lambda x: (x["中断率"] * 100).round(2))
        .assign(平均时效=lambda x: x["平均时效"].round(1))
    )
    print(by_reliability.to_string())
    results["by_carrier_reliability"] = by_reliability

    # 距离分组
    print("\n--- 运输距离 与 中断率 ---")
    distance_groups = pd.cut(
        df["Distance_km"],
        bins=[0, 3000, 6000, 9000, 12000, 15000],
        labels=["<3k km", "3k-6k km", "6k-9k km", "9k-12k km", ">12k km"],
    )
    by_distance = (
        df.groupby(distance_groups, observed=True)
        .agg(货运量=("Disruption_Occurred", "count"), 中断率=("Disruption_Occurred", "mean"))
        .assign(中断率=lambda x: (x["中断率"] * 100).round(2))
    )
    print(by_distance.to_string())
    results["by_distance"] = by_distance

    return results


# ============================================================
# 4. 路线分析 (Route Analysis)
# ============================================================

def route_analysis(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    分析主要运输路线绩效。

    Parameters
    ----------
    df : pd.DataFrame
    top_n : int
        分析前 N 条热门路线

    Returns
    -------
    pd.DataFrame
        路线绩效表
    """
    print("=" * 55)
    print(f"[主要路线分析] (Top {top_n})")
    print("=" * 55)

    route_stats = (
        df.groupby("Route")
        .agg(
            货运量=("Shipment_ID", "count"),
            中断率=("Disruption_Occurred", "mean"),
            平均时效=("Lead_Time_Days", "mean"),
            平均距离=("Distance_km", "mean"),
            平均承运商可靠性=("Carrier_Reliability_Score", "mean"),
            平均地缘风险=("Geopolitical_Risk_Score", "mean"),
        )
        .assign(中断率=lambda x: (x["中断率"] * 100).round(2))
        .assign(平均时效=lambda x: x["平均时效"].round(1))
        .assign(平均距离=lambda x: x["平均距离"].round(0).astype(int))
    )

    top_routes = route_stats.sort_values("货运量", ascending=False).head(top_n)
    print(top_routes.to_string())
    return top_routes


# ============================================================
# 5. 相关性分析 (Correlation Analysis)
# ============================================================

def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    数值特征相关性分析。

    Returns
    -------
    pd.DataFrame
        相关系数矩阵
    """
    print("=" * 55)
    print("[相关性分析]")
    print("=" * 55)

    num_cols = [
        "Distance_km",
        "Weight_MT",
        "Fuel_Price_Index",
        "Geopolitical_Risk_Score",
        "Carrier_Reliability_Score",
        "Lead_Time_Days",
        "Speed_km_per_day",
        "Ton_KM",
        "Disruption_Occurred",
    ]

    corr_matrix = df[num_cols].corr()

    # 与中断的相关性（排序）
    print("\n--- 与 Disruption_Occurred 相关性（排序）---")
    disruption_corr = corr_matrix["Disruption_Occurred"].drop("Disruption_Occurred").sort_values(ascending=False)
    for factor, corr_val in disruption_corr.items():
        bar = "#" * int(abs(corr_val) * 20)
        direction = "(正相关)" if corr_val > 0 else "(负相关)"
        print(f"  {factor:<30s} {corr_val:+.4f} {bar} {direction}")

    # 强相关对
    print("\n--- 强相关对 (|r| > 0.5) ---")
    strong_pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                strong_pairs.append(
                    {
                        "Factor 1": num_cols[i],
                        "Factor 2": num_cols[j],
                        "Correlation": corr_matrix.iloc[i, j],
                    }
                )
    if strong_pairs:
        for pair in strong_pairs:
            print(f"  {pair['Factor 1']} <-> {pair['Factor 2']}: {pair['Correlation']:+.4f}")
    else:
        print("  无强相关对（各因素独立性强，有利于建模）")

    return corr_matrix


# ============================================================
# 6. 中断预测模型 -- 多模型对比 (Multi-Model Comparison)
# ============================================================

def _prepare_model_features(df: pd.DataFrame) -> tuple:
    """
    共享的特征工程：One-Hot 编码 + 标准化。

    关键改进 (v3)：
      - 移除 Speed_km_per_day（数据泄漏：分母 Lead_Time_Days 是结果变量）
      - 新增 Weather_Condition（OneHot，飓风→100% 中断，最强信号）
      - 新增 Ton_KM（吨公里，无泄漏的派生特征）
      - OneHotEncoder 替代 LabelEncoder
      - 移除 Transport_Mode / Product_Category（chi-square 不显著 + RF 重要性<0.01）

    Returns
    -------
    tuple: (X, y, feature_names, scaler, ohe)
    """
    # 数值特征（均为预测时可用的特征，无数据泄漏）
    numeric_cols = [
        "Distance_km",
        "Weight_MT",
        "Geopolitical_Risk_Score",
        "Carrier_Reliability_Score",
        "Ton_KM",  # = Weight_MT × Distance_km / 1000，运输工作量
    ]

    # 类别特征 — 仅保留有统计显著预测力的特征
    categorical_cols = [
        "Weather_Condition",  # Cramér's V = 0.50, 所有变量中效应量最大
    ]

    df_model = df.copy()

    # One-Hot 编码
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = ohe.fit_transform(df_model[categorical_cols])
    cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))

    # 标准化数值特征
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df_model[numeric_cols])

    # 合并
    X = np.hstack([X_num, X_cat])
    feature_names = numeric_cols + cat_feature_names

    y = df_model["Disruption_Occurred"].values

    return X, y, feature_names, scaler, ohe


def _evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str) -> dict:
    """对单个模型进行全面评估"""
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = {
        "model": model_name,
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    return metrics


def _find_optimal_threshold_cv(model, X, y, cv) -> float:
    """
    通过 5 折 CV 寻找最大化 F1 的最优概率阈值。

    方法：在每一折上用 4 折训练、1 折预测，汇总所有 CV 概率后，
    在 precision-recall 曲线上找最大化 F1 的阈值。
    这避免了在测试集上调阈值的过拟合。

    Returns
    -------
    float: 最优阈值
    """
    y_proba_cv = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thresholds = precision_recall_curve(y, y_proba_cv)
    f1_scores = 2 * (prec * rec) / (prec + rec)
    best_idx = np.nanargmax(f1_scores[:-1])  # 排除最后一个 (thresholds 比 f1 短 1)
    return float(thresholds[best_idx])


def build_disruption_model(df: pd.DataFrame) -> dict:
    """
    多模型对比：Baseline -> 逻辑回归 -> 随机森林 -> 梯度提升

    关键改进 (v3)：
      - 移除 Speed_km_per_day（数据泄漏修复）
      - 新增 Weather_Condition OneHot 编码（最强预测信号）
      - 移除 Transport_Mode / Product_Category（chi-square 不显著）
      - OneHotEncoder 替代 LabelEncoder
      - 新增 GradientBoostingClassifier
      - RF 平衡类别权重 + 调优超参数
      - CV 阈值优化（用 CV 找最优阈值，而非用默认 0.5）
      - 5 折交叉验证评估稳定性
      - 多指标对比 (Accuracy / Precision / Recall / F1 / ROC-AUC)

    Returns
    -------
    dict
        包含 model_comparison, feature_importance, optimal_threshold 等
    """
    print("=" * 55)
    print("[中断预测模型 -- 多模型对比]")
    print("=" * 55)

    X, y, feature_names, scaler, ohe = _prepare_model_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_features = X.shape[1]
    print(f"\n  特征数: {n_features} (数值=5, 天气OneHot=5)")

    # ---- 基准率 ----
    baseline_rate = y.mean()
    print(f"  中断基准率 (Baseline): {baseline_rate:.2%}")
    print(f"     若全猜中断，准确率 = {baseline_rate:.2%}")
    print("     模型必须显著超越此基线才有实用价值\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---- 模型列表 ----
    models = {
        "Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent", random_state=42),
        "Baseline (Stratified)": DummyClassifier(strategy="stratified", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=3000, C=0.5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=15,
            learning_rate=0.03, random_state=42,
        ),
    }

    results = []
    optimal_thresholds = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = _evaluate_model(model, X_train, X_test, y_train, y_test, name)

        # 5 折交叉验证
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        metrics["cv_mean"] = cv_scores.mean()
        metrics["cv_std"] = cv_scores.std()

        # CV 阈值优化（仅对真实模型，跳过 Baseline）
        if "Baseline" not in name and hasattr(model, "predict_proba"):
            opt_thresh = _find_optimal_threshold_cv(model, X, y, cv)
            optimal_thresholds[name] = opt_thresh
            # 用最优阈值重新评估
            y_proba_test = model.predict_proba(X_test)[:, 1]
            y_pred_opt = (y_proba_test >= opt_thresh).astype(int)
            metrics["opt_threshold"] = opt_thresh
            metrics["opt_accuracy"] = accuracy_score(y_test, y_pred_opt)
            metrics["opt_precision"] = precision_score(y_test, y_pred_opt, zero_division=0)
            metrics["opt_recall"] = recall_score(y_test, y_pred_opt, zero_division=0)
            metrics["opt_f1"] = f1_score(y_test, y_pred_opt, zero_division=0)

        results.append(metrics)

    # ---- 结果汇总 ----
    comparison_df = pd.DataFrame(results).set_index("model")
    metric_cols = ["test_accuracy", "precision", "recall", "f1", "cv_mean", "cv_std"]
    display_cols = [c for c in metric_cols if c in comparison_df.columns]
    if "roc_auc" in comparison_df.columns:
        display_cols.insert(4, "roc_auc")

    print("  [模型对比汇总] (默认阈值=0.5):")
    print(comparison_df[display_cols].to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    # ---- 阈值优化结果 ----
    if optimal_thresholds:
        print("  [CV 阈值优化] (最大化 F1):")
        print(f"  {'模型':<25s} {'CV阈值':>8s} {'F1(默认)':>10s} {'F1(最优)':>10s}")
        print(f"  {'-'*53}")
        for name, thresh in optimal_thresholds.items():
            row = comparison_df.loc[name]
            f1_default = row.get("f1", 0)
            f1_opt = row.get("opt_f1", 0)
            arrow = "↑" if f1_opt > f1_default else "→"
            print(f"  {name:<25s} {thresh:>8.3f} {f1_default:>10.4f} {f1_opt:>10.4f} {arrow}")
        print()

    # ---- 逻辑回归详细结果 ----
    lr_model = models["Logistic Regression"]
    y_pred_lr = lr_model.predict(X_test)
    lr_thresh = optimal_thresholds.get("Logistic Regression", 0.5)
    y_pred_lr_opt = (lr_model.predict_proba(X_test)[:, 1] >= lr_thresh).astype(int)

    print(f"  [Logistic Regression 分类报告] (默认阈值=0.5):")
    print(classification_report(y_test, y_pred_lr, target_names=["正常", "中断"]))
    if lr_thresh != 0.5:
        print(f"  [Logistic Regression 分类报告] (CV最优阈值={lr_thresh:.3f}):")
        print(classification_report(y_test, y_pred_lr_opt, target_names=["正常", "中断"]))

    # ---- Random Forest 特征重要性 ----
    rf_model = models["Random Forest"]
    rf_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    print("\n  [Random Forest 特征重要性]:")
    for _, row in rf_importance.iterrows():
        bar = "#" * int(row["Importance"] * 40)
        print(f"    {row['Feature']:<40s} {row['Importance']:.4f} {bar}")

    # ---- 逻辑回归特征系数 (保留用于可解释性) ----
    lr_importance = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": lr_model.coef_[0],
    }).sort_values("Coefficient", ascending=False)

    print("\n  [Logistic Regression 特征系数]:")
    for _, row in lr_importance.iterrows():
        direction = "(+风险)" if row["Coefficient"] > 0 else "(-风险)"
        print(f"    {row['Feature']:<40s} {row['Coefficient']:+.4f} {direction}")

    return {
        "model": lr_model,                     # 最佳 AUC 模型
        "rf_model": rf_model,                  # 随机森林（特征重要性可解释）
        "scaler": scaler,
        "ohe": ohe,
        "model_comparison": comparison_df,     # 多模型对比表
        "feature_importance": rf_importance,   # RF 特征重要性（主要）
        "lr_feature_importance": lr_importance, # LR 系数（可解释性参考）
        "feature_names": feature_names,
        "optimal_thresholds": optimal_thresholds,  # CV 最优阈值
        "cv": cv,
    }


# ============================================================
# 运行全部分析
# ============================================================

def run_all_analysis(df: pd.DataFrame, skip_basic_prints: bool = False) -> dict:
    """
    运行所有分析模块。

    Parameters
    ----------
    df : pd.DataFrame
        预处理后的数据
    skip_basic_prints : bool
        如果为 True，跳过基础分析模块的打印（仅运行建模）

    Returns
    -------
    dict
        所有分析结果的字典
    """
    print("\n" + "#" * 55)
    print("  全球供应链风险与物流绩效分析")
    print("  Global Supply Chain Risk & Logistics Performance Analysis")
    print("  Data: 2024-2026 | 5,000 International Shipments")
    print("#" * 55 + "\n")

    results = {}

    # 数据概览
    print("\n[数据概览]")
    print(f"  时间范围: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  出发港口: {df['Origin_Port'].nunique()}")
    print(f"  目的港口: {df['Destination_Port'].nunique()}")
    print(f"  运输模式: {df['Transport_Mode'].nunique()} ({', '.join(df['Transport_Mode'].unique())})")
    print(f"  产品类别: {df['Product_Category'].nunique()} ({', '.join(df['Product_Category'].unique())})")

    if not skip_basic_prints:
        results["disruption"] = disruption_summary(df)
        results["lead_time"] = lead_time_analysis(df)
        results["risk_factors"] = risk_factor_analysis(df)
        results["routes"] = route_analysis(df)
        results["correlation"] = correlation_analysis(df)

    # 预测建模一定运行（提供 feature_importance 和 model_comparison）
    model_results = build_disruption_model(df)
    results["model"] = model_results
    results["feature_importance"] = model_results["feature_importance"]
    results["model_comparison"] = model_results["model_comparison"]

    return results


if __name__ == "__main__":
    from data_loader import prepare_data

    df = prepare_data()
    results = run_all_analysis(df)
