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

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
    """共享的特征工程：编码 + 标准化，返回 X, y, feature_cols, scaler"""
    feature_cols = [
        "Distance_km",
        "Weight_MT",
        "Fuel_Price_Index",
        "Geopolitical_Risk_Score",
        "Carrier_Reliability_Score",
    ]

    df_model = df.copy()
    le_mode = LabelEncoder()
    df_model["Transport_Mode_Encoded"] = le_mode.fit_transform(df_model["Transport_Mode"])

    le_product = LabelEncoder()
    df_model["Product_Category_Encoded"] = le_product.fit_transform(df_model["Product_Category"])

    feature_cols.extend(["Transport_Mode_Encoded", "Product_Category_Encoded"])

    X = df_model[feature_cols].values
    y = df_model["Disruption_Occurred"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols, scaler


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


def build_disruption_model(df: pd.DataFrame) -> dict:
    """
    多模型对比：Baseline -> 逻辑回归 -> 随机森林

    关键改进：
      - DummyClassifier 作为基准线（避免准确率不如瞎猜的尴尬）
      - 随机森林捕捉非线性关系
      - 5 折交叉验证评估稳定性
      - 多指标对比 (Accuracy / Precision / Recall / F1 / ROC-AUC)

    Returns
    -------
    dict
        包含 model_comparison, feature_importance 等
    """
    print("=" * 55)
    print("[中断预测模型 -- 多模型对比]")
    print("=" * 55)

    X, y, feature_cols, scaler = _prepare_model_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- 基准率 ----
    baseline_rate = y.mean()
    print(f"\n  中断基准率 (Baseline): {baseline_rate:.2%}")
    print(f"     若全猜中断，准确率 = {baseline_rate:.2%}")
    print("     模型必须显著超越此基线才有实用价值\n")

    # ---- 模型列表 ----
    models = {
        "Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent", random_state=42),
        "Baseline (Stratified)": DummyClassifier(strategy="stratified", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            random_state=42, n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = _evaluate_model(model, X_train, X_test, y_train, y_test, name)

        # 5 折交叉验证
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        metrics["cv_mean"] = cv_scores.mean()
        metrics["cv_std"] = cv_scores.std()

        results.append(metrics)

    # ---- 结果汇总 ----
    comparison_df = pd.DataFrame(results).set_index("model")
    metric_cols = ["test_accuracy", "precision", "recall", "f1", "cv_mean", "cv_std"]
    display_cols = [c for c in metric_cols if c in comparison_df.columns]
    if "roc_auc" in comparison_df.columns:
        display_cols.insert(4, "roc_auc")

    print("  [模型对比汇总]:")
    print(comparison_df[display_cols].to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    # ---- Random Forest 详细结果 ----
    rf_model = models["Random Forest"]
    y_pred_rf = rf_model.predict(X_test)
    print("  [Random Forest 分类报告]:")
    print(classification_report(y_test, y_pred_rf, target_names=["正常", "中断"]))

    # ---- Random Forest 特征重要性 ----
    rf_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    print("\n  [Random Forest 特征重要性]:")
    for _, row in rf_importance.iterrows():
        bar = "#" * int(row["Importance"] * 40)
        print(f"    {row['Feature']:<32s} {row['Importance']:.4f} {bar}")

    # ---- 逻辑回归特征系数 (保留用于可解释性) ----
    lr_model = models["Logistic Regression"]
    lr_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": lr_model.coef_[0],
    }).sort_values("Coefficient", ascending=False)

    return {
        "model": rf_model,                     # 最佳模型
        "scaler": scaler,
        "model_comparison": comparison_df,     # 多模型对比表
        "feature_importance": rf_importance,   # RF 特征重要性（主要）
        "lr_feature_importance": lr_importance, # LR 系数（可解释性参考）
        "feature_names": feature_cols,
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
