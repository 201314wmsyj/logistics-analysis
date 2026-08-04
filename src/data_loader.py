"""
数据加载与预处理模块
====================
功能：加载全球供应链数据集，进行数据清洗、特征工程和预处理。

数据集：Global Supply Chain Risk & Logistics (2024-2026)
来源：Kaggle - nudratabbas/global-supply-chain-risk-and-logistics-2024-2026
记录数：5,000 条国际货运记录
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_raw_data(data_path: str | None = None) -> pd.DataFrame:
    """
    加载原始 CSV 数据。

    Parameters
    ----------
    data_path : str, optional
        数据文件路径，默认为项目 data/ 目录下的文件

    Returns
    -------
    pd.DataFrame
        原始数据框
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "global_supply_chain_risk_2026.csv"

    df = pd.read_csv(data_path, parse_dates=["Date"])
    print(f"[数据加载] 已加载 {len(df):,} 条记录, {len(df.columns)} 个字段")
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据质量校验（在清洗前运行）：
      - 检查必需列是否存在
      - 检查数值列是否有不合理值（负时效、零距离等）
      - 检查类别列取值范围
      - 检查数据完整性

    如果发现严重问题，抛出 ValueError 并说明具体原因。
    轻度问题打印警告但继续执行。

    Returns
    -------
    pd.DataFrame（校验通过的数据框）
    """
    # ---- 必需列 ----
    expected_cols = [
        "Shipment_ID", "Date", "Origin_Port", "Destination_Port",
        "Transport_Mode", "Product_Category", "Distance_km", "Weight_MT",
        "Fuel_Price_Index", "Geopolitical_Risk_Score", "Weather_Condition",
        "Carrier_Reliability_Score", "Lead_Time_Days", "Disruption_Occurred",
    ]
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"缺失必需列: {missing_cols}")      #raise主动抛出异常,阻止后续代码运行，如果空，跳过执行
    print(f"[数据校验] 所有 {len(expected_cols)} 个必需列存在 OK")

    # ---- 数值范围校验 ----
    checks = [
        ("Distance_km", 0, 30000, "距离"),
        ("Weight_MT", 0, 10000, "重量"),
        ("Lead_Time_Days", 0, 365, "时效"),
        ("Fuel_Price_Index", 0, 10, "燃油价格指数"),
        ("Geopolitical_Risk_Score", 0, 10, "地缘政治风险评分"),
        ("Carrier_Reliability_Score", 0, 1.0, "承运商可靠性"),
    ]
    for col, lo, hi, label in checks:
        if col in df.columns:
            outliers = df[(df[col] < lo) | (df[col] > hi)]
            if len(outliers) > 0:
                pct = len(outliers) / len(df) * 100
                print(f"[数据校验] WARNING {label} ({col}): {len(outliers)} 条({pct:.1f}%) 超出 [{lo}, {hi}]")

    # ---- 类别列校验 ----
    expected_modes = {"Air", "Rail", "Road", "Sea"}
    actual_modes = set(df["Transport_Mode"].unique()) if "Transport_Mode" in df.columns else set()   #三元运算符 结果 if 条件 else 结果B 
    if actual_modes - expected_modes:
        print(f"[数据校验] WARNING 未知运输模式: {actual_modes - expected_modes}")

    expected_products = {"Automotive", "Electronics", "Perishables", "Pharmaceuticals", "Textiles"}
    actual_products = set(df["Product_Category"].unique()) if "Product_Category" in df.columns else set()
    if actual_products - expected_products:
        print(f"[数据校验] WARNING 未知产品类别: {actual_products - expected_products}")

    expected_weather = {"Clear", "Fog", "Hurricane", "Rain", "Storm"}
    actual_weather = set(df["Weather_Condition"].unique()) if "Weather_Condition" in df.columns else set()
    if actual_weather - expected_weather:
        print(f"[数据校验] WARNING 未知天气条件: {actual_weather - expected_weather}")

    # ---- Disruption 列 ----
    if "Disruption_Occurred" in df.columns:
        valid = set(df["Disruption_Occurred"].unique())
        if valid - {0, 1}:
            print(f"[数据校验] WARNING Disruption_Occurred 包含非 0/1 值: {valid - {0, 1}}")

    print("[数据校验] 质量校验完成 OK")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗：
    - 处理缺失值
    - 转换数据类型
    - 移除异常值

    Parameters
    ----------
    df : pd.DataFrame
        原始数据框

    Returns
    -------
    pd.DataFrame
        清洗后的数据框
    """
    # 检查缺失值
    missing = df.isnull().sum()    #对每个单元格判断是否缺失（NaN、None、NaT 等），返回布尔型 DataFrame，按列求和，True=1，False=0，得到每列的缺失值数量
    if missing.sum() > 0:
        print(f"[数据清洗] 发现缺失值:\n{missing[missing > 0]}")
        # 数值列用中位数填充
        num_cols = df.select_dtypes(include=[np.number]).columns   #按类型筛选，只保留数值列
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        # 类别列用众数填充
        cat_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])  #计算众数，由于众数可能有多个，取第一个
    else:
        print("[数据清洗] 无缺失值，数据质量良好 OK")

    # 确保日期格式
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    print(f"[数据清洗] 清洗完成, 剩余 {len(df):,} 条记录")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征工程：
    - 提取时间特征（年、月、季度、星期）
    - 计算派生指标（单位距离时效、吨公里等）
    - 创建风险等级分类

    Parameters
    ----------
    df : pd.DataFrame
        清洗后的数据框

    Returns
    -------
    pd.DataFrame
        包含新特征的数据框
    """
    df = df.copy()

    # --- 时间特征 ---
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Month_Name"] = df["Date"].dt.month_name()
    df["DayOfWeek"] = df["Date"].dt.dayofweek  # 0=星期一

    # --- 派生指标 ---
    # 时效效率：每天运输多少公里（越高越好）
    df["Speed_km_per_day"] = df["Distance_km"] / df["Lead_Time_Days"].replace(0, np.nan)
    df["Speed_km_per_day"] = df["Speed_km_per_day"].fillna(df["Speed_km_per_day"].median())

    # 吨公里 = 重量 x 距离（运输工作量）
    df["Ton_KM"] = df["Weight_MT"] * df["Distance_km"] / 1000  # 千吨公里

    # 单位重量时效（每吨货物的在途天数）
    df["Lead_Time_per_Ton"] = df["Lead_Time_Days"] / df["Weight_MT"].replace(0, np.nan)
    df["Lead_Time_per_Ton"] = df["Lead_Time_per_Ton"].fillna(df["Lead_Time_per_Ton"].median())

    # --- 风险等级分类 ---
    risk_bins = [0, 3, 6, 10]
    risk_labels = ["Low", "Medium", "High"]
    df["Risk_Level"] = pd.cut(
        df["Geopolitical_Risk_Score"],
        bins=risk_bins,
        labels=risk_labels,
        include_lowest=True,
    )

    # --- 时效等级分类 ---
    lead_bins = [0, 5, 15, 30, float("inf")]
    lead_labels = ["Express (<5d)", "Standard (5-15d)", "Long (15-30d)", "Very Long (>30d)"]
    df["Lead_Time_Category"] = pd.cut(
        df["Lead_Time_Days"],
        bins=lead_bins,
        labels=lead_labels,
        include_lowest=True,
    )

    # --- 路线名称 ---
    df["Route"] = df["Origin_Port"] + " -> " + df["Destination_Port"]

    print(f"[特征工程] 新增 {len(df.columns) - 14} 个特征")

    return df


def prepare_data(data_path: str | None = None) -> pd.DataFrame:
    """
    完整的数据准备流程：加载 -> 校验 -> 清洗 -> 特征工程

    Parameters
    ----------
    data_path : str, optional
        数据文件路径

    Returns
    -------
    pd.DataFrame
        处理后的完整数据框
    """
    df = load_raw_data(data_path)
    df = validate_data(df)              #  清洗前先校验数据质量
    df = clean_data(df)
    df = engineer_features(df)
    print(f"[数据准备] 完成！数据维度: {df.shape}")
    return df


if __name__ == "__main__":
    df = prepare_data()
    print("\n数据概览：")
    print(df.info())
    print("\n数值统计：")
    print(df.describe())
