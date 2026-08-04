"""
可视化模块
==========
功能：使用 matplotlib 和 seaborn 创建专业级供应链分析图表。

图表清单：
  1. 中断率按运输模式 + 产品类别 (分组柱状图)
  2. 时效分布对比 (箱线图)
  3. 月度中断率趋势 (折线图)
  4. 地缘政治风险 vs 中断率 (柱状图)
  5. 天气条件影响 (水平柱状图)
  6. 相关系数热力图
  7. 前10路线中断率 vs 时效 (散点气泡图)
  8. 特征重要性 (水平柱状图)
  9. 蒙特卡洛中断率分布 (Bootstrap 直方图)
  10. 压力测试对比 (多子图)
  11. 多模型对比 (分组柱状图)
  12. 路线 VaR (水平柱状图)
  13. 总成本分布 (模拟直方图)
  14. 统计效应量汇总 (水平柱状图)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# 全局样式设置
# ============================================================

# 中文字体设置 - 尝试多种字体以兼容不同系统
import matplotlib.font_manager as fm

# 查找可用的中文字体
_chinese_fonts = [f.name for f in fm.fontManager.ttflist if any(kw in f.name.lower() for kw in ['microsoft yahei', 'simhei', 'simsun', 'noto sans cjk', 'wenquanyi', 'source han'])]

if _chinese_fonts:
    plt.rcParams["font.family"] = _chinese_fonts[0]
    print(f"[可视化] 使用中文字体: {_chinese_fonts[0]}")

# Seaborn 风格
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

# 自定义调色板
PALETTE_BLUE = "#2E86AB"
PALETTE_RED = "#A23B72"
PALETTE_GREEN = "#3A7D44"
PALETTE_ORANGE = "#F18F01"
PALETTE_PURPLE = "#6A4C93"
CAT_PALETTE = ["#2E86AB", "#A23B72", "#3A7D44", "#F18F01", "#6A4C93", "#D62828", "#457B9D"]
DIVERGING_PALETTE = sns.diverging_palette(240, 10, as_cmap=True)

# 输出目录
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
DPI = 150  # 高分辨率输出


def _save_and_show(fig, filename: str):
    """保存图片到 output 目录"""
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"  [OK] 已保存: {path}")
    plt.close(fig)


# ============================================================
# 图表 1: 中断率 - 按运输模式 & 产品类别
# ============================================================

def plot_disruption_by_mode_and_product(df: pd.DataFrame):
    """分组柱状图：各运输模式 + 产品类别的中断率"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：按运输模式
    ax = axes[0]
    by_mode = (
        df.groupby("Transport_Mode")
        .agg(Disruption_Rate=("Disruption_Occurred", "mean"))
        .assign(Disruption_Rate=lambda x: x["Disruption_Rate"] * 100)
        .sort_values("Disruption_Rate", ascending=False)
    )
    bars = ax.bar(by_mode.index, by_mode["Disruption_Rate"], color=CAT_PALETTE[:len(by_mode)], edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, by_mode["Disruption_Rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_title("Disruption Rate by Transport Mode", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Disruption Rate (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, by_mode["Disruption_Rate"].max() * 1.2)
    ax.tick_params(axis="x", rotation=15)

    # 右图：按产品类别
    ax = axes[1]
    by_product = (
        df.groupby("Product_Category")
        .agg(Disruption_Rate=("Disruption_Occurred", "mean"))
        .assign(Disruption_Rate=lambda x: x["Disruption_Rate"] * 100)
        .sort_values("Disruption_Rate", ascending=False)
    )
    bars = ax.bar(by_product.index, by_product["Disruption_Rate"], color=CAT_PALETTE[:len(by_product)], edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, by_product["Disruption_Rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{val:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_title("Disruption Rate by Product Category", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Disruption Rate (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, by_product["Disruption_Rate"].max() * 1.2)
    ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Supply Chain Disruption Rate Analysis", fontsize=16, fontweight="bold", y=1.03)
    plt.tight_layout()
    _save_and_show(fig, "01_disruption_by_mode_product.png")
    return fig


# ============================================================
# 图表 2: 时效分布 - 箱线图
# ============================================================

def plot_lead_time_distribution(df: pd.DataFrame):
    """箱线图：各运输模式的时效分布"""
    fig, ax = plt.subplots(figsize=(12, 6))

    order = df.groupby("Transport_Mode")["Lead_Time_Days"].median().sort_values().index.tolist()
    palette = dict(zip(order, CAT_PALETTE[:len(order)]))

    # 过滤极端值以便可视化（保留99%分位以内）
    q99 = df["Lead_Time_Days"].quantile(0.99)
    df_plot = df[df["Lead_Time_Days"] <= q99]

    sns.boxplot(
        data=df_plot,
        x="Transport_Mode",
        y="Lead_Time_Days",
        order=order,
        hue="Transport_Mode",
        palette=palette,
        width=0.55,
        linewidth=1.2,
        fliersize=3,
        legend=False,
        ax=ax,
    )

    # 添加均值标注
    means = df_plot.groupby("Transport_Mode")["Lead_Time_Days"].mean()
    for i, mode in enumerate(order):
        ax.annotate(
            f"mean={means[mode]:.1f}d",
            xy=(i, means[mode]),
            fontsize=9,
            fontweight="bold",
            color="darkred",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    ax.set_title("Lead Time Distribution by Transport Mode", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Lead Time (Days)")
    ax.set_xlabel("")
    ax.axhline(y=df["Lead_Time_Days"].median(), color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax.text(len(order) - 0.5, df["Lead_Time_Days"].median() + 1, f"Overall Median: {df['Lead_Time_Days'].median():.1f}d",
            fontsize=9, color="red", ha="right")

    plt.tight_layout()
    _save_and_show(fig, "02_lead_time_distribution.png")
    return fig


# ============================================================
# 图表 3: 月度中断率趋势
# ============================================================

def plot_monthly_disruption_trend(df: pd.DataFrame):
    """折线图 + 柱状图：月度中断率趋势（双Y轴）"""
    monthly = (
        df.groupby(["Year", "Month"])
        .agg(Total=("Disruption_Occurred", "count"), Rate=("Disruption_Occurred", "mean"))
        .reset_index()
    )
    monthly["Date_Label"] = monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str).str.zfill(2)
    monthly["Rate"] = monthly["Rate"] * 100

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 柱状图 - 货运量
    bars = ax1.bar(
        monthly["Date_Label"],
        monthly["Total"],
        color="#2E86AB",
        alpha=0.3,
        label="Total Shipments",
    )
    ax1.set_ylabel("Number of Shipments", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)

    # 折线图 - 中断率（右Y轴）
    ax2 = ax1.twinx()
    ax2.plot(
        monthly["Date_Label"],
        monthly["Rate"],
        color="#A23B72",
        linewidth=2.5,
        marker="o",
        markersize=6,
        label="Disruption Rate",
    )
    # 添加趋势线
    x_range = np.arange(len(monthly))
    z = np.polyfit(x_range, monthly["Rate"], 1)
    p = np.poly1d(z)
    ax2.plot(monthly["Date_Label"], p(x_range), color="#A23B72", linestyle="--", alpha=0.4, linewidth=2, label="Trend")
    ax2.set_ylabel("Disruption Rate (%)", fontsize=12)
    ax2.set_ylim(0, monthly["Rate"].max() * 1.3)

    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    ax1.set_title("Monthly Shipment Volume & Disruption Rate Trend (2024-2026)", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    _save_and_show(fig, "03_monthly_disruption_trend.png")
    return fig


# ============================================================
# 图表 4: 地缘政治风险 vs 中断率
# ============================================================

def plot_risk_vs_disruption(df: pd.DataFrame):
    """柱状图 + 折线：不同风险等级下的中断率和时效"""
    fig, ax = plt.subplots(figsize=(10, 6))

    risk_levels = df.groupby("Risk_Level", observed=True).agg(
        Disruption_Rate=("Disruption_Occurred", "mean"),
        Avg_Lead_Time=("Lead_Time_Days", "mean"),
        Count=("Disruption_Occurred", "count"),
    )
    risk_levels["Disruption_Rate"] = risk_levels["Disruption_Rate"] * 100
    risk_order = ["Low", "Medium", "High"]
    risk_levels = risk_levels.reindex(risk_order)

    bars = ax.bar(
        risk_levels.index,
        risk_levels["Disruption_Rate"],
        color=["#3A7D44", "#F18F01", "#D62828"],
        edgecolor="white",
        linewidth=1.2,
        width=0.5,
    )
    for bar, val, count in zip(bars, risk_levels["Disruption_Rate"], risk_levels["Count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.2f}%\n(n={count:,})", ha="center", va="bottom", fontweight="bold", fontsize=11)

    # 时效折线
    ax2 = ax.twinx()
    ax2.plot(risk_levels.index, risk_levels["Avg_Lead_Time"], color="#2E86AB",
             linewidth=3, marker="D", markersize=10, label="Avg Lead Time")
    ax2.set_ylabel("Average Lead Time (Days)", fontsize=12, color="#2E86AB")
    ax2.tick_params(axis="y", colors="#2E86AB")
    for i, (idx, row) in enumerate(risk_levels.iterrows()):
        ax2.annotate(f"{row['Avg_Lead_Time']:.1f} days", xy=(i, row["Avg_Lead_Time"]),
                     fontsize=10, color="#2E86AB", ha="center", va="bottom",
                     fontweight="bold")

    ax.set_title("Impact of Geopolitical Risk on Disruption Rate & Lead Time", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Disruption Rate (%)", fontsize=12)
    ax.set_xlabel("Geopolitical Risk Level")
    ax.set_ylim(0, risk_levels["Disruption_Rate"].max() * 1.35)

    plt.tight_layout()
    _save_and_show(fig, "04_risk_vs_disruption.png")
    return fig


# ============================================================
# 图表 5: 天气影响
# ============================================================

def plot_weather_impact(df: pd.DataFrame):
    """水平柱状图：各天气条件下的中断率"""
    weather_stats = df.groupby("Weather_Condition").agg(
        Disruption_Rate=("Disruption_Occurred", "mean"),
        Count=("Disruption_Occurred", "count"),
    )
    weather_stats["Disruption_Rate"] = weather_stats["Disruption_Rate"] * 100
    weather_stats = weather_stats.sort_values("Disruption_Rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 用颜色深浅表示中断率高低
    norm = plt.Normalize(weather_stats["Disruption_Rate"].min(), weather_stats["Disruption_Rate"].max())
    colors = plt.cm.RdYlGn_r(norm(weather_stats["Disruption_Rate"]))

    bars = ax.barh(weather_stats.index, weather_stats["Disruption_Rate"], color=colors, edgecolor="gray", linewidth=0.8)
    for bar, val, count in zip(bars, weather_stats["Disruption_Rate"], weather_stats["Count"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%  (n={count:,})", va="center", fontsize=10, fontweight="bold")

    ax.set_title("Disruption Rate by Weather Condition", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Disruption Rate (%)")
    ax.set_xlim(0, weather_stats["Disruption_Rate"].max() * 1.25)

    plt.tight_layout()
    _save_and_show(fig, "05_weather_impact.png")
    return fig


# ============================================================
# 图表 6: 相关系数热力图
# ============================================================

def plot_correlation_heatmap(df: pd.DataFrame):
    """热力图：数值特征之间的相关性"""
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
    labels = [
        "Distance (km)",
        "Weight (MT)",
        "Fuel Price Index",
        "Geopolitical Risk",
        "Carrier Reliability",
        "Lead Time (Days)",
        "Speed (km/day)",
        "Ton-KM (k)",
        "Disruption",
    ]

    corr_matrix = df[num_cols].corr()

    # 遮罩上三角
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=DIVERGING_PALETTE,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=1.5,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        ax=ax,
    )
    ax.set_title("Correlation Matrix of Logistics Performance Factors", fontsize=14, fontweight="bold", pad=20)
    ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    _save_and_show(fig, "06_correlation_heatmap.png")
    return fig


# ============================================================
# 图表 7: 路线绩效 - 气泡图
# ============================================================

def plot_route_performance(df: pd.DataFrame, top_n: int = 15):
    """散点气泡图：Top N 路线 - X=中断率, Y=平均时效, 气泡大小=货运量"""
    route_stats = df.groupby("Route").agg(
        Shipments=("Shipment_ID", "count"),
        Disruption_Rate=("Disruption_Occurred", "mean"),
        Avg_Lead_Time=("Lead_Time_Days", "mean"),
    ).query("Shipments >= 20")
    route_stats["Disruption_Rate"] = route_stats["Disruption_Rate"] * 100
    top_routes = route_stats.nlargest(top_n, "Shipments")

    fig, ax = plt.subplots(figsize=(14, 8))

    scatter = ax.scatter(
        top_routes["Disruption_Rate"],
        top_routes["Avg_Lead_Time"],
        s=top_routes["Shipments"] * 0.8,
        c=top_routes["Disruption_Rate"],
        cmap="RdYlGn_r",
        alpha=0.7,
        edgecolors="gray",
        linewidth=0.5,
        zorder=5,
    )

    # 标注路线名
    for route, row in top_routes.iterrows():
        parts = route.split(" -> ")
        if len(parts) == 2:
            short_label = f"{parts[0][:3]}->{parts[1][:3]}"
        else:
            short_label = route[:15]
        ax.annotate(
            short_label,
            (row["Disruption_Rate"], row["Avg_Lead_Time"]),
            fontsize=7,
            ha="center",
            va="bottom",
            alpha=0.8,
            xytext=(0, 8),
            textcoords="offset points",
        )

    ax.axhline(y=top_routes["Avg_Lead_Time"].median(), color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=top_routes["Disruption_Rate"].median(), color="gray", linestyle="--", alpha=0.5, linewidth=1)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Disruption Rate (%)", fontsize=10)

    ax.set_title(f"Route Performance Map: Top {top_n} Routes by Volume", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Disruption Rate (%)", fontsize=12)
    ax.set_ylabel("Average Lead Time (Days)", fontsize=12)

    ax.text(0.98, 0.02, "Bubble size = Shipment volume\nDashed lines = median values",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    _save_and_show(fig, "07_route_performance_bubble.png")
    return fig


# ============================================================
# 图表 8: 特征重要性
# ============================================================

def plot_feature_importance(feature_importance: pd.DataFrame):
    """
    水平柱状图：特征重要性。
    自动适配两种格式：
      - Random Forest: 'Importance' 列（0-1，全正）
      - Logistic Regression: 'Coefficient' 列（可正可负）
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    fi = feature_importance.copy()

    # 检测格式
    if "Importance" in fi.columns:
        fi = fi.sort_values("Importance", ascending=True)
        values = fi["Importance"]
        colors = ["#2E86AB"] * len(fi)
        xlabel = "Feature Importance (Random Forest)"
        title = "Feature Importance for Disruption Prediction (Random Forest)"
        fmt = "{:.4f}"
    elif "Coefficient" in fi.columns:
        fi["Abs_Val"] = fi["Coefficient"].abs()
        fi = fi.sort_values("Abs_Val", ascending=True)
        values = fi["Abs_Val"]
        colors = ["#D62828" if c > 0 else "#2E86AB" for c in fi["Coefficient"]]
        xlabel = "Absolute Coefficient Magnitude"
        title = "Feature Importance for Disruption Prediction (Logistic Regression)"
        fmt = "{:+.3f}"
    else:
        raise KeyError("feature_importance must have 'Importance' or 'Coefficient' column")

    bars = ax.barh(fi["Feature"], values, color=colors, edgecolor="white", linewidth=1, height=0.6)

    for bar, (_, row) in zip(bars, fi.iterrows()):
        if "Importance" in fi.columns:
            label = f"{row['Importance']:.4f}"
        else:
            sign = "+" if row["Coefficient"] > 0 else ""
            label = f"{sign}{row['Coefficient']:.3f}"
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=11, fontweight="bold")

    if "Coefficient" in fi.columns:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#D62828", label="Increases Disruption Risk"),
            Patch(facecolor="#2E86AB", label="Reduces Disruption Risk"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_xlim(0, values.max() * 1.3)

    plt.tight_layout()
    _save_and_show(fig, "08_feature_importance.png")


# ============================================================
# 附加图表: 总体仪表盘
# ============================================================

def plot_dashboard(df: pd.DataFrame):
    """综合仪表盘：2x3 子图布局"""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Global Supply Chain Risk & Logistics Dashboard (2024-2026)", fontsize=16, fontweight="bold", y=0.98)

    # ---- 1. 中断率按运输模式 (左上) ----
    ax1 = fig.add_subplot(2, 3, 1)
    by_mode = df.groupby("Transport_Mode")["Disruption_Occurred"].mean() * 100
    by_mode.sort_values(ascending=False).plot(kind="barh", ax=ax1, color=CAT_PALETTE[:len(by_mode)], edgecolor="white")
    ax1.set_title("Disruption Rate by Mode")
    ax1.set_xlabel("%")
    for i, v in enumerate(by_mode.sort_values(ascending=False)):
        ax1.text(v + 0.5, i, f"{v:.1f}%", va="center", fontweight="bold", fontsize=9)

    # ---- 2. 时效分布 (中上) ----
    ax2 = fig.add_subplot(2, 3, 2)
    order = df.groupby("Transport_Mode")["Lead_Time_Days"].median().sort_values().index
    q99 = df["Lead_Time_Days"].quantile(0.98)
    sns.violinplot(data=df[df["Lead_Time_Days"] <= q99], x="Transport_Mode", y="Lead_Time_Days",
                   order=order, hue="Transport_Mode", palette=CAT_PALETTE[:len(order)], inner="quartile",
                   legend=False, ax=ax2, linewidth=0.8)
    ax2.set_title("Lead Time Distribution")
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=20)

    # ---- 3. 月度趋势 (右上) ----
    ax3 = fig.add_subplot(2, 3, 3)
    monthly = df.groupby(["Year", "Month"])["Disruption_Occurred"].mean() * 100
    monthly.index = [f"{y}-{m:02d}" for y, m in monthly.index]
    ax3.fill_between(range(len(monthly)), monthly.values, alpha=0.3, color="#2E86AB")
    ax3.plot(range(len(monthly)), monthly.values, color="#A23B72", linewidth=2, marker="o", markersize=4)
    ax3.set_xticks(range(0, len(monthly), 3))
    ax3.set_xticklabels(monthly.index[::3], rotation=45, fontsize=8)
    ax3.set_title("Monthly Disruption Rate Trend")
    ax3.set_ylabel("%")

    # ---- 4. 风险等级影响 (左下) ----
    ax4 = fig.add_subplot(2, 3, 4)
    risk = df.groupby("Risk_Level", observed=True).agg(Rate=("Disruption_Occurred", "mean"), LT=("Lead_Time_Days", "mean"))
    risk["Rate"] = risk["Rate"] * 100
    risk = risk.reindex(["Low", "Medium", "High"])
    ax4.bar(risk.index, risk["Rate"], color=["#3A7D44", "#F18F01", "#D62828"], edgecolor="white")
    ax4_twin = ax4.twinx()
    ax4_twin.plot(risk.index, risk["LT"], color="#2E86AB", linewidth=2.5, marker="s", markersize=8)
    ax4.set_title("Geopolitical Risk Impact")
    ax4.set_ylabel("Disruption Rate (%)")
    ax4_twin.set_ylabel("Avg Lead Time (Days)", color="#2E86AB")

    # ---- 5. 天气影响 (中下) ----
    ax5 = fig.add_subplot(2, 3, 5)
    weather = df.groupby("Weather_Condition")["Disruption_Occurred"].agg(["mean", "count"])
    weather["mean"] = weather["mean"] * 100
    weather = weather.sort_values("mean")
    ax5.barh(weather.index, weather["mean"], color=plt.cm.RdYlGn_r(
        np.linspace(0.2, 0.8, len(weather))), edgecolor="gray", linewidth=0.5)
    ax5.set_title("Disruption Rate by Weather")
    ax5.set_xlabel("%")

    # ---- 6. 货运量分布 (右下) ----
    ax6 = fig.add_subplot(2, 3, 6)
    top_routes = df["Route"].value_counts().head(10)
    ax6.barh(range(len(top_routes)), top_routes.values, color=CAT_PALETTE, edgecolor="white")
    ax6.set_yticks(range(len(top_routes)))
    short_labels = [" -> ".join([p[:4] for p in r.split(" -> ")]) for r in top_routes.index]
    ax6.set_yticklabels(short_labels, fontsize=8)
    ax6.set_title("Top 10 Routes by Volume")
    ax6.set_xlabel("Shipments")
    ax6.invert_yaxis()

    plt.tight_layout()
    _save_and_show(fig, "00_dashboard.png")
    return fig


# ============================================================
# 图表 9: 蒙特卡洛 -- 中断率 Bootstrap 分布
# ============================================================

def plot_mc_disruption_distribution(mc_results: dict):
    """直方图 + 置信区间：Bootstrap 中断率抽样分布"""
    if mc_results is None or "disruption_rate" not in mc_results:
        return None

    dr = mc_results["disruption_rate"]
    samples = dr["samples"]
    observed = dr["observed"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(samples, bins=50, color="#2E86AB", alpha=0.7, edgecolor="white", density=True)

    ax.axvline(observed, color="#D62828", linewidth=3, linestyle="-", label=f"Observed: {observed:.3f}")

    ax.axvline(dr["ci_95_lower"], color="#F18F01", linewidth=2, linestyle="--",
               label=f"95% CI: [{dr['ci_95_lower']:.3f}, {dr['ci_95_upper']:.3f}]")
    ax.axvline(dr["ci_95_upper"], color="#F18F01", linewidth=2, linestyle="--")

    ax.set_title("Monte Carlo Bootstrap: Disruption Rate Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Simulated Disruption Rate")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)

    plt.tight_layout()
    _save_and_show(fig, "09_mc_disruption_distribution.png")
    return fig


# ============================================================
# 图表 10: 蒙特卡洛 -- 压力测试对比
# ============================================================

def plot_mc_stress_test(mc_results: dict):
    """并排直方图：Baseline vs 压力情景下的中断率分布"""
    if mc_results is None:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Monte Carlo Stress Test: Disruption Rate Distributions", fontsize=14, fontweight="bold")

    scenarios = [
        (mc_results.get("stress_geo"), "Geopolitical Risk +2.0"),
        (mc_results.get("stress_weather"), "Severe Weather x3"),
    ]

    for col, (result, title) in enumerate(scenarios):
        if result is None:
            continue

        ax = axes[0, col]
        ax.hist(result["baseline_samples"], bins=40, color="#2E86AB", alpha=0.5,
                edgecolor="white", label="Baseline", density=True)
        ax.hist(result["stressed_samples"], bins=40, color="#D62828", alpha=0.5,
                edgecolor="white", label="Stressed", density=True)
        ax.axvline(np.mean(result["baseline_samples"]), color="#2E86AB", linewidth=2, linestyle="-")
        ax.axvline(np.mean(result["stressed_samples"]), color="#D62828", linewidth=2, linestyle="-")
        ax.set_title(f"{title}\nDelta = {result['pct_change']:+.1f}%", fontsize=12, fontweight="bold")
        ax.set_xlabel("Disruption Rate")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        ax = axes[1, col]
        delta = result["stressed_samples"] - result["baseline_samples"]
        ax.hist(delta, bins=40, color="#6A4C93", alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(delta), color="#D62828", linewidth=2, linestyle="-",
                   label=f"Mean Delta: {np.mean(delta):.4f}")
        ax.set_title(f"{title} -- Change Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Delta Disruption Rate (Stressed - Baseline)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save_and_show(fig, "10_mc_stress_test.png")
    return fig


# ============================================================
# 图表 11: 模型对比
# ============================================================

def plot_model_comparison(model_comparison):
    """分组柱状图：多模型多指标对比"""
    if model_comparison is None:
        return None

    fig, ax = plt.subplots(figsize=(12, 7))

    df_plot = model_comparison.reset_index().rename(columns={"index": "model"})
    if "Model" in df_plot.columns:
        df_plot = df_plot.rename(columns={"Model": "model"})

    metrics = ["test_accuracy", "precision", "recall", "f1", "roc_auc"]
    available = [m for m in metrics if m in df_plot.columns]
    if not available:
        available = ["test_accuracy", "precision", "recall", "f1"]

    x = np.arange(len(df_plot))
    width = 0.15
    colors = ["#2E86AB", "#A23B72", "#3A7D44", "#F18F01", "#6A4C93"]

    for i, metric in enumerate(available):
        offset = (i - len(available) / 2 + 0.5) * width
        values = df_plot[metric].values
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["model"], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Supply Chain Disruption Prediction", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.15)
    # 动态计算 baseline
    baseline_val = model_comparison.iloc[0]["test_accuracy"] if "test_accuracy" in model_comparison.columns else 0.613
    ax.axhline(y=baseline_val, color="gray", linestyle="--", alpha=0.5, linewidth=1,
               label=f"Naive Baseline ({baseline_val:.1%})")

    plt.tight_layout()
    _save_and_show(fig, "11_model_comparison.png")
    return fig


# ============================================================
# 图表 12: 路线 VaR
# ============================================================

def plot_route_var(mc_results: dict):
    """水平柱状图：各路线 95% VaR 时效"""
    if mc_results is None or "route_var" not in mc_results:
        return None

    rv = mc_results["route_var"].sort_values("VaR_95", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    y_pos = range(len(rv))
    ax.barh(y_pos, rv["Observed_Mean_LT"], height=0.4, color="#2E86AB",
            alpha=0.6, label="Observed Mean LT", edgecolor="white")
    ax.barh(y_pos, rv["VaR_95"], height=0.4, color="#D62828", alpha=0.6,
            label="VaR 95% LT", edgecolor="white")

    for i, (_, row) in enumerate(rv.iterrows()):
        ax.text(row["VaR_95"] + 0.5, i, f"{row['VaR_95']:.1f}d", va="center", fontsize=8, fontweight="bold")

    short_labels = [" -> ".join([p[:4] for p in r.split(" -> ")]) for r in rv["Route"]]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_labels, fontsize=8)

    ax.set_title("Route-Level Lead Time Value-at-Risk (VaR 95%)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Lead Time (Days)")
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    _save_and_show(fig, "12_route_var.png")
    return fig


# ============================================================
# 图表 13: 总成本模拟分布
# ============================================================

def plot_cost_distribution(mc_results: dict):
    """直方图：总风险成本模拟分布 + VaR 标记"""
    if mc_results is None or "total_cost" not in mc_results:
        return None

    tc = mc_results["total_cost"]
    samples = tc["cost_samples"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(samples / 1e6, bins=50, color="#2E86AB", alpha=0.7, edgecolor="white")

    ax.axvline(tc["VaR_95_cost"] / 1e6, color="#F18F01", linewidth=2.5, linestyle="--",
               label=f"VaR 95%: ${tc['VaR_95_cost']/1e6:.1f}M")
    ax.axvline(tc["VaR_99_cost"] / 1e6, color="#D62828", linewidth=2.5, linestyle="--",
               label=f"VaR 99%: ${tc['VaR_99_cost']/1e6:.1f}M")
    ax.axvline(tc["mean_cost"] / 1e6, color="#3A7D44", linewidth=2.5, linestyle="-",
               label=f"Mean: ${tc['mean_cost']/1e6:.1f}M")

    ax.set_title("Monte Carlo Simulation: Total Supply Chain Risk Cost", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Total Cost (Million USD)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=10)

    plt.tight_layout()
    _save_and_show(fig, "13_cost_distribution.png")
    return fig


# ============================================================
# 图表 14: 统计检验结果
# ============================================================

def plot_statistical_results(test_results: dict):
    """水平柱状图：各检验的效应量"""
    if test_results is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    items = []

    def _collect(name, result):
        if result is None:
            return
        if "effect_size_cohens_d" in result:
            items.append((name, abs(result["effect_size_cohens_d"]), "Cohen's d"))
        elif "effect_size_eta_squared" in result:
            items.append((name, result["effect_size_eta_squared"], "eta^2"))
        elif "cramers_v" in result:
            items.append((name, result["cramers_v"], "Cramer's V"))

    _collect("Lead Time ~ Disruption", test_results.get("ttest_lt_disruption"))
    _collect("Lead Time ~ Mode", test_results.get("anova_lt_mode"))
    _collect("Disruption ~ Weather", test_results.get("chi2_weather"))
    _collect("Disruption ~ Risk Level", test_results.get("chi2_risk"))

    chi2_ind = test_results.get("chi2_disruption_independence", {})
    for key in ["Transport_Mode", "Product_Category"]:
        if key in chi2_ind:
            _collect(f"Disruption ~ {key}", chi2_ind[key])

    if not items:
        plt.close(fig)
        return None

    items.sort(key=lambda x: x[1], reverse=True)
    names = [i[0] for i in items]
    values = [i[1] for i in items]
    es_types = [i[2] for i in items]

    colors = []
    for v in values:
        if v > 0.5:
            colors.append("#D62828")
        elif v > 0.3:
            colors.append("#F18F01")
        else:
            colors.append("#2E86AB")

    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.5)
    for bar, val, es_t in zip(bars, values, es_types):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f} ({es_t})", va="center", fontsize=10, fontweight="bold")

    for threshold, label, ls in [(0.1, "Small", "--"), (0.3, "Medium", "-."), (0.5, "Large", ":")]:
        ax.axvline(threshold, color="gray", linestyle=ls, alpha=0.4)
        ax.text(threshold, len(items) - 0.3, label, fontsize=8, color="gray", ha="center")

    ax.set_title("Statistical Test Effect Sizes Summary", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Effect Size")
    ax.set_xlim(0, max(values) * 1.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#D62828", label="Large"),
        Patch(facecolor="#F18F01", label="Medium"),
        Patch(facecolor="#2E86AB", label="Small"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    _save_and_show(fig, "14_statistical_effects.png")
    return fig


# ============================================================
# 运行所有可视化
# ============================================================

def run_all_visualizations(
    df: pd.DataFrame,
    feature_importance: pd.DataFrame = None,
    model_comparison=None,
    mc_results: dict = None,
    test_results: dict = None,
):
    """生成所有图表（包括可选的高级分析图表）"""
    print("\n" + "=" * 55)
    print("[生成可视化图表]")
    print("=" * 55)

    # ---- 基础图表 ----
    plot_dashboard(df)
    plot_disruption_by_mode_and_product(df)
    plot_lead_time_distribution(df)
    plot_monthly_disruption_trend(df)
    plot_risk_vs_disruption(df)
    plot_weather_impact(df)
    plot_correlation_heatmap(df)
    plot_route_performance(df)

    if feature_importance is not None:
        plot_feature_importance(feature_importance)

    # ---- 高级图表 ----
    if model_comparison is not None:
        print("\n  [高级] 生成模型对比图表...")
        plot_model_comparison(model_comparison)

    if mc_results is not None:
        print("  [高级] 生成蒙特卡洛模拟图表...")
        plot_mc_disruption_distribution(mc_results)
        plot_mc_stress_test(mc_results)
        plot_route_var(mc_results)
        plot_cost_distribution(mc_results)

    if test_results is not None:
        print("  [高级] 生成统计检验图表...")
        plot_statistical_results(test_results)

    print(f"\n[OK] 所有图表已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import prepare_data
    from analysis import run_all_analysis

    df = prepare_data()
    results = run_all_analysis(df)
    run_all_visualizations(
        df,
        feature_importance=results["feature_importance"],
        model_comparison=results["model_comparison"],
    )
