"""
全球供应链风险与物流绩效分析 -- 主程序
============================================

项目结构:
  logistics-analysis/
  |-- data/          <-- 原始数据
  |-- src/           <-- Python 模块
  |   |-- data_loader.py   数据加载与预处理
  |   |-- analysis.py      多维度分析
  |   |-- statistical_tests.py  统计假设检验
  |   |-- monte_carlo.py        蒙特卡洛模拟
  |   |-- visualization.py 可视化
  |-- output/        <-- 生成的图表
  |-- tests/         <-- 单元测试
  |-- main.py        <-- 当前文件

运行:
  python main.py
"""

import sys
import os
from pathlib import Path

# 修复 Windows 终端编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


from data_loader import prepare_data
from analysis import run_all_analysis
from statistical_tests import run_statistical_tests
from monte_carlo import MonteCarloEngine, run_monte_carlo_analysis
from visualization import run_all_visualizations


def main():
    print("=" * 60)
    print("  全球供应链风险与物流绩效分析系统")
    print("  Global Supply Chain Risk & Logistics Performance")
    print("  Data: 2024-2026 | 5,000 International Shipments")
    print("=" * 60)

    # ---- Step 1: 数据准备 ----
    print("\n[Step 1/5] 数据准备")
    print("-" * 40)
    df = prepare_data()

    # ---- Step 2: 统计检验 ----
    print("\n[Step 2/5] 统计假设检验")
    print("-" * 40)
    test_results = run_statistical_tests(df)

    # ---- Step 3: 分析 & 建模 ----
    print("\n[Step 3/5] 多维度分析与预测建模")
    print("-" * 40)
    analysis_results = run_all_analysis(df, skip_basic_prints=True)

    # ---- Step 4: 蒙特卡洛模拟 ----
    print("\n[Step 4/5] 蒙特卡洛模拟")
    print("-" * 40)
    engine = MonteCarloEngine(df, random_seed=42)
    mc_results = run_monte_carlo_analysis(engine, n_samples=10000)

    # ---- Step 5: 可视化 ----
    print("\n[Step 5/5] 生成可视化图表")
    print("-" * 40)
    run_all_visualizations(
        df=df,
        feature_importance=analysis_results["feature_importance"],
        model_comparison=analysis_results.get("model_comparison"),
        mc_results=mc_results,
        test_results=test_results,
    )

    # ---- 完成 ----
    print("\n" + "=" * 60)
    print("  全部分析完成！")
    print(f"  图表输出目录: {Path(__file__).parent / 'output'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
