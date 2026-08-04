# 全球供应链风险与物流绩效分析

## Global Supply Chain Risk & Logistics Performance Analysis

> 数据分析 / 供应链分析 作品集项目 -- 5,000 条国际货运记录 -- 从描述性分析到蒙特卡洛模拟的完整闭环

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-1.5+-green.svg)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.10+-blueviolet.svg)](https://scipy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 项目概述

本项目对 2024-2026 年全球供应链的 5,000 条国际货运记录进行端到端分析，覆盖 描述性分析 -> 统计推断 -> 预测建模 -> 蒙特卡洛模拟 四个层次。

### 分析框架

| 层次 | 维度 | 核心问题 | 方法 |
|------|------|---------|------|
| 描述分析 | 中断 / 时效 / 风险 | 哪些因素与高中断率相关？ | 分组聚合、交叉分析、多维度对比 |
| 统计推断 | 假设检验 | 组间差异是否统计显著？ | t 检验、ANOVA、卡方检验 + 效应量 |
| 预测建模 | 多模型对比 | 能否预测中断？哪个模型最优？ | Baseline / 逻辑回归 / 随机森林 |
| 蒙特卡洛 | 风险量化 | 中断率的置信区间？压力情景？VaR？ | Bootstrap、场景模拟、成本分布 |

---

## 快速开始

### 环境要求

- Python 3.10+
- 依赖库：pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, openpyxl

### 安装与运行

```bash
# 1. 进入项目目录
cd logistics-analysis

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行主程序（完整分析流程）
python main.py

# 4. 或运行 Jupyter Notebook（交互式）
jupyter notebook logistics_analysis.ipynb

# 5. 或单独运行各模块
python src/analysis.py          # 描述分析 + 多模型对比
python src/statistical_tests.py # 统计假设检验
python src/monte_carlo.py       # 蒙特卡洛模拟
python src/visualization.py     # 生成全部图表
```

---

## 项目结构

```
logistics-analysis/
|-- data/
|   |-- global_supply_chain_risk_2026.csv   # Kaggle 数据集 (5,000条)
|-- src/
|   |-- data_loader.py         # 数据加载、清洗、特征工程、质量校验
|   |-- analysis.py            # 6 大分析 + 多模型对比 (Baseline/LR/RF)
|   |-- statistical_tests.py   # 5 项统计检验 + Bonferroni 校正 + Tukey HSD
|   |-- monte_carlo.py         # 蒙特卡洛模拟引擎 (Bootstrap/压力测试/VaR/ROI)
|   |-- visualization.py       # 15 张专业可视化图表
|-- tests/
|   |-- test_data_loader.py    # 数据加载单元测试
|   |-- test_statistical_tests.py  # 统计检验单元测试
|   |-- test_monte_carlo.py    # 蒙特卡洛单元测试
|-- models/                    # 训练模型持久化目录
|-- output/                    # 生成的图表 (PNG, 150 DPI)
|-- .github/workflows/ci.yml   # GitHub Actions CI 流水线
|-- config.yaml                # 全局配置文件
|-- pytest.ini                 # Pytest 配置
|-- .gitignore                 # Git 忽略规则
|-- logistics_analysis.ipynb   # Jupyter Notebook 交互入口
|-- main.py                    # 主程序入口
|-- requirements.txt           # Python 依赖
|-- README.md                  # 本文件
```

---

## 核心发现

### 1. 中断分析

- 整体中断率: 61.26%（3,063 / 5,000 条货运记录经历了中断）
- 运输模式差异: 四种模式中断率均在 61%-62%，差异不显著
- 产品类别差异: 纺织品中断率最高 (64.85%)，电子产品最低 (59.55%)
- 天气影响: 飓风条件下中断率 100%，暴风雨 79.5%，晴天仅 37.0%

### 2. 时效绩效

| 运输模式 | 平均时效 | 中位时效 | 标准差 |
|---------|---------|---------|-------|
| 空运 (Air) | 1.64 天 | 0.93 天 | 2.93 |
| 公路 (Road) | 16.45 天 | 9.30 天 | 23.39 |
| 铁路 (Rail) | 19.95 天 | 11.54 天 | 30.71 |
| 海运 (Sea) | 39.80 天 | 21.79 天 | 47.67 |

> ANOVA 检验: F 统计量显著（p < 0.001），eta^2 效应量大，运输模式可解释大部分时效差异。

### 3. 统计检验结论

| 检验 | p-value | 显著性 | 效应量 |
|------|---------|--------|--------|
| 中断 vs 正常 时效差异 (t-test) | p < 0.001 | SIG | Cohen's d 中等 |
| 运输模式间时效差异 (ANOVA) | p < 0.001 | SIG | eta^2 大 |
| 中断 x 天气条件 (chi^2) | p < 0.001 | SIG | Cramer's V 大 |
| 中断 x 风险等级 (chi^2) | p < 0.001 | SIG | Cramer's V 中等 |
| 中断 x 运输模式 (chi^2) | 不显著 | NS | Cramer's V 极小 |

### 4. 多模型对比

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|----------|-----------|--------|-----|---------|
| Baseline (Most Frequent) | 0.613 | -- | -- | -- | -- |
| Logistic Regression | 0.631 | 0.632 | 0.983 | 0.770 | 0.635 |
| Random Forest | 0.660 | 0.648 | 0.948 | 0.770 | 0.716 |

> 随机森林相对朴素基线提升 4.7pp，ROC-AUC = 0.716 表明模型捕获了真实信号。

Top 5 特征 (Random Forest):

1. 地缘政治风险 -- 最重要的中断驱动因素
2. 承运商可靠性 -- 选择高可靠性承运商可降低中断
3. Lead Time -- 时效越长越容易中断
4. 运输距离
5. 燃油价格指数

### 5. 蒙特卡洛模拟

| 情景 | 结果 |
|------|------|
| 全局中断率 95% CI | Bootstrap: [59.9%, 62.6%] |
| 地缘风险 +2.0 压力测试 | 中断率显著上升 |
| 恶劣天气 x3 压力测试 | 中断率显著上升 |
| 路线 VaR 95% | 部分路线时效风险极高 |
| 总成本 VaR 95% | 见成本模拟分布图 |

---

## 可视化图表清单

| # | 图表 | 文件 | 类型 |
|---|------|------|------|
| 0 | 综合仪表盘 (6合1) | `00_dashboard.png` | Dashboard |
| 1 | 中断率 x 运输模式/产品 | `01_disruption_by_mode_product.png` | 分组柱状图 |
| 2 | 时效分布 | `02_lead_time_distribution.png` | 箱线图 |
| 3 | 月度中断趋势 | `03_monthly_disruption_trend.png` | 折线+柱状图 |
| 4 | 地缘政治风险影响 | `04_risk_vs_disruption.png` | 柱状图+折线 |
| 5 | 天气条件影响 | `05_weather_impact.png` | 水平柱状图 |
| 6 | 相关系数热力图 | `06_correlation_heatmap.png` | 热力图 |
| 7 | 路线绩效气泡图 | `07_route_performance_bubble.png` | 散点气泡图 |
| 8 | 特征重要性 | `08_feature_importance.png` | 水平柱状图 |
| 9 | 蒙特卡洛中断率分布 | `09_mc_disruption_distribution.png` | Bootstrap 直方图 |
| 10 | 压力测试对比 | `10_mc_stress_test.png` | 多子图对比 |
| 11 | 多模型对比 | `11_model_comparison.png` | 分组柱状图 |
| 12 | 路线 VaR | `12_route_var.png` | 水平柱状图 |
| 13 | 总成本分布 | `13_cost_distribution.png` | 模拟直方图 |
| 14 | 统计效应量汇总 | `14_statistical_effects.png` | 水平柱状图 |

---

## 技术栈

| 层 | 工具 | 用途 |
|----|------|------|
| 数据处理 | pandas, numpy | ETL、分组聚合、特征工程 |
| 统计推断 | scipy.stats | t 检验、ANOVA、卡方检验、效应量 |
| 机器学习 | scikit-learn | 逻辑回归、随机森林、交叉验证、Dummy Baseline |
| 模拟 | numpy (自实现) | Bootstrap、压力测试、VaR、成本模拟 |
| 可视化 | matplotlib, seaborn | 15 张图表，150 DPI，统一配色 |

---

## 业务建议

1. 优先选择高可靠性承运商 -- 这是仅次于地缘政治的第二大可控因素
2. 建立天气预警机制 -- 飓风/暴风雨条件下中断率是晴天的 2-3 倍
3. 重点监控高风险路线 -- 参考路线 VaR 报告
4. 将预测模型嵌入运营系统 -- 随机森林可提前识别高风险货运
5. 差异化库存策略 -- 对纺织品等高中断率品类设置更高安全库存
6. 地缘风险对冲 -- 对高风险地区路线设置备选方案

---

## 数据来源

- 数据集: [Global Supply Chain Risk & Logistics (2024-2026)](https://www.kaggle.com/datasets/nudratabbas/global-supply-chain-risk-and-logistics-2024-2026)
- 来源: Kaggle（合成数据，用于教学/演示目的）
- 获取方式: 从 Kaggle 下载 `global_supply_chain_risk_2026.csv`，放置于 `data/` 目录下
- 规模: 5,000 条记录，14 个原始字段
- 时间范围: 2024-01 ~ 2025-12
- 覆盖: 8 个出发港、9 个目的港、4 种运输模式、5 类产品

---

## 作者

- 本项目为供应链/物流数据分析方向作品集项目
- 展示技能：Python 数据处理 / 统计推断 / 机器学习 / 蒙特卡洛模拟 / 数据可视化 / 业务洞察

---

## 局限性声明 (Limitations)

本项目为教学/演示用途的分析作品集，使用者需了解以下局限：

### 数据相关
- 合成数据: 数据来源于 Kaggle 合成数据集，所有统计量和结论描述的是数据生成器的特性，不代表真实全球供应链状况
- 样本量: 5,000 条记录对于 8x9 个港口组合 x 4 种运输模式 x 5 类产品的完整交叉来说是稀疏的
- 时间范围: 两年数据无法捕捉长周期趋势（如经济周期、贸易协定变更）

### 方法论相关
- 关联不等于因果: 所有因素分析为关联性估计。未观测混杂因素（如承运商规模、航线基础设施、海关效率）可能同时影响可靠性评分和中断率
- 逻辑回归限制: 逻辑回归对非线性关系捕捉有限，其准确率仅略高于朴素基准。随机森林表现更好（+4.7pp），但可解释性更差
- ROI 模拟: 基于模型预测的反事实推断，非实际实验数据。`simulate_reliability_improvement` 使用了 train/test 分离和 AUC 评估以保证诚实性
- Bonferroni 校正: 6 个假设检验已使用 Bonferroni 校正，但大样本下 p 值容易显著 -- 应结合效应量（Cohen's d / Cramer's V）解读

### 工程相关
- 本项目是一个分析管道，不是生产级工具。没有模型服务 API、监控、数据漂移检测
- 成本模拟中的美元金额基于假设参数（$5,000/次中断, $200/天超时），实际应用前需标定

---

## License

MIT License
