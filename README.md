# 选择者期权定价模型：真实数据、机器学习与极端情景压力测试

中金公司量化研究实习 · 第 8 周结项工程

本项目以摩根大通（JPM）2018—2024 年的日频数据为样本，复现并扩展 Rubinstein（1991）的选择者期权（Chooser Option）定价框架，构建了一条从数据清洗、蒙特卡洛基准、机器学习修正、可解释性分析到交互工具与压力测试的完整链路。

---

## 1. 项目解决什么问题

选择者期权允许持有者在 T1 时点决定这份合约最终是看涨还是看跌。Rubinstein 给出了无套利分解：

```
Chooser(S, K, T1, T2) = Call(S, K, T2) + Put(S, K·exp(-r·(T2-T1)), T1)
```

分解本身是精确的，但真实市场并不满足 BSM 的假设：波动率并非常数，价格会跳空，买卖存在价差。项目要回答三个问题：

1. BSM 在真实数据上偏离多少，偏离来自哪里？
2. 用机器学习学习这部分偏离，能修正到什么程度？
3. 当市场进入训练样本从未覆盖的极端区间，修正还可靠吗？

第 3 个问题是本项目的核心结论，也是第 8 周加练的内容。

---

## 2. 关键结果

| 阶段 | 指标 / 结论 |
|---|---|
| 蒙特卡洛基准 | 15.1180 美元（10 万路径，Heston + 5% 跳跃 + 5% 买卖价差） |
| BSM 波动率消融 | σ=20% 时 MAE 最小（0.5579 美元）；偏离到 10% / 30% 升至 4.91 / 6.23 美元 |
| 方案一：预测波动率 | 测试集 R² = −0.5418，方向可辨但精度不足 |
| 方案二：GBDT 残差修正 | MAE 0.1732 → 0.0196，RMSE 0.4770 → 0.1835，R² 0.9350 → 0.9904，MAPE 1.21% → 0.12% |
| SHAP | Vol_60、Vol_20、MA_Gap 是修正边际的前三大驱动 |
| 敏感性分析 | σ 从 20% → 50%，选择者价格 20.78 → 49.77 美元；r 从 5% → 9%，仅 20.78 → 22.24 美元 |
| 黑天鹅测试 | 修正方向 100% 正确，量级仅 78% 落在 ±20% 内；波动率梯上修正恒定 0.0015 美元 |

**核心结论**：GBDT 在正常与中度压力市场能提供稳定的残差修正，但在极端黑天鹅下会结构性饱和——树模型只能内插、不能外推。深跌情景下未约束的修正达到 BSM 价格的 83%，因此工具必须配合 ±20% 护栏与越界警告，而不是把模型输出直接当成价格。

---

## 3. 目录结构

```
WelcomeScreen/
├── pricing_core.py               # 定价与模型核心（BSM、Rubinstein、特征、护栏、OOD）
├── Week8_App.py                  # 最终交付的 Streamlit 交互工具
├── Week8_BlackSwan_Test.py       # 第 8 周极端黑天鹅测试（5 组情景 / 18 个案例）
├── make_report_figures.py        # 结项报告配图生成
├── make_banner.py                # PPT 深色概览图生成
│
├── Week 2.py                     # 第 1–2 周：数据清洗与特征工程
├── visualize.py                  # 数据可视化
├── config.py                     # 参数配置
├── Week3_bsm.py                  # 第 3 周：BSM 与选择者期权首版实现
├── Week4_MonteCarlo_Sim.py       # 第 4 周：带摩擦的蒙特卡洛基准
├── Week4_Ablation_Study.py       # 第 4 周：波动率消融
├── Week4_Error_Metrics.py        # 误差指标工具
├── Week5_ML_Approach1.py         # 第 5 周：方案一（预测波动率）
├── Week5_ML_Approach2.py         # 第 5 周：方案二（端到端 / 残差）
├── Week5_Evaluation_Metrics.py   # 第 5 周：统一评估
├── Week6_Tuning_and_SHAP.py      # 第 6 周：调参与 SHAP
├── Week6_Tuning_Comparison.py    # 第 6 周：三种调参方法对比
├── Week6_SHAP_Visuals.py         # 第 6 周：SHAP 出图
├── Week7_Stress_Test.py          # 第 7 周：波动率与利率敏感性
│
├── jpm_stock.csv                 # JPM 日频行情
├── vix.csv                       # VIX 指数
├── treasury.csv                  # 美国 10 年期国债收益率
├── features.csv / features.parquet  # 清洗后的特征面板
├── Week6_Best_GBDT_Model.pkl     # 最终 GBDT 模型
│
├── overview.png / overview_banner.png
├── correlation.png
├── Week4_Ablation_Table.csv
├── Week5_Approach1_Results.csv / Week5_Approach2_Results.csv
├── Week6_SHAP_Summary.png / Week6_SHAP_Summary_V2.png
├── Week6_Tuning_Convergence.png
├── Week7_Stress_Test.png
├── Week8_BlackSwan_Results.csv
├── Week8_BlackSwan_Analysis.png
├── Week8_Correction_Saturation.png
├── app_shot_normal.png / app_shot_blackswan.png
│
├── requirements.txt
├── README.md
└── _archive/                     # 旧版工具、旧图与运行日志（不参与运行）
```

---

## 4. 环境与运行

### 4.1 安装

建议 Python 3.10+。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 4.2 启动交互工具

```bash
streamlit run Week8_App.py
```

浏览器打开后可以看到四个标签页：

- **Pricing**：输入 S、K、T1、T2、r、σ，得到 BSM 价格、GBDT 修正后价格、修正边际及其占比。
- **Sensitivity**：查看价格随波动率、标的价格变化的曲线，以及修正边际曲线。
- **Model diagnostics**：模型训练输出区间、特征分位数、越界情况。
- **About**：方法说明与指标口径。

侧边栏提供三种预设：

| 预设 | 参数 | 特征 | BSM 价格 |
|---|---|---|---|
| Normal market | S=K=150，T1=0.5，T2=1.0，r=5%，σ=20% | 常规水平 | 20.78 |
| High-volatility stress | σ=50% | 波动率与 VIX 同步抬升 | 49.77 |
| Black swan (COVID-style) | σ=80% | 单日急跌、VIX 跳升、相关性反转 | 78.29 |

预设会同时设定 σ 与 10 个市场特征；选 “Custom” 则保留当前手动输入的取值。

勾选 “Link market features to sigma and spot” 后，特征会随参数联动；取消勾选可手动编辑 10 个特征。

### 4.3 复现实验

```bash
python "Week 2.py"                 # 数据清洗与特征
python Week4_MonteCarlo_Sim.py     # 蒙特卡洛基准
python Week4_Ablation_Study.py     # 波动率消融
python Week5_ML_Approach1.py
python Week5_ML_Approach2.py
python Week5_Evaluation_Metrics.py
python Week6_Tuning_and_SHAP.py
python Week6_Tuning_Comparison.py
python Week7_Stress_Test.py
python Week8_BlackSwan_Test.py
```

输出文件会写回项目根目录。`Week8_BlackSwan_Test.py` 会生成结果 CSV 与两张诊断图。

### 4.4 生成结项材料

```bash
python make_report_figures.py      # 报告配图
python make_banner.py              # PPT 概览底图
```

Word / PPT 由交付目录中的构建脚本生成，不属于运行时依赖。

---

## 5. 数据字典

原始数据均来自公开接口（yfinance / FRED），按 JPM 交易日对齐。

| 文件 | 字段 | 说明 |
|---|---|---|
| `jpm_stock.csv` | Date, Close, High, Low, Open, Volume | JPM 日频行情（2018-01-02 至 2024-12-31，1,700+ 交易日） |
| `vix.csv` | Date, VIX | CBOE 波动率指数 |
| `treasury.csv` | Date, 10Y | 美国 10 年期国债收益率（%） |
| `features.csv` / `.parquet` | 见下表 | 清洗并构造后的建模面板 |

模型使用的 10 个特征（顺序固定，与 `pricing_core.FEATURE_ORDER` 一致）：

| 特征 | 含义 |
|---|---|
| `Return` | 当日收益率 |
| `Vol_5` | 5 日滚动年化波动率 |
| `Vol_20` | 20 日滚动年化波动率 |
| `Vol_60` | 60 日滚动年化波动率 |
| `Volume_MA5` | 5 日平均成交量 |
| `VIX_Change` | VIX 日变化 |
| `VIX_JPM_Corr` | VIX 与 JPM 收益率的滚动相关性 |
| `Rate_Momentum` | 国债收益率动量 |
| `Return_Lag1` | 前一日收益率 |
| `MA_Gap` | 价格相对均线的偏离度 |

缺失值处理只做前向填充，不使用未来数据回填；IQR 识别异常值但保留 2020 年 3 月等真实极端行情。

---

## 6. 方法说明

### 6.1 蒙特卡洛基准

- Heston 随机波动率（方差过程全截断，避免负值）
- 跳跃扩散：跳跃概率 5%，跳跃幅度服从对数正态
- 杠杆效应：价格与波动率相关系数 ρ = −0.7
- 流动性摩擦：最终成交价双边扣减 5%
- 10 万条路径、252 个时间步、对偶变量法降方差、固定随机种子

### 6.2 两条机器学习路线

**方案一**：XGBoost 预测下一日 ATM 隐含波动率，再代入 BSM。保留完整可解释结构，但日频特征难以预测波动率的 regime shift。

**方案二**：BSM 提供基准，GBDT 只学习基准与目标之间的残差。最终模型参数为 `learning_rate=0.01, max_depth=3, n_estimators=50`，由 TimeSeriesSplit（3 折）网格搜索确定，随机搜索与 Optuna 得到一致结果。

数据按时间顺序 70% / 15% / 15% 切分，测试集 256 个交易日，不做随机打乱。

### 6.3 指标口径（重要）

第 5、6 周的 MAE、RMSE、R²、MAPE **以 BSM/VIX 代理价格为基准计算**，不是以交易所真实成交价计算。由于公开接口拿不到 JPM 期权逐笔成交，目标价格本身由 VIX 隐含波动率代入 BSM 得到。因此 0.9904 的 R² 应理解为“模型没有破坏基准”，而不是“模型能精确预测市场”。

### 6.4 安全护栏

- 修正幅度限制在 BSM 价格的 ±20% 以内，原始修正单独展示
- 特征与训练集 1%–99% 分位比较，越界时在界面显式警告
- 最终价格不得为负

护栏的作用不是提高精度，而是防止模型在失效区间给出误导性数字。

---

## 7. 第 8 周黑天鹅测试

测试包含 5 组情景、18 个案例：

1. **波动率梯**：σ = 20% / 30% / 50% / 80% / 100% / 120% / 150%
2. **股价崩跌梯**：S = 150 / 135 / 120 / 100 / 80 / 60（σ=80%，r=1%）
3. **利率梯**：r = 5% / 7% / 9%（σ=50%）
4. **复合黑天鹅**：S=100、σ=80%、r=1%，叠加单日 −15%、VIX 单日 +50% 等恐慌特征
5. **历史对照**：2020-03-16 COVID 冲击

五条合理性判据：方向为正、|修正| ≤ 20% BSM、落在训练输出区间、最终价格非负、越界特征数量。

结果：

- 训练输出区间为 `[-5.3717, +3.8341]`
- 波动率梯上修正恒定 `0.001527` 美元
- 深跌梯相对修正从 `0.0032%` 升至 `83.27%`
- 方向正确率 100%，量级达标率 78%，存在越界特征的情景占 89%

结论与工具处置写在报告第 8 章。

---

## 8. 已知局限

1. **缺少真实期权成交价**：目标价是 VIX + BSM 的代理值，指标不能直接解读为市场预测精度。
2. **样本量有限**：约 1,700 个交易日，极端样本更少，树模型难以学到尾部规律。
3. **特征维度有限**：缺少期权链偏斜、期限结构、持仓量等更有信息量的变量。
4. **树模型不能外推**：这是结构性限制，不能靠调参解决；需要护栏、越界检测，或换用带外推能力的参数化模型。
5. **未做交易成本与对冲模拟**：当前是定价研究，不是完整交易策略回测。

---

## 9. 交付物

| 文件 | 说明 |
|---|---|
| `Ziyue Qian_Week8_Weekly Report.docx` | 第 8 周周报 |
| `Ziyue Qian_Final_Report.docx` / `.pdf` | 结项报告（15 页，8,500+ 汉字，13 图 10 表） |
| `Ziyue Qian_Advanced Chooser Option Pricing.pptx` | 结项汇报 PPT（15 页） |
| `Ziyue Qian_Demo_Video_Script.docx` | 5–10 分钟演示视频录制脚本 |
| `Week8_App.py` | 最终交付的交互工具 |

---

## 10. 参考

- Rubinstein, M. (1991). *Options for the undecided*. Risk, 4(9).
- Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. JPE.
- Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility*. RFS.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.