---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: '62dae807-f6db-4955-a5bd-cbf2c8867e34'
  PropagateID: '62dae807-f6db-4955-a5bd-cbf2c8867e34'
  ReservedCode1: '629a7d7f-0d48-4a7e-91ac-6a96916127de'
  ReservedCode2: '629a7d7f-0d48-4a7e-91ac-6a96916127de'
---

# WAF-ML — 攻击检测模型训练

从 Nginx Access 日志训练机器学习模型，判断请求是否为攻击。产出 LightGBM 模型供 [Mini-WAF](../mini-waf) 的 ml_detect 插件使用。

## 训练管线

13 个步骤覆盖完整闭环，可单独运行或一键全流程：

```bash
# 一键全流程
python pipeline.py

# 单步运行
python pipeline.py --step 1    # 解析
python pipeline.py --step 2    # 标注
python pipeline.py --step 3    # 特征
python pipeline.py --step 4    # 训练
```

### 步骤说明

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 1 | step1_parse.py | Nginx 日志 → Parquet（流式解析，20万行/块） |
| 2 | step2_label.py | 规则标注：SEO spam / 敏感文件 / 漏洞扫描 / 路径穿越 / SQL注入 / XSS / 命令注入 |
| 2b | step2b_find_mislabeled.py | 漏标检测 |
| 3 | step3_features.py | 28 个特征工程（URL/UA/请求属性/统计/路径语义） |
| 4 | step4_train.py | LightGBM + XGBoost 训练对比 |
| 4b | step4b_threshold_analysis.py | 阈值分析 |
| 5 | step5_retrain.py | 修复数据泄露后重训 |
| 6 | step6_fix_leak.py | 移除 7 个响应特征（status_code 等 WAF 决策时拿不到） |
| 7 | step7_fnr_analysis.py | 漏报分析 |
| 8 | step8_replay.py | Replay 重放验证（WAF 必须先启动） |
| 8b-d | step8b/c/d_*.py | 离线 Replay / 完整重放变体 |
| 9 | step9_redteam.py | 红队测试 |
| 10 | step10_threshold_optimize.py | 阈值寻优 |
| 11 | step11_diagnose.py | 误报根因诊断 |
| 12 | step12_train_v4.py | v4 重训（移除统计特征） |
| 12b | step12b_train_v4_tuned.py | v4 超参搜索（3 组对比） |
| 13 | step13_rule_coverage.py | 旧规则覆盖率诊断 |
| 13b | step13b_rule_coverage_new.py | 新规则覆盖率诊断 |
| 14 | step14_multiclass_train.py | 多分类模型训练（攻击类型判断） |

## 关键结论

### 模型迭代

| 版本 | AUC | FPR(离线) | FPR(Replay) | 结论 |
|------|-----|-----------|-------------|------|
| v1 | 0.98+ | 3.3% | — | 含数据泄露（7个响应特征） |
| v3 | 0.9833 | 3.32% | 52.8% | 移除泄露后，统计特征线上失效 |
| v4 | 0.86 | 0% | — | 移除统计特征，F1=0，纯 ML 走不通 |

### v3 模型 Replay 失效根因

模型 42.8% 的重要性来自 3 个统计特征：

| 特征 | 重要性 | 问题 |
|------|--------|------|
| req_count_60s | 19.3% | 线上新 IP 首访必为 0 |
| unique_url_60s | 14.9% | 同上 |
| err_count_60s | 8.6% | 同上 |

训练集中正常用户这些值远高于攻击者（正常用户浏览多页面→req_count 高），但线上每个新 IP 首次访问时这些值必然为 0，导致正常流量被大量误判为攻击。

### 最终架构决策

从"ML 为主"转向 **规则为主 + 频率限制 + ML 观察模式**：
- 第一层：blacklist 正则规则（60+ URI + 17 UA + 20 body）
- 第二层：limit 频率限制（200次/分钟）
- 第三层：badbehavior 行为封禁（10次 4xx → 封禁 1 小时）
- 第四层：ml_detect 双模型 observe 模式（二分类判断是否攻击 + 多分类判断攻击类型，只打分不拦截）

Replay FPR 从 52.8% 降到 5.0%。

### 多分类模型（Step 14）

在 v3 二分类基础上新增多分类模型，判断攻击的具体类型：

| 指标 | 值 |
|------|-----|
| Accuracy | 95.38% |
| Normal FPR | 3.98%（<5% 红线） |
| Macro F1 | 0.4291 |
| Weighted F1 | 0.9619 |

各类别表现：

| 类别 | 样本数 | F1 |
|------|--------|-----|
| normal | 1,776,202 | 0.9777 |
| automated_tool | 40,318 | 0.6797 |
| sensitive_scan | 9,938 | 0.5822 |
| vuln_scan | 12,060 | 0.4753 |
| seo_spam_slash | 11,301 | 0.4530 |
| path_traversal | 1,373 | 0.2367 |
| path_padding | 214 | 0（样本太少） |
| sql_injection | 79 | 0.0281（样本太少） |

设计：与 v3 二分类并行使用。二分类负责"是否攻击"的 FPR 控制，多分类负责"什么类型"的分类，互不干扰。

## 产出文件

最终使用的文件（Mini-WAF 依赖）：

| 文件 | 说明 |
|------|------|
| model/lightgbm_model_v3.pkl | LightGBM v3 二分类模型（判断是否攻击，observe 模式） |
| model/feature_columns_v3.json | 二分类 28 个特征列名 |
| model/lightgbm_model_mc.pkl | LightGBM 多分类模型（判断攻击类型，observe 模式） |
| model/feature_columns_mc.json | 多分类 28 个特征列名 |
| model/attack_types.json | 攻击类型名称 → 索引映射 |

其他模型（归档参考）：

| 文件 | 说明 |
|------|------|
| model/lightgbm_model_v4.pkl | v4 模型（AUC=0.86，F1=0，不可用） |
| model/xgboost_model_v3.pkl | XGBoost v3（AUC 差 0.0003，无需融合） |
| model/training_report_v3.json | v3 训练报告 |
| model/training_report_v4.json | v4 三组超参报告 |
| model/training_report_mc.json | 多分类训练报告 |
| model/lightgbm_feature_importance_v3.csv | v3 特征重要性 |
| model/lightgbm_feature_importance_mc.csv | 多分类特征重要性 |
| model/attack_types.json | 攻击类型映射 |
| model/replay_results.csv | 最新 Replay 1000 条结果 |

## 数据依赖

训练数据来自两个 Nginx 站点的 access 日志：

| 文件 | 大小 | 说明 |
|------|------|------|
| parsed/www.zstzpt.com.parquet | ~100MB | 主站点解析后数据 |
| parsed/data.zstzpt.com.parquet | ~4MB | 子站点解析后数据 |
| labeled/*.parquet | ~107MB | 标注后数据 |
| features/*.parquet | ~69MB | 特征工程后数据 |

原始日志文件不在本目录中，parsed/ 下是解析后的 Parquet。

## 环境要求

- Python 3.12+
- 路径不能有中文（LightGBM C 库写文件会失败，已用 `D:\training-data`）
- 依赖见 requirements.txt

```bash
pip install -r requirements.txt
```

## 与 Mini-WAF 的关系

```
waf-ml (本项目)                     mini-waf
┌───────────────────────┐          ┌──────────────────────────────┐
│ 日志解析→标注          │          │ config.yaml                  │
│   →特征→训练           │          │   ml_detect:                 │
│                        │  产出     │     model_path: ─────────────┼──→ model/lightgbm_model_v3.pkl
│ lightgbm_model_v3.pkl  │──────→   │     mc_model_path: ──────────┼──→ model/lightgbm_model_mc.pkl
│ lightgbm_model_mc.pkl  │          │   (observe_mode)             │
│ attack_types.json      │          └──────────────────────────────┘
└───────────────────────┘
```

WAF 的 config.yaml 中配置模型路径：

```yaml
ml_detect:
  model_path: "D:/training-data/waf-ml/model/lightgbm_model_v3.pkl"
  mc_model_path: "D:/training-data/waf-ml/model/lightgbm_model_mc.pkl"
  observe_mode: true
```

## 项目结构

```
waf-ml/
├── pipeline.py              # 一键全流程入口
├── step1_parse.py          # 日志解析
├── step2_label.py          # 规则标注
├── step2b_find_mislabeled.py
├── step3_features.py       # 特征工程（28特征）
├── step4_train.py          # 模型训练（LightGBM + XGBoost）
├── step4b_threshold_analysis.py
├── step5_retrain.py        # 重训（修复泄露后）
├── step6_fix_leak.py       # 数据泄露修复
├── step7_fnr_analysis.py   # 漏报分析
├── step8_replay.py          # Replay 验证
├── step8b/c/d_*.py         # Replay 变体
├── step9_redteam.py        # 红队测试
├── step10_threshold_optimize.py
├── step11_diagnose.py      # 误报根因诊断
├── step12_train_v4.py       # v4 重训
├── step12b_train_v4_tuned.py
├── step13_rule_coverage.py  # 规则覆盖率
├── step13b_rule_coverage_new.py
├── step14_multiclass_train.py  # 多分类模型训练（攻击类型判断）
├── parsed/                  # 解析后 Parquet
├── labeled/                 # 标注后 Parquet
├── features/                # 特征 Parquet
└── model/                   # 模型和评估产出
```

> AI生成