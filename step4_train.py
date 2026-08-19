#!/usr/bin/env python3
"""
Step 4: WAF 攻击检测模型训练
支持 LightGBM 和 XGBoost，可单独训练或同时训练对比。

用法:
    python step4_train.py                    # 默认训练 LightGBM
    python step4_train.py --model lightgbm    # 只训练 LightGBM
    python step4_train.py --model xgboost    # 只训练 XGBoost
    python step4_train.py --model both       # 两个都训练并对比
"""

import argparse
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
)

# ── 配置 ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "features"
MODEL_DIR   = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# 特征文件（实际文件名带域名）
FEATURE_FILES = [
    FEATURE_DIR / "www.zstzpt.com.parquet",
    FEATURE_DIR / "data.zstzpt.com.parquet",
]

# 二分类标签列
LABEL_COL = "is_attack"

# 非特征列（训练时剔除）
NON_FEATURE_COLS = [LABEL_COL, "attack_type", "ip", "path", "source"]

RANDOM_STATE = 42
TEST_SIZE    = 0.2


def log(msg):
    """带时间戳的 flush 打印"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 数据加载 ──────────────────────────────────────────
def load_data():
    """加载 Parquet，只读需要的列以节省内存"""
    dfs = []
    for f in FEATURE_FILES:
        if not f.exists():
            log(f"WARNING: 文件不存在 {f}")
            continue
        log(f"加载 {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB) ...")

        # 先读列名，确定要读哪些列
        pf = pd.read_parquet(f, columns=None)
        # 只保留数值特征列 + 标签列（丢弃 ip/path/source/attack_type 等字符串列）
        cols_to_keep = [c for c in pf.columns if c not in NON_FEATURE_COLS or c == LABEL_COL]
        pf = pf[cols_to_keep]
        dfs.append(pf)
        log(f"  -> {len(pf):,} 行, {pf.shape[1]} 列")

    if not dfs:
        raise FileNotFoundError(f"没有找到特征文件，请检查 {FEATURE_DIR}")

    df = pd.concat(dfs, ignore_index=True)
    log(f"合并完成: {len(df):,} 行, {df.shape[1]} 列")

    # 分离特征与标签
    y = df[LABEL_COL]
    X = df.drop(columns=[LABEL_COL])

    # 确保所有特征列都是数值型
    for col in X.columns:
        if X[col].dtype == object:
            log(f"WARNING: 列 {col} 是 object 类型，尝试转为数值...")
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    log(f"特征数: {X.shape[1]}")
    log(f"正样本(攻击)比例: {y.mean():.4f} ({y.sum():,} / {len(y):,})")
    log(f"标签分布:\n{y.value_counts().to_string()}")

    return X, y


# ── 数据划分 ──────────────────────────────────────────
def split_data(X, y):
    """
    随机 shuffle 划分（不用 stratify，400万行 stratify 内存开销太大）。
    正样本占 65%，随机划分后训练集/测试集分布差异可忽略。
    """
    log("划分训练集/测试集 (80/20, 随机 shuffle) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    log(f"训练集: {len(X_train):,}  测试集: {len(X_test):,}")
    log(f"训练集正样本比例: {y_train.mean():.4f}  测试集正样本比例: {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test


# ── 评估 ──────────────────────────────────────────────
def evaluate(model, name, X_test, y_test):
    """评估模型并返回指标字典"""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    log(f"\n{'='*60}")
    log(f"  {name} 测试集评估")
    log(f"{'='*60}")
    log(f"  Accuracy:  {acc:.4f}")
    log(f"  Precision: {prec:.4f}")
    log(f"  Recall:    {rec:.4f}")
    log(f"  F1:        {f1:.4f}")
    log(f"  AUC:       {auc:.4f}")
    log(f"  混淆矩阵: TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    log(f"  误报率(FPR): {fp/(fp+tn):.4f}")
    log(f"  漏报率(FNR): {fn/(fn+tp):.4f}")

    # 分类报告
    report = classification_report(y_test, y_pred, target_names=["normal", "attack"], digits=4)
    log(f"\n分类报告:\n{report}")

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": round(fp/(fp+tn), 4),
        "fnr": round(fn/(fn+tp), 4),
    }


# ── 特征重要性 ────────────────────────────────────────
def feature_importance(model, name, feature_names):
    """打印并保存特征重要性"""
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    log(f"\n{name} 特征重要性 Top-15:")
    log(imp.head(15).to_string(index=False))

    # 保存完整列表
    imp_path = MODEL_DIR / f"{name}_feature_importance.csv"
    imp.to_csv(imp_path, index=False)
    log(f"特征重要性已保存: {imp_path}")


# ── 训练 LightGBM ────────────────────────────────────
def train_lightgbm(X_train, y_train, X_test, y_test, feature_names):
    import lightgbm as lgb

    log("\n" + "="*60)
    log("训练 LightGBM ...")
    log("="*60)

    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )
    elapsed = time.time() - t0
    log(f"LightGBM 训练完成，耗时 {elapsed:.1f}s，最佳迭代 {model.best_iteration_}")

    # 评估
    metrics = evaluate(model, "LightGBM", X_test, y_test)
    metrics["train_time_sec"] = round(elapsed, 1)
    metrics["best_iteration"] = int(model.best_iteration_)

    # 特征重要性
    feature_importance(model, "lightgbm", feature_names)

    # 保存模型
    model_path = MODEL_DIR / "lightgbm_model.pkl"
    import joblib
    joblib.dump(model, model_path)
    log(f"模型已保存: {model_path}")

    return model, metrics


# ── 训练 XGBoost ──────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    import xgboost as xgb

    log("\n" + "="*60)
    log("训练 XGBoost ...")
    log("="*60)

    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
        early_stopping_rounds=50,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    elapsed = time.time() - t0
    log(f"XGBoost 训练完成，耗时 {elapsed:.1f}s")

    # 评估
    metrics = evaluate(model, "XGBoost", X_test, y_test)
    metrics["train_time_sec"] = round(elapsed, 1)

    # 特征重要性
    feature_importance(model, "xgboost", feature_names)

    # 保存模型
    model_path = MODEL_DIR / "xgboost_model.pkl"
    import joblib
    joblib.dump(model, model_path)
    log(f"模型已保存: {model_path}")

    return model, metrics


# ── 主流程 ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="WAF 攻击检测模型训练")
    parser.add_argument("--model", choices=["lightgbm", "xgboost", "both"], default="lightgbm",
                        help="选择训练的模型 (default: lightgbm)")
    args = parser.parse_args()

    log("="*60)
    log("  WAF 攻击检测模型训练")
    log(f"  模型: {args.model}")
    log("="*60)

    # 加载数据
    X, y = load_data()
    feature_names = X.columns.tolist()

    # 划分
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 保存特征列表（推理时对齐）
    feat_path = MODEL_DIR / "feature_columns.json"
    with open(feat_path, "w") as f:
        json.dump(feature_names, f)
    log(f"特征列表已保存: {feat_path}")

    all_metrics = {}

    # 训练
    if args.model in ("lightgbm", "both"):
        _, m = train_lightgbm(X_train, y_train, X_test, y_test, feature_names)
        all_metrics["lightgbm"] = m

    if args.model in ("xgboost", "both"):
        _, m = train_xgboost(X_train, y_train, X_test, y_test, feature_names)
        all_metrics["xgboost"] = m

    # 对比汇总
    if len(all_metrics) > 1:
        log("\n" + "="*60)
        log("  模型对比汇总")
        log("="*60)
        summary = pd.DataFrame(all_metrics).T
        log(summary.to_string())

    # 保存评估结果
    report_path = MODEL_DIR / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    log(f"\n评估报告已保存: {report_path}")

    log("\n全部完成!")


if __name__ == "__main__":
    main()
