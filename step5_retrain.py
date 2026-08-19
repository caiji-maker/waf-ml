#!/usr/bin/env python3
"""
Step 5: 移除 seo_spam + 补漏标规则 + 重新训练
关键改动：
  1. 移除 seo_spam（216万条，用规则拦截即可）
  2. 补漏标规则：?/ 模式、Nacos 路径、可疑 PHP、双斜杠、?p=/ 等
  3. 用清洗后的数据重新训练 LightGBM 和 XGBoost

用法:
    python step5_retrain.py --model lightgbm
    python step5_retrain.py --model xgboost
    python step5_retrain.py --model both
"""
import argparse
import sys
import time
import json
import re
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

FEATURE_FILES = [
    FEATURE_DIR / "www.zstzpt.com.parquet",
    FEATURE_DIR / "data.zstzpt.com.parquet",
]

LABEL_COL = "is_attack"
NON_FEATURE_COLS = [LABEL_COL, "attack_type", "ip", "path", "source"]
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# 补漏标规则（匹配 path 的正则，命中则标为 attack）
MISLABEL_PATTERNS = [
    # ?/ 模式的 SEO 垃圾（当前只覆盖了 ?and/，漏了 ?/）
    (r"/\?/", "seo_spam_slash"),
    # Nacos 扫描
    (r"/nacos/", "vuln_scan"),
    # 可疑随机命名 PHP（8位以上随机字符.php）
    (r"/[a-zA-Z0-9]{6,}\.php(?:/|$)", "sensitive_scan"),
    # 双斜杠路径扫描
    (r"^//", "vuln_scan"),
    # ?p=/ 路径注入
    (r"\?p=/", "path_traversal"),
    # /pki-validationwp.php
    (r"pki-validation.*\.php", "sensitive_scan"),
    # /runtime/archive/*.php
    (r"/runtime/archive/.*\.php", "sensitive_scan"),
    # Nacos API 路径（不带 /nacos/ 前缀的）
    (r"/v1/(auth|cs)/", "vuln_scan"),
    (r"/v1/core/cluster/", "vuln_scan"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 数据加载与清洗 ─────────────────────────────────────
def load_and_clean():
    """加载数据 → 移除 seo_spam → 补漏标规则 → 返回清洗后的 X, y"""
    dfs = []
    for f in FEATURE_FILES:
        if not f.exists():
            log(f"WARNING: 文件不存在 {f}")
            continue
        log(f"加载 {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB) ...")
        df = pd.read_parquet(f)
        dfs.append(df)
        log(f"  -> {len(df):,} 行, attack_type 分布:")
        log(f"     {df['attack_type'].value_counts().head(5).to_dict()}")

    df = pd.concat(dfs, ignore_index=True)
    log(f"合并完成: {len(df):,} 行")

    # ── 关键步骤1：移除 seo_spam ──
    before = len(df)
    seo_count = (df["attack_type"] == "seo_spam").sum()
    df = df[df["attack_type"] != "seo_spam"].copy()
    log(f"移除 seo_spam: {seo_count:,} 条, 剩余 {len(df):,} 条")

    # ── 关键步骤2：补充漏标规则 ──
    # 从 path 列匹配漏标攻击（标为 normal 但实际是攻击的）
    path_col = df["path"].fillna("").astype(str)
    mislabeled_count = 0
    for pattern, _ in MISLABEL_PATTERNS:
        mask = path_col.str.contains(pattern, case=False, regex=True, na=False)
        # 只修改当前标为 normal 的样本
        hit = mask & (df["attack_type"] == "normal") & (df[LABEL_COL] == 0)
        hit_count = hit.sum()
        if hit_count > 0:
            df.loc[hit, LABEL_COL] = 1
            df.loc[hit, "attack_type"] = "mislabeled_fixed"
            mislabeled_count += hit_count
            log(f"  补漏标 [{pattern[:40]}]: 修正 {hit_count:,} 条")

    log(f"共修正漏标: {mislabeled_count:,} 条")

    # ── 数据清洗后统计 ──
    y = df[LABEL_COL]
    log(f"清洗后样本: {len(df):,} 条")
    log(f"  正样本(攻击): {y.sum():,} ({y.mean():.2%})")
    log(f"  负样本(正常): {(1-y).sum():,}")
    log(f"  attack_type 分布:")
    log(f"  {df['attack_type'].value_counts().to_dict()}")

    # ── 分离特征与标签 ──
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    X = X.fillna(0)

    # 确保所有特征列都是数值型
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    return X, y


# ── 数据划分 ──────────────────────────────────────────
def split_data(X, y):
    log("划分训练集/测试集 (80/20, 随机 shuffle) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    log(f"训练集: {len(X_train):,}  测试集: {len(X_test):,}")
    log(f"训练集正样本比例: {y_train.mean():.4f}  测试集: {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test


# ── 评估 ──────────────────────────────────────────────
def evaluate(model, name, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    log(f"\n{'='*60}")
    log(f"  {name} 测试集评估")
    log(f"{'='*60}")
    log(f"  Accuracy:  {acc:.4f}")
    log(f"  Precision: {prec:.4f}")
    log(f"  Recall:    {rec:.4f}")
    log(f"  F1:        {f1:.4f}")
    log(f"  AUC:       {auc:.4f}")
    log(f"  混淆矩阵: TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    log(f"  误报率(FPR): {fpr:.2%}")
    log(f"  漏报率(FNR): {fnr:.2%}")

    # 与旧模型对比
    log(f"\n  --- 与旧模型对比 ---")
    log(f"  {'指标':<16s} {'旧LightGBM':>12s} {'新' + name:>12s} {'变化':>10s}")
    log(f"  {'-'*52}")
    old = {"f1": 0.9370, "fpr": 0.1487, "fnr": 0.0118, "auc": 0.9599}
    new = {"f1": f1, "fpr": fpr, "fnr": fnr, "auc": auc}
    for metric in ["f1", "fpr", "fnr", "auc"]:
        o, n = old[metric], new[metric]
        delta = n - o
        log(f"  {metric:<16s} {o:>12.4f} {n:>12.4f} {delta:>+10.4f}")

    # 阈值扫描
    log(f"\n  --- 阈值扫描 ---")
    log(f"  {'阈值':<8s} {'F1':<10s} {'FPR':<10s} {'FNR':<10s} {'FP数':<10s} {'FN数':<10s}")
    log(f"  {'-'*56}")
    for th in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        y_pred_th = (y_proba >= th).astype(int)
        tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_th).ravel()
        f1_2 = f1_score(y_test, y_pred_th)
        fpr2 = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0
        fnr2 = fn2 / (fn2 + tp2) if (fn2 + tp2) > 0 else 0
        log(f"  {th:<8.2f} {f1_2:<10.4f} {fpr2:<10.2%} {fnr2:<10.2%} {fp2:<10,} {fn2:<10,}")

    return {
        "model": name, "accuracy": round(acc, 4),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4), "auc": round(auc, 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": round(fpr, 4), "fnr": round(fnr, 4),
    }


# ── 特征重要性 ────────────────────────────────────────
def feature_importance(model, name, feature_names):
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    log(f"\n{name} 特征重要性 Top-15:")
    log(imp.head(15).to_string(index=False))

    imp_path = MODEL_DIR / f"{name}_feature_importance_no_seospam.csv"
    imp.to_csv(imp_path, index=False)
    log(f"特征重要性已保存: {imp_path}")


# ── 训练 LightGBM ────────────────────────────────────
def train_lightgbm(X_train, y_train, X_test, y_test, feature_names):
    import lightgbm as lgb

    log("\n" + "="*60)
    log("训练 LightGBM (移除seo_spam后, 加权) ...")
    log("="*60)

    # 计算正负样本比，用于 scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = neg_count / pos_count
    log(f"正样本: {pos_count:,}  负样本: {neg_count:,}  scale_pos_weight={spw:.2f}")

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
        scale_pos_weight=spw,  # 关键：补偿正负样本不平衡
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

    metrics = evaluate(model, "LightGBM", X_test, y_test)
    metrics["train_time_sec"] = round(elapsed, 1)
    metrics["best_iteration"] = int(model.best_iteration_)

    feature_importance(model, "lightgbm", feature_names)

    import joblib
    model_path = MODEL_DIR / "lightgbm_model_no_seospam.pkl"
    joblib.dump(model, model_path)
    log(f"模型已保存: {model_path}")

    return model, metrics


# ── 训练 XGBoost ──────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    import xgboost as xgb

    log("\n" + "="*60)
    log("训练 XGBoost (移除seo_spam后, 加权) ...")
    log("="*60)

    # 计算正负样本比，用于 scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = neg_count / pos_count
    log(f"正样本: {pos_count:,}  负样本: {neg_count:,}  scale_pos_weight={spw:.2f}")

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
        scale_pos_weight=spw,  # 关键：补偿正负样本不平衡
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

    metrics = evaluate(model, "XGBoost", X_test, y_test)
    metrics["train_time_sec"] = round(elapsed, 1)

    feature_importance(model, "xgboost", feature_names)

    import joblib
    model_path = MODEL_DIR / "xgboost_model_no_seospam.pkl"
    joblib.dump(model, model_path)
    log(f"模型已保存: {model_path}")

    return model, metrics


# ── 主流程 ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="WAF 重训(移除seo_spam)")
    parser.add_argument("--model", choices=["lightgbm", "xgboost", "both"], default="both",
                        help="选择训练的模型 (default: both)")
    args = parser.parse_args()

    log("="*60)
    log("  WAF 重训：移除 seo_spam + 补漏标规则")
    log(f"  模型: {args.model}")
    log("="*60)

    # 加载并清洗数据
    X, y = load_and_clean()
    feature_names = X.columns.tolist()

    # 划分
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 保存特征列表
    feat_path = MODEL_DIR / "feature_columns_no_seospam.json"
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
        log("  新模型对比汇总")
        log("="*60)
        summary = pd.DataFrame(all_metrics).T
        log(summary.to_string())

    # 保存评估结果
    report_path = MODEL_DIR / "training_report_no_seospam.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    log(f"\n评估报告已保存: {report_path}")

    log("\n全部完成!")


if __name__ == "__main__":
    main()
