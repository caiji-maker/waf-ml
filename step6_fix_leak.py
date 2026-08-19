#!/usr/bin/env python3
"""
Step 6: 修复训练 —— 解决数据泄露 + LightGBM 崩溃问题

关键改动：
  1. 移除 7 个响应特征（WAF 决策时拿不到）：status_code, is_4xx, is_5xx, is_444, is_404, is_403, body_size
  2. 新增 4 个请求时可用特征：path_is_root, is_static_file, has_hash_filename, is_known_api_prefix
  3. LightGBM: 用 is_unbalance=True 代替 scale_pos_weight=23.52 + 降学习率 + 加正则
  4. XGBoost: scale_pos_weight 从 23.52 降到 sqrt(23.52)≈4.85（更温和）
  5. 两模型都加更严格的正则化防止过拟合

用法:
    python step6_fix_leak.py --model lightgbm
    python step6_fix_leak.py --model xgboost
    python step6_fix_leak.py --model both
"""
import argparse
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
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

# 响应特征（WAF 决策时不可用，数据泄露！）
LEAK_FEATURES = ["status_code", "is_4xx", "is_5xx", "is_444", "is_404", "is_403", "body_size"]

RANDOM_STATE = 42
TEST_SIZE    = 0.2

# 补漏标规则
MISLABEL_PATTERNS = [
    (r"/\?/", "seo_spam_slash"),
    (r"/nacos/", "vuln_scan"),
    (r"/[a-zA-Z0-9]{6,}\.php(?:/|$)", "sensitive_scan"),
    (r"^//", "vuln_scan"),
    (r"\?p=/", "path_traversal"),
    (r"pki-validation.*\.php", "sensitive_scan"),
    (r"/runtime/archive/.*\.php", "sensitive_scan"),
    (r"/v1/(auth|cs)/", "vuln_scan"),
    (r"/v1/core/cluster/", "vuln_scan"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_request_features(df):
    """新增 4 个请求时可用的路径语义特征（直接从 path 列生成）"""
    path = df["path"].fillna("").astype(str)

    # 1. 是否根路径 / 或 /index.*
    df["path_is_root"] = (
        path.isin(["/", "/index.html", "/index.htm", "/index.php"])
    ).astype(int)

    # 2. 是否静态文件（.css, .js, .jpg, .png, .gif, .ico, .woff, .ttf, .svg, .map）
    static_ext = r"\.(css|js|jpg|jpeg|png|gif|ico|woff2?|ttf|svg|map|eot|mp4|mp3|webp|webm|flv|swf)(?:\?|$)"
    df["is_static_file"] = path.str.contains(static_ext, case=False, regex=True, na=False).astype(int)

    # 3. 是否哈希命名文件（随机 8+ 位十六进制或字母数字组合.扩展名）
    hash_pattern = r"/[a-f0-9]{8,}\.\w{2,5}$|/[a-f0-9]{32,}(?:\.js|\.css)?$"
    df["has_hash_filename"] = path.str.contains(hash_pattern, case=False, regex=True, na=False).astype(int)

    # 4. 是否已知 API 前缀
    api_prefix = r"^/api/|^/v[0-9]+/|^/graphql|^/webhook"
    df["is_known_api_prefix"] = path.str.contains(api_prefix, case=False, regex=True, na=False).astype(int)

    return df


# ── 数据加载与清洗 ─────────────────────────────────────
def load_and_clean():
    dfs = []
    for f in FEATURE_FILES:
        if not f.exists():
            log(f"WARNING: 文件不存在 {f}")
            continue
        log(f"加载 {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB) ...")
        df = pd.read_parquet(f)
        dfs.append(df)
        log(f"  -> {len(df):,} 行")

    df = pd.concat(dfs, ignore_index=True)
    log(f"合并完成: {len(df):,} 行")

    # 移除 seo_spam
    seo_count = (df["attack_type"] == "seo_spam").sum()
    df = df[df["attack_type"] != "seo_spam"].copy()
    log(f"移除 seo_spam: {seo_count:,} 条, 剩余 {len(df):,} 条")

    # 补漏标
    path_col = df["path"].fillna("").astype(str)
    mislabeled_count = 0
    for pattern, _ in MISLABEL_PATTERNS:
        mask = path_col.str.contains(pattern, case=False, regex=True, na=False)
        hit = mask & (df["attack_type"] == "normal") & (df[LABEL_COL] == 0)
        hit_count = hit.sum()
        if hit_count > 0:
            df.loc[hit, LABEL_COL] = 1
            df.loc[hit, "attack_type"] = "mislabeled_fixed"
            mislabeled_count += hit_count
            log(f"  补漏标 [{pattern[:40]}]: 修正 {hit_count:,} 条")
    log(f"共修正漏标: {mislabeled_count:,} 条")

    # 新增请求时可用特征
    log("新增 4 个路径语义特征...")
    df = add_request_features(df)

    # 分离特征与标签
    y = df[LABEL_COL]
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])

    # 移除响应特征（数据泄露）
    leak_cols = [c for c in LEAK_FEATURES if c in X.columns]
    if leak_cols:
        log(f"移除 {len(leak_cols)} 个响应特征（数据泄露）: {leak_cols}")
        X = X.drop(columns=leak_cols)

    X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    log(f"最终特征数: {len(X.columns)}")
    log(f"  正样本(攻击): {y.sum():,} ({y.mean():.2%})")
    log(f"  负样本(正常): {(1-y).sum():,}")

    return X, y


def split_data(X, y):
    log("划分训练集/测试集 (80/20, 随机 shuffle) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    log(f"训练集: {len(X_train):,}  测试集: {len(X_test):,}")
    log(f"训练集正样本比例: {y_train.mean():.4f}  测试集: {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test


def evaluate(model, name, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
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

    # 阈值扫描
    log(f"\n  --- 阈值扫描 ---")
    log(f"  {'阈值':<8s} {'F1':<10s} {'FPR':<10s} {'FNR':<10s} {'FP数':<10s} {'FN数':<10s}")
    log(f"  {'-'*56}")
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
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


def feature_importance(model, name, feature_names):
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    log(f"\n{name} 特征重要性 Top-15:")
    log(imp.head(15).to_string(index=False))

    imp_path = MODEL_DIR / f"{name}_feature_importance_v3.csv"
    imp.to_csv(imp_path, index=False)
    log(f"特征重要性已保存: {imp_path}")


# ── 训练 LightGBM ────────────────────────────────────
def train_lightgbm(X_train, y_train, X_test, y_test, feature_names):
    import lightgbm as lgb

    log("\n" + "="*60)
    log("训练 LightGBM (移除泄露特征 + is_unbalance + 强正则) ...")
    log("="*60)

    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    spw_raw = neg_count / pos_count
    spw_sqrt = spw_raw ** 0.5  # 用 sqrt(spw) 代替 is_unbalance，更温和
    log(f"正样本: {pos_count:,}  负样本: {neg_count:,}  不平衡比: {spw_raw:.2f}, sqrt={spw_sqrt:.2f}")

    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        min_child_samples=100,
        min_gain_to_split=0.5,      # 防止分裂出太小的叶子
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        scale_pos_weight=spw_sqrt,   # 用 sqrt(spw) 而非 is_unbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),  # 给更多容忍
            lgb.log_evaluation(period=100),
        ],
    )
    elapsed = time.time() - t0
    log(f"LightGBM 训练完成，耗时 {elapsed:.1f}s，最佳迭代 {model.best_iteration_}")

    metrics = evaluate(model, "LightGBM", X_test, y_test)
    metrics["train_time_sec"] = round(elapsed, 1)
    metrics["best_iteration"] = int(model.best_iteration_)

    feature_importance(model, "lightgbm", feature_names)

    import joblib
    model_path = MODEL_DIR / "lightgbm_model_v3.pkl"
    joblib.dump(model, model_path)
    log(f"模型已保存: {model_path}")

    return model, metrics


# ── 训练 XGBoost ──────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    import xgboost as xgb

    log("\n" + "="*60)
    log("训练 XGBoost (移除泄露特征 + sqrt(spw) 加权) ...")
    log("="*60)

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    # 用 sqrt(scale_pos_weight) 代替原始值，更温和
    spw_raw = neg_count / pos_count
    spw_sqrt = spw_raw ** 0.5
    log(f"正样本: {pos_count:,}  负样本: {neg_count:,}")
    log(f"原始 spw={spw_raw:.2f}, sqrt(spw)={spw_sqrt:.2f}")

    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=5,         # 比之前大（3→5）
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.5,                   # 比之前大（0.1→0.5）
        reg_alpha=1.0,               # 比之前大
        reg_lambda=2.0,              # 比之前大
        scale_pos_weight=spw_sqrt,   # 温和加权
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
        early_stopping_rounds=100,
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
    model_path = MODEL_DIR / "xgboost_model_v3.pkl"
    joblib.dump(model, model_path)
    log(f"模型已保存: {model_path}")

    return model, metrics


# ── 主流程 ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="WAF 训练 v3 (修复数据泄露)")
    parser.add_argument("--model", choices=["lightgbm", "xgboost", "both"], default="both")
    args = parser.parse_args()

    log("="*60)
    log("  WAF 训练 v3: 移除泄露特征 + 平衡修正")
    log(f"  模型: {args.model}")
    log("="*60)

    X, y = load_and_clean()
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = split_data(X, y)

    feat_path = MODEL_DIR / "feature_columns_v3.json"
    with open(feat_path, "w") as f:
        json.dump(feature_names, f)
    log(f"特征列表已保存: {feat_path}")
    log(f"特征列: {feature_names}")

    all_metrics = {}

    if args.model in ("lightgbm", "both"):
        _, m = train_lightgbm(X_train, y_train, X_test, y_test, feature_names)
        all_metrics["lightgbm"] = m

    if args.model in ("xgboost", "both"):
        _, m = train_xgboost(X_train, y_train, X_test, y_test, feature_names)
        all_metrics["xgboost"] = m

    if len(all_metrics) > 1:
        log("\n" + "="*60)
        log("  V3 模型对比汇总")
        log("="*60)
        summary = pd.DataFrame(all_metrics).T
        log(summary.to_string())

    report_path = MODEL_DIR / "training_report_v3.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    log(f"\n评估报告已保存: {report_path}")

    log("\n全部完成!")


if __name__ == "__main__":
    main()
