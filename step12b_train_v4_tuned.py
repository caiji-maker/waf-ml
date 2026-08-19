#!/usr/bin/env python3
"""
Step 12b: v4 训练调优 —— 放松正则化 + 多组超参搜索

v4 首次训练结果：FNR=100%, F1=0, best_iteration=14
原因：移除统计特征后 25 个特征区分力弱 + 正则化太强 + sqrt(spw) 太温和 → 模型直接摆烂

策略：
  1. 尝试 3 组超参配置，从温和到激进
  2. 放松 min_gain_to_split、reg_alpha、reg_lambda
  3. 尝试更高 scale_pos_weight（raw 23.52 而非 sqrt 4.85）
  4. 降低 min_child_samples 让树能分得更细
  5. 对比各组结果，选最优

用法: python step12b_train_v4_tuned.py
"""
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

BASE_DIR    = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "features"
MODEL_DIR   = BASE_DIR / "model"

FEATURE_FILES = [
    FEATURE_DIR / "www.zstzpt.com.parquet",
    FEATURE_DIR / "data.zstzpt.com.parquet",
]

LABEL_COL = "is_attack"
NON_FEATURE_COLS = [LABEL_COL, "attack_type", "ip", "path", "source"]
LEAK_FEATURES = ["status_code", "is_4xx", "is_5xx", "is_444", "is_404", "is_403", "body_size"]
STAT_FEATURES = ["req_count_60s", "err_count_60s", "unique_url_60s"]

RANDOM_STATE = 42
TEST_SIZE    = 0.2

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
    path = df["path"].fillna("").astype(str)
    df["path_is_root"] = path.isin(["/", "/index.html", "/index.htm", "/index.php"]).astype(int)
    static_ext = r"\.(css|js|jpg|jpeg|png|gif|ico|woff2?|ttf|svg|map|eot|mp4|mp3|webp|webm|flv|swf)(?:\?|$)"
    df["is_static_file"] = path.str.contains(static_ext, case=False, regex=True, na=False).astype(int)
    hash_pattern = r"/[a-f0-9]{8,}\.\w{2,5}$|/[a-f0-9]{32,}(?:\.js|\.css)?$"
    df["has_hash_filename"] = path.str.contains(hash_pattern, case=False, regex=True, na=False).astype(int)
    api_prefix = r"^/api/|^/v[0-9]+/|^/graphql|^/webhook"
    df["is_known_api_prefix"] = path.str.contains(api_prefix, case=False, regex=True, na=False).astype(int)
    return df


def load_and_clean():
    dfs = []
    for f in FEATURE_FILES:
        if not f.exists():
            continue
        log(f"加载 {f.name} ...")
        df = pd.read_parquet(f)
        dfs.append(df)
        log(f"  -> {len(df):,} 行")
    df = pd.concat(dfs, ignore_index=True)
    log(f"合并: {len(df):,} 行")

    seo_count = (df["attack_type"] == "seo_spam").sum()
    df = df[df["attack_type"] != "seo_spam"].copy()
    log(f"移除 seo_spam: {seo_count:,} 条, 剩余 {len(df):,} 条")

    path_col = df["path"].fillna("").astype(str)
    for pattern, _ in MISLABEL_PATTERNS:
        mask = path_col.str.contains(pattern, case=False, regex=True, na=False)
        hit = mask & (df["attack_type"] == "normal") & (df[LABEL_COL] == 0)
        if hit.sum() > 0:
            df.loc[hit, LABEL_COL] = 1
            df.loc[hit, "attack_type"] = "mislabeled_fixed"

    df = add_request_features(df)

    y = df[LABEL_COL]
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    X = X.drop(columns=[c for c in LEAK_FEATURES if c in X.columns])
    X = X.drop(columns=[c for c in STAT_FEATURES if c in X.columns])
    X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    log(f"特征数: {len(X.columns)}, 正样本: {y.sum():,} ({y.mean():.2%})")
    return X, y


def evaluate(model, name, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    log(f"\n  {name}:")
    log(f"    AUC={auc:.4f}  F1={f1:.4f}  FPR={fpr:.2%}  FNR={fnr:.2%}")
    log(f"    TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

    # 阈值扫描（只看关键阈值）
    best_f1 = 0
    best_th = 0.5
    for th in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]:
        y_pred_th = (y_proba >= th).astype(int)
        tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_th).ravel()
        f1_2 = f1_score(y_test, y_pred_th, zero_division=0)
        fpr2 = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0
        fnr2 = fn2 / (fn2 + tp2) if (fn2 + tp2) > 0 else 0
        if f1_2 > best_f1:
            best_f1 = f1_2
            best_th = th
        log(f"    th={th:.2f}  F1={f1_2:.4f}  FPR={fpr2:.2%}  FNR={fnr2:.2%}  TP={tp2:,}  FP={fp2:,}")

    log(f"    >> Best threshold={best_th:.2f}  F1={best_f1:.4f}")

    return {
        "name": name, "auc": round(auc, 4), "f1": round(f1, 4),
        "fpr": round(fpr, 4), "fnr": round(fnr, 4),
        "best_threshold": best_th, "best_f1": round(best_f1, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def train_config(X_train, y_train, X_test, y_test, config_name, params):
    import lightgbm as lgb

    log(f"\n{'='*60}")
    log(f"  训练配置: {config_name}")
    log(f"  参数: {params}")
    log(f"{'='*60}")

    t0 = time.time()
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )
    elapsed = time.time() - t0
    log(f"  训练完成, 耗时 {elapsed:.1f}s, 最佳迭代 {model.best_iteration_}")

    metrics = evaluate(model, config_name, X_test, y_test)
    metrics["train_time_sec"] = round(elapsed, 1)
    metrics["best_iteration"] = int(model.best_iteration_)

    return model, metrics


def main():
    log("="*60)
    log("  v4 超参搜索: 3 组配置")
    log("="*60)

    X, y = load_and_clean()
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = neg / pos
    log(f"训练集: {len(X_train):,}  正: {pos:,}  负: {neg:,}  spw={spw:.2f} sqrt={spw**0.5:.2f}")

    base_params = {
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }

    configs = {
        # 配置 A: v3 原参数（已知会摆烂，作基线）
        "A_v3_baseline": {
            **base_params,
            "n_estimators": 1000,
            "learning_rate": 0.02,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 100,
            "min_gain_to_split": 0.5,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            "scale_pos_weight": spw ** 0.5,
        },
        # 配置 B: 放松正则 + 提高权重 + 更细的树
        "B_relaxed": {
            **base_params,
            "n_estimators": 2000,
            "learning_rate": 0.01,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 20,
            "min_gain_to_split": 0.0,  # 不限制分裂
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,         # 大幅放松
            "reg_lambda": 0.5,        # 大幅放松
            "scale_pos_weight": spw,   # 用原始权重而非 sqrt
        },
        # 配置 C: 极致放松 + is_unbalance + 更多迭代
        "C_aggressive": {
            **base_params,
            "n_estimators": 3000,
            "learning_rate": 0.01,
            "num_leaves": 127,
            "max_depth": 10,
            "min_child_samples": 10,
            "min_gain_to_split": 0.0,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.01,
            "reg_lambda": 0.1,
            "is_unbalance": True,      # 让 LightGBM 自动处理不平衡
        },
    }

    all_metrics = {}
    best_model = None
    best_f1 = 0
    best_name = ""

    for name, params in configs.items():
        model, metrics = train_config(X_train, y_train, X_test, y_test, name, params)
        all_metrics[name] = metrics

        # 特征重要性
        imp = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        log(f"\n  {name} 特征重要性 Top 5:")
        for _, row in imp.head(5).iterrows():
            log(f"    {row['feature']:<25s} {row['importance']:>8d}")

        if metrics["best_f1"] > best_f1:
            best_f1 = metrics["best_f1"]
            best_model = model
            best_name = name

    # 汇总对比
    log(f"\n{'='*60}")
    log(f"  超参搜索结果汇总")
    log(f"{'='*60}")
    log(f"  {'Config':<20s} {'AUC':>8s} {'F1':>8s} {'FPR':>8s} {'FNR':>8s} {'BestTh':>8s} {'BestF1':>8s} {'Iters':>8s}")
    log(f"  {'-'*80}")
    for name, m in all_metrics.items():
        log(f"  {name:<20s} {m['auc']:8.4f} {m['f1']:8.4f} {m['fpr']:8.2%} {m['fnr']:8.2%} {m['best_threshold']:8.2f} {m['best_f1']:8.4f} {m['best_iteration']:8d}")

    log(f"\n  最优配置: {best_name} (F1={best_f1:.4f})")

    # 保存最优模型
    if best_model is not None:
        import joblib
        model_path = MODEL_DIR / "lightgbm_model_v4.pkl"
        joblib.dump(best_model, model_path)
        log(f"  最优模型已保存: {model_path}")

        feat_path = MODEL_DIR / "feature_columns_v4.json"
        with open(feat_path, "w") as f:
            json.dump(feature_names, f)
        log(f"  特征列表已保存: {feat_path}")

        # 保存最优模型特征重要性
        imp = pd.DataFrame({
            "feature": feature_names,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        imp_path = MODEL_DIR / "lightgbm_feature_importance_v4.csv"
        imp.to_csv(imp_path, index=False)
        log(f"  特征重要性已保存: {imp_path}")

    report_path = MODEL_DIR / "training_report_v4.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    log(f"\n  评估报告已保存: {report_path}")
    log("\n全部完成!")


if __name__ == "__main__":
    main()
