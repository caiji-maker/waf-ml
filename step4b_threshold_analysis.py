#!/usr/bin/env python3
"""
Step 4b: 阈值调优 + 误报样本深度分析

1. 加载已保存的 LightGBM 模型
2. 在测试集上扫描不同阈值，输出 FPR/FNR/F1 表
3. 选定阈值后，提取误报样本，对比特征均值
4. 从原始 Parquet 加载完整数据（含 path/attack_type），分析误报根源
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# ── 配置 ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "features"
MODEL_DIR   = BASE_DIR / "model"

FEATURE_FILES = [
    FEATURE_DIR / "www.zstzpt.com.parquet",
    FEATURE_DIR / "data.zstzpt.com.parquet",
]

LABEL_COL = "is_attack"
NON_FEATURE_COLS = [LABEL_COL, "attack_type", "ip", "path", "source"]
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 特征列
FEATURE_COLS = [
    "url_length", "num_special_chars", "num_digits", "num_dots", "num_slashes",
    "num_params", "has_and_pattern", "encoded_chars", "path_depth",
    "digit_ratio", "special_char_ratio",
    "ua_length", "is_bot_ua", "is_legitimate_bot", "is_empty_ua",
    "method_code", "is_get", "is_post", "is_options", "is_head",
    "status_code", "is_4xx", "is_5xx", "is_444", "is_404", "is_403",
    "body_size", "is_empty_referer",
    "req_count_60s", "err_count_60s", "unique_url_60s",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 加载完整数据（含原始列）────────────────────────────
def load_full_data():
    """加载完整 Parquet，保留所有列（包括 path/attack_type 等用于分析）"""
    dfs = []
    for f in FEATURE_FILES:
        if not f.exists():
            continue
        log(f"加载 {f.name} ...")
        df = pd.read_parquet(f)
        dfs.append(df)
        log(f"  -> {len(df):,} 行, {df.shape[1]} 列")

    df = pd.concat(dfs, ignore_index=True)
    log(f"合并完成: {len(df):,} 行")
    return df


def main():
    log("=" * 60)
    log("  阈值调优 + 误报样本深度分析")
    log("=" * 60)

    # 1. 加载模型
    model_path = MODEL_DIR / "lightgbm_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")
    model = joblib.load(model_path)
    log(f"已加载模型: {model_path}")

    # 2. 加载完整数据
    df = load_full_data()

    # 3. 分离特征与标签（和训练时完全一致的方式）
    y = df[LABEL_COL].values
    X = df[FEATURE_COLS].copy()
    X = X.fillna(0)

    # 4. 划分测试集（和训练时相同的 random_state + shuffle）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    # 同时对完整 df 做同样的划分，拿到测试集的原始列（path/attack_type 等）
    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    # 确保 X_test 和 df_test 的索引一致
    X_test = X_test.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True).values

    log(f"测试集: {len(X_test):,} 行")

    # 5. 预测概率
    log("计算预测概率 ...")
    y_proba = model.predict_proba(X_test)[:, 1]

    # ══════════════════════════════════════════════════════
    # 第一步：阈值扫描
    # ══════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  阈值扫描")
    log("=" * 80)
    log(f"{'阈值':<8} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FPR':<12} {'FNR':<12} {'FP数':<10} {'FN数':<10}")
    log("-" * 80)

    thresholds = np.arange(0.50, 1.00, 0.05)
    threshold_results = []

    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        acc  = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0

        log(f"{th:<8.2f} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {fpr:<12.2%} {fnr:<12.2%} {fp:<10,} {fn:<10,}")
        threshold_results.append({
            "threshold": round(th, 2),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "fp": int(fp),
            "fn": int(fn),
        })

    # 保存阈值表
    th_df = pd.DataFrame(threshold_results)
    th_path = MODEL_DIR / "threshold_analysis.csv"
    th_df.to_csv(th_path, index=False)
    log(f"\n阈值表已保存: {th_path}")

    # ══════════════════════════════════════════════════════
    # 第二步：选定阈值，提取误报样本并分析
    # ══════════════════════════════════════════════════════
    # 选几个阈值同时分析: 0.75, 0.85, 0.90
    for ANALYSIS_TH in [0.75, 0.85, 0.90]:
        log(f"\n{'='*80}")
        log(f"  误报分析 @ 阈值 {ANALYSIS_TH}")
        log(f"{'='*80}")

        y_pred_th = (y_proba >= ANALYSIS_TH).astype(int)

        # 误报: 真实=0, 预测=1
        fp_mask = (y_test == 0) & (y_pred_th == 1)
        # 漏报: 真实=1, 预测=0
        fn_mask = (y_test == 1) & (y_pred_th == 0)
        # 正确的正常: 真实=0, 预测=0
        tn_mask = (y_test == 0) & (y_pred_th == 0)

        fp_count = fp_mask.sum()
        fn_count = fn_mask.sum()
        tn_count = tn_mask.sum()
        log(f"误报(FP): {fp_count:,}  漏报(FN): {fn_count:,}  正确正常(TN): {tn_count:,}")

        if fp_count == 0:
            log("无误报样本，跳过")
            continue

        # ── 2a. 误报样本的 attack_type 分布 ──
        fp_attack_types = df_test.loc[fp_mask, "attack_type"].value_counts()
        log(f"\n误报样本的 attack_type 分布:")
        log(fp_attack_types.to_string())

        # ── 2b. 误报样本 vs 正确正常样本的特征均值对比 ──
        fp_features = X_test.loc[fp_mask]
        tn_features = X_test.loc[tn_mask]

        log(f"\n误报样本 vs 正确正常样本 — 特征均值对比 (Top 重要特征):")
        log(f"{'特征':<25} {'误报均值':<15} {'正确正常均值':<15} {'差异':<15}")
        log("-" * 70)

        compare_cols = [
            "req_count_60s", "body_size", "ua_length", "unique_url_60s",
            "url_length", "digit_ratio", "err_count_60s", "num_digits",
            "special_char_ratio", "num_slashes", "num_dots",
            "status_code", "is_404", "is_403", "is_4xx",
            "is_empty_referer", "is_bot_ua", "is_legitimate_bot",
            "has_and_pattern", "path_depth",
        ]
        for col in compare_cols:
            fp_mean = fp_features[col].mean()
            tn_mean = tn_features[col].mean()
            diff = fp_mean - tn_mean
            log(f"{col:<25} {fp_mean:<15.2f} {tn_mean:<15.2f} {diff:<+15.2f}")

        # ── 2c. 误报样本的 path 抽样（看原始 URL）──
        log(f"\n误报样本 path 抽样 (前30条):")
        fp_paths = df_test.loc[fp_mask, "path"].head(30).tolist()
        for i, p in enumerate(fp_paths):
            log(f"  {i+1:3d}. {p[:120]}")

        # ── 2d. 误报样本的 UA 抽样 ──
        # UA 不在特征列里，但从特征可以看 ua_length
        log(f"\n误报样本 ua_length 分布:")
        ua_dist = fp_features["ua_length"].describe()
        log(ua_dist.to_string())

        # ── 2e. 误报样本的 source 分布 ──
        log(f"\n误报样本 source 分布:")
        fp_sources = df_test.loc[fp_mask, "source"].value_counts()
        log(fp_sources.to_string())

        # ── 2f. 保存误报样本到 CSV（前1000条）──
        fp_sample = df_test.loc[fp_mask, ["path", "attack_type", "source", "ip", "is_attack"]].head(1000)
        fp_sample["predicted_prob"] = y_proba[fp_mask][:1000]
        fp_path = MODEL_DIR / f"false_positives_th{int(ANALYSIS_TH*100)}.csv"
        fp_sample.to_csv(fp_path, index=False)
        log(f"误报样本(前1000条)已保存: {fp_path}")

    # ══════════════════════════════════════════════════════
    # 第三步：漏报样本分析（看哪些攻击被漏掉了）
    # ══════════════════════════════════════════════════════
    ANALYSIS_TH = 0.85
    y_pred_th = (y_proba >= ANALYSIS_TH).astype(int)
    fn_mask = (y_test == 1) & (y_pred_th == 0)

    if fn_mask.sum() > 0:
        log(f"\n{'='*80}")
        log(f"  漏报分析 @ 阈值 {ANALYSIS_TH}")
        log(f"{'='*80}")
        log(f"漏报数: {fn_mask.sum():,}")

        fn_attack_types = df_test.loc[fn_mask, "attack_type"].value_counts()
        log(f"\n漏报样本的 attack_type 分布:")
        log(fn_attack_types.to_string())

        log(f"\n漏报样本 path 抽样 (前30条):")
        fn_paths = df_test.loc[fn_mask, "path"].head(30).tolist()
        for i, p in enumerate(fn_paths):
            log(f"  {i+1:3d}. {p[:120]}")

    log("\n分析完成!")


if __name__ == "__main__":
    main()
