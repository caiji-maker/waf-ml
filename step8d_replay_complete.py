"""
Step 8d: 离线 Replay 验证（用 features parquet 补全特征）

从 features parquet 加载 24 个旧特征 + 从 labeled parquet 补上 4 个路径语义特征。
统计特征使用训练时的真实值，最接近模型实际能力。
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import re

FEATURE_DIR = Path(r"D:\training-data\waf-ml\features")
LABELED_DIR = Path(r"D:\training-data\waf-ml\labeled")
MODEL_DIR = Path(r"D:\training-data\waf-ml\model")

FEATURE_COLS = json.loads(
    (MODEL_DIR / "feature_columns_v3.json").read_text(encoding="utf-8")
)

# ── 路径语义特征提取 ──

STATIC_EXT_RE = re.compile(
    r'\.(css|js|jpg|jpeg|png|gif|ico|woff2?|ttf|svg|map|eot|mp4|mp3|webp|webm|flv|swf)(?:\?|$)',
    re.IGNORECASE,
)
HASH_FILE_RE = re.compile(
    r'/[a-f0-9]{8,}\.\w{2,5}$|/[a-f0-9]{32,}(?:\.js|\.css)?$',
    re.IGNORECASE,
)
API_PREFIX_RE = re.compile(
    r'^/api/|^/v[0-9]+/|^/graphql|^/webhook',
    re.IGNORECASE,
)


def compute_path_features(uri):
    path_is_root = 1 if uri in ("/", "/index.html", "/index.htm", "/index.php") else 0
    is_static_file = 1 if STATIC_EXT_RE.search(uri) else 0
    has_hash_filename = 1 if HASH_FILE_RE.search(uri) else 0
    is_known_api_prefix = 1 if API_PREFIX_RE.search(uri) else 0
    return path_is_root, is_static_file, has_hash_filename, is_known_api_prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.40)
    args = parser.parse_args()

    print("Step 8d: 离线 Replay 验证（补全 28 特征）")
    print(f"  Threshold: {args.threshold}")

    # 1. 加载 features
    print("\nLoading features parquet...")
    feat_dfs = []
    for f in sorted(FEATURE_DIR.glob("*.parquet")):
        feat_dfs.append(pd.read_parquet(f))
    feat_df = pd.concat(feat_dfs, ignore_index=True)
    print(f"  Features: {len(feat_df):,} rows, columns: {len(feat_df.columns)}")

    # 2. 加载 labeled（需要 is_attack + attack_type + path）
    print("\nLoading labeled parquet...")
    lab_dfs = []
    for f in sorted(LABELED_DIR.glob("*.parquet")):
        lab_dfs.append(pd.read_parquet(f, columns=["is_attack", "attack_type", "path"]))
    lab_df = pd.concat(lab_dfs, ignore_index=True)
    print(f"  Labels: {len(lab_df):,} rows")

    # 对齐
    assert len(feat_df) == len(lab_df), f"Length mismatch: {len(feat_df)} vs {len(lab_df)}"

    # 合并标签
    feat_df["is_attack"] = lab_df["is_attack"].values
    feat_df["attack_type"] = lab_df["attack_type"].values

    # 排除 seo_spam
    mask = feat_df["attack_type"] != "seo_spam"
    n_excluded = (~mask).sum()
    print(f"  Excluding seo_spam: {n_excluded:,} rows")
    df = feat_df[mask].copy()

    # 3. 补全缺失的 4 个路径语义特征
    print("\nComputing path semantic features...")
    paths = lab_df["path"].values[mask.values]
    path_features = [compute_path_features(str(p)) for p in paths]
    df["path_is_root"] = [f[0] for f in path_features]
    df["is_static_file"] = [f[1] for f in path_features]
    df["has_hash_filename"] = [f[2] for f in path_features]
    df["is_known_api_prefix"] = [f[3] for f in path_features]

    # 验证所有特征都存在
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  ERROR: Still missing: {missing}")
        sys.exit(1)
    print(f"  All {len(FEATURE_COLS)} features ready!")

    # 4. 准备数据
    X = df[FEATURE_COLS].astype(np.float32).values
    y_true = df["is_attack"].values
    attack_types = df["attack_type"].values

    # 5. 预测
    print("\nPredicting...")
    import joblib
    model = joblib.load(MODEL_DIR / "lightgbm_model_v3.pkl")
    y_proba = model.predict_proba(X)[:, 1]

    threshold = args.threshold
    y_pred = (y_proba >= threshold).astype(int)

    # 6. 统计
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    total_attack = tp + fn
    total_normal = fp + tn

    print("\n" + "=" * 60)
    print("  离线 Replay 验证报告（完整 28 特征 + 真实统计值）")
    print("=" * 60)

    print(f"\n--- 总体效果 (threshold={threshold}) ---")
    print(f"  攻击样本: {total_attack:,}")
    print(f"    被拦截 (TP): {tp:,}")
    print(f"    漏过   (FN): {fn:,}")
    print(f"    拦截率 (Recall): {tp/max(total_attack,1)*100:.2f}%")
    print(f"    漏报率 (FNR):   {fn/max(total_attack,1)*100:.2f}%")

    print(f"\n  正常样本: {total_normal:,}")
    print(f"    被误拦 (FP): {fp:,}")
    print(f"    正确放行 (TN): {tn:,}")
    print(f"    误报率 (FPR):   {fp/max(total_normal,1)*100:.2f}%")

    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"    精确率 (Precision): {precision*100:.2f}%")

    # 按攻击类型
    print(f"\n--- 按攻击类型拦截率 ---")
    by_type = defaultdict(lambda: {"tp": 0, "total": 0, "fp": 0, "normal_total": 0})
    for i in range(len(df)):
        at = str(attack_types[i]) if attack_types[i] and str(attack_types[i]) != "nan" else "normal"
        if y_true[i] == 1:
            by_type[at]["total"] += 1
            if y_pred[i] == 1:
                by_type[at]["tp"] += 1
        else:
            by_type[at]["normal_total"] += 1
            if y_pred[i] == 1:
                by_type[at]["fp"] += 1

    for at in sorted(by_type.keys()):
        info = by_type[at]
        if info["total"] > 0:
            rate = info["tp"] / max(info["total"], 1) * 100
            print(f"  {at:25s}  {info['tp']:>6,}/{info['total']:>6,}  blocked ({rate:5.1f}%)")
        else:
            fpr = info["fp"] / max(info["normal_total"], 1) * 100
            print(f"  {at:25s}  FP={info['fp']:>6,}/{info['normal_total']:>6,}  (FPR={fpr:.1f}%)")

    # 阈值扫描
    print(f"\n--- 阈值扫描 ---")
    for t in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]:
        pred_t = (y_proba >= t).astype(int)
        tp_t = int(((pred_t == 1) & (y_true == 1)).sum())
        fn_t = int(((pred_t == 0) & (y_true == 1)).sum())
        fp_t = int(((pred_t == 1) & (y_true == 0)).sum())
        tn_t = int(((pred_t == 0) & (y_true == 0)).sum())
        fpr_t = fp_t / max(fp_t + tn_t, 1) * 100
        fnr_t = fn_t / max(tp_t + fn_t, 1) * 100
        print(f"  threshold={t:.2f}  FPR={fpr_t:5.2f}%  FNR={fnr_t:5.2f}%  TP={tp_t:>6,}  FP={fp_t:>6,}")

    print("\n" + "=" * 60)

    # 保存
    out_path = MODEL_DIR / "replay_offline_v3_full.csv"
    result_df = pd.DataFrame({
        "is_attack": y_true,
        "attack_type": attack_types,
        "attack_proba": y_proba,
        "predicted": y_pred,
    })
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
