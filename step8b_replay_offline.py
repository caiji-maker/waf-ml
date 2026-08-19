"""
Step 8b: 离线 Replay 验证（直接用模型预测）

绕过 WAF 的在线重放（统计特征会因重放环境失真），
直接从标注数据提取特征，用模型预测，统计 FPR/FNR。

这样测出来的是纯模型能力，不受网络/时序环境影响。
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(r"D:\training-data\waf-ml\features")
MODEL_DIR = Path(r"D:\training-data\waf-ml\model")

# 28 个最终特征列
FEATURE_COLS = json.loads(
    (MODEL_DIR / "feature_columns_v3.json").read_text(encoding="utf-8")
)

# 排除 seo_spam（训练时已移除，不参与评估）
EXCLUDE_TYPES = {"seo_spam"}


def load_model():
    import joblib
    model = joblib.load(MODEL_DIR / "lightgbm_model_v3.pkl")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.40, help="判定阈值")
    parser.add_argument("--sample", type=int, default=0, help="抽样数，0=全量")
    args = parser.parse_args()

    print("Step 8b: 离线 Replay 验证（直接用模型预测）")
    print(f"  Threshold: {args.threshold}")

    # 1. 加载特征数据
    print("\nLoading features...")
    dfs = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(df):,} rows")

    # 2. 加载标注信息（需要 is_attack + attack_type）
    labeled_dir = Path(r"D:\training-data\waf-ml\labeled")
    label_dfs = []
    for f in sorted(labeled_dir.glob("*.parquet")):
        label_dfs.append(pd.read_parquet(f, columns=["is_attack", "attack_type"]))
    labels = pd.concat(label_dfs, ignore_index=True)
    print(f"  Labels: {len(labels):,} rows")

    # 合并标签
    if len(df) == len(labels):
        df["is_attack"] = labels["is_attack"].values
        df["attack_type"] = labels["attack_type"].values
    else:
        print("  WARNING: Feature/label length mismatch, doing merge by index")
        df = df.join(labels)

    # 排除 seo_spam
    if "attack_type" in df.columns:
        mask = df["attack_type"] != "seo_spam"
        print(f"  Excluding seo_spam: {(~mask).sum():,} rows")
        df = df[mask].copy()

    # 3. 抽样（可选）
    if args.sample > 0 and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42)
        print(f"  Sampled: {len(df):,} rows")

    # 4. 提取特征
    print(f"\nPreparing features ({len(FEATURE_COLS)} cols)...")
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        print(f"  WARNING: Missing feature columns: {missing_cols}")
        # 尝试从数据中构造
        for c in missing_cols:
            if c not in df.columns:
                df[c] = 0

    X = df[FEATURE_COLS].astype(np.float32).values
    y_true = df["is_attack"].values
    attack_types = df["attack_type"].values if "attack_type" in df.columns else np.array([""] * len(df))

    # 5. 加载模型并预测
    print("\nLoading model and predicting...")
    model = load_model()
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= args.threshold).astype(int)

    # 6. 统计
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    total_attack = tp + fn
    total_normal = fp + tn

    print("\n" + "=" * 60)
    print("  离线 Replay 验证报告（纯模型能力）")
    print("=" * 60)

    print(f"\n--- 总体效果 (threshold={args.threshold}) ---")
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

    # 7. 按攻击类型拆分
    print(f"\n--- 按攻击类型拦截率 ---")
    by_type = {}
    for i in range(len(df)):
        at = str(attack_types[i]) if attack_types[i] and str(attack_types[i]) != "nan" else "normal"
        if at not in by_type:
            by_type[at] = {"tp": 0, "total": 0, "fp": 0, "normal_total": 0}
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

    # 8. 不同阈值的扫描
    print(f"\n--- 阈值扫描 ---")
    for t in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        pred_t = (y_proba >= t).astype(int)
        tp_t = int(((pred_t == 1) & (y_true == 1)).sum())
        fn_t = int(((pred_t == 0) & (y_true == 1)).sum())
        fp_t = int(((pred_t == 1) & (y_true == 0)).sum())
        tn_t = int(((pred_t == 0) & (y_true == 0)).sum())
        fpr_t = fp_t / max(fp_t + tn_t, 1) * 100
        fnr_t = fn_t / max(tp_t + fn_t, 1) * 100
        print(f"  threshold={t:.2f}  FPR={fpr_t:5.2f}%  FNR={fnr_t:5.2f}%  TP={tp_t:>6,}  FP={fp_t:>6,}")

    print("\n" + "=" * 60)

    # 9. 保存概率分布
    out_path = MODEL_DIR / "replay_offline_results.csv"
    result_df = df[["is_attack"]].copy() if "is_attack" in df.columns else pd.DataFrame()
    result_df["attack_type"] = attack_types
    result_df["attack_proba"] = y_proba
    result_df["predicted"] = y_pred
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
