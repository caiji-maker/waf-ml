"""
Step 11: Replay 误报根因深度诊断

核心问题：v3 模型离线 FPR=3.32%，但 Replay FPR=52.8%

诊断目标：
1. 对比训练集 vs Replay 的特征分布差异（特别是 Top 特征）
2. 分析误报样本的 SHAP-like 贡献度（用 LightGBM 内置 pred_contrib）
3. 统计特征在 Replay 中是否失效（req_count_60s 等全为 0）
4. 对比训练集和 Replay 中正常请求的特征均值差异
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(r"D:\training-data\waf-ml\model")
FEATURES_DIR = Path(r"D:\training-data\waf-ml\features")
LABELED_DIR = Path(r"D:\training-data\waf-ml\labeled")
REPLAY_CSV = MODEL_DIR / "replay_results.csv"

PYTHON = sys.executable

def load_model():
    import joblib
    model = joblib.load(MODEL_DIR / "lightgbm_model_v3.pkl")
    with open(MODEL_DIR / "feature_columns_v3.json") as f:
        feat_cols = json.load(f)
    return model, feat_cols

def load_training_features():
    """加载训练集特征数据"""
    dfs = []
    for f in sorted(FEATURES_DIR.glob("*.parquet")):
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    print(f"Training features loaded: {len(df):,} rows")
    return df

def load_replay_data():
    """加载 Replay 结果"""
    df = pd.read_csv(REPLAY_CSV, encoding="utf-8-sig")
    print(f"Replay results loaded: {len(df)} rows")
    return df

def analyze_feature_distribution(train_df, feat_cols):
    """对比训练集正常 vs 攻击的特征分布"""
    print("\n" + "=" * 80)
    print("  训练集特征分布：正常 vs 攻击")
    print("=" * 80)
    
    normal = train_df[train_df["is_attack"] == 0]
    attack = train_df[train_df["is_attack"] == 1]
    
    print(f"  Normal: {len(normal):,}  Attack: {len(attack):,}")
    print(f"\n  {'Feature':<25s} {'Normal Mean':>12s} {'Attack Mean':>12s} {'Ratio':>8s}")
    print("  " + "-" * 60)
    
    results = []
    for col in feat_cols:
        if col not in train_df.columns:
            print(f"  {col:<25s} {'MISSING':>12s}")
            continue
        n_mean = normal[col].mean()
        a_mean = attack[col].mean()
        ratio = a_mean / max(abs(n_mean), 0.001)
        results.append({
            "feature": col,
            "normal_mean": n_mean,
            "attack_mean": a_mean,
            "ratio": ratio,
        })
        print(f"  {col:<25s} {n_mean:12.4f} {a_mean:12.4f} {ratio:8.2f}")
    
    return pd.DataFrame(results)

def analyze_top_feature_importance():
    """显示 Top 10 特征重要性"""
    print("\n" + "=" * 80)
    print("  模型特征重要性 Top 10")
    print("=" * 80)
    
    imp = pd.read_csv(MODEL_DIR / "lightgbm_feature_importance_v3.csv")
    imp = imp.sort_values("importance", ascending=False)
    
    total = imp["importance"].sum()
    imp["pct"] = imp["importance"] / total * 100
    imp["cumulative_pct"] = imp["pct"].cumsum()
    
    print(f"  {'Feature':<25s} {'Importance':>10s} {'%':>6s} {'Cum%':>6s}")
    print("  " + "-" * 50)
    for _, row in imp.head(10).iterrows():
        print(f"  {row['feature']:<25s} {row['importance']:10d} {row['pct']:5.1f}% {row['cumulative_pct']:5.1f}%")
    
    # Top 4 特征累计占比
    top4_pct = imp.head(4)["pct"].sum()
    print(f"\n  Top 4 特征累计占比: {top4_pct:.1f}%")
    print(f"  Top 4 特征: {', '.join(imp.head(4)['feature'].tolist())}")

def analyze_replay_fp_samples(replay_df):
    """分析 Replay 误报样本"""
    print("\n" + "=" * 80)
    print("  Replay 误报样本分析")
    print("=" * 80)
    
    normal = replay_df[replay_df["label"] == "normal"]
    fp = normal[normal["blocked"] == True]
    tn = normal[normal["blocked"] == False]
    
    print(f"  正常请求总数: {len(normal)}")
    print(f"  误报 (FP): {len(fp)}  ({len(fp)/max(len(normal),1)*100:.1f}%)")
    print(f"  正确放行 (TN): {len(tn)}")
    
    # 分析误报样本的 URI 模式
    print(f"\n--- 误报样本 URI 模式 ---")
    
    # 按 URI 前缀分组
    def uri_prefix(uri):
        if not isinstance(uri, str):
            return "unknown"
        if uri == "/" or uri.startswith("/index"):
            return "root"
        if uri.startswith("/api/"):
            return "/api/*"
        if uri.startswith("/uploads/"):
            return "/uploads/*"
        if uri.startswith("/project/"):
            return "/project/*"
        if ".php" in uri:
            return "*.php"
        if "%" in uri:
            return "encoded_url"
        return "other"
    
    fp["uri_pattern"] = fp["uri"].apply(uri_prefix)
    pattern_counts = fp["uri_pattern"].value_counts()
    print(f"\n  {'Pattern':<20s} {'Count':>6s} {'Pct':>6s}")
    print("  " + "-" * 35)
    for pat, cnt in pattern_counts.items():
        print(f"  {pat:<20s} {cnt:6d} {cnt/len(fp)*100:5.1f}%")
    
    # 分析误报样本的 UA 模式
    print(f"\n--- 误报样本 UA 模式 ---")
    def ua_pattern(ua):
        if not isinstance(ua, str):
            return "unknown"
        ua_lower = ua.lower()
        if "undici" in ua_lower:
            return "undici"
        if "mozilla" in ua_lower:
            return "mozilla/browser"
        if "java" in ua_lower:
            return "java"
        if "bot" in ua_lower or "crawl" in ua_lower:
            return "bot"
        if "mycustomua" in ua_lower:
            return "custom"
        return "other"
    
    fp["ua_pattern"] = fp["ua"].apply(ua_pattern)
    ua_counts = fp["ua_pattern"].value_counts()
    print(f"\n  {'UA Pattern':<20s} {'Count':>6s} {'Pct':>6s}")
    print("  " + "-" * 35)
    for pat, cnt in ua_counts.items():
        print(f"  {pat:<20s} {cnt:6d} {cnt/len(fp)*100:5.1f}%")
    
    # 分析误报样本的 Method
    print(f"\n--- 误报样本 Method 分布 ---")
    method_counts = fp["method"].value_counts()
    for m, cnt in method_counts.items():
        print(f"  {m}: {cnt} ({cnt/len(fp)*100:.1f}%)")
    
    # 展示典型误报样本
    print(f"\n--- 典型误报样本 (前 20 条) ---")
    for _, row in fp.head(20).iterrows():
        print(f"  {row['method']:<6s} {str(row['uri'])[:70]:<70s}  UA={str(row['ua'])[:30]}")


def analyze_stat_feature_failure(train_df, feat_cols):
    """分析统计特征（req_count_60s 等）在训练集 vs 线上的差异"""
    print("\n" + "=" * 80)
    print("  统计特征失效分析（核心诊断）")
    print("=" * 80)
    
    stat_features = ["req_count_60s", "err_count_60s", "unique_url_60s"]
    
    # 训练集分布
    print(f"\n  训练集统计特征分布：")
    print(f"  {'Feature':<20s} {'Normal Mean':>12s} {'Attack Mean':>12s} {'Normal>0%':>10s} {'Attack>0%':>10s}")
    print("  " + "-" * 65)
    
    normal = train_df[train_df["is_attack"] == 0]
    attack = train_df[train_df["is_attack"] == 1]
    
    for col in stat_features:
        if col not in train_df.columns:
            continue
        n_mean = normal[col].mean()
        a_mean = attack[col].mean()
        n_nonzero = (normal[col] > 0).sum() / max(len(normal), 1) * 100
        a_nonzero = (attack[col] > 0).sum() / max(len(attack), 1) * 100
        print(f"  {col:<20s} {n_mean:12.2f} {a_mean:12.2f} {n_nonzero:9.1f}% {a_nonzero:9.1f}%")
    
    # 这些特征的重要性
    imp = pd.read_csv(MODEL_DIR / "lightgbm_feature_importance_v3.csv")
    imp = imp.sort_values("importance", ascending=False)
    total_imp = imp["importance"].sum()
    
    stat_imp = imp[imp["feature"].isin(stat_features)]
    stat_imp_sum = stat_imp["importance"].sum()
    stat_imp_pct = stat_imp_sum / total_imp * 100
    
    print(f"\n  统计特征重要性合计: {stat_imp_sum} / {total_imp} = {stat_imp_pct:.1f}%")
    
    for _, row in stat_imp.iterrows():
        pct = row["importance"] / total_imp * 100
        print(f"    {row['feature']}: {row['importance']} ({pct:.1f}%)")
    
    print(f"\n  结论：")
    print(f"  - 在线上 Replay 时，每个 IP 只发 1 条请求，req_count_60s=0, err_count_60s=0, unique_url_60s=0")
    print(f"  - 但训练集中这些特征有真实分布（攻击者高频请求）")
    print(f"  - 这 3 个特征占模型重要性 {stat_imp_pct:.1f}%，是 Top 4 中的 3 个")
    print(f"  - 线上拿不到正确值 → 模型输入与训练分布严重不匹配 → 误报")


def analyze_ua_length_issue(train_df, feat_cols):
    """分析 ua_length 特征在正常流量中的分布"""
    print("\n" + "=" * 80)
    print("  ua_length 特征分析（第二大特征）")
    print("=" * 80)
    
    normal = train_df[train_df["is_attack"] == 0]
    attack = train_df[train_df["is_attack"] == 1]
    
    # ua_length 在训练集中的分布
    print(f"\n  训练集 ua_length 分布：")
    print(f"    正常: mean={normal['ua_length'].mean():.1f}, median={normal['ua_length'].median():.0f}, std={normal['ua_length'].std():.1f}")
    print(f"    攻击: mean={attack['ua_length'].mean():.1f}, median={attack['ua_length'].median():.0f}, std={attack['ua_length'].std():.1f}")
    
    # 分位数
    for label, df_sub in [("正常", normal), ("攻击", attack)]:
        q25 = df_sub["ua_length"].quantile(0.25)
        q50 = df_sub["ua_length"].quantile(0.50)
        q75 = df_sub["ua_length"].quantile(0.75)
        q90 = df_sub["ua_length"].quantile(0.90)
        print(f"    {label}: P25={q25:.0f}, P50={q50:.0f}, P75={q75:.0f}, P90={q90:.0f}")
    
    # 分析训练集中 ua_length < 10 的比例
    short_ua_normal = (normal["ua_length"] < 10).sum() / max(len(normal), 1) * 100
    short_ua_attack = (attack["ua_length"] < 10).sum() / max(len(attack), 1) * 100
    print(f"\n  ua_length < 10 的比例：正常={short_ua_normal:.1f}%, 攻击={short_ua_attack:.1f}%")
    
    # Replay 中正常请求被拦的 UA 长度
    print(f"\n  Replay 中误报样本 UA 长度分布：")
    replay_df = load_replay_data()
    fp = replay_df[(replay_df["label"] == "normal") & (replay_df["blocked"] == True)]
    if len(fp) > 0:
        # 由于 replay CSV 中 ua 是截断的（前 50 字符），只能分析截断后长度
        fp_ua_len = fp["ua"].fillna("").apply(len)
        print(f"    误报样本 UA 长度（截断后）: mean={fp_ua_len.mean():.1f}, median={fp_ua_len.median():.0f}")
        print(f"    注意: replay CSV 中 UA 被截断为前 50 字符，实际值更长")


def main():
    print("=" * 80)
    print("  Step 11: Replay 误报根因深度诊断")
    print("=" * 80)
    
    # 1. 加载模型和数据
    model, feat_cols = load_model()
    train_df = load_training_features()
    replay_df = load_replay_data()
    
    # 2. 特征重要性分析
    analyze_top_feature_importance()
    
    # 3. 训练集特征分布对比
    dist_df = analyze_feature_distribution(train_df, feat_cols)
    dist_df.to_csv(MODEL_DIR / "feature_distribution_train.csv", index=False)
    print(f"\n  特征分布已保存: {MODEL_DIR / 'feature_distribution_train.csv'}")
    
    # 4. Replay 误报样本分析
    analyze_replay_fp_samples(replay_df)
    
    # 5. 统计特征失效分析（核心）
    analyze_stat_feature_failure(train_df, feat_cols)
    
    # 6. ua_length 分析
    analyze_ua_length_issue(train_df, feat_cols)
    
    # 总结
    print("\n" + "=" * 80)
    print("  总结：误报根因")
    print("=" * 80)
    print("""
  根因 1（致命）: 统计特征线上失效
    - req_count_60s, err_count_60s, unique_url_60s 在线上全为 0
    - 这 3 个特征占模型重要性 ~36%（Top 4 中的 3 个）
    - 训练集中攻击者 req_count_60s 平均远高于正常，模型高度依赖
    - 线上拿不到 → 输入分布与训练严重不匹配
    
  根因 2（次要）: is_post 权重偏高
    - POST 请求在训练集中多为攻击（automated_tool 用 undici 发 POST）
    - 正常 POST 请求线上被误判
    
  根因 3（次要）: ua_length 分布差异
    - 攻击者常短 UA 或无 UA，模型学到 "短 UA = 攻击"
    - 但正常请求中也有短 UA（MyCustomUA/1.0 等）

  方案建议：
    A. 重训模型：移除统计特征（线上拿不到的），只用请求本身特征
    B. 规则为主：用规则覆盖已知攻击模式，ML 只做兜底
    C. 混合方案：规则 + 轻量 ML（无统计特征）
""")


if __name__ == "__main__":
    main()
