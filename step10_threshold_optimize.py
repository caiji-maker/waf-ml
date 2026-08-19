"""
Step 10: 基于 Replay 样本的阈值寻优

从 labeled parquet 中加载与 step8 相同的 1000 条抽样，
用模型重新打分，在不同阈值下统计 FPR/FNR，
找到 FPR < 5% 的最优阈值。

同时输出：
- 阈值-FPR-FNR 曲线数据
- 不同阈值下各攻击类型的拦截率
- 推荐阈值
"""

import sys
import json
import random
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# ── 常量 ──────────────────────────────────────────────────────

PROJECT_DIR = Path(r"D:\training-data\waf-ml")
DATA_DIR = PROJECT_DIR / "labeled"
FEATURES_DIR = PROJECT_DIR / "features"
MODEL_DIR = PROJECT_DIR / "model"

SEED = 42
N_ATTACK = 500
N_NORMAL = 500

# 分析的阈值范围
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_model():
    """加载模型和特征列"""
    import joblib

    model_path = MODEL_DIR / "lightgbm_model_v3.pkl"
    feat_path = MODEL_DIR / "feature_columns_v3.json"

    model = joblib.load(model_path)
    with open(feat_path, "r") as f:
        feature_cols = json.load(f)

    print(f"Model loaded: {model_path.name}")
    print(f"Features: {len(feature_cols)}")
    return model, feature_cols


def load_and_sample():
    """与 step8 相同的抽样逻辑"""
    print("Loading labeled data...")
    dfs = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows: {len(df):,}")

    # 列别名
    if "path" not in df.columns and "uri" in df.columns:
        df["path"] = df["uri"]
    if "ua" not in df.columns and "user_agent" in df.columns:
        df["ua"] = df["user_agent"]

    # 排除 seo_spam
    if "attack_type" in df.columns:
        seo_mask = df["attack_type"] == "seo_spam"
        print(f"  Excluding {seo_mask.sum():,} seo_spam rows")
        df = df[~seo_mask].copy()

    attack_df = df[df["is_attack"] == 1].copy()
    normal_df = df[df["is_attack"] == 0].copy()
    print(f"  Attack rows (excl. seo_spam): {len(attack_df):,}")
    print(f"  Normal rows: {len(normal_df):,}")

    random.seed(SEED)
    n_attack = min(N_ATTACK, len(attack_df))
    n_normal = min(N_NORMAL, len(normal_df))

    attack_sample = attack_df.sample(n=n_attack, random_state=SEED)
    normal_sample = normal_df.sample(n=n_normal, random_state=SEED)

    attack_sample = attack_sample.copy()
    attack_sample["_label"] = "attack"
    normal_sample = normal_sample.copy()
    normal_sample["_label"] = "normal"

    return pd.concat([attack_sample, normal_sample], ignore_index=True)


def extract_features_for_row(row, feature_cols):
    """与 ml_detect.py 的 _extract_features 完全一致的特征提取"""
    import re

    SPECIAL_CHARS = set('?&=%\'";<>|(){}[]^~`!@#$')

    BOT_UA_RE = re.compile(
        r'bot|crawler|spider|scan|fetch|python|curl|wget|scrapy|hutool|undici|'
        r'go-http|java/|httpclient|masscan|zgrab|zmeu',
        re.IGNORECASE,
    )
    LEGIT_BOT_RE = re.compile(
        r'bingbot|googlebot|baiduspider|bytespider|yandexbot|duckduckbot',
        re.IGNORECASE,
    )
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
    SEO_SPAM_RE = re.compile(r'/\?and/', re.IGNORECASE)

    uri = str(row.get("path", row.get("uri", "/")))
    ua = str(row.get("ua", row.get("user_agent", "")))
    method = str(row.get("method", "GET"))
    referer = ""  # 日志里没有 referer，默认空

    # URL 特征
    url_length = len(uri)
    num_special_chars = sum(1 for c in uri if c in SPECIAL_CHARS)
    num_digits = sum(1 for c in uri if c.isdigit())
    num_dots = uri.count('.')
    num_slashes = uri.count('/')
    num_params = uri.count('&') + uri.count('=')
    has_and_pattern = 1 if SEO_SPAM_RE.search(uri) else 0
    encoded_chars = uri.count('%')
    path_depth = num_slashes - (1 if uri.startswith('/') else 0)
    digit_ratio = num_digits / max(url_length, 1)
    special_char_ratio = num_special_chars / max(url_length, 1)

    # UA 特征
    ua_length = len(ua) if ua and ua != "nan" else 0
    is_bot_ua = 1 if BOT_UA_RE.search(ua) else 0
    is_legitimate_bot = 1 if LEGIT_BOT_RE.search(ua) else 0
    is_empty_ua = 1 if (not ua or ua == "nan" or ua == "-" or ua == "") else 0

    # 请求属性
    method_map = {"GET": 0, "POST": 1, "OPTIONS": 2, "HEAD": 3, "PUT": 4, "DELETE": 5}
    method_code = method_map.get(method, 6)
    is_get = 1 if method == "GET" else 0
    is_post = 1 if method == "POST" else 0
    is_options = 1 if method == "OPTIONS" else 0
    is_head = 1 if method == "HEAD" else 0
    is_empty_referer = 1  # 日志里没有 referer，总是 1

    # 统计特征（离线分析无法获取实时统计，用 0 占位）
    req_count_60s = 0
    err_count_60s = 0
    unique_url_60s = 0

    # 路径语义
    path_is_root = 1 if uri in ("/", "/index.html", "/index.htm", "/index.php") else 0
    is_static_file = 1 if STATIC_EXT_RE.search(uri) else 0
    has_hash_filename = 1 if HASH_FILE_RE.search(uri) else 0
    is_known_api_prefix = 1 if API_PREFIX_RE.search(uri) else 0

    features = {
        "url_length": url_length,
        "num_special_chars": num_special_chars,
        "num_digits": num_digits,
        "num_dots": num_dots,
        "num_slashes": num_slashes,
        "num_params": num_params,
        "has_and_pattern": has_and_pattern,
        "encoded_chars": encoded_chars,
        "path_depth": path_depth,
        "digit_ratio": digit_ratio,
        "special_char_ratio": special_char_ratio,
        "ua_length": ua_length,
        "is_bot_ua": is_bot_ua,
        "is_legitimate_bot": is_legitimate_bot,
        "is_empty_ua": is_empty_ua,
        "method_code": method_code,
        "is_get": is_get,
        "is_post": is_post,
        "is_options": is_options,
        "is_head": is_head,
        "is_empty_referer": is_empty_referer,
        "req_count_60s": req_count_60s,
        "err_count_60s": err_count_60s,
        "unique_url_60s": unique_url_60s,
        "path_is_root": path_is_root,
        "is_static_file": is_static_file,
        "has_hash_filename": has_hash_filename,
        "is_known_api_prefix": is_known_api_prefix,
    }

    return {col: float(features.get(col, 0)) for col in feature_cols}


def main():
    print("=" * 60)
    print("  Step 10: 阈值寻优分析")
    print("=" * 60)

    # 1. 加载模型
    model, feature_cols = load_model()

    # 2. 加载抽样（与 step8 相同）
    samples = load_and_sample()
    print(f"\n  Sampled {len(samples)} requests")

    # 3. 逐条提取特征并打分
    print("\nExtracting features and scoring...")
    t0 = time.time()
    scores = []
    for _, row in samples.iterrows():
        feat = extract_features_for_row(row, feature_cols)
        X = np.array([[feat[col] for col in feature_cols]], dtype=np.float32)
        proba = model.predict_proba(X)[0, 1]
        scores.append(float(proba))

    samples["_score"] = scores
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # 4. 分数分布
    attack_scores = samples[samples["_label"] == "attack"]["_score"]
    normal_scores = samples[samples["_label"] == "normal"]["_score"]

    print(f"\n--- 分数分布 ---")
    print(f"  Attack scores: mean={attack_scores.mean():.4f}, median={attack_scores.median():.4f}, std={attack_scores.std():.4f}")
    print(f"    min={attack_scores.min():.4f}, max={attack_scores.max():.4f}")
    print(f"    Q25={attack_scores.quantile(0.25):.4f}, Q75={attack_scores.quantile(0.75):.4f}")
    print(f"  Normal scores: mean={normal_scores.mean():.4f}, median={normal_scores.median():.4f}, std={normal_scores.std():.4f}")
    print(f"    min={normal_scores.min():.4f}, max={normal_scores.max():.4f}")
    print(f"    Q25={normal_scores.quantile(0.25):.4f}, Q75={normal_scores.quantile(0.75):.4f}")

    # 5. 阈值分析
    print(f"\n--- 阈值分析 ---")
    print(f"  {'Threshold':>10}  {'FPR':>8}  {'FNR':>8}  {'Recall':>8}  {'Precision':>10}  {'FP':>4}  {'FN':>4}  {'TP':>4}  {'TN':>4}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}")

    results = []
    for th in THRESHOLDS:
        tp = int(sum((samples["_label"] == "attack") & (samples["_score"] >= th)))
        fn = int(sum((samples["_label"] == "attack") & (samples["_score"] < th)))
        fp = int(sum((samples["_label"] == "normal") & (samples["_score"] >= th)))
        tn = int(sum((samples["_label"] == "normal") & (samples["_score"] < th)))

        total_attack = tp + fn
        total_normal = fp + tn
        fpr = fp / max(total_normal, 1)
        fnr = fn / max(total_attack, 1)
        recall = tp / max(total_attack, 1)
        precision = tp / max(tp + fp, 1)

        results.append({
            "threshold": th,
            "fpr": fpr,
            "fnr": fnr,
            "recall": recall,
            "precision": precision,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        })

        marker = ""
        if fpr < 0.05:
            marker = " <-- FPR < 5%"
        print(f"  {th:>10.2f}  {fpr:>8.2%}  {fnr:>8.2%}  {recall:>8.2%}  {precision:>10.2%}  {fp:>4d}  {fn:>4d}  {tp:>4d}  {tn:>4d}{marker}")

    # 6. 找最优阈值
    # 优先级：FPR < 5%，在满足的前提下 FNR 最低
    valid = [r for r in results if r["fpr"] < 0.05]
    if valid:
        best = min(valid, key=lambda x: x["fnr"])
        print(f"\n=== 推荐阈值: {best['threshold']:.2f} ===")
        print(f"  FPR = {best['fpr']:.2%} (红线 < 5%)")
        print(f"  FNR = {best['fnr']:.2%}")
        print(f"  Recall = {best['recall']:.2%}")
        print(f"  Precision = {best['precision']:.2%}")
        print(f"  TP={best['tp']}, FN={best['fn']}, FP={best['fp']}, TN={best['tn']}")
    else:
        print("\n=== 没有阈值能满足 FPR < 5% ===")
        # 退而求其次：找 FPR 最低的
        best = min(results, key=lambda x: x["fpr"])
        print(f"  最低 FPR 阈值: {best['threshold']:.2f}")
        print(f"  FPR = {best['fpr']:.2%}")
        print(f"  FNR = {best['fnr']:.2%}")
        print(f"  Recall = {best['recall']:.2%}")
        print(f"  Precision = {best['precision']:.2%}")

    # 7. 各攻击类型在推荐阈值下的拦截率
    if valid:
        th = best["threshold"]
    else:
        th = best["threshold"]

    print(f"\n--- 推荐阈值 {th:.2f} 下各攻击类型拦截率 ---")
    attack_samples = samples[samples["_label"] == "attack"]
    by_type = defaultdict(list)
    for _, row in attack_samples.iterrows():
        at = str(row.get("attack_type", "unknown"))
        if at == "nan":
            at = "unknown"
        by_type[at].append(row["_score"])

    for at in sorted(by_type.keys()):
        scs = by_type[at]
        blocked = sum(1 for s in scs if s >= th)
        total = len(scs)
        rate = blocked / max(total, 1) * 100
        print(f"  {at:25s}  {blocked:4d}/{total:4d}  ({rate:5.1f}%)")

    # 8. 保存详细数据
    out_path = MODEL_DIR / "threshold_analysis_replay.csv"
    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nThreshold analysis saved to: {out_path}")

    # 保存带分数的样本
    samples_out = samples[["uri", "method", "ua", "label", "attack_type", "_score"]].copy()
    samples_out.to_csv(MODEL_DIR / "replay_scores.csv", index=False, encoding="utf-8-sig")
    print(f"Scored samples saved to: {MODEL_DIR / 'replay_scores.csv'}")


if __name__ == "__main__":
    main()
