"""
Step 8c: 离线 Replay 验证（从原始数据重新提取完整 28 特征）

之前 8b 用 features parquet 缺了 4 个 v3 新增特征（path_is_root 等），
导致模型输入偏差。本脚本从 labeled parquet 原始数据重新提取全部 28 特征。
"""

import sys
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

DATA_DIR = Path(r"D:\training-data\waf-ml\labeled")
MODEL_DIR = Path(r"D:\training-data\waf-ml\model")

FEATURE_COLS = json.loads(
    (MODEL_DIR / "feature_columns_v3.json").read_text(encoding="utf-8")
)

# ── 特征提取逻辑（与 ml_detect.py 一致）────────────────────

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


def extract_features(row):
    """从一行 labeled 数据提取 28 个特征"""
    uri = str(row.get("path", "/"))
    ua = str(row.get("user_agent", ""))
    method = str(row.get("method", "GET"))
    referer = str(row.get("referer", ""))

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
    ua_length = len(ua)
    is_bot_ua = 1 if BOT_UA_RE.search(ua) else 0
    is_legitimate_bot = 1 if LEGIT_BOT_RE.search(ua) else 0
    is_empty_ua = 1 if (ua == "" or ua == "-" or ua == "nan") else 0

    # 请求属性
    method_map = {"GET": 0, "POST": 1, "OPTIONS": 2, "HEAD": 3, "PUT": 4, "DELETE": 5}
    method_code = method_map.get(method, 6)
    is_get = 1 if method == "GET" else 0
    is_post = 1 if method == "POST" else 0
    is_options = 1 if method == "OPTIONS" else 0
    is_head = 1 if method == "HEAD" else 0
    is_empty_referer = 1 if (referer == "" or referer == "-" or referer == "nan") else 0

    # 统计特征（重放环境下用默认值 0）
    req_count_60s = 0
    err_count_60s = 0
    unique_url_60s = 0

    # 路径语义特征
    path_is_root = 1 if uri in ("/", "/index.html", "/index.htm", "/index.php") else 0
    is_static_file = 1 if STATIC_EXT_RE.search(uri) else 0
    has_hash_filename = 1 if HASH_FILE_RE.search(uri) else 0
    is_known_api_prefix = 1 if API_PREFIX_RE.search(uri) else 0

    return {
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--sample", type=int, default=0, help="0=全量")
    parser.add_argument("--batch-size", type=int, default=50000)
    args = parser.parse_args()

    print("Step 8c: 离线 Replay 验证（完整 28 特征）")
    print(f"  Threshold: {args.threshold}")

    # 1. 加载 labeled 数据
    print("\nLoading labeled data...")
    dfs = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(df):,} rows")

    # 排除 seo_spam
    if "attack_type" in df.columns:
        mask = df["attack_type"] != "seo_spam"
        n_excluded = (~mask).sum()
        print(f"  Excluding seo_spam: {n_excluded:,} rows")
        df = df[mask].copy()

    # 2. 抽样（可选）
    if args.sample > 0 and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42)
        print(f"  Sampled: {len(df):,} rows")

    # 3. 批量提取特征
    print(f"\nExtracting features for {len(df):,} rows...")
    batch_size = args.batch_size
    all_proba = []

    import joblib
    model = joblib.load(MODEL_DIR / "lightgbm_model_v3.pkl")
    print("  Model loaded.")

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch = df.iloc[start:end]

        features_list = []
        for _, row in batch.iterrows():
            features_list.append(extract_features(row))

        X = np.array(
            [[f.get(col, 0) for col in FEATURE_COLS] for f in features_list],
            dtype=np.float32,
        )
        proba = model.predict_proba(X)[:, 1]
        all_proba.append(proba)

        if end % 200000 < batch_size:
            print(f"  Progress: {end:,}/{len(df):,}")

    y_proba = np.concatenate(all_proba)
    y_true = df["is_attack"].values
    attack_types = df["attack_type"].values if "attack_type" in df.columns else np.array([""] * len(df))

    # 4. 统计
    threshold = args.threshold
    y_pred = (y_proba >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    total_attack = tp + fn
    total_normal = fp + tn

    print("\n" + "=" * 60)
    print("  离线 Replay 验证报告（完整 28 特征）")
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
    for t in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]:
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
    out_path = MODEL_DIR / "replay_offline_v3.csv"
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
