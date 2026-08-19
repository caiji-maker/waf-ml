"""
Step 3: 特征工程 —— 向量化版本
从标注后的日志提取模型可用的数值特征
"""
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

SPECIAL_CHARS = set('?&=%\'";<>|(){}[]^~`!@#$')
BOT_UA_KEYWORDS = re.compile(
    r'bot|crawler|spider|scan|fetch|python|curl|wget|scrapy|hutool|undici|'
    r'go-http|java/|httpclient|masscan|zgrab|zmeu',
    re.IGNORECASE
)
LEGITIMATE_BOT = re.compile(
    r'bingbot|googlebot|baiduspider|bytespider|yandexbot|duckduckbot',
    re.IGNORECASE
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """向量化构建所有特征"""
    print(f"构建特征, {len(df):,} 行")

    # 确保字符串列
    df['path'] = df['path'].fillna('').astype(str)
    df['user_agent'] = df['user_agent'].fillna('').astype(str)
    df['referer'] = df['referer'].fillna('').astype(str)

    features = pd.DataFrame(index=df.index)

    # ===== 1. URL 特征 =====
    print("  URL 特征...")
    features['url_length'] = df['path'].str.len()
    features['num_special_chars'] = df['path'].apply(lambda s: sum(1 for c in s if c in SPECIAL_CHARS))
    features['num_digits'] = df['path'].str.count(r'\d')
    features['num_dots'] = df['path'].str.count(r'\.')
    features['num_slashes'] = df['path'].str.count('/')
    features['num_params'] = df['path'].str.count('&') + df['path'].str.count('=')
    features['has_and_pattern'] = df['path'].str.contains(r'\?and/', case=False, regex=True, na=False).astype(int)
    features['encoded_chars'] = df['path'].str.count('%')
    features['path_depth'] = features['num_slashes'] - df['path'].str.startswith('/').astype(int)
    features['digit_ratio'] = features['num_digits'] / features['url_length'].clip(lower=1)
    features['special_char_ratio'] = features['num_special_chars'] / features['url_length'].clip(lower=1)

    # ===== 2. UA 特征 =====
    print("  UA 特征...")
    features['ua_length'] = df['user_agent'].str.len()
    features['is_bot_ua'] = df['user_agent'].str.contains(BOT_UA_KEYWORDS, regex=True, na=False).astype(int)
    features['is_legitimate_bot'] = df['user_agent'].str.contains(LEGITIMATE_BOT, regex=True, na=False).astype(int)
    features['is_empty_ua'] = ((df['user_agent'] == '-') | (df['user_agent'] == '')).astype(int)

    # ===== 3. 请求属性特征 =====
    print("  请求属性特征...")
    method_map = {'GET': 0, 'POST': 1, 'OPTIONS': 2, 'HEAD': 3, 'PUT': 4, 'DELETE': 5}
    features['method_code'] = df['method'].map(method_map).fillna(6).astype(int)
    features['is_get'] = (df['method'] == 'GET').astype(int)
    features['is_post'] = (df['method'] == 'POST').astype(int)
    features['is_options'] = (df['method'] == 'OPTIONS').astype(int)
    features['is_head'] = (df['method'] == 'HEAD').astype(int)
    features['status_code'] = df['status']
    features['is_4xx'] = ((df['status'] >= 400) & (df['status'] < 500)).astype(int)
    features['is_5xx'] = ((df['status'] >= 500) & (df['status'] < 600)).astype(int)
    features['is_444'] = (df['status'] == 444).astype(int)
    features['is_404'] = (df['status'] == 404).astype(int)
    features['is_403'] = (df['status'] == 403).astype(int)
    features['body_size'] = df['size']
    features['is_empty_referer'] = ((df['referer'] == '') | (df['referer'] == '-')).astype(int)

    # ===== 4. 统计特征 (时间窗口) =====
    # 用 pandas rolling 的内置聚合 (sum)，C 底层实现，无 Python 逐行回调
    print("  统计特征 (时间窗口)...")
    df['timestamp'] = pd.to_datetime(df['time'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')
    df = df.sort_values(['ip', 'timestamp']).reset_index(drop=True)

    df['_one'] = 1
    df['_is404'] = (df['status'] == 404).astype(int)
    # 同 IP 上一条请求是否同 path (用于近似不同 URL 数)
    prev_path = df.groupby('ip', sort=False)['path'].shift(1)
    df['_same_url'] = (df['path'] == prev_path).astype(int)

    # 一次 rolling 同时算三个 sum
    print("    计算 rolling 窗口 (60s)...")
    rolled = df.groupby('ip', sort=False).rolling(window='60s', on='timestamp')
    req_cnt = rolled['_one'].sum().reset_index(level=0, drop=True)
    err_cnt = rolled['_is404'].sum().reset_index(level=0, drop=True)
    same_cnt = rolled['_same_url'].sum().reset_index(level=0, drop=True)

    # 不同 URL 数 ≈ 请求数 - 连续重复数
    unique_url = (req_cnt.values - same_cnt.values).clip(min=1).astype(np.int32)

    features['req_count_60s'] = req_cnt.values.astype(np.int32)
    features['err_count_60s'] = err_cnt.values.astype(np.int32)
    features['unique_url_60s'] = unique_url

    # 清理
    df.drop(columns=['timestamp', '_one', '_is404', '_same_url'], inplace=True)

    # 添加标签和元信息
    features['is_attack'] = df['is_attack'].values
    features['attack_type'] = df['attack_type'].values
    features['ip'] = df['ip'].values
    features['path'] = df['path'].values
    features['source'] = df['source'].values if 'source' in df.columns else ''

    return features


def main():
    import argparse
    parser = argparse.ArgumentParser(description="特征工程")
    parser.add_argument("--input-dir", default=r"D:\training-data\waf-ml\labeled")
    parser.add_argument("--out", default=r"D:\training-data\waf-ml\features")
    parser.add_argument("--sample", type=int, default=0, help="采样行数，0=全部")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquets = sorted(input_dir.glob("*.parquet"))
    if not parquets:
        print("没有找到标注文件，请先运行 step2_label.py")
        sys.exit(1)

    for pq in parquets:
        print(f"\n=== 特征工程 {pq.name} ===")
        df = pd.read_parquet(pq)

        if args.sample > 0 and len(df) > args.sample:
            print(f"采样 {args.sample:,} 行 (调试模式)")
            df = df.sample(args.sample, random_state=42)

        print(f"加载 {len(df):,} 行")

        feature_df = build_features(df)

        out_path = out_dir / pq.name
        feature_df.to_parquet(out_path, engine='pyarrow', index=False)
        print(f"已保存: {out_path}")

        meta_cols = {'is_attack', 'attack_type', 'ip', 'path', 'source'}
        feat_cols = [c for c in feature_df.columns if c not in meta_cols]
        print(f"特征数: {len(feat_cols)}")
        print(f"特征列: {feat_cols}")


if __name__ == "__main__":
    main()
