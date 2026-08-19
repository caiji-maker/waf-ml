#!/usr/bin/env python3
"""
Step 7: 漏报分析 —— 按 attack_type 拆分 FNR，找出漏报重灾区

目的：
  1. 加载 v3 模型和测试数据
  2. 按 attack_type 统计每类攻击的漏报率
  3. 输出漏报样本分析（哪些被模型放过了）
  4. 针对漏报重灾区推荐规则补强方案

用法:
    python step7_fnr_analysis.py
"""
import time
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import joblib

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
LEAK_FEATURES = ["status_code", "is_4xx", "is_5xx", "is_444", "is_404", "is_403", "body_size"]

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

RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.40  # 推荐阈值

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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_request_features(df):
    path = df["path"].fillna("").astype(str)
    df["path_is_root"] = path.isin(["/", "/index.html", "/index.htm", "/index.php"]).astype(int)
    df["is_static_file"] = path.str.contains(STATIC_EXT_RE, regex=True, na=False).astype(int)
    df["has_hash_filename"] = path.str.contains(HASH_FILE_RE, regex=True, na=False).astype(int)
    df["is_known_api_prefix"] = path.str.contains(API_PREFIX_RE, regex=True, na=False).astype(int)
    return df


def load_and_clean():
    dfs = []
    for f in FEATURE_FILES:
        if not f.exists():
            continue
        log(f"加载 {f.name} ...")
        dfs.append(pd.read_parquet(f))

    df = pd.concat(dfs, ignore_index=True)

    # 移除 seo_spam
    df = df[df["attack_type"] != "seo_spam"].copy()

    # 补漏标
    path_col = df["path"].fillna("").astype(str)
    for pattern, _ in MISLABEL_PATTERNS:
        mask = path_col.str.contains(pattern, case=False, regex=True, na=False)
        hit = mask & (df["attack_type"] == "normal") & (df[LABEL_COL] == 0)
        if hit.sum() > 0:
            df.loc[hit, LABEL_COL] = 1
            df.loc[hit, "attack_type"] = "mislabeled_fixed"

    # 新增特征
    df = add_request_features(df)

    # 分离
    y = df[LABEL_COL]
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    leak_cols = [c for c in LEAK_FEATURES if c in X.columns]
    if leak_cols:
        X = X.drop(columns=leak_cols)
    X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    return X, y, df


def main():
    log("=" * 60)
    log("  漏报分析：按 attack_type 拆分 FNR")
    log("=" * 60)

    # 加载模型
    model_path = MODEL_DIR / "lightgbm_model_v3.pkl"
    if not model_path.exists():
        log(f"模型文件不存在: {model_path}")
        return

    model = joblib.load(model_path)
    log(f"模型已加载: {model_path}")

    feat_path = MODEL_DIR / "feature_columns_v3.json"
    with open(feat_path, "r") as f:
        feature_cols = json.load(f)

    # 加载数据（保留 attack_type 和 path）
    X, y, df_full = load_and_clean()

    # 用相同的随机种子划分，确保和训练时的测试集一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    # 同时划分 attack_type 和 path
    df_train, df_test = train_test_split(
        df_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    # 模型预测
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    log(f"测试集: {len(y_test):,} 条, 阈值={THRESHOLD}")
    log(f"整体: FPR={((y_pred == 1) & (y_test == 0)).sum() / (y_test == 0).sum():.2%}  "
        f"FNR={((y_pred == 0) & (y_test == 1)).sum() / (y_test == 1).sum():.2%}")

    # ── 按 attack_type 拆分分析 ──
    attack_types = df_test["attack_type"].values
    paths = df_test["path"].values
    y_test_arr = y_test.values

    log(f"\n{'='*60}")
    log(f"  按 attack_type 漏报率")
    log(f"{'='*60}")

    # 只分析攻击类型
    attack_mask = y_test_arr == 1
    attack_type_vals = attack_types[attack_mask]
    y_pred_attack = y_pred[attack_mask]
    y_proba_attack = y_proba[attack_mask]
    path_attack = paths[attack_mask]

    results = []
    for atype in sorted(set(attack_type_vals)):
        mask = attack_type_vals == atype
        total = mask.sum()
        fn = ((y_pred_attack[mask] == 0)).sum()
        tp = ((y_pred_attack[mask] == 1)).sum()
        fnr = fn / total if total > 0 else 0
        avg_score = y_proba_attack[mask].mean()

        results.append({
            "attack_type": atype,
            "total": int(total),
            "TP": int(tp),
            "FN": int(fn),
            "FNR": round(fnr, 4),
            "avg_score": round(avg_score, 4),
        })
        log(f"  {atype:<25s} 总数={total:>6,}  TP={tp:>5,}  FN={fn:>5,}  FNR={fnr:>6.2%}  均分={avg_score:.4f}")

    # 保存结果
    results_df = pd.DataFrame(results)
    out_path = MODEL_DIR / "fnr_by_attack_type.csv"
    results_df.to_csv(out_path, index=False)
    log(f"\n漏报率表已保存: {out_path}")

    # ── 漏报样本分析 ──
    log(f"\n{'='*60}")
    log(f"  漏报样本分析（被放过但实际是攻击）")
    log(f"{'='*60}")

    fn_mask = (y_pred == 0) & (y_test_arr == 1)
    fn_df = pd.DataFrame({
        "attack_type": attack_types[fn_mask],
        "path": paths[fn_mask],
        "score": y_proba[fn_mask],
    })

    if len(fn_df) > 0:
        # 按类型分组展示
        for atype in sorted(fn_df["attack_type"].unique()):
            sub = fn_df[fn_df["attack_type"] == atype]
            log(f"\n  [{atype}] 漏报 {len(sub)} 条，均分={sub['score'].mean():.4f}，最高分={sub['score'].max():.4f}")

            # 显示 Top 10 漏报样本
            top_fn = sub.nlargest(min(10, len(sub)), "score")
            for _, row in top_fn.iterrows():
                log(f"    score={row['score']:.4f}  path={row['path'][:100]}")

        # 保存全部漏报样本
        fn_out = MODEL_DIR / "false_negatives_th40.csv"
        fn_df.to_csv(fn_out, index=False)
        log(f"\n全部漏报样本已保存: {fn_out}")

    # ── 规则补强建议 ──
    log(f"\n{'='*60}")
    log(f"  规则补强建议")
    log(f"{'='*60}")

    suggestions = []
    for r in results:
        atype = r["attack_type"]
        fnr_val = r["FNR"]
        total = r["total"]

        if fnr_val > 0.15 and total >= 10:
            suggestion = "高优先级"
            action = f"加正则规则兜底 + 考虑增加特征"
        elif fnr_val > 0.05 and total >= 10:
            suggestion = "中优先级"
            action = f"观察线上漏报情况，适时加规则"
        else:
            suggestion = "低优先级"
            action = f"模型已能较好识别"

        log(f"  [{atype}] FNR={fnr_val:.2%} → {suggestion}: {action}")
        suggestions.append({
            "attack_type": atype,
            "FNR": fnr_val,
            "priority": suggestion,
            "action": action,
        })

    # 特别关注：样本少但漏报多的类型
    log(f"\n  --- 样本量极少的攻击类型 ---")
    for r in results:
        if r["total"] < 100 and r["total"] > 0:
            log(f"    {r['attack_type']}: 仅 {r['total']} 条样本 → "
                f"模型很难学好，建议 100% 用正则规则拦截")

    sug_path = MODEL_DIR / "rule_suggestions.json"
    with open(sug_path, "w", encoding="utf-8") as f:
        json.dump(suggestions, f, indent=2, ensure_ascii=False)
    log(f"\n规则建议已保存: {sug_path}")

    log("\n分析完成!")


if __name__ == "__main__":
    main()
