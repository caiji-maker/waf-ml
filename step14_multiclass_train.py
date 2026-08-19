#!/usr/bin/env python3
"""
Step 14: 多分类模型训练 —— 判断攻击类型

与 v3 二分类模型并行使用：
  - v3 二分类：判断"是否攻击"（FPR<5% 红线）
  - mc 多分类：判断"什么类型攻击"（观察模式，给 Dashboard 提供分类标签）

训练数据：
  - 复用 features/*.parquet（已有 attack_type 字段）
  - 移除 seo_spam（正则已 100% 拦截，不进模型）
  - 补漏标（复用 step6 的 MISLABEL_PATTERNS）
  - 28 特征不变（与 v3 一致）

输出：
  - model/lightgbm_model_mc.pkl
  - model/feature_columns_mc.json
  - model/attack_types.json          (类别名 → 索引映射)
  - model/training_report_mc.json
  - model/lightgbm_feature_importance_mc.csv

用法:
    python step14_multiclass_train.py
"""
import argparse
import time
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
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

# 非特征列（训练时剔除）
NON_FEATURE_COLS = ["is_attack", "attack_type", "ip", "path", "source"]

# 响应特征（WAF 决策时不可用，数据泄露）
LEAK_FEATURES = ["status_code", "is_4xx", "is_5xx", "is_444", "is_404", "is_403", "body_size"]

# 路径语义正则（与 step6 / ml_detect.py 保持一致）
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

RANDOM_STATE = 42
TEST_SIZE    = 0.2

# 补漏标规则（复用 step6）
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


# ── 数据加载与清洗 ─────────────────────────────────────

def load_and_clean():
    """加载 features Parquet，清洗后返回 X, y(attack_type 字符串)"""
    dfs = []
    for f in FEATURE_FILES:
        if not f.exists():
            log(f"WARNING: 文件不存在 {f}")
            continue
        log(f"加载 {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB) ...")
        df = pd.read_parquet(f)
        dfs.append(df)
        log(f"  -> {len(df):,} 行")

    if not dfs:
        raise FileNotFoundError(f"没有找到特征文件: {FEATURE_DIR}")

    df = pd.concat(dfs, ignore_index=True)
    log(f"合并完成: {len(df):,} 行")

    # ── 移除 seo_spam ──
    seo_count = (df["attack_type"] == "seo_spam").sum()
    df = df[df["attack_type"] != "seo_spam"].copy()
    log(f"移除 seo_spam: {seo_count:,} 条, 剩余 {len(df):,} 条")

    # ── 补漏标 ──
    path_col = df["path"].fillna("").astype(str)
    mislabeled_count = 0
    for pattern, fix_type in MISLABEL_PATTERNS:
        mask = path_col.str.contains(pattern, case=False, regex=True, na=False)
        hit = mask & (df["attack_type"] == "normal") & (df["is_attack"] == 0)
        hit_count = hit.sum()
        if hit_count > 0:
            df.loc[hit, "attack_type"] = fix_type
            df.loc[hit, "is_attack"] = 1
            mislabeled_count += hit_count
            log(f"  补漏标 [{pattern[:40]}]: 修正 {hit_count:,} 条 → {fix_type}")
    log(f"共修正漏标: {mislabeled_count:,} 条")

    # ── 新增 4 个路径语义特征（与 step6 / ml_detect.py 一致） ──
    log("新增 4 个路径语义特征...")
    path = df["path"].fillna("").astype(str)
    df["path_is_root"] = path.isin(["/", "/index.html", "/index.htm", "/index.php"]).astype(int)
    df["is_static_file"] = path.str.contains(STATIC_EXT_RE, regex=True, na=False).astype(int)
    df["has_hash_filename"] = path.str.contains(HASH_FILE_RE, regex=True, na=False).astype(int)
    df["is_known_api_prefix"] = path.str.contains(API_PREFIX_RE, regex=True, na=False).astype(int)

    # ── 类别分布 ──
    log(f"\nattack_type 分布:")
    vc = df["attack_type"].value_counts()
    for cls, cnt in vc.items():
        log(f"  {cls:<25s} {cnt:>8,}  ({cnt/len(df)*100:.2f}%)")
    log(f"  {'总数':<25s} {len(df):>8,}")

    # ── 分离特征与标签 ──
    y_str = df["attack_type"].values  # 字符串标签
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

    log(f"\n最终特征数: {len(X.columns)}")
    log(f"特征列: {X.columns.tolist()}")

    return X, y_str


# ── 评估 ──────────────────────────────────────────────

def evaluate_multiclass(model, X_test, y_test, label_encoder):
    """多分类评估：整体指标 + 各类别指标 + 混淆矩阵"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)

    # 各类别指标
    target_names = label_encoder.classes_.tolist()
    report = classification_report(
        y_test, y_pred, target_names=target_names, digits=4, zero_division=0
    )
    log(f"\n{'='*60}")
    log("  LightGBM 多分类测试集评估")
    log(f"{'='*60}")
    log(f"  Accuracy: {acc:.4f}")
    log(f"\n分类报告:\n{report}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"实际_{c}" for c in target_names],
        columns=[f"预测_{c}" for c in target_names],
    )
    log(f"\n混淆矩阵:\n{cm_df.to_string()}")

    # 各类别 FPR（正常被误判为某类攻击的比例）
    normal_idx = list(target_names).index("normal") if "normal" in target_names else -1
    if normal_idx >= 0:
        normal_total = cm[normal_idx].sum()
        normal_correct = cm[normal_idx, normal_idx]
        normal_misclassified = normal_total - normal_correct
        fpr = normal_misclassified / normal_total if normal_total > 0 else 0
        log(f"\n  Normal FPR（正常→攻击）: {fpr:.4f} ({normal_misclassified}/{normal_total})")

    # macro / weighted 指标
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    log(f"\n  Macro F1:     {macro_f1:.4f}")
    log(f"  Weighted F1:   {weighted_f1:.4f}")
    log(f"  Macro Prec:   {macro_prec:.4f}")
    log(f"  Macro Rec:    {macro_rec:.4f}")

    # 各类别 precision/recall/f1
    per_class = {}
    for i, cls in enumerate(target_names):
        per_class[cls] = {
            "precision": round(precision_score(y_test == i, y_pred == i, zero_division=0), 4),
            "recall": round(recall_score(y_test == i, y_pred == i, zero_division=0), 4),
            "f1": round(f1_score(y_test == i, y_pred == i, zero_division=0), 4),
            "support": int((y_test == i).sum()),
        }

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "normal_fpr": round(fpr, 4) if normal_idx >= 0 else None,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_order": target_names,
    }


# ── 训练 ──────────────────────────────────────────────

def train_lightgbm_mc(X_train, y_train, X_test, y_test, feature_names, label_encoder):
    import lightgbm as lgb

    log("\n" + "="*60)
    log("训练 LightGBM 多分类 (objective=multiclass) ...")
    log("="*60)

    n_classes = len(label_encoder.classes_)
    log(f"类别数: {n_classes}")
    log(f"类别映射: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # 类别权重：用 sqrt 反频率加权，防止大类碾压小类
    class_counts = np.bincount(y_train, minlength=n_classes)
    total = len(y_train)
    # sqrt(neg/pos) 的泛化版：weight_i = sqrt(total / (n_classes * count_i))
    class_weights = np.sqrt(total / (n_classes * class_counts.astype(float)))
    class_weights = class_weights / class_weights.sum() * n_classes  # 归一化到均值 1
    class_weight_map = {i: float(w) for i, w in enumerate(class_weights)}
    sample_weights = np.array([class_weight_map[c] for c in y_train])

    log(f"\n类别权重:")
    for i, cls in enumerate(label_encoder.classes_):
        log(f"  {cls:<25s} count={class_counts[i]:>8,}  weight={class_weight_map[i]:.4f}")

    t0 = time.time()
    model = lgb.LGBMClassifier(
        objective="multiclass",
        n_classes=n_classes,
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        min_child_samples=100,
        min_gain_to_split=0.5,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
        ],
    )
    elapsed = time.time() - t0
    log(f"\nLightGBM 多分类训练完成，耗时 {elapsed:.1f}s，最佳迭代 {model.best_iteration_}")

    # 评估
    metrics = evaluate_multiclass(model, X_test, y_test, label_encoder)
    metrics["train_time_sec"] = round(elapsed, 1)
    metrics["best_iteration"] = int(model.best_iteration_)

    # 特征重要性
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    log(f"\n特征重要性 Top-15:")
    log(imp.head(15).to_string(index=False))

    imp_path = MODEL_DIR / "lightgbm_feature_importance_mc.csv"
    imp.to_csv(imp_path, index=False)
    log(f"特征重要性已保存: {imp_path}")

    # 保存模型
    import joblib
    model_path = MODEL_DIR / "lightgbm_model_mc.pkl"
    joblib.dump(model, model_path)
    log(f"模型已保存: {model_path}")

    return model, metrics


# ── 主流程 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WAF 多分类模型训练")
    parser.add_argument("--model", choices=["lightgbm"], default="lightgbm",
                        help="选择训练的模型 (目前只支持 lightgbm)")
    args = parser.parse_args()

    log("="*60)
    log("  WAF 多分类模型训练 (攻击类型判断)")
    log("="*60)

    # 加载数据
    X, y_str = load_and_clean()
    feature_names = X.columns.tolist()

    # 标签编码
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    log(f"\n标签编码完成:")
    for cls, idx in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        log(f"  {cls:<25s} → {idx}")

    # 划分
    log("\n划分训练集/测试集 (80/20, stratify) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True, stratify=y
    )
    log(f"训练集: {len(X_train):,}  测试集: {len(X_test):,}")

    # 保存特征列
    feat_path = MODEL_DIR / "feature_columns_mc.json"
    with open(feat_path, "w") as f:
        json.dump(feature_names, f)
    log(f"特征列表已保存: {feat_path}")

    # 保存类别映射
    attack_types = {
        "classes": label_encoder.classes_.tolist(),
        "mapping": {cls: int(idx) for cls, idx in
                     zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))},
    }
    types_path = MODEL_DIR / "attack_types.json"
    with open(types_path, "w", encoding="utf-8") as f:
        json.dump(attack_types, f, indent=2, ensure_ascii=False)
    log(f"类别映射已保存: {types_path}")

    # 训练
    _, metrics = train_lightgbm_mc(X_train, y_train, X_test, y_test, feature_names, label_encoder)

    # 保存报告
    report_path = MODEL_DIR / "training_report_mc.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log(f"\n评估报告已保存: {report_path}")

    log("\n全部完成!")
    log(f"  模型: {MODEL_DIR / 'lightgbm_model_mc.pkl'}")
    log(f"  特征: {feat_path}")
    log(f"  类别: {types_path}")
    log(f"  报告: {report_path}")


if __name__ == "__main__":
    main()
