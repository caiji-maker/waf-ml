#!/usr/bin/env python3
"""
Step 2b: 分析原始标注数据中的漏标攻击模式
从 features Parquet 加载完整数据，提取所有标为 normal 的样本，
分析其 path 模式，找出被规则遗漏的攻击。
"""

import re
from pathlib import Path
from collections import Counter

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "features"

FEATURE_FILES = [
    FEATURE_DIR / "www.zstzpt.com.parquet",
    FEATURE_DIR / "data.zstzpt.com.parquet",
]


def main():
    # 加载完整数据
    dfs = []
    for f in FEATURE_FILES:
        if f.exists():
            print(f"加载 {f.name} ...")
            df = pd.read_parquet(f)
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"总行数: {len(df):,}")

    # 提取标为 normal 的样本
    normal = df[df["attack_type"] == "normal"].copy()
    print(f"normal 样本: {len(normal):,}")

    # 提取标为 attack 的样本，看规则已覆盖哪些模式
    attack = df[df["is_attack"] == 1]
    print(f"attack 样本: {len(attack):,}")

    # ── 分析 normal 样本的 path 模式 ──────────────────────
    # 1. 按路径首段分组
    print("\n=== normal 样本 path 首段分布 (Top 30) ===")
    def first_segment(path):
        parts = str(path).split("/")
        return parts[1] if len(parts) > 1 else parts[0]
    normal["first_seg"] = normal["path"].apply(first_segment)
    print(normal["first_seg"].value_counts().head(30).to_string())

    # 2. 找出可疑的 normal 样本（可能是漏标攻击）
    print("\n=== 可疑 normal 样本（含已知攻击关键词）===")
    suspicious_patterns = {
        "nacos": r"/nacos/",
        "actuator": r"/actuator/",
        "admin": r"/admin",
        "phpinfo": r"phpinfo",
        "env": r"/\.env",
        "git": r"/\.git",
        "svn": r"/\.svn",
        "shell": r"shell|webshell|backdoor",
        "config": r"/config\.|/configs?",
        "backup": r"/backup|\.bak",
        "wp_": r"/wp-admin|/wp-login|/wp-content",
        "struts": r"struts|ognl",
        "jndi": r"jndi|ldap|rmi",
        "log4j": r"log4j|\$\{",
        "spring": r"spring|cloud|gateway",
        "swagger": r"/swagger|/v2/api-docs|/v3/api-docs",
        "druid": r"/druid",
        "manager": r"/manager/html|/host-manager",
        "solr": r"/solr",
        "jenkins": r"/jenkins",
        "phpmyadmin": r"phpmyadmin|pma",
        "cgi": r"/cgi-bin",
        "eval": r"eval\(|base64_decode|system\(",
        "command": r";cat|;ls|;id|;whoami|;uname|wget|curl|nc|bash|/bin/sh",
        "xss": r"javascript:|<script|onerror=|onload=",
        "sqli": r"union\s+select|sleep\(|benchmark\(|information_schema|load_file",
        "traversal": r"\.\./|etc/passwd|etc/shadow",
        "upload_php": r"\.php\b.*upload|upload.*\.php",
        "random_php": r"/[a-zA-Z0-9]{8,}\.php",  # 随机命名 PHP 文件
        "question_slash": r"/\?/",  # ?/ 模式
        "question_and": r"\?and/",  # ?and 模式（应已被标为 seo_spam）
        "api_open": r"/api/openApi",
    }

    for name, pattern in suspicious_patterns.items():
        mask = normal["path"].str.contains(pattern, case=False, regex=True, na=False)
        count = mask.sum()
        if count > 0:
            print(f"\n  [{name}] 匹配 {count} 条, 示例:")
            samples = normal.loc[mask, "path"].head(5).tolist()
            for s in samples:
                print(f"    {str(s)[:150]}")

    # 3. 统计 normal 中 status_code 分布（看有没有大量 4xx/444）
    print("\n=== normal 样本 status_code 分布 ===")
    print(normal["status_code"].value_counts().head(20).to_string())

    # 4. normal 中 is_empty_ua 的数量
    print(f"\n=== normal 中空 UA 数量: {normal['is_empty_ua'].sum():,} ===")

    # 5. 提取 normal 中 path 包含 .php 的样本
    print("\n=== normal 中含 .php 的样本 ===")
    php_mask = normal["path"].str.contains(r"\.php", case=False, regex=True, na=False)
    print(f"数量: {php_mask.sum()}")
    if php_mask.sum() > 0:
        print(normal.loc[php_mask, "path"].value_counts().head(20).to_string())

    # 6. normal 中 path 包含 .env/.git/.svn 等敏感文件
    print("\n=== normal 中含敏感文件后缀的样本 ===")
    sensitive_mask = normal["path"].str.contains(
        r"\.(env|git|svn|bak|sql|log|ini|conf|sh|bat)$",
        case=False, regex=True, na=False
    )
    print(f"数量: {sensitive_mask.sum()}")
    if sensitive_mask.sum() > 0:
        print(normal.loc[sensitive_mask, "path"].value_counts().head(20).to_string())

    # 7. 统计 seo_spam 在 normal 中的残留（has_and_pattern）
    print("\n=== normal 中 has_and_pattern=True 的样本 ===")
    and_mask = normal["has_and_pattern"] == 1
    print(f"数量: {and_mask.sum()}")
    if and_mask.sum() > 0:
        print(normal.loc[and_mask, "path"].head(20).to_string())

    # 8. 按攻击类型的 path 模式做汇总
    print("\n=== 各 attack_type 的 path 示例 (每类5条) ===")
    for at in df["attack_type"].unique():
        samples = df[df["attack_type"] == at]["path"].head(5).tolist()
        print(f"\n  [{at}]")
        for s in samples:
            print(f"    {str(s)[:150]}")


if __name__ == "__main__":
    main()
