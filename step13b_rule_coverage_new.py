#!/usr/bin/env python3
"""
Step 13b: 用更新后的规则集重新诊断覆盖率

对比旧规则 vs 新规则的拦截率和误报率
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

LABELED_DIR = Path(r"D:\training-data\waf-ml\labeled")

# ── 新 blacklist URI 正则（从更新后的 config.yaml 复制）──
NEW_URI_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'/\.env',
        r'/\.git',
        r'/\.ssh',
        r'/\.htaccess',
        r'/\.htpasswd',
        r'/backup',
        r'/database\.sql',
        r'/dump\.sql',
        r'/config\.(yml|yaml|json|php|conf|ini)',
        r'/wp-config',
        r'/phpmyadmin',
        r'/xmlrpc\.php',
        # 路径穿越
        r'\.\./',
        r'\.\.%2f',
        r'\.\.%5c',
        r'/etc/passwd',
        r'/proc/self',
        # 漏洞扫描路径
        r'/nacos/',
        r'/actuator',
        r'/swagger',
        r'/api-docs',
        r'/v2/api-docs',
        r'/console',
        r'/debug',
        r'/trace',
        r'/metrics',
        r'/health',
        r'/mappings',
        r'/env$',
        r'/refresh',
        r'/jolokia',
        r'/hsqldb',
        r'/graphql',
        r'/__debug__',
        # 后台/管理路径
        r'/admin',
        r'/wp-admin',
        r'/wp-login',
        r'/manager/html',
        r'/host-manager',
        # 随机PHP扫描
        r'/[a-zA-Z0-9]{6,}\.php(?:/|$)',
        r'pki-validation.*\.php',
        r'/runtime/archive/.*\.php',
        # 双斜杠扫描
        r'^//',
        # Nacos API
        r'/v1/(auth|cs)/',
        r'/v1/core/cluster/',
        # SEO spam
        r'\?and/',
        r'/\?/',
        r'\?p=/',
        # SQL注入
        r'sleep\(',
        r'benchmark\(',
        r'updatexml\(',
        r'extractvalue\(',
        r'union\s+select',
        r'load_file\(',
        r'into\s+outfile',
        r'information_schema',
        # XSS
        r'<script[^>]*>',
        r'javascript:',
        r'onerror\s*=',
        r'onload\s*=',
        # 命令注入
        r'cmd\.exe',
        r'powershell',
        r'bash\s+-i',
        r'nc\s+-e',
        # Shellshock
        r'\(\)\s*\{',
        # Log4Shell
        r'\$\{jndi',
        # Struts
        r'\.action',
    ]
]

# ── 新 blacklist UA 正则 ──
NEW_UA_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'python-requests',
        r'python-urllib',
        r'masscan',
        r'zgrab',
        r'zmeu',
        r'curl/',
        r'wget/',
        r'scrapy',
        r'httpclient',
        r'go-http',
        r'java/',
        r'nikto',
        r'sqlmap',
        r'nmap',
        r'dirbuster',
        r'gobuster',
        r'ffuf',
        r'hutool',
    ]
]

# ── 旧规则（对比用）──
OLD_URI_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'/\.env', r'/\.git', r'/backup', r'/database\.sql',
        r'/config\.(yml|yaml|json)', r'/xmlrpc\.php', r'/nacos/',
        r'/actuator', r'/swagger', r'/api-docs', r'\?and/',
        r'sleep\(', r'benchmark\(', r'updatexml\(', r'extractvalue\(',
        r'union\s+select',
    ]
]
OLD_UA_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'python-requests', r'masscan', r'zgrab', r'zmeu',
    ]
]
MISLABEL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'/\?/', r'/nacos/', r'/[a-zA-Z0-9]{6,}\.php(?:/|$)',
        r'^//', r'\?p=/', r'pki-validation.*\.php',
        r'/runtime/archive/.*\.php', r'/v1/(auth|cs)/', r'/v1/core/cluster/',
    ]
]


def match_rules(uri, ua, uri_patterns, ua_patterns, include_mislabel=False):
    """模拟规则匹配"""
    for p in uri_patterns:
        if p.search(uri):
            return True
    if include_mislabel:
        for p in MISLABEL_PATTERNS:
            if p.search(uri):
                return True
    for p in ua_patterns:
        if p.search(ua):
            return True
    return False


def main():
    print("=" * 80)
    print("  Step 13b: 新规则覆盖率诊断（对比旧规则）")
    print("=" * 80)

    # 加载数据
    dfs = []
    for f in sorted(LABELED_DIR.glob("*.parquet")):
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)

    # 移除 seo_spam
    df = df[df["attack_type"] != "seo_spam"].copy()

    # 补漏标
    path_col = df["path"].fillna("").astype(str)
    for pattern in [
        r'/\?/', r'/nacos/', r'/[a-zA-Z0-9]{6,}\.php(?:/|$)',
        r'^//', r'\?p=/', r'pki-validation.*\.php',
        r'/runtime/archive/.*\.php', r'/v1/(auth|cs)/', r'/v1/core/cluster/',
    ]:
        mask = path_col.str.contains(pattern, case=False, regex=True, na=False)
        hit = mask & (df["attack_type"] == "normal") & (df["is_attack"] == 0)
        df.loc[hit, "is_attack"] = 1
        df.loc[hit, "attack_type"] = "mislabeled_fixed"

    print(f"\n  总样本: {len(df):,}")
    print(f"  攻击: {df['is_attack'].sum():,}  正常: {(1-df['is_attack']).sum():,}")

    # 抽样
    attack_df = df[df["is_attack"] == 1]
    normal_df = df[df["is_attack"] == 0]
    attack_sample = attack_df.sample(n=min(500, len(attack_df)), random_state=42)
    normal_sample = normal_df.sample(n=min(500, len(normal_df)), random_state=42)
    samples = pd.concat([attack_sample, normal_sample], ignore_index=True)

    # 准备 uri 和 ua 列
    samples["uri"] = samples["path"].fillna("").astype(str)
    samples["uri"] = samples["uri"].apply(lambda x: x if x.startswith("/") else "/" + x)
    samples["ua"] = samples.get("user_agent", samples.get("ua", "")).fillna("").astype(str) if "user_agent" in samples.columns else samples.get("ua", "").fillna("").astype(str)

    # ── 旧规则匹配 ──
    old_results = []
    for _, row in samples.iterrows():
        blocked = match_rules(row["uri"], row["ua"], OLD_URI_PATTERNS, OLD_UA_PATTERNS, include_mislabel=True)
        old_results.append({"blocked": blocked, "label": "attack" if row["is_attack"] == 1 else "normal",
                           "attack_type": row.get("attack_type", "")})

    # ── 新规则匹配 ──
    new_results = []
    for _, row in samples.iterrows():
        blocked = match_rules(row["uri"], row["ua"], NEW_URI_PATTERNS, NEW_UA_PATTERNS, include_mislabel=False)
        new_results.append({"blocked": blocked, "label": "attack" if row["is_attack"] == 1 else "normal",
                           "attack_type": row.get("attack_type", "")})

    # ── 对比统计 ──
    def calc_stats(results):
        attack_res = [r for r in results if r["label"] == "attack"]
        normal_res = [r for r in results if r["label"] == "normal"]
        tp = sum(1 for r in attack_res if r["blocked"])
        fn = sum(1 for r in attack_res if not r["blocked"])
        fp = sum(1 for r in normal_res if r["blocked"])
        tn = sum(1 for r in normal_res if not r["blocked"])
        return tp, fn, fp, tn

    old_tp, old_fn, old_fp, old_tn = calc_stats(old_results)
    new_tp, new_fn, new_fp, new_tn = calc_stats(new_results)

    print(f"\n{'='*70}")
    print(f"  旧规则 vs 新规则 对比")
    print(f"{'='*70}")
    print(f"  {'':25s} {'旧规则':>15s} {'新规则':>15s} {'变化':>10s}")
    print(f"  {'-'*65}")
    print(f"  {'攻击拦截 (TP)':<25s} {old_tp:>15d} {new_tp:>15d} {new_tp-old_tp:>+10d}")
    print(f"  {'攻击漏过 (FN)':<25s} {old_fn:>15d} {new_fn:>15d} {new_fn-old_fn:>+10d}")
    print(f"  {'正常误拦 (FP)':<25s} {old_fp:>15d} {new_fp:>15d} {new_fp-old_fp:>+10d}")
    print(f"  {'正常放行 (TN)':<25s} {old_tn:>15d} {new_tn:>15d} {new_tn-old_tn:>+10d}")
    print()
    old_tpr = old_tp / max(old_tp + old_fn, 1) * 100
    new_tpr = new_tp / max(new_tp + new_fn, 1) * 100
    old_fpr = old_fp / max(old_fp + old_tn, 1) * 100
    new_fpr = new_fp / max(new_fp + new_tn, 1) * 100
    print(f"  {'攻击拦截率':<25s} {old_tpr:>14.1f}% {new_tpr:>14.1f}% {new_tpr-old_tpr:>+9.1f}%")
    print(f"  {'正常误报率':<25s} {old_fpr:>14.1f}% {new_fpr:>14.1f}% {new_fpr-old_fpr:>+9.1f}%")

    # ── 新规则按攻击类型拆分 ──
    print(f"\n{'='*70}")
    print(f"  新规则按攻击类型拦截率")
    print(f"{'='*70}")
    print(f"  {'Attack Type':<25s} {'Total':>6s} {'Blocked':>8s} {'Rate':>8s}")
    print(f"  {'-'*50}")

    by_type = defaultdict(list)
    for r in new_results:
        if r["label"] == "attack":
            at = r["attack_type"] if r["attack_type"] and r["attack_type"] != "nan" else "unknown"
            by_type[at].append(r)

    for at in sorted(by_type.keys()):
        items = by_type[at]
        blocked = sum(1 for r in items if r["blocked"])
        total = len(items)
        rate = blocked / max(total, 1) * 100
        print(f"  {at:<25s} {total:6d} {blocked:8d} {rate:7.1f}%")

    # ── 漏过样本分析 ──
    missed = [r for r in new_results if r["label"] == "attack" and not r["blocked"]]
    print(f"\n{'='*70}")
    print(f"  新规则漏过样本分析")
    print(f"{'='*70}")
    missed_by_type = defaultdict(int)
    for r in missed:
        at = r["attack_type"] if r["attack_type"] and r["attack_type"] != "nan" else "unknown"
        missed_by_type[at] += 1
    for at, cnt in sorted(missed_by_type.items(), key=lambda x: -x[1]):
        print(f"  {at}: {cnt}")

    # 展示漏过的样本
    print(f"\n--- 漏过样本 Top 20 ---")
    missed_rows = [row for _, row in samples.iterrows()
                   if row["is_attack"] == 1 and not match_rules(
                       row["uri"] if row["uri"].startswith("/") else "/" + str(row.get("path", "")),
                       str(row.get("user_agent", row.get("ua", ""))),
                       NEW_URI_PATTERNS, NEW_UA_PATTERNS)]
    for row in missed_rows[:20]:
        uri = str(row.get("path", "/"))[:60]
        ua = str(row.get("user_agent", row.get("ua", "")))[:30]
        at = row.get("attack_type", "")
        print(f"  [{at}] {str(row.get('method','')):<5s} {uri:<60s}  UA={ua}")

    # ── 误报样本分析 ──
    fp_samples = [r for r in new_results if r["label"] == "normal" and r["blocked"]]
    print(f"\n{'='*70}")
    print(f"  新规则误报样本（正常被拦）")
    print(f"{'='*70}")
    if fp_samples:
        fp_rows = [row for _, row in samples.iterrows()
                   if row["is_attack"] == 0 and match_rules(
                       row["uri"] if row["uri"].startswith("/") else "/" + str(row.get("path", "")),
                       str(row.get("user_agent", row.get("ua", ""))),
                       NEW_URI_PATTERNS, NEW_UA_PATTERNS)]
        for row in fp_rows[:20]:
            uri = str(row.get("path", "/"))[:60]
            ua = str(row.get("user_agent", row.get("ua", "")))[:30]
            print(f"  {str(row.get('method','')):<5s} {uri:<60s}  UA={ua}")
    else:
        print("  无误报")


if __name__ == "__main__":
    main()
