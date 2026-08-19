#!/usr/bin/env python3
"""
Step 13: 纯规则覆盖率诊断

对 Replay 的 1000 条样本，用现有 blacklist 规则做离线匹配，
统计：
1. 纯规则对攻击样本的拦截率（按 attack_type 拆分）
2. 纯规则对正常样本的误报率
3. 哪些攻击类型规则覆盖不到（需要补规则或 ML 兜底）

不启动 WAF，直接用 Python 正则匹配，快速诊断。
"""

import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

MODEL_DIR = Path(r"D:\training-data\waf-ml\model")
LABELED_DIR = Path(r"D:\training-data\waf-ml\labeled")

# ── 现有 blacklist URI 正则（从 config.yaml 复制）──
URI_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'/\.env',
        r'/\.git',
        r'/backup',
        r'/database\.sql',
        r'/config\.(yml|yaml|json)',
        r'/xmlrpc\.php',
        r'/nacos/',
        r'/actuator',
        r'/swagger',
        r'/api-docs',
        r'\?and/',
        r'sleep\(',
        r'benchmark\(',
        r'updatexml\(',
        r'extractvalue\(',
        r'union\s+select',
    ]
]

# ── 现有 blacklist UA 正则 ──
UA_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'python-requests',
        r'masscan',
        r'zgrab',
        r'zmeu',
    ]
]

# ── ml_detect 中的 SEO spam 正则 ──
SEO_SPAM_RE = re.compile(r'/\?and/', re.IGNORECASE)

# ── ml_detect 中的 body 注入预检正则 ──
BODY_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'union\s+select', r'sleep\s*\(', r'benchmark\s*\(', r'updatexml\s*\(',
        r'extractvalue\s*\(', r'<script[^>]*>', r'javascript:', r'onerror\s*=',
        r'onload\s*=', r'eval\s*\(', r'document\.cookie', r'\.\./\.\./',
        r'/etc/passwd', r'cmd\.exe', r'powershell', r'nc\s+-e', r'bash\s+-i',
        r'information_schema', r'load_file\s*\(', r'into\s+outfile',
    ]
]

# ── 补漏标正则（step6 中的 MISLABEL_PATTERNS）──
MISLABEL_PATTERNS = [
    (re.compile(r'/\?/', re.IGNORECASE), "seo_spam_slash"),
    (re.compile(r'/nacos/', re.IGNORECASE), "vuln_scan"),
    (re.compile(r'/[a-zA-Z0-9]{6,}\.php(?:/|$)', re.IGNORECASE), "sensitive_scan"),
    (re.compile(r'^//', re.IGNORECASE), "vuln_scan"),
    (re.compile(r'\?p=/', re.IGNORECASE), "path_traversal"),
    (re.compile(r'pki-validation.*\.php', re.IGNORECASE), "sensitive_scan"),
    (re.compile(r'/runtime/archive/.*\.php', re.IGNORECASE), "sensitive_scan"),
    (re.compile(r'/v1/(auth|cs)/', re.IGNORECASE), "vuln_scan"),
    (re.compile(r'/v1/core/cluster/', re.IGNORECASE), "vuln_scan"),
]

def match_rules(uri, ua, method, body=""):
    """模拟 WAF 规则匹配，返回 (blocked, rule_type, matched_pattern)"""
    # URI 匹配
    for i, p in enumerate(URI_PATTERNS):
        if p.search(uri):
            return True, "uri_rule", p.pattern[:40]
    
    # SEO spam
    if SEO_SPAM_RE.search(uri):
        return True, "seo_spam", "?and/"
    
    # 补漏标规则（这些在训练时被标为攻击，但 WAF blacklist 里没有全部加）
    for p, name in MISLABEL_PATTERNS:
        if p.search(uri):
            return True, "mislabel_rule", name
    
    # UA 匹配
    for p in UA_PATTERNS:
        if p.search(ua):
            return True, "ua_rule", p.pattern[:40]
    
    # Body 匹配（Replay 不发 body，所以这里基本不会命中）
    if method in ("POST", "PUT", "PATCH") and body:
        for p in BODY_INJECTION_PATTERNS:
            if p.search(body):
                return True, "body_rule", p.pattern[:40]
    
    return False, None, None


def main():
    print("=" * 80)
    print("  Step 13: 纯规则覆盖率诊断")
    print("=" * 80)
    
    # 加载 labeled 数据
    dfs = []
    for f in sorted(LABELED_DIR.glob("*.parquet")):
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    
    # 排除 seo_spam（正则 100% 拦截，无测试意义）
    df = df[df["attack_type"] != "seo_spam"].copy()
    
    # 补漏标
    path_col = df["path"].fillna("").astype(str)
    for pattern, atype in [
        (r"/\?/", "seo_spam_slash"), (r"/nacos/", "vuln_scan"),
        (r"/[a-zA-Z0-9]{6,}\.php(?:/|$)", "sensitive_scan"),
        (r"^//", "vuln_scan"), (r"\?p=/", "path_traversal"),
        (r"pki-validation.*\.php", "sensitive_scan"),
        (r"/runtime/archive/.*\.php", "sensitive_scan"),
        (r"/v1/(auth|cs)/", "vuln_scan"),
        (r"/v1/core/cluster/", "vuln_scan"),
    ]:
        mask = path_col.str.contains(pattern, case=False, regex=True, na=False)
        hit = mask & (df["attack_type"] == "normal") & (df["is_attack"] == 0)
        df.loc[hit, "is_attack"] = 1
        df.loc[hit, "attack_type"] = "mislabeled_fixed"
    
    print(f"\n  总样本: {len(df):,}")
    print(f"  攻击: {df['is_attack'].sum():,}  正常: {(1-df['is_attack']).sum():,}")
    
    # 抽样 1000 条（与 Replay 一致）
    attack_df = df[df["is_attack"] == 1]
    normal_df = df[df["is_attack"] == 0]
    
    attack_sample = attack_df.sample(n=min(500, len(attack_df)), random_state=42)
    normal_sample = normal_df.sample(n=min(500, len(normal_df)), random_state=42)
    
    print(f"\n  抽样: 攻击 {len(attack_sample)} + 正常 {len(normal_sample)} = {len(attack_sample)+len(normal_sample)}")
    
    # 用规则匹配
    results = []
    for _, row in pd.concat([attack_sample, normal_sample], ignore_index=True).iterrows():
        uri = str(row.get("path", row.get("uri", "/")))
        if not uri.startswith("/"):
            uri = "/" + uri
        ua = str(row.get("ua", row.get("user_agent", "")))
        method = str(row.get("method", "GET"))
        label = "attack" if row["is_attack"] == 1 else "normal"
        attack_type = str(row.get("attack_type", ""))
        
        blocked, rule_type, pattern = match_rules(uri, ua, method)
        results.append({
            "uri": uri,
            "ua": ua[:50],
            "method": method,
            "label": label,
            "attack_type": attack_type,
            "blocked": blocked,
            "rule_type": rule_type,
            "matched_pattern": pattern,
        })
    
    rdf = pd.DataFrame(results)
    
    # ── 总体统计 ──
    attack_res = rdf[rdf["label"] == "attack"]
    normal_res = rdf[rdf["label"] == "normal"]
    
    tp = (attack_res["blocked"] == True).sum()
    fn = (attack_res["blocked"] == False).sum()
    fp = (normal_res["blocked"] == True).sum()
    tn = (normal_res["blocked"] == False).sum()
    
    print(f"\n{'='*60}")
    print(f"  纯规则拦截效果（离线模拟）")
    print(f"{'='*60}")
    print(f"  攻击样本: {len(attack_res)}")
    print(f"    拦截 (TP): {tp}  ({tp/max(len(attack_res),1)*100:.1f}%)")
    print(f"    漏过 (FN): {fn}  ({fn/max(len(attack_res),1)*100:.1f}%)")
    print(f"  正常样本: {len(normal_res)}")
    print(f"    误拦 (FP): {fp}  ({fp/max(len(normal_res),1)*100:.1f}%)")
    print(f"    放行 (TN): {tn}  ({tn/max(len(normal_res),1)*100:.1f}%)")
    
    # ── 按攻击类型拆分 ──
    print(f"\n{'='*60}")
    print(f"  按攻击类型拦截率")
    print(f"{'='*60}")
    print(f"  {'Attack Type':<25s} {'Total':>6s} {'Blocked':>8s} {'Rate':>8s} {'Main Rule':>20s}")
    print(f"  {'-'*70}")
    
    by_type = defaultdict(list)
    for _, r in attack_res.iterrows():
        at = r["attack_type"] if r["attack_type"] and r["attack_type"] != "nan" else "unknown"
        by_type[at].append(r)
    
    type_stats = []
    for at in sorted(by_type.keys()):
        items = by_type[at]
        blocked = sum(1 for r in items if r["blocked"])
        total = len(items)
        rate = blocked / max(total, 1) * 100
        # 主要命中的规则
        rules = [r["rule_type"] for r in items if r["blocked"]]
        main_rule = max(set(rules), key=rules.count) if rules else "—"
        type_stats.append({"attack_type": at, "total": total, "blocked": blocked, "rate": rate, "main_rule": main_rule})
        print(f"  {at:<25s} {total:6d} {blocked:8d} {rate:7.1f}% {main_rule:>20s}")
    
    # ── 漏过的攻击样本分析 ──
    missed = attack_res[attack_res["blocked"] == False]
    print(f"\n{'='*60}")
    print(f"  漏过的攻击样本（按类型）")
    print(f"{'='*60}")
    missed_by_type = missed["attack_type"].value_counts()
    for at, cnt in missed_by_type.items():
        print(f"  {at}: {cnt}")
    
    # 展示漏过的样本
    print(f"\n--- 漏过样本 Top 20 ---")
    for _, r in missed.head(20).iterrows():
        print(f"  [{r['attack_type']}] {r['method']:<6s} {str(r['uri'])[:60]:<60s}  UA={str(r['ua'])[:25]}")
    
    # ── 误报样本分析 ──
    fp_samples = normal_res[normal_res["blocked"] == True]
    print(f"\n{'='*60}")
    print(f"  误报样本（正常被规则拦）")
    print(f"{'='*60}")
    if len(fp_samples) > 0:
        for _, r in fp_samples.iterrows():
            print(f"  {r['method']:<6s} {str(r['uri'])[:60]:<60s}  rule={r['rule_type']} pattern={r['matched_pattern']}")
    else:
        print("  无误报")
    
    # ── 漏过攻击的 URI 模式分析 ──
    print(f"\n{'='*60}")
    print(f"  漏过攻击的 URI 模式分析")
    print(f"{'='*60}")
    
    # 分析哪些模式没被规则覆盖
    missed_uris = missed["uri"].fillna("").astype(str)
    
    # 常见攻击模式检测
    patterns_to_check = [
        ("路径穿越 ../", r'\.\./'),
        ("路径穿越 ..%2f", r'\.\.%2f'),
        ("SQL注入 ' OR ", r"'\s*or\s"),
        ("SQL注入 UNION", r'union'),
        ("SQL注入 sleep", r'sleep'),
        ("XSS <script", r'<script'),
        ("XSS javascript:", r'javascript:'),
        ("敏感文件 .env", r'\.env'),
        ("敏感文件 .git", r'\.git'),
        ("敏感文件 .ssh", r'\.ssh'),
        ("敏感文件 wp-config", r'wp-config'),
        ("备份文件 .bak", r'\.bak'),
        ("Nacos", r'nacos'),
        ("Actuator", r'actuator'),
        ("Swagger", r'swagger'),
        ("随机PHP", r'[a-zA-Z0-9]{6,}\.php'),
        ("双斜杠 //", r'^//'),
        ("?and/", r'\?and/'),
        ("?p=/ 路径注入", r'\?p=/'),
        ("/.well-known", r'\.well-known'),
        ("/v1/auth", r'/v1/auth'),
        ("/phpmyadmin", r'phpmyadmin'),
        ("/admin", r'/admin'),
        ("/wp-admin", r'wp-admin'),
        ("/console", r'console'),
        ("/debug", r'debug'),
        ("/test", r'test'),
        ("/api/v2", r'/v2/'),
        ("Shellshock", r'\(\)\s*\{'),
        ("Struts", r'\.action'),
        ("Log4Shell", r'\$\{jndi'),
        ("F5 BIG-IP", r'hsqldb'),
        ("JSON注入", r'\$ref'),
    ]
    
    print(f"\n  {'Pattern':<25s} {'Missed Count':>12s} {'Attack Types':>30s}")
    print(f"  {'-'*70}")
    for name, pattern in patterns_to_check:
        mask = missed_uris.str.contains(pattern, case=False, regex=True, na=False)
        cnt = mask.sum()
        if cnt > 0:
            types = missed[mask]["attack_type"].value_counts().to_dict()
            types_str = ", ".join(f"{k}:{v}" for k, v in list(types.items())[:3])
            print(f"  {name:<25s} {cnt:12d} {types_str:>30s}")
    
    # ── 建议 ──
    print(f"\n{'='*60}")
    print(f"  总结")
    print(f"{'='*60}")
    print(f"""
  纯规则效果：
    攻击拦截率: {tp/max(len(attack_res),1)*100:.1f}% ({tp}/{len(attack_res)})
    正常误报率: {fp/max(len(normal_res),1)*100:.1f}% ({fp}/{len(normal_res)})
    
  如果拦截率 > 80% 且误报率 < 5%：
    → 纯规则方案可行，ML 降级为观察模式
    
  如果拦截率 < 50%：
    → 需要大幅补充规则，或保留 ML 作为兜底
    
  如果误报率 > 5%：
    → 规则太激进，需要收窄
""")


if __name__ == "__main__":
    main()
