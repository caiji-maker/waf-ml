"""
Step 9: WAF 红队验证测试

包含：
1. SQL 注入测试（SQLMap 风格 payload）
2. 目录扫描/敏感文件探测
3. 路径穿越测试
4. 命令注入测试
5. XSS 测试
6. 正常流量基线（确保不误拦）

统计：总拦截率 + 按攻击类型拆分
"""

import asyncio
import sys
import time
import random
import string
from pathlib import Path

try:
    import aiohttp
except ImportError:
    import os
    os.system(f'{sys.executable} -m pip install aiohttp -q')
    import aiohttp

WAF_URL = "http://127.0.0.1:8082"
CONCURRENCY = 5
TIMEOUT_SEC = 10

# ── 测试 payload 定义 ──────────────────────────────────────

TESTS = {
    "sql_injection": [
        # Classic SQLi
        "/?id=1' OR '1'='1",
        "/?id=1' UNION SELECT NULL,NULL,NULL--",
        "/?id=1; DROP TABLE users--",
        "/?id=1' AND 1=1--",
        "/?id=1' AND SLEEP(5)--",
        "/?id=1' AND BENCHMARK(10000000,SHA1('test'))--",
        "/?id=1 AND UPDATEXML(NULL,CONCAT(0x7e,VERSION()),0)--",
        "/?id=1 AND EXTRACTVALUE(NULL,CONCAT(0x7e,VERSION()))--",
        "/?id=1' UNION SELECT username,password FROM users--",
        "/?id=1' OR 1=1#",
        "/?id=1' OR '1'='1' /*",
        "/?id=1 UNION SELECT 1,2,3,4,5,6,7,8,9,10--",
        "/?search=' OR 1=1--",
        "/?user=admin'--",
        "/?id=(SELECT 1 FROM(SELECT COUNT(*),CONCAT((SELECT VERSION()),0x7e,FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a)",
        # Blind SQLi
        "/?id=1' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
        "/?id=1' AND SUBSTRING.VERSION(),1,1)=5--",
        "/?id=1' AND ASCII(SUBSTRING((SELECT username FROM users LIMIT 1),1,1))>64--",
        # Error-based
        "/?id=1' AND EXP(~(SELECT * FROM (SELECT CONCAT(VERSION(),0x7e)))a)--",
        "/?id=-1' OR 1 GROUP BY CONCAT(0x7e,VERSION(),0x7e) HAVING 1=1--",
    ],
    "path_traversal": [
        "/../../../../etc/passwd",
        "/..%2f..%2f..%2fetc%2fpasswd",
        "/%2e%2e/%2e%2e/%2e%2e/etc/passwd",
        "/....//....//....//etc/passwd",
        "/..%252f..%252f..%252fetc/passwd",
        "/..%c0%af..%c0%af..%c0%afetc/passwd",
        "/var/www/../../etc/passwd",
        "/cgi-bin/../../../../etc/passwd",
        "/wp-content/../../../../etc/passwd",
        "/static/../../../etc/shadow",
    ],
    "sensitive_scan": [
        "/.env",
        "/.git/config",
        "/.git/HEAD",
        "/.htaccess",
        "/.htpasswd",
        "/backup.sql",
        "/database.sql",
        "/config.yml",
        "/config.json",
        "/config.yaml",
        "/phpinfo.php",
        "/info.php",
        "/server-status",
        "/server-info",
        "/.DS_Store",
        "/robots.txt",
        "/sitemap.xml",
        "/wp-config.php",
        "/web.config",
        "/xmlrpc.php",
    ],
    "vuln_scan": [
        "/nacos/",
        "/actuator",
        "/actuator/health",
        "/actuator/env",
        "/swagger-ui.html",
        "/api-docs",
        "/v2/api-docs",
        "/v3/api-docs",
        "/graphql",
        "/.well-known/security.txt",
        "/console",
        "/manager/html",
        "/admin",
        "/admin/login",
        "/phpmyadmin/",
        "/wp-admin/",
        "/wp-login.php",
        "/solr/",
        "/jenkins/",
        "/.svn/entries",
    ],
    "command_injection": [
        "/?cmd=; cat /etc/passwd",
        "/?cmd=| ls -la",
        "/?cmd=`id`",
        "/?cmd=$(whoami)",
        "/?ping=; cat /etc/passwd",
        "/?ip=127.0.0.1; cat /etc/shadow",
        "/?cmd=| cat /etc/passwd #",
        "/?cmd=& dir C:\\",
    ],
    "xss": [
        "/?q=<script>alert(1)</script>",
        "/?q=<img src=x onerror=alert(1)>",
        "/?q=<svg onload=alert(1)>",
        "/?q=javascript:alert(1)",
        "/?q=<body onload=alert(1)>",
        "/?q=<iframe src='javascript:alert(1)'>",
        "/?q=<input onfocus=alert(1) autofocus>",
        "/?q='%22--%3E%3Cscript%3Ealert(1)%3C/script%3E",
    ],
    "path_padding": [
        "//",
        "///",
        "/././././",
        "/../",
        "/./.././",
        "/test/..;/admin",
        "/;./admin",
        "/test/..;/etc/passwd",
    ],
}

# 正常流量（不应被拦截）
NORMAL_REQUESTS = [
    "/",
    "/index.html",
    "/about",
    "/contact",
    "/products",
    "/services",
    "/blog",
    "/faq",
    "/privacy-policy",
    "/terms-of-service",
    "/uploads/2024/report.pdf",
    "/images/logo.png",
    "/css/style.css",
    "/js/app.js",
    "/api/v1/users",
    "/api/v1/products?page=1",
    "/search?q=laptop",
    "/category/electronics",
    "/article/how-to-use-our-product",
    "/user/profile",
]


async def send_request(session, url, method="GET", ua=None, sem=None):
    """发送请求到 WAF"""
    headers = {}
    if ua:
        headers["User-Agent"] = ua
    else:
        headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

    async_func = sem.__aenter__ if sem else None
    
    try:
        if sem:
            async with sem:
                async with session.request(
                    method=method, url=url, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC),
                    allow_redirects=False,
                ) as resp:
                    return resp.status
        else:
            async with session.request(
                method=method, url=url, headers=headers,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC),
                allow_redirects=False,
            ) as resp:
                return resp.status
    except Exception as e:
        return 0


async def main():
    print("Step 9: WAF Red Team Validation")
    print(f"  Target: {WAF_URL}")

    # 1. 健康检查
    print("\nChecking WAF...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{WAF_URL}/__waf/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print(f"  WAF returned {resp.status}")
                    sys.exit(1)
                print("  WAF is online!")
    except Exception as e:
        print(f"  Cannot connect: {e}")
        sys.exit(1)

    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, limit_per_host=CONCURRENCY)
    
    results = {}

    async with aiohttp.ClientSession(connector=connector) as session:
        # 2. 攻击测试
        print("\n--- 攻击测试 ---")
        for attack_type, payloads in TESTS.items():
            print(f"\n  Testing {attack_type} ({len(payloads)} payloads)...")
            blocked = 0
            type_results = []
            
            for payload in payloads:
                url = f"{WAF_URL}{payload}"
                # 每个请求用随机 IP 避免被 ban
                random_ip = f"10.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
                
                async with sem:
                    try:
                        async with session.request(
                            method="GET", url=url,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                                "X-Forwarded-For": random_ip,
                            },
                            timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC),
                            allow_redirects=False,
                        ) as resp:
                            status = resp.status
                    except Exception as e:
                        status = 0

                is_blocked = status == 403
                if is_blocked:
                    blocked += 1
                type_results.append({
                    "payload": payload,
                    "status": status,
                    "blocked": is_blocked,
                })

                short_payload = payload[:60] + "..." if len(payload) > 60 else payload
                mark = "BLOCKED" if is_blocked else "PASS"
                print(f"    [{mark}] {short_payload} -> {status}")

            rate = blocked / max(len(payloads), 1) * 100
            results[attack_type] = {
                "total": len(payloads),
                "blocked": blocked,
                "rate": rate,
                "details": type_results,
            }
            print(f"  => {attack_type}: {blocked}/{len(payloads)} blocked ({rate:.0f}%)")

        # 3. 正常流量测试
        print(f"\n--- 正常流量测试 ({len(NORMAL_REQUESTS)} requests) ---")
        normal_blocked = 0
        for uri in NORMAL_REQUESTS:
            url = f"{WAF_URL}{uri}"
            random_ip = f"192.168.{random.randint(1,254)}.{random.randint(1,254)}"
            
            async with sem:
                try:
                    async with session.request(
                        method="GET", url=url,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                            "X-Forwarded-For": random_ip,
                            "Referer": "https://www.example.com/",
                        },
                        timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC),
                        allow_redirects=False,
                    ) as resp:
                        status = resp.status
                except Exception:
                    status = 0

            is_blocked = status == 403
            if is_blocked:
                normal_blocked += 1
            mark = "BLOCKED(!)" if is_blocked else "OK"
            print(f"    [{mark}] {uri} -> {status}")

        results["normal"] = {
            "total": len(NORMAL_REQUESTS),
            "blocked": normal_blocked,
            "rate": normal_blocked / max(len(NORMAL_REQUESTS), 1) * 100,
        }

    # 4. 汇总
    print("\n" + "=" * 60)
    print("  WAF Red Team Validation Report")
    print("=" * 60)
    print(f"\n  {'Type':20s}  {'Blocked':>8s}  {'Total':>8s}  {'Rate':>8s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}")

    total_attack_blocked = 0
    total_attack = 0
    for attack_type, r in results.items():
        if attack_type == "normal":
            continue
        total_attack_blocked += r["blocked"]
        total_attack += r["total"]
        print(f"  {attack_type:20s}  {r['blocked']:>8d}  {r['total']:>8d}  {r['rate']:>7.0f}%")

    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}")
    overall_rate = total_attack_blocked / max(total_attack, 1) * 100
    print(f"  {'TOTAL ATTACK':20s}  {total_attack_blocked:>8d}  {total_attack:>8d}  {overall_rate:>7.0f}%")

    nr = results["normal"]
    print(f"  {'NORMAL (should pass)':20s}  {nr['blocked']:>8d}  {nr['total']:>8d}  {nr['rate']:>7.0f}%")

    print("\n" + "=" * 60)

    # 5. 未拦截的攻击详情
    print("\n--- Missed Attacks (need rule supplement) ---")
    for attack_type, r in results.items():
        if attack_type == "normal":
            continue
        missed = [d["payload"] for d in r["details"] if not d["blocked"]]
        if missed:
            print(f"\n  {attack_type}:")
            for p in missed:
                print(f"    - {p[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
