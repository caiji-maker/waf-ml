"""
Step 8: 日志 Replay 重放验证

从标注数据中抽样攻击和正常请求，重放给 WAF，统计实际 FPR/FNR。

逻辑：
1. 从 labeled parquet 读取数据（含原始 ip/path/ua/method 等字段 + is_attack 标签 + attack_type）
2. 抽样：攻击 500 条 + 正常 500 条（默认）
3. 用 aiohttp 异步重放给 WAF
4. 统计：TP/TN/FP/FN + 各攻击类型的拦截率

注意：
- WAF 必须先启动（observe_mode=false 才能测试真实拦截）
- 抽样时排除 seo_spam（已用正则 100% 拦截，测试无意义）
- 正常请求可能返回各种状态码，只有 403 才算被拦截
"""

import asyncio
import sys
import os
import time
import random
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

# 确保 aiohttp 可用
try:
    import aiohttp
except ImportError:
    print("Installing aiohttp...")
    os.system(f'{sys.executable} -m pip install aiohttp -q')
    import aiohttp

# ── 常量 ──────────────────────────────────────────────────────

DATA_DIR = Path(r"D:\training-data\waf-ml\labeled")
WAF_URL = "http://127.0.0.1:8082"
CONCURRENCY = 20          # 并发数
TIMEOUT_SEC = 10          # 单请求超时

# ── 抽样 ──────────────────────────────────────────────────────

def load_and_sample(n_attack=500, n_normal=500, seed=42):
    """从 labeled 数据加载并抽样"""
    print("Loading labeled data...")
    dfs = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows: {len(df):,}")

    # 基本列检查
    required_cols = ["ip", "path", "ua", "method", "is_attack"]
    for col in required_cols:
        if col not in df.columns:
            # 尝试别名
            alias = {"path": "uri", "ua": "user_agent"}.get(col)
            if alias and alias in df.columns:
                df[col] = df[alias]
            else:
                raise ValueError(f"Column '{col}' not found. Available: {df.columns.tolist()}")

    # 排除 seo_spam（正则已 100% 拦截）
    attack_col = "attack_type" if "attack_type" in df.columns else None
    if attack_col:
        seo_mask = df[attack_col] == "seo_spam"
        print(f"  Excluding {seo_mask.sum():,} seo_spam rows")
        df = df[~seo_mask].copy()

    # 拆分攻击/正常
    attack_df = df[df["is_attack"] == 1].copy()
    normal_df = df[df["is_attack"] == 0].copy()
    print(f"  Attack rows (excl. seo_spam): {len(attack_df):,}")
    print(f"  Normal rows: {len(normal_df):,}")

    # 抽样
    random.seed(seed)
    n_attack = min(n_attack, len(attack_df))
    n_normal = min(n_normal, len(normal_df))

    attack_sample = attack_df.sample(n=n_attack, random_state=seed)
    normal_sample = normal_df.sample(n=n_normal, random_state=seed)

    # 打上标签
    attack_sample["_label"] = "attack"
    normal_sample["_label"] = "normal"

    return pd.concat([attack_sample, normal_sample], ignore_index=True)


# ── 重放 ──────────────────────────────────────────────────────

async def replay_request(session, row, sem, waf_url):
    """重放单条请求到 WAF"""
    uri = str(row.get("path", row.get("uri", "/")))
    ua = str(row.get("ua", row.get("user_agent", "")))
    method = str(row.get("method", "GET"))
    host = str(row.get("host", "www.example.com"))

    # 确保 uri 以 / 开头
    if not uri.startswith("/"):
        uri = "/" + uri

    url = f"{waf_url}{uri}"

    headers = {
        "User-Agent": ua if ua and ua != "nan" else "",
        "Host": host if host and host != "nan" else "www.example.com",
    }

    # 用原始 IP 作为 X-Forwarded-For，让 WAF 看到不同客户端
    src_ip = str(row.get("ip", ""))
    if src_ip and src_ip != "nan":
        headers["X-Forwarded-For"] = src_ip

    async with sem:
        try:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC),
                allow_redirects=False,
            ) as resp:
                status = resp.status
                return {
                    "status": status,
                    "blocked": status == 403,
                    "uri": uri,
                    "method": method,
                    "ua": ua[:50],
                    "label": row["_label"],
                    "attack_type": str(row.get("attack_type", "")),
                    "error": None,
                }
        except Exception as e:
            return {
                "status": 0,
                "blocked": False,
                "uri": uri,
                "method": method,
                "ua": ua[:50],
                "label": row["_label"],
                "attack_type": str(row.get("attack_type", "")),
                "error": str(e),
            }


async def run_replay(samples_df, waf_url):
    """异步重放所有抽样请求"""
    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, limit_per_host=CONCURRENCY)

    results = []
    total = len(samples_df)
    done = 0

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for _, row in samples_df.iterrows():
            tasks.append(replay_request(session, row, sem, waf_url))

        # 带进度条的并发执行
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  Progress: {done}/{total}")

    return results


# ── 统计 ──────────────────────────────────────────────────────

def analyze_results(results):
    """统计拦截效果"""
    print("\n" + "=" * 60)
    print("  WAF Replay 验证报告")
    print("=" * 60)

    # 分组
    attack_results = [r for r in results if r["label"] == "attack"]
    normal_results = [r for r in results if r["label"] == "normal"]

    # 错误统计
    errors = [r for r in results if r["error"]]
    if errors:
        print(f"\n[WARNING] {len(errors)} requests had errors (excluded from stats)")
        for e in errors[:5]:
            print(f"  {e['method']} {e['uri'][:60]} -> {e['error'][:80]}")

    # 有效结果
    attack_ok = [r for r in attack_results if not r["error"]]
    normal_ok = [r for r in normal_results if not r["error"]]

    # 二分类统计
    tp = sum(1 for r in attack_ok if r["blocked"])    # 攻击被拦截
    fn = sum(1 for r in attack_ok if not r["blocked"])  # 攻击漏过
    fp = sum(1 for r in normal_ok if r["blocked"])     # 正常被误拦
    tn = sum(1 for r in normal_ok if not r["blocked"])  # 正常放行

    total_attack = tp + fn
    total_normal = fp + tn

    print(f"\n--- 总体拦截效果 ---")
    print(f"  攻击样本: {total_attack}")
    print(f"    被拦截 (TP): {tp}")
    print(f"    漏过   (FN): {fn}")
    print(f"    拦截率 (Recall): {tp/max(total_attack,1)*100:.2f}%")
    print(f"    漏报率 (FNR):   {fn/max(total_attack,1)*100:.2f}%")

    print(f"\n  正常样本: {total_normal}")
    print(f"    被误拦 (FP): {fp}")
    print(f"    正确放行 (TN): {tn}")
    print(f"    误报率 (FPR):   {fp/max(total_normal,1)*100:.2f}%")

    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"    精确率 (Precision): {precision*100:.2f}%")

    # 按攻击类型拆分
    print(f"\n--- 按攻击类型拦截率 ---")
    by_type = defaultdict(list)
    for r in attack_ok:
        at = r["attack_type"] if r["attack_type"] and r["attack_type"] != "nan" else "unknown"
        by_type[at].append(r)

    for at in sorted(by_type.keys()):
        items = by_type[at]
        blocked = sum(1 for r in items if r["blocked"])
        total = len(items)
        rate = blocked / max(total, 1) * 100
        print(f"  {at:25s}  {blocked:4d}/{total:4d}  ({rate:5.1f}%)")

    # 漏报样本展示
    missed = [r for r in attack_ok if not r["blocked"]]
    if missed:
        print(f"\n--- 漏报样本 Top 10 ---")
        for r in missed[:10]:
            print(f"  [{r['attack_type']}] {r['method']} {r['uri'][:80]}  UA={r['ua'][:30]}")

    # 误报样本展示
    false_blocked = [r for r in normal_ok if r["blocked"]]
    if false_blocked:
        print(f"\n--- 误报样本 Top 10 ---")
        for r in false_blocked[:10]:
            print(f"  {r['method']} {r['uri'][:80]}  UA={r['ua'][:30]}")

    print("\n" + "=" * 60)
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "fnr": fn / max(total_attack, 1),
        "fpr": fp / max(total_normal, 1),
    }


# ── 主入口 ────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="WAF Replay Validation")
    parser.add_argument("--n-attack", type=int, default=500, help="攻击抽样数")
    parser.add_argument("--n-normal", type=int, default=500, help="正常抽样数")
    parser.add_argument("--waf-url", type=str, default=WAF_URL, help="WAF 地址")
    parser.add_argument("--full", action="store_true", help="全量重放（不抽样）")
    args = parser.parse_args()

    waf_url = args.waf_url

    print(f"WAF Replay Validation")
    print(f"  Target: {waf_url}")
    print(f"  Attack samples: {args.n_attack}")
    print(f"  Normal samples: {args.n_normal}")

    # 1. 检查 WAF 是否在线
    print("\nChecking WAF availability...")
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{waf_url}/__waf/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    print("  WAF is online!")
                else:
                    print(f"  WAF returned status {resp.status}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to WAF at {waf_url}")
        print(f"  Detail: {e}")
        print(f"\n  Please start WAF first:")
        print(f"    cd {Path(__file__).parent}")
        print(f"    ..\\mini-waf\\start.bat")
        print(f"  Or with observe_mode=false for real blocking test.")
        sys.exit(1)

    # 2. 加载抽样
    if args.full:
        # 全量模式
        df = pd.read_parquet(DATA_DIR / "www.zstzpt.com.parquet")
        if "is_attack" in df.columns:
            samples_df = df.copy()
            samples_df["_label"] = df["is_attack"].map({1: "attack", 0: "normal"})
        print(f"  Full mode: {len(samples_df):,} rows")
    else:
        samples_df = load_and_sample(args.n_attack, args.n_normal)

    print(f"\n  Sampled {len(samples_df):,} requests for replay")

    # 3. 执行重放
    print("\nReplaying requests...")
    t0 = time.time()
    results = await run_replay(samples_df, waf_url)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(results)/elapsed:.0f} req/s)")

    # 4. 统计分析
    stats = analyze_results(results)

    # 5. 保存详细结果
    out_dir = Path(r"D:\training-data\waf-ml\model")
    out_path = out_dir / "replay_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
