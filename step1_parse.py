"""
Step 1: 解析 Nginx Access 日志 → 结构化 DataFrame (Parquet)
流式解析大文件，分块临时 pickle 存储，最后合并输出 Parquet
"""
import re
import sys
import pickle
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Nginx combined log format 正则
LOG_PATTERN = re.compile(
    r'(?P<ip>[\da-fA-F.:]+)'                     # IP (支持 IPv4/IPv6)
    r' - '
    r'(?P<user>\S+)'                               # remote_user (- 或用户名)
    r' \[(?P<time>[^\]]+)\]'                       # time_local
    r' "(?P<method>\S+)'                            # request method
    r' (?P<path>\S+)'                               # request path/URL
    r' (?P<protocol>[^"]*)"'                        # protocol
    r' (?P<status>\d{3})'                           # status code
    r' (?P<size>\d+)'                               # body_bytes_sent
    r' "(?P<referer>[^"]*)"'                        # referer
    r' "(?P<user_agent>[^"]*)"'                     # user_agent
)

CHUNK_SIZE = 200_000  # 每20万行写一次临时文件


def parse_line(line: str) -> dict | None:
    """解析单行日志，返回字典或 None"""
    m = LOG_PATTERN.match(line.strip())
    if not m:
        return None
    d = m.groupdict()
    d['status'] = int(d['status'])
    d['size'] = int(d['size'])
    if d['user'] == '-':
        d['user'] = None
    return d


def parse_file(filepath: str, source: str, output_dir: Path) -> str:
    """流式解析日志文件，分块写 pickle，最后合并输出 Parquet"""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filepath).stem
    parquet_path = output_dir / f"{stem}.parquet"

    chunk = []
    chunk_files = []
    total = 0
    parsed = 0
    failed = 0
    chunk_idx = 0

    # 先统计行数用于进度条
    print(f"[{source}] 正在统计行数...")
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        line_count = sum(1 for _ in f)
    print(f"[{source}] 共 {line_count:,} 行，开始解析...")

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in tqdm(f, total=line_count, desc=f"解析 {source}", unit="行"):
            total += 1
            d = parse_line(line)
            if d is None:
                failed += 1
                continue
            d['source'] = source
            chunk.append(d)
            parsed += 1

            if len(chunk) >= CHUNK_SIZE:
                # 写临时 pickle 分块
                tmp_path = output_dir / f"_tmp_{stem}_{chunk_idx}.pkl"
                with open(tmp_path, 'wb') as pf:
                    pickle.dump(chunk, pf, protocol=pickle.HIGHEST_PROTOCOL)
                chunk_files.append(tmp_path)
                chunk = []
                chunk_idx += 1

    # 写剩余
    if chunk:
        tmp_path = output_dir / f"_tmp_{stem}_{chunk_idx}.pkl"
        with open(tmp_path, 'wb') as pf:
            pickle.dump(chunk, pf, protocol=pickle.HIGHEST_PROTOCOL)
        chunk_files.append(tmp_path)
        chunk = []

    print(f"[{source}] 解析完成: {parsed:,} / {total:,}, 失败 {failed:,}")
    print(f"[{source}] 合并 {len(chunk_files)} 个分块...")

    # 合并分块 → Parquet
    dfs = []
    for tmp_path in tqdm(chunk_files, desc=f"合并 {source}"):
        with open(tmp_path, 'rb') as pf:
            dfs.append(pd.DataFrame(pickle.load(pf)))
        # 删除临时文件
        tmp_path.unlink()

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(parquet_path, engine='pyarrow', index=False)

    print(f"[{source}] 输出: {parquet_path} ({len(combined):,} 行)")
    return str(parquet_path)


def main():
    parser = argparse.ArgumentParser(description="解析 Nginx Access 日志")
    parser.add_argument("--www", default=r"D:\training-data\www.zstzpt.com.log")
    parser.add_argument("--data", default=r"D:\training-data\data.zstzpt.com.log")
    parser.add_argument("--out", default=r"D:\training-data\waf-ml\parsed")
    args = parser.parse_args()

    out_dir = Path(args.out)

    for filepath, source in [(args.www, "www"), (args.data, "data")]:
        if not Path(filepath).exists():
            print(f"文件不存在: {filepath}，跳过")
            continue
        parse_file(filepath, source, out_dir)

    # 合并统计
    print("\n=== 合并统计 ===")
    parquets = list(out_dir.glob("*.parquet"))
    if parquets:
        dfs = [pd.read_parquet(p) for p in parquets]
        combined = pd.concat(dfs, ignore_index=True)
        print(f"总行数: {len(combined):,}")
        print(f"字段: {list(combined.columns)}")
        print(f"\n状态码分布:\n{combined['status'].value_counts().sort_index()}")
        print(f"\n请求方法分布:\n{combined['method'].value_counts().head(10)}")
        print(f"\n来源分布:\n{combined['source'].value_counts()}")


if __name__ == "__main__":
    main()
