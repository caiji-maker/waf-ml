"""
Step 2: 规则标注 —— 向量化版本，快速处理百万级数据
为每条日志打上 attack_type 和 is_attack 标签
"""
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ===================== 规则定义 =====================

# 1. SEO 垃圾外链注入: URL 含 ?and/ 模式
SEO_SPAM_PATTERN = re.compile(r'\?and/', re.IGNORECASE)

# 2. 敏感文件扫描
SENSITIVE_FILE_PATTERN = re.compile(
    r'/\.env|/\.git|/phpinfo\.php|/\.htaccess|/web\.config|'
    r'/config\.(php|yml|yaml|json|ini)|/database\.(sql|db|sqlite)|'
    r'/backup|/\.svn|/DS_Store|/wp-config\.php',
    re.IGNORECASE
)

# 3. 漏洞扫描 (GeoServer / Actuator / Swagger / admin 等)
VULN_SCAN_PATTERN = re.compile(
    r'/geoserver|/actuator|/swagger|/api-docs|/v[123]/api-docs|'
    r'/admin(\.php)?(\?|/|$)|/manager|/console|/login\.php|'
    r'/data/admin/ver\.txt|/wp-admin|/wp-login\.php|/xmlrpc\.php',
    re.IGNORECASE
)

# 4. 路径遍历
PATH_TRAVERSAL_PATTERN = re.compile(
    r'\.\./\.\.|/etc/passwd|/etc/shadow|lang=\.\.|controllername=\w+\.\w+',
    re.IGNORECASE
)

# 5. SQL 注入
SQL_INJECTION_PATTERN = re.compile(
    r'union\s+select|(?<=[&?=])\s*(?:1=1|1\s*=\s*1|true)|'
    r'sleep\s*\(|benchmark\s*\(|concat\s*\(|information_schema|'
    r'load_file\s*\(|into\s+outfile|OR\s+1|AND\s+\d+=\d+|'
    r'CASE\s+WHEN',
    re.IGNORECASE
)

# 6. 自动化工具 UA
AUTOMATED_UA_PATTERN = re.compile(
    r'python-requests|python-urllib|curl/|wget/|Scrapy|Hutool|'
    r'go-http|Java/|Apache-HttpClient|httpclient|undici|ZmEu|'
    r'masscan|zgrab',
    re.IGNORECASE
)

# 7. 重复路径填充
PATH_PADDING_PATTERN = re.compile(r'/zstzpt\.com/zstzpt\.com', re.IGNORECASE)


def label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """向量化标注：用 str.contains + 优先级覆盖"""
    print(f"标注 {len(df):,} 行...")

    # 确保字符串列
    df['path'] = df['path'].fillna('').astype(str)
    df['user_agent'] = df['user_agent'].fillna('').astype(str)
    df['referer'] = df['referer'].fillna('').astype(str)

    # 初始化为 normal
    df['attack_type'] = 'normal'
    df['is_attack'] = 0

    # 用于组合检测的列
    path_ref = df['path'] + ' ' + df['referer']

    # 按优先级标注 (后标注的会覆盖前面的，但同一优先级内先匹配的生效)
    # 低优先级 → 高优先级

    # 6. 自动化工具 UA (低优先级，可能也是正常爬虫)
    mask = df['user_agent'].str.contains(AUTOMATED_UA_PATTERN, regex=True, na=False)
    df.loc[mask & (df['attack_type'] == 'normal'), 'attack_type'] = 'automated_tool'
    df.loc[mask & (df['is_attack'] == 0), 'is_attack'] = 1

    # 7. 重复路径填充
    mask = df['path'].str.contains(PATH_PADDING_PATTERN, regex=True, na=False)
    df.loc[mask & (df['attack_type'] == 'normal'), 'attack_type'] = 'path_padding'
    df.loc[mask & (df['is_attack'] == 0), 'is_attack'] = 1

    # 2. 敏感文件扫描
    mask = df['path'].str.contains(SENSITIVE_FILE_PATTERN, regex=True, na=False)
    df.loc[mask, 'attack_type'] = 'sensitive_scan'
    df.loc[mask, 'is_attack'] = 1

    # 3. 漏洞扫描
    mask = df['path'].str.contains(VULN_SCAN_PATTERN, regex=True, na=False)
    df.loc[mask, 'attack_type'] = 'vuln_scan'
    df.loc[mask, 'is_attack'] = 1

    # 4. 路径遍历
    mask = path_ref.str.contains(PATH_TRAVERSAL_PATTERN, regex=True, na=False)
    df.loc[mask, 'attack_type'] = 'path_traversal'
    df.loc[mask, 'is_attack'] = 1

    # 5. SQL 注入
    mask = path_ref.str.contains(SQL_INJECTION_PATTERN, regex=True, na=False)
    df.loc[mask, 'attack_type'] = 'sql_injection'
    df.loc[mask, 'is_attack'] = 1

    # 1. SEO 垃圾外链 (最高优先级，因为量最大，单独处理)
    mask = df['path'].str.contains(SEO_SPAM_PATTERN, regex=True, na=False)
    df.loc[mask, 'attack_type'] = 'seo_spam'
    df.loc[mask, 'is_attack'] = 1

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="规则标注")
    parser.add_argument("--input-dir", default=r"D:\training-data\waf-ml\parsed")
    parser.add_argument("--out", default=r"D:\training-data\waf-ml\labeled")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquets = sorted(input_dir.glob("*.parquet"))
    if not parquets:
        print("没有找到解析后的 Parquet 文件，请先运行 step1_parse.py")
        sys.exit(1)

    for pq in parquets:
        print(f"\n=== 标注 {pq.name} ===")
        df = pd.read_parquet(pq)
        print(f"加载 {len(df):,} 行")

        df = label_dataframe(df)

        print(f"\n攻击类型分布:")
        print(df['attack_type'].value_counts().to_string())
        print(f"\nis_attack 分布:")
        print(df['is_attack'].value_counts().to_string())

        out_path = out_dir / pq.name
        df.to_parquet(out_path, engine='pyarrow', index=False)
        print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
