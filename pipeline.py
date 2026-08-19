"""
一键运行全流程: 解析 → 标注 → 特征工程 → 训练
支持每步单独运行或全流程运行
"""
import sys
import time
from pathlib import Path

# 项目路径
PROJECT_DIR = Path(r"D:\training-data\waf-ml")
PARSED_DIR = PROJECT_DIR / "parsed"
LABELED_DIR = PROJECT_DIR / "labeled"
FEATURES_DIR = PROJECT_DIR / "features"
MODEL_DIR = PROJECT_DIR / "model"

# TeleAgent Python
PYTHON = "python"


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def install_deps():
    """安装必要依赖"""
    print("=== 安装依赖 ===")
    import subprocess
    deps = ['pandas', 'pyarrow', 'tqdm', 'scikit-learn', 'matplotlib', 'lightgbm']
    for dep in deps:
        print(f"  安装 {dep}...")
        subprocess.check_call(
            [PYTHON, '-m', 'pip', 'install', dep, '-q', '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple'],
            timeout=300
        )
    print("依赖安装完成\n")


def run_step(step_name: str, script: str, desc: str):
    """运行单个步骤"""
    print(f"\n{'='*60}")
    print(f"  Step: {desc}")
    print(f"{'='*60}")
    import subprocess
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, script],
        capture_output=True, text=True, timeout=3600
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[FAIL] {step_name} 失败 (耗时 {elapsed:.1f}s)")
        print(f"STDOUT:\n{result.stdout[-2000:]}")
        print(f"STDERR:\n{result.stderr[-2000:]}")
        return False

    print(f"[OK] {step_name} 完成 (耗时 {elapsed:.1f}s)")
    # 打印最后几行输出
    lines = result.stdout.strip().split('\n')
    for line in lines[-5:]:
        print(f"  {line}")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WAF-ML 全流程")
    parser.add_argument("--step", type=int, default=0,
                        help="运行指定步骤 (1=解析,2=标注,3=特征,4=训练), 0=全部")
    parser.add_argument("--skip-deps", action="store_true", help="跳过依赖安装")
    args = parser.parse_args()

    if not args.skip_deps:
        install_deps()

    steps = {
        1: ("step1_parse", str(PROJECT_DIR / "step1_parse.py"), "日志解析"),
        2: ("step2_label", str(PROJECT_DIR / "step2_label.py"), "规则标注"),
        3: ("step3_features", str(PROJECT_DIR / "step3_features.py"), "特征工程"),
        4: ("step4_train", str(PROJECT_DIR / "step4_train.py"), "模型训练"),
    }

    if args.step > 0:
        # 运行指定步骤
        name, script, desc = steps[args.step]
        ok = run_step(name, script, desc)
        if not ok:
            sys.exit(1)
    else:
        # 运行全部
        t_total = time.time()
        for step_num in sorted(steps.keys()):
            name, script, desc = steps[step_num]
            ok = run_step(name, script, desc)
            if not ok:
                print(f"\n第 {step_num} 步失败，停止执行")
                sys.exit(1)

        elapsed = time.time() - t_total
        print(f"\n{'='*60}")
        print(f"  全流程完成! 总耗时 {elapsed:.1f}s")
        print(f"  模型输出: {MODEL_DIR}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
