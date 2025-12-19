import wandb
import pandas as pd
import argparse
import os
from datetime import datetime
import sys

def export_all_metrics(run_path, output_dir="wandb_data"):
    print(f"🔄 正在连接 WandB API，尝试获取 Run: {run_path} ...")
    
    try:
        api = wandb.Api()
        run = api.run(run_path)
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        sys.exit(1)

    print(f"✅ 连接成功！Run 名称: {run.name}")
    print("🔍 正在扫描所有可用指标...")

    # 1. 尝试直接获取所有历史数据（不指定 keys）
    try:
        # samples 设置为 100000 以确保尽可能多的数据点
        history = run.history(pandas=True, samples=100000)
    except Exception as e:
        print(f"❌ 下载数据失败: {e}")
        sys.exit(1)

    if history.empty:
        print("⚠️ 警告: 该 Run 似乎没有任何历史数据记录。请确认 WandB 网页上是否能看到图表。")
        return

    # 2. 准备保存
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成文件名
    run_id = run_path.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"wandb_dump_{run_id}_{timestamp}.csv"
    file_path = os.path.join(output_dir, filename)

    # 3. 保存 CSV
    history.to_csv(file_path, index=False)
    
    print("\n✅ 导出成功！")
    print(f"📊 包含指标 (列): {list(history.columns)}")  # 打印出来看看有哪些列
    print(f"📈 数据行数: {len(history)}")
    print(f"💾 文件已保存: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出 WandB 所有指标数据")
    parser.add_argument(
        "run_path", 
        type=str, 
        help="WandB Run 的路径 (例如: mkygogo-shuaimeng/mkrobot_hil_serl/ywvnamwi)"
    )
    
    args = parser.parse_args()
    export_all_metrics(args.run_path)