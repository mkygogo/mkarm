import os
import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

def analyze_pi05_data(repo_id, root_dir="data"):
    print(f"🔍 正在分析数据集: {repo_id} ...\n")
    
    try:
        # 加载数据集
        dataset = LeRobotDataset(repo_id, root=root_dir)
    except Exception as e:
        print(f"❌ 无法加载数据集: {e}")
        return

    # 1. 修正后的基础统计
    num_episodes = dataset.num_episodes
    num_frames = dataset.num_frames # 修复 AttributeError
    print(f"📊 基础统计:")
    print(f"  - 总 Episode 数: {num_episodes} (建议: >50)")
    print(f"  - 总 帧 数: {num_frames}")
    print(f"  - FPS: {dataset.fps}")

    # 2. 检查关键特征 (Features)
    required_features = ["observation.state", "action", "observation.images.top", "observation.images.wrist"]
    print(f"\n🧩 特征完整性检查:")
    for feat in required_features:
        if feat in dataset.features:
            shape = dataset.features[feat]["shape"]
            print(f"  ✅ {feat:25} 存在, Shape: {shape}")
            
            # 检查图像是否为 pi0.5 要求的正方形 448
            if "images" in feat:
                if shape[1] != 448 or shape[2] != 448:
                    print(f"     ⚠️ 警告: 图像不是 448x448，OpenPI 处理器将报错！")
        else:
            print(f"  ❌ {feat:25} 缺失！这是 pi0.5 必需的。")

    # 3. 检查任务指令 (Task)
    print(f"\n📝 任务指令检查:")
    try:
        first_frame = dataset[0]
        if "task" in first_frame:
            print(f"  ✅ Task 字段内容预览: \"{first_frame['task']}\"")
        else:
            print(f"  ❌ 每一帧中未发现 'task' 文本。")
    except Exception:
        print(f"  ⚠️ 无法读取任务文本，请检查 tasks.parquet 是否生成。")

    # 4. 数值归一化检查 (对 pi0.5 极其关键)
    print(f"\n⚖️ 数值归一化检查 (离散化阈值检测):")
    stats = dataset.meta.stats # 从元数据读取统计
    if stats and "observation.state" in stats:
        s_min = np.array(stats["observation.state"]["min"])
        s_max = np.array(stats["observation.state"]["max"])
        
        # pi0.5 处理器假设输入在 [-1, 1] 之间进行 256-bin 离散化
        if np.any(s_min < -1.1) or np.any(s_max > 1.1):
            print(f"  ❌ 严重警告: observation.state 超出 [-1, 1] 范围!")
            print(f"     实际 Min: {s_min}")
            print(f"     实际 Max: {s_max}")
            print(f"     👉 修改建议: pi0.5 会将超出 1.0 的值全部挤在最后一个索引，导致失控。")
            print(f"     请在采集脚本存入 dataset 之前除以关节限位（如 3.14）。")
        else:
            print(f"  ✅ observation.state 范围基本符合离散化要求。")

    print(f"\n🔚 分析完成。")

if __name__ == "__main__":
    analyze_pi05_data(repo_id="mkygogo/mkrobot_pi05_cube", root_dir="data")