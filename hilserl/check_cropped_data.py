import pandas as pd
from pathlib import Path

# === 配置路径 ===
ORIGINAL_ROOT = "data"
CROPPED_ROOT = "data_cropped_resized"  # 你的新目录名

def load_dataset_stats(root_dir, name):
    path = Path(root_dir)
    print(f"📂 正在扫描 {name}: {path}")
    
    # 递归查找所有 parquet 数据文件
    files = sorted(path.glob("data/**/*.parquet"))
    if not files:
        print(f"   ❌ {name}: 未找到数据文件！")
        return None

    # 合并读取
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"   ⚠️ 无法读取 {f.name}: {e}")
    
    if not dfs: return None
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # 统计关键指标
    stats = {
        "total_frames": len(full_df),
        "num_episodes": full_df["episode_index"].nunique(),
        "success_frames": full_df["next.success"].sum() if "next.success" in full_df else 0,
        "reward_sum": full_df["next.reward"].sum() if "next.reward" in full_df else 0,
    }
    
    print(f"   📊 {name} 统计:")
    print(f"      - 总帧数: {stats['total_frames']}")
    print(f"      - 总集数: {stats['num_episodes']}")
    print(f"      - 成功帧: {stats['success_frames']}")
    
    return stats

def compare():
    print("="*40)
    print("⚖️  数据集一致性校验")
    print("="*40)
    
    orig = load_dataset_stats(ORIGINAL_ROOT, "原始数据")
    print("-" * 20)
    crop = load_dataset_stats(CROPPED_ROOT, "裁剪数据")
    print("="*40)
    
    if not orig or not crop:
        print("❌ 无法读取数据，终止。")
        return

    # 核心对比
    is_frame_match = orig['total_frames'] == crop['total_frames']
    is_ep_match = orig['num_episodes'] == crop['num_episodes']
    is_success_match = orig['success_frames'] == crop['success_frames']
    
    print("\n🧐 对比结果:")
    print(f"   1. 帧数一致性:   {'✅ 通过' if is_frame_match else '❌ 失败'} ({orig['total_frames']} vs {crop['total_frames']})")
    print(f"   2. 集数一致性:   {'✅ 通过' if is_ep_match else '❌ 失败'} ({orig['num_episodes']} vs {crop['num_episodes']})")
    print(f"   3. 成功标签一致: {'✅ 通过' if is_success_match else '❌ 失败'}")

    if is_frame_match and is_ep_match and is_success_match:
        print("\n🎉 结论：数据转换完美！没有任何丢失。")
    else:
        print("\n⚠️ 结论：发现数据差异，请检查上述红叉项。")

if __name__ == "__main__":
    compare()