import pandas as pd
import os
from pathlib import Path

# === 配置路径 ===
DATASET_ROOT = "data" 

def check_all_shards():
    root_path = Path(DATASET_ROOT)
    if not root_path.exists():
        print(f"❌ 找不到目录: {root_path}")
        return

    # 1. 自动搜索所有数据分片 (file-000, file-001, ...)
    # LeRobot 的数据存储在 data/chunk-XXX/file-XXX.parquet
    data_files = sorted(root_path.glob("data/**/*.parquet"))
    
    if not data_files:
        print("❌ 未找到任何 .parquet 数据文件！")
        return

    print(f"🔍 发现 {len(data_files)} 个数据分片文件:")
    for f in data_files:
        print(f"   - {f.relative_to(root_path)}")

    try:
        # 2. 批量读取并合并
        print("\n⏳ 正在合并所有数据...")
        dfs = []
        for f in data_files:
            df = pd.read_parquet(f)
            dfs.append(df)
            print(f"   > 已加载 {f.name}: {len(df)} 帧")
            
        full_df = pd.concat(dfs, ignore_index=True)
        
        print("\n" + "="*40)
        print(f"✅ 合并完成！")
        print(f"📊 总帧数 (Total Frames): {len(full_df)}")
        print("="*40)

        # 3. 核心字段检查
        if "next.reward" not in full_df.columns:
            print("❌ [严重] 缺少 next.reward 字段！")
            return

        # 4. 统计正负样本
        # 注意：next.success 可能是 bool 也可能是 float(0.0/1.0)，兼容处理
        success_frames = full_df[full_df["next.success"] == True]
        success_count = len(success_frames)
        
        reward_frames = full_df[full_df["next.reward"] > 0.5] # 容错，大于0.5算1
        reward_count = len(reward_frames)

        # 统计 Episode 数量 (通过 next.done)
        episode_count = full_df["next.done"].sum()

        print(f"\n📈 HIL-SERL 数据统计报告:")
        print(f"   - 录制总集数 (Episodes): {episode_count}")
        print(f"   - 成功帧数 (Reward=1)  : {reward_count}")
        print(f"   - 成功标记 (Success=T) : {success_count}")
        
        print("\n🧐 样本分布诊断:")
        if reward_count == 0:
            print("   ⚠️ [警告] 依然没有检测到成功的正样本！请确认是否按了Y键？")
        elif reward_count < 50:
            print(f"   ⚠️ [提示] 正样本较少 ({reward_count}帧)。建议录制到 50+ 个成功帧以保证分类器训练效果。")
        else:
            print(f"   ✅ [优秀] 正样本数量充足 ({reward_count}帧)。可以开始训练了！")

        # 打印最后几行看看
        print("\n📋 数据尾部采样:")
        cols = ["episode_index", "frame_index", "next.reward", "next.success", "next.done"]
        # 仅显示存在的列
        valid_cols = [c for c in cols if c in full_df.columns]
        print(full_df[valid_cols].tail(5))

    except Exception as e:
        print(f"❌ 处理失败: {e}")

if __name__ == "__main__":
    check_all_shards()