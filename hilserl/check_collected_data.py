import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
import numpy as np

# === 修改配置适配新数据 ===
REPO_ID = "mkygogo/mk_arm_hil_serl_v1"  # 你的新数据集ID
ROOT_DIR = "data"                       # 你的新数据存放目录

def check_dataset():
    print(f"🚀 开始检查数据集: {REPO_ID} (Root: {ROOT_DIR})")
    
    try:
        # 1. 加载数据集
        dataset = LeRobotDataset(repo_id=REPO_ID, root=ROOT_DIR)
        print(f"✅ 数据集加载成功！")
        print(f"📊 总帧数 (Total Frames): {len(dataset)}")
        print(f"🎬 总集数 (Total Episodes): {dataset.num_episodes}")
        
        # 计算平均帧数
        avg_frames = len(dataset) / dataset.num_episodes if dataset.num_episodes > 0 else 0
        print(f"📏 平均每集帧数: {avg_frames:.1f}")

        # 2. 探查数据结构 (读取第一帧)
        if len(dataset) > 0:
            item = dataset[0]
            print("\n🔍 [数据结构探查] 第一帧包含的字段:")
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    info = f"Tensor shape={list(value.shape)}"
                elif isinstance(value, np.ndarray):
                    info = f"Numpy shape={value.shape}"
                else:
                    info = f"Type={type(value)}"
                print(f"   - {key:<30} : {info}")
            
            # 检查是否有 reward
            has_reward = any("reward" in k for k in item.keys())
            if not has_reward:
                print("\n⚠️ [注意] 数据中未检测到 'reward' 或 'next.reward' 字段。")
                print("   -> 这意味着无法直接通过脚本统计 '成功率'。")
                print("   -> 如果你是做 HIL-SERL，需要在采集代码中把 is_success 写入 frame['next.reward']。")
        
        # 3. 批量扫描 (检查是否有坏数据)
        print("\n正在快速扫描所有数据完整性...")
        dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
        
        for i, batch in enumerate(dataloader):
            # 简单的非空检查
            if batch['action'].isnan().any():
                print(f"❌ 发现 NaN (空值) 动作数据，在第 {i} 批次！")
            
            if i % 10 == 0:
                print(f"   已扫描 {i*32}/{len(dataset)} 帧...", end='\r')
                
        print(f"\n✅ 扫描完成！数据格式基本完整。")

    except Exception as e:
        print(f"\n❌ 检查过程中出错: {e}")
        # 如果是找不到数据集，提示路径
        import os
        if not os.path.exists(os.path.join(ROOT_DIR, REPO_ID)):
            print(f"   可能原因: 目录 '{os.path.join(ROOT_DIR, REPO_ID)}' 不存在。")

if __name__ == "__main__":
    check_dataset()