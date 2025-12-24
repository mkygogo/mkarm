import os
import cv2
import numpy as np
import argparse
import torch
import gc
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# 直接从 LeRobot 核心库导入删除工具，避开命令行工具的 Bug
from lerobot.datasets.dataset_tools import delete_episodes

def visualize_dataset(dataset_path_str):
    dataset_path = Path(dataset_path_str).resolve()
    
    print(f"🔍 正在加载本地数据集路径: {dataset_path}")
    if not (dataset_path / "meta/info.json").exists():
        print(f"❌ 错误: 在 {dataset_path} 下找不到 meta/info.json")
        print("提示: 如果之前运行失败，请检查文件夹是否被改名为了 data_old")
        return

    # 使用本地模式加载
    dataset = LeRobotDataset(repo_id="local_data", root=dataset_path)
    num_episodes = dataset.num_episodes
    episodes_to_delete = []
    
    cv2.namedWindow("Data Visualization", cv2.WINDOW_NORMAL)
    print("\n" + "="*50)
    print("🎮 操作说明: [Space]暂停 | [K]保留 | [S]跳过 | [D]删除 | [Q]结算退出")
    print("="*50 + "\n")

    try:
        for ep_idx in range(num_episodes):
            all_indices = torch.where(torch.tensor(dataset.hf_dataset["episode_index"]) == ep_idx)[0]
            if len(all_indices) == 0: continue
            
            from_idx, to_idx = int(all_indices[0]), int(all_indices[-1]) + 1
            print(f"🎞️ 预览 Episode {ep_idx}/{num_episodes-1} ({to_idx - from_idx} 帧)...")
            
            paused, curr_frame_idx = False, from_idx
            while curr_frame_idx < to_idx:
                if not paused:
                    frame = dataset[curr_frame_idx]
                    img_top = frame["observation.images.top"].permute(1, 2, 0).numpy()
                    img_wrist = frame["observation.images.wrist"].permute(1, 2, 0).numpy()
                    if img_top.max() <= 1.0:
                        img_top, img_wrist = (img_top * 255).astype(np.uint8), (img_wrist * 255).astype(np.uint8)
                    
                    combined = np.hstack((img_top, img_wrist))
                    cv2.imshow("Data Visualization", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                    curr_frame_idx += 1

                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '): paused = not paused
                elif key in [ord('k'), ord('s')]: break
                elif key == ord('d'):
                    print(f"❌ 标记删除 Ep {ep_idx}"); episodes_to_delete.append(ep_idx); break
                elif key == ord('q'): break
            if key == ord('q'): break
    finally:
        cv2.destroyAllWindows()

    if episodes_to_delete:
        print(f"\n⚠️ 准备剔除 Episode: {episodes_to_delete}")
        if input("\n确认执行删除操作吗？(yes/no): ").lower() == 'yes':
            print("\n🚀 正在执行物理删除与重新编码...")
            try:
                # 1. 调用官方接口在缓存生成新数据
                new_dataset = delete_episodes(dataset, episode_indices=episodes_to_delete)
                
                # 2. 物理搬运逻辑
                import shutil
                source_path = Path(new_dataset.root)
                target_path = Path(dataset_path).resolve()
                
                print(f"📦 正在同步清理后的数据至: {target_path}")
                
                # 如果目标目录已存在，先删除它以防 copytree 冲突
                if target_path.exists():
                    shutil.rmtree(target_path)
                
                # 将缓存中的新数据拷贝回原始位置
                shutil.copytree(source_path, target_path)
                
                print("\n✅ 物理删除并覆盖成功！数据现在是干净的了。")
                
            except Exception as e:
                print(f"\n❌ 处理过程中发生错误: {e}")
                print("提示：如果是因为权限或目录不存在，请检查当前目录结构。")
        else:
            print("\n操作取消。")
    
    # 彻底释放句柄
    del dataset
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 这里的 root 必须指向包含 meta 文件夹的那个 'data' 目录
    parser.add_argument("--root", type=str, default="data")
    args = parser.parse_args()
    visualize_dataset(args.root)