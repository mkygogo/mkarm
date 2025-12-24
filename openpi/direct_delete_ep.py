import shutil
import gc
import argparse
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes

def force_delete_episodes(dataset_path_str, episode_indices):
    dataset_path = Path(dataset_path_str).resolve()
    cache_path = Path("/home/jr/.cache/huggingface/lerobot/local_data_modified")

    # 1. 强制清理缓存残留
    if cache_path.exists():
        print(f"🧹 正在清理缓存残留: {cache_path}")
        shutil.rmtree(cache_path)

    # 2. 加载数据集
    print(f"🔍 正在加载数据集: {dataset_path}")
    dataset = LeRobotDataset(repo_id="local_data", root=dataset_path)

    try:
        # 3. 执行删除操作
        print(f"🚀 正在物理删除 Episode: {episode_indices}...")
        new_dataset = delete_episodes(dataset, episode_indices=episode_indices)
        
        # 4. 彻底物理覆盖原始目录
        print(f"📦 正在同步数据至原始目录...")
        # 必须先释放句柄，否则 Windows/Linux 某些文件可能无法删除
        source_root = new_dataset.root
        del dataset
        del new_dataset
        gc.collect()

        # 移动数据
        shutil.rmtree(dataset_path)
        shutil.copytree(source_root, dataset_path)
        
        print(f"\n✅ 成功删除了 Episode {episode_indices}！")
        print(f"📢 别忘了重新运行 compute_stats.sh 来刷新统计量！")

    except Exception as e:
        print(f"\n❌ 删除失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--ep", type=int, nargs='+', required=True, help="要删除的 Episode 索引，例如 --ep 44 或 --ep 10 11 12")
    args = parser.parse_args()
    
    force_delete_episodes(args.root, args.ep)