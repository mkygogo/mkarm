rm data/meta/stats.json

python /home/jr/lerobot/lerobot/src/lerobot/datasets/v30/augment_dataset_quantile_stats.py --root data --repo-id local_data

cat data/meta/stats.json | grep -A 16 "observation.state"