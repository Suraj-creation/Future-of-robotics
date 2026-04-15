#!/bin/bash
set -e

echo "========================================================"
echo "    SO-101 (ACT) Action Chunking Policy Trainer"
echo "========================================================"

DATASET_PATH="LeRobot_ACT_Dataset"

if [ ! -d "$DATASET_PATH" ]; then
    echo "[!] Error: Dataset not found at $DATASET_PATH!"
    echo "    1. Run: python headless_data_generator.py (Collects data buffers)"
    echo "    2. Run: python dataset_converter.py (Constructs Parquet dataset)"
    exit 1
fi

echo "[✓] Dataset located. Initializing PyTorch MPS backpropagation..."
echo "[*] Depending on your Mac's RAM, this process could take several hours."

# LeRobot provides an easy way to train models on local datasets
python -m lerobot.scripts.train \
  --dataset.repo_id=$DATASET_PATH \
  --policy.type=act \
  --env.type=gym_pusht \
  --policy.act.dim_action=6 \
  --policy.act.chunk_size=100 \
  --policy.act.n_action_steps=100 \
  --training.log_freq=10 \
  --training.save_freq=1000 \
  --wandb.enable=false \
  --device=mps \
  --output_dir=outputs/train/act_so101_hackathon

echo "========================================================"
echo "[✓] Training Event Finished!"
echo "Weights saved to: outputs/train/act_so101_hackathon/checkpoints/"
echo "Update 'autonomous_pick_place.py' --policy_path argument to utilize new weights!"
