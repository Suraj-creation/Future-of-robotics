import os
import numpy as np
from PIL import Image

try:
    from datasets import Dataset, Features, Sequence, Value, Image as HFImage
except ImportError:
    print("Please install datasets using: pip install datasets")
    exit(1)

def convert_to_lerobot_dataset(source_dir="ACT_Dataset", out_dir="LeRobot_ACT_Dataset"):
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist. Run headless offline generator first!")
        return

    episodes = [f for f in os.listdir(source_dir) if f.endswith('.npz')]
    if not episodes:
        print(f"No npz episodes found in '{source_dir}'")
        return

    print(f"[*] Found {len(episodes)} offline episodes. Compiling to HuggingFace Parquet format...")

    data_dict = {
        "observation.image": [],
        "observation.state": [],
        "action": [],
        "episode_index": [],
        "frame_index": [],
        "timestamp": [],
        "task_index": []
    }
    
    for ep_idx, ep_file in enumerate(episodes):
        data = np.load(os.path.join(source_dir, ep_file))
        images = data['images'] # [N, H, W, 3] RGB
        qpos = data['qpos']     # [N, 6]
        ctrl = data['ctrl']     # [N, 6]
        
        for i in range(len(images)):
            # Downsample to 256x256 for VRAM efficiency during ACT memory attention if desired, or keep 640x480
            # We'll stick to original size for now
            data_dict["observation.image"].append(Image.fromarray(images[i]))
            data_dict["observation.state"].append(qpos[i].tolist())
            data_dict["action"].append(ctrl[i].tolist())
            data_dict["episode_index"].append(ep_idx)
            data_dict["frame_index"].append(i)
            # Timestamp derived from simulated steps between records
            data_dict["timestamp"].append(float(i * 0.02))
            data_dict["task_index"].append(0)
            
    print("[*] Encoding into LeRobot Dataset standard structures...")
    features = Features({
        "observation.image": HFImage(),
        "observation.state": Sequence(Value('float32'), length=6),
        "action": Sequence(Value('float32'), length=6),
        "episode_index": Value('int64'),
        "frame_index": Value('int64'),
        "timestamp": Value('float32'),
        "task_index": Value('int64')
    })
    
    hf_dataset = Dataset.from_dict(data_dict, features=features)
    
    # Save the dataset to disk compatible format
    print(f"[*] Saving chunked format to '{out_dir}'")
    hf_dataset.save_to_disk(out_dir)
    print(f"[✓] Hackathon ACT Dataset compiled successfully in tabular parquet format!")

if __name__ == '__main__':
    convert_to_lerobot_dataset()
