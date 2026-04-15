macOS pipeline scaffold for Physical AI Hackathon

Overview
--------
This folder contains a minimal scaffold to run a macOS-native MuJoCo + perception + policy pipeline.

Files
-----
- `setup_and_run.sh`: creates a Python venv and installs core packages.
- `requirements.txt`: pip packages to install.
- `run_pipeline.py`: small runner to test imports and guide next steps.
- `sim/robots/SO101.xml`: placeholder for your SO101 MJCF (place uploaded model here).

Notes / Next steps
------------------
1. Upload your SO101 MJCF to `sim/robots/SO101.xml`.
2. If you want to use DenseFusion and ACT, training must run on a GPU. You chose `cloud:yes` — I will scaffold Colab/AWS training notebooks next.
3. For cubes only, the scaffold supports an analytic pose estimator (centroid + PCA); integrate that in `run_pipeline.py` or a dedicated module `perception/pose_analytic.py`.
4. To run the setup script from the repo root:

```bash
cd ~/Downloads/Future-of-robotics-main/physical-ai-challenge-2026
./macos_pipeline/setup_and_run.sh
```

MuJoCo license / binaries
-------------------------
The `mujoco` Python package may still require MuJoCo binaries or a license key depending on the version. If `import mujoco` fails, follow the official MuJoCo install instructions and place your `mjkey.txt` under `~/.mujoco/mjkey.txt` or set `MUJOCO_KEY_PATH` accordingly.

Cloud training
--------------
I will scaffold Colab notebooks for YOLOv8 training and ACT training that you can run on a GPU. Tell me which cloud provider you prefer (Colab/GCP/AWS) and I will prepare them.
