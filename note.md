## Setup

- Environments

```
python3 -m venv .env
source .env/bin/activate

pip install pip==22.2.1
pip install setuptools==69.5.1 wheel ninja
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install h5py pandas matplotlib tqdm transforms3d open3d
sudo apt update
sudo apt install -y libopenblas-dev g++-9 gcc-9
CC=gcc-9 CXX=g++-9 pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
    --install-option="--blas=openblas"
```

- Dataset Preparation (NCLT)

```
data_root
├── 2012-01-22
│   ├── velodyne_left
│   │   ├── xxx.bin
│   │   ├── xxx.bin
│   │   ├── …
    ├── velodyne_sync
│   │   ├── xxx.bin
│   │   ├── xxx.bin
│   │   ├── …
│   ├── velodyne_left_False.h5
├── …
```

- Dataset Form Preparation (NCLT)

```
python nclt_process.py \
    --data_root /mnt/wall_maria/NCLT \
    --seqs 2012-02-12 \
    --workers 8
```

- h5 From GoogleDrive (NCLT)

```
https://drive.google.com/drive/folders/1IAPbppgy88fr3KEgcKHJHUvdC0q1TJTo
```

- Make NCLT_pose_stats.txt (train하면 만드는거라 train 전에는 직접 만들어야함. input 좌표 regularize 하는 용도?)

```
python - <<'PY'
from pathlib import Path
import h5py
import numpy as np

data_root = Path("/mnt/wall_maria/NCLT")
seqs = ["2012-02-12", "2012-02-19", "2012-03-31", "2012-05-26"]

poses = []
for seq in seqs:
    h5_path = data_root / seq / "velodyne_left_False.h5"
    print("load", h5_path)
    with h5py.File(h5_path, "r") as h5:
        poses.append(h5["poses"][5:-5])

poses = np.vstack(poses)
mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)

out = data_root / "NCLT_pose_stats.txt"
np.savetxt(out, mean_t, fmt="%8.7f")

print("saved", out)
print("mean_t:", mean_t)
PY
```

 

## Demo

TODO

 

## Train (NCLT)

- Train Classifier

```
python train.py \
    --sample_cls=True \
    --generate_clusters=True \
    --batch_size=512 \
    --epochs=50 \
    --level_cluster=100 \
    --training_buffer_size=120000 \
    --voxel_size=0.3
```

- Train Regressor

```
python train.py \
    --sample_cls=False \
    --generate_clusters=False \
    --batch_size=256 \
    --epochs=30 \
    --rsd=True \
    --prune_ratio=0.15 \
    --level_cluster=100 \
    --voxel_size=0.3
```

 

## Test (NCLT)

```
python test.py \
    --scene /mnt/wall_maria/NCLT \
    --test_seqs 2012-02-12 \
    --encoder_path weight/Backbone.pth \
    --classifier_path weight/NCLT/49_cls.pth \
    --regressor_path weight/NCLT/29_reg.pth \
    --output_path log \
    --voxel_size 0.3
```

- 2012-02-12 (in paper, 0.98, 2.76)

```
Mean Position Error(m): 1.479417
Mean XY Position Error(m): 1.457102
Mean Orientation Error(degrees): 2.772779
Median Position Error(m): 1.150624
Median XY Position Error(m): 1.134311
Median Orientation Error(degrees): 1.944432
Mean Network Cost Time(s): 0.005225
```

- 2012-02-19 (in paper, 0.89, 2.51)

```
Mean Position Error(m): 1.428108
Mean XY Position Error(m): 1.408327
Mean Orientation Error(degrees): 2.455116
Median Position Error(m): 1.120016
Median XY Position Error(m): 1.103497
Median Orientation Error(degrees): 1.793442
Mean Network Cost Time(s): 0.005558
```

- 2012-03-31 (in paper, 0.86, 2.67)

```
Mean Position Error(m): 1.388758
Mean XY Position Error(m): 1.366191
Mean Orientation Error(degrees): 2.698463
Median Position Error(m): 1.166522
Median XY Position Error(m): 1.150391
Median Orientation Error(degrees): 1.897878
Mean Network Cost Time(s): 0.005034
```

- 2012-05-26 (in paper, 3.10, 3.26)

```
Mean Position Error(m): 4.348762
Mean XY Position Error(m): 4.322061
Mean Orientation Error(degrees): 3.495975
Median Position Error(m): 1.255632
Median XY Position Error(m): 1.236081
Median Orientation Error(degrees): 2.004355
Mean Network Cost Time(s): 0.004729
```

