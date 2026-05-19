## Setup

- Environments

```
recommend : 3.8, 1.11.0, 11.3
python3 -m venv .env
source .env/bin/activate
```

```
pip install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit==11.3
pip install openblas-devel # Install MinkowskiEngine
pip uninstall setuptools -y
pip install setuptools==69.5.1
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
pip install matplotlib
pip install tqdm
pip install h5py
pip install pandas
pip install transforms3d
pip install open3d
```

```
pip install pip==22.2.1
pip install setuptools==69.5.1 wheel ninja
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
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
    --voxel_size=0.3
```

