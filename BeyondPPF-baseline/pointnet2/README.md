# PointNet++
## Install
1. Create a conda virtual environment.
    ```bash
    conda create -n pointnet2 python=3.9
    conda activate pointnet2
    ```

2. Install [PyTorch](https://pytorch.org/).
    ```bash
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    ```

3. Install [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).

    Please make sure that you have installed `openblas` and `ninja`.

    You can install `openblas` via `sudo apt install build-essential python3-dev libopenblas-dev`.

    You can install `ninja` via `pip install ninja`.

    Then run:
    ```bash
    pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps
    ```
    If any error occurs, please refer to [MinkowskiEngine's GitHub](https://github.com/NVIDIA/MinkowskiEngine) for more information.

4. Install other dependencies.
    ```bash
    pip install -r requirements.txt
    ```

## Get Started
```bash
# prepare your dataset

# prepare your config.yaml

# train
python train.py

# test
python test.py --weight_path $WEIGHT_PATH

# inference
python inference.py --data_path $DATA_PATH --weight_path $WEIGHT_PATH

# prepare your other dataset

# prepare your config_finetune.yaml

# fine tune
python finetune.py

# test fine-tuning
python test.py --weight_path $WEIGHT_PATH

# inference fine-tuning
python inference.py --data_path $DATA_PATH --weight_path $WEIGHT_PATH
```

## Performance
```yaml
# Microwave corresponding to `weights/04-20-21-34`
avg_plane_error: 2.627 cm
avg_angle_error: 1.228 deg
```
```yaml
# Oven corresponding to `weights/04-26-16-43`
avg_plane_error: 3.450 cm
avg_angle_error: 2.060 deg
```
```yaml
# Box corresponding to `weights/04-26-22-25`
avg_plane_error: 1.397 cm
avg_angle_error: 1.668 deg
```
```yaml
# Drawer corresponding to `weights/04-26-11-57`
avg_along_error: 0.761 cm
avg_angle_error: 4.772 deg
```
```yaml
# Real L515 Microwave with table noisy points corresponding to `weights/04-20-21-34`
avg_plane_error: 29.031 cm
avg_angle_error: 24.226 deg
```
```yaml
# Real L515 Microwave without table noisy points corresponding to `weights/04-20-21-34`
avg_plane_error: 20.304 cm
avg_angle_error: 18.210 deg
```
Drop me email if you want to obtain the checkpoints.

## Note
* The SHOT feature detector may need rebuild due to mismatching with python version.
* Currently support multiple joints category-level kinematic structure estimation with optional joint states.
