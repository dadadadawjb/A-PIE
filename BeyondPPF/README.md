# BeyondPPF
## Install
1. Create a conda virtual environment.
    ```bash
    conda create -n beyondppf python=3.9
    conda activate beyondppf
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

4. Install [pytorch-gradual-warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr).

    ```bash
    pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
    ```

5. Install other dependencies.
    ```bash
    pip install -r requirements.txt
    ```

## Get Started
```bash
# prepare your dataset

# prepare your `config.yaml`

# train
python train.py

# test
python test.py --weight_path $WEIGHT_PATH

# inference
python inference.py --data_path $DATA_PATH --weight_path $WEIGHT_PATH

# prepare your other dataset

# prepare your `config_finetune.yaml`

# fine tune
python finetune.py

# test fine-tuning
python test.py --weight_path $WEIGHT_PATH

# inference fine-tuning
python inference.py --data_path $DATA_PATH --weight_path $WEIGHT_PATH

# manipulate to get your loop dataset

# prepare your `config_looptune.yaml`

# loop tune
python looptune.py

# test loop-tuning
python test.py --weight_path $WEIGHT_PATH

# inference loop-tuning
python inference.py --data_path $DATA_PATH --weight_path $WEIGHT_PATH
```

## Performance
```yaml
# Microwave corresponding to `weights/04-21-13-58`
avg_plane_error: 2.317 cm
avg_angle_error: 4.949 deg
```
```yaml
# Oven corresponding to `weights/04-26-15-20`
avg_plane_error: 3.204 cm
avg_angle_error: 4.576 deg
```
```yaml
# Box corresponding to `weights/04-26-20-49`
avg_plane_error: 2.466 cm
avg_angle_error: 8.879 deg
```
```yaml
# Drawer corresponding to `weights/04-26-10-37`
avg_along_error: 0.602 cm
avg_angle_error: 8.429 deg
```
```yaml
# Real L515 Microwave with table noisy points corresponding to `weights/04-21-13-58`
avg_plane_error: 10.071 cm
avg_angle_error: 11.241 deg
```
```yaml
# Real L515 Microwave without table noisy points corresponding to `weights/04-21-13-58`
avg_plane_error: 4.200 cm
avg_angle_error: 4.949 deg
```
Drop me email if you want to obtain the checkpoints.

## Note
* The SHOT feature may need rebuild due to mismatching with python version.
* The `cuda_runtime.h` include in `models/helper_math.cuh` may need modified according to the correct path.
* The `helper_math.cuh` include in `models/voting.py` may need modified according to the correct path.
* Currently support multiple joints category-level kinematic structure estimation with optional joint states.
