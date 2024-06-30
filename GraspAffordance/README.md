# GraspAffordance
## Install
1. Create a conda virtual environment.
    ```bash
    conda create -n grasp-affordance python=3.9
    conda activate grasp-affordance
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

# prepare your config.yaml

# train
python train.py

# test
python test.py --weight_path $WEIGHT_PATH

# inference
python inference.py --data_path $DATA_PATH --joints_path $JOINTS_PATH --grasps_path $GRASPS_PATH --weight_path $WEIGHT_PATH

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
# Microwave 4 classification corresponding to `weights/04-23-20-09`
Accuracy: 78.346%
Precision: 84.373%
```
```yaml
# Oven 4 classification corresponding to `weights/04-27-10-57`
Accuracy: 72.483%
Precision: 75.277%
```
```yaml
# Box 4 classification corresponding to `weights/04-27-19-00`
Accuracy: 85.520%
Precision: 90.205%
```
```yaml
# Drawer 4 classification corresponding to `weights/04-27-09-18`
Accuracy: 82.560%
Precision: 90.448%
```
```yaml
# Real L515 Microwave without table noisy points 4 classification corresponding to `weights/04-23-20-09`
Accuracy: 66.177%
Precision: 92.819%
```
Drop me email if you want to obtain the checkpoints.

## Note
* The SHOT feature may need rebuild due to mismatching with python version.
* Currently support multiple joints grasp affordance estimation with optional joint states.
