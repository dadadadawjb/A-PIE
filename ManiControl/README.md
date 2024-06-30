# ManiControl
## Install
1. Create a conda virtual environment.
    ```bash
    conda create -n mani-control python=3.9
    conda activate mani-control
    ```

2. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```

## Get Started
It acts as module, containing controller, robot and loop parts.

Currently support PID controller, panda gripper and joint loop-tunning.

The manipulation control in real environment see `real`.
