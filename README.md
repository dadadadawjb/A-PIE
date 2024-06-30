# A-PIE
**Articulation-PerceptionImaginationExecution** (pronounced "*a pie*"): perception and imagination and execution around articulated objects.

## Articulation
`articulation-generation` generates required datasets of articulation objects. See [articulation-generation/README.md](articulation-generation/README.md).

## Perception
`BeyondPPF` receives point cloud and estimates the joint poses, i.e. the task of visual kinematic model estimation for articulated objects, focusing on category-level, single-view and sim2real. See [BeyondPPF/README.md](BeyondPPF/README.md).

`BeyondPPF-baseline/pointnet2` still performs the task of visual kinematic model estimation for articulated objects, but maybe in lack of the sim2real ability. See [BeyondPPF-baseline/pointnet2/README.md](BeyondPPF-baseline/pointnet2/README.md).

## Imagination
`GraspAffordance` receives point cloud, joint poses and grasp poses, then estimates each grasp pose's affordance against each joint, featured at constraining grasp space with pre-filtering by [AnyGrasp](https://graspnet.net/anygrasp.html) here. See [GraspAffordance/README.md](GraspAffordance/README.md).

## Execution
`ManiControl` contains controller and robot parts to perform manipulation and control for articulated objects, featured at non-complex design. See [ManiControl/README.md](ManiControl/README.md).

## Get Started
* `sim_pipeline.py` integrates all the 3 parts in simulation environment.
    ```bash
    # prepare your config in `configs`

    # run
    python sim_pipeline.py --config configs/*.txt
    ```

* `real_pipeline.py` experiments all the 3 parts in real environment, with RealSense L515 camera and Franka robot.
    ```bash
    # prepare your config in `configs`

    # run
    python real_pipeline.py --config configs/*.txt
    ```

## Performance
```yaml
# Microwave
pass@1: 69.756%
pass@5: 73.659%
pass@10: 74.634%
```
```yaml
# Oven
pass@1: 80.255%
pass@5: 82.166%
pass@10: 82.803%
```
```yaml
# Box
pass@1: 73.279%
pass@5: 77.328%
pass@10: 79.757%
```
```yaml
# Drawer
pass@1: 81.633%
pass@5: 86.735%
pass@10: 89.796%
```
```yaml
# Real L515+Franka Microwave
pass@1: 7 success + 5 no_affordable + 7 out_limit + 1 fail
```

## Note
This is the outstanding undergraduate thesis project in SJTU CS.
