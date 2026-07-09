[![Static checks](https://github.com/SonyResearch/feasible_action_for_optimal_control/actions/workflows/static-checks.yml/badge.svg)](https://github.com/SonyResearch/feasible_action_for_optimal_control/actions/workflows/static-checks.yml)
[![Tests](https://github.com/SonyResearch/feasible_action_for_optimal_control/actions/workflows/tests.yml/badge.svg)](https://github.com/SonyResearch/feasible_action_for_optimal_control/actions/workflows/tests.yml)

# Feasible Action for Optimal Control (FAOC)

Lightweight implementation of the Mapping Algorithm and Motion Planning Optimal Control Problem from the FAOC framework for controlling dynamical systems.
This implementation, combined with a Reinforcement Learning agent, was used for controlling the ACE robot in the 2026 work [*Outplaying elite table tennis players with an autonomous robot*](https://www.nature.com/articles/s41586-026-10338-5).

![FAOC](doc/FAOC_diagram.jpg)

> [!NOTE]
> The current implementation supports only Cubic splines and 1- or 2-dimensional action spaces.

> [!IMPORTANT]
> Currently, FAOC needs to be compiled for a specific number of joints (Degrees of Freedom)

## What is FAOC?

FAOC is a control framework that provides a simple, static action space for controlling a constrained dynamical system.
Actions selected from this action space (which we call _abstract set_) automatically yield a unique feasible trajectory to control the system in open loop for a short window of time.
This is achieved by first mapping the chosen abstract action to the set of feasible terminal constraints for an underlying motion planning problem, given the current state of the system.
This mapping is bijective, invertible, and it takes into account of the sets' shapes such that distributions do not degenerate when mapped (i.e. accumulation and dispersion of points is reduced), so it is easy to learn (e.g. via Reinforcement Learning) how to select the optimal actions.

The following is a visualization of the mapping for a particular control problem (note the shape of the target set depends on the constraints and the current state of the system and is implicitly inferred):

![Mapping](doc/Mapping.png)

See [the practical example](#example-joint-control-of-a-8-dof-robot-arm) for more information about how to use FAOC.

## Getting started
### Recommended prerequisites for building and using FAOC in an Ubuntu system:

- CMake 3.22
- Clang 14
- Eigen 3.4.0
- Python >= 3.10
- Pybind11 2.6.2
- Numpy 1.26.4 (<2 necessary)
- DAQP 0.8.6
- QHull 8.0
- C-Thread-Pool


Installation instructions:
```bash
sudo ./scripts/install_3rd_party_dependencies.sh
```

### Building and installing FAOC:
Build the FAOC framework with clang by **specifying the number of joints** (degrees of freedom) of your system.
For example, for the case of 8 joints:
```bash
mkdir build
cd build
cmake .. \
    -DFAOC_N_JOINTS=8
    -DCMAKE_C_COMPILER=/usr/bin/clang-14 \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++-14 \
    -DCMAKE_BUILD_TYPE=Release
make -j4
sudo make install
```

To install the libraries in your local user, please use the following cmake flags, e.g.:
```bash
    -DCMAKE_INSTALL_PREFIX=$HOME/.local \
    -DPYTHON_INSTALL_DIR=$HOME/.local/lib/python3/site-packages
```

## Example: Joint control of a 8-DoF robot arm
FAOC allows to generate smooth trajectories for a robot system.
First, we need to define the FAOC hyperparameters, depending on the requirements of the robot application.
All of these parameters trade-off performance and computation time.
- **Closed loop control frequency**: For this application, we choose to control the arm at 20Hz. This implies that we need to generate **trajectory segments of 50ms in length**. One segment contains one or more concatenated 3rd degree order polynomials.
- **Frequency bandwidth**: In a segment, all polynomials are the same time-length. This length determines the frequency bandwith of the motion plan, as increasing the number of polynomials given the total segment length allows for higher frequency trajectories. For very reactive systems, a recommended target would be for each polynomial segment to be about 8 to 10ms long.
- **Motion plan frequency**: This should be set to the low-level control frequency of the system. For example, 200Hz, 1000Hz, etc.
- **Action space dimension**: Our current implementation allows to have 1 or 2 dimensional abstract action spaces per joint. The first dimension maps to the terminal joint position, and the (optional) second one to the joint velocity. A 2D action space thus simultaneously constraints the terminal joint position and velocity.

### Using the Python library
After the build and installation are successful, one can import the python libraries by simply:
```py
from faoc import CubicSpline, MPOnlineSettings, JointData, ObjectiveFunction
from faoc_utils import EXIT_SUCCESSFUL, get_solution_dict, stack_solution
```

To initialize the FAOC class, the kinodynamic limits of the system must be provided.
The implementation assumes that velocity, acceleration and jerk limits are symmetric (i.e. v_max = -v_min and so on).
Additionally, whether the robot kinematic structure is symmetrical about a specific axis or not must also be specified.
For example, a humanoid robot is usually symmetric along the body length axis.
Symmetry properties can be useful for augmenting the training data by mirroring the motion plans along such an axis.
For every joint, one of the following options must be selected
- `0`: the robot does not have an axis of symmetry, or symmetric augmentation is not needed.
- `1`: the robot has an axis of symmetry, but the joint does not apply (for example, a revolute joint with the axis of rotation perpendicular to the symmetry axis)
- `-1`: otherwise
```py
    # This data corresponds to the configuration of the ACE table tennis robot
    joint_data = JointData(
        joint_mirroring=[1, -1, -1, -1, -1, -1, -1, -1],
        joint_pos_min=[-0.925, -0.675, -3.14, -1.57, -3.14, -2.44, -2.40, -6.28],
        joint_pos_max=[0.925, 0.675, 3.14, 1.57, 3.14, 2.44, 2.40, 6.28],
        joint_vel_max=[4.,  4.,  7.85,  7.85, 12.56, 12.56, 7.85, 31.42],
        joint_acc_max=[10.8, 15.7, 39.27, 39.27, 125.66, 125.66, 78.54, 628.32],
        joint_jerk_max=[ 600., 600., 750., 750., 1500., 1500., 1000., 6000.]
    )
```

The FAOC class is initialized as follows.
`MPOnlineSettings` supports optional arguments to limit the computation time for the motion planning, but for most applications, it can be initialized by default.
```py
    faoc = CubicSpline(
        tau_c=0.01,                   # Length of one polynomial in seconds
        n_l=5,                        # Number of polynomials per segment
        sampling_freq=1000,           # Sampling frequency of the robot
        abstract_set_dim=2            # Dimensionality of the action space
        joint_data=joint_data,
        online_settings=MPOnlineSettings(),
    )
    status = faoc.initialize(ObjectiveFunction.difference)
```

The initial configuration of the system must be informed to the solver as follows. For the case of the 8-DoF robot, the initial state is an 8x3 matrix composed of the initial joint positions, velocities and accelerations. The initial state must be within the kinodynamic limits specified in `JointData`.
```py
x_0 = np.zeros((8, 3))
status = faoc.set_initial_state(x_0)
```

Finally, the following steps are repeatedly executed, where `action` is an 8x1 or 8x2 matrix (depending on the abstract set dimension). 
Note that the FAOC calls should in practice be run in parallel to the execution to ensure that the next solution plan is available before the previous one ends.
```py
for step in steps:
    action = get_faoc_action()  # Using an RL agent or other control policy
    status = faoc.set_abstract_action_and_solve(action)
    sol = get_solution_dict(faoc, "last")
    robot.execute(sol)

# Reset the FAOC state for the next episode
faoc.reset()
```

### Visualization of the motion plan generated by FAOC for one action:
The 2D action mapped into the feasible set of terminal constraints is denoted with the red dot in the position and velocity rows for all joints. 
The FAOC solution consists of 5 concatenated cubic splines (with piece-wise constaint jerk) that connect the current state of the system (in position, velocity and acceleration) to the desired terminal state.
Since each spline is 10ms long, the total solution lasts for 50ms, and since the robot control frequency is 1000Hz, it has 50 samples in total. 
![Plot](doc/FAOC_traj_example.png)


For more information and examples, including the usage of the symmetric augmentation, fallback deceleration trajectory planning and more, take a look at the [Python unit tests](test/test_python_bindings.py).


## Contributing
This repository will be released as an archived repository and will not be maintained.

