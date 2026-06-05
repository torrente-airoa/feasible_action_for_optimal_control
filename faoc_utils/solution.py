"""Utility helpers around FAOC solution extraction."""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

EXIT_SUCCESSFUL = 0


def _collect_per_joint(faoc: Any, getter_name: str) -> list[np.ndarray]:
    """Return one numpy array per joint by calling a FAOC getter with joint index."""
    getter = getattr(faoc, getter_name)
    return [np.asarray(getter(joint_i)) for joint_i in range(faoc.get_n_joints())]


def get_solution_dict(faoc: Any, mode: str = "both") -> dict[str, list[np.ndarray]]:
    """Collect solution arrays from a FAOC instance.

    mode:
    - "last": only latest step
    - "complete": only accumulated full trajectory
    - "both": include both views
    """
    mode = mode.lower()
    if mode not in {"last", "complete", "both"}:
        raise ValueError("mode must be one of: 'last', 'complete', 'both'")

    out: dict[str, list[np.ndarray]] = {}

    if mode in {"last", "both"}:
        out["pos"] = _collect_per_joint(faoc, "get_u_sol")
        out["vel"] = _collect_per_joint(faoc, "get_du_sol")
        out["acc"] = _collect_per_joint(faoc, "get_ddu_sol")
        out["jerk"] = _collect_per_joint(faoc, "get_dddu_sol")
        out["t_u"] = [np.asarray(faoc.get_time_axis(joint_i, False)) for joint_i in range(faoc.get_n_joints())]

    if mode in {"complete", "both"}:
        out["complete_pos"] = _collect_per_joint(faoc, "get_complete_u")
        out["complete_vel"] = _collect_per_joint(faoc, "get_complete_du")
        out["complete_acc"] = _collect_per_joint(faoc, "get_complete_ddu")
        out["complete_t_u"] = [np.asarray(faoc.get_time_axis(joint_i, True)) for joint_i in range(faoc.get_n_joints())]

    return out


def stack_solution(joint_series: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
    """Stack a list of per-joint 1D arrays into shape [n_joints, n_samples]."""
    if len(joint_series) == 0:
        return np.zeros((0, 0))
    return np.vstack([np.asarray(series).reshape(1, -1) for series in joint_series])


def plot_solution(
    solution: dict[str, list[np.ndarray]],
    solution_type: str,
    final_action: Optional[tuple[float, np.ndarray]] = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot position, velocity, and acceleration for a complete FAOC solution."""
    assert solution_type in {
        "last",
        "complete",
    }, "solution_type must be 'last' or 'complete'"

    if solution_type == "last":
        time_axes = solution["t_u"]
        positions = solution["pos"]
        velocities = solution["vel"]
        accelerations = solution["acc"]
        jerk = solution["jerk"]
    else:
        time_axes = solution["complete_t_u"]
        positions = solution["complete_pos"]
        velocities = solution["complete_vel"]
        accelerations = solution["complete_acc"]
        jerk = [np.gradient(acc) / np.gradient(t) for acc, t in zip(accelerations, time_axes)]

    n_joints = len(positions)
    figure, axes = plt.subplots(4, n_joints, sharex="all", squeeze=False)

    for joint_i in range(n_joints):
        axes[0, joint_i].plot(time_axes[joint_i], positions[joint_i], "-o", markersize=2, label="Position")
        axes[1, joint_i].plot(
            time_axes[joint_i],
            velocities[joint_i],
            "-o",
            markersize=2,
            label="Velocity",
        )
        axes[2, joint_i].plot(
            time_axes[joint_i],
            accelerations[joint_i],
            "-o",
            markersize=2,
            label="Acceleration",
        )
        axes[3, joint_i].plot(time_axes[joint_i], jerk[joint_i], "-o", markersize=2, label="Jerk")
        for axis_i in range(axes.shape[0]):
            axes[axis_i, joint_i].grid()
        axes[0, joint_i].set_title(f"Joint {joint_i + 1}")
        axes[-1, joint_i].set_xlabel("Time [s]")

    if final_action is not None:
        t_f, x_f = final_action
        x_f = np.atleast_2d(x_f)
        for joint_i in range(n_joints):
            axes[0, joint_i].scatter(t_f, x_f[joint_i, 0], color="r")
            if x_f.shape[1] == 2:
                axes[1, joint_i].scatter(t_f, x_f[joint_i, 1], color="r")

    axes[0, 0].set_ylabel("Pos")
    axes[1, 0].set_ylabel("Vel")
    axes[2, 0].set_ylabel("Acc")
    axes[3, 0].set_ylabel("Jerk")

    if show:
        plt.show()

    return figure, axes
