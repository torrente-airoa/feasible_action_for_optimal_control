import itertools
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from faoc_utils.solution import EXIT_SUCCESSFUL
from faoc import CubicSpline, MPOnlineSettings, JointData, ObjectiveFunction


def sample_mapping(faoc: CubicSpline, sampled_points: np.ndarray, initial_state: np.ndarray) -> np.ndarray:
    """
    Samples the mapping from abstract space to joint space for a given FAOC object and a set of abstract points.
    Args:
        faoc (CubicSpline): FAOC object to sample the mapping from
        sampled_points (np.ndarray): Points in the abstract space to be mapped (Nx3 for cubic)
        Returns:
            np.ndarray: Mapped points in the joint space
    """
    mapped_points = np.zeros_like(sampled_points)
    abs_action = np.zeros((faoc.get_n_joints(), 2))

    # Computing joint-space actions for only one joint
    for i, point in enumerate(sampled_points):
        faoc.reset()
        faoc.set_initial_state(initial_state)
        abs_action[0, :] = point
        assert faoc.set_abstract_action_and_solve(abs_action) == EXIT_SUCCESSFUL
        action = faoc.get_last_joint_action()[0]
        mapped_points[i, 0] = action[0]
        mapped_points[i, 1] = action[1]
    return mapped_points


def visualize_color_mapping(n_points: int = 1000, sampling_modes: Optional[list[str]] = None):
    """Visualizes the mapping between abstract space and feasible polytope"""

    if sampling_modes is None:
        sampling_modes = ["grid"]

    joint_data = JointData(
        joint_mirroring=[1, -1, -1, -1, -1, -1, -1, -1],
        joint_pos_min=[
            -0.925,
            -0.675,
            -3.14159,
            -1.5708,
            -3.14159,
            -2.44346095,
            -2.40855437,
            -6.28319,
        ],
        joint_pos_max=[
            0.925,
            0.675,
            3.14159,
            1.5708,
            3.14159,
            2.44346095,
            2.40855437,
            6.28319,
        ],
        joint_vel_max=[4.0, 4.0, 7.85398, 7.85398, 12.5664, 12.5664, 7.8539816, 31.41593],
        joint_acc_max=[10.8, 15.7, 39.27, 39.27, 125.66, 125.66, 78.54, 628.32],
        joint_jerk_max=[600.0, 600.0, 750.0, 750.0, 1500.0, 1500.0, 1000.0, 6000.0],
    )

    faoc = CubicSpline(
        tau_c=0.01,
        n_l=4,
        sampling_freq=1000,
        n_joints=len(joint_data.mirroring_logic),
        joint_data=joint_data,
        online_settings=MPOnlineSettings(),
        abstract_set_dim=2,
    )
    faoc.initialize(ObjectiveFunction.difference)

    _, axes = plt.subplots(
        len(sampling_modes),
        2,
        gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1] * len(sampling_modes)},
        figsize=(6, 3 * len(sampling_modes)),
    )
    axes = np.atleast_2d(axes)

    axes[0, 0].set_title("Abstract points", fontsize=12)
    axes[0, 1].set_title("Mapped points", fontsize=12)

    initial_state = np.zeros((faoc.get_n_joints(), 3))

    for j, sampling_mode in enumerate(sampling_modes):
        if sampling_mode == "grid":
            n_points_per_axis = int(np.sqrt(n_points))
            axis_points = np.linspace(-0.98, 0.98, n_points_per_axis)
            sampled_points = np.array(list(itertools.product(axis_points, axis_points)))
            mean_point = np.array([0.0, 0.0])
        elif sampling_mode == "squashed_gaussian":
            mean = np.array([0.5, -0.5])
            std = np.array([0.5, 0.8])
            sampled_points = np.tanh(np.random.multivariate_normal(mean=mean, cov=np.diag(std**2), size=n_points))
            mean_point = np.tanh(mean)
        elif sampling_mode == "uniform":
            x_lin = np.random.uniform(-0.98, 0.98, size=n_points)
            y_lin = np.random.uniform(-0.98, 0.98, size=n_points)
            mean_point = np.array([0.0, 0.0])
            sampled_points = np.stack((x_lin, y_lin), axis=-1)
        else:
            raise ValueError(f"Unknown sampling mode {sampling_mode}")

        for axis in axes[j, :]:
            axis.tick_params(axis="both", which="both", length=0)
            axis.tick_params(labelleft=False, labelbottom=False)

        # Build per-point colors from normalised abstract-space coordinates so
        # that the mapping is correct regardless of x_lin / y_lin ordering.
        normalized_points = (sampled_points + 1) / 2
        # R = x, G = (1-x)*(1-y), B = y
        colors = np.stack(
            [
                normalized_points[:, 0],
                (1 - normalized_points[:, 0]) * (1 - normalized_points[:, 1]),
                normalized_points[:, 1],
            ],
            axis=-1,
        )

        size = 2.5
        edges_abs = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [-1, -1]], dtype=float)
        axes[j, 0].scatter(sampled_points[:, 0], sampled_points[:, 1], c=colors, s=size, marker="o", edgecolors="none")
        axes[j, 0].scatter(mean_point[0], mean_point[1], s=25, marker="o", edgecolors="white", facecolors="k")
        axes[j, 0].plot(edges_abs[:, 0], edges_abs[:, 1], color="black", linewidth=1)
        axes[j, 0].set_xlim(-1.1, 1.1)
        axes[j, 0].set_ylim(-1.1, 1.1)

        faoc.reset()
        faoc.set_initial_state(initial_state)

        # Get vertices of parameter space
        edges_aux = []
        for i in range(len(edges_abs) - 1):
            actions = np.stack(
                (
                    np.linspace(edges_abs[i][0], edges_abs[i + 1][0], 50),
                    np.linspace(edges_abs[i][1], edges_abs[i + 1][1], 50),
                ),
                axis=1,
            )
            edges_aux.append(sample_mapping(faoc=faoc, sampled_points=actions, initial_state=initial_state))
        edges = np.vstack(edges_aux)

        points = sample_mapping(faoc=faoc, sampled_points=sampled_points, initial_state=initial_state)
        mean_map = sample_mapping(faoc=faoc, sampled_points=mean_point[np.newaxis], initial_state=initial_state)
        axes[j, 1].scatter(points[:, 0], points[:, 1], c=colors, s=size, marker="o", edgecolors="none")
        axes[j, 1].scatter(mean_map[0, 0], mean_map[0, 1], s=25, marker="o", edgecolors="white", facecolors="k")
        axes[j, 1].plot(edges[:, 0], edges[:, 1], color="black", linewidth=1)

    plt.tight_layout()
    # plt.savefig("mapping_comparison.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    visualize_color_mapping(n_points=5000, sampling_modes=["grid"])
