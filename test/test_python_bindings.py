import os
import json
import unittest
import numpy as np
import matplotlib.pyplot as plt
from faoc import CubicSpline, MPOnlineSettings, JointData, ObjectiveFunction
from faoc_utils import EXIT_SUCCESSFUL, get_solution_dict, plot_invariant_set, plot_solution, stack_solution

FAOC_PLOT_TESTS = 0

JOINT_DATA = JointData(
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
N_JOINTS = len(JOINT_DATA.mirroring_logic)

INIT = np.array(
    [
        [0.646357, -0.422468],
        [0.396339, 1.19376],
        [2.58645, -2.37543],
        [-0.517664, 2.10667],
        [-1.39628, 0.678208],
        [-0.110459, 1.61944],
        [-0.651348, 0.105251],
        [5.68289, 13.0752],
    ]
)
ABSTRACT_STEPS = np.array(
    [
        [
            [0.271423, 0.434593],
            [-0.716794, 0.213938],
            [-0.967398, -0.514226],
            [-0.725536, 0.608353],
            [-0.686641, -0.198111],
            [-0.740418, -0.782382],
            [0.997848, -0.563486],
            [0.0258648, 0.678224],
        ],
        [
            [0.225279, -0.407936],
            [0.275104, 0.0485743],
            [-0.012834, 0.945549],
            [-0.414966, 0.542715],
            [0.0534899, 0.539827],
            [-0.199543, 0.783058],
            [-0.43337, -0.295083],
            [0.615448, 0.838052],
        ],
        [
            [-0.860489, 0.898653],
            [0.0519906, -0.827887],
            [-0.615572, 0.326454],
            [0.780464, -0.302214],
            [-0.871656, -0.959953],
            [-0.0845964, -0.873807],
            [-0.52344, 0.941267],
            [0.804415, 0.701839],
        ],
        [
            [-0.466668, 0.0795206],
            [-0.249586, 0.520497],
            [0.0250707, 0.335447],
            [0.0632128, -0.921438],
            [-0.124725, 0.863669],
            [0.861619, 0.441904],
            [-0.431413, 0.477068],
            [0.279957, -0.291902],
        ],
        [
            [0.375722, -0.668051],
            [-0.119791, 0.76015],
            [0.658402, -0.339325],
            [-0.542063, 0.786744],
            [-0.299279, 0.373339],
            [0.912936, 0.17728],
            [0.314608, 0.717352],
            [-0.12088, 0.847939],
        ],
    ]
)


def xy_mirror(vec: np.ndarray) -> np.ndarray:
    vec[1:, :] *= -1
    return vec


def plots_enabled() -> bool:
    return int(os.getenv("FAOC_PLOT_TESTS", str(FAOC_PLOT_TESTS))) == 1


# pylint: disable=too-many-statements
def test_abstract_multistep():
    """This test first uses the pre-defined initial joint state and abstract actions to recursively call the multistep
    FAOC planner. While doing that, saves the pairs (abstract_action, found joint action). Then, does the same in
    reverse; starting from the initial joint state, finds back the abstract_action's from the found_joint_action's.
    """
    tau_c = 0.01  # the length of a single polynomial in seconds
    n_l = 5  # the number of polynomials per solution
    f_s = 1000  # the sampling frequency of the solution in Hz

    faoc = CubicSpline(
        tau_c=tau_c,
        n_l=n_l,
        sampling_freq=f_s,
        n_joints=N_JOINTS,
        joint_data=JOINT_DATA,
        online_settings=MPOnlineSettings(),
        abstract_set_dim=2,
    )

    # We also generate a mirrored solution
    faoc_mirrored = CubicSpline(
        tau_c=tau_c,
        n_l=n_l,
        sampling_freq=f_s,
        n_joints=N_JOINTS,
        joint_data=JOINT_DATA,
        online_settings=MPOnlineSettings(),
        abstract_set_dim=2,
    )
    tolerance_inv_map = 1e-5

    x0_3d = np.hstack((INIT, np.zeros((8, 1))))

    # Initialize both FAOC's
    gains = 0.5 * np.ones(faoc.get_n_joints())
    z_tol = np.zeros(faoc.get_n_joints())
    for faoc_i in [faoc, faoc_mirrored]:
        assert faoc_i.set_velocity_limit_gain(gains) == EXIT_SUCCESSFUL
        assert faoc_i.set_acceleration_limit_gain(gains) == EXIT_SUCCESSFUL
        assert faoc_i.set_tolerances(p_tol=z_tol, v_tol=z_tol) == EXIT_SUCCESSFUL
        assert faoc_i.initialize(ObjectiveFunction.difference) == EXIT_SUCCESSFUL

    assert faoc.is_zero_tolerance

    # Get the size of the step in samples (will be 40 in this case)
    step_size = faoc.get_step_size()
    assert step_size == tau_c * f_s * n_l

    # We will be stacking the solution ourselves for testing purposes
    full_t_sol = np.zeros(0)
    full_u_sol = np.zeros((faoc.get_n_joints(), 0))

    # Set feasible initial joint state
    if faoc.get_joint_space_dimension() == 2:
        assert faoc.set_initial_state(INIT) == EXIT_SUCCESSFUL
    else:
        assert faoc.set_initial_state(x0_3d) == EXIT_SUCCESSFUL

    # No solution yet
    assert len(get_solution_dict(faoc, "last")["pos"][0]) == 0

    joint_actions = []
    joint_states = []

    # Step with abstract actions
    for i, action in enumerate(ABSTRACT_STEPS):
        print(f"Fwd step {i}")
        centroids = np.zeros((faoc.get_n_joints(), faoc.get_abstract_space_dimension()))
        for j in range(faoc.get_n_joints()):
            code, centroids[j, :] = faoc.get_unique_centroid(j)
            assert code == EXIT_SUCCESSFUL
        code, centroid = faoc.map_to_abstract_set(centroids)
        assert code == EXIT_SUCCESSFUL
        assert np.isclose(np.sum(centroid), 0.0, atol=1e-9)

        assert faoc.set_abstract_action_and_solve(action) == EXIT_SUCCESSFUL

        sol = get_solution_dict(faoc, "both")
        t_sol = sol["t_u"][0]  # For the time, we can keep just the array for one joint (they are all the same)
        u_sol = stack_solution(sol["pos"])
        final_action = (
            (faoc.get_fusion_index() + step_size - 1) / f_s,
            faoc.get_last_joint_action(),
        )
        for j in range(faoc.get_n_joints()):
            assert len(sol["vel"][j]) == len(sol["pos"][j]) == len(sol["acc"][j]) == step_size
            assert (
                len(sol["complete_vel"][j])
                == len(sol["complete_pos"][j])
                == len(sol["complete_acc"][j])
                == (i + 1) * step_size
            )
            assert np.all(np.isclose(sol["complete_vel"][j][-step_size:], sol["vel"][j]))
            assert np.all(np.isclose(sol["complete_acc"][j][-step_size:], sol["acc"][j]))

        if plots_enabled():
            plot_solution(sol, solution_type="last", final_action=final_action, show=False)

        # Save action in joint space for second part of test
        joint_actions += [faoc.get_last_joint_action()]
        joint_states += [faoc.get_last_joint_state()]

        # Due to the zero tolerance, the chosen action must be the same as the last pos
        assert np.isclose(u_sol[:, -1], joint_actions[-1][:, 0]).all()
        assert np.isclose(u_sol[:, -1], joint_states[-1][:, 0]).all()

        if i == 0:
            # Check that the position trajectory starts from one step ahead of the initial state
            assert not np.isclose(u_sol[:, 0], INIT[:, 0], atol=1e-10).all()

        # The solutions are one sample longer than the step_size since they overlap
        assert len(t_sol) == u_sol.shape[1] == step_size

        # Get fusion index and verify its correctness
        fusion_idx = faoc.get_fusion_index()
        assert fusion_idx == i * step_size
        if i > 0:
            # The current solution must also not overlap with the previous one
            assert not np.isclose(full_u_sol[:, -1], u_sol[:, 0], atol=1e-10).all()

        # Concatenate
        full_t_sol = np.concatenate((full_t_sol[:fusion_idx], t_sol))
        full_u_sol = np.hstack((full_u_sol[:, :fusion_idx], u_sol))

    if plots_enabled():
        plt.show()

    complete_u = stack_solution(get_solution_dict(faoc, "complete")["complete_pos"])
    assert complete_u.shape == full_u_sol.shape
    assert np.sum(np.linalg.norm(complete_u - full_u_sol, axis=1)) < 1e-10

    # Now go backwards and extract the abstract actions from the solution
    faoc.reset()

    mirrored_actions = []

    # Set initial joint state
    if faoc.get_joint_space_dimension() == 2:
        assert faoc.set_initial_state(INIT) == EXIT_SUCCESSFUL
        assert faoc_mirrored.set_initial_state(faoc_mirrored.mirror_joint_state(INIT)) == EXIT_SUCCESSFUL
    else:
        assert faoc.set_initial_state(x0_3d) == EXIT_SUCCESSFUL
        assert faoc_mirrored.set_initial_state(faoc_mirrored.mirror_joint_state(x0_3d)) == EXIT_SUCCESSFUL

    for i, joint_action in enumerate(joint_actions):
        print(f"Bwd step {i}")
        res = faoc.map_to_abstract_set(joint_action)
        assert res[0] == EXIT_SUCCESSFUL
        assert np.isclose(res[1], ABSTRACT_STEPS[i], atol=tolerance_inv_map).all()

        res_mirrored = faoc_mirrored.map_to_abstract_set(faoc_mirrored.mirror_joint_state(joint_action))
        assert res_mirrored[0] == EXIT_SUCCESSFUL

        assert faoc.set_abstract_action_and_solve(res[1]) == EXIT_SUCCESSFUL
        assert faoc_mirrored.set_abstract_action_and_solve(res_mirrored[1]) == EXIT_SUCCESSFUL

        mirrored_actions.append(res_mirrored[1])

    # Reconstruct mirrored solution to compare it with the forward solution
    complete_u = stack_solution(get_solution_dict(faoc, "complete")["complete_pos"])
    complete_u_mirror = stack_solution(get_solution_dict(faoc_mirrored, "complete")["complete_pos"])
    assert complete_u.shape == complete_u_mirror.shape == full_u_sol.shape
    assert np.sum(np.linalg.norm(complete_u - full_u_sol, axis=1)) < 1e-9
    assert np.sum(np.linalg.norm(xy_mirror(complete_u_mirror) - full_u_sol, axis=1)) < 1e-9


def test_tolerance():
    """Test that the tolerant mode of FAOC respects the velocity limits"""
    f_s = 1000
    tau_c = 0.01
    n_l = 5
    faoc = CubicSpline(
        tau_c=tau_c,
        n_l=n_l,
        sampling_freq=f_s,
        n_joints=N_JOINTS,
        joint_data=JOINT_DATA,
        online_settings=MPOnlineSettings(),
        abstract_set_dim=2,
    )

    pos_tolerance = np.array([0.0002, 0.0002, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003])
    vel_tolerance = np.array([0.08, 0.08, 0.157, 0.157, 0.25, 0.25, 0.157, 0.62])
    v_max_ratio = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    a_max_ratio = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    assert faoc.set_tolerances(p_tol=pos_tolerance, v_tol=vel_tolerance) == EXIT_SUCCESSFUL
    assert faoc.set_velocity_limit_gain(v_max_ratio) == EXIT_SUCCESSFUL
    assert faoc.set_acceleration_limit_gain(a_max_ratio) == EXIT_SUCCESSFUL
    assert faoc.initialize(ObjectiveFunction.magnitude) == EXIT_SUCCESSFUL

    with open(os.path.join(os.path.dirname(__file__), "rl_test_data.json"), encoding="utf8") as file:
        loads = json.load(file)["dataset"]

    x0_2d = np.squeeze(np.stack(loads[0]))
    x0_3d = np.hstack((x0_2d, np.zeros((faoc.get_n_joints(), 1))))
    assert faoc.set_initial_state(x_0=x0_3d) == EXIT_SUCCESSFUL

    for action in loads[1]:
        assert faoc.set_abstract_action_and_solve(np.stack(action)) == EXIT_SUCCESSFUL

    complete_sol = get_solution_dict(faoc, "complete")

    for j in range(faoc.get_n_joints()):
        assert np.all(-JOINT_DATA.vel_max[j] * v_max_ratio[j] * 1.00001 <= complete_sol["complete_vel"][j])
        assert np.all(complete_sol["complete_vel"][j] <= JOINT_DATA.vel_max[j] * v_max_ratio[j] * 1.00001)
        assert np.all(-JOINT_DATA.acc_max[j] * a_max_ratio[j] * 1.00001 <= complete_sol["complete_acc"][j])
        assert np.all(complete_sol["complete_acc"][j] <= JOINT_DATA.acc_max[j] * a_max_ratio[j] * 1.00001)

        p_diff = np.gradient(complete_sol["complete_pos"][j]) / np.gradient(complete_sol["complete_t_u"][j])
        assert np.median(np.abs(p_diff[1:-1] - complete_sol["complete_vel"][j][1:-1])) < 1e-4

    if plots_enabled():
        plot_solution(complete_sol, solution_type="complete")


def test_tolerance_with_mirroring_and_solving():
    x_init = np.array(
        [
            [-0.25, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    abstract_action_1 = np.array(
        [
            [0.35439157, 0.8461826],
            [0.48252606, -0.6166909],
            [-0.51079535, -0.5686023],
            [-0.7092712, -0.9548507],
            [0.68722177, 0.91676664],
            [0.10141778, -0.5556567],
            [0.00449109, -0.87670994],
            [0.5572705, 0.8474796],
        ]
    )

    abstract_action_2 = np.array(
        [
            [0.8928478, -0.4453938],
            [-0.18884611, 0.4706881],
            [0.22879648, -0.14419556],
            [-0.01165986, 0.9106164],
            [-0.818161, -0.25259924],
            [0.35097432, -0.5955472],
            [-0.07742786, -0.40463257],
            [0.51541066, 0.2928276],
        ]
    )

    f_s = 1000
    tau_c = 0.01
    n_l = 5
    pos_tolerance = np.array([0.0002, 0.0002, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003])
    vel_tolerance = np.array([0.08, 0.08, 0.157, 0.157, 0.25, 0.25, 0.157, 0.62])
    gains = np.array([0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    faoc = CubicSpline(
        tau_c=tau_c,
        n_l=n_l,
        sampling_freq=f_s,
        n_joints=N_JOINTS,
        joint_data=JOINT_DATA,
        online_settings=MPOnlineSettings(),
        abstract_set_dim=2,
    )
    assert faoc.set_tolerances(p_tol=pos_tolerance, v_tol=vel_tolerance) == EXIT_SUCCESSFUL
    assert faoc.set_velocity_limit_gain(gains) == EXIT_SUCCESSFUL
    assert faoc.set_acceleration_limit_gain(gains) == EXIT_SUCCESSFUL
    assert faoc.initialize(ObjectiveFunction.difference) == EXIT_SUCCESSFUL

    faoc_mirror = CubicSpline(
        tau_c=tau_c,
        n_l=n_l,
        sampling_freq=f_s,
        n_joints=N_JOINTS,
        joint_data=JOINT_DATA,
        online_settings=MPOnlineSettings(),
        abstract_set_dim=2,
    )
    assert faoc_mirror.set_tolerances(p_tol=pos_tolerance, v_tol=vel_tolerance) == EXIT_SUCCESSFUL
    assert faoc_mirror.set_velocity_limit_gain(gains) == EXIT_SUCCESSFUL
    assert faoc_mirror.set_acceleration_limit_gain(gains) == EXIT_SUCCESSFUL
    assert faoc_mirror.initialize(ObjectiveFunction.difference) == EXIT_SUCCESSFUL

    faoc.reset()
    faoc_mirror.reset()

    assert faoc.set_initial_state(x_init[:, : faoc.get_joint_space_dimension()]) == EXIT_SUCCESSFUL
    assert (
        faoc_mirror.set_initial_state(faoc.mirror_joint_state(x_init[:, : faoc.get_joint_space_dimension()]))
        == EXIT_SUCCESSFUL
    )

    print("> Step 1")
    assert faoc.set_abstract_action_and_solve(abstract_action_1) == EXIT_SUCCESSFUL
    joint_action_1 = faoc.get_last_joint_action()
    abs_action_mirror_1 = faoc_mirror.map_to_abstract_set(faoc_mirror.mirror_joint_state(joint_action_1))
    assert abs_action_mirror_1[0] == EXIT_SUCCESSFUL
    assert faoc_mirror.set_abstract_action_and_solve(abs_action_mirror_1[1]) == EXIT_SUCCESSFUL

    print("> Step 2")
    assert faoc.set_abstract_action_and_solve(abstract_action_2) == EXIT_SUCCESSFUL
    joint_action_2 = faoc.get_last_joint_action()
    abs_action_mirror_2 = faoc_mirror.map_to_abstract_set(faoc_mirror.mirror_joint_state(joint_action_2))
    assert abs_action_mirror_2[0] == EXIT_SUCCESSFUL
    assert faoc_mirror.set_abstract_action_and_solve(abs_action_mirror_2[1]) == EXIT_SUCCESSFUL

    print("Done")


def test_reset_faoc():
    f_s = 1000
    tau_c = 0.01
    n_l = 5
    faoc = CubicSpline(
        tau_c=tau_c,
        n_l=n_l,
        sampling_freq=f_s,
        n_joints=N_JOINTS,
        joint_data=JOINT_DATA,
        online_settings=MPOnlineSettings(),
        abstract_set_dim=2,
    )

    faoc.set_velocity_limit_gain(np.ones(N_JOINTS) * 0.7)
    faoc.set_acceleration_limit_gain(np.ones(N_JOINTS) * 0.7)
    faoc.initialize(ObjectiveFunction.magnitude)

    x0_3d = np.hstack((INIT, np.zeros((N_JOINTS, 1))))
    p_f_low = np.array([0.0, -0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p_f_up = np.array([0.8, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p_f = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    reset_kwargs = dict(
        p_reset_low=p_f_low,
        p_reset_up=p_f_up,
        mult=2,
        add_steps=3,
        n_threads=1,
        max_n_l=300,
        reset_sync=True,
    )

    # Will fail since p_f_up[1] > p_max[1] and p_f_low[1] < p_min[1]
    assert faoc.initialize_reset_planner(**reset_kwargs) != EXIT_SUCCESSFUL

    # Now it should work
    reset_kwargs["p_reset_up"][1] = 0.6
    reset_kwargs["p_reset_low"][1] = -0.6
    assert faoc.initialize_reset_planner(**reset_kwargs) == EXIT_SUCCESSFUL

    assert faoc.compute_reset_trajectory(current_state=x0_3d, reset_pos=p_f) == EXIT_SUCCESSFUL
    reset_pos, reset_vel, reset_acc, reset_jerk = faoc.get_reset_plan()
    assert not np.any(reset_vel[-3] == 0)  # When using reset_sync, all joints should reach 0 velocity almost at the end
    print(f"Final pos:\n{reset_pos[-5:]}")
    print(f"Final vel:\n{reset_vel[-5:]}")
    print(f"Final acc:\n{reset_acc[-5:]}")
    print(f"Final jerk:\n{reset_jerk[-5:]}")
    assert np.linalg.norm(reset_pos[-1] - p_f) < 1e-10
    assert np.linalg.norm(reset_vel[-1]) < 1e-10
    assert np.linalg.norm(reset_acc[-1]) < 1e-10
    # Jerk can be anything inside the limits


def test_faoc_approx_1d():
    f_s = 1000
    tau_c = 0.01
    n_l = 5
    faoc = CubicSpline(
        tau_c=tau_c,
        n_l=n_l,
        sampling_freq=f_s,
        n_joints=N_JOINTS,
        joint_data=JOINT_DATA,
        online_settings=MPOnlineSettings(),
        abstract_set_dim=1,  # Set to 1D
    )

    assert faoc.get_abstract_space_dimension() == 1
    assert faoc.get_n_joints() == 8

    faoc.set_velocity_limit_gain(np.ones(faoc.get_n_joints()) * 0.7)
    faoc.set_acceleration_limit_gain(np.ones(faoc.get_n_joints()) * 0.7)
    faoc.initialize(ObjectiveFunction.magnitude)

    assert faoc.set_initial_state(np.zeros((faoc.get_n_joints(), faoc.get_joint_space_dimension()))) == EXIT_SUCCESSFUL

    for i in range(20):
        print(f"Fwd step {i}")
        assert (
            faoc.set_abstract_action_and_solve(
                np.random.uniform(
                    -1.0,
                    1.0,
                    (faoc.get_n_joints(), faoc.get_abstract_space_dimension()),
                )
            )
            == EXIT_SUCCESSFUL
        )


def test_invariant_set_unscaled():
    """The MCIS returned in physical units must agree with the internal membership check, also when the position
    limits are not symmetric about zero."""
    joint_data = JointData(
        joint_mirroring=[1, 1],
        joint_pos_min=[-0.2, -0.675],
        joint_pos_max=[1.0, 0.675],
        joint_vel_max=[4.0, 4.0],
        joint_acc_max=[10.8, 15.7],
        joint_jerk_max=[600.0, 600.0],
    )
    n_joints = len(joint_data.mirroring_logic)
    faoc = CubicSpline(
        tau_c=0.01,
        n_l=5,
        sampling_freq=1000,
        n_joints=n_joints,
        joint_data=joint_data,
        online_settings=MPOnlineSettings(),
        abstract_set_dim=2,
    )
    assert faoc.initialize(ObjectiveFunction.magnitude) == EXIT_SUCCESSFUL

    pos_lims = faoc.get_position_limits()
    vel_lims = faoc.get_velocity_limits()
    acc_lims = faoc.get_acceleration_limits()
    assert np.allclose(pos_lims, np.column_stack((joint_data.pos_min, joint_data.pos_max)))

    sets = [faoc.get_max_controlled_invariant_set(j) for j in range(n_joints)]
    for j, (a_mat, b_vec) in enumerate(sets):
        h_mat, h_vec = faoc.get_max_controlled_invariant_set_scaled(j)
        assert h_mat.shape == (len(h_vec), 3)
        assert a_mat.shape == (2 * len(h_vec), 3)
        assert b_vec.shape == (2 * len(h_vec),)

    rng = np.random.default_rng(0)
    n_samples = 2000
    n_inside = np.zeros(n_joints, dtype=int)
    for _ in range(n_samples):
        state = np.column_stack(
            (
                rng.uniform(pos_lims[:, 0], pos_lims[:, 1]),
                rng.uniform(-vel_lims, vel_lims),
                rng.uniform(-acc_lims, acc_lims),
            )
        )
        code, inside = faoc.check_state_in_invariant_set(state)
        assert code == EXIT_SUCCESSFUL
        for j, (a_mat, b_vec) in enumerate(sets):
            assert bool(np.all(a_mat @ state[j] <= b_vec + 1e-9)) == bool(inside[j])
        n_inside += inside
    assert np.all(n_inside > 0) and np.all(n_inside < n_samples)

    a_mat, b_vec = sets[0]
    assert np.all(a_mat @ np.array([0.9, 0.0, 0.0]) <= b_vec)
    assert not np.all(a_mat @ np.array([-0.9, 0.0, 0.0]) <= b_vec)

    if plots_enabled():
        fig = plt.figure(figsize=(6 * n_joints, 5))
        for j, (a_mat, b_vec) in enumerate(sets):
            ax = fig.add_subplot(1, n_joints, j + 1, projection="3d")
            limits = np.array([pos_lims[j], [-vel_lims[j], vel_lims[j]], [-acc_lims[j], acc_lims[j]]])
            plot_invariant_set(a_mat, b_vec, ax, state_limits=limits, title=f"Joint {j + 1} MCIS")
        plt.show()


def test_inverse_map_offset_invariance():
    """Shifting the position limits and the whole problem by a constant must leave the abstract action
    unchanged and shift the joint action by exactly that constant."""

    def solve_shifted(shift: float):
        joint_data = JointData(
            joint_mirroring=[0, 0],
            joint_pos_min=[-1.0 + shift, -1.0 + shift],
            joint_pos_max=[1.0 + shift, 1.0 + shift],
            joint_vel_max=[2.0, 2.0],
            joint_acc_max=[8.0, 8.0],
            joint_jerk_max=[80.0, 80.0],
        )
        faoc = CubicSpline(
            tau_c=0.02,
            n_l=5,
            sampling_freq=1000,
            n_joints=2,
            joint_data=joint_data,
            online_settings=MPOnlineSettings(),
            abstract_set_dim=2,
        )
        assert faoc.initialize(ObjectiveFunction.magnitude) == EXIT_SUCCESSFUL

        x_0 = np.zeros((2, 3))
        x_0[:, 0] = shift
        assert faoc.set_initial_state(x_0.copy()) == EXIT_SUCCESSFUL

        wanted = np.array([[0.004 + shift, 0.05], [0.004 + shift, 0.05]])
        code, z_action = faoc.map_to_abstract_set(wanted.copy())
        assert code == EXIT_SUCCESSFUL
        assert faoc.set_abstract_action_and_solve(z_action) == EXIT_SUCCESSFUL
        reached = faoc.get_last_joint_action()

        # A target inside the feasible set must come back out of the round trip unchanged
        assert np.allclose(reached, wanted, atol=1e-9)
        return z_action, reached

    z_centred, reached_centred = solve_shifted(0.0)
    z_offset, reached_offset = solve_shifted(0.5)

    assert np.allclose(z_centred, z_offset, atol=1e-9)
    assert np.allclose(reached_offset[:, 0] - reached_centred[:, 0], 0.5, atol=1e-9)
    assert np.allclose(reached_offset[:, 1], reached_centred[:, 1], atol=1e-9)


class TestFAOCBindings(unittest.TestCase):
    """Test suite for FAOC Python bindings"""

    def test_abstract_multistep_method(self):
        """Wrapper for test_abstract_multistep"""
        print("Test 1", flush=True)
        test_abstract_multistep()

    def test_tolerance_method(self):
        """Wrapper for test_tolerance"""
        print("Test 2", flush=True)
        test_tolerance()

    def test_tolerance_with_mirroring_and_solving_method(self):
        """Wrapper for test_tolerance_with_mirroring_and_solving"""
        print("Test 3", flush=True)
        test_tolerance_with_mirroring_and_solving()

    def test_reset_faoc_method(self):
        """Wrapper for test_reset_faoc"""
        print("Test 4", flush=True)
        test_reset_faoc()

    def test_faoc_approx_1d_method(self):
        """Wrapper for test_faoc_approx_1d"""
        print("Test 5", flush=True)
        test_faoc_approx_1d()

    def test_invariant_set_unscaled_method(self):
        """Wrapper for test_invariant_set_unscaled"""
        print("Test 6", flush=True)
        test_invariant_set_unscaled()

    def test_inverse_map_offset_invariance_method(self):
        """Wrapper for test_inverse_map_offset_invariance"""
        print("Test 7", flush=True)
        test_inverse_map_offset_invariance()


if __name__ == "__main__":
    np.random.seed(0)
    np.set_printoptions(precision=5, suppress=True)
    unittest.main()
