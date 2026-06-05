#include "helper.h"
#include "math_utils.h"
#include "resetAPI.h"

int main(void) {
  // reset planner parameters
  const int dim = 3;                     // order of spline (only dim = 3 implemented yet)
  const obj_t obj_type = MAGN;           // objective type in MPC (see h-file for definition)
  const double tau_c = 8e-3;             // length of single spline interval [s]
  const scale_t scale_type = NORMALIZE;  // state/input scaling method
  const double f_s = 1e3;                // ZOH sampling frequency of position trajectory
  const bool PARALLEL_INIT = true;       // true: parallel init with n_threads_init, false: no parallelization
  const int n_threads_init = 4;          // number of threads in initialization
  const bool PARALLEL_RESET = true;      // true: parallel reset planner with n_threads_reset, false: no parallelization
  const int n_threads_reset = 4;         // number of threads in reset planner
  const int joint_order_reset[] = {0, 1, 2, 3,
                                   4, 5, 6, 7};  // order of joints assigned to threads if PARALLEL_RESET is true
  const int mult = 2;  // multiple of tau_c used for roughly and quickly determining minimum horizon length
  const double speed_accel_prefact = 0.7;  // scaling factor for speed and acceleration limits
  const int n_scenarios =
    1e3;                    // number of different scenarios (one scenario means one reset trajectory for all joints)
  const int max_N_l = 300;  // maximum horizon length for reset planner (valid for all joints, valid for tau_c case)
  const rp_mode reset_mode = SYNC;  // mode for reset planner (DIST | SYNC)
  const map_t map_type = POS_VEL;   // mapping type (POS (1D) | POS_VEL (2D))

  // algorithm parameters
  const maxctrlinvset_params mcis_pars = {
    .zero_tol = 1e-14,
    .primal_tol = 1e-11,
    .shift_tol = 1e-8,
    .h_rel_tol = 1e-8,
    .Hh_abs_tol = 1e-12,
    .max_iter = 200,
    .n_constr_max = 2000,
    .re_method = "convh"};  // parameters for maximum control invariant set computations

  const algo_params mpc_alg_pars = {
    .zero_tol = 1e-12,
    .primal_tol = 1e-11,
    .dual_tol = 1e-12};  // parameters for MPC QP (both for tau_c and mult*tau_c discretization)

  const bisect_params bs_params_mult = {
    .dN_max = 2, .add_steps_init = 4, .add_steps = 3};  // for mult*tau_c discretization only

  // define joint limits in SI units
  const int n_joints = 8;
  joint_limits joint_lims[n_joints];

  // axis X
  joint_lims[0].qup = 0.95;
  joint_lims[0].qdotup = 4.0;
  joint_lims[0].qddotup = 10.8;
  joint_lims[0].qdddotup = 600.0;

  // axis Y
  joint_lims[1].qup = 0.7;
  joint_lims[1].qdotup = 4.0;
  joint_lims[1].qddotup = 15.7;
  joint_lims[1].qdddotup = 600.0;

  // axis J1
  joint_lims[2].qup = M_PI;
  joint_lims[2].qdotup = 7.85398;
  joint_lims[2].qddotup = 39.27;
  joint_lims[2].qdddotup = 750.0;

  // axis J2
  joint_lims[3].qup = M_PI / 2;
  joint_lims[3].qdotup = 7.85398;
  joint_lims[3].qddotup = 39.27;
  joint_lims[3].qdddotup = 750.0;

  // axis J3
  joint_lims[4].qup = M_PI;
  joint_lims[4].qdotup = 12.5664;
  joint_lims[4].qddotup = 125.66;
  joint_lims[4].qdddotup = 1500.0;

  // axis J4
  joint_lims[5].qup = 2.44346;
  joint_lims[5].qdotup = 12.5664;
  joint_lims[5].qddotup = 125.66;
  joint_lims[5].qdddotup = 1500.0;

  // axis J5
  joint_lims[6].qup = 2.40855;
  joint_lims[6].qdotup = 7.8539816;
  joint_lims[6].qddotup = 78.539816;
  joint_lims[6].qdddotup = 1000.0;

  // axis J6
  joint_lims[7].qup = 2 * M_PI;
  joint_lims[7].qdotup = 31.41593;
  joint_lims[7].qddotup = 628.32;
  joint_lims[7].qdddotup = 6000.0;

  // define maximum control horizons valid for all states in respective MCIS
  // Note: If any of the maximum horizon lengths are unknown for the respective setting, then simply
  //       define the corresponding entry as zero. In this case, initResetData() will compute the
  //       missing entry.
  int N_l_max[] = {0, 0, 0, 0, 0, 0, 0, 0};       // for tau_c discretization
  int N_l_max_mult[] = {0, 0, 0, 0, 0, 0, 0, 0};  // for mult*tau_c discretization

  // define reset state ranges
  // Note: - only position (first entry) might be defined as an interval with lower/upper bounds,
  //         speed and acceleration reset values must be zero (currently)
  //       - every reset state must be in the MCIS
  const double reset_states_lo[n_joints][dim] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
                                                 {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};  // unscaled
  const double reset_states_up[n_joints][dim] = {{0.5, 0, 0},      {0.35, 0, 0},     {M_PI / 2, 0, 0},
                                                 {M_PI / 4, 0, 0}, {M_PI / 2, 0, 0}, {1, 0, 0},
                                                 {1, 0, 0},        {M_PI, 0, 0}};  // unscaled
  const double *states_lo_ptr = &reset_states_lo[0][0];
  const double *states_up_ptr = &reset_states_up[0][0];

  double z_min[2] = {-1, -1};
  double z_max[2] = {1, 1};

  // init motion planner and reset planner data
  mpdata mpd[n_joints];
  rpdata rpd[n_joints];
  double init_time = 0;
  double total_init_time = 0;
  for (int i = 0; i < n_joints; i++) {
    mpd[i].joint_lims.qup = joint_lims[i].qup;
    mpd[i].joint_lims.qdotup = speed_accel_prefact * joint_lims[i].qdotup;
    mpd[i].joint_lims.qddotup = speed_accel_prefact * joint_lims[i].qddotup;
    mpd[i].joint_lims.qdddotup = joint_lims[i].qdddotup;
    rpd[i].joint_lims.qup = mpd[i].joint_lims.qup;
    rpd[i].joint_lims.qdotup = mpd[i].joint_lims.qdotup;
    rpd[i].joint_lims.qddotup = mpd[i].joint_lims.qddotup;
    rpd[i].joint_lims.qdddotup = mpd[i].joint_lims.qdddotup;

    int ret = initMapping(&mpd[i].mapd, dim, 5, z_min, z_max, map_type, &init_time);
    if (ret < 0) {
      printf("Mapping initialization failed with return code %d. Exiting.\n", ret);
      return -1;
    }
  }

  // compute MCIS and init rest of data
  int ret = initData(
    mpd, obj_type, n_joints, tau_c, f_s, scale_type, PARALLEL_INIT, n_threads_init, &mcis_pars, &mpc_alg_pars,
    &mpc_alg_pars, &mpc_alg_pars, &mpc_alg_pars, &mpc_alg_pars,
    &init_time);  // Note: We provide fake parameters for the optimization problems since they are not needed here
  total_init_time = init_time;
  ret = min(ret, initResetData(rpd, dim, obj_type, n_joints, tau_c, mult, max_N_l, N_l_max, N_l_max_mult, f_s,
                               scale_type, PARALLEL_INIT, n_threads_init, states_lo_ptr, states_up_ptr, &mcis_pars,
                               &mpc_alg_pars, &bs_params_mult, reset_mode, &init_time));
  total_init_time += init_time;
  if (ret < 0) {
    printf("Initialization failed with return code %d. Exiting.\n", ret);
    return -1;
  } else
    printf("Initialization including maximum control invariant set computations took %.3fs.\n\n", total_init_time);

  printf("Results of horizon computations:\n");
  printArrInt("N_l_max", n_joints, 1, N_l_max);
  printArrInt("N_l_max_mult", n_joints, 1, N_l_max_mult);

  // set the _scaled_ terminal reset state x_f
  // Note: - As in the multi-step planner, the user is responsible for appropriate scaling.
  //       - If the reset state x_f is a parameter, it must be set online.
  srand(0);  // to get different initializations of the RNG use time(NULL) as argument
  const double theta = ((double)rand() / RAND_MAX);  // theta \in [0, 1]
  for (int i = 0; i < n_joints; i++)
    for (int j = 0; j < dim; j++)
      rpd[i].x_f[j] =
        rpd[i].D[j] * (reset_states_lo[i][j] +
                       theta * (reset_states_up[i][j] - reset_states_lo[i][j]));  // choose a random reset position

  // run tests
  struct timespec start, end;
  double run_time = 0;
  double total_run_time = 0;
  double max_run_time = 0;
  int n_failed_scenarios = 0;
  int ok = 0;
  threadpool thpool = NULL;
  if (PARALLEL_RESET) thpool = thpool_init(n_threads_reset);
  for (int j = 0; j < n_scenarios; j++) {
    // (I) Set initial state for every joint
    for (int i = 0; i < n_joints; i++) {
      // compute a random (scaled) initial state x_0 w.r.t. the tau_c initialization
      while (1) {
        getRandomInitialState(mpd + i);
        // Note: The reset planner below assumes that the scaled initial state is inside the maximum
        //       control invariant set, so verify this and sample again, if it is outside.
        if (isInsideMaxCtrlInvSetFast(mpd[i].H_inf_rm, mpd[i].h_inf, (const double **)&(mpd[i].x_0), 1, 1.0,
                                      mpd[i].n_inf, dim))
          break;
      }
      // provide reset planner with the initial state
      for (int iota = 0; iota < dim; iota++) rpd[i].x_0[iota] = mpd[i].x_0[iota];
      // set previous scaled jerk (only relevant if obj_type == DIFF or obj_type == MIXED)
      rpd[i].udddh_m1 = rpd[i].d * 0;
    }

    // (II) compute reset trajectories for all joints
    clock_gettime(CLOCK_REALTIME, &start);
    for (int i = 0; i < n_joints; i++) {
      if (PARALLEL_RESET)
        thpool_add_work(thpool, computeResetTraj, (void *)(rpd + joint_order_reset[i]));
      else
        computeResetTraj((void *)(rpd + i));
    }
    if (PARALLEL_RESET) thpool_wait(thpool);
    if (reset_mode == SYNC) syncTrajs(rpd, n_joints, thpool);
    clock_gettime(CLOCK_REALTIME, &end);
    run_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    if (run_time > max_run_time) max_run_time = run_time;
    total_run_time += run_time;

    // (III) check if computations were done right
    for (int i = 0; i < n_joints; i++) {
      if (rpd[i].planner_ret == TRAJ_OK) {
        ok = validateTraj(rpd + i);
        if (!ok) {
          printf("scenario %d: validation failed for joint %d.\n", j, i);
          n_failed_scenarios++;
          break;
        }
      } else {
        printf("scenario %d: rpd[%d].planner_ret = %d\n", j, i, rpd[i].planner_ret);
        n_failed_scenarios++;
        break;
      }
    }
  }

  // print statistics
  char thr_info[20];
  sprintf(thr_info, "%d thread%s", (PARALLEL_RESET) ? (n_threads_reset) : (1),
          (PARALLEL_RESET && (n_threads_reset > 1)) ? ("s") : (""));
  printf("Avg/Max runtime of computeResetTraj() for all %d joints (%s, %s mode): %.3fms/%.3fms.\n", n_joints, thr_info,
         (reset_mode == SYNC) ? ("SYNC") : ("DIST"), (total_run_time / n_scenarios) * 1e3, max_run_time * 1e3);
  printf("Number of successfully completed scenarios (each with %d reset trajectories): %d/%d\n", n_joints,
         n_scenarios - n_failed_scenarios, n_scenarios);

  // free memory
  if ((PARALLEL_RESET) && (thpool != NULL)) thpool_destroy(thpool);
  freeData(mpd, n_joints);
  freeResetData(rpd, n_joints);

  return 0;
}
