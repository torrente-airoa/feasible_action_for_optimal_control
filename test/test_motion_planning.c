#include "helper.h"
#include "multiStepAPI.h"

int main(void) {
  // motion planner parameters
  const int dim = 3;            // order of spline (only dim = 3 implemented yet)
  const obj_t obj_type = DIFF;  // objective type in MPC (DIFF | MAGN) (see h-file for definition)
  const int N_l = 4;            // number of intervals of cubic spline
  const double tau_c = 8e-3;    // length of single spline interval [s]
  const scale_t scale_type =
    NORMALIZE;                       // state/input scaling method (TIME_BASED | NORMALIZE) (tested & tuned: NORMALIZE)
  const double f_s = 1e3;            // zero-order-hold sampling frequency of cubic spline [Hz]
  const bool PARALLEL_INIT = true;   // true: parallel init with n_threads_init, false: no parallelization
  const int n_threads_init = 4;      // number of threads in initialization
  const bool PARALLEL_MULTI = true;  // true: parallel multistep planner with n_threads_multi, false: no parallelization
  const int n_threads_multi = 4;     // number of threads in multistep planner
  const double z_min[] = {-1, -1};   // lower bounds on abstract rectangular action set
  const double z_max[] = {1, 1};     // upper bounds on abstract rectangular action set
  const map_t map_type = POS_VEL;    // mapping type (POS (1D) | POS_VEL (2D))

  // test parameters
  const int n_episodes = 1e4;  // number of episodes in test loop
  const int n_steps = 25;      // number of consecutive steps in multi-step MPC (per episode)

  // algorithm parameters
  const maxctrlinvset_params mcis_pars = {
    .zero_tol = 1e-14,
    .primal_tol = 1e-11,
    .shift_tol = 1e-8,
    .h_rel_tol = 1e-8,
    .Hh_abs_tol = 1e-12,
    .max_iter = 200,
    .n_constr_max = 2000,
    .re_method = "convh"};  // parameters for maximum control invariant set computation

  const algo_params mpc_alg_pars = {
    .zero_tol = 1e-12, .primal_tol = 1e-11, .dual_tol = 1e-12};  // parameters for MPC QP

  const algo_params chebyshev_alg_pars = {
    .zero_tol = 1e-12,
    .primal_tol = 1e-11,
    .dual_tol = 1e-12,
    .eps_prox = 1e1,
    .eta_prox = 1e-10};  // parameters for Chebyshev LP and unique norm QP (only for POS_VEL)

  const algo_params direct_chebyshev_alg_pars = {
    .zero_tol = 1e-12,
    .primal_tol = 1e-11,
    .dual_tol = 1e-12,
    .eps_prox = 1e0,
    .eta_prox = 1e-10};  // parameters for direct Chebyshev LP (only for POS_VEL)

  const algo_params beta_alg_pars = {
    .zero_tol = 1e-12,
    .primal_tol = 1e-11,
    .dual_tol = 1e-12,
    .eps_prox = 1e1,
    .eta_prox = 1e-10};  // parameters for LP to compute scale factor beta (only for POS_VEL)

  const algo_params pos_set_alg_pars = {.zero_tol = 1e-12,
                                        .primal_tol = 1e-11,
                                        .dual_tol = 1e-12,
                                        .eps_prox = 2e4,
                                        .eta_prox = 1e-10};  // parameters for LP to compute action set (only for POS)

  // define speed and acceleration prefactors
  const double speed_prefact = 1;
  const double accel_prefact = 1;

  // define allowed deviations from terminal pos/speed target in SI units
  double delta_pos_abs = 1e-4;    // allowed absolute deviation in terminal position [m] or [rad]
  double delta_speed_abs = 1e-2;  // allowed absolute deviation in terminal speed [m/s] or [rad/s]

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

  // init data
  mpdata mpd[n_joints];
  threadpool thpool = NULL;
  double init_time = 0;
  for (int i = 0; i < n_joints; i++) {
    int ret = initMapping(&mpd[i].mapd, dim, N_l, z_min, z_max, map_type, &init_time);
    if (ret < 0) {
      printf("Initialization of mapdata struct for joint %d failed with return code %d. Exiting.\n", i, ret);
      exit(-1);  // not nice, since we don't clean up all memory, but ok for our purpose, since this may only happen at
                 // initialization time
    }

    // set joint limits and allowed deviations in position and speed at the terminal time instant
    mpd[i].joint_lims.qup = joint_lims[i].qup;
    mpd[i].joint_lims.qdotup = speed_prefact * joint_lims[i].qdotup;
    mpd[i].joint_lims.qddotup = accel_prefact * joint_lims[i].qddotup;
    mpd[i].joint_lims.qdddotup = joint_lims[i].qdddotup;
    mpd[i].term_state_devs.delta_qlow = -delta_pos_abs;
    mpd[i].term_state_devs.delta_qup = delta_pos_abs;
    mpd[i].term_state_devs.delta_qdotlow = -delta_speed_abs;
    mpd[i].term_state_devs.delta_qdotup = delta_speed_abs;
  }
  printf("Initialization of mapdata structs for all %d joints took %.3fs.\n", n_joints, init_time);
  int ret =
    initData(mpd, obj_type, n_joints, tau_c, f_s, scale_type, PARALLEL_INIT, n_threads_init, &mcis_pars, &mpc_alg_pars,
             &chebyshev_alg_pars, &direct_chebyshev_alg_pars, &beta_alg_pars, &pos_set_alg_pars, &init_time);
  if (ret < 0) {
    printf("Initialization failed with return code %d. Exiting.\n", ret);
    exit(-1);  // not nice, but freeData() would free memory otherwise that is not initialized yet
  } else
    printf("Initialization including maximum control invariant set computation took %.3fs.\n", init_time);

  // run tests
  if (PARALLEL_MULTI) thpool = thpool_init(n_threads_multi);
  srand(0);  // to get different initializations of the RNG use time(NULL) as argument
  struct timespec start, end;
  double run_time = 0;

  int n_breaks = 0;
  for (int j = 0; j < n_episodes; j++) {
    // (I) Set initial state for every joint
    for (int i = 0; i < n_joints; i++) {
      while (1)  // compute a random initial state x_0 until it is feasible
      {
        getRandomInitialState(mpd + i);
        // Note: The MPC planner below assumes that the scaled initial state is inside the maximum
        //       control invariant set, so verify this and sample again, if it is outside.
        if (isInsideMaxCtrlInvSetFast(mpd[i].H_inf_rm, mpd[i].h_inf, (const double **)&(mpd[i].x_0), 1, 1.0,
                                      mpd[i].n_inf, dim))
          break;
      }
    }

    // (II) For n_steps in time, run the MPC planner or every joint starting from the initial state
    for (int i = 0; i < n_joints; i++)
      mpd[i].udddh_m1 = mpd[i].d * 0;  // only relevant if obj_type == DIFF or obj_type == MIXED
    for (int m = 0; m < n_steps; m++) {
      // get an abstract action
      for (int i = 0; i < n_joints; i++) getRandomAbstractAction(mpd + i);  // draw sample from action set

      // compute step
      clock_gettime(CLOCK_REALTIME, &start);
      for (int i = 0; i < n_joints; i++) {
        if (PARALLEL_MULTI)
          thpool_add_work(thpool, computeStep, (void *)(mpd + i));
        else
          computeStep((void *)(mpd + i));
      }
      if (PARALLEL_MULTI) thpool_wait(thpool);
      clock_gettime(CLOCK_REALTIME, &end);
      run_time += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

      // check if computations were done right
      int BREAK = 0;
      for (int i = 0; i < n_joints; i++) {
        if (mpd[i].planner_ret != TRAJ_OK) {
          printf("episode %d, step %d, mpd[%d].planner_ret = %d\n", j, m, i, mpd[i].planner_ret);
          n_breaks++;
          BREAK = 1;
          break;  // start over with new initial state, i.e. dismiss this episode
        }
      }
      if (BREAK) break;

      // prepare for next simulation step
      for (int i = 0; i < n_joints; i++) {
        for (int iota = 0; iota < dim; iota++) mpd[i].x_0[iota] = mpd[i].x_N_l[iota];
        mpd[i].udddh_m1 = mpd[i].udddh[N_l - 1];  // we set it even if obj_type != DIFF
      }
    }
  }
  char thr_info[20];
  sprintf(thr_info, "%d thread%s", (PARALLEL_MULTI) ? (n_threads_multi) : (1),
          (PARALLEL_MULTI && (n_threads_multi > 1)) ? ("s") : (""));
  printf("Total/avg runtime of computeStep() for all %d joints (%s): %.3fs/%.3fus.\n", n_joints, thr_info, run_time,
         (run_time / (n_episodes * n_steps)) * 1e6);
  printf("Number of successfully completed episodes each with %d steps: %d/%d\n", n_steps, n_episodes - n_breaks,
         n_episodes);

  if ((PARALLEL_MULTI) && (thpool != NULL)) thpool_destroy(thpool);
  freeData(mpd, n_joints);

  return 0;
}
