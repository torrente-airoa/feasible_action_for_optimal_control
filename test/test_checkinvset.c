#include "helper.h"

int main(void) {
  // parameters
  const int dim = 3;                       // order of spline (only dim = 3 implemented yet)
  const obj_t obj_type = MAGN;             // objective type in MPC (see h-file for definition)
  const double tau_c = 8e-3;               // length of single spline interval [s]
  const scale_t scale_type = NORMALIZE;    // state/input scaling method
  const double f_s = 1e3;                  // zero-order-hold sampling frequency of cubic spline [Hz]
  const bool PARALLEL_INIT = true;         // true: parallel init with n_threads_init, false: no parallelization
  const int n_threads_init = 2;            // number of threads in initialization
  const double speed_accel_prefact = 0.7;  // scaling factor for speed and acceleration limits
  const int n_episodes = 107;              // number of episodes for timing (<= 107)
  const map_t map_type = POS_VEL;          // mapping type (POS (1D) | POS_VEL (2D))

  // directory for unit testdata
  const char utd_dir[] = "./unit_test_data/checkinvset/";

  // algorithm parameters (can be modified for tests)
  const maxctrlinvset_params mcis_pars = {.zero_tol = 1e-14,
                                          .primal_tol = 1e-11,
                                          .shift_tol = 1e-8,
                                          .h_rel_tol = 1e-8,
                                          .Hh_abs_tol = 1e-12,
                                          .max_iter = 100,
                                          .n_constr_max = 1000,
                                          .re_method = "convh"};

  const algo_params mpc_alg_pars = {
    .zero_tol = 1e-12,
    .primal_tol = 1e-11,
    .dual_tol = 1e-12};  // parameters for MPC QP (both for tau_c and mult*tau_c discretization)

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
  double z_min[2] = {-1, -1};
  double z_max[2] = {1, 1};
  double init_time = 0;
  for (int i = 0; i < n_joints; i++) {
    mpd[i].joint_lims.qup = joint_lims[i].qup;
    mpd[i].joint_lims.qdotup = speed_accel_prefact * joint_lims[i].qdotup;
    mpd[i].joint_lims.qddotup = speed_accel_prefact * joint_lims[i].qddotup;
    mpd[i].joint_lims.qdddotup = joint_lims[i].qdddotup;

    int ret = initMapping(&mpd[i].mapd, dim, 5, z_min, z_max, map_type, &init_time);
    if (ret < 0) {
      printf("Mapping initialization failed with return code %d. Exiting.\n", ret);
      return -1;
    }
  }
  int ret = initData(
    mpd, obj_type, n_joints, tau_c, f_s, scale_type, PARALLEL_INIT, n_threads_init, &mcis_pars, &mpc_alg_pars,
    &mpc_alg_pars, &mpc_alg_pars, &mpc_alg_pars, &mpc_alg_pars,
    &init_time);  // Note: We provide fake parameters for the optimization problems since they are not needed here
  if (ret < 0) {
    printf("Initialization failed with return code %d. Exiting.\n", ret);
    return -1;
  } else
    printf("Initialization including maximum control invariant set computation took %.3fs.\n", init_time);

  // start timing
  struct timespec start, end;
  double run_time = 0;
  double total_run_time = 0;
  double max_run_time = 0;
  int N[n_joints] = {0};
  int N_total = 0;
  int n_inside_MCIS = 0;
  int IN = 0;
  double *pos[n_joints] = {NULL};
  double *speed[n_joints] = {NULL};
  double *accel[n_joints] = {NULL};
  FILE *f_data[n_joints] = {NULL};
  double **X_0 = malloc(n_joints * sizeof(double *));
  for (int i = 0; i < n_joints; i++) X_0[i] = malloc(dim * sizeof(double));
  for (int j = 0; j < n_episodes; j++) {
    // open data files and read test data
    for (int i = 0; i < n_joints; i++) {
      getFileHandleCheckInvSet(utd_dir, i, j, &(f_data[i]));
      fread(N + i, sizeof(int), 1, f_data[i]);  // read trajectory length
      if (i > 0)
        if (N[i] != N[i - 1]) {
          printf("Trajectory lengths are not consistent.\n");
          return -1;
        }
      pos[i] = malloc(N[i] * sizeof(double));
      fread(pos[i], sizeof(double), N[i], f_data[i]);  // read position trajectory
      speed[i] = malloc(N[i] * sizeof(double));
      fread(speed[i], sizeof(double), N[i], f_data[i]);  // read speed trajectory
      accel[i] = malloc(N[i] * sizeof(double));
      fread(accel[i], sizeof(double), N[i], f_data[i]);  // read accel trajectory
    }

    // check MCIS membership for all joints for every timestep
    N_total += N[0];
    for (int k = 0; k < N[0]; k++) {
      // set initial states
      // Note: Sony's raw states are unscaled, so scale them first
      for (int i = 0; i < n_joints; i++) {
        X_0[i][0] = mpd[i].D[0] * pos[i][k];
        X_0[i][1] = mpd[i].D[1] * speed[i][k];
        X_0[i][2] = mpd[i].D[2] * accel[i][k];
      }

      // check for MCIS membership
      IN = 1;
      clock_gettime(CLOCK_REALTIME, &start);
      for (int i = 0; i < n_joints; i++) {
        // IN &= isInsideMaxCtrlInvSet(mpd + i, (const double **)&(X_0[i]), 1, 1.0); // slow, original function
        IN &= isInsideMaxCtrlInvSetFast(mpd[i].H_inf_rm, mpd[i].h_inf, (const double **)&(X_0[i]), 1, 1.0, mpd[i].n_inf,
                                        dim);  // fast, new function
      }
      clock_gettime(CLOCK_REALTIME, &end);
      if (IN)
        n_inside_MCIS++;  // Note: Since the initial states are based on ruckig, it's not guaranteed that all initial
                          // states are within their MCIS
      run_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
      if (run_time > max_run_time) max_run_time = run_time;
      total_run_time += run_time;
    }

    // close data files and free memory
    for (int i = 0; i < n_joints; i++) {
      fclose(f_data[i]);
      free(pos[i]);
      free(speed[i]);
      free(accel[i]);
    }
  }

  // print statistics
  printf("Avg/Max runtime of isInsideMaxCtrlInvSet() for all %d joints (%d time steps): %.3fus/%.3fus.\n", n_joints,
         N_total, (total_run_time / N_total) * 1e6, max_run_time * 1e6);
  printf("Time steps for which the initial states of all joints are inside their MCISs: %d/%d\n", n_inside_MCIS,
         N_total);
  printf("NOTE: Since the initial states are from ruckig, some of them are outside of the respective MCIS.\n");

  // free memory
  freeData(mpd, n_joints);
  for (int i = 0; i < n_joints; i++) free(X_0[i]);
  free(X_0);

  return 0;
}
