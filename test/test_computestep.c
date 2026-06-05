#include "helper.h"
#include "multiStepAPI.h"

int main(void) {
  // motion planner parameters
  // Note: The following parameters must not be modified since they need to be the same as in the data-generating Matlab
  // file
  const int dim = 3;                     // order of spline
  const obj_t obj_type = DIFF;           // objective type in MPC (see h-file for definition)
  const int N_l = 4;                     // number of intervals of quadratic spline
  const double tau_c = 10e-3;            // length of single spline interval [s]
  const scale_t scale_type = NORMALIZE;  // state/input scaling method
  const double f_s = 1e3;                // zero-order-hold sampling frequency of quadratic spline [Hz]
  const bool PARALLEL_INIT = true;       // true: parallel init with n_threads_init, false: no parallelization
  const int n_threads_init = 4;          // number of threads in initialization
  const double z_min[2] = {-1, -1};      // lower bounds on abstract rectangular action set
  const double z_max[2] = {1, 1};        // upper bounds on abstract rectangular action set
  const map_t map_type = POS_VEL;  // mapping type (Note: This test function is only valid for POS_VEL (2D mapping))

  // directory for unit testdata
  const char utd_dir[] = "./unit_test_data/computestep/";

  // algorithm parameters
  const maxctrlinvset_params mcis_pars = {
    .zero_tol = 1e-14,
    .primal_tol = 1e-11,
    .shift_tol = 1e-8,
    .h_rel_tol = 1e-8,
    .Hh_abs_tol = 1e-12,
    .max_iter = 100,
    .n_constr_max = 1000,
    .re_method = "convh"};  // parameters for maximum control invariant set computation

  const algo_params mpc_alg_pars = {
    .zero_tol = 1e-12, .primal_tol = 1e-11, .dual_tol = 1e-12};  // parameters for MPC QP

  const algo_params chebyshev_alg_pars = {.zero_tol = 1e-12,
                                          .primal_tol = 1e-11,
                                          .dual_tol = 1e-12,
                                          .eps_prox = 1e1,
                                          .eta_prox = 1e-10};  // parameters for Chebyshev LP

  const algo_params direct_chebyshev_alg_pars = {.zero_tol = 1e-12,
                                                 .primal_tol = 1e-11,
                                                 .dual_tol = 1e-12,
                                                 .eps_prox = 1e0,
                                                 .eta_prox = 1e-10};  // parameters for direct Chebyshev LP

  const algo_params beta_alg_pars = {.zero_tol = 1e-12,
                                     .primal_tol = 1e-11,
                                     .dual_tol = 1e-12,
                                     .eps_prox = 1e1,
                                     .eta_prox = 1e-10};  // parameters for LP to compute scale factor beta

  const algo_params pos_set_alg_pars = {.zero_tol = 1e-12,
                                        .primal_tol = 1e-11,
                                        .dual_tol = 1e-12,
                                        .eps_prox = 2e4,
                                        .eta_prox = 1e-10};  // parameters for LP to compute action set (only for POS)

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

  // other parameters from Matlab
  const double speed_accel_prefact = 1.0;
  double delta_pos_abs = 1e-4;    // allowed absolute deviation in terminal position [m] or [rad]
  double delta_speed_abs = 1e-2;  // allowed absolute deviation in terminal speed [m/s] or [rad/s]
  double x_0_scalings[] = {
    0,    0.1,  0.5,    0.8,      0.9,
    0.95, 0.99, 0.9999, 0.999999, 1};  // scalings of sampled initial states w.r.t. boundary of max ctr inv set
  const int n_x_0_scalings = 10;
  mpdata mpd[n_joints];

  // init data
  double init_time = 0;
  for (int i = 0; i < n_joints; i++) {
    // set joint limits and allowed deviations in position and speed at the terminal time instant
    mpd[i].joint_lims.qup = joint_lims[i].qup;
    mpd[i].joint_lims.qdotup = speed_accel_prefact * joint_lims[i].qdotup;
    mpd[i].joint_lims.qddotup = speed_accel_prefact * joint_lims[i].qddotup;
    mpd[i].joint_lims.qdddotup = joint_lims[i].qdddotup;
    mpd[i].term_state_devs.delta_qlow = -delta_pos_abs;
    mpd[i].term_state_devs.delta_qup = delta_pos_abs;
    mpd[i].term_state_devs.delta_qdotlow = -delta_speed_abs;
    mpd[i].term_state_devs.delta_qdotup = delta_speed_abs;

    int ret = initMapping(&mpd[i].mapd, dim, N_l, z_min, z_max, map_type, &init_time);
    if (ret < 0) {
      printf("Mapping initialization failed with return code %d. Exiting.\n", ret);
      return -1;
    }
  }

  // compute maximum control invariant sets and init rest of data
  int ret =
    initData(mpd, obj_type, n_joints, tau_c, f_s, scale_type, PARALLEL_INIT, n_threads_init, &mcis_pars, &mpc_alg_pars,
             &chebyshev_alg_pars, &direct_chebyshev_alg_pars, &beta_alg_pars, &pos_set_alg_pars, &init_time);
  if (ret < 0) {
    printf("Initialization failed with return code %d. Exiting.\n", ret);
    return -1;
  } else
    printf("Initialization including maximum control invariant set computation took %.3fs.\n\n", init_time);

  // run mapping tests
  struct timespec start, end;
  double run_time = 0;
  int n_x_0 = 0;
  int total_num_ok = 0;
  for (int i = 0; i < n_joints; i++) {
    mpd[i].udddh_m1 = mpd[i].d * 0;  // only relevant if obj_type == DIFF or obj_type == MIXED
    for (int j = 0; j < n_x_0_scalings; j++) {
      // open file for reading data
      FILE *f_data = NULL;
      getFileHandle(utd_dir, i, speed_accel_prefact, obj_type, x_0_scalings[j], &f_data);

      // loop over all possible pairs (initial state, abstract action)
      fread(&n_x_0, sizeof(int), 1, f_data);
      int num_ok = 0;
      int ok = 0;
      for (int iota = 0; iota < n_x_0; iota++) {
        // set initial state and abstract action
        fread(mpd[i].x_0, sizeof(double), dim, f_data);  // Note: initial state is already scaled
        fread(mpd[i].mapd->z_N_l, sizeof(double), 2, f_data);

        // compute step
        clock_gettime(CLOCK_REALTIME, &start);
        computeStep((void *)(mpd + i));
        clock_gettime(CLOCK_REALTIME, &end);
        run_time += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        // validate result
        if (mpd[i].planner_ret == TRAJ_OK) {
          ok = validateStep(mpd + i, f_data);
          if (ok) num_ok++;
        } else
          fseek(f_data, (N_l + 2 + N_l + 1) * sizeof(double), SEEK_CUR);  // skip validation data
      }
      total_num_ok += num_ok;
      fclose(f_data);

      printf("Joint #%d: initial states with scaling %.6f: %d/%d%s", i, x_0_scalings[j], num_ok, n_x_0,
             (j < (n_x_0_scalings - 1)) ? ("\n") : ("\n\n"));
    }
  }

  printf("%d/%d test cases solved successfully.\n", total_num_ok, n_joints * n_x_0_scalings * n_x_0);
  printf("Total/avg runtime of computeStep() (single core): %.3fs/%.3fus.\n", run_time,
         (run_time / (n_joints * n_x_0_scalings * n_x_0)) * 1e6);

  // free memory
  freeData(mpd, n_joints);

  return 0;
}
