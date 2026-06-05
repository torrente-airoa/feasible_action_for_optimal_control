#include "helper.h"
#include "multiStepAPI.h"

int main(void) {
  // motion planner parameters
  // Note: The following parameters must not be modified since they need to be the same as in the data-generating Matlab
  // file
  const int dim = 3;            // order of spline (only working with dim = 3, i.e. cubic spline, at the moment)
  const obj_t obj_type = MAGN;  // choose objective type in MPC (see h-file for definition)
  const int N_l = 4;            // number of intervals of quadratic spline
  const double tau_c = 10e-3;   // length of single spline interval [s]
  const scale_t scale_type = TIME_BASED;  // Note: Only TIME_BASED allowed for test function
  const double f_s = 1e3;                 // zero-order-hold sampling frequency of quadratic spline [Hz]
  const bool PARALLEL_INIT = false;       // true: parallel init with n_threads_init, false: no parallelization
  const int n_threads_init = 4;           // number of threads in initialization
  const double z_min[2] = {-1, -1};       // lower bounds on abstract rectangular action set
  const double z_max[2] = {1, 1};         // upper bounds on abstract rectangular action set
  const map_t map_type = POS_VEL;         // mapping type (POS (1D) | POS_VEL (2D))

  // directory for unit testdata
  const char utd_dir[] = "./unit_test_data/ctrlinvset/";

  // algorithm parameters
  const maxctrlinvset_params mcis_pars = {
    .zero_tol = 1e-14,
    .primal_tol = 1e-11,
    .shift_tol = 1e-8,
    .h_rel_tol = 1e-8,
    .Hh_abs_tol = 1e-12,
    .max_iter = 100,
    .n_constr_max = 1000,
    .re_method = "opt"};  // parameters for maximum control invariant set computation

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

  const algo_params *pos_set_alg_pars =
    &beta_alg_pars;  // enough for test reasons, since MCIS computations are not affected by these parameters

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

  // compute maximum control invariant sets for different speed/accel prefactors
  const double speed_accel_prefact[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  mpdata mpd[n_joints];

  double init_time = 0;
  int all_ok = 1;
  for (int iota = 0; iota < 10; iota++) {
    printf("\nComputing maximum control invariant sets with speed/accel prefactor of %.1f.\n",
           speed_accel_prefact[iota]);

    // adapt kinematic joint limits
    for (int i = 0; i < n_joints; i++) {
      mpd[i].joint_lims.qup = joint_lims[i].qup;
      mpd[i].joint_lims.qdotup = speed_accel_prefact[iota] * joint_lims[i].qdotup;
      mpd[i].joint_lims.qddotup = speed_accel_prefact[iota] * joint_lims[i].qddotup;
      mpd[i].joint_lims.qdddotup = joint_lims[i].qdddotup;

      int ret = initMapping(&mpd[i].mapd, dim, N_l, z_min, z_max, map_type, &init_time);
      if (ret < 0) {
        printf("Mapping initialization failed with return code %d. Exiting.\n", ret);
        freeData(mpd, n_joints);
        return -1;
      }
    }

    // compute maximum control invariant sets and init other data
    int ret = initData(mpd, obj_type, n_joints, tau_c, f_s, scale_type, PARALLEL_INIT, n_threads_init, &mcis_pars,
                       &mpc_alg_pars, &chebyshev_alg_pars, &direct_chebyshev_alg_pars, &beta_alg_pars, pos_set_alg_pars,
                       &init_time);
    if (ret < 0) {
      printf("Initialization failed with return code %d. Exiting.\n", ret);
      freeData(mpd, n_joints);
      return -1;
    } else
      printf("Initialization (including maximum control invariant set computation took %.3fs.\n", init_time);

    // compare with validated Matlab solution stored in binary files
    printf("Validation: ");
    for (int i = 0; i < n_joints; i++) {
      int ok = validateSet((const double **)mpd[i].H_inf, mpd[i].h_inf, mpd[i].n_inf, dim, utd_dir, i,
                           speed_accel_prefact[iota], &mcis_pars);
      all_ok &= ok;
      printf("Joint #%d: %s%s", i, (ok > 0) ? ("OK") : ("NOT OK"), (i < (n_joints - 1)) ? (", ") : ("\n"));
    }

    freeData(mpd, n_joints);
  }
  if (all_ok)
    printf("\nAll tests passed successfully.\n");
  else {
    printf("\nAt least one test case did not pass successfully.\n");
    if (!strcmp(mcis_pars.re_method, "convh")) {
      printf("NOTE: Validation uses Matlab results that are based on the \"opt\" redundancy elimination.\n");
      printf("      It turns out that for the redundancy elimination method \"convh\" the results are\n");
      printf("      slightly different and thus the validation says that not all tests have passed.\n");
      printf("      However, as verified, the results based on \"convh\" are indeed correct, but\n");
      printf("      in some cases come with a smaller number of (non-redundant) hyperplanes,\n");
      printf("      which is due to a higher accuracy of this redundancy elimination method.\n");
    }
  }

  return 0;
}
