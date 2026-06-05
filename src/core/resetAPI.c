#include "resetAPI.h"

#include "helper.h"
#include "math_utils.h"

int initResetData(rpdata *rpd, const int dim, const obj_t obj_type, const int n_joints, const double tau_c,
                  const int mult, const int max_N_l, int *N_l_max, int *N_l_max_mult, const double f_s,
                  const scale_t scale_type, const bool PARALLEL, const int n_threads, const double *reset_states_lo,
                  const double *reset_states_up, const maxctrlinvset_params *mcis_pars_mult,
                  const algo_params *mpc_alg_pars, const bisect_params *bs_params_mult, const rp_mode reset_mode,
                  double *init_time) {
  if (dim != 3)  // only cubic splines allowed so far
  {
    *init_time = 0;
    return -2;
  }

  if (scale_type == TIME_BASED)  // no time-based scaling implemented for reset planner so far
  {
    *init_time = 0;
    return -2;
  }

  struct timespec start, end;
  clock_gettime(CLOCK_REALTIME, &start);

  // compute N_l_max values (tau_c case), if applicable, and
  // compute N_l_max_mult values (mult*tau_c case), if applicable
  // Note: The implementation could be made more efficient in terms of making use of
  //       existing MCIS, however, this would make the code unnecessarily complex.
  //       For this reason we sacrifice speed for code clarity.
  if (N_l_max) {
    int ret = getMaxHorizons(rpd, dim, n_joints, tau_c, max_N_l, f_s, scale_type, PARALLEL, n_threads, reset_states_lo,
                             reset_states_up, mcis_pars_mult, mpc_alg_pars, N_l_max);
    if (ret < 0) {
      *init_time = 0;  // if initialization fails, this value is not of interest
      return -2;
    }
  } else {
    *init_time = 0;
    return -2;
  }

  if (N_l_max_mult) {
    int ret = getMaxHorizons(rpd, dim, n_joints, mult * tau_c, max_N_l / mult, f_s, scale_type, PARALLEL, n_threads,
                             reset_states_lo, reset_states_up, mcis_pars_mult, mpc_alg_pars,
                             N_l_max_mult);  // Note: Integer division is intended
    if (ret < 0) {
      *init_time = 0;  // if initialization fails, this value is not of interest
      return -2;
    }
  } else {
    *init_time = 0;
    return -2;
  }

  // take into account the reset planner mode
  // Note: This is easiest, if we overwrite the values of N_l_max and N_l_max_mult by the respective max
  int N_l_max_orig[n_joints];       // used to restore the original N_l_max values
  int N_l_max_mult_orig[n_joints];  // used to restore the original N_l_max_mult values
  if (reset_mode == SYNC) {
    int max_N_l_max = 0;
    int max_N_l_max_mult = 0;
    for (int i = 0; i < n_joints; i++) {
      N_l_max_orig[i] = N_l_max[i];
      N_l_max_mult_orig[i] = N_l_max_mult[i];
      max_N_l_max = (N_l_max[i] > max_N_l_max) ? (N_l_max[i]) : (max_N_l_max);
      max_N_l_max_mult = (N_l_max_mult[i] > max_N_l_max_mult) ? (N_l_max_mult[i]) : (max_N_l_max_mult);
    }
    for (int i = 0; i < n_joints; i++) {
      N_l_max[i] = max_N_l_max;
      N_l_max_mult[i] = max_N_l_max_mult;
    }
  }

  for (int i = 0; i < n_joints; i++) {
    // (i) init data fields and/or allocate memory
    rpd[i].dim = dim;
    rpd[i].obj_type = obj_type;
    rpd[i].N_l_max_mult = N_l_max_mult[i] + bs_params_mult->add_steps;
    // N_l_max must fulfill the following requirements for the implemented methods to work correctly:
    // - it is an integer multiple of mult
    // - it is at least mult * N_l_max_mult
    rpd[i].N_l_max = (int)max((double)(mult * ceil(N_l_max[i] / (double)mult) + mult * bs_params_mult->add_steps),
                              (double)(mult * rpd[i].N_l_max_mult));
    rpd[i].tau_c = tau_c;
    rpd[i].mult = (mult < 1) ? (1) : (mult);
    rpd[i].tau_s = 1.0 / f_s;
    rpd[i].scale_type = scale_type;
    rpd[i].mcis_pars_mult.primal_tol = mcis_pars_mult->primal_tol;
    rpd[i].mcis_pars_mult.zero_tol = mcis_pars_mult->zero_tol;
    rpd[i].mcis_pars_mult.shift_tol = mcis_pars_mult->shift_tol;
    rpd[i].mcis_pars_mult.h_rel_tol = mcis_pars_mult->h_rel_tol;
    rpd[i].mcis_pars_mult.Hh_abs_tol = mcis_pars_mult->Hh_abs_tol;
    rpd[i].mcis_pars_mult.max_iter = mcis_pars_mult->max_iter;
    rpd[i].mcis_pars_mult.n_constr_max = mcis_pars_mult->n_constr_max;
    strcpy(rpd[i].mcis_pars_mult.re_method, mcis_pars_mult->re_method);
    rpd[i].bs_params_mult.dN_max = (bs_params_mult->dN_max < 1) ? (1) : (bs_params_mult->dN_max);
    rpd[i].bs_params_mult.add_steps_init =
      (bs_params_mult->add_steps_init < 0) ? (0) : (bs_params_mult->add_steps_init);
    rpd[i].bs_params_mult.add_steps = (bs_params_mult->add_steps < 0) ? (0) : (bs_params_mult->add_steps);
    // D, d initialized when computing the MCIS
    rpd[i].N = -1;  // N gets set once length of control horizon N_l has been determined
    rpd[i].N_l = -1;
    // H_inf_mult, H_inf_mult_rm, h_inf_mult and n_inf_mult initialized when computing the MCIS
    rpd[i].x_f = malloc(dim * sizeof(double));
    rpd[i].x_0 = malloc(dim * sizeof(double));
    rpd[i].rowcoef = malloc(rpd[i].N_l_max * dim * sizeof(double));
    rpd[i].rowcoef_mult = malloc(rpd[i].N_l_max_mult * dim * sizeof(double));
    // A_mult, B_mult, x_max_mult and u_max_mult are allocated/defined when computing the MCIS
    // A, B, x_max and u_max are allocated/defined separately below
    rpd[i].uh = malloc((rpd[i].N_l_max + 1) * sizeof(double));
    rpd[i].udh = malloc((rpd[i].N_l_max + 1) * sizeof(double));
    rpd[i].uddh = malloc((rpd[i].N_l_max + 1) * sizeof(double));
    rpd[i].udddh = malloc((rpd[i].N_l_max) * sizeof(double));
    int N_max = floor((rpd[i].N_l_max * rpd[i].tau_c) * f_s);
    rpd[i].u_zoh = malloc((N_max + 1) * sizeof(double));
    rpd[i].du_zoh = malloc((N_max + 1) * sizeof(double));
    rpd[i].ddu_zoh = malloc((N_max + 1) * sizeof(double));
    rpd[i].dddu_zoh = malloc((N_max + 1) * sizeof(double));
    rpd[i].lambda_star_mult = malloc((rpd[i].N_l_max_mult + rpd[i].N_l_max_mult * dim) * sizeof(double));

    rpd[i].planner_ret = INIT_PLANNER_RET;
  }

  threadpool thpool;
  mpdata mpd[n_joints];  // helper array to make use of the existing functions
  if (PARALLEL) thpool = thpool_init(n_threads);
  for (int i = 0; i < n_joints; i++) {
    mpd[i].mapd = malloc(sizeof(map_data));
    rpd2mpd(&(rpd[i]), &(mpd[i]));  // Note: Also sets rpd[i].tau_c = rpd[i].mult*rpd[i].tau_c as required

    // (ii) compute MCIS
    if (PARALLEL)
      thpool_add_work(thpool, computeMaxCtrlInvSetSymmetric, (void *)(mpd + i));
    else {
      computeMaxCtrlInvSetSymmetric((void *)(mpd + i));
      mpd2rpd(&(mpd[i]), &(rpd[i]));
    }
  }
  if (PARALLEL) {
    thpool_wait(thpool);
    for (int i = 0; i < n_joints; i++) mpd2rpd(&(mpd[i]), &(rpd[i]));
    thpool_destroy(thpool);
  }
  for (int i = 0; i < n_joints; i++) {
    if (rpd[i].mcis_ret != ALL_OK) {
      clock_gettime(CLOCK_REALTIME, &end);
      *init_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
      return rpd[i].mcis_ret;
    }
  }

  // allocate/define A, B, x_max and u_max (tau_c case)
  double D[3];
  double d;
  double Delta = 0;      // shrinkage value for position in unscaled (original) units
  double Delta_dot = 0;  // shrinkage value for speed in unscaled (original) units
  for (int i = 0; i < n_joints; i++) {
    scaleDyn(rpd[i].scale_type, &(rpd[i].joint_lims), rpd[i].tau_c, rpd[i].A, rpd[i].B, D,
             &d);  // Note: D and d are already set
    getShrinkageDeltas(&(rpd[i].joint_lims), rpd[i].tau_c, &Delta_dot, &Delta);
    rpd[i].x_max = malloc(rpd[i].dim * sizeof(double));
    rpd[i].x_max[0] = rpd[i].D[0] * (rpd[i].joint_lims.qup - Delta);
    rpd[i].x_max[1] = rpd[i].D[1] * (rpd[i].joint_lims.qdotup - Delta_dot);
    rpd[i].x_max[2] = rpd[i].D[2] * rpd[i].joint_lims.qddotup;
    rpd[i].u_max = rpd[i].d * rpd[i].joint_lims.qdddotup;
  }

  // initialize optimization problems
  int ret = 0;
  for (int i = 0; i < n_joints; i++) {
    memset(&(rpd[i].mpc), 0, sizeof(DAQPProblem));
    memset(&(rpd[i].work), 0, sizeof(DAQPWorkspace));
    ret = min(ret, initMPCProb(&(rpd[i].mpc), &(rpd[i].work), mpc_alg_pars, rpd[i].A, rpd[i].B, rpd[i].N_l_max,
                               rpd[i].dim, rpd[i].rowcoef));
    memset(&(rpd[i].mpc_mult), 0, sizeof(DAQPProblem));
    memset(&(rpd[i].work_mult), 0, sizeof(DAQPWorkspace));
    ret = min(ret, initMPCProb(&(rpd[i].mpc_mult), &(rpd[i].work_mult), mpc_alg_pars, rpd[i].A_mult, rpd[i].B_mult,
                               rpd[i].N_l_max_mult, rpd[i].dim, rpd[i].rowcoef_mult));
  }

  // compute return value as minimum of all return values so far
  int min_ret = ret;
  for (int i = 0; i < n_joints; i++) min_ret = min(min_ret, rpd[i].mcis_ret);

  // stop timer
  clock_gettime(CLOCK_REALTIME, &end);
  *init_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  // restore N_l_max and N_l_max_mult values, if applicable
  if (reset_mode == SYNC)
    for (int i = 0; i < n_joints; i++) {
      N_l_max[i] = N_l_max_orig[i];
      N_l_max_mult[i] = N_l_max_mult_orig[i];
    }

  return min_ret;
}

void computeResetTraj(void *arg) {
  rpdata *rpd = (rpdata *)arg;
  const int dim = rpd->dim;
  const double shrink_fact = 1 - 1e-8;  // see note in header text of shrinkState()

  // (i) For mult*tau_c discretization: Determine a near time-optimal horizon length using bisection
  //     Note: x_0 could be infeasible w.r.t. the mult*tau_c discretization, so, if this is the case,
  //           i.e. x_0 is not in the MCIS for mult*tau_c, then we shrink it until it is inside this set.
  double x_0_mult[dim];
  if (!isInsideMaxCtrlInvSetFast(rpd->H_inf_mult_rm, rpd->h_inf_mult, (const double **)&(rpd->x_0), 1, 1.0,
                                 rpd->n_inf_mult, rpd->dim))
    shrinkState(rpd->x_0, (const double **)rpd->H_inf_mult, rpd->h_inf_mult, rpd->n_inf_mult, dim, shrink_fact,
                x_0_mult);
  else
    for (int iota = 0; iota < dim; iota++) x_0_mult[iota] = rpd->x_0[iota];
  // at this point, x_0_mult is a feasible initial state w.r.t. the mult*tau_c discretization -- and not much different
  // than the original initial state x_0 -- i.e. it can be brought to the reset state in finite time (but the
  // horizon length is still unknown)

  // finally, do the bisection to find a near time-optimal horizon length for the mult*tau_c discretization
  int N_l_lo_mult = ceil(minTContTime(x_0_mult, rpd->x_f[0], rpd->x_max_mult, rpd->D) /
                         (rpd->mult * rpd->tau_c));  // lower bound on control horizon
  int N_l_up_mult = rpd->N_l_max_mult;               // upper bound on control horizon
  int N_l_init_mult = (int)min((double)(N_l_lo_mult + rpd->bs_params_mult.add_steps_init),
                               (double)N_l_up_mult);  // initial control horizon
  if (N_l_lo_mult > N_l_up_mult) {
    rpd->planner_ret = NO_FEAS_HORIZON;
    return;
  }
  // Note: We use MAGN as objective type for the bisection _always_ since it is faster than DIFF and
  //       for determining the horizon length, the objective type is not relevant
  //       (as long as it is strongly convex). However, the Lagrange multiplier depends on the objective
  //       type and hence, we expect worse performance through warmstarting.
  int N_l_mult = mpcBisect(&(rpd->mpc_mult), &(rpd->work_mult), rpd->dim, rpd->A_mult, rpd->rowcoef_mult,
                           rpd->x_max_mult, rpd->u_max_mult, N_l_lo_mult, N_l_init_mult, N_l_up_mult, x_0_mult,
                           rpd->x_f, rpd->udddh_m1, MAGN, &(rpd->bs_params_mult), rpd->lambda_star_mult);

  // error checks
  if (N_l_mult <= 0) {
    // something went wrong in the bisection method (always related to DAQP)
    rpd->planner_ret = BISECT_ERR;
    return;
  }
  if (N_l_mult > N_l_up_mult) {
    // no feasible horizon length exists
    rpd->planner_ret = NO_FEAS_HORIZON;
    return;
  }

  // (ii) solve the original MPC problem with tau_c discretization using mult*N_mult as the horizon length
  //      and use warm-starting information in Lagrange multiplier rpd->lambda_star_mult
  int N_l = rpd->mult * N_l_mult;  // always feasible w.r.t. allocated memory
  int ret =
    mpcSolve(&(rpd->mpc), &(rpd->work), rpd->dim, rpd->A, rpd->rowcoef, rpd->x_max, rpd->u_max, &N_l, rpd->N_l_max,
             rpd->x_0, rpd->x_f, rpd->udddh_m1, rpd->obj_type, rpd->mult, rpd->lambda_star_mult, true, rpd->udddh);
  if (ret < 0) {
    // problem in MPC
    rpd->planner_ret = MPC_ERR;
    return;
  }
  rpd->N_l = N_l;
  rpd->N = floor((N_l * rpd->tau_c) / rpd->tau_s);

  // (iii) simulate state trajectory (needed for ZOH sampling)
  rpd->uh[0] = rpd->x_0[0];
  rpd->udh[0] = rpd->x_0[1];
  rpd->uddh[0] = rpd->x_0[2];
  double x[dim];
  double x_plus[dim];
  x[0] = rpd->uh[0];
  x[1] = rpd->udh[0];
  x[2] = rpd->uddh[0];
  for (int l = 0; l < N_l; l++) {
    matVecMul2(dim, dim, rpd->A, x, x_plus);
    for (int iota = 0; iota < dim; iota++) x[iota] = x_plus[iota] + rpd->B[iota] * rpd->udddh[l];
    rpd->uh[l + 1] = x[0];
    rpd->udh[l + 1] = x[1];
    rpd->uddh[l + 1] = x[2];
  }

  // (iv) ZOH sampling
  // Note: In order to use the existing ZOH sampling function, we use an mpdata struct:
  mpdata mpd;
  mpd.mapd = malloc(sizeof(map_data));
  for (int iota = 0; iota < dim; iota++) mpd.D[iota] = rpd->D[iota];
  mpd.d = rpd->d;
  mpd.N = rpd->N;
  mpd.mapd->N_l = rpd->N_l;
  mpd.tau_s = rpd->tau_s;
  mpd.tau_c = rpd->tau_c;
  mpd.u_zoh = rpd->u_zoh;
  mpd.ud_zoh = rpd->du_zoh;
  mpd.udd_zoh = rpd->ddu_zoh;
  mpd.uddd_zoh = rpd->dddu_zoh;
  mpd.uh = rpd->uh;
  mpd.udh = rpd->udh;
  mpd.uddh = rpd->uddh;
  mpd.udddh = rpd->udddh;

  zohSampleSpline(&mpd);
  free(mpd.mapd);
  rpd->planner_ret = TRAJ_OK;
}

void syncTrajs(rpdata *rpd, const int n_joints, threadpool thpool) {
  // sync trajectories, i.e. make them all the same length (based on the slowest one)

  // make sure that all trajectories are valid and determine the largest control horizon
  int max_N_l_max = 0;
  for (int i = 0; i < n_joints; i++) {
    if (rpd[i].planner_ret != TRAJ_OK) return;
    max_N_l_max = (rpd[i].N_l > max_N_l_max) ? (rpd[i].N_l) : (max_N_l_max);
  }

  // re-solve all reset trajectories so that they are of equal length
  // (except for the ones which are of max length already)
  for (int i = 0; i < n_joints; i++) {
    if (rpd[i].N_l < max_N_l_max) {
      rpd[i].N_l = max_N_l_max;
      if (thpool)
        thpool_add_work(thpool, computeSyncResetTraj, (void *)(rpd + i));
      else
        computeSyncResetTraj((void *)(rpd + i));
    }
  }
  if (thpool) thpool_wait(thpool);
}

int freeResetData(rpdata *rpd, const int n_joints) {
  for (int i = 0; i < n_joints; i++) {
    // free MPC QP (tau_c discretization)
    free(rpd[i].mpc.H);
    free(rpd[i].mpc.f);
    free(rpd[i].mpc.blower);
    free(rpd[i].mpc.bupper);
    free(rpd[i].mpc.A);
    free(rpd[i].mpc.sense);
    free_daqp_workspace(&(rpd[i].work));
    free_daqp_ldp(&(rpd[i].work));

    // free MPC QP (mult*tau_c discretization)
    free(rpd[i].mpc_mult.H);
    free(rpd[i].mpc_mult.f);
    free(rpd[i].mpc_mult.blower);
    free(rpd[i].mpc_mult.bupper);
    free(rpd[i].mpc_mult.A);
    free(rpd[i].mpc_mult.sense);
    free_daqp_workspace(&(rpd[i].work_mult));
    free_daqp_ldp(&(rpd[i].work_mult));

    if (rpd[i].H_inf_mult) {
      for (int j = 0; j < rpd[i].n_inf_mult; j++)
        if (rpd[i].H_inf_mult[j]) free(rpd[i].H_inf_mult[j]);
      free(rpd[i].H_inf_mult);
    }
    if (rpd[i].H_inf_mult_rm) free(rpd[i].H_inf_mult_rm);
    if (rpd[i].h_inf_mult) free(rpd[i].h_inf_mult);
    free(rpd[i].x_f);
    free(rpd[i].x_0);
    free(rpd[i].rowcoef);
    free(rpd[i].rowcoef_mult);
    free(rpd[i].x_max);
    free(rpd[i].x_max_mult);
    free(rpd[i].uh);
    free(rpd[i].udh);
    free(rpd[i].uddh);
    free(rpd[i].udddh);
    free(rpd[i].u_zoh);
    free(rpd[i].du_zoh);
    free(rpd[i].ddu_zoh);
    free(rpd[i].dddu_zoh);
    free(rpd[i].lambda_star_mult);
  }
  return 0;
}
