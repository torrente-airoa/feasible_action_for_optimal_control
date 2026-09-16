#include "multiStepAPI.h"

#include "daqp/utils.h"
#include "helper.h"
#include "math_utils.h"

int initData(mpdata *mpd, const obj_t obj_type, const int n_joints, const double tau_c, const double f_s,
             const scale_t scale_type, const bool PARALLEL, const int n_threads, const maxctrlinvset_params *mcis_pars,
             const algo_params *mpc_alg_pars, const algo_params *chebyshev_alg_pars,
             const algo_params *direct_chebyshev_alg_pars, const algo_params *beta_alg_pars,
             const algo_params *pos_set_alg_pars, double *init_time) {
  struct timespec start, end;
  clock_gettime(CLOCK_REALTIME, &start);
  for (int i = 0; i < n_joints; i++) {
    // (i) init data fields and/or allocate memory
    mpd[i].mcis_pars.primal_tol = mcis_pars->primal_tol;
    mpd[i].mcis_pars.zero_tol = mcis_pars->zero_tol;
    mpd[i].mcis_pars.shift_tol = mcis_pars->shift_tol;
    mpd[i].mcis_pars.h_rel_tol = mcis_pars->h_rel_tol;
    mpd[i].mcis_pars.Hh_abs_tol = mcis_pars->Hh_abs_tol;
    mpd[i].mcis_pars.max_iter = mcis_pars->max_iter;
    mpd[i].mcis_pars.n_constr_max = mcis_pars->n_constr_max;
    strcpy(mpd[i].mcis_pars.re_method, mcis_pars->re_method);
    mpd[i].obj_type = obj_type;
    mpd[i].scale_type = scale_type;
    // D, d initialized when computing control invariant set
    mpd[i].tau_c = tau_c;
    mpd[i].tau_s = 1.0 / f_s;
    mpd[i].N = floor((mpd[i].mapd->N_l * mpd[i].tau_c) * f_s);
    // H_inf, H_inf_rm, h_inf and n_inf initialized when computing control invariant set
    mpd[i].x_0 = malloc(mpd[i].mapd->dim * sizeof(double));
    // A, B, V_X, x_max, H_0, h_0 and u_max allocated/defined when computing control invariant set
    mpd[i].x_N_l = malloc(mpd[i].mapd->dim * sizeof(double));
    mpd[i].uh = malloc((mpd[i].mapd->N_l + 1) * sizeof(double));
    mpd[i].udh = malloc((mpd[i].mapd->N_l + 1) * sizeof(double));
    mpd[i].uddh = malloc((mpd[i].mapd->N_l + 1) * sizeof(double));
    mpd[i].udddh = malloc((mpd[i].mapd->N_l) * sizeof(double));
    mpd[i].u_zoh = malloc((mpd[i].N + 1) * sizeof(double));
    mpd[i].ud_zoh = malloc((mpd[i].N + 1) * sizeof(double));
    mpd[i].udd_zoh = malloc((mpd[i].N + 1) * sizeof(double));
    mpd[i].uddd_zoh = malloc((mpd[i].N + 1) * sizeof(double));
    mpd[i].E_full = malloc(mpd[i].mapd->dim * sizeof(double *));
    for (int j = 0; j < mpd[i].mapd->dim; j++) mpd[i].E_full[j] = malloc(mpd[i].mapd->N_l * sizeof(double));
    if (mpd[i].mapd->map_type == POS_VEL) {
      mpd[i].E = malloc(POS_VEL * sizeof(double *));
      for (int j = 0; j < POS_VEL; j++) mpd[i].E[j] = malloc(mpd[i].mapd->N_l * sizeof(double));
    }
    mpd[i].planner_ret = INIT_PLANNER_RET;
  }

  threadpool thpool;
  if (PARALLEL) thpool = thpool_init(n_threads);
  for (int i = 0; i < n_joints; i++) {
    // (ii) compute maximum control invariant set
    if (PARALLEL)
      thpool_add_work(thpool, computeMaxCtrlInvSetSymmetric, (void *)(mpd + i));
    else
      computeMaxCtrlInvSetSymmetric((void *)(mpd + i));
  }
  if (PARALLEL) {
    thpool_wait(thpool);
    thpool_destroy(thpool);
  }
  for (int i = 0; i < n_joints; i++) {
    if (mpd[i].mcis_ret != ALL_OK) {
      clock_gettime(CLOCK_REALTIME, &end);
      *init_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
      return mpd[i].mcis_ret;
    }
  }

  // set up optimization problems
  int ret = 0;
  for (int i = 0; i < n_joints; i++) {
    if (mpd[i].mapd->map_type == POS_VEL) {
      ret = min(ret, setupOptProbsPosSpeed(mpd + i, mpc_alg_pars, chebyshev_alg_pars, direct_chebyshev_alg_pars,
                                           beta_alg_pars));
    } else {
      ret = min(ret, setupOptProbsPos(mpd + i, mpc_alg_pars, pos_set_alg_pars));
    }
  }

  // compute return value as minimum of all return values so far
  int min_ret = ret;
  for (int i = 0; i < n_joints; i++) min_ret = min(min_ret, mpd[i].mcis_ret);

  // stop timer
  clock_gettime(CLOCK_REALTIME, &end);
  *init_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  return min_ret;
}

void computeMaxCtrlInvSetSymmetric(void *arg) {
  // Compute maximum control invariant set assuming symmetry of the state and input set (cf. notes from Aug 12 2024)
  // and prototype Matlab function computeMaxCtrlInvSetSymmetric()

  mpdata *mpd = (mpdata *)arg;
  const int dim = mpd->mapd->dim;

  // (A) independent of the user-defined scaling, always scale states and inputs according to
  //     TIME_BASED scaling for the computation of the maximum control invariant set, since
  //     it turns out to lead to the best numerical stability. But need to re-scale the resulting
  //     set after computation to adhere to the scaling in scale_type
  scaleDynNSets(mpd, TIME_BASED);

  // (B) allocate memory for helper variables
  double **H_k = malloc(
    mpd->mcis_pars.n_constr_max *
    sizeof(double *));  // coefficient matrix of symmetric H-representation of set X_k in max ctrl inv set computation
  for (int j = 0; j < mpd->mcis_pars.n_constr_max; j++) H_k[j] = malloc(dim * sizeof(double));
  double *h_k =
    malloc(mpd->mcis_pars.n_constr_max *
           sizeof(double));  // lhs/rhs vector of symmetric H-representation of set X_k in max ctrl inv set computation
  double *g_k = malloc(mpd->mcis_pars.n_constr_max * sizeof(double));  // auxiliary vector
  int *I = malloc(mpd->mcis_pars.n_constr_max * sizeof(int));          // auxiliary index vector
  int *J = malloc(mpd->mcis_pars.n_constr_max * sizeof(int));          // auxiliary index vector
  double **H_kp1 =
    malloc(mpd->mcis_pars.n_constr_max *
           sizeof(double *));  // coefficient matrix of H-representation of set X_kp1 in max ctrl inv set computation
  double *h_kp1 =
    malloc(mpd->mcis_pars.n_constr_max *
           sizeof(double));  // rhs vector of H-representation of set X_kp1 in max ctrl inv set computation
  int *R =
    malloc(mpd->mcis_pars.n_constr_max * sizeof(int));  // holds indices of redundant constraints in (H_kp1, h_kp1)
  for (int j = 0; j < mpd->mcis_pars.n_constr_max; j++) H_kp1[j] = malloc(dim * sizeof(double));
  double b_u_diff[dim];
  double D_TIME_BASED[dim];

  // conditional memory allocation, based on method for redundancy elimination
  double *H_kp1_rm = NULL;
  double *H_final = NULL;
  int *auxR = NULL;
  const int aux_num_vert = mpd->mcis_pars.n_constr_max;
  if (!strcmp(mpd->mcis_pars.re_method, "convh")) {
    // needed for dual-based redundancy elimination
    H_kp1_rm = malloc(mpd->mcis_pars.n_constr_max * dim *
                      sizeof(double));  // coefficient vector of H-representation of set X_kp1 in row-major format
    H_final = malloc(2 * aux_num_vert * dim *
                     sizeof(double));  // coefficient vector of relevant entries in final round (2* due to symmetry)
    auxR = malloc(aux_num_vert * sizeof(int));  // auxiliary vector
  }

  // (C) initialize variables
  const int m_0 = dim;  // number of hyperplances in symmetric H-representation of feasible state set X
  double f_bar = 0;
  for (int j = 0; j < dim; j++) {
    b_u_diff[j] = -2 * mpd->B[j] * mpd->u_max;
    f_bar += (1.0 / 2) * pow(mpd->x_max[j], 2);
  }
  for (int i = 0; i < m_0; i++) {
    h_k[i] = mpd->h_0[i];                                      // init h_k
    for (int j = 0; j < dim; j++) H_k[i][j] = mpd->H_0[i][j];  // init H_k
  }
  int m_k = m_0;
  int m_kp1 = 0;
  int k = 0;
  int n_R = 0;
  int iota = 0;
  int n_iters = 0;
  int EQUIV = 0;
  int j_R = 0;
  double l1_norm = 0;
  mpd->H_inf = NULL;
  mpd->H_inf_rm = NULL;
  mpd->h_inf = NULL;
  mpd->n_inf = 0;
  double *swap = NULL;
  double *vert_rm = NULL;
  int n_vert = 0;
  mcis_ret_code ret = ALL_OK;
  mcis_ret_code ret_mix = ALL_OK;

  // (D) compute maximum control invariant set recursively
  while (1) {
    if (k == mpd->mcis_pars.max_iter) {
      printf("Increase max_iter in the MCIS computation parameter struct.\n");
      ret = ITER_LIMIT;
      goto cleanup;
    }

    // (i) compute all vertices of the set from the previous iteration, which is a superset
    //     or identical (when convergence is achieved) to the next set. These vertices will be used
    //     to quickly identify constraints, which are surely redundant. This will not only save time
    //     in the redundancy check later on, but also drastically reduce memory consumption for matrix H_kp1
    if (!vert_rm) {
      int ret_vert = computeVertices((const double **)H_k, h_k, m_k, dim, &vert_rm, &n_vert);
      if (ret_vert < 0) {
        printf("Problem with vertex computation.\n");
        ret = VERTEX_COMP;
        goto cleanup;
      }
    }

    // (ii) define set X_kp1 = intersect(inv(A)*(X_k - B*U), X)  as { x | -h_kp1 <= H_kp1*x <= h_kp1 }
    //      and in order to do so, we start with partitioning the coefficients of the g_k variable into
    //      negative and positive ones and fill in rows/components of intermediate matrix H_kp1 and vector h_kp1
    matVecMul(m_k, dim, (const double **)H_k, b_u_diff, g_k);  // g_k <- H_k*b_u_diff
    int n_I = 0, n_J = 0;
    m_kp1 = 0;
    for (int i = 0; i < m_k; i++) {
      // compute a constraint candidate for (H_kp1, h_kp1)
      vecMatMul(dim, dim, H_k[i], mpd->A, H_kp1[m_kp1]);  // H_kp1(m_kp1,:) <- H_k(i,:)*A
      h_kp1[m_kp1] = h_k[i];
      if (g_k[i] > mpd->mcis_pars.zero_tol) {
        h_kp1[m_kp1] += g_k[i] / 2;
        I[n_I++] = i;
      } else if (g_k[i] < -mpd->mcis_pars.zero_tol) {
        h_kp1[m_kp1] -= g_k[i] / 2;
        J[n_J++] = i;
      }

      // do a quick check if this constraint is redundant (only sufficient condition)
      if (isRedundant(H_kp1[m_kp1], h_kp1[m_kp1], vert_rm, n_vert, dim, mpd->mcis_pars.shift_tol))
        continue;  // don't increase m_kp1 so that current constraint candidate gets overwritten in next iteration
      else
        m_kp1++;
    }

    // next, add mixed constraints
    // Note: Before adding these constraints to (H_kp1, h_kp1), we do very basic but powerful redundancy checks
    // (ii.a) mix within I set
    ret_mix = mixConstraintsSameSet((const double **)H_k, h_k, dim, g_k, I, n_I, mpd->A, mpd->mcis_pars.n_constr_max,
                                    &(mpd->mcis_pars), H_kp1, h_kp1, &m_kp1, vert_rm, n_vert);
    if (ret_mix < 0) {
      ret = ret_mix;
      goto cleanup;
    }

    // (ii.b) mix within J set
    ret_mix = mixConstraintsSameSet((const double **)H_k, h_k, dim, g_k, J, n_J, mpd->A, mpd->mcis_pars.n_constr_max,
                                    &(mpd->mcis_pars), H_kp1, h_kp1, &m_kp1, vert_rm, n_vert);
    if (ret_mix < 0) {
      ret = ret_mix;
      goto cleanup;
    }

    // (ii.c) mix I and J sets
    ret_mix = mixConstraints((const double **)H_k, h_k, dim, g_k, I, n_I, J, n_J, mpd->A, mpd->mcis_pars.n_constr_max,
                             &(mpd->mcis_pars), H_kp1, h_kp1, &m_kp1, vert_rm, n_vert);
    if (ret_mix < 0) {
      ret = ret_mix;
      goto cleanup;
    }

    // and finally, add state constraints
    if (m_kp1 > (mpd->mcis_pars.n_constr_max - m_0)) {
      printf("Increase n_constr_max in the MCIS computation parameter struct.\n");
      ret = MEMORY_LIMIT;
      goto cleanup;
    }
    for (int i = 0; i < m_0; i++) {
      for (int j = 0; j < dim; j++) H_kp1[m_kp1][j] = mpd->H_0[i][j];
      h_kp1[m_kp1++] = mpd->h_0[i];
      // don't do a redundancy check for this type of constraints
    }

    // vertices of the previous set are not used anymore, so free memory
    free(vert_rm);
    vert_rm = NULL;

    // (iii) remove redundant hyperplances in (H_kp1, h_kp1)
    if (!strcmp(mpd->mcis_pars.re_method, "convh"))
      n_iters = redundancyEliminationConvexHull((const double **)H_kp1, h_kp1, m_kp1, dim, H_kp1_rm, H_final, auxR,
                                                aux_num_vert, R, &n_R, &vert_rm, &n_vert);
    else
      n_iters = redundancyElimination((const double **)H_kp1, h_kp1, m_kp1, dim, f_bar, &(mpd->mcis_pars), R, &n_R);
    if (n_iters < 0) {
      printf("Problem in redundancy elimination method.");
      ret = REDUNDANCY_ERR;
      goto cleanup;
    }

    // (iv) check termination criterion and prepare for next iteration
    k++;
    if (m_k == (m_kp1 - n_R))  // necessary condition for equivalence
    {
      // before we start the worst case quadratic search, we do a simple linear complexity check
      double h_k_one_norm = oneNorm(m_k, h_k);
      double h_kp1_one_norm = 0;
      j_R = 0;
      for (int j = 0; j < m_kp1; j++) {
        if ((j_R < n_R) && (j == R[j_R])) {
          j_R++;
        } else {
          h_kp1_one_norm += h_kp1[j];  // note that h_kp1 is a positive vector
        }
      }
      if (fabs(h_kp1_one_norm - h_k_one_norm) <
          (mpd->mcis_pars.h_rel_tol * h_k_one_norm))  // make this check to avoid many expensive checks below
      {
        for (int i = 0; i < m_k; i++) {
          EQUIV = 0;
          j_R = 0;
          for (int j = 0; j < m_kp1;
               j++)  // Note: This is not the most efficient way to do the comparison, however, it's more readable
          {
            if ((j_R < n_R) && (j == R[j_R])) {
              j_R++;
            } else {
              l1_norm = 0;
              for (iota = 0; iota < dim; iota++) l1_norm += fabs(H_kp1[j][iota] - H_k[i][iota]);
              l1_norm += fabs(h_kp1[j] - h_k[i]);
              if (l1_norm <= mpd->mcis_pars.Hh_abs_tol) {
                EQUIV = 1;
                break;
              }
            }
          }
          if (!EQUIV) break;
        }

        if (EQUIV) break;  // quits while loop for recursive set computation
      }
    }

    // (v) prepare for next iteration
    if ((m_kp1 - n_R) > mpd->mcis_pars.n_constr_max)  // make sure that there is enough memory for (H_k, h_k)
    {
      printf("Increase n_constr_max in the MCIS computation parameter struct.\n");
      ret = MEMORY_LIMIT;
      goto cleanup;
    }

    // (H_k, h_k) <- (H_kp1, h_kp1) minus redundant constraints
    m_k = 0;
    j_R = 0;
    for (int j = 0; j < m_kp1; j++) {
      if ((j_R < n_R) && (j == R[j_R])) {
        j_R++;
      } else {
        // swap rows
        swap = H_k[m_k];
        H_k[m_k] = H_kp1[j];
        H_kp1[j] = swap;
        h_k[m_k++] = h_kp1[j];
      }
    }
  }

  // (E) scale set (H_k, h_k) according to scale_type
  if (mpd->scale_type != TIME_BASED) {
    for (int i = 0; i < dim; i++) D_TIME_BASED[i] = mpd->D[i];

    // free H_0, h_0 and x_max before 2nd call to scaleDynNSets()
    for (int i = 0; i < dim; i++) free(mpd->H_0[i]);
    free(mpd->H_0);
    free(mpd->h_0);
    free(mpd->x_max);
    scaleDynNSets(mpd, mpd->scale_type);  // finally create data according to user-defined scaling
    for (int i = 0; i < m_k; i++)
      for (int j = 0; j < dim; j++)
        H_k[i][j] *= D_TIME_BASED[j] / mpd->D[j];  // re-scale according to user-defined scaling
  }

  // (F) save maximum control invariant set
  mpd->n_inf = m_k;
  mpd->H_inf = malloc(mpd->n_inf * sizeof(double *));
  mpd->H_inf_rm = malloc(mpd->n_inf * mpd->mapd->dim * sizeof(double));
  mpd->h_inf = malloc(mpd->n_inf * sizeof(double));
  for (int i = 0; i < mpd->n_inf; i++) {
    mpd->H_inf[i] = H_k[i];
    for (int iota = 0; iota < mpd->mapd->dim; iota++) mpd->H_inf_rm[i * mpd->mapd->dim + iota] = H_k[i][iota];
    mpd->h_inf[i] = h_k[i];
    H_k[i] = NULL;  // just as a safeguard
  }

  // // normalize rows for better numerics
  // // Note: Did not show any advantage.
  // double one_norm = 0;
  // for (int i = 0; i < mpd->n_inf; i++)
  // {
  //     one_norm = oneNorm(dim, mpd->H_inf[i]);
  //     for (int j = 0; j < dim; j++)
  //         mpd->H_inf[i][j] /= one_norm;
  //     mpd->h_inf[i] /= one_norm;
  // }

// (G) clean up and return
cleanup:
  for (int i = mpd->n_inf; i < mpd->mcis_pars.n_constr_max; i++) free(H_k[i]);
  free(H_k);
  free(h_k);
  free(g_k);
  free(I);
  free(J);
  for (int j = 0; j < mpd->mcis_pars.n_constr_max; j++) free(H_kp1[j]);
  free(H_kp1);
  free(h_kp1);
  free(R);
  if (!strcmp(mpd->mcis_pars.re_method, "convh")) {
    free(H_kp1_rm);
    free(H_final);
    free(auxR);
  }
  if (vert_rm) free(vert_rm);
  mpd->mcis_ret = ret;

  // printf("set iterations: %d, n_inf: %d, exit code: %d\n", k, mpd->n_inf, ret);
}

int setupOptProbsPosSpeed(mpdata *mpd, const algo_params *mpc_alg_pars, const algo_params *chebyshev_alg_pars,
                          const algo_params *direct_chebyshev_alg_pars, const algo_params *beta_alg_pars) {
  // (i) set up condensed MPC problem and compute matrix E and direction-mapping matrix G (if applicable)
  int ret = setupMPC(mpd, mpc_alg_pars);

  ret = min(ret, SetupOptProbs2D(mpd->mapd, mpd->n_inf, mpd->mpc.A, (const double **)mpd->E, chebyshev_alg_pars,
                                 direct_chebyshev_alg_pars, beta_alg_pars));

  return ret;
}

int setupOptProbsPos(mpdata *mpd, const algo_params *mpc_alg_pars, const algo_params *pos_set_alg_pars) {
  // (i) set up condensed MPC problem and compute matrix E and direction-mapping matrix G (if applicable)
  int ret = setupMPC(mpd, mpc_alg_pars);

  // (ii) set up action set computation LP (computation of position interval)
  ret = min(ret, SetupOptProbs1D(mpd->mapd, mpd->n_inf, mpd->mpc.A, (const double **)mpd->E_full, pos_set_alg_pars));

  return ret;
}

void mappingUpdateBounds(mpdata *mpd, map_runtime_bounds *bounds, c_float *lower, c_float *upper, double *e_full) {
  map_data *mapd = mpd->mapd;
  const int N_l = mapd->N_l;
  const int dim = mapd->dim;

  // Update mapping bounds for current x_0 and y_N_l.
  updateBounds(mpd, lower, lower + N_l, NULL, lower + N_l + dim * (N_l - 1), upper, upper + N_l, NULL,
               upper + N_l + dim * (N_l - 1), NULL, e_full);

  bounds->lower = lower;
  bounds->upper = upper;
  bounds->e_full = e_full;
}

void computeStep(void *arg) {
  mpdata *mpd = (mpdata *)arg;
  map_data *mapd = mpd->mapd;

  const int n_feas = mapd->N_l + mapd->dim * (mapd->N_l - 1) + mpd->n_inf;
  c_float lower[n_feas];
  c_float upper[n_feas];
  double e_full[mapd->dim];
  map_runtime_bounds bounds;
  mappingUpdateBounds(mpd, &bounds, lower, upper, e_full);

  int ret = (int)mapFromAbstractSetWithBounds(mpd->mapd, mpd->n_inf, &bounds);
  if (ret < 0) {
    mpd->planner_ret = MAP_ERR;
    return;
  }

  ret = mpcPlanner(mpd);
  if (ret < 0) {
    mpd->planner_ret = MPC_ERR;
    return;
  }

  zohSampleSpline(mpd);
  mpd->planner_ret = TRAJ_OK;
}

int mpcPlanner(mpdata *mpd) {
  // Computes a joint position trajectory that is zero-order-hold sampled with 1/tau_s.
  // Note: Caller is responsible for making sure that the pair of initial and terminal
  //       states/positions/speeds (x_0, y_N_l) is feasible. Under this condition, this
  //       function will always provide a position trajectory -- unless numerical issues
  //       pop up, which will be reported by a return value of <0.

  const int N_l = mpd->mapd->N_l;
  const int dim = mpd->mapd->dim;
  int update_mask;
  int exit_flag;
  double e_full[mpd->mapd->dim];

  // update lhs and rhs vectors of inequality constraints (dependent on x_0)
  updateBounds(mpd, mpd->mpc.blower, mpd->mpc.blower + N_l, mpd->mpc.blower + N_l + dim * (N_l - 1),
               mpd->mpc.blower + N_l + dim * (N_l - 1) + 2, mpd->mpc.bupper, mpd->mpc.bupper + N_l,
               mpd->mpc.bupper + N_l + dim * (N_l - 1), mpd->mpc.bupper + N_l + dim * (N_l - 1) + 2, NULL, e_full);

  // update gradient in objective function, if applicable
  if (mpd->obj_type == DIFF || mpd->obj_type == MIXED)
    mpd->mpc.f[0] = -mpd->udddh_m1;  // Note: We require mpd->udddh_m1 to be a _scaled_ jerk

  // if there are no deviations in terminal pos/speed, mark this explicitly in the sense
  if (mpd->term_state_devs.delta_qlow == mpd->term_state_devs.delta_qup)
    mpd->mpc.sense[N_l + dim * (N_l - 1)] = DAQP_ACTIVE + DAQP_IMMUTABLE;
  if ((mpd->term_state_devs.delta_qdotlow == mpd->term_state_devs.delta_qdotup) && (mpd->mapd->map_type == POS_VEL))
    mpd->mpc.sense[N_l + dim * (N_l - 1) + 1] = DAQP_ACTIVE + DAQP_IMMUTABLE;

  // solve problem
  update_mask = DAQP_UPDATE_v + DAQP_UPDATE_sense;
  daqp_update_ldp(update_mask, &(mpd->mpc_work), mpd->mpc_work.qp);
  exit_flag = daqp_ldp(&(mpd->mpc_work));

  if (exit_flag != DAQP_EXIT_OPTIMAL) {
    daqp_deactivate_constraints(&(mpd->mpc_work));
    reset_daqp_workspace(&(mpd->mpc_work));
    return -1;
  } else {
    ldp2qp_solution(&(mpd->mpc_work));                                 // convert solution
    for (int l = 0; l < N_l; l++) mpd->udddh[l] = mpd->mpc_work.x[l];  // save solution
  }

  // simulate state trajectory
  mpd->uh[0] = mpd->x_0[0];
  mpd->udh[0] = mpd->x_0[1];
  mpd->uddh[0] = mpd->x_0[2];
  double x[dim];
  double x_plus[dim];
  x[0] = mpd->uh[0];
  x[1] = mpd->udh[0];
  x[2] = mpd->uddh[0];
  for (int l = 0; l < N_l; l++) {
    matVecMul2(dim, dim, mpd->A, x, x_plus);
    for (int iota = 0; iota < dim; iota++) x[iota] = x_plus[iota] + mpd->B[iota] * mpd->udddh[l];
    mpd->uh[l + 1] = x[0];
    mpd->udh[l + 1] = x[1];
    mpd->uddh[l + 1] = x[2];
  }

  // compute and set full-dimensional terminal state
  // Note: It turns out that in order to enhance numerical stability and to compensate for
  //       minor numerical errors, we slightly shrink the terminal state so that it is
  //       surely within the maximum control invariant set
  matVecMul(dim, N_l, (const double **)mpd->E_full, mpd->udddh, mpd->x_N_l);  // use (E,e) for numerical reasons
  vecVecDiff(dim, (1 - 1e-8), mpd->x_N_l, -(1 - 1e-8), e_full, mpd->x_N_l);

  // check terminal state condition
  double err_uh_N_l = mpd->x_N_l[0] - mpd->mapd->y_N_l[0];
  if ((err_uh_N_l > mpd->term_state_devs.delta_qup * mpd->D[0] + 1e-6) ||
      (err_uh_N_l < mpd->term_state_devs.delta_qlow * mpd->D[0] - 1e-6))
    return -1;

  if (mpd->mapd->map_type == POS_VEL) {
    double err_udh_N_l = mpd->x_N_l[1] - mpd->mapd->y_N_l[1];
    if ((err_udh_N_l > mpd->term_state_devs.delta_qdotup * mpd->D[1] + 1e-6) ||
        (err_udh_N_l < mpd->term_state_devs.delta_qdotlow * mpd->D[1] - 1e-6))
      return -1;
  }

  // check feasibility of (reduced) position and speed limits and (original) acceleration and jerk limits
  for (int l = 0; l < N_l; l++) {
    if ((mpd->uh[l] < (1 + 1e-6) * (-mpd->x_max[0])) || (mpd->uh[l] > (1 + 1e-6) * mpd->x_max[0]) ||
        (mpd->udh[l] < (1 + 1e-6) * (-mpd->x_max[1])) || (mpd->udh[l] > (1 + 1e-6) * mpd->x_max[1]) ||
        (mpd->uddh[l] < (1 + 1e-6) * (-mpd->x_max[2])) || (mpd->uddh[l] > (1 + 1e-6) * mpd->x_max[2]) ||
        (mpd->udddh[l] < (1 + 1e-6) * (-mpd->u_max)) || (mpd->udddh[l] > (1 + 1e-6) * mpd->u_max))
      return -1;
  }

  return 0;
}

int isInsideMaxCtrlInvSet(const mpdata *mpd, const double **X, const int n_X, const double shrink_factor) {
  for (int i = 0; i < n_X; i++) {
    for (int j = 0; j < mpd->n_inf; j++) {
      if (fabs(scalarProd(mpd->mapd->dim, mpd->H_inf[j], X[i])) > shrink_factor * mpd->h_inf[j]) return 0;
    }
  }

  return 1;
}

int isInsideMaxCtrlInvSetFast(const double *H_inf_rm, const double *h_inf, const double **X, const int n_X,
                              const double shrink_factor, const int n_inf, const int dim) {
  for (int i = 0; i < n_X; i++)
    for (int j = 0; j < n_inf; j++)
      if (fabs(scalarProd(dim, H_inf_rm + j * dim, X[i])) > shrink_factor * h_inf[j]) return 0;

  return 1;
}

int freeData(mpdata *mpd, const int n_joints) {
  for (int i = 0; i < n_joints; i++) {
    // free MPC QP
    free(mpd[i].mpc.H);
    free(mpd[i].mpc.f);
    free(mpd[i].mpc.blower);
    free(mpd[i].mpc.bupper);
    free(mpd[i].mpc.A);
    free(mpd[i].mpc.sense);
    free_daqp_workspace(&(mpd[i].mpc_work));
    free_daqp_ldp(&(mpd[i].mpc_work));

    if (mpd[i].H_inf) {
      for (int j = 0; j < mpd[i].n_inf; j++)
        if (mpd[i].H_inf[j]) free(mpd[i].H_inf[j]);
      free(mpd[i].H_inf);
    }
    if (mpd[i].H_inf_rm) free(mpd[i].H_inf_rm);
    if (mpd[i].h_inf) free(mpd[i].h_inf);
    free(mpd[i].x_0);
    free(mpd[i].x_max);
    for (int j = 0; j < mpd[i].mapd->dim; j++) free(mpd[i].H_0[j]);
    free(mpd[i].H_0);
    free(mpd[i].h_0);
    free(mpd[i].x_N_l);
    free(mpd[i].uh);
    free(mpd[i].udh);
    free(mpd[i].uddh);
    free(mpd[i].udddh);
    free(mpd[i].u_zoh);
    free(mpd[i].ud_zoh);
    free(mpd[i].udd_zoh);
    free(mpd[i].uddd_zoh);
    for (int j = 0; j < mpd[i].mapd->dim; j++) free(mpd[i].E_full[j]);
    free(mpd[i].E_full);
    if (mpd[i].mapd->map_type == POS_VEL) {
      for (int j = 0; j < POS_VEL; j++) free(mpd[i].E[j]);
      free(mpd[i].E);
    }

    // Free mapping data
    freeMapping(mpd[i].mapd);
  }
  return 0;
}
