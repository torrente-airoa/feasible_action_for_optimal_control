#include "helper.h"

#include "daqp/utils.h"
#include "daqp_lp.h"
#include "libqhull_r/qhull_ra.h"
#include "math_utils.h"

void scaleDynNSets(mpdata *mpd, const scale_t scale_type) {
  // Scale dynamics and input/state sets
  scaleDyn(scale_type, &(mpd->joint_lims), mpd->tau_c, mpd->A, mpd->B, mpd->D, &(mpd->d));

  // shrink position and speed limits so that they are not violated after sampling
  double Delta = 0;      // shrinkage value for position in unscaled (original) units
  double Delta_dot = 0;  // shrinkage value for speed in unscaled (original) units
  getShrinkageDeltas(&(mpd->joint_lims), mpd->tau_c, &Delta_dot, &Delta);
  mpd->x_max = malloc(mpd->mapd->dim * sizeof(double));
  mpd->x_max[0] = mpd->D[0] * (mpd->joint_lims.qup - Delta);
  mpd->x_max[1] = mpd->D[1] * (mpd->joint_lims.qdotup - Delta_dot);
  mpd->x_max[2] = mpd->D[2] * mpd->joint_lims.qddotup;

  // define symmetric feasible state set both in V- and H-representation
  mpd->V_X[0][0] = -mpd->x_max[0];  // Note: vertex set in CCW order required
  mpd->V_X[1][0] = -mpd->x_max[0];
  mpd->V_X[2][0] = -mpd->x_max[0];
  mpd->V_X[3][0] = -mpd->x_max[0];
  mpd->V_X[4][0] = mpd->x_max[0];
  mpd->V_X[5][0] = mpd->x_max[0];
  mpd->V_X[6][0] = mpd->x_max[0];
  mpd->V_X[7][0] = mpd->x_max[0];

  mpd->V_X[0][1] = -mpd->x_max[1];
  mpd->V_X[1][1] = -mpd->x_max[1];
  mpd->V_X[2][1] = mpd->x_max[1];
  mpd->V_X[3][1] = mpd->x_max[1];
  mpd->V_X[4][1] = -mpd->x_max[1];
  mpd->V_X[5][1] = -mpd->x_max[1];
  mpd->V_X[6][1] = mpd->x_max[1];
  mpd->V_X[7][1] = mpd->x_max[1];

  mpd->V_X[0][2] = -mpd->x_max[2];
  mpd->V_X[1][2] = mpd->x_max[2];
  mpd->V_X[2][2] = -mpd->x_max[2];
  mpd->V_X[3][2] = mpd->x_max[2];
  mpd->V_X[4][2] = -mpd->x_max[2];
  mpd->V_X[5][2] = mpd->x_max[2];
  mpd->V_X[6][2] = -mpd->x_max[2];
  mpd->V_X[7][2] = mpd->x_max[2];

  mpd->H_0 = malloc(mpd->mapd->dim * sizeof(double *));
  mpd->h_0 = malloc(mpd->mapd->dim * sizeof(double));
  for (int j = 0; j < mpd->mapd->dim; j++) {
    mpd->H_0[j] = malloc(mpd->mapd->dim * sizeof(double));
    for (int iota = 0; iota < mpd->mapd->dim; iota++) mpd->H_0[j][iota] = (j == iota) ? (1) : (0);
    mpd->h_0[j] = mpd->x_max[j];
  }

  // finally, set u_max
  mpd->u_max = mpd->d * mpd->joint_lims.qdddotup;
}

void scaleDyn(const scale_t scale_type, const joint_limits *joint_lims, const double tau_c, double A[3][3], double B[3],
              double D[3], double *d) {
  // define scaling coefficients and scale dynamics matrix and input vector
  switch (scale_type) {
    case NO_SCALING:
      D[0] = 1;
      D[1] = 1;
      D[2] = 1;
      *d = 1;
      break;
    case NORMALIZE:
      // normalize variables to [-1,1] using their max values
      D[0] = 1.0 / joint_lims->qup;
      D[1] = 1.0 / joint_lims->qdotup;
      D[2] = 1.0 / joint_lims->qddotup;
      *d = 1.0 / joint_lims->qdddotup;
      break;
    case TIME_BASED:
      // scaling based on tau_c (makes dynamics independent of tau_c)
      D[0] = 1;
      D[1] = tau_c;
      D[2] = pow(tau_c, 2);
      *d = pow(tau_c, 3);
      break;
  }

  // scale A and B: A <- D*A*inv(D), B <- D*B*(1/d)
  A[0][0] = 1;
  A[0][1] = (D[0] * tau_c) / D[1];
  A[0][2] = (D[0] * pow(tau_c, 2)) / (2 * D[2]);
  A[1][0] = 0;
  A[1][1] = 1;
  A[1][2] = (D[1] * tau_c) / D[2];
  A[2][0] = 0;
  A[2][1] = 0;
  A[2][2] = 1;
  B[0] = (D[0] * pow(tau_c, 3)) / (6 * (*d));
  B[1] = (D[1] * pow(tau_c, 2)) / (2 * (*d));
  B[2] = (D[2] * tau_c) / (*d);
}

void getShrinkageDeltas(const joint_limits *lims, const double tau_c, double *Delta_dot, double *Delta) {
  *Delta_dot = min((1.0 / 2) * lims->qdddotup, lims->qddotup / tau_c) * pow(tau_c, 2) / 4.0;
  double q_star_2 = min(lims->qddotup / 2.0, (lims->qdotup - *Delta_dot) / tau_c - (1.0 / 6) * lims->qdddotup * tau_c);
  double t_star = -q_star_2 / ((1.0 / 2) * lims->qdddotup) +
                  sqrt(pow(q_star_2 / ((1.0 / 2) * lims->qdddotup), 2) +
                       ((1.0 / 6) * lims->qdddotup * pow(tau_c, 2) + q_star_2 * tau_c) / ((1.0 / 2) * lims->qdddotup));
  *Delta = (-1.0 / 6) * lims->qdddotup * pow(t_star, 3) - q_star_2 * pow(t_star, 2) +
           (1.0 / 6 * lims->qdddotup * pow(tau_c, 2) + q_star_2 * tau_c) * t_star;
}

void getRandomInitialState(mpdata *mpd) {
  for (int i = 0; i < mpd->mapd->dim; i++) mpd->x_0[i] = uniformSample(-mpd->x_max[i], mpd->x_max[i]);
}

void getRandomAbstractAction(mpdata *mpd) {
  for (int i = 0; i < (int)mpd->mapd->map_type; i++)
    mpd->mapd->z_N_l[i] = uniformSample(mpd->mapd->z_min[i], mpd->mapd->z_max[i]);
}

void zohSampleSpline(mpdata *mpd) {
  // Zero-order hold sampling of cubic spline.

  // Note: sample at last time step as well, i.e. there are N+1 sampling points
  double t_k = 0;
  int l_tau_c = 0;
  double sampl_coeff[4] = {1.0 / mpd->D[0], 1.0 / mpd->D[1], 1.0 / (2 * mpd->D[2]), 1.0 / (6 * mpd->d)};
  double sampl_coeff_vel[3] = {1.0 / mpd->D[1], 1.0 / mpd->D[2], 1.0 / (2 * mpd->d)};
  double sampl_coeff_acc[2] = {1.0 / mpd->D[2], 1.0 / mpd->d};
  double sampl_coeff_jerk[1] = {1.0 / mpd->d};
  double dt = 0;
  for (int k = 0; k < (mpd->N + 1); k++) {
    // determine in which interval l_tau_c of length tau_c the point t_k = k*tau_s lies
    // (assuming 0-indexing of tau_c intervals)
    t_k = k * mpd->tau_s;
    if (k == mpd->N)
      l_tau_c = mpd->mapd->N_l - 1;
    else
      l_tau_c = floor(t_k / mpd->tau_c);

    // sample continuous position trajectory
    dt = max(min((t_k - (l_tau_c * mpd->tau_c)), mpd->tau_c), 0.0);
    mpd->u_zoh[k] = sampl_coeff[0] * mpd->uh[l_tau_c] +
                    (sampl_coeff[1] * (mpd->udh[l_tau_c]) + sampl_coeff[2] * (mpd->uddh[l_tau_c]) * dt +
                     sampl_coeff[3] * (mpd->udddh[l_tau_c]) * dt * dt) *
                      dt;
    mpd->ud_zoh[k] = sampl_coeff_vel[0] * mpd->udh[l_tau_c] +
                     (sampl_coeff_vel[1] * (mpd->uddh[l_tau_c]) + sampl_coeff_vel[2] * (mpd->udddh[l_tau_c]) * dt) * dt;
    mpd->udd_zoh[k] = sampl_coeff_acc[0] * mpd->uddh[l_tau_c] + sampl_coeff_acc[1] * mpd->udddh[l_tau_c] * dt;
    mpd->uddd_zoh[k] = sampl_coeff_jerk[0] * mpd->udddh[l_tau_c];
  }
}

void printArr(const char *name, const int m, const int n, const void *V) {
  printf("%s = [\n", name);
  if (n > 1) {
    const double **VV = (const double **)V;
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) printf("\t%18.15f%s", VV[i][j], (j == (n - 1)) ? ("\n") : (","));
  } else {
    const double *VV = (const double *)V;
    for (int i = 0; i < m; i++) printf("\t%18.15f\n", VV[i]);
  }
  printf("]\n");
}

void printArrInt(const char *name, const int m, const int n, const void *V) {
  printf("%s = [\n", name);
  if (n > 1) {
    const int **VV = (const int **)V;
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) printf("\t%d%s", VV[i][j], (j == (n - 1)) ? ("\n") : (","));
  } else {
    const int *VV = (const int *)V;
    for (int i = 0; i < m; i++) printf("\t%d\n", VV[i]);
  }
  printf("]\n");
}

void setCol(c_float *A, const int num_cols, const int start_row, const int col, const int len,
            const c_float rowcoef[]) {
  // Set column col of coefficient matrix with num_cols columns
  // Note: Matrix A is represented in row-major order
  int index = 0;
  for (int i = start_row; i < (start_row + len); i++) {
    index = i * num_cols + col;
    A[index] = rowcoef[i - start_row];
  }
}

int redundancyElimination(const double **H, const double *h, const int m, const int n, const double f_bar,
                          const maxctrlinvset_params *mcis_pars, int *R, int *n_R) {
  // Identifies redundant constraints in -h <= H*x <= h, where m is the number of rows of H and n the
  // number of columns. Index vector R contains the indices of the n_R redundant constraints.
  //
  // Note: - Weak constraints are not removed

  c_float A[m * n];  // Note: row-major order
  int k = 0;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) A[k++] = H[i][j];

  DAQPProblem model;
  DAQPWorkspace work;
  memset(&model, 0, sizeof(DAQPProblem));
  memset(&work, 0, sizeof(DAQPWorkspace));

  c_float Hess[n * n];  // Note: row-major order
  c_float f[n];
  c_float blower[m];
  c_float bupper[m];
  int sense[m];
  int n_iters = 0;

  // define the QP (see types.h in DAQP header files and notes for method developed on Aug 13 2024)
  model.n = n;
  model.ms = 0;
  model.m = m;
  model.H = Hess;
  for (int iota = 0; iota < (n * n); iota++) model.H[iota] = 0;
  for (int iota = 0; iota < n; iota++) model.H[iota * n + iota] = 1;  // set diagonal entries of Hessian to 1
  model.f = f;
  for (int iota = 0; iota < n; iota++) model.f[iota] = 0;
  model.blower = blower;
  model.bupper = bupper;
  for (int iota = 0; iota < m; iota++) {
    model.blower[iota] = -h[iota];
    model.bupper[iota] = h[iota];
  }
  model.A = A;
  model.sense = sense;
  for (int iota = 0; iota < m; iota++) model.sense[iota] = 0;  // all constraints are inequality constraints

  // do redundancy elimination
  work.settings = NULL;  // if not set, we will get a seg fault later on
  int ret = setup_daqp(&model, &work, NULL);
  if (ret != 1)
    return -1;
  else {
    work.settings->primal_tol = mcis_pars->primal_tol;
    work.settings->zero_tol = mcis_pars->zero_tol;
    work.settings->fval_bound = f_bar + mcis_pars->zero_tol;
  }

  int update_mask = DAQP_UPDATE_v + DAQP_UPDATE_sense;
  for (int i = 0; i < m; i++) {
    // make i'th constraint an equality constraint
    model.bupper[i] -= mcis_pars->shift_tol;  // make decision more robust
    model.blower[i] = model.bupper[i];

    // solve problem
    daqp_update_ldp_cold(update_mask, &work, work.qp);
    // daqp_deactivate_constraints(&work); // coldstart
    // reset_daqp_workspace(&work);
    // activate_constraints(&work);
    int exit_flag = daqp_ldp(&work);
    if (exit_flag == DAQP_EXIT_INFEASIBLE) {
      R[i] = 1;  // if problem is infeasible, constraint i is redundant
      // proposed workaround by Daniel Arnstroem (June 17 2024)
      daqp_deactivate_constraints(&work);
      reset_daqp_workspace(&work);
    } else if (exit_flag != DAQP_EXIT_OPTIMAL) {
      free_daqp_workspace(&work);
      free_daqp_ldp(&work);
      return -1;  // some problem in DAQP occured
    } else {
      // if problem is optimal, all active constraints are not redundant for sure (from DAQP minHRep function)
      for (int iota = 0; iota < work.n_active; iota++) R[work.WS[iota]] = 0;
    }

    // reset bounds
    model.bupper[i] = h[i];
    model.blower[i] = -h[i];

    // keep track of statistics
    n_iters += work.iterations;
    // n_iters += result.iter;
  }

  // postprocess array R, which currently consists of 1 (if constraint is redundant) and 0 (otherwise),
  // so that after postprocessing the array consists only of (absolute) indices of redundant constraints
  *n_R = 0;
  for (int iota = 0; iota < m; iota++) {
    if (R[iota]) R[(*n_R)++] = iota;
  }

  // free memory in work struct
  free_daqp_workspace(&work);
  free_daqp_ldp(&work);

  return n_iters;
}

int redundancyEliminationConvexHull(const double **H, const double *h, const int m, const int n, double *H_rm,
                                    double *H_final, int *nonR, const int m_aux, int *R, int *n_R, double **vert_rm,
                                    int *n_vert) {
  // Identifies redundant constraints in -h <= H*x <= h, where m is the number of rows of H and n is the
  // number of columns. Index vector R contains the indices of the n_R redundant constraints.
  //
  // Note: This function is based on the newly found method -- seemingly unknown in literature -- based on
  //       computing convex hulls. Using qhull is not the most efficient approach, CGAL is seemingly
  //       based on a more efficient implementation that also allows to add constraints online, which would
  //       save a considerable amount of memory. Yet, CGAL is under GPL (at least the needed function),
  //       which is why we use qhull.
  // Note: The constraints are two-sided, but the convex hull algorithm cannot exploit this property,
  //       which is why we need to exploit this fact at a higher level (cf. own notes for basic idea)

  *vert_rm = NULL;
  *n_vert = 0;

  // (i) prepare data in required format and normalize rows so that the new rhs vector becomes the one-vector
  //     Note: We are concentrating on upper bounds only in the first round, i.e. dismiss the constraints
  //           that correspond to the lower bounds
  //     Note: There cannot be a memory overflow for H_rm by construction, so, no checks are required.
  int m_eff = 0;
  int i = 0;
  int j = 0;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) H_rm[m_eff * n + j] = H[i][j] / h[i];  // Note: h[i] > 0
    m_eff++;
  }

  // (ii) compute the convex hull of all 'points' (i.e. normalized rows) in H_rm
  qhT qh_qh;
  qhT *qh = &qh_qh;
  boolT ismalloc = False;
  char flags[] = "qhull";  // Tv for verification, Ts for statistics, Tn for trace, Qt for triangulated facets
  int curlong;
  int totlong;
  qh_zero(qh, NULL);
  int exitcode = qh_new_qhull(qh, n, m_eff, H_rm, ismalloc, flags, NULL, stderr);
  if ((exitcode) || (qh->num_vertices > m_aux))  // 2*m_aux is the number of allocated vertices in H_final and m_aux is
                                                 // the number of allocated ints in nonR
  {
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    if (curlong || totlong)
      printf("qhull internal warning: did not free %d bytes of long memory (%d pieces)\n", totlong, curlong);
    return -1;  // some problem in qhull occured (check exitcode) or not enough memory for H_final/nonR was allocated
  }
  // printf("Convex Hull with %d vertices and %d facets:\n", qh->num_vertices, qh->num_facets);

  // (iii) isolate the very 'points' that are part of the convex hull, only those are needed in the second ('final')
  // round
  //       Note: No memory overflow can occur since qh->num_vertices <= m_aux as verified above
  vertexT *vertex;
  int m_eff_final = 0;
  for (vertex = qh->vertex_list; vertex && vertex->next; vertex = vertex->next) {
    i = qh_pointid(qh, vertex->point);
    for (int j = 0; j < n; j++) H_final[m_eff_final * n + j] = H[i][j] / h[i];  // Note: h[i] > 0
    nonR[m_eff_final++] = i;
  }

  // (iv) add the mirrored 'points' to H_final (this takes into account the symmetry of the set)
  for (i = 0; i < (m_eff_final * n); i++) H_final[m_eff_final * n + i] = -H_final[i];
  m_eff_final *= 2;

  // (v) compute the convex hull of all 'points' in H_final
  qhT qh_qh_final;
  qhT *qh_final = &qh_qh_final;
  qh_zero(qh_final, NULL);
  exitcode = qh_new_qhull(qh_final, n, m_eff_final, H_final, ismalloc, flags, NULL, stderr);
  if ((exitcode) || (qh_final->num_vertices %
                     2))  // Note: Due to symmetry, there must be an even number of vertices in the convex hull
  {
    // first free the memory from the first call to qhull
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    if (curlong || totlong)
      printf("qhull internal warning: did not free %d bytes of long memory (%d pieces)\n", totlong, curlong);

    // free memory from second run to qhull
    qh_freeqhull(qh_final, !qh_ALL);
    qh_memfreeshort(qh_final, &curlong, &totlong);
    if (curlong || totlong)
      printf("qhull internal warning: did not free %d bytes of long memory (%d pieces)\n", totlong, curlong);
    return -1;  // some problem in qhull occured (check exitcode)
  }

  // (vi) recover indices of _non_redundant constraints
  *n_R = 0;
  for (vertex = qh_final->vertex_list; vertex && vertex->next; vertex = vertex->next) {
    i = qh_pointid(qh_final, vertex->point);
    if (i < (m_eff_final / 2))
      R[(*n_R)++] = nonR[i];  // Note: The stored indices are the ones of the non-redundant constraints (so, R is abused
                              // in some sense)
  }

  // (vii) in order to comply with the definition of the interface, we next compute the absolute indices
  //       of the redundant constraints
  qsort(R, *n_R, sizeof(int),
        compare);  // sort indices of non-redundant constraints in ascending order (prerequisite for next steps)
  for (i = 0; i < *n_R; i++) nonR[i] = R[i];  // copy relevant indices because we need R as a return argument
  j = 0;
  int k = 0;
  for (i = 0; i < m; i++) {
    if (i == nonR[j]) {
      if (j < (*n_R - 1)) j++;
    } else
      R[k++] = i;
  }
  // assert(j + 1 != *n_R)
  // assert(k != (m - *n_R))
  *n_R = m - *n_R;  // now, *n_R holds the number of redundant constraints

  // (viii) get vertices of the MCIS
  *n_vert = qh_final->num_facets;
  *vert_rm = malloc(*n_vert * n * sizeof(double));
  facetT *facet;
  int n_aux = 0;
  for (facet = qh_final->facet_list; facet && facet->next; facet = facet->next) {
    for (int j = 0; j < n; j++) (*vert_rm)[n_aux * n + j] = (facet->normal[j] / facet->offset);
    n_aux++;
  }

  // (ix) free qhull memory
  // first free the memory from the first call to qhull
  qh_freeqhull(qh, !qh_ALL);
  qh_memfreeshort(qh, &curlong, &totlong);
  if (curlong || totlong)
    printf("qhull internal warning: did not free %d bytes of long memory (%d pieces)\n", totlong, curlong);

  // free memory from second run to qhull
  qh_freeqhull(qh_final, !qh_ALL);
  qh_memfreeshort(qh_final, &curlong, &totlong);
  if (curlong || totlong)
    printf("qhull internal warning: did not free %d bytes of long memory (%d pieces)\n", totlong, curlong);

  return 0;
}

mcis_ret_code mixConstraintsSameSet(const double **H_k, const double *h_k, const int dim, const double *g_k,
                                    const int *I, const int n_I, const double A[dim][dim], const int n_constr_max,
                                    const maxctrlinvset_params *mcis_pars, double **H_kp1, double *h_kp1, int *m_kp1,
                                    const double *vert_rm, const int n_vert) {
  // Mixing of constraints within same index set (Fourier-Motzkin elimination)

  double aux = 0;
  double aux_vec[dim];
  for (int i1 = 0; i1 < n_I; i1++) {
    for (int i2 = i1 + 1; i2 < n_I; i2++) {
      // first, make sure that we have enough memory available
      if (*m_kp1 == n_constr_max) {
        printf("Increase n_constr_max in the MCIS computation parameter struct.\n");
        return MEMORY_LIMIT;
      }

      aux = g_k[I[i2]] / g_k[I[i1]];  // Note: g_k is always a positive vector
      vecVecDiff(dim, 1.0, H_k[I[i2]], aux, H_k[I[i1]],
                 aux_vec);                             // aux_vec <- (H_k(i2,:) - (g_k(i2)/g_k(i1))*H_k(i1,:))
      vecMatMul(dim, dim, aux_vec, A, H_kp1[*m_kp1]);  // H_kp1(end,:) <- aux_vec'*A
      h_kp1[*m_kp1] = h_k[I[i2]] + aux * h_k[I[i1]];   // h_kp1(iota) <- h_k(i2) + (g_k(i2)/g_k(i1))*h_k(i1);

      // do a quick check if this constraint is redundant (only sufficient condition)
      if (isRedundant(H_kp1[*m_kp1], h_kp1[*m_kp1], vert_rm, n_vert, dim, mcis_pars->shift_tol))
        continue;  // don't increase *m_kp1 so that current constraint candidate gets overwritten in next iteration
      else
        (*m_kp1)++;
    }
  }

  return ALL_OK;
}

mcis_ret_code mixConstraints(const double **H_k, const double *h_k, const int dim, const double *g_k, const int *I,
                             const int n_I, const int *J, const int n_J, const double A[dim][dim],
                             const int n_constr_max, const maxctrlinvset_params *mcis_pars, double **H_kp1,
                             double *h_kp1, int *m_kp1, const double *vert_rm, const int n_vert) {
  // Mixing of constraints between different index sets (Fourier-Motzkin elimination)

  double aux = 0;
  double aux_vec[dim];
  for (int i = 0; i < n_I; i++) {
    for (int j = 0; j < n_J; j++) {
      // first, make sure that we have enough memory available
      if (*m_kp1 == n_constr_max) {
        printf("Increase n_constr_max in the MCIS computation parameter struct.\n");
        return MEMORY_LIMIT;
      }

      aux = g_k[I[i]] / g_k[J[j]];                               // Note: g_k is always a positive vector
      vecVecDiff(dim, 1.0, H_k[I[i]], aux, H_k[J[j]], aux_vec);  // aux_vec <- (H_k(i,:) - (g_k(i)/g_k(j))*H_k(j,:))
      vecMatMul(dim, dim, aux_vec, A, H_kp1[*m_kp1]);            // H_kp1(end,:) <- aux_vec'*A
      h_kp1[*m_kp1] = h_k[I[i]] - aux * h_k[J[j]];               // h_kp1(iota) <- h_k(i) - (g_k(i)/g_k(j))*h_k(j);

      // do a quick check if this constraint is redundant (only sufficient condition)
      if (isRedundant(H_kp1[*m_kp1], h_kp1[*m_kp1], vert_rm, n_vert, dim, mcis_pars->shift_tol))
        continue;  // don't increase *m_kp1 so that current constraint candidate gets overwritten in next iteration
      else
        (*m_kp1)++;
    }
  }

  return ALL_OK;
}

int validateSet(const double **H, const double *h, const int m, const int n, const char dir[], const int i,
                const double prefact, const maxctrlinvset_params *mcis_pars) {
  // Validate if computed maximum control invariant set is identical to a offline-validated set computed
  // with MPT3 in Matlab.
  //
  // Note: The order of the constraints in both set representations is identical.

  char full_fname_data[1000];
  char subdir_name[100];
  char fname[100];
  FILE *f_data = NULL;
  strcpy(full_fname_data, dir);
  sprintf(subdir_name, "joint%d/", i);
  sprintf(fname, "prefact_%.1f.bin", prefact);
  strcat(full_fname_data, subdir_name);
  strcat(full_fname_data, fname);
  f_data = fopen(full_fname_data, "rb");  // binary mode
  if (!f_data) {
    printf("File %s cannot be opened. Exiting.\n", full_fname_data);
    exit(-1);
  }

  // read from binary file and validate
  int m_val;
  fread(&m_val, sizeof(int), 1, f_data);
  if (m != m_val) {
    fclose(f_data);
    return 0;
  }

  int n_val;
  fread(&n_val, sizeof(int), 1, f_data);
  if (n != n_val) {
    fclose(f_data);
    return 0;
  }

  double H_row_val[n_val];
  double diff[n_val];
  double diff_norm = 0;
  for (int i = 0; i < m_val; i++) {
    fread(H_row_val, sizeof(double), n_val, f_data);
    vecVecDiff(n_val, 1.0, H_row_val, 1.0, H[i], diff);
    diff_norm = oneNorm(n_val, diff);
    if (diff_norm > mcis_pars->Hh_abs_tol) {
      fclose(f_data);
      return 0;
    }
  }

  double h_val;
  for (int i = 0; i < m_val; i++) {
    fread(&h_val, sizeof(double), 1, f_data);
    if (fabs(h_val - h[i]) > mcis_pars->Hh_abs_tol) {
      fclose(f_data);
      return 0;
    }
  }

  fclose(f_data);
  return 1;
}

int validateStep(const mpdata *mpd, FILE *f) {
  fseek(f, mpd->mapd->N_l * sizeof(double), SEEK_CUR);  // skip Chebyshev center U_bar

  // validate action
  double y[2];
  double diff_y[2];
  fread(y, sizeof(double), 2, f);
  vecVecDiff(2, 1.0, y, 1.0, mpd->mapd->y_N_l, diff_y);
  if (oneNorm(2, diff_y) / oneNorm(2, y) > 1e-6) {
    fseek(f, (mpd->mapd->N_l + 1) * sizeof(double), SEEK_CUR);  // skip remaining data
    return 0;
  }

  // validate MPC solution (Note: The MPC solution is always unique)
  double U_star[mpd->mapd->N_l];
  double diff_U[mpd->mapd->N_l];
  fread(U_star, sizeof(double), mpd->mapd->N_l, f);
  fseek(f, sizeof(double), SEEK_CUR);  // skip remaining data
  vecVecDiff(mpd->mapd->N_l, 1.0, U_star, 1.0, mpd->udddh, diff_U);
  if (oneNorm(mpd->mapd->N_l, diff_U) / oneNorm(mpd->mapd->N_l, U_star) > 1e-6) return 0;

  return 1;
}

int validateReset(const rpdata *rpd, FILE *f) {
  // validate reset trajectory (in terms of jerk)
  int N_l = 0;
  double udddh[rpd->N_l_max];

  fread(&N_l, sizeof(int), 1, f);
  fread(udddh, sizeof(double), N_l, f);
  if (N_l != rpd->N_l) return 0;

  if (rpd->obj_type == MAGN)  // optimal solution vector is valid for MAGN objective only
  {
    double diff_udddh[N_l];
    vecVecDiff(N_l, 1.0, rpd->udddh, 1.0, udddh, diff_udddh);
    if (oneNorm(N_l, diff_udddh) / oneNorm(N_l, udddh) > 1e-6) return 0;
  }

  return 1;
}

int validateTraj(const rpdata *rpd) {
  // check if terminal state equals reset state
  double x_f_MPC[rpd->dim];
  double delta_x_f[rpd->dim];
  x_f_MPC[0] = rpd->uh[rpd->N_l];
  x_f_MPC[1] = rpd->udh[rpd->N_l];
  x_f_MPC[2] = rpd->uddh[rpd->N_l];
  vecVecDiff(rpd->dim, 1.0, x_f_MPC, 1.0, rpd->x_f, delta_x_f);
  if (oneNorm(rpd->dim, delta_x_f) > 1e-6) return 0;

  // check feasibility of (reduced) position and speed limits and (original) acceleration and jerk limits
  for (int l = 0; l < rpd->N_l; l++) {
    if ((rpd->uh[l] < (1 + 1e-6) * (-rpd->x_max[0])) || (rpd->uh[l] > (1 + 1e-6) * rpd->x_max[0]) ||
        (rpd->udh[l] < (1 + 1e-6) * (-rpd->x_max[1])) || (rpd->udh[l] > (1 + 1e-6) * rpd->x_max[1]) ||
        (rpd->uddh[l] < (1 + 1e-6) * (-rpd->x_max[2])) || (rpd->uddh[l] > (1 + 1e-6) * rpd->x_max[2]) ||
        (rpd->udddh[l] < (1 + 1e-6) * (-rpd->u_max)) || (rpd->udddh[l] > (1 + 1e-6) * rpd->u_max))
      return 0;
  }

  return 1;
}

void getFileHandle(const char dir[], const int joint_num, const double prefact, const obj_t obj_type,
                   const double x_0_scaling, FILE **f_data) {
  // Get file handle for computestep test data.
  char full_fname_data[1000];
  char subdir_name[100];
  char fname[100];
  char obj_type_str[10];
  if (obj_type == DIFF)
    sprintf(obj_type_str, "DIFF");
  else if (obj_type == MIXED)
    sprintf(obj_type_str, "MIXED");
  else
    sprintf(obj_type_str, "MAGN");

  strcpy(full_fname_data, dir);
  sprintf(subdir_name, "joint%d/", joint_num);
  sprintf(fname, "prefact_%.1f_%s_x_0_scaling_%.6f.bin", prefact, obj_type_str, x_0_scaling);
  strcat(full_fname_data, subdir_name);
  strcat(full_fname_data, fname);
  *f_data = fopen(full_fname_data, "rb");  // binary mode
  if (!(*f_data)) {
    printf("File %s cannot be opened. Exiting.\n", full_fname_data);
    exit(-1);
  }
}

void compute_relative_to_dirname(const char *current_dir, const char *target_dirname, char *relative_path) {
  // Find the position of the target directory name in the current directory
  const char *target_pos = strstr(current_dir, target_dirname);
  if (!target_pos) {
    fprintf(stderr, "Error: Target directory '%s' not found in current path '%s'\n", target_dirname, current_dir);
    strcpy(relative_path, "Error");
    return;
  }

  // Count how many levels we need to go up
  int up_levels = 0;
  for (const char *p = target_pos + strlen(target_dirname); *p != '\0'; p++) {
    if (*p == '/') {
      up_levels++;
    }
  }

  // Add "../" for each level up
  char buffer[256] = {0};
  for (int i = 0; i < up_levels; i++) {
    strcat(buffer, "../");
  }

  strcpy(relative_path, buffer);
}

void getFileHandleReset(const char dir[], const int joint_num, const double prefact, const obj_t obj_type,
                        const double tau_c, const int mult, FILE **f_data) {
  // Get file handle for resetplanner test data.
  char full_fname_data[1000];
  char subdir_name[100];
  char fname[100];
  char obj_type_str[10];
  if (obj_type == DIFF)
    sprintf(obj_type_str, "DIFF");
  else if (obj_type == MIXED)
    sprintf(obj_type_str, "MIXED");
  else
    sprintf(obj_type_str, "MAGN");

  strcpy(full_fname_data, dir);
  sprintf(subdir_name, "joint%d/", joint_num);
  sprintf(fname, "prefact_%.1f_%s_tau_c_%.3f_mult_%d.bin", prefact, obj_type_str, tau_c, mult);
  strcat(full_fname_data, subdir_name);
  strcat(full_fname_data, fname);
  char cwd[PATH_MAX];
  getcwd(cwd, sizeof(cwd));
  char relative_path[PATH_MAX];
  compute_relative_to_dirname(cwd, "faoc_cubic_approx", relative_path);
  strcat(relative_path, full_fname_data);
  *f_data = fopen(relative_path, "rb");  // binary mode
  if (!(*f_data)) {
    printf("Error %d (%s). File %s cannot be opened. Exiting.\n", errno, strerror(errno), relative_path);
    exit(-1);
  }
}

void getFileHandleCheckInvSet(const char dir[], const int joint_num, const int episode_num, FILE **f_data) {
  // Get file handle for reading invariant set membership data
  char full_fname_data[1000];
  char subdir_name[100];
  char fname[100];

  strcpy(full_fname_data, dir);
  sprintf(subdir_name, "joint%d/", joint_num);
  sprintf(fname, "episode_%d.bin", episode_num);
  strcat(full_fname_data, subdir_name);
  strcat(full_fname_data, fname);
  *f_data = fopen(full_fname_data, "rb");  // binary mode
  if (!(*f_data)) {
    printf("File %s cannot be opened. Exiting.\n", full_fname_data);
    exit(-1);
  }
}

int setupMPC(mpdata *mpd, const algo_params *alg_pars) {
  // Set up MPC problem with allowed deviations from the terminal pos/speed constraint
  // Note: We assume the following variable ordering: (u_0, u_1, ..., u_{N_l-1})

  DAQPProblem *model = &(mpd->mpc);
  DAQPWorkspace *work = &(mpd->mpc_work);

  memset(model, 0, sizeof(DAQPProblem));
  memset(work, 0, sizeof(DAQPWorkspace));

  const int N_l = mpd->mapd->N_l;
  const int dim = mpd->mapd->dim;
  const int n_inf = mpd->n_inf;

  // (i) define the size of the QP (see types.h in DAQP header files)
  model->n = N_l;
  model->ms = N_l;
  model->m = model->ms + dim * (N_l - 1) + 2 + n_inf;

  // (ii) define objective
  model->H = malloc(N_l * N_l * sizeof(c_float));  // Note: row-major order
  for (int iota = 0; iota < (N_l * N_l); iota++) model->H[iota] = 0;
  if (mpd->obj_type == MAGN) {
    for (int l = 0; l < N_l; l++) model->H[l * N_l + l] = 1;  // set diagonal entries of Hessian to 1
  } else if (mpd->obj_type == DIFF) {
    for (int l = 0; l < N_l; l++) model->H[l * N_l + l] = 2;  // set diagonal entries of Hessian to 2
    model->H[N_l * N_l - 1] = 1;  // don't penalize (future - last accleration)^2, so this entry is 1, not 2
    for (int l = 1; l < N_l; l++) model->H[l * N_l + l - 1] = -1;  // set sub-diagonal entries of Hessian to -1
    for (int l = 0; l < (N_l - 1); l++)
      model->H[l * N_l + l + 1] = -1;  // set sup-diagonal entries of Hessian to -1
                                       // in addition, the linear gradient vector f is
    // f = [-uddh_m1, 0, 0, ..., 0], but since it depends on a parameter,
    // it needs to be set online
  } else if (mpd->obj_type == MIXED) {
    for (int l = 0; l < N_l; l++) model->H[l * N_l + l] = 1;  // set diagonal entries of Hessian to 1
    model->H[0] = 2;
    // in addition, the linear gradient vector f is
    // f = [-uddh_m1, 0, 0, ..., 0], but since it depends on a parameter,
    // it needs to be set online
  } else {
    printf("Unknown obj_type %d. Exiting.\n", mpd->obj_type);
    exit(-1);  // not that nice but otherwise freeData() becomes cumbersome since pointers in DAQPProblem are not
               // initialized to NULL
  }
  model->f = malloc(N_l * sizeof(c_float));
  for (int l = 0; l < N_l; l++) model->f[l] = 0;

  // (iii) define bound vectors (both for constraints and variables)
  model->blower = malloc((model->m) * sizeof(c_float));
  model->bupper = malloc((model->m) * sizeof(c_float));
  for (int iota = 0; iota < model->m; iota++) {
    model->blower[iota] = -DAQP_INF;  // will be set online to a meaningful value
    model->bupper[iota] = DAQP_INF;   // will be set online to a meaningful value
  }

  // (iv) define state constraints
  model->A = malloc((model->m - model->ms) * model->n * sizeof(c_float));
  for (int iota = 0; iota < ((model->m - model->ms) * model->n); iota++) model->A[iota] = 0;

  // for auxiliary purposes, compute the vector [B; A*B; A^2*B; ...; A^(N_l-1)*B]
  c_float rowcoef[dim * N_l];
  for (int iota = 0; iota < dim; iota++) rowcoef[iota] = mpd->B[iota];
  for (int l = 1; l < N_l; l++) matVecMul2(dim, dim, mpd->A, rowcoef + (l - 1) * dim, rowcoef + l * dim);

  // set columns that encode state constraints (dim*(N_l-1) constraints,
  // but set dim*N_l constraints next and overwrite the last one later)
  for (int l = 0; l < N_l; l++)
    setCol(model->A, N_l, dim * l, l, dim * N_l - dim * l, rowcoef);  // set coefficient matrix

  // (v) define matrices E and E_full
  for (int i = 0; i < dim; i++)
    for (int j = 0; j < N_l; j++) {
      mpd->E_full[i][j] = model->A[dim * (N_l - 1) * N_l + i * N_l + j];
      if ((i < 2) && (mpd->mapd->map_type == POS_VEL)) mpd->E[i][j] = model->A[dim * (N_l - 1) * N_l + i * N_l + j];
    }

  // (vii) terminal pos/speed constraints: Note that they are already there, but we are using only the first two rows of
  // the last bunch of state constraints
  //       Also note that we always have a pos and a speed constraint in the formulation, independent of the map type
  //       (we simply adjust the lower/upper bounds appropriately at runtime to deactivate the speed constraint, if
  //       applicable)

  // (viii) encode maximum control invariant set
  int ind_A = -1;
  for (int j = 0; j < n_inf; j++) {
    ind_A = (dim * (N_l - 1) + 2 + j) * model->n;
    for (int iota = 0; iota < model->n; iota++)
      model->A[ind_A + iota] = scalarProd(dim, mpd->H_inf[j], rowcoef + (N_l * dim - (iota + 1) * dim));
  }

  // (ix) define sense vector
  model->sense = malloc(model->m * sizeof(int));
  for (int iota = 0; iota < model->m; iota++) model->sense[iota] = 0;  // all constraints are inequality constraints

  // (x) prepare for an efficient approach of using DAQP online
  work->settings = NULL;  // if not set, we will get a seg fault later on
  work->iterations = 0;   // if not set, then warm-starting might not work
  int ret = setup_daqp(model, work, NULL);
  if (ret != 1)
    return -1;
  else {
    work->settings->primal_tol = alg_pars->primal_tol;
    work->settings->dual_tol = alg_pars->dual_tol;
    work->settings->zero_tol = alg_pars->zero_tol;
  }

  return 0;
}

void exportDAQP(DAQPProblem *model, DAQPSettings *set) {
  // export DAQP problem struct and settings into binary file
  static int count = -1;
  count++;
  // open file for binary write access
  char fname[100];
  sprintf(fname, "lpdata/lp%d.bin", count);
  FILE *f_data = fopen(fname, "wb");
  if (!f_data) {
    printf("File %s cannot be opened. Exiting.\n", fname);
    exit(-1);
  }
  // write problem struct
  fwrite(&model->n, sizeof(int), 1, f_data);
  fwrite(&model->m, sizeof(int), 1, f_data);
  fwrite(&model->ms, sizeof(int), 1, f_data);
  if (model->H) fwrite(model->H, sizeof(c_float), model->n * model->n, f_data);
  if (model->f) fwrite(model->f, sizeof(c_float), model->n, f_data);
  fwrite(model->A, sizeof(c_float), (model->m - model->ms) * model->n, f_data);
  fwrite(model->bupper, sizeof(c_float), model->m, f_data);
  fwrite(model->blower, sizeof(c_float), model->m, f_data);
  fwrite(model->sense, sizeof(int), model->m, f_data);
  // write settings
  fwrite(set, sizeof(DAQPSettings), 1, f_data);
  fclose(f_data);
}

void exportTrajs(const rpdata *rpd, const int n_joints, const char dirname[], const int j) {
  // export reset trajectories to binary files

  char fname[250];
  for (int i = 0; i < n_joints; i++) {
    // open file for binary write access
    sprintf(fname, "%s/joint%d/scenario%d.bin", dirname, i, j);
    FILE *f_data = fopen(fname, "wb");
    if (!f_data) {
      printf("File %s cannot be opened. Exiting.\n", fname);
      exit(-1);
    }

    // unscale acceleration trajectory
    const int len = rpd[i].N_l + 1;
    double udd[len];
    for (int l = 0; l < len; l++) udd[l] = rpd[i].uddh[l] / rpd[i].D[2];

    // write data
    fwrite(&len, sizeof(int), 1, f_data);
    fwrite(udd, sizeof(double), len, f_data);
    fclose(f_data);
  }
}

void rpd2mpd(rpdata *rpd, mpdata *mpd) {
  // init only those fields that are necessary to compute the MCIS
  mpd->mapd->dim = rpd->dim;
  mpd->joint_lims.qup = rpd->joint_lims.qup;
  mpd->joint_lims.qdotup = rpd->joint_lims.qdotup;
  mpd->joint_lims.qddotup = rpd->joint_lims.qddotup;
  mpd->joint_lims.qdddotup = rpd->joint_lims.qdddotup;
  mpd->tau_c = rpd->mult * rpd->tau_c;  // (!)
  mpd->scale_type = rpd->scale_type;
  mpd->mcis_pars.primal_tol = rpd->mcis_pars_mult.primal_tol;
  mpd->mcis_pars.zero_tol = rpd->mcis_pars_mult.zero_tol;
  mpd->mcis_pars.shift_tol = rpd->mcis_pars_mult.shift_tol;
  mpd->mcis_pars.h_rel_tol = rpd->mcis_pars_mult.h_rel_tol;
  mpd->mcis_pars.Hh_abs_tol = rpd->mcis_pars_mult.Hh_abs_tol;
  mpd->mcis_pars.max_iter = rpd->mcis_pars_mult.max_iter;
  mpd->mcis_pars.n_constr_max = rpd->mcis_pars_mult.n_constr_max;
  strcpy(mpd->mcis_pars.re_method, rpd->mcis_pars_mult.re_method);
}

void mpd2rpd(mpdata *mpd, rpdata *rpd) {
  // map back only the relevant results after MCIS computation
  for (int i = 0; i < mpd->mapd->dim; i++) {
    rpd->B_mult[i] = mpd->B[i];
    rpd->D[i] = mpd->D[i];
    for (int j = 0; j < mpd->mapd->dim; j++) rpd->A_mult[i][j] = mpd->A[i][j];
  }
  rpd->d = mpd->d;
  rpd->x_max_mult = mpd->x_max;  // we are responsible to free this memory later on
  mpd->x_max = NULL;             // just to be sure
  rpd->u_max_mult = mpd->u_max;
  rpd->H_inf_mult = mpd->H_inf;        // we are responsible to free this memory later on
  rpd->H_inf_mult_rm = mpd->H_inf_rm;  // we are responsible to free this memory later on
  rpd->h_inf_mult = mpd->h_inf;        // we are responsible to free this memory later on
  rpd->n_inf_mult = mpd->n_inf;
  rpd->mcis_ret = mpd->mcis_ret;

  // H_0 and h_0 are allocated in MCIS computation but not needed anymore in rpd, so free this memory
  for (int j = 0; j < mpd->mapd->dim; j++) free(mpd->H_0[j]);
  free(mpd->H_0);
  free(mpd->h_0);
}

int initMPCProb(DAQPProblem *model, DAQPWorkspace *work, const algo_params *alg_pars, const double A[3][3],
                const double B[3], const int N_l, const int dim, double *rowcoef) {
  // init MPC problem, i.e. only allocate the maximally needed memory, nullify it
  // and pre-compute some of the data so that online the problem can be constructed efficiently

  int n = N_l;
  model->H = malloc(n * n * sizeof(c_float));
  for (int iota = 0; iota < (n * n); iota++) model->H[iota] = 0;

  model->f = malloc(n * sizeof(c_float));
  for (int l = 0; l < n; l++) model->f[l] = 0;

  int m = n + N_l * dim;
  model->blower = malloc(m * sizeof(c_float));
  model->bupper = malloc(m * sizeof(c_float));
  for (int iota = 0; iota < m; iota++) {
    model->blower[iota] = -DAQP_INF;  // will be set online to a meaningful value
    model->bupper[iota] = DAQP_INF;   // will be set online to a meaningful value
  }

  model->A = malloc((m - n) * n * sizeof(c_float));
  for (int iota = 0; iota < ((m - n) * n); iota++) model->A[iota] = 0;

  model->sense = malloc(m * sizeof(int));
  for (int iota = 0; iota < m; iota++) model->sense[iota] = 0;

  // in order to set up the MPC problem efficiently online, we pre-compute
  // the vector [B; A*B; A^2*B; ...; A^(N_l-1)*B]
  for (int iota = 0; iota < dim; iota++) rowcoef[iota] = B[iota];
  for (int l = 1; l < N_l; l++) matVecMul2(dim, dim, A, rowcoef + (l - 1) * dim, rowcoef + l * dim);

  // prepare for an efficient approach of using DAQP online
  // Note: The idea is to set up the work struct for the max size, but online use only the
  //       relevant data chunks (requires to adjust sizes in the work struct appropriately!)
  model->n = n;
  model->ms = n;
  model->m = m;

  // set diagonal entries of Hessian to 1 (this is just to ensure that the work struct below can be initialized)
  for (int l = 0; l < N_l; l++) model->H[l * N_l + l] = 1;

  work->settings = NULL;  // if not set, we will get a seg fault later on
  work->iterations = 0;   // if not set, then warm-starting might not work
  int ret = setup_daqp(model, work, NULL);
  if (ret != 1)
    return -1;
  else {
    work->settings->primal_tol = alg_pars->primal_tol;
    work->settings->dual_tol = alg_pars->dual_tol;
    work->settings->zero_tol = alg_pars->zero_tol;
  }

  return 0;
}

void shrinkState(const double *x_out, const double **H, const double *h, const int m, const int n,
                 const double shrink_fact, double *x_in) {
  // Given x_out that is not in a symmetric polytope {x | -h <= H*x < h, H \in \Reals{m \times n}}, this
  // function computes a new point x_in = alpha*x_out with 0 <= alpha < 1 being the largest shrink factor such
  // that x_in is in the symmetric polytope.
  // Note: It turns out that for numerical robustness it makes sense to shrink x_in a bit more such that it
  //       does not end up at the boundary of the symmetric polytope (helps with feasibility issues in
  //       subsequent MPC problems (cf. shrink_fact)).

  double g[m];
  double alpha_plus = 1e1;
  double alpha_minus = 1e1;
  matVecMul(m, n, H, x_out, g);  // g <- H*x_out
  for (int i = 0; i < m; i++) {
    if (g[i] > 0)
      alpha_plus = min(alpha_plus, h[i] / g[i]);
    else if (g[i] < 0)
      alpha_minus = min(alpha_minus, -h[i] / g[i]);
  }
  double alpha = min(alpha_plus, alpha_minus);
  for (int i = 0; i < n; i++) x_in[i] = shrink_fact * alpha * x_out[i];
}

int mpcBisect(DAQPProblem *model, DAQPWorkspace *work, const int dim, const double A[3][3], const double *rowcoef,
              const double *x_max, const double u_max, int N_l_lo, const int N_l_init, int N_l_up, const double *x_0,
              const double *x_f, const double udddh_m1, const obj_t obj_type, const bisect_params *bs_params,
              double *lambda_star) {
  // implement a bisection method to determine the (near) minimum-time control horizon for MPC
  //
  // Note: It must be guaranteed by the caller, that there is enough memory available
  //       for the maximum MPC horizon length of N_l_up.

  int N_l_max = N_l_up;
  int N_l_min = N_l_max + 1;  // (near) optimal horizon length (solution of bisection) Note: A value > N_l_max indicates
                              // that there is no valid horizon length so far
  N_l_lo = (int)max((double)dim, (double)N_l_lo);        // always have at least a control horizon of dim
  int N_l = (int)max((double)N_l_lo, (double)N_l_init);  // set initial horizon to test for feasibility
  int update_mask =
    DAQP_UPDATE_Rinv + DAQP_UPDATE_M + DAQP_UPDATE_v + DAQP_UPDATE_d + DAQP_UPDATE_sense;  // update everything
  int exit_flag;
  int ret = 0;
  while (1) {
    // for a horizon length of N_l set up the MPC problem
    ret = mpcProb(model, work, dim, A, rowcoef, x_max, u_max, N_l, x_0, x_f, udddh_m1, obj_type);
    if (ret < 0) return ret;

    // solve the MPC problem
    ret = daqp_update_ldp_cold(update_mask, work, work->qp);
    if (ret < 0) return ret;
    exit_flag = daqp_ldp(work);

    // Note: We are not transforming the primal/dual solution with ldp2qp_solution(work) to save time
    //       So, the lower bound of 1e10 below on the 1-norm might have to be set differently than in
    //       the Matlab prototype. But for warmstarting the MPC problem at the original tau_c discretization,
    //       only the signs of the Lagrange multipliers are important, not the actual scaling.
    if ((exit_flag == DAQP_EXIT_INFEASIBLE) ||
        (oneNorm(work->n_active, work->lam_star) >
         1e10))  // Note: We introduce the norm condition to avoid corner cases in infeasibility detection, leading to
                 // too short horizons for the MPC probelm with tau_c discretization later on
    {
      // problem is infeasible -> increase N_l_lo
      N_l_lo = N_l;
    } else if (exit_flag == DAQP_EXIT_OPTIMAL) {
      // problem is feasible -> decrease N_l_up
      N_l_up = N_l;
      // save currently best horizon length together with Lagrange multiplier
      N_l_min = N_l;
      for (int i = 0; i < model->m; i++) lambda_star[i] = 0;
      for (int i = 0; i < work->n_active; i++)
        lambda_star[work->WS[i]] = work->lam_star[i];  // save Lagrange multiplier for potential warmstarting
    } else {
      daqp_deactivate_constraints(work);
      reset_daqp_workspace(work);
      return -1;
    }

    // make sure DAQP behaves correctly in next iteration
    daqp_deactivate_constraints(work);
    reset_daqp_workspace(work);

    // check termination criterion
    // Note: Either terminate when a feasible horizon length was found and the gap between this horizon length
    //       and the lower bound is small enough OR if the gap vanishes, which happens only if no feasible horizon
    //       length exists (in this case, N_l_up must be chosen larger by the caller of this function)
    if (((N_l_min <= N_l_max) && ((N_l_min - N_l_lo) <= bs_params->dN_max)) || (N_l_up - N_l_lo == 0)) break;

    // prepare for next iteration
    N_l = ceil((N_l_lo + N_l_up) / 2.0);
  }

  // If we end up here, we have found a feasible horizon length N_l_min. Now, if the user requested
  // to extend this horizon length by add_steps, we compute another optimization problem next with this
  // new horizon length in order to allow for a warmstart in mpcSolve() (cf. computeResetTraj())
  if (bs_params->add_steps > 0) {
    if ((int)min((double)(N_l_min + bs_params->add_steps), (double)(N_l_max)) > N_l_min) {
      // Note: N_l_max has the meaning of the maximum possible horizon length (in our implementation
      //       this corresponds to the maximum possible problem size for which memory was allocated)
      N_l_min = (int)min((double)(N_l_min + bs_params->add_steps), (double)(N_l_max));
      ret = mpcProb(model, work, dim, A, rowcoef, x_max, u_max, N_l_min, x_0, x_f, udddh_m1, obj_type);
      if (ret < 0) return ret;
      ret = daqp_update_ldp_cold(update_mask, work, work->qp);
      if (ret < 0) return ret;
      exit_flag = daqp_ldp(work);
      if (exit_flag == DAQP_EXIT_OPTIMAL) {
        // save Lagrange multiplier for potential warmstarting in mpcSolve()
        for (int i = 0; i < model->m; i++) lambda_star[i] = 0;
        for (int i = 0; i < work->n_active; i++) lambda_star[work->WS[i]] = work->lam_star[i];
      } else {
        daqp_deactivate_constraints(work);
        reset_daqp_workspace(work);
        return -1;
      }
    }
  }

  return N_l_min;
}

double minTContTime(const double *x_0, double s_f, const double *x_max, const double *D) {
  // Compute minimum traversal time t_f for continuous time-optimal control of a double integrator
  // starting at (1./D)*x_0 (state ordering: pos [m], speed [m/s]) and going to terminal position
  // (1/D[0])*s_f [m], where terminal speed is assumed to be zero. Position, speed and acceleration
  // limits are given by (1./D)*x_max (3D vector with coordinates in [m], [m/s], [m/s^2]).
  //
  // See notes from Dec 3-5 2024 for new derivation. Assumes symmetric speed and acceleration limits.

  double s_0 = x_0[0] / D[0];
  double v_0 = x_0[1] / D[1];
  s_f /= D[0];
  double v_up = x_max[1] / D[1];
  double a_up = x_max[2] / D[2];

  // Note: The theory is derived for case s_f >= s_0 only. Due to symmetry, we may mirror the initial and
  //       end conditions in the other case:
  if (s_f < s_0) {
    s_0 = -s_0;
    v_0 = -v_0;
    s_f = -s_f;
  }

  // distinguish between overshoot and non-overhoot case
  double t_f = -1;
  if ((v_0 > 0) && (pow(v_0, 2) / (2 * a_up) > (s_f - s_0))) {
    // overshoot case
    double t_s = v_0 / a_up + sqrt(pow(v_0, 2) / (2 * pow(a_up, 2)) - (1 / a_up) * (s_f - s_0));
    t_f = 2 * t_s - v_0 / a_up;
  } else {
    // no-overshoot case
    double t_a_up = (v_up - v_0) / a_up;
    double t_a = -v_0 / a_up + sqrt(pow(v_0, 2) / (2 * pow(a_up, 2)) + (s_f - s_0) / a_up);
    if ((t_a >= 0) && (t_a <= t_a_up)) {
      // CASE A (t_a = t_d, i.e. no constant speed phase)
      t_f = 2 * t_a + v_0 / a_up;
    } else {
      // CASE B (t_a < t_d)
      t_a = t_a_up;
      double t_d = (1 / v_up) * (s_f - s_0 + (v_0 / a_up) * (v_0 / 2 - v_up));
      t_f = t_d + t_a + v_0 / a_up;
    }
  }

  return t_f;
}

int mpcProb(DAQPProblem *model, DAQPWorkspace *work, const int dim, const double A[3][3], const double *rowcoef,
            const double *x_max, const double u_max, const int N_l, const double *x_0, const double *x_f,
            const double udddh_m1, const obj_t obj_type) {
  int n = N_l;
  int ms = N_l;
  int m = ms + N_l * dim;

  // set problem sizes
  model->n = n;
  model->ms = ms;
  model->m = m;

  // it turns out that if the problem gets updated before a solve, the problem dimensions are not updated,
  // so we take care of this ourselves
  work->n = n;
  work->ms = ms;
  work->m = m;

  // Hessian
  for (int iota = 0; iota < (N_l * N_l); iota++) model->H[iota] = 0;
  if (obj_type == MAGN) {
    for (int l = 0; l < N_l; l++) model->H[l * N_l + l] = 1;  // set diagonal entries of Hessian to 1
  } else if (obj_type == DIFF) {
    for (int l = 0; l < N_l; l++) model->H[l * N_l + l] = 2;  // set diagonal entries of Hessian to 2
    model->H[N_l * N_l - 1] = 1;  // don't penalize (future - last jerk)^2, so this entry is 1, not 2
    for (int l = 1; l < N_l; l++) model->H[l * N_l + l - 1] = -1;        // set sub-diagonal entries of Hessian to -1
    for (int l = 0; l < (N_l - 1); l++) model->H[l * N_l + l + 1] = -1;  // set sup-diagonal entries of Hessian to -1
  } else if (obj_type == MIXED) {
    for (int l = 0; l < N_l; l++) model->H[l * N_l + l] = 1;  // set diagonal entries of Hessian to 1
    model->H[0] = 2;
    // in addition, the linear gradient vector f is
    // f = [-uddh_m1, 0, 0, ..., 0], but since it depends on a parameter,
    // it needs to be set online
  } else
    return -2;

  // gradient of linear term
  if (obj_type == DIFF || obj_type == MIXED)
    model->f[0] = -udddh_m1;  // Note: We require mpd->udddh_m1 to be a _scaled_ jerk

  // set bound vectors
  for (int l = 0; l < N_l; l++) {
    model->blower[l] = -u_max;
    model->bupper[l] = u_max;
  }

  // compute max possible objective value for early termination in case the MPC problem is infeasible
  double f_upper = DAQP_INF;
  if (obj_type == MAGN) f_upper = 0.5 * N_l * pow(u_max, 2) + work->settings->zero_tol;
  work->settings->fval_bound = f_upper;

  // set lower/upper bounds on state constraints
  c_float res[dim];
  c_float prod[dim];
  for (int iota = 0; iota < dim; iota++) prod[iota] = x_0[iota];
  int offset = N_l;
  for (int l = 0; l < (N_l - 1); l++) {
    matVecMul2(dim, dim, A, prod, res);
    for (int iota = 0; iota < dim; iota++) prod[iota] = res[iota];
    for (int iota = 0; iota < dim; iota++) {
      model->blower[offset + l * dim + iota] = (c_float)-x_max[iota] - prod[iota];
      model->bupper[offset + l * dim + iota] = (c_float)x_max[iota] - prod[iota];
    }
  }

  // set lower/upper bounds on terminal state constraint
  matVecMul2(dim, dim, A, prod, res);  // res = A^{N_l}*x_0
  offset += (N_l - 1) * dim;
  for (int iota = 0; iota < dim; iota++) {
    model->blower[offset + iota] = (c_float)x_f[iota] - res[iota];
    model->bupper[offset + iota] = (c_float)x_f[iota] - res[iota];
  }

  // set coefficient matrix
  for (int iota = 0; iota < ((m - ms) * n); iota++) model->A[iota] = 0;
  for (int l = 0; l < n; l++) setCol(model->A, N_l, dim * l, l, dim * N_l - dim * l, rowcoef);

  // finally, set the sense vector
  for (int iota = 0; iota < (m - dim); iota++) model->sense[iota] = 0;
  for (int iota = m - dim; iota < m; iota++)
    model->sense[iota] = DAQP_ACTIVE + DAQP_IMMUTABLE;  // terminal state constraint is an equality constraint

  return 0;
}

int mpcSolve(DAQPProblem *model, DAQPWorkspace *work, const int dim, const double A[3][3], const double *rowcoef,
             const double *x_max, const double u_max, int *N_l, const int N_l_max, const double *x_0, const double *x_f,
             const double udddh_m1, const obj_t obj_type, const int mult, const double *lambda_star_mult,
             bool EXTEND_ON_INF, double *udddh) {
  // Solve MPC problem with horizon length N_l, which could probably be infeasible. Note that
  // N_l_max is the maximum possible horizon length (in terms of allocated memory) and this function
  // extends the horizon length by mult steps everytime the chosen horizon leads to an infeasible problem.
  // If lambda_star_mult != NULL, we use it for warmstarting of the initial working set.

  // set up problem
  int ret = mpcProb(model, work, dim, A, rowcoef, x_max, u_max, *N_l, x_0, x_f, udddh_m1, obj_type);
  if (ret < 0) return ret;

  // initialize working set ('warmstarting'), if applicable
  if (lambda_star_mult) {
    int N_l_mult = *N_l / mult;  // works always, i.e. division is without remainder

    // determine active input constraints
    for (int i = 0; i < mult; i++)
      for (int k = 0; k < N_l_mult; k++)
        if (lambda_star_mult[k] < -work->settings->dual_tol)
          model->sense[k * mult + i] = DAQP_ACTIVE + DAQP_LOWER;
        else if (lambda_star_mult[k] > work->settings->dual_tol)
          model->sense[k * mult + i] = DAQP_ACTIVE;

    // determine active state constraints
    for (int j = 0; j < dim; j++)  // first position, then speed, then acceleration
      for (int i = 0; i < mult; i++)
        for (int k = 0; k < N_l_mult - 1; k++)
          if (lambda_star_mult[N_l_mult + j + k * dim] < -work->settings->dual_tol)
            model->sense[*N_l + j + k * dim * mult + i * dim + dim] = DAQP_ACTIVE + DAQP_LOWER;
          else if (lambda_star_mult[N_l_mult + j + k * dim] > work->settings->dual_tol)
            model->sense[*N_l + j + k * dim * mult + i * dim + dim] = DAQP_ACTIVE;

    // correct sense of terminal state constraint, which is always an equality constraint
    for (int j = 0; j < dim; j++) model->sense[*N_l + (*N_l - 1) * dim + j] = DAQP_ACTIVE + DAQP_IMMUTABLE;
  }

  // solve the MPC problem
  int update_mask =
    DAQP_UPDATE_Rinv + DAQP_UPDATE_M + DAQP_UPDATE_v + DAQP_UPDATE_d + DAQP_UPDATE_sense;  // update everything
  ret = daqp_update_ldp_cold(update_mask, work, work->qp);
  if (ret < 0)  // Note: The problem seems to be related to an invalid sense vector
  {
    // as a workaround perform a coldstart
    for (int iota = 0; iota < (model->m - dim); iota++) model->sense[iota] = 0;
    for (int iota = model->m - dim; iota < model->m; iota++)
      model->sense[iota] = DAQP_ACTIVE + DAQP_IMMUTABLE;  // terminal state constraint is an equality constraint
    ret = daqp_update_ldp_cold(update_mask, work, work->qp);
    if (ret < 0) return ret;  // coldstart did not help
  }
  int exit_flag = daqp_ldp(work);

  if ((EXTEND_ON_INF) && (exit_flag == DAQP_EXIT_INFEASIBLE)) {
    // horizon was too short and we are allowed to extend the horizon length
    *N_l += mult;
    if (*N_l > N_l_max) return -1;  // problem remains infeasible, e.g., because N_l_max was chosen too short

    // reset DAQP and solve with longer horizon
    daqp_deactivate_constraints(work);
    reset_daqp_workspace(work);
    ret = mpcSolve(model, work, dim, A, rowcoef, x_max, u_max, N_l, N_l_max, x_0, x_f, udddh_m1, obj_type, mult, NULL,
                   EXTEND_ON_INF, udddh);  // no warmstarting!
    return ret;
  } else if (exit_flag == DAQP_EXIT_OPTIMAL) {
    // save optimal (scaled) jerk sequence
    ldp2qp_solution(work);                                 // convert solution
    for (int l = 0; l < *N_l; l++) udddh[l] = work->x[l];  // save solution
    return 0;
  } else
    return -1;
}

int compare(const void *a, const void *b) {
  // needed for qsort
  int int_a = *((int *)a);
  int int_b = *((int *)b);

  return (int_a > int_b) - (int_a < int_b);
}

int getMaxHorizons(const rpdata *rpd, const int dim, const int n_joints, const double tau_c, const int max_N_l,
                   const double f_s, const scale_t scale_type, const bool PARALLEL, const int n_threads,
                   const double *reset_states_lo, const double *reset_states_up, const maxctrlinvset_params *mcis_pars,
                   const algo_params *mpc_alg_pars, int *N_l_max) {
  // compute _minimum_ horizon length N_l_max for joint i such that all initial states
  // from the corresponding MCIS can be brought to the corresponding reset states within
  // that horizon

  threadpool thpool;
  if (PARALLEL) thpool = thpool_init(n_threads);
  max_horizon_data mhd[n_joints];
  for (int i = 0; i < n_joints; i++)
    if (N_l_max[i] <= 0) {
      // (a) fill data into mhd struct
      mhd[i].joint_lims = &(rpd[i].joint_lims);
      mhd[i].dim = dim;
      mhd[i].tau_c = tau_c;
      mhd[i].f_s = f_s;
      mhd[i].scale_type = scale_type;
      mhd[i].mcis_pars = mcis_pars;
      mhd[i].mpc_alg_pars = mpc_alg_pars;
      mhd[i].max_N_l = max_N_l;
      mhd[i].reset_state_lo = reset_states_lo + i * dim;
      mhd[i].reset_state_up = reset_states_up + i * dim;
      if ((mhd[i].reset_state_lo[1] != 0) || (mhd[i].reset_state_lo[2] != 0) || (mhd[i].reset_state_up[1] != 0) ||
          (mhd[i].reset_state_up[2] != 0)) {
        printf("The speed and acceleration components of the reset states must be zero.\n");
        if (PARALLEL) thpool_destroy(thpool);
        return -1;
      }

      if (PARALLEL)
        thpool_add_work(thpool, computeMaxHorizon, (void *)(mhd + i));
      else
        computeMaxHorizon((void *)(mhd + i));
    }
  if (PARALLEL) {
    thpool_wait(thpool);
    thpool_destroy(thpool);
  }

  // check if an error occured, otherwise read out computed horizon length
  for (int i = 0; i < n_joints; i++)
    if (N_l_max[i] <= 0) {
      if (mhd[i].N_l_max <= 0)
        return -1;
      else
        N_l_max[i] = mhd[i].N_l_max;
    }

  return 0;
}

void computeMaxHorizon(void *arg) {
  // for a given joint, compute N_l_max, i.e. the _minimum_ horizon length N_l_max
  // such that all initial states from the corresponding MCIS can be brought to the
  // corresponding reset states within that horizon

  max_horizon_data *mhd = (max_horizon_data *)arg;

  // compute MCIS and other data needed for time-optimal MPC problem that we need
  // to solve in calculateMaxHorizon() below
  int N_l = mhd->dim;  // set arbitrarily
  const double z_min[2] = {-1, -1};
  const double z_max[2] = {1, 1};
  double init_time = 0;
  mpdata mpd;
  int ret = initMapping(&mpd.mapd, mhd->dim, N_l, z_min, z_max, POS, &init_time);
  if (ret < 0) {
    printf("Initialization of mapdata struct failed with return code %d. Exiting.\n", ret);
    exit(-1);  // not nice, since we don't clean up all memory, but ok for our purpose, since this may only happen at
               // initialization time
  }

  mpd.joint_lims.qup = mhd->joint_lims->qup;
  mpd.joint_lims.qdotup = mhd->joint_lims->qdotup;
  mpd.joint_lims.qddotup = mhd->joint_lims->qddotup;
  mpd.joint_lims.qdddotup = mhd->joint_lims->qdddotup;
  ret = initData(&mpd, MAGN, 1, mhd->tau_c, mhd->f_s, mhd->scale_type, false, 1, mhd->mcis_pars, mhd->mpc_alg_pars,
                 mhd->mpc_alg_pars, mhd->mpc_alg_pars, mhd->mpc_alg_pars, mhd->mpc_alg_pars, &init_time);
  if (ret < 0) {
    printf("Initialization of mpdata struct failed with return code %d. Exiting.\n", ret);
    exit(-1);  // not nice, since we don't clean up all memory, but ok for our purpose, since this may only happen at
               // initialization time
  }

  int N_l_max_lb = 0;
  calculateMaxHorizon(&mpd, mhd->reset_state_lo, mhd->max_N_l, N_l_max_lb, mhd->mpc_alg_pars, &(mhd->N_l_max));
  if (mhd->N_l_max <= 0)  // some problem occured in calculateMaxHorion
  {
    freeData(&mpd, 1);
    return;
  }
  if (mhd->reset_state_lo[0] != mhd->reset_state_up[0]) {
    N_l_max_lb =
      mhd->N_l_max;  // to speed up computations, we set the lower bound to the previously found max horizon length
    calculateMaxHorizon(&mpd, mhd->reset_state_up, mhd->max_N_l, N_l_max_lb, mhd->mpc_alg_pars, &(mhd->N_l_max));
    // no error check necessary here, since we free up the memory of the mpd struct and then return anyway
  }

  // wrap up
  freeData(&mpd, 1);
}

void calculateMaxHorizon(const mpdata *mpd, const double *reset_state, const int max_N_l, const int N_l_max_lb,
                         const algo_params *mpc_alg_params, int *N_l_max) {
  // given an MCIS in the mpdata struct, compute N_l_max by computing all vertices of the MCIS and
  // chosing the longest of the shortest horizons as N_l_max

  // (i) compute convex hull of dual MCIS polytope
  double H_inf_rm_sc[2 * mpd->n_inf * mpd->mapd->dim];
  for (int iota = 0; iota < mpd->n_inf; iota++)  // prepare data for qhull
    for (int j = 0; j < mpd->mapd->dim; j++)
      H_inf_rm_sc[iota * mpd->mapd->dim + j] =
        mpd->H_inf[iota][j] / mpd->h_inf[iota];  // normalization is central to approach
  for (int j = 0; j < mpd->n_inf * mpd->mapd->dim; j++) H_inf_rm_sc[mpd->n_inf * mpd->mapd->dim + j] = -H_inf_rm_sc[j];

  qhT qh_qh;
  qhT *qh = &qh_qh;
  boolT ismalloc = False;
  char flags[] = "qhull";  // Tv for verification, Ts for statistics, Tn for trace, Qt for triangulated facets
  int curlong;
  int totlong;
  qh_zero(qh, NULL);
  int exitcode = qh_new_qhull(qh, mpd->mapd->dim, 2 * mpd->n_inf, H_inf_rm_sc, ismalloc, flags, NULL, stderr);
  if ((exitcode) || (qh->num_facets %
                     2))  // the number of facets of the dual polytope is the same as the number of vertices of the MCIS
  {
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    if (curlong || totlong)
      printf("qhull internal warning: did not free %d bytes of long memory (%d pieces)\n", totlong, curlong);
    *N_l_max = -1;  // some problem in qhull occured (check exitcode or number of facets)
    return;
  }

  // (ii) set up MPC problem to determine time-optimal horizon length for each vertex of the MCIS
  //      as an initial state
  DAQPProblem mpc;     // condensed MPC model
  DAQPWorkspace work;  // workspace for mpc
  memset(&mpc, 0, sizeof(DAQPProblem));
  memset(&work, 0, sizeof(DAQPWorkspace));
  double rowcoeff[max_N_l * mpd->mapd->dim];
  int ret = initMPCProb(&mpc, &work, mpc_alg_params, mpd->A, mpd->B, max_N_l, mpd->mapd->dim, rowcoeff);
  if (ret < 0) {
    // free what can be freed
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    freeData((mpdata *)mpd, 1);
    printf("Initialization of MPC problem failed with return code %d. Exiting.\n", ret);
    exit(-1);  // not nice, since we don't clean up all memory for the MPC problem, but ok for our purpose, since this
               // may only happen at initialization time
  }

  // (iii) finally determine N_l_max using bisection
  facetT *facet;
  double x_0[mpd->mapd->dim];
  double x_f[mpd->mapd->dim];
  for (int j = 0; j < mpd->mapd->dim; j++) x_f[j] = reset_state[j] * mpd->D[j];  // scale reset state
  *N_l_max = N_l_max_lb;
  int N_l = 0;
  int N_l_lo = 0;
  int N_l_up = max_N_l;  // upper bound on control horizon
  int N_l_init = 0;
  bisect_params bs_params = {.add_steps = 0, .add_steps_init = 0, .dN_max = 1};  // dN_max = 1 for optimality
  double lambda_star[max_N_l + max_N_l * mpd->mapd->dim];
  int j = 0;
  const double pre_fact = 1 - 1e-11;  // to increase numerical robustness (avoids infeasibilities in MPC problem)
  for (facet = qh->facet_list; facet && facet->next; facet = facet->next) {
    // (a) get vertex and set it as initial state of the MPC problem
    for (j = 0; j < mpd->mapd->dim; j++)
      x_0[j] =
        pre_fact * (facet->normal[j] /
                    facet->offset);  // scaled since MCIS is scaled (pre_fact to avoid infeasible MPC problems below)

    // // exploit symmetry of the MCIS: if the negative initial state was already visited, then skip it (only valid if
    // reset state is the origin) if (facet->previous)
    // {
    //     for (facetT *prev_facet = facet->previous; prev_facet; prev_facet = prev_facet->previous)
    //     {
    //         for (j = 0; j < mpd->mapd->dim; j++)
    //             if (fabs(x_0[j] + (pre_fact * (prev_facet->normal[j] / prev_facet->offset))) > 1e-16) // from
    //             empirical tests, choose 1e-16 or value below
    //                 break;
    //         if (j == mpd->mapd->dim) // -x_0 was visited already
    //             break;
    //     }
    //     if (j == mpd->mapd->dim)
    //         continue;
    // }

    // (b) compute horizon bounds and initial horizon for bisection
    N_l_lo = ceil(minTContTime(x_0, x_f[0], mpd->x_max, mpd->D) / (mpd->tau_c));  // lower bound on control horizon
    N_l_lo = max(N_l_lo, *N_l_max);                                               // consider incumbent for N_l_max
    N_l_init = N_l_lo;  // initial control horizon (eventually, this will be the best choice in the loop, but not
                        // necessarily in the beginning)
    if (N_l_lo > N_l_up) {
      // no feasible horizon (i.e. max_N_l needs to be increased!)
      printf("Increase max_N_l.\n");
      *N_l_max = -1;
      break;
    }

    // (c) perform bisection
    N_l = mpcBisect(&mpc, &work, mpd->mapd->dim, mpd->A, rowcoeff, mpd->x_max, mpd->u_max, N_l_lo, N_l_init, N_l_up,
                    x_0, x_f, 0, MAGN, &bs_params, lambda_star);
    // error checks
    if (N_l <= 0) {
      // something went wrong in the bisection method (always related to DAQP)
      *N_l_max = N_l;
      break;
    }
    if (N_l > N_l_up) {
      // no feasible horizon length exists (i.e. max_N_l needs to be increased!)
      printf("Increase max_N_l.\n");
      *N_l_max = -1;
      break;
    }

    // prepare for next iteration
    *N_l_max = N_l;  // no max operation necessary since the max() is already applied for setting the lower bound in the
                     // bisection
  }

  // free memory of qhull
  qh_freeqhull(qh, !qh_ALL);
  qh_memfreeshort(qh, &curlong, &totlong);

  // free memory of MPC problem
  free(mpc.H);
  free(mpc.f);
  free(mpc.blower);
  free(mpc.bupper);
  free(mpc.A);
  free(mpc.sense);
  free_daqp_workspace(&work);
  free_daqp_ldp(&work);
}

int computeVertices(const double **H, const double *h, const int m, const int n, double **vert_rm, int *n_vert) {
  // Given {x | -h <= H*x <= h, h \in \Reals{m}}, this function computes all vertices
  // of this set.

  // bring representation into format required by approach (normalized rows, row-major order)
  double H_rm_sc[2 * m * n];
  for (int iota = 0; iota < m; iota++)
    for (int j = 0; j < n; j++) H_rm_sc[iota * n + j] = H[iota][j] / h[iota];  // normalization is central to approach
  for (int j = 0; j < m * n; j++) H_rm_sc[m * n + j] = -H_rm_sc[j];

  // compute convex hull of dual polytope
  qhT qh_qh;
  qhT *qh = &qh_qh;
  boolT ismalloc = False;
  char flags[] = "qhull";  // Tv for verification, Ts for statistics, Tn for trace, Qt for triangulated facets
  int curlong;
  int totlong;
  qh_zero(qh, NULL);
  int exitcode = qh_new_qhull(qh, n, 2 * m, H_rm_sc, ismalloc, flags, NULL, stderr);
  int ret = 0;
  if ((exitcode) || (qh->num_facets %
                     2))  // the number of facets of the dual polytope is the same as the number of vertices of the MCIS
    ret = -1;

  // get vertices of original set
  *n_vert = qh->num_facets;
  *vert_rm = malloc(*n_vert * n * sizeof(double));
  facetT *facet;
  int n_aux = 0;
  for (facet = qh->facet_list; facet && facet->next; facet = facet->next) {
    for (int j = 0; j < n; j++) (*vert_rm)[n_aux * n + j] = (facet->normal[j] / facet->offset);
    n_aux++;
  }

  // wrap up
  qh_freeqhull(qh, !qh_ALL);
  qh_memfreeshort(qh, &curlong, &totlong);
  if (curlong || totlong)
    printf("qhull internal warning: did not free %d bytes of long memory (%d pieces)\n", totlong, curlong);

  return ret;
}

bool isRedundant(const double *a, const double b, const double *vert_rm, const int n_vert, const int n,
                 const double tol) {
  // Determines if constraint -b <= a'*x <= b is redundant for the vertex set given by vert_rm with
  // n_vert vertices. If false, it could still be redundant.

  (void)n;  // just to silence compiler

  // Note: It turns out that hardcoding the loop as follows significantly speeds up computation time.
  //       For the original loop see the commented code below.
  // Note: Assumes n=3 (cubic spline case)
  for (int iota = 0; iota < n_vert; iota++)
    if (fabs(a[0] * vert_rm[iota * 3] + a[1] * vert_rm[iota * 3 + 1] + a[2] * vert_rm[iota * 3 + 2]) > (b - tol))
      return false;

  // // slow, original version
  // for (int iota = 0; iota < n_vert; iota++)
  //     if (fabs(scalarProd(n, a, vert_rm + iota * n)) > (b - tol)) // -tol to be on the safe side, i.e. better declare
  //     a constraint as non-redundant than redundant
  //         return false;

  return true;  // indicates that constraint (a,b) can be dismissed
}

void computeSyncResetTraj(void *arg) {
  // compute synced reset trajectory

  rpdata *rpd = (rpdata *)arg;
  const int dim = rpd->dim;

  // (i) solve the original MPC problem with tau_c discretization using rpd->N_l as the horizon length
  int ret =
    mpcSolve(&(rpd->mpc), &(rpd->work), rpd->dim, rpd->A, rpd->rowcoef, rpd->x_max, rpd->u_max, &(rpd->N_l),
             rpd->N_l_max, rpd->x_0, rpd->x_f, rpd->udddh_m1, rpd->obj_type, rpd->mult, NULL, false, rpd->udddh);
  if (ret < 0) {
    // problem in MPC
    rpd->planner_ret = MPC_ERR;
    return;
  }
  rpd->N = floor((rpd->N_l * rpd->tau_c) / rpd->tau_s);

  // (ii) simulate state trajectory (needed for ZOH sampling)
  rpd->uh[0] = rpd->x_0[0];
  rpd->udh[0] = rpd->x_0[1];
  rpd->uddh[0] = rpd->x_0[2];
  double x[dim];
  double x_plus[dim];
  x[0] = rpd->uh[0];
  x[1] = rpd->udh[0];
  x[2] = rpd->uddh[0];
  for (int l = 0; l < rpd->N_l; l++) {
    matVecMul2(dim, dim, rpd->A, x, x_plus);
    for (int iota = 0; iota < dim; iota++) x[iota] = x_plus[iota] + rpd->B[iota] * rpd->udddh[l];
    rpd->uh[l + 1] = x[0];
    rpd->udh[l + 1] = x[1];
    rpd->uddh[l + 1] = x[2];
  }

  // (iii) ZOH sampling
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

void updateBounds(mpdata *mpd, c_float *U_min, c_float *X_min, c_float *pos_speed_min, c_float *x_term_min,
                  c_float *U_max, c_float *X_max, c_float *pos_speed_max, c_float *x_term_max, double *e,
                  double *e_full) {
  const int N_l = mpd->mapd->N_l;
  const int dim = mpd->mapd->dim;
  const int n_inf = mpd->n_inf;
  c_float res[dim];
  c_float prod[dim];

  for (int l = 0; l < N_l; l++) {
    U_min[l] = -mpd->u_max;
    U_max[l] = mpd->u_max;
  }

  for (int iota = 0; iota < dim; iota++) prod[iota] = mpd->x_0[iota];

  for (int l = 0; l < (N_l - 1); l++) {
    matVecMul2(dim, dim, mpd->A, prod, res);
    for (int iota = 0; iota < dim; iota++) prod[iota] = res[iota];

    for (int iota = 0; iota < dim; iota++) {
      *(X_min++) = (c_float)-mpd->x_max[iota] - prod[iota];
      *(X_max++) = (c_float)mpd->x_max[iota] - prod[iota];
    }
  }

  if ((pos_speed_min) && (pos_speed_max)) {
    matVecMul2(dim, dim, mpd->A, prod, res);
    pos_speed_min[0] = (c_float)(mpd->mapd->y_N_l[0] + mpd->term_state_devs.delta_qlow * mpd->D[0]) - res[0];
    pos_speed_max[0] = (c_float)(mpd->mapd->y_N_l[0] + mpd->term_state_devs.delta_qup * mpd->D[0]) - res[0];
    if (mpd->mapd->map_type == POS_VEL) {
      pos_speed_min[1] = (c_float)(mpd->mapd->y_N_l[1] + mpd->term_state_devs.delta_qdotlow * mpd->D[1]) - res[1];
      pos_speed_max[1] = (c_float)(mpd->mapd->y_N_l[1] + mpd->term_state_devs.delta_qdotup * mpd->D[1]) - res[1];
    } else {
      pos_speed_min[1] = -DAQP_INF;
      pos_speed_max[1] = DAQP_INF;
    }
  }

  matVecMul2(dim, dim, mpd->A, prod, res);
  for (int iota = 0; iota < n_inf; iota++) {
    const double aux = scalarProd(dim, mpd->H_inf[iota], res);
    x_term_min[iota] = (c_float)(-mpd->h_inf[iota] - aux);
    x_term_max[iota] = (c_float)(mpd->h_inf[iota] - aux);
  }

  if (e) {
    e[0] = res[0];
    e[1] = res[1];
  }

  if (e_full)
    for (int iota = 0; iota < dim; iota++) e_full[iota] = res[iota];
}
