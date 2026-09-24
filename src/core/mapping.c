#include "mapping.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "daqp/constants.h"
#include "daqp/utils.h"
#include "daqp_lp.h"
#include "math_utils.h"

static int validateRuntimeBounds(const map_runtime_bounds *bounds) {
  return (bounds && bounds->lower && bounds->upper && bounds->e_full) ? 0 : -1;
}

static void applyFeasibleBoundsToPosSet(map_data *mapd, const int n_inf, const map_runtime_bounds *bounds) {
  const int n_mpc_constr = mapd->N_l + mapd->dim * (mapd->N_l - 1) + n_inf;
  for (int i = 0; i < n_mpc_constr; i++) {
    mapd->pos_set.blower[i] = bounds->lower[i];
    mapd->pos_set.bupper[i] = bounds->upper[i];
  }
}

static void applyFeasibleBoundsToChebyshev(map_data *mapd, const int n_inf, const map_runtime_bounds *bounds) {
  const int n_mpc_constr = mapd->N_l + mapd->dim * (mapd->N_l - 1) + n_inf;
  const int offset = n_mpc_constr;
  for (int i = 0; i < n_mpc_constr; i++) {
    mapd->chebyshev.bupper[i] = bounds->upper[i];
    mapd->chebyshev.blower[offset + i] = bounds->lower[i];
  }
}

static void applyFeasibleBoundsToBeta(map_data *mapd, const int n_inf, const map_runtime_bounds *bounds) {
  const int N_l = mapd->N_l;
  const int n_x = mapd->dim * (N_l - 1);

  for (int i = 0; i < N_l; i++) {
    mapd->beta_comp.blower[i] = bounds->lower[i];
    mapd->beta_comp.bupper[i] = bounds->upper[i];
  }
  for (int i = 0; i < n_x; i++) {
    mapd->beta_comp.blower[N_l + 1 + i] = bounds->lower[N_l + i];
    mapd->beta_comp.bupper[N_l + 1 + i] = bounds->upper[N_l + i];
  }
  for (int i = 0; i < n_inf; i++) {
    mapd->beta_comp.blower[N_l + 1 + n_x + i] = bounds->lower[N_l + n_x + i];
    mapd->beta_comp.bupper[N_l + 1 + n_x + i] = bounds->upper[N_l + n_x + i];
  }
}

static void applyFeasibleBoundsToDirectChebyshev(map_data *mapd, const int n_inf, const c_float *blower_feas,
                                                 const c_float *bupper_feas, const double *e_tilde) {
  const int N_l = mapd->N_l;
  const int dim = mapd->dim;

  // (i) set upper bounds of 4 * 2 inequalities
  int i = 1;
  mapd->direct_chebyshev.bupper[i++] = -e_tilde[0];
  mapd->direct_chebyshev.bupper[i++] = -e_tilde[1];
  mapd->direct_chebyshev.bupper[i++] = e_tilde[0];
  mapd->direct_chebyshev.bupper[i++] = e_tilde[1];
  mapd->direct_chebyshev.bupper[i++] = -e_tilde[1];
  mapd->direct_chebyshev.bupper[i++] = e_tilde[0];
  mapd->direct_chebyshev.bupper[i++] = e_tilde[1];
  mapd->direct_chebyshev.bupper[i++] = -e_tilde[0];

  // (ii) set lower/upper bounds corresponding to feasible sets of MPC problem (without terminal pos/speed constraints)
  const int n_mpc_constr = N_l + dim * (N_l - 1) + n_inf;
  for (i = 0; i < 4; i++) {
    for (int l = 0; l < n_mpc_constr; l++) {
      mapd->direct_chebyshev.blower[9 + i * n_mpc_constr + l] = blower_feas[l];
      mapd->direct_chebyshev.bupper[9 + i * n_mpc_constr + l] = bupper_feas[l];
    }
  }
}

int initMapping(map_data **mapd_out, const int dim, const int N_l, const double z_min[], const double z_max[],
                const map_t map_type, double *init_time) {
  if (!mapd_out || !init_time || !z_min || !z_max) return -1;

  if (dim != 3)  // only cubic splines allowed so far
  {
    *init_time = 0;
    return -2;
  }

  *mapd_out = calloc(1, sizeof(map_data));
  if (!*mapd_out) {
    *init_time = 0;
    return -1;
  }

  map_data *mapd = *mapd_out;
  mapd->dim = dim;
  mapd->N_l = N_l;
  mapd->map_type = map_type;
  mapd->z_N_l = malloc(map_type * sizeof(double));
  mapd->z_min = malloc(map_type * sizeof(double));
  mapd->z_max = malloc(map_type * sizeof(double));
  mapd->y_N_l = malloc(map_type * sizeof(double));
  for (int iota = 0; iota < (int)map_type; iota++) {
    mapd->z_min[iota] = z_min[iota];
    mapd->z_max[iota] = z_max[iota];
  }
  if (map_type == POS_VEL) {
    mapd->G = malloc(POS_VEL * sizeof(double *));
    for (int j = 0; j < POS_VEL; j++) mapd->G[j] = malloc(POS_VEL * sizeof(double));
    mapd->G_inv = malloc(POS_VEL * sizeof(double *));
    for (int j = 0; j < POS_VEL; j++) mapd->G_inv[j] = malloc(POS_VEL * sizeof(double));
  }
  *init_time = 0;
  return 0;
}

int freeMapping(map_data *mapd) {
  free(mapd->z_min);
  free(mapd->z_max);
  free(mapd->y_N_l);
  free(mapd->z_N_l);
  if (mapd->map_type == POS_VEL) {
    // free Chebyshev LP
    free(mapd->chebyshev.f);
    free(mapd->chebyshev.blower);
    free(mapd->chebyshev.bupper);
    free(mapd->chebyshev.A);
    free(mapd->chebyshev.sense);
    free_daqp_workspace(&(mapd->chebyshev_work));
    free_daqp_ldp(&(mapd->chebyshev_work));

    // free Chebyshev minimum-center norm QP
    free(mapd->chebyshev_unique.H);
    free(mapd->chebyshev_unique.f);
    free(mapd->chebyshev_unique.blower);
    free(mapd->chebyshev_unique.bupper);
    free(mapd->chebyshev_unique.A);
    free(mapd->chebyshev_unique.sense);
    free_daqp_workspace(&(mapd->unique_work));
    free_daqp_ldp(&(mapd->unique_work));

    // free direct Chebyshev LP
    free(mapd->direct_chebyshev.f);
    free(mapd->direct_chebyshev.blower);
    free(mapd->direct_chebyshev.bupper);
    free(mapd->direct_chebyshev.A);
    free(mapd->direct_chebyshev.sense);
    free_daqp_workspace(&(mapd->direct_work));
    free_daqp_ldp(&(mapd->direct_work));

    // free direct Chebyshev minimum-center norm QP
    free(mapd->direct_unique.H);
    free(mapd->direct_unique.f);
    free(mapd->direct_unique.blower);
    free(mapd->direct_unique.bupper);
    free(mapd->direct_unique.A);
    free(mapd->direct_unique.sense);
    free_daqp_workspace(&(mapd->direct_unique_work));
    free_daqp_ldp(&(mapd->direct_unique_work));

    // free LP for beta computation
    free(mapd->beta_comp.f);
    free(mapd->beta_comp.blower);
    free(mapd->beta_comp.bupper);
    free(mapd->beta_comp.A);
    free(mapd->beta_comp.sense);
    free_daqp_workspace(&(mapd->beta_work));
    free_daqp_ldp(&(mapd->beta_work));

    for (int j = 0; j < POS_VEL; j++) free(mapd->G[j]);
    free(mapd->G);
    for (int j = 0; j < POS_VEL; j++) free(mapd->G_inv[j]);
    free(mapd->G_inv);
  } else {
    free(mapd->pos_set.f);
    free(mapd->pos_set.blower);
    free(mapd->pos_set.bupper);
    free(mapd->pos_set.A);
    free(mapd->pos_set.sense);
    free_daqp_workspace(&(mapd->pos_work));
    free_daqp_ldp(&(mapd->pos_work));
  }
  free(mapd);
  return 0;
}

map_ret_code mapFromAbstractSetWithBounds(map_data *mapd, const int n_inf, const map_runtime_bounds *bounds) {
  const int N_l = mapd->N_l;
  int update_mask;
  int exit_flag;
  map_ret_code ret = 0;

  if (validateRuntimeBounds(bounds) != 0) return OPT_ERR;

  if (mapd->map_type == POS) {
    // in this case both the abstract and the action set are 1D intervals only, which
    // considerably simplifies the mapping
    // (i) compute the action set interval
    double y_lo = 0;  // lower interval end
    double y_up = 0;  // upper interval end
    ret = getActionSetPosWithBounds(mapd, n_inf, bounds, &y_lo, &y_up);
    if (ret == OPT_ERR)
      return ret;
    else if (ret == NO_INV) {
      mapd->y_N_l[0] = y_lo;
      return ret;
    } else {
      // (ii) if the interval is not a singleton (and no error occurred), then perform the mapping
      //      Idea: There is an affine map between [z_min, z_max] and [y_lo, y_up], which is given by:
      mapd->y_N_l[0] = y_lo + ((y_up - y_lo) / (mapd->z_max[0] - mapd->z_min[0])) * (mapd->z_N_l[0] - mapd->z_min[0]);
      return ret;
    }
  } else {
    // (i) compute unique interior point y_bar of action set
    double y_bar[2] = {0};
    double e[2] = {0};
    ret = getIntPointActionSetWithBounds(mapd, n_inf, bounds, y_bar, e);
    if (ret == OPT_ERR)
      return ret;
    else if (ret == NO_INV) {
      // if the action set has no interior, we map the abstract action to the feasible point y_bar (which is not an
      // interior point)
      mapd->y_N_l[0] = y_bar[0];
      mapd->y_N_l[1] = y_bar[1];
      return ret;
    }

    // (ii) map sample z_N_l \in [z_min, z_max] into an element y in the action set Y
    double z_bar[2] = {0};  // centroid of rectangle [z_min, z_max]
    z_bar[0] = (1.0 / 2) * (mapd->z_min[0] + mapd->z_max[0]);
    z_bar[1] = (1.0 / 2) * (mapd->z_min[1] + mapd->z_max[1]);
    double delta_z[2] = {0};
    delta_z[0] = mapd->z_N_l[0] - z_bar[0];
    delta_z[1] = mapd->z_N_l[1] - z_bar[1];
    if (max(fabs(delta_z[0]), fabs(delta_z[1])) < 1e-12) {
      mapd->y_N_l[0] = y_bar[0];
      mapd->y_N_l[1] = y_bar[1];
      return INV;
    }

    // compute alpha >= 1
    double alpha = DBL_MAX;
    for (int i = 0; i < 2; i++) {
      if (delta_z[i] > 0)
        alpha = min(alpha, (mapd->z_max[i] - z_bar[i]) / delta_z[i]);
      else if (delta_z[i] < 0)
        alpha = min(alpha, (mapd->z_min[i] - z_bar[i]) / delta_z[i]);
      // Note: The case where delta_z is the zero vector is excluded by the if clause above
    }

    // compute delta_y
    double delta_y[2] = {0};
    matVecMul(2, 2, (const double **)mapd->G, delta_z, delta_y);
    // normalize delta_y for better numerics in the coefficient matrix of the optimization problem (but does not
    // increase accuracy for some reason)
    double one_norm_delta_y = oneNorm(2, delta_y);
    if (one_norm_delta_y < 1e-4) {
      delta_y[0] /= one_norm_delta_y;
      delta_y[1] /= one_norm_delta_y;
    }

    // compute beta > 0
    applyFeasibleBoundsToBeta(mapd, n_inf, bounds);

    // in addition, we need to update the lhs/rhs of the equality constraints...
    mapd->beta_comp.blower[mapd->beta_comp.m - 2] = y_bar[0] - e[0];
    mapd->beta_comp.blower[mapd->beta_comp.m - 1] = y_bar[1] - e[1];
    mapd->beta_comp.bupper[mapd->beta_comp.m - 2] = y_bar[0] - e[0];
    mapd->beta_comp.bupper[mapd->beta_comp.m - 1] = y_bar[1] - e[1];
    // ... and set -delta_y in the coefficient matrix
    int i_0 = (mapd->beta_comp.m - mapd->beta_comp.ms - 2) * mapd->beta_comp.n + N_l;
    mapd->beta_comp.A[i_0] = -delta_y[0];
    mapd->beta_comp.A[i_0 + mapd->beta_comp.n] = -delta_y[1];

    // mapd->beta_comp.blower[N_l] = -DAQP_INF; // Note: Has no effect.
    mapd->beta_comp.sense[N_l] = DAQP_ACTIVE + DAQP_LOWER;  // Note: Sufficient to do this once, but ok this way.
    update_mask = DAQP_UPDATE_M + DAQP_UPDATE_sense + DAQP_UPDATE_v;
    daqp_update_ldp_cold(update_mask, &(mapd->beta_work), mapd->beta_work.qp);
    exit_flag = daqp_prox(&(mapd->beta_work));
    if (exit_flag != DAQP_EXIT_OPTIMAL) {
      daqp_deactivate_constraints(&(mapd->beta_work));
      reset_daqp_workspace(&(mapd->beta_work));
      return OPT_ERR;
    }
    double beta = mapd->beta_work.x[N_l];

    // compute mapped point
    mapd->y_N_l[0] = y_bar[0] + (beta / alpha) * delta_y[0];
    mapd->y_N_l[1] = y_bar[1] + (beta / alpha) * delta_y[1];

    return INV;
  }
}

map_ret_code mapToAbstractSetWithBounds(map_data *mapd, const int n_inf, const map_runtime_bounds *bounds) {
  const int N_l = mapd->N_l;
  int update_mask;
  int exit_flag;
  map_ret_code ret = 0;

  if (validateRuntimeBounds(bounds) != 0) return OPT_ERR;

  if (mapd->map_type == POS) {
    // in this case both the abstract and the action set are 1D intervals only, which
    // considerably simplifies the mapping
    // (i) compute the action set interval
    double y_lo = 0;  // lower interval end
    double y_up = 0;  // upper interval end
    ret = getActionSetPosWithBounds(mapd, n_inf, bounds, &y_lo, &y_up);
    if (ret == OPT_ERR)
      return ret;
    else if (ret == NO_INV) {
      // if the action set has no interior, we return the center point of the abstract action set
      mapd->z_N_l[0] = (1.0 / 2) * (mapd->z_min[0] + mapd->z_max[0]);
      return ret;
    } else {
      // (ii) if the interval is not a singleton (and no error occurred), then perform the mapping
      //      Idea: There is an affine map between [y_lo, y_up] and [z_min, z_max], which is given by:
      mapd->z_N_l[0] = mapd->z_min[0] + ((mapd->z_max[0] - mapd->z_min[0]) / (y_up - y_lo)) * (mapd->y_N_l[0] - y_lo);
      mapd->z_N_l[0] = max(min(mapd->z_N_l[0], mapd->z_max[0]), mapd->z_min[0]);  // clip to avoid numerical issues
      return ret;
    }
  } else {
    // computed centroid of rectangle [z_min, z_max]
    double z_bar[2] = {0};
    z_bar[0] = (1.0 / 2) * (mapd->z_min[0] + mapd->z_max[0]);
    z_bar[1] = (1.0 / 2) * (mapd->z_min[1] + mapd->z_max[1]);

    // (i) compute unique interior point y_bar of action set
    double y_bar[2] = {0};
    double e[2] = {0};
    ret = getIntPointActionSetWithBounds(mapd, n_inf, bounds, y_bar, e);
    if (ret == OPT_ERR)
      return ret;
    else if (ret == NO_INV) {
      // if the action set has no interior, we return the center point of the abstract action set
      mapd->z_N_l[0] = z_bar[0];
      mapd->z_N_l[1] = z_bar[1];
      return ret;
    }

    // (ii) compute delta_y and scaling factor alpha, if applicable
    double delta_y[2] = {0};
    delta_y[0] = mapd->y_N_l[0] - y_bar[0];
    delta_y[1] = mapd->y_N_l[1] - y_bar[1];
    if (max(fabs(delta_y[0]), fabs(delta_y[1])) < 1e-12) {
      mapd->z_N_l[0] = z_bar[0];
      mapd->z_N_l[1] = z_bar[1];
      return INV;
    }
    // normalize delta_y for better numerics in the coefficient matrix of the optimization problem
    double one_norm_delta_y = oneNorm(2, delta_y);
    if (one_norm_delta_y < 1e-4) {
      // Note: If we normalize delta_y, then alpha not necessarily >=1 anymore in the computation below
      //       (Finally, alpha >= 1 though, after de-normalizing with 1/one_norm_delta_y)
      delta_y[0] /= one_norm_delta_y;
      delta_y[1] /= one_norm_delta_y;
    } else
      one_norm_delta_y = 1.0;

    applyFeasibleBoundsToBeta(mapd, n_inf, bounds);
    // in addition, we need to update the lhs/rhs of the equality constraints...
    mapd->beta_comp.blower[mapd->beta_comp.m - 2] = y_bar[0] - e[0];
    mapd->beta_comp.blower[mapd->beta_comp.m - 1] = y_bar[1] - e[1];
    mapd->beta_comp.bupper[mapd->beta_comp.m - 2] = y_bar[0] - e[0];
    mapd->beta_comp.bupper[mapd->beta_comp.m - 1] = y_bar[1] - e[1];
    // ... and set -delta_y in the coefficient matrix
    int i_0 = (mapd->beta_comp.m - mapd->beta_comp.ms - 2) * mapd->beta_comp.n + N_l;
    mapd->beta_comp.A[i_0] = -delta_y[0];
    mapd->beta_comp.A[i_0 + mapd->beta_comp.n] = -delta_y[1];

    // activate lower bound in initial working set
    mapd->beta_comp.sense[N_l] = DAQP_ACTIVE + DAQP_LOWER;
    update_mask = DAQP_UPDATE_M + DAQP_UPDATE_sense + DAQP_UPDATE_v;
    daqp_update_ldp_cold(update_mask, &(mapd->beta_work), mapd->beta_work.qp);
    exit_flag = daqp_prox(&(mapd->beta_work));
    if (exit_flag != DAQP_EXIT_OPTIMAL) {
      daqp_deactivate_constraints(&(mapd->beta_work));
      reset_daqp_workspace(&(mapd->beta_work));
      return OPT_ERR;
    }
    double alpha = mapd->beta_work.x[N_l];
    alpha = max(1.0, alpha / one_norm_delta_y);  // correct scaling, note that alpha >= 1 necessarily

    // (iv) compute direction delta_z
    double delta_z[2] = {0};
    matVecMul(2, 2, (const double **)mapd->G_inv, delta_y, delta_z);

    // (v) compute scaling factor beta (> 0)
    double beta = DBL_MAX;
    for (int i = 0; i < 2; i++) {
      if (delta_z[i] > 0)
        beta = min(beta, (mapd->z_max[i] - z_bar[i]) / delta_z[i]);
      else if (delta_z[i] < 0)
        beta = min(beta, (mapd->z_min[i] - z_bar[i]) / delta_z[i]);
      // Note: The case where delta_z is the zero vector is excluded by the if clause above
    }

    // compute mapped point
    mapd->z_N_l[0] = z_bar[0] + (beta / alpha) * delta_z[0];  // always inside [z_min, z_max] if alpha >= 1
    mapd->z_N_l[1] = z_bar[1] + (beta / alpha) * delta_z[1];

    return INV;
  }
}

map_ret_code getActionSetPosWithBounds(map_data *mapd, const int n_inf, const map_runtime_bounds *bounds, double *y_lo,
                                       double *y_up) {
  // compute lower and upper bound of action set (position interval)

  const int N_l = mapd->N_l;
  int update_mask = DAQP_UPDATE_sense + DAQP_UPDATE_v;  // DAQP_UPDATE_v implies UPDATE_d
  int exit_flag;

  if (validateRuntimeBounds(bounds) != 0) return OPT_ERR;

  // (i) compute lower bound of action set
  // update lhs and rhs vectors of inequality constraints (dependent on x_0)
  applyFeasibleBoundsToPosSet(mapd, n_inf, bounds);
  daqp_update_ldp_cold(update_mask, &(mapd->pos_work), mapd->pos_work.qp);
  exit_flag = daqp_prox(&(mapd->pos_work));
  if (exit_flag != DAQP_EXIT_OPTIMAL) {
    daqp_deactivate_constraints(&(mapd->pos_work));
    reset_daqp_workspace(&(mapd->pos_work));
    return OPT_ERR;
  }
  *y_lo = bounds->e_full[0];
  for (int i = 0; i < N_l; i++) *y_lo += mapd->pos_set.f[i] * mapd->pos_work.x[i];

  // (ii) compute upper bound of action set
  for (int i = 0; i < N_l; i++) mapd->pos_set.f[i] = -mapd->pos_set.f[i];  // update objective gradient (to maximize)
  for (int iota = 0; iota < mapd->pos_set.m; iota++)
    mapd->pos_set.sense[iota] = 0;  // reset sense vector (to avoid warmstarting)
  daqp_update_ldp_cold(update_mask, &(mapd->pos_work), mapd->pos_work.qp);
  exit_flag = daqp_prox(&(mapd->pos_work));
  if (exit_flag != DAQP_EXIT_OPTIMAL) {
    daqp_deactivate_constraints(&(mapd->pos_work));
    reset_daqp_workspace(&(mapd->pos_work));
    return OPT_ERR;
  }
  *y_up = bounds->e_full[0];
  for (int i = 0; i < N_l; i++) {
    mapd->pos_set.f[i] = -mapd->pos_set.f[i];  // reverse sign of objective gradient first
    *y_up += mapd->pos_set.f[i] * mapd->pos_work.x[i];
  }

  // (iii) check correctness
  if (*y_lo > (*y_up + mapd->pos_work.settings->primal_tol)) {
    // something went wrong numerically in the optimization
    return OPT_ERR;
  } else if (*y_lo > (*y_up - mapd->pos_work.settings->primal_tol)) {
    // we collapse the interval to a singleton (no inverse exists in this case)
    *y_lo = 0.5 * (*y_lo) + 0.5 * (*y_up);
    *y_up = *y_lo;
    return NO_INV;
  } else
    return INV;
}

map_ret_code getIntPointActionSetWithBounds(map_data *mapd, const int n_inf, const map_runtime_bounds *bounds,
                                            double y_bar[2], double e[2]) {
  const int N_l = mapd->N_l;
  int update_mask = DAQP_UPDATE_sense + DAQP_UPDATE_v;
  int exit_flag;

  if (validateRuntimeBounds(bounds) != 0) return OPT_ERR;

  e[0] = bounds->e_full[0];
  e[1] = bounds->e_full[1];

  // (i) compute Chebyshev center of feasible input set
  applyFeasibleBoundsToChebyshev(mapd, n_inf, bounds);
  daqp_update_ldp_cold(update_mask, &(mapd->chebyshev_work), mapd->chebyshev_work.qp);
  exit_flag = daqp_prox(&(mapd->chebyshev_work));
  if (exit_flag != DAQP_EXIT_OPTIMAL) {
    daqp_deactivate_constraints(&(mapd->chebyshev_work));
    reset_daqp_workspace(&(mapd->chebyshev_work));
    return OPT_ERR;
  }

  // (ii) if the radius of the Chebyshev ball is below the primal tolerance of the solver,
  // we solve for the (scaled inf-norm) Chebyshev ball in the action set _directly_ to
  // see if the action set really has nonempty interior (this is a certificate then)
  double r = (double)mapd->chebyshev_work.x[N_l];
  if (r < mapd->chebyshev_work.settings->primal_tol) {
    // update x_0-dependent lhs/rhs vectors of inequalities and solve problem
    double e_tilde[2] = {0};
    matVecMul(2, 2, (const double **)mapd->G_inv, e, e_tilde);  // take scaling of inf-norm ball into account
    applyFeasibleBoundsToDirectChebyshev(mapd, n_inf, bounds->lower, bounds->upper, e_tilde);
    daqp_update_ldp_cold(update_mask, &(mapd->direct_work), mapd->direct_work.qp);
    exit_flag = daqp_prox(&(mapd->direct_work));
    if (exit_flag != DAQP_EXIT_OPTIMAL) {
      daqp_deactivate_constraints(&(mapd->direct_work));
      reset_daqp_workspace(&(mapd->direct_work));
      return OPT_ERR;
    }

    // compute the unique minimum-norm Chebyshev center
    double r_actset = (double)mapd->direct_work.x[0];
    for (int i = 0; i < mapd->direct_unique.m; i++) {
      // set lhs/rhs vectors of constraints
      mapd->direct_unique.blower[i] = mapd->direct_chebyshev.blower[i];
      mapd->direct_unique.bupper[i] = mapd->direct_chebyshev.bupper[i];
    }
    // fix radius
    mapd->direct_unique.blower[0] = r_actset;
    mapd->direct_unique.bupper[0] = r_actset;

    // update problem and solve
    daqp_update_ldp_cold(update_mask, &(mapd->direct_unique_work), mapd->direct_unique_work.qp);
    exit_flag = daqp_prox(&(mapd->direct_unique_work));  // solve non-strongly convex QP
    if (exit_flag != DAQP_EXIT_OPTIMAL) {
      daqp_deactivate_constraints(&(mapd->direct_unique_work));
      reset_daqp_workspace(&(mapd->direct_unique_work));
      return OPT_ERR;
    }

    // based on computed radius determine further steps
    r_actset = (double)mapd->direct_unique_work.x[0];
    y_bar[0] = (double)mapd->direct_unique_work.x[1];
    y_bar[1] = (double)mapd->direct_unique_work.x[2];
    if (r_actset < mapd->direct_unique_work.settings->primal_tol) {
      // if the radius is below the primal tolerance of the solver, we declare the action set
      // as having no interior, which means that we cannot continue with the mapping algorithm
      // but return the center of the Chebyshev ball as a result. Note that the inverse does not
      // exist in this case. Also note that this means that either the 2D action set is a line or a point.
      return NO_INV;
    } else
      return INV;
  } else {
    // compute interior point of 2D action set based on affine transformation, but before
    // compute the unique minimum-norm Chebyshev center
    for (int i = 0; i < mapd->chebyshev_unique.m; i++) {
      // set lhs/rhs vectors of constraints
      mapd->chebyshev_unique.blower[i] = mapd->chebyshev.blower[i];
      mapd->chebyshev_unique.bupper[i] = mapd->chebyshev.bupper[i];
    }
    // fix radius
    mapd->chebyshev_unique.blower[mapd->chebyshev_unique.m - 1] = r;
    mapd->chebyshev_unique.bupper[mapd->chebyshev_unique.m - 1] = r;

    // warmstart QP from LP solution
    // for (int i = 0; i < mapd->chebyshev_unique.m; i++)
    //     mapd->chebyshev_unique.sense[i] = 0; // reset sense before setting initial working set

    // // try via Lagrange multipliers of LP (Note: Not effective)
    // for (int i = 0; i < mapd->chebyshev_work.n_active; i++)
    // {
    //     if (fabs(mapd->chebyshev_work.lam_star[i]) > mapd->chebyshev_work.settings->dual_tol) // otherwise sign
    //     cannot be determined properly
    //     {
    //         int ind = mapd->chebyshev_work.WS[i];
    //         if ((mapd->chebyshev_work.lam_star[i]) >= 0)
    //             mapd->chebyshev_unique.sense[ind] = DAQP_ACTIVE; // active upper bound in initial working set
    //         else
    //             mapd->chebyshev_unique.sense[ind] = DAQP_ACTIVE + DAQP_LOWER; // active lower bound in initial
    //             working set
    //     }
    // }
    // mapd->chebyshev_unique.sense[mapd->chebyshev_unique.m - 1] = DAQP_ACTIVE + DAQP_IMMUTABLE; // fixing radius to
    // given value

    // // try via feasibility of primal solution (Note: Better, but no clear advantage over coldstart)
    // double val = 0;
    // int added = 0;
    // for (int i = 0; i < mapd->chebyshev_unique.m; i++)
    // {
    //     if (added == mapd->chebyshev_unique.n - 1) // reasoning: Since there is an additional equality constraint, we
    //     add only n-1 constraints
    //         break;

    //     val = scalarProd(mapd->chebyshev_unique.n, mapd->chebyshev_unique.A + i * mapd->chebyshev_unique.n,
    //     mapd->chebyshev_work.x); if (val >= (mapd->chebyshev_unique.bupper[i] -
    //     mapd->unique_work.settings->primal_tol))
    //     {
    //         mapd->chebyshev_unique.sense[i] = DAQP_ACTIVE; // active upper bound in initial working set
    //         added++;
    //     }
    //     else if (val <= (mapd->chebyshev_unique.blower[i] + mapd->unique_work.settings->primal_tol))
    //     {
    //         mapd->chebyshev_unique.sense[i] = DAQP_ACTIVE + DAQP_LOWER; // active lower bound in initial working set
    //         added++;
    //     }
    // }
    // mapd->chebyshev_unique.sense[mapd->chebyshev_unique.m - 1] = DAQP_ACTIVE + DAQP_IMMUTABLE; // fixing radius to
    // given value

    // update problem and solve
    daqp_update_ldp(update_mask, &(mapd->unique_work), mapd->unique_work.qp);
    exit_flag = daqp_ldp_retry(update_mask, &(mapd->unique_work), mapd->unique_work.qp);  // solve strongly convex QP
    if (exit_flag != DAQP_EXIT_OPTIMAL) {
      daqp_deactivate_constraints(&(mapd->unique_work));
      reset_daqp_workspace(&(mapd->unique_work));
      return OPT_ERR;
    }

    // get minimum-norm center point and compute interior point y_bar of action set
    ldp2qp_solution(&(mapd->unique_work));  // convert solution
    double *U_bar = (double *)mapd->unique_work.x;
    matVecMul(2, N_l, (const double **)mapd->E, U_bar, y_bar);
    vecVecDiff(2, 1.0, y_bar, -1.0, e, y_bar);
    return INV;
  }
}

int SetupOptProbs1D(map_data *mapd, const int n_inf, const double *A_mpc, const double **E_full,
                    const algo_params *algo_params) {
  return setupPosSet(mapd, n_inf, A_mpc, E_full, algo_params);
}

int SetupOptProbs2D(map_data *mapd, const int n_inf, const double *A_mpc, const double **E,
                    const algo_params *chebyshev_alg_pars, const algo_params *direct_chebyshev_alg_pars,
                    const algo_params *beta_alg_pars) {
  // Confirm the map type is 2D
  if (mapd->map_type != POS_VEL) {
    return -1;  // Return an error code if the map type is not 2D
  }

  // (i) compute direction-mapping matrix G via SVD (only needed for POS_VEL mapping, but cheap to do anyway and
  // simplifies code structure)
  mapd->E = E;
  ComputeGMatrix(mapd, E);

  // (ii) set up Chebyshev ball LP
  int ret = setupChebyshev(mapd, n_inf, A_mpc, chebyshev_alg_pars);

  // (iii) set up QP to compute unique minimum norm center of Chebyshev ball
  ret = min(ret, setupChebyshevMinNormCenter(mapd, chebyshev_alg_pars));

  // (iv) set up Chebyshev ball LP (direct approach)
  ret = min(ret, setupDirectChebyshev(mapd, n_inf, A_mpc, E, direct_chebyshev_alg_pars));

  // (v) set up QP to compute unique minimum norm center of Chebyshev ball (direct approach)
  ret = min(ret, setupDirectChebyshevMinNormCenter(mapd, direct_chebyshev_alg_pars));

  // (vi) set up LP to compute beta
  ret = min(ret, setupBeta(mapd, n_inf, A_mpc, E, beta_alg_pars));

  return ret;
}

void ComputeGMatrix(map_data *mapd, const double **E) {
  // Compute direction-mapping matrix G via SVD Idea: Instead of doing an SVD, compute eigenvectors and eigenvalues of
  // matrix E*E' (2x2!).
  double AUX[2][2] = {0};
  double GG[2][2] = {0};
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) AUX[i][j] = scalarProd(mapd->N_l, E[i], E[j]);
  double lambda[2] = {0};
  lambda[0] = (AUX[0][0] + AUX[1][1] -
               sqrt(pow(AUX[0][0], 2) + 4 * AUX[0][1] * AUX[1][0] - 2 * AUX[0][0] * AUX[1][1] + pow(AUX[1][1], 2))) /
              2;
  lambda[1] = (AUX[0][0] + AUX[1][1] +
               sqrt(pow(AUX[0][0], 2) + 4 * AUX[0][1] * AUX[1][0] - 2 * AUX[0][0] * AUX[1][1] + pow(AUX[1][1], 2))) /
              2;
  GG[0][0] = -(-AUX[0][0] + AUX[1][1] +
               sqrt(pow(AUX[0][0], 2) + 4 * AUX[0][1] * AUX[1][0] - 2 * AUX[0][0] * AUX[1][1] + pow(AUX[1][1], 2))) /
             (2 * AUX[1][0]);
  GG[1][0] = 1.0;
  GG[0][1] = -(-AUX[0][0] + AUX[1][1] -
               sqrt(pow(AUX[0][0], 2) + 4 * AUX[0][1] * AUX[1][0] - 2 * AUX[0][0] * AUX[1][1] + pow(AUX[1][1], 2))) /
             (2 * AUX[1][0]);
  GG[1][1] = 1.0;

  // normalize the columns of G, so that G is orthonormal (this is matrix U in the SVD of matrix E)
  double length = 0;
  for (int j = 0; j < 2; j++) {
    length = sqrt(pow(GG[0][j], 2) + pow(GG[1][j], 2));
    for (int i = 0; i < 2; i++) GG[i][j] /= length;
  }
  // it remains to multiply the columns of G by the square root of the associated eigenvalue
  double frob_norm_G = 0;
  for (int j = 0; j < 2; j++)
    for (int i = 0; i < 2; i++) {
      GG[i][j] *= sqrt(lambda[j]);
      frob_norm_G += pow(GG[i][j], 2);
    }
  frob_norm_G = sqrt(frob_norm_G);

  // finally, change columns and signs such that the result is identical to the one of the SVD method used in Matlab
  // (needed for test_computestep validation)
  mapd->G[0][0] = -GG[0][1] / frob_norm_G;  // normalization by Frobenius norm so that radius computation in direct
                                            // Chebyshev approach is more meaningful
  mapd->G[1][0] = -GG[1][1] / frob_norm_G;
  mapd->G[0][1] = -GG[0][0] / frob_norm_G;
  mapd->G[1][1] = -GG[1][0] / frob_norm_G;

  // compute the inverse of G for the inverse mapping
  double det = mapd->G[0][0] * mapd->G[1][1] - mapd->G[0][1] * mapd->G[1][0];
  mapd->G_inv[0][0] = (1.0 / det) * mapd->G[1][1];
  mapd->G_inv[1][0] = (1.0 / det) * (-mapd->G[1][0]);
  mapd->G_inv[0][1] = (1.0 / det) * (-mapd->G[0][1]);
  mapd->G_inv[1][1] = (1.0 / det) * mapd->G[0][0];
}

int setupChebyshev(map_data *mapd, const int n_inf, const double *A_mpc, const algo_params *alg_pars) {
  // Set up LP to compute the Chebyshev ball for the set of feasible control inputs
  // Note: We assume the following variable ordering: (u_0, u_1, ..., u_{N_l-1}, r)

  // Note: The idea is to use the setup for the MPC problem, so, we assume that the MPC problem
  //       in mapd->mpc was set up before calling this function.

  // Structure of constraints (AA is MPC coefficient matrix, b_lower, b_upper lhs/rhs vectors from MPC)
  // -inf      <= [I       1     ]       <= b_upper_1
  // -inf      <= [AA  [|AA_i|_2]]       <= b_upper_2
  // b_lower_1 <= [I      -1     ] [U;r] <= inf
  // b_lower_2 <= [AA -[|AA_i|_2]]       <= inf
  // 0         <= [0       1     ]       <= inf

  DAQPProblem *model = &(mapd->chebyshev);
  DAQPWorkspace *work = &(mapd->chebyshev_work);
  const int N_l = mapd->N_l;
  const int dim = mapd->dim;

  // (i) define the size of the LP (see types.h in DAQP header files)
  model->n = N_l + 1;
  model->ms = 0;
  model->m = model->ms + 2 * (N_l + dim * (N_l - 1) + n_inf) + 1;

  // (ii) define objective
  model->H = NULL;
  model->f = malloc(model->n * sizeof(c_float));
  for (int l = 0; l < model->n; l++) model->f[l] = 0;
  model->f[model->n - 1] = -1;  // maximize radius

  // (iii) define bound vectors (both for constraints and variables)
  model->blower = malloc((model->m) * sizeof(c_float));
  model->bupper = malloc((model->m) * sizeof(c_float));
  for (int iota = 0; iota < model->m; iota++) {
    model->blower[iota] = -DAQP_INF;  // will be set online to a meaningful value
    model->bupper[iota] = DAQP_INF;   // will be set online to a meaningful value
  }
  model->blower[model->m - 1] = 0;  // r >= 0

  // (iv) define state constraints
  model->A = malloc((model->m - model->ms) * model->n * sizeof(c_float));
  for (int iota = 0; iota < ((model->m - model->ms) * model->n); iota++) model->A[iota] = 0;

  // use existing MPC formulation to fill in constraint matrix A
  for (int iota = 0; iota < N_l; iota++) {
    // set identity matrices
    model->A[iota * model->n + iota] = 1;
    model->A[(N_l + dim * (N_l - 1) + n_inf) * model->n + iota * model->n + iota] = 1;

    // set one vectors
    model->A[iota * model->n + N_l] = 1;
    model->A[(N_l + dim * (N_l - 1) + n_inf) * model->n + iota * model->n + N_l] = -1;
  }

  for (int iota = 0; iota < (dim * (N_l - 1)); iota++) {
    // set state constraint entries from MPC problem in coefficient matrix (until right before terminal pos/speed
    // constraints)
    double sq_sum = 0;
    for (int j = 0; j < N_l; j++) {
      model->A[N_l * model->n + iota * model->n + j] = A_mpc[iota * N_l + j];
      model->A[(N_l + dim * (N_l - 1) + n_inf) * model->n + N_l * model->n + iota * model->n + j] =
        A_mpc[iota * N_l + j];
      sq_sum += pow(A_mpc[iota * N_l + j], 2);
    }
    model->A[N_l * model->n + iota * model->n + N_l] = sqrt(sq_sum);
    model->A[(N_l + dim * (N_l - 1) + n_inf) * model->n + N_l * model->n + iota * model->n + N_l] = -sqrt(sq_sum);
  }
  for (int iota = 0; iota < n_inf; iota++) {
    // set terminal state constraint entries from MPC problem in coefficient matrix
    double sq_sum = 0;
    for (int j = 0; j < N_l; j++) {
      model->A[(N_l + (dim * (N_l - 1))) * model->n + iota * model->n + j] =
        A_mpc[(dim * (N_l - 1) + 2) * N_l + iota * N_l + j];
      model
        ->A[(N_l + dim * (N_l - 1) + n_inf) * model->n + (N_l + (dim * (N_l - 1))) * model->n + iota * model->n + j] =
        A_mpc[(dim * (N_l - 1) + 2) * N_l + iota * N_l + j];
      sq_sum += pow(A_mpc[(dim * (N_l - 1) + 2) * N_l + iota * N_l + j], 2);
    }
    model->A[(N_l + (dim * (N_l - 1))) * model->n + iota * model->n + N_l] = sqrt(sq_sum);
    model
      ->A[(N_l + dim * (N_l - 1) + n_inf) * model->n + (N_l + (dim * (N_l - 1))) * model->n + iota * model->n + N_l] =
      -sqrt(sq_sum);
  }
  model->A[(model->m - model->ms) * model->n - 1] = 1;  // r>=0 constraint

  // (v) define sense vector
  model->sense = malloc(model->m * sizeof(int));
  for (int iota = 0; iota < model->m; iota++) model->sense[iota] = 0;  // all constraints are inequality constraints

  // (vi) prepare for an efficient approach of using DAQP online
  work->settings = malloc(sizeof(DAQPSettings));  // if not set, we will get a seg fault later on
  daqp_default_settings(work->settings);
  work->settings->zero_tol = alg_pars->zero_tol;
  work->settings->primal_tol = alg_pars->primal_tol;
  work->settings->dual_tol = alg_pars->dual_tol;
  work->settings->eps_prox = alg_pars->eps_prox;
  work->settings->eta_prox = alg_pars->eta_prox;
  work->iterations = 0;  // if not set, then warm-starting might not work
  int ret = setup_daqp(model, work, NULL);
  if (ret != 1)
    return -1;
  else
    return 0;
}

int setupChebyshevMinNormCenter(map_data *mapd, const algo_params *alg_pars) {
  // Set up QP to compute minimum-norm center of previously computed Chebyshev ball.
  // Note: Assumes that setupChebyshev() was called prior to calling this function

  DAQPProblem *model = &(mapd->chebyshev_unique);
  DAQPWorkspace *work = &(mapd->unique_work);
  DAQPProblem *model_src = &(mapd->chebyshev);

  // idea: simply copy the constraints of the LP model mapd->chebyshev
  // (i) define the size of the QP
  model->n = model_src->n;
  model->ms = model_src->ms;
  model->m = model_src->m;

  // (ii) define QP objective
  model->H = malloc(model->n * model->n * sizeof(c_float));  // Note: row-major order
  for (int iota = 0; iota < (model->n * model->n); iota++) model->H[iota] = 0;
  for (int l = 0; l < model->n; l++) model->H[l * model->n + l] = 1;  // set diagonal entries of Hessian to 1
  model->f = malloc(model->n * sizeof(c_float));
  for (int l = 0; l < model->n; l++) model->f[l] = 0;

  // (iii) define bound vectors (both for constraints and variables)
  model->blower = malloc((model->m) * sizeof(c_float));
  model->bupper = malloc((model->m) * sizeof(c_float));
  for (int iota = 0; iota < model->m; iota++) {
    model->blower[iota] = -DAQP_INF;  // will be set online to a meaningful value
    model->bupper[iota] = DAQP_INF;   // will be set online to a meaningful value
  }
  model->blower[model->m - 1] = 0;  // r >= 0

  // (iv) define state constraints
  model->A = malloc((model->m - model->ms) * model->n * sizeof(c_float));
  for (int iota = 0; iota < ((model->m - model->ms) * model->n); iota++) model->A[iota] = model_src->A[iota];

  // (v) define sense vector
  model->sense = malloc(model->m * sizeof(int));
  for (int iota = 0; iota < model->m; iota++)
    model->sense[iota] = 0;  // all constraints are inequality constraints, except for the last one (fixes radius)
  model->sense[model->m - 1] = DAQP_ACTIVE + DAQP_IMMUTABLE;

  // (vi) prepare for an efficient approach of using DAQP online
  work->settings = malloc(sizeof(DAQPSettings));  // if not set, we will get a seg fault later on
  daqp_default_settings(work->settings);
  work->settings->zero_tol = alg_pars->zero_tol;
  work->settings->primal_tol = alg_pars->primal_tol;
  work->settings->dual_tol = alg_pars->dual_tol;
  work->iterations = 0;  // if not set, then warm-starting might not work
  int ret = setup_daqp(model, work, NULL);
  if (ret != 1)
    return -1;
  else
    return 0;
}

int setupDirectChebyshev(map_data *mapd, const int n_inf, const double *A_mpc, const double **E,
                         const algo_params *alg_pars) {
  // Set up LP to compute the Chebyshev ball for the action set directly (specifically, the scaled inf-norm ball)
  // Note: We assume the following variable ordering: (r, y_c, U_1, U_2, U_3, U_4), where
  //       r \in \Reals_+, y_c \in \Reals{2}, U_i \in \Reals{N_l}.

  // Note: The idea is to use the setup for the MPC problem, so, we assume that the MPC problem
  //       in mapd->mpc was set up before calling this function.

  // See notes from Oct 10 2024 (and references therein for derivation)

  DAQPProblem *model = &(mapd->direct_chebyshev);
  DAQPWorkspace *work = &(mapd->direct_work);
  const int N_l = mapd->N_l;
  const int dim = mapd->dim;
  const double w_11[2] = {1, 0};
  const double w_21[2] = {0, 1};
  const double w_31[2] = {-1, 0};
  const double w_41[2] = {0, -1};
  const double w_12[2] = {0, 1};
  const double w_22[2] = {-1, 0};
  const double w_32[2] = {0, -1};
  const double w_42[2] = {1, 0};

  // compute scaled directions
  double w_11_tilde[2] = {0};
  double w_21_tilde[2] = {0};
  double w_31_tilde[2] = {0};
  double w_41_tilde[2] = {0};
  double w_12_tilde[2] = {0};
  double w_22_tilde[2] = {0};
  double w_32_tilde[2] = {0};
  double w_42_tilde[2] = {0};
  vecMatDPMul(2, 2, w_11, (const double **)mapd->G_inv, w_11_tilde);
  vecMatDPMul(2, 2, w_21, (const double **)mapd->G_inv, w_21_tilde);
  vecMatDPMul(2, 2, w_31, (const double **)mapd->G_inv, w_31_tilde);
  vecMatDPMul(2, 2, w_41, (const double **)mapd->G_inv, w_41_tilde);
  vecMatDPMul(2, 2, w_12, (const double **)mapd->G_inv, w_12_tilde);
  vecMatDPMul(2, 2, w_22, (const double **)mapd->G_inv, w_22_tilde);
  vecMatDPMul(2, 2, w_32, (const double **)mapd->G_inv, w_32_tilde);
  vecMatDPMul(2, 2, w_42, (const double **)mapd->G_inv, w_42_tilde);

  // (i) define the size of the LP (see types.h in DAQP header files)
  const int n_mpc_constr = N_l + dim * (N_l - 1) + n_inf;
  model->n = 1 + 2 + 4 * N_l;
  model->ms = 0;
  model->m = model->ms + 1 + 4 * 2 + 4 * n_mpc_constr;

  // (ii) define objective
  model->H = NULL;
  model->f = malloc(model->n * sizeof(c_float));
  for (int l = 0; l < model->n; l++) model->f[l] = 0;
  model->f[0] = -1;  // maximize radius

  // (iii) define bound vectors (both for constraints and variables)
  model->blower = malloc((model->m) * sizeof(c_float));
  model->bupper = malloc((model->m) * sizeof(c_float));
  for (int iota = 0; iota < model->m; iota++) {
    model->blower[iota] = -DAQP_INF;  // will be set online to a meaningful value
    model->bupper[iota] = DAQP_INF;   // will be set online to a meaningful value
  }
  model->blower[0] = 0;  // r >= 0

  // (iv) define state constraints
  model->A = malloc((model->m - model->ms) * model->n * sizeof(c_float));
  for (int iota = 0; iota < ((model->m - model->ms) * model->n); iota++) model->A[iota] = 0;

  // manually define the specific first 1 + 4 * 2 inequality constraints
  c_float aux[N_l];
  // constr #1
  int i = 0;
  model->A[i] = 1;
  // constr #2
  i = model->n;
  model->A[i++] = 1;
  model->A[i++] = -w_11_tilde[0];
  model->A[i++] = -w_11_tilde[1];
  vecMatDPMul(2, N_l, w_11_tilde, E, aux);
  for (int iota = 0; iota < N_l; iota++) model->A[i++] = aux[iota];
  // constr #3
  i = 2 * model->n;
  model->A[i++] = 1;
  model->A[i++] = -w_21_tilde[0];
  model->A[i++] = -w_21_tilde[1];
  i += N_l;
  vecMatDPMul(2, N_l, w_21_tilde, E, aux);
  for (int iota = 0; iota < N_l; iota++) model->A[i++] = aux[iota];
  // constr #4
  i = 3 * model->n;
  model->A[i++] = 1;
  model->A[i++] = -w_31_tilde[0];
  model->A[i++] = -w_31_tilde[1];
  i += 2 * N_l;
  vecMatDPMul(2, N_l, w_31_tilde, E, aux);
  for (int iota = 0; iota < N_l; iota++) model->A[i++] = aux[iota];
  // constr #5
  i = 4 * model->n;
  model->A[i++] = 1;
  model->A[i++] = -w_41_tilde[0];
  model->A[i++] = -w_41_tilde[1];
  i += 3 * N_l;
  vecMatDPMul(2, N_l, w_41_tilde, E, aux);
  for (int iota = 0; iota < N_l; iota++) model->A[i++] = aux[iota];
  // constr #6
  i = 5 * model->n;
  model->A[i++] = 1;
  model->A[i++] = -w_12_tilde[0];
  model->A[i++] = -w_12_tilde[1];
  vecMatDPMul(2, N_l, w_12_tilde, E, aux);
  for (int iota = 0; iota < N_l; iota++) model->A[i++] = aux[iota];
  // constr #7
  i = 6 * model->n;
  model->A[i++] = 1;
  model->A[i++] = -w_22_tilde[0];
  model->A[i++] = -w_22_tilde[1];
  i += N_l;
  vecMatDPMul(2, N_l, w_22_tilde, E, aux);
  for (int iota = 0; iota < N_l; iota++) model->A[i++] = aux[iota];
  // constr #8
  i = 7 * model->n;
  model->A[i++] = 1;
  model->A[i++] = -w_32_tilde[0];
  model->A[i++] = -w_32_tilde[1];
  i += 2 * N_l;
  vecMatDPMul(2, N_l, w_32_tilde, E, aux);
  for (int iota = 0; iota < N_l; iota++) model->A[i++] = aux[iota];
  // constr #9
  i = 8 * model->n;
  model->A[i++] = 1;
  model->A[i++] = -w_42_tilde[0];
  model->A[i++] = -w_42_tilde[1];
  i += 3 * N_l;
  vecMatDPMul(2, N_l, w_42_tilde, E, aux);
  for (int iota = 0; iota < N_l; iota++) model->A[i++] = aux[iota];

  // use existing MPC formulation to fill in the rest of constraint matrix A
  int offset = 0;
  for (int k = 0; k < 4; k++) {
    offset = (9 + k * n_mpc_constr) * model->n + 1 + 2 + k * N_l;
    for (int iota = 0; iota < N_l; iota++)
      model->A[offset + iota * model->n + iota] = 1;  // set identity matrix for bound constraints on U

    offset += N_l * model->n;
    for (int iota = 0; iota < (dim * (N_l - 1)); iota++)
      for (int j = 0; j < N_l; j++) model->A[offset + iota * model->n + j] = A_mpc[iota * N_l + j];

    offset += (dim * (N_l - 1)) * model->n;
    for (int iota = 0; iota < n_inf; iota++)
      for (int j = 0; j < N_l; j++)
        model->A[offset + iota * model->n + j] = A_mpc[(dim * (N_l - 1) + 2) * N_l + iota * N_l + j];
  }

  // (v) define sense vector
  model->sense = malloc(model->m * sizeof(int));
  for (int iota = 0; iota < model->m; iota++) model->sense[iota] = 0;  // all constraints are inequality constraints

  // (vi) prepare for an efficient approach of using DAQP online
  work->settings = malloc(sizeof(DAQPSettings));  // if not set, we will get a seg fault later on
  daqp_default_settings(work->settings);
  work->settings->zero_tol = alg_pars->zero_tol;
  work->settings->primal_tol = alg_pars->primal_tol;
  work->settings->dual_tol = alg_pars->dual_tol;
  work->settings->eps_prox = alg_pars->eps_prox;
  work->settings->eta_prox = alg_pars->eta_prox;
  work->iterations = 0;  // if not set, then warm-starting might not work
  int ret = setup_daqp(model, work, NULL);
  if (ret != 1)
    return -1;
  else
    return 0;
}

int setupDirectChebyshevMinNormCenter(map_data *mapd, const algo_params *alg_pars) {
  // Set up QP to compute minimum-norm center of previously computed Chebyshev ball (direct approach).
  // Note: Assumes that setupDirectChebyshev() was called prior to calling this function

  DAQPProblem *model = &(mapd->direct_unique);
  DAQPWorkspace *work = &(mapd->direct_unique_work);
  DAQPProblem *model_src = &(mapd->direct_chebyshev);

  // idea: simply copy the constraints of the LP model mapd->direct_chebyshev
  // (i) define the size of the QP
  model->n = model_src->n;
  model->ms = model_src->ms;
  model->m = model_src->m;

  // (ii) define QP objective
  model->H = malloc(model->n * model->n * sizeof(c_float));  // Note: row-major order
  for (int iota = 0; iota < (model->n * model->n); iota++) model->H[iota] = 0;
  for (int l = 0; l < 3; l++)
    model->H[l * model->n + l] = 1;  // quadratic penalty only for radius and center of action set
  model->f = malloc(model->n * sizeof(c_float));
  for (int l = 0; l < model->n; l++) model->f[l] = 0;

  // (iii) define bound vectors (both for constraints and variables)
  model->blower = malloc((model->m) * sizeof(c_float));
  model->bupper = malloc((model->m) * sizeof(c_float));
  for (int iota = 0; iota < model->m; iota++) {
    model->blower[iota] = -DAQP_INF;  // will be set online to a meaningful value
    model->bupper[iota] = DAQP_INF;   // will be set online to a meaningful value
  }
  model->blower[0] = 0;  // r >= 0

  // (iv) define state constraints
  model->A = malloc((model->m - model->ms) * model->n * sizeof(c_float));
  for (int iota = 0; iota < ((model->m - model->ms) * model->n); iota++) model->A[iota] = model_src->A[iota];

  // (v) define sense vector
  model->sense = malloc(model->m * sizeof(int));
  for (int iota = 0; iota < model->m; iota++)
    model->sense[iota] = 0;  // all constraints are inequality constraints, except for the last one (fixes radius)
  model->sense[0] = DAQP_ACTIVE + DAQP_IMMUTABLE;

  // (vi) prepare for an efficient approach of using DAQP online
  work->settings = malloc(sizeof(DAQPSettings));  // if not set, we will get a seg fault later on
  daqp_default_settings(work->settings);
  work->settings->zero_tol = alg_pars->zero_tol;
  work->settings->primal_tol = alg_pars->primal_tol;
  work->settings->dual_tol = alg_pars->dual_tol;
  work->settings->eps_prox = 1e-5;  // empirically determined and hard-coded for scaletype = NORMALIZE
  work->settings->eta_prox = alg_pars->eta_prox;
  work->iterations = 0;  // if not set, then warm-starting might not work
  int ret = setup_daqp(model, work, NULL);
  if (ret != 1)
    return -1;
  else
    return 0;
}

int setupBeta(map_data *mapd, const int n_inf, const double *A_mpc, const double **E, const algo_params *alg_pars) {
  // Set up LP to compute the scaling factor beta in the mapping algorithm
  // Note: We assume the following variable ordering: (u_0, u_1, ..., u_{N_l-1}, beta)

  // Note: The idea is to use the setup for the MPC problem, so, we assume that the MPC problem
  //       in mapd->mpc was set up before calling this function.

  // Structure of constraints (AA is MPC coefficient matrix, b_lower, b_upper lhs/rhs vectors from MPC)
  // b_lower_1 <= [I         ]           <= b_upper_1
  //   0       <= [         1]           <= inf
  // b_lower_2 <= [AA       0] [z; beta] <= b_upper_2
  // y_bar - e <= [E -delta_y]           <= y_bar - e

  DAQPProblem *model = &(mapd->beta_comp);
  DAQPWorkspace *work = &(mapd->beta_work);
  const int N_l = mapd->N_l;
  const int dim = mapd->dim;

  // (i) define the size of the LP (see types.h in DAQP header files)
  model->n = N_l + 1;
  model->ms = N_l + 1;
  model->m = model->ms + dim * (N_l - 1) + n_inf + 2;

  // (ii) define objective
  model->H = NULL;
  model->f = malloc(model->n * sizeof(c_float));
  for (int l = 0; l < model->n; l++) model->f[l] = 0;
  model->f[model->n - 1] = -1;  // maximize beta

  // (iii) define bound vectors (both for constraints and variables)
  model->blower = malloc((model->m) * sizeof(c_float));
  model->bupper = malloc((model->m) * sizeof(c_float));
  for (int iota = 0; iota < model->m; iota++) {
    model->blower[iota] = 0;         // will be set online to a meaningful value
    model->bupper[iota] = DAQP_INF;  // will be set online to a meaningful value
  }
  // set equality constraints
  model->blower[model->m - 2] = 0;
  model->blower[model->m - 1] = 0;
  model->bupper[model->m - 2] = 0;
  model->bupper[model->m - 1] = 0;

  // (iv) define state constraints
  model->A = malloc((model->m - model->ms) * model->n * sizeof(c_float));
  for (int iota = 0; iota < ((model->m - model->ms) * model->n); iota++) model->A[iota] = 0;

  // use existing MPC formulation to fill in constraint matrix A
  for (int iota = 0; iota < (dim * (N_l - 1)); iota++) {
    // set state constraint entries from MPC problem in coefficient matrix (until right before terminal pos/speed
    // constraints)
    for (int j = 0; j < N_l; j++) model->A[iota * model->n + j] = A_mpc[iota * N_l + j];
  }
  for (int iota = 0; iota < n_inf; iota++) {
    // set terminal state constraint entries from MPC problem in coefficient matrix
    for (int j = 0; j < N_l; j++)
      model->A[(dim * (N_l - 1)) * model->n + iota * model->n + j] =
        A_mpc[(dim * (N_l - 1) + 2) * N_l + iota * N_l + j];
  }
  for (int iota = 0; iota < 2; iota++) {
    for (int j = 0; j < N_l; j++) model->A[((dim * (N_l - 1)) + n_inf) * model->n + iota * model->n + j] = E[iota][j];
  }

  // (v) define sense vector
  model->sense = malloc(model->m * sizeof(int));
  for (int iota = 0; iota < model->m; iota++)
    model->sense[iota] = 0;        // all constraints are inequality constraints, except ...
  model->sense[model->m - 2] = 5;  // ... for the last two, which are equality constraints
  model->sense[model->m - 1] = 5;

  // (vi) prepare for an efficient approach of using DAQP online
  work->settings = malloc(sizeof(DAQPSettings));  // if not set, we will get a seg fault later on
  daqp_default_settings(work->settings);
  work->settings->zero_tol = alg_pars->zero_tol;
  work->settings->primal_tol = alg_pars->primal_tol;
  work->settings->dual_tol = alg_pars->dual_tol;
  work->settings->eps_prox = alg_pars->eps_prox;
  work->settings->eta_prox = alg_pars->eta_prox;
  work->iterations = 0;  // if not set, then warm-starting might not work
  int ret = setup_daqp(model, work, NULL);
  if (ret != 1)
    return -1;
  else
    return 0;
}

int setupPosSet(map_data *mapd, const int n_inf, const double *A_mpc, const double **E_full,
                const algo_params *alg_pars) {
  // Set up LP problem to compute the 1D action set (position only)

  DAQPProblem *model = &(mapd->pos_set);
  DAQPWorkspace *work = &(mapd->pos_work);
  const int N_l = mapd->N_l;
  const int dim = mapd->dim;

  // (i) define the size of the LP (see types.h in DAQP header files)
  model->n = N_l;
  model->ms = N_l;
  model->m = model->ms + dim * (N_l - 1) + n_inf;

  // (ii) define objective
  model->H = NULL;
  model->f = malloc(N_l * sizeof(c_float));
  for (int l = 0; l < N_l; l++)
    model->f[l] =
      E_full[0]
            [l];  // f'*U gives the position after N_l steps due to the input U (without considering the initial state)

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

  // use existing MPC formulation to fill in constraint matrix A
  for (int iota = 0; iota < (dim * (N_l - 1)); iota++) {
    // set state constraint entries from MPC problem in coefficient matrix (until right before terminal pos/speed
    // constraints)
    for (int j = 0; j < N_l; j++) model->A[iota * N_l + j] = A_mpc[iota * N_l + j];
  }
  for (int iota = 0; iota < n_inf; iota++) {
    // set terminal state constraint entries from MPC problem in coefficient matrix
    for (int j = 0; j < N_l; j++)
      model->A[(dim * (N_l - 1)) * N_l + iota * N_l + j] =
        A_mpc[(dim * (N_l - 1) + 2) * N_l + iota * N_l + j];  // +2 due to pos/speed constraints in MPC formulation
  }

  // (v) define sense vector
  model->sense = malloc(model->m * sizeof(int));
  for (int iota = 0; iota < model->m; iota++) model->sense[iota] = 0;  // all constraints are inequality constraints

  // (vi) prepare for an efficient approach of using DAQP online
  work->settings = malloc(sizeof(DAQPSettings));  // if not set, we will get a seg fault later on
  daqp_default_settings(work->settings);
  work->settings->zero_tol = alg_pars->zero_tol;
  work->settings->primal_tol = alg_pars->primal_tol;
  work->settings->dual_tol = alg_pars->dual_tol;
  work->settings->eps_prox = alg_pars->eps_prox;
  work->settings->eta_prox = alg_pars->eta_prox;
  work->iterations = 0;  // if not set, then warm-starting might not work
  int ret = setup_daqp(model, work, NULL);
  if (ret != 1)
    return -1;
  else
    return 0;
}
