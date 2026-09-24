#include "daqp_lp.h"

#include "daqp/utils.h"

static int projgradient_step(DAQPWorkspace *work);
static void compute_CSP_lp(DAQPWorkspace *work);
static int compute_projgrad(DAQPWorkspace *work);
static int remove_constraint_lp(DAQPWorkspace *work);

int daqp_update_ldp_cold(int update_mask, DAQPWorkspace *work, DAQPProblem *qp) {
  // for the LPs, whose optimum is a face: warm-started, the vertex returned depended on the
  // previous call. The reset must precede the update, which re-activates the immutable constraints.
  daqp_deactivate_constraints(work);
  reset_daqp_workspace(work);
  return daqp_update_ldp(update_mask, work, qp);
}

int daqp_ldp_retry(int update_mask, DAQPWorkspace *work, DAQPProblem *qp) {
  // for the strictly convex problems, whose optimum is unique so a warm start only saves
  // iterations. A stale active set can still report infeasible, and on a degenerate problem the
  // MPC's tolerances (1e-11, far below DAQP's defaults) make its active set cycle even from cold,
  // so the retry is cold and at DAQP's default tolerances with a wider cycle guard.
  int exit_flag = daqp_ldp(work);
  if (exit_flag != DAQP_EXIT_OPTIMAL) {
    DAQPSettings kept = *work->settings;
    work->settings->primal_tol = DAQP_DEFAULT_PRIM_TOL;
    work->settings->zero_tol = DAQP_DEFAULT_ZERO_TOL;
    work->settings->cycle_tol = 10 * DAQP_DEFAULT_CYCLE_TOL;
    daqp_update_ldp_cold(update_mask, work, qp);
    exit_flag = daqp_ldp(work);
    *work->settings = kept;
  }
  return exit_flag;
}

int daqp_lp(DAQPWorkspace *work) {
  int i, total_iter = 0;
  // const int nx = work->n;
  int exitflag;
  // c_float *swp_ptr;

  // Initial phase to get primal feasible solution
  exitflag = daqp_ldp(work);

  total_iter += work->iterations;
  if (exitflag < 0)
    return exitflag;  // Could not solve LDP -> return
  else
    ldp2qp_solution(work);  // Get qp solution

  // printf("Initial phase done\n");
  // printf("xinit: ");
  // for(i = 0; i < work->n; i++){
  //     printf(" %f",work->x[i]);
  // }
  // printf("\n");

  // Do a quick swap such that x and u are seperate
  for (i = 0; i < work->n; i++) work->xold[i] = work->x[i];
  work->x = work->xold;

  // Cleanup workspace
  work->reuse_ind = 0;
  // deactivate_constraints(work);
  // reset_daqp_workspace(work);
  // activate_constraints(work);

  // Start primal phase
  // printf("Starting primal phase \n");
  while (total_iter++ < work->settings->iter_limit) {
    // printf(" WS: { ");
    // for (i = 0; i < work->n_active; i++)
    //     printf("%d ",work->WS[i]);
    // printf("}\n");
    //  Solve KKT system
    // printf("Computing CSP \n");
    compute_CSP_lp(work);
    // printf("compute projgrad\n");
    if (work->n_active == work->n || compute_projgrad(work) != 1) {  // |p| = 0
                                                                     // Check lambda >= 0
      // printf("removing constraint\n");
      if (remove_constraint_lp(work) == DAQP_EMPTY_IND) {
        exitflag = DAQP_EXIT_OPTIMAL;
        break;
      }
    } else {  // |p| =/= 0 -> take gradient step
      // printf("x: ");
      // for(i = 0; i < work->n; i++){
      //     printf(" %f",work->x[i]);
      // }
      // printf("\n p: ");
      // for(i = 0; i < work->n; i++){
      //     printf(" %f",work->u[i]);
      // }
      // printf("\n");
      if (projgradient_step(work) == DAQP_EMPTY_IND) {
        exitflag = DAQP_EXIT_UNBOUNDED;
        break;
      }
    }
  }
  if (total_iter >= work->settings->iter_limit) exitflag = DAQP_EXIT_ITERLIMIT;
  work->iterations = total_iter;
  for (i = 0; i < work->n_active; i++) work->lam_star[i] *= work->scaling[work->WS[i]];
  return exitflag;
}

// Gradient step
static int projgradient_step(DAQPWorkspace *work) {
  int j, k, disp, add_ind = DAQP_EMPTY_IND;
  const int nx = work->n;
  const int m = work->m;
  const int ms = work->ms;
  c_float Ax, delta_s, min_alpha = DAQP_INF;
  int isupper;
  // Find constraint j to add: j =  argmin_j s_j
  // Simple bounds
  for (j = 0, disp = 0; j < ms; j++) {
    if (work->sense[j] & (DAQP_ACTIVE + DAQP_IMMUTABLE)) continue;
    delta_s = work->u[j];
    // printf("#%d| ds:%f | bu:%f | bl:%f | xj: %f\n",j,delta_s,work->qp->bupper[j],work->qp->blower[j],work->x[j]);
    // printf("#%d | alphau:%f, alphal:%f\n",j,(work->qp->bupper[j] - work->x[j])/delta_s,(work->qp->blower[j] -
    // work->x[j])/delta_s);

    if (delta_s > work->settings->zero_tol &&  // Feasible descent direction
        work->qp->bupper[j] < DAQP_INF &&      // Not single-sided
        work->qp->bupper[j] - work->x[j] < min_alpha * delta_s) {
      add_ind = j;
      isupper = 1;
      min_alpha = (work->qp->bupper[j] - work->x[j]) / delta_s;
    } else if (delta_s < -work->settings->zero_tol &&  // Feasible descent direction
               work->qp->blower[j] > -DAQP_INF &&      // Not single-sided
               work->qp->blower[j] - work->x[j] > min_alpha * delta_s) {
      add_ind = j;
      isupper = 0;
      min_alpha = (work->qp->blower[j] - work->x[j]) / delta_s;
    }
  }
  // General bounds
  for (j = ms, disp = 0; j < m; j++) {
    if (work->sense[j] & (DAQP_ACTIVE + DAQP_IMMUTABLE)) {
      disp += nx;  // Skip ahead in A
      continue;
    }
    // delta_s[j] = A[j,:]*delta_x
    for (k = 0, delta_s = 0, Ax = 0; k < nx; k++) {  // compute s = A(x-xold) and Ax
      Ax += work->qp->A[disp] * work->x[k];
      delta_s += work->qp->A[disp++] * work->u[k];
    }

    //&printf("#%d | alphau:%f, alphal:%f\n",j,(work->qp->bupper[j] - Ax)/delta_s,(work->qp->blower[j] - Ax)/delta_s);
    // printf("#%d| ds:%f | bu:%f | bl:%f | Ax: %f\n",j,delta_s,work->qp->bupper[j],work->qp->blower[j],Ax);
    if (delta_s > work->settings->zero_tol &&  // Feasible descent direction
        work->qp->bupper[j] < DAQP_INF &&      // Not single-sided
        work->qp->bupper[j] - Ax < delta_s * min_alpha) {
      add_ind = j;
      isupper = 1;
      min_alpha = (work->qp->bupper[j] - Ax) / delta_s;
    } else if (delta_s < -work->settings->zero_tol &&  // Feasible descent direction
               work->qp->blower[j] > -DAQP_INF &&      // Not single-sided
               work->qp->blower[j] - Ax > delta_s * min_alpha) {
      add_ind = j;
      isupper = 0;
      min_alpha = (work->qp->blower[j] - Ax) / delta_s;
    }
  }
  // printf("add_ind: %d | min_alpha:%f\n", add_ind,min_alpha);
  //  update iterate
  if (add_ind != DAQP_EMPTY_IND) {
    for (k = 0; k < nx; k++)  // x <-- x+alpha deltax
      work->x[k] += min_alpha * work->u[k];
    if (isupper) {
      DAQP_SET_UPPER(add_ind);
      daqp_add_constraint(work, add_ind, 1);
    } else {
      DAQP_SET_LOWER(add_ind);
      daqp_add_constraint(work, add_ind, -1);
    }
  }

  return add_ind;
}

static void compute_CSP_lp(DAQPWorkspace *work) {
  int i, j, disp, start_disp;
  c_float sum;
  // Forward substitution (xi <-- L\d)
  for (i = work->reuse_ind, disp = DAQP_ARSUM(work->reuse_ind); i < work->n_active; i++) {
    // Setup RHS
    // since d = b + Af => -Af = b - d
    sum = work->scaling[work->WS[i]] * work->qp->bupper[work->WS[i]] - work->dupper[work->WS[i]];
    // printf("-Af[%d]: %f\n", i+1, sum);
    for (j = 0; j < i; j++) sum -= work->L[disp++] * work->xldl[j];
    disp++;  // Skip 1 in L
    work->xldl[i] = sum;
  }
  // Scale with D  (zi = xi/di)
  for (i = work->reuse_ind; i < work->n_active; i++) work->zldl[i] = work->xldl[i] / work->D[i];
  // Backward substitution  (lam_star <-- L'\z)
  start_disp = DAQP_ARSUM(work->n_active) - 1;
  for (i = work->n_active - 1; i >= 0; i--) {
    sum = work->zldl[i];
    disp = start_disp--;
    for (j = work->n_active - 1; j > i; j--) {
      sum -= work->lam_star[j] * work->L[disp];
      disp -= j;
    }
    work->lam_star[i] = sum;
  }
  work->reuse_ind = work->n_active;  // Save forward substitution information
                                     // printf("lam: ");
                                     // for(i = 0; i < work->n_active; i++)
                                     //     printf(" %f",work->lam_star[i]);
                                     // printf("\n");
}

static int compute_projgrad(DAQPWorkspace *work) {
  int i, j, disp;
  for (j = 0; j < work->n; j++) work->u[j] = -work->qp->f[j];
  // u[m] <-- Mk'*lam_star (zero if empty set)
  for (i = 0; i < work->n_active; i++) {
    if (work->WS[i] < work->ms)
      work->u[work->WS[i]] -= work->lam_star[i];  // Hessian is identity
    else                                          // General constraint
      for (j = 0, disp = work->n * (work->WS[i] - work->ms); j < work->n; j++)
        work->u[j] -= work->M[disp++] * work->lam_star[i];
  }

  // Check if |p| = 0
  for (j = 0; j < work->n; j++) {
    if (work->u[j] > work->settings->eta_prox || work->u[j] < -work->settings->eta_prox) return 1;
  }
  return 0;
}

static int remove_constraint_lp(DAQPWorkspace *work) {
  int i;
  int rm_id = DAQP_EMPTY_IND;
  c_float val = -work->settings->dual_tol;
  for (i = 0; i < work->n_active; i++) {
    if (DAQP_IS_IMMUTABLE(work->WS[i])) continue;
    if (DAQP_IS_LOWER(work->WS[i])) {
      // printf("l: %f\n",work->lam_star[i]);
      if (-work->lam_star[i] <= val) {
        rm_id = i;
        val = -work->lam_star[i];
      }
    } else {
      // printf("u: %f\n",work->lam_star[i]);
      if (work->lam_star[i] <= val) {
        val = work->lam_star[i];
        rm_id = i;
      }
    }
  }
  if (rm_id != DAQP_EMPTY_IND) {
    daqp_remove_constraint(work, rm_id);
  }
  return rm_id;
}
