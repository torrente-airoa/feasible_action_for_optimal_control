#ifndef MULTISTEPAPI_H
#define MULTISTEPAPI_H

#include <assert.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "mapping.h"

#ifndef M_PI
#define M_PI (3.14159265358979323846264338327950288)
#endif

#ifdef __cplusplus
extern "C" {
#endif
// MAGN: minimize sum of squared input
// DIFF: minimize sum of squared input diffs (requires last input spline_input_m1 to be provided)
// MIXED: minimize sum of squared input and diff with last segment (requires last accel spline_input_m1 to be provided)
typedef enum { MAGN, DIFF, MIXED } obj_t;

typedef enum { NO_SCALING = 0, NORMALIZE = 1, TIME_BASED = 2 } scale_t;

typedef enum {
  DIST,  // in this case, the number of steps of every reset plan is chosen individually
  SYNC   // sync mode for reset planner (i.e. all reset plans have the same number of steps)
} rp_mode;

typedef enum {
  VERTEX_COMP = -4,
  REDUNDANCY_ERR = -3,
  MEMORY_LIMIT = -2,
  ITER_LIMIT = -1,
  ALL_OK = 0
} mcis_ret_code;  // return code of maximum control invariant set computation

typedef enum {
  INIT_PLANNER_RET = -10,
  NO_FEAS_HORIZON = -4,
  BISECT_ERR = -3,
  MAP_ERR = -2,
  MPC_ERR = -1,
  TRAJ_OK = 0
} planner_ret_code;

typedef struct {
  double zero_tol;    // if |x| <= zero_tol, x is considered to be zero
  double primal_tol;  // tolerance for primal infeasibility in DAQP
  double shift_tol;   // amount by which rhs of an inequality constraint is decreased before redundancy is checked
  double h_rel_tol;   // relative tolerance when comparing lhs/rhs vectors
  double Hh_abs_tol;  // absolute tolerance when comparing [H,h] representations
  int max_iter;       // maximum number of iterations
  int n_constr_max;   // (>0) maximum number of constraints in matrix H_k and H_kp1 (for max ctrl inv set computations)
  char
    re_method[10];  // method for redundancy elimination: "opt" (old method) | "convh" (new method, scales much better)
} maxctrlinvset_params;

typedef struct {
  double qlow;      // THIS IS UNUSED BUT PUT HERE TO KEEP CONSISTENCY WITH OTHER MULTISTEP CODE!!
  double qup;       // [rad] or [m]
  double qdotup;    // [rad/s] or [m/s]
  double qddotup;   // [rad/s^2] or [m/s^2]
  double qdddotup;  // [rad/s^3] or [m/s^3]
} joint_limits;

typedef struct {
  map_data *mapd;
  joint_limits joint_lims;
  struct terminal_state_deviations {
    // Note: The deviations in this struct are used in the following way:
    //       q_target + delta_qlow <= u_N_l <= q_target + delta_qup
    //       qdot_target + delta_qdotlow <= ud_N_l <= qdot_target + delta_qdotup
    // Note: For map_type == POS, only the first line of inequalities is enforced
    double delta_qlow;     // allowed lower deviation from terminal position target [rad] or [m]
    double delta_qup;      // allowed upper deviation from terminal position target [rad] or [m]
    double delta_qdotlow;  // allowed lower deviation from terminal speed target [rad/s] or [m/s]
    double delta_qdotup;   // allowed upper deviation from terminal speed target [rad/s] or [m/s]
  } term_state_devs;
  maxctrlinvset_params
    mcis_pars;     // holds parameters for maximum control invariant set computation (to allow for parallel computation)
  obj_t obj_type;  // holds type of objective in MPC model
  DAQPProblem mpc;         // condensed MPC model (allows for deviations in the terminal position and speed)
  DAQPWorkspace mpc_work;  // workspace for mpc
  scale_t scale_type;      // scaling type for input and states
  double D[3];             // diagonal scaling matrix for 3D state: x_tilde = D*x
  double d;                // scaling coefficient for input: u_tilde = d*u
  double tau_c;            // length of single spline interval [s]
  double tau_s;            // sampling interval length [s]
  int N;                   // number of sampling intervals spaced at tau_s
  int n_inf;               // number of rows of H_inf
  double **H_inf;          // Coefficient matrix of H-representation of scaled symmetric maximum control invariant set
  double *H_inf_rm;        // H_inf in row-major order (to speed up MCIS membership computation)
  double *h_inf;   // LHS/RHS vector of scaled symmetric maximum control invariant set {x | -h_inf <= H_inf*x <= h_inf}
  double *x_0;     // initial state (3D, scaled) {pos, speed, acceleration}
  double A[3][3];  // dynamic matrix (scaled triple integrator)
  double B[3];     // input matrix (scaled triple integrator)
  double V_X[8]
            [3];   // vertices of scaled feasible set (used for redundancy elimination in max ctrl inv set computations)
  double *x_max;   // upper state bound (scaled) of symmetric state set
  double **H_0;    // coefficient matrix of symmetric H-represenation of scaled feasible state set
  double *h_0;     // vector of symmetric H-represenation of scaled feasible state set {x | -h_0 <= H_0 <= h_0}
  double u_max;    // scaled maximum input (jerk)
  double *x_N_l;   // scaled _actual_ terminal state (with dim dimensions) at t=N_l*tau_c (pos (and/or speed) could be
                   // different from y_N_l if deviations in term_state_devs are nonzero)
  double *uh;      // optimal scaled position of length N_l+1 from MPC
  double *udh;     // optimal scaled speed of length N_l+1 from MPC
  double *uddh;    // optimal scaled acceleration of length N_l+1 from MPC
  double *udddh;   // optimal scaled jerk of length N_l from MPC
  double *u_zoh;   // ZOH sampled position trajectory of length N+1
  double *ud_zoh;  // ZOH sampled velocity trajectory of length N+1
  double *udd_zoh;               // ZOH sampled acceleration trajectory of length N+1
  double *uddd_zoh;              // ZOH sampled jerk trajectory of length N+1
  double udddh_m1;               // last jerk of previous planning (scaled)
  double **E_full;               // maps inputs to state at terminal time
  double **E;                    // maps inputs to position and speed at terminal time (only if map_type == POS_VEL)
  mcis_ret_code mcis_ret;        // exit code of maximum control invariant set computation
  planner_ret_code planner_ret;  // exit code of MPC planner
} mpdata;                        // all data needed by multi-step planner for a single axis

int initData(mpdata *mpd, const obj_t obj_type, const int n_joints, const double tau_c, const double f_s,
             const scale_t scale_type, const bool PARALLEL, const int n_threads, const maxctrlinvset_params *mcis_pars,
             const algo_params *mpc_alg_pars, const algo_params *chebyshev_alg_pars,
             const algo_params *direct_chebyshev_alg_pars, const algo_params *beta_alg_pars,
             const algo_params *pos_set_alg_pars, double *init_time);

int setupOptProbsPosSpeed(mpdata *mpd, const algo_params *mpc_alg_pars, const algo_params *chebyshev_alg_pars,
                          const algo_params *direct_chebyshev_alg_pars, const algo_params *beta_alg_pars);

int setupOptProbsPos(mpdata *mpd, const algo_params *mpc_alg_pars, const algo_params *pos_set_alg_pars);

void mappingUpdateBounds(mpdata *mpd, map_runtime_bounds *bounds, c_float *lower, c_float *upper, double *e_full);

void computeStep(void *arg);

int mpcPlanner(mpdata *mpd);

void computeMaxCtrlInvSetSymmetric(void *arg);

int isInsideMaxCtrlInvSet(const mpdata *mpd, const double **X, const int n_X, const double shrink_factor);

int isInsideMaxCtrlInvSetFast(const double *H_inf_rm, const double *h_inf, const double **X, const int n_X,
                              const double shrink_factor, const int n_inf, const int dim);

int freeData(mpdata *mpd, const int n_joints);

#ifdef __cplusplus
}
#endif

#endif
