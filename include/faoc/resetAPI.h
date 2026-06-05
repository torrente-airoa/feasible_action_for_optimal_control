#ifndef RESETAPI_H
#define RESETAPI_H

#include "multiStepAPI.h"
#include "thpool.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int dN_max;  // max allowed diff from minimum horizon length (bisection termination criterion; >= 1)
  int
    add_steps_init;  // number of steps added to lower bound in bisection to obtain an init value for the horizon length
  int add_steps;  // number of steps finally added to the horizon length that was determined by bisection (to get less
                  // aggressive reset trajectories)
} bisect_params;

typedef struct {
  const joint_limits *joint_lims;         // joint limits
  int dim;                                // dimension of state vector, i.e. order of spline (3: cubic)
  double tau_c;                           // length of single spline interval [s]
  double f_s;                             // sampling frequency [Hz]
  scale_t scale_type;                     // scaling type for input and states
  const maxctrlinvset_params *mcis_pars;  // holds parameters for maximum control invariant set computation
  const algo_params *mpc_alg_pars;        // DAQP parameters for MPC problem
  // double A[3][3];                  // scaled dynamic matrix
  // double B[3];                     // scaled input vector
  int max_N_l;                   // maximum horizon length for which MPC problem is set up
  const double *reset_state_lo;  // unscaled reset state (lower bound)
  const double *reset_state_up;  // unscaled reset state (upper bound)
  int N_l_max;  // maximum horizon length so that every state from the MCIS can be brought to the reset state in N_l_max
                // steps
} max_horizon_data;  // summarizes all required data to compute N_l_max and N_l_max_mult

typedef struct {
  joint_limits joint_lims;  // joint limits
  int dim;                  // dimension of state vector, i.e. order of spline (3: cubic)
  obj_t obj_type;           // holds type of objective in MPC model
  int N_l_max;              // maximum horizon length for mpc
  int N_l_max_mult;         // maximum horizon length for mpc_mult
  double tau_c;             // length of single spline interval [s]
  int mult;                 // multiple of tau_c used for roughly and quickly determining minimum horizon length
  double tau_s;             // sampling interval length [s]
  scale_t scale_type;       // scaling type for input and states
  maxctrlinvset_params
    mcis_pars_mult;  // holds parameters for maximum control invariant set computation (mult*tau_c discretization)
  bisect_params bs_params_mult;  // parameters controlling bisection algorithm (mult*tau_c discretization)
  DAQPProblem mpc;               // condensed MPC model with sampling interval length of tau_c
  DAQPWorkspace work;            // workspace for mpc
  DAQPProblem mpc_mult;          // condensed MPC model with sampling interval length of mult*tau_c
  DAQPWorkspace work_mult;       // workspace for mpc_mult
  double D[3];                   // diagonal scaling matrix for 3D state: x_tilde = D*x
  double d;                      // scaling coefficient for input: u_tilde = d*u
  int N;                         // number of sampling intervals spaced at tau_s
  int N_l;                       // actual horizon length for mpc (tau_c discretization)
  double **H_inf_mult;       // Coefficient matrix of H-representation of scaled symmetric maximum control invariant set
                             // (mult*tau_c discretization)
  double *H_inf_mult_rm;     // H_inf_mult in row-major order (to speed up MCIS membership computation)
  double *h_inf_mult;        // LHS/RHS vector of scaled symmetric maximum control invariant set {x | -h_inf_mult <=
                             // H_inf_mult*x <= h_inf_mult} (mult*tau_c discretization)
  int n_inf_mult;            // number of rows of H_inf_mult
  double *x_f;               // terminal state (3D, scaled) of reset planner
  double *x_0;               // initial state (3D, scaled) {pos, speed, acceleration}
  double *rowcoef;           // vector [B; A*B; A^2*B; ...; A^(N_l_max-1)*B]
  double *rowcoef_mult;      // vector [B_mult; A_mult*B_mult; A_mult^2*B_mult; ...; A_mult^(N_l_max_mult-1)*B_mult]
  double A[3][3];            // dynamic matrix (scaled triple integrator) (tau_c discretization)
  double A_mult[3][3];       // dynamic matrix (scaled triple integrator) (mult*tau_c discretization)
  double B[3];               // input matrix (scaled triple integrator) (tau_c discretization)
  double B_mult[3];          // input matrix (scaled triple integrator) (mult*tau_c discretization)
  double *x_max;             // upper state bound (scaled) of symmetric state set (tau_c discretization)
  double *x_max_mult;        // upper state bound (scaled) of symmetric state set (mult*tau_c discretization)
  double u_max;              // scaled maximum input (jerk) (tau_c discretization)
  double u_max_mult;         // scaled maximum input (jerk) (mult*tau_c discretization)
  double *uh;                // optimal scaled position of length N_l+1 from MPC (tau_c discretization)
  double *udh;               // optimal scaled speed of length N_l+1 from MPC (tau_c discretization)
  double *uddh;              // optimal scaled acceleration of length N_l+1 from MPC (tau_c discretization)
  double *udddh;             // optimal scaled jerk of length N_l from MPC (tau_c discretization)
  double *u_zoh;             // ZOH sampled position trajectory of length N+1
  double *du_zoh;            // ZOH sampled velocity trajectory of length N+1
  double *ddu_zoh;           // ZOH sampled acceleration trajectory of length N+1
  double *dddu_zoh;          // ZOH sampled jerk trajectory of length N+1
  double *lambda_star_mult;  // Lagrange multiplier vector for warmstarting (mult*tau_c discretization)
  double udddh_m1;           // last jerk of previous plan (scaled)
  mcis_ret_code mcis_ret;    // exit code of maximum control invariant set computation
  planner_ret_code planner_ret;  // exit code of MPC planner
} rpdata;                        // all data needed by reset planner for a single axis

int initResetData(rpdata *rpd, const int dim, const obj_t obj_type, const int n_joints, const double tau_c,
                  const int mult, const int max_N_l, int *N_l_max, int *N_l_max_mult, const double f_s,
                  const scale_t scale_type, const bool PARALLEL, const int n_threads, const double *reset_states_lo,
                  const double *reset_states_up, const maxctrlinvset_params *mcis_pars_mult,
                  const algo_params *mpc_alg_pars, const bisect_params *bs_params_mult, const rp_mode reset_mode,
                  double *init_time);

void computeResetTraj(void *arg);

void syncTrajs(rpdata *rpd, const int n_joints, threadpool thpool);

int freeResetData(rpdata *rpd, const int n_joints);

#ifdef __cplusplus
}
#endif

#endif
