#ifndef HELPER_H
#define HELPER_H

#include <stdbool.h>
#include <stdio.h>

#include "resetAPI.h"

#ifdef __cplusplus
extern "C" {
#endif
void scaleDynNSets(mpdata *mpd, const scale_t scale_type);

void scaleDyn(const scale_t scale_type, const joint_limits *joint_lims, const double tau_c, double A[3][3], double B[3],
              double D[3], double *d);

void getShrinkageDeltas(const joint_limits *lims, const double tau_c, double *Delta_dot, double *Delta);

void getRandomInitialState(mpdata *mpd);

void getRandomAbstractAction(mpdata *mpd);

void zohSampleSpline(mpdata *mpd);

void printArr(const char *name, const int m, const int n, const void *V);

void printArrInt(const char *name, const int m, const int n, const void *V);

void setCol(c_float *A, const int num_cols, const int start_row, const int col, const int len, const c_float rowcoef[]);

int redundancyElimination(const double **H, const double *h, const int m, const int n, const double f_bar,
                          const maxctrlinvset_params *mcis_pars, int *R, int *n_R);

int redundancyEliminationConvexHull(const double **H, const double *h, const int m, const int n, double *H_rm,
                                    double *H_final, int *nonR, const int m_aux, int *R, int *n_R, double **vert_rm,
                                    int *n_vert);

mcis_ret_code mixConstraintsSameSet(const double **H_k, const double *h_k, const int dim, const double *g_k,
                                    const int *I, const int n_I, const double A[dim][dim], const int n_constr_max,
                                    const maxctrlinvset_params *mcis_pars, double **H_kp1, double *h_kp1, int *m_kp1,
                                    const double *vert_rm, const int n_vert);

mcis_ret_code mixConstraints(const double **H_k, const double *h_k, const int dim, const double *g_k, const int *I,
                             const int n_I, const int *J, const int n_J, const double A[dim][dim],
                             const int n_constr_max, const maxctrlinvset_params *mcis_pars, double **H_kp1,
                             double *h_kp1, int *m_kp1, const double *vert_rm, const int n_vert);

int validateSet(const double **H, const double *h, const int m, const int n, const char dir[], const int i,
                const double prefact, const maxctrlinvset_params *mcis_pars);

int validateStep(const mpdata *mpd, FILE *f);

int validateReset(const rpdata *rpd, FILE *f);

void compute_relative_to_dirname(const char *current_dir, const char *target_dirname, char *relative_path);

int validateTraj(const rpdata *rpd);

void getFileHandle(const char dir[], const int joint_num, const double prefact, const obj_t obj_type,
                   const double x_0_scaling, FILE **f_data);

void getFileHandleReset(const char dir[], const int joint_num, const double prefact, const obj_t obj_type,
                        const double tau_c, const int mult, FILE **f_data);

void getFileHandleCheckInvSet(const char dir[], const int joint_num, const int episode_num, FILE **f_data);

int setupMPC(mpdata *mpd, const algo_params *alg_pars);

void updateBounds(mpdata *mpd, c_float *U_min, c_float *X_min, c_float *pos_speed_min, c_float *x_term_min,
                  c_float *U_max, c_float *X_max, c_float *pos_speed_max, c_float *x_term_max, double *e,
                  double *e_full);

void exportDAQP(DAQPProblem *prob, DAQPSettings *set);

void exportTrajs(const rpdata *rpd, const int n_joints, const char dirname[], const int j);

void rpd2mpd(rpdata *rpd, mpdata *mpd);

void mpd2rpd(mpdata *mpd, rpdata *rpd);

int initMPCProb(DAQPProblem *model, DAQPWorkspace *work, const algo_params *alg_pars, const double A[3][3],
                const double B[3], const int N_l, const int dim, double *rowcoef);

void shrinkState(const double *x_out, const double **H, const double *h, const int m, const int n,
                 const double shrink_fact, double *x_in);

int mpcBisect(DAQPProblem *model, DAQPWorkspace *work, const int dim, const double A[3][3], const double *rowcoef,
              const double *x_max, const double u_max, int N_l_lo, const int N_l_init, int N_l_up, const double *x_0,
              const double *x_f, const double udddh_m1, const obj_t obj_type, const bisect_params *bs_params,
              double *lambda_star);

double minTContTime(const double *x_0, double s_f, const double *x_max, const double *D);

int mpcProb(DAQPProblem *model, DAQPWorkspace *work, const int dim, const double A[3][3], const double *rowcoef,
            const double *x_max, const double u_max, const int N_l, const double *x_0, const double *x_f,
            const double udddh_m1, const obj_t obj_type);

int mpcSolve(DAQPProblem *model, DAQPWorkspace *work, const int dim, const double A[3][3], const double *rowcoef,
             const double *x_max, const double u_max, int *N_l, const int N_l_max, const double *x_0, const double *x_f,
             const double udddh_m1, const obj_t obj_type, const int mult, const double *lambda_star_mult,
             bool EXTEND_ON_INF, double *udddh);

int compare(const void *a, const void *b);

int getMaxHorizons(const rpdata *rpd, const int dim, const int n_joints, const double tau_c, const int max_N_l,
                   const double f_s, const scale_t scale_type, const bool PARALLEL, const int n_threads,
                   const double *reset_states_lo, const double *reset_states_up, const maxctrlinvset_params *mcis_pars,
                   const algo_params *mpc_alg_pars, int *N_l_max);

void computeMaxHorizon(void *arg);

void calculateMaxHorizon(const mpdata *mpd, const double *reset_state, const int max_N_l, const int N_l_max_lb,
                         const algo_params *mpc_alg_params, int *N_l_max);

int computeVertices(const double **H, const double *h, const int m, const int n, double **vert_rm, int *n_vert);

bool isRedundant(const double *a, const double b, const double *vert_rm, const int n_vert, const int n,
                 const double tol);

void computeSyncResetTraj(void *arg);

#ifdef __cplusplus
}
#endif

#endif
