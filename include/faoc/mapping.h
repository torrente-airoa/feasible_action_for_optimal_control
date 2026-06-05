#include "daqp/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { POS = 1, POS_VEL = 2 } map_t;  // mapping type: POS (1D) or POS_VEL (2D)

typedef enum {
  INV = 1,      // mapping is invertible
  NO_INV = 0,   // mapping is not invertible (since action set has no interior)
  OPT_ERR = -1  // numerical issues in DAQP solver
} map_ret_code;

typedef struct {
  double zero_tol;    // DAQP: values below zero_tol are regarded as zero
  double primal_tol;  // DAQP: tolerance for primal infeasibility
  double dual_tol;    // DAQP: tolerance for dual infeasibility
  double eps_prox;  // DAQP: regularization parameter used for proximal-point iterations (0 means that no proximal-point
                    // iterations are performed)
  double eta_prox;  // DAQP: tolerance that determines if a fix-point has been reached during proximal-point iterations
} algo_params;

typedef struct {
  int n_inf;         // number of rows of H_inf
  double **H_inf;    // coefficient matrix of MCIS H-representation
  double *H_inf_rm;  // row-major copy of H_inf
  double *h_inf;     // rhs vector of MCIS H-representation
} mcis_data;

typedef struct {
  const c_float *lower;  // feasible lower bounds: [U, X, X_term]
  const c_float *upper;  // feasible upper bounds: [U, X, X_term]
  const double *e_full;  // terminal free response A^N_l x_0 (length dim)
} map_runtime_bounds;

typedef struct {
  map_t map_type;           // mapping type: POS (1D) or POS_VEL (2D)
  int N_l;                  // number of steps in the horizon
  int dim;                  // dimension of action set (1 for POS, 2 for POS_VEL)
  double *z_min;            // lower corner of rectangle abstract action set
  double *z_max;            // upper corner of rectangle abstract action set
  double *z_N_l;            // base 'abstract' state from [z_min, z_max]
  double *y_N_l;            // result of mapping
  const double **E;         // maps inputs to terminal position/speed (for POS_VEL inverse mapping)
  double **G;               // direction-mapping matrix (mapping algorithm) (only if map_type == POS_VEL)
  double **G_inv;           // inverse of direction-mapping matrix (mapping algorithm) (only if map_type == POS_VEL)
  DAQPProblem beta_comp;    // model for computation of scaling factor beta
  DAQPWorkspace beta_work;  // workspace for beta_comp
  DAQPProblem pos_set;      // model for computation of min/max position
  DAQPWorkspace pos_work;   // workspace for pos_set
  DAQPProblem chebyshev;    // model for Chebyshev ball computation (based on condensed MPC model)
  DAQPWorkspace chebyshev_work;  // workspace for chebyshev
  DAQPProblem direct_chebyshev;  // model for computing a scaled inf-norm ball in the action set directly
  DAQPWorkspace direct_work;     // workspace for direct_chebyshev
  DAQPProblem
    direct_unique;  // QP model for computing a unique minimum-norm Chebyshev center (with radius of ball fixed)
  DAQPWorkspace direct_unique_work;  // workspace for direct_unique
  DAQPProblem
    chebyshev_unique;  // QP model for computing a unique minimum-norm Chebyshev center (with radius of ball fixed)
  DAQPWorkspace unique_work;  // workspace for chebyshev_unique
} map_data;

int initMapping(map_data **mapd, const int dim, const int N_l, const double z_min[], const double z_max[],
                const map_t map_type, double *init_time);

int freeMapping(map_data *mapd);

map_ret_code mapFromAbstractSetWithBounds(map_data *mapd, const int n_inf, const map_runtime_bounds *bounds);

map_ret_code mapToAbstractSetWithBounds(map_data *mapd, const int n_inf, const map_runtime_bounds *bounds);

map_ret_code getActionSetPosWithBounds(map_data *mapd, const int n_inf, const map_runtime_bounds *bounds, double *y_lo,
                                       double *y_up);

map_ret_code getIntPointActionSetWithBounds(map_data *mapd, const int n_inf, const map_runtime_bounds *bounds,
                                            double y_bar[2], double e[2]);

int SetupOptProbs1D(map_data *mapd, const int n_inf, const double *A_mpc, const double **E_full,
                    const algo_params *algo_params);

int SetupOptProbs2D(map_data *mapd, const int n_inf, const double *A_mpc, const double **E,
                    const algo_params *chebyshev_alg_pars, const algo_params *direct_chebyshev_alg_pars,
                    const algo_params *beta_alg_pars);

void ComputeGMatrix(map_data *mapd, const double **E);

int setupChebyshev(map_data *mapd, const int n_inf, const double *A_mpc, const algo_params *alg_pars);

int setupChebyshevMinNormCenter(map_data *mapd, const algo_params *alg_pars);

int setupDirectChebyshev(map_data *mapd, const int n_inf, const double *A_mpc, const double **E,
                         const algo_params *alg_pars);

int setupDirectChebyshevMinNormCenter(map_data *mapd, const algo_params *alg_pars);

int setupBeta(map_data *mapd, const int n_inf, const double *A_mpc, const double **E, const algo_params *alg_pars);

int setupPosSet(map_data *mapd, const int n_inf, const double *A_mpc, const double **E_full,
                const algo_params *alg_pars);

#ifdef __cplusplus
}
#endif
