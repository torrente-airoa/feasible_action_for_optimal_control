#ifndef DAQP_LP_H
#define DAQP_LP_H

#ifdef __cplusplus
extern "C" {
#endif  // ifdef __cplusplus

#include "daqp/constants.h"
#include "daqp/daqp.h"
#include "daqp/types.h"

int daqp_lp(DAQPWorkspace *work);
// daqp_update_ldp from a cleared active set, so the solve that follows is never warm-started.
int daqp_update_ldp_cold(int update_mask, DAQPWorkspace *work, DAQPProblem *qp);
// daqp_ldp warm-started, repeated once from a cleared active set if it does not reach the optimum.
int daqp_ldp_retry(int update_mask, DAQPWorkspace *work, DAQPProblem *qp);

#ifdef __cplusplus
}
#endif  // ifdef __cplusplus

#endif  // ifndef DAQP_LP_H
