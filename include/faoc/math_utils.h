#ifndef MATH_UTILS_H
#define MATH_UTILS_H
#endif

#include <limits.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

double uniformSample(const double min, const double max);

double scalarProd(const int n, const double *a, const double *b);

void matMatMul(const int n_rows_A, const int n_cols_A, const int n_cols_B, const double **A,
               const double B[n_cols_A][n_cols_B], double **C);

void vecMatMul(const int n_rows_B, const int n_cols_B, const double *a, const double B[n_rows_B][n_cols_B], double *c);

void vecMatDPMul(const int n_rows_B, const int n_cols_B, const double *a, const double **B, double *c);

void matVecMul(const int n_rows_A, const int n_cols_A, const double **A, const double *b, double *c);

void matVecMul2(const int n_rows_A, const int n_cols_A, const double A[n_rows_A][n_cols_A], const double *b, double *c);

void vecVecDiff(const int n_rows, const double alpha, const double *a, const double beta, const double *b, double *c);

double min(const double x, const double y);

double max(const double x, const double y);

double oneNorm(const int n, const double *x);

#ifdef __cplusplus
}
#endif
