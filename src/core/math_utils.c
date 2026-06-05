#include <math.h>
#include <stdlib.h>

double uniformSample(const double min, const double max) { return (max - min) * ((double)rand() / RAND_MAX) + min; }

double scalarProd(const int n, const double *a, const double *b) {
  // compute c = a'*b
  double c = 0;
  for (int i = 0; i < n; i++) c += a[i] * b[i];

  return c;
}

void matMatMul(const int n_rows_A, const int n_cols_A, const int n_cols_B, const double **A,
               const double B[n_cols_A][n_cols_B], double **C) {
  double coeff = 0;
  // Compute C = A * B
  for (int i = 0; i < n_rows_A; i++) {
    for (int j = 0; j < n_cols_B; j++) {
      coeff = 0;
      for (int k = 0; k < n_cols_A; k++) {
        coeff += A[i][k] * B[k][j];
      }
      C[i][j] = coeff;
    }
  }
}

void vecMatMul(const int n_rows_B, const int n_cols_B, const double *a, const double B[n_rows_B][n_cols_B], double *c) {
  // Compute c' = a'*B
  for (int j = 0; j < n_cols_B; j++) {
    c[j] = 0;
    for (int i = 0; i < n_rows_B; i++) c[j] += a[i] * B[i][j];
  }
}

void vecMatDPMul(const int n_rows_B, const int n_cols_B, const double *a, const double **B, double *c) {
  // Compute c' = a'*B
  for (int j = 0; j < n_cols_B; j++) {
    c[j] = 0;
    for (int i = 0; i < n_rows_B; i++) c[j] += a[i] * B[i][j];
  }
}

void matVecMul(const int n_rows_A, const int n_cols_A, const double **A, const double *b, double *c) {
  // Compute c = A*b
  for (int i = 0; i < n_rows_A; i++) {
    c[i] = 0;
    for (int j = 0; j < n_cols_A; j++) c[i] += A[i][j] * b[j];
  }
}

void matVecMul2(const int n_rows_A, const int n_cols_A, const double A[n_rows_A][n_cols_A], const double *b,
                double *c) {
  // Compute c = A*b
  for (int i = 0; i < n_rows_A; i++) {
    c[i] = 0;
    for (int j = 0; j < n_cols_A; j++) c[i] += A[i][j] * b[j];
  }
}

void vecVecDiff(const int n_rows, const double alpha, const double *a, const double beta, const double *b, double *c) {
  // Compute c = alpha*a - beta*b
  for (int i = 0; i < n_rows; i++) c[i] = alpha * a[i] - beta * b[i];
}

double min(const double x, const double y) { return (x < y) ? x : y; }

double max(const double x, const double y) { return (x > y) ? x : y; }

double oneNorm(const int n, const double *x) {
  // returns the 1-norm of vector x
  double one_norm = 0;
  for (int i = 0; i < n; i++) one_norm += fabs(x[i]);

  return one_norm;
}
