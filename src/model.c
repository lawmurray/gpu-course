#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <model.h>
#include <data.h>
#include <init.h>
#include <function.h>

void model_init(model_t* m, const int M, const int B, const int L,
    const int* u) {
  /* count number of parameters and units */
  int U = u[0];  // number of units
  int P = u[0]*(M - 1) + u[0];  // number of parameters
  for (int l = 1; l < L; ++l) {
    U += u[l];
    P += u[l]*u[l - 1] + u[l];
  }

  /* allocate */
  cudaMallocManaged((void**)&m->theta, P*sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&m->dtheta, P*sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&m->A, U*B*sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&m->dA, U*B*sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&m->l, B*sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&m->ll, sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&m->ones, B*sizeof(float), cudaMemAttachGlobal);

  /* convenience pointers into allocations */
  m->W = (float**)malloc(L*sizeof(float*));
  m->dW = (float**)malloc(L*sizeof(float*));
  m->b = (float**)malloc(L*sizeof(float*));
  m->db = (float**)malloc(L*sizeof(float*));
  m->Z = (float**)malloc(L*sizeof(float*));
  m->dZ = (float**)malloc(L*sizeof(float*));

  m->W[0] = m->theta;
  m->dW[0] = m->dtheta;
  m->b[0] = m->W[0] + u[0]*(M - 1);
  m->db[0] = m->dW[0] + u[0]*(M - 1);
  m->Z[0] = m->A;
  m->dZ[0] = m->dA;
  for (int l = 1; l < L; ++l) {
    m->W[l] = m->b[l - 1] + u[l - 1];
    m->dW[l] = m->db[l - 1] + u[l - 1];
    m->b[l] = m->W[l] + u[l]*u[l - 1];
    m->db[l] = m->dW[l] + u[l]*u[l - 1];
    m->Z[l] = m->Z[l - 1] + B*u[l - 1];
    m->dZ[l] = m->dZ[l - 1] + B*u[l - 1];
  }

  /* size */
  m->U = U;
  m->P = P;
  m->M = M;
  m->B = B;
  m->L = L;
  m->u = u;

  /* initialize */
  for (int i = 0; i < B; ++i) {
    m->ones[i] = 1.0f;
  }
  for (int l = 0; l < L; ++l) {
    int u_prev = (l == 0) ? M - 1 : u[l - 1];
    for (int i = 0; i < u[l]*u_prev; ++i) {
      m->W[l][i] = (2.0*drand48() - 1.0)/u_prev;
    }
    for (int i = 0; i < u[l]; ++i) {
      m->b[l][i] = 0.0f;
    }
  }
}

void model_term(model_t* m) {
  free(m->W);
  free(m->dW);
  free(m->b);
  free(m->db);
  free(m->Z);
  free(m->dZ);

  m->W = NULL;
  m->dW = NULL;
  m->b = NULL;
  m->db = NULL;
  m->Z = NULL;
  m->dZ = NULL;

  cudaFree(m->theta);
  cudaFree(m->dtheta);
  cudaFree(m->A);
  cudaFree(m->dA);
  cudaFree(m->l);
  cudaFree(m->ll);
  cudaFree(m->ones);

  m->theta = NULL;
  m->dtheta = NULL;
  m->A = NULL;
  m->dA = NULL;
  m->l = NULL;
  m->ll = NULL;
  m->ones = NULL;

  m->U = 0;
  m->P = 0;
  m->M = 0;
  m->B = 0;
  m->L = 0;
  m->u = NULL;
}

void model_forward(model_t* m, float* X, const int B) {
  assert(B <= m->B);

  float** W = m->W;
  float** b = m->b;
  float** Z = m->Z;
  int M = m->M;
  int L = m->L;
  const int* u = m->u;
  float* ones = m->ones;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, u[0], B, M - 1, scalar1, W[0],
      u[0], X, M, scalar0, Z[0], u[0]);
  cublasSger(handle, u[0], B, scalar1, b[0], 1, ones, 1, Z[0], u[0]);
  for (int l = 1; l < L; ++l) {
    rectify(u[l - 1], B, Z[l - 1], u[l - 1]);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, u[l], B, u[l - 1], scalar1,
        W[l], u[l], Z[l - 1], u[l - 1], scalar0, Z[l], u[l]);
    cublasSger(handle, u[l], B, scalar1, b[l], 1, ones, 1, Z[l], u[l]);
  }
}

void model_backward(model_t* m, float* X, const int B) {
  assert(B <= m->B);

  float** W = m->W;
  float** dW = m->dW;
  float** b = m->b;
  float** db = m->db;
  float** Z = m->Z;
  float** dZ = m->dZ;
  float* ones = m->ones;

  int M = m->M;
  int L = m->L;
  const int* u = m->u;

  squared_error_grad(B, X + M - 1, M, Z[L - 1], 1, dZ[L - 1], 1);
  for (int l = L - 1; l > 0; --l) {
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, u[l - 1], B, u[l],
        scalar1, W[l], u[l], dZ[l], u[l], scalar0, dZ[l - 1], u[l - 1]);
    rectify_grad(u[l - 1], B, Z[l - 1], u[l - 1], dZ[l - 1], u[l - 1]);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, u[l], u[l - 1], B, scalar1,
        dZ[l], u[l], Z[l - 1], u[l - 1], scalar0, dW[l], u[l]);
    cublasSgemv(handle, CUBLAS_OP_N, u[l], B, scalar1, dZ[l], u[l], ones,
        1, scalar0, db[l], 1);
  }
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, u[0], M - 1, B, scalar1,
      dZ[0], u[0], X, M, scalar0, dW[0], u[0]);
  cublasSgemv(handle, CUBLAS_OP_N, u[0], B, scalar1, dZ[0], u[0], ones, 1,
      scalar0, db[0], 1);
}

void model_loss_clear(model_t* m) {
  cudaMemcpyAsync(m->ll, scalar0, sizeof(float), cudaMemcpyDefault,
      cudaStreamDefault);
}

void model_loss_accumulate(model_t* m, float* X, const int B) {
  float** Z = m->Z;
  float* l = m->l;
  float* ll = m->ll;
  float* ones = m->ones;
  int M = m->M;
  int L = m->L;
  const int* u = m->u;

  squared_error(B, X + M - 1, M, Z[L - 1], 1, l, 1);
  cublasSgemv(handle, CUBLAS_OP_N, 1, B, scalar1, m->l, 1, ones, 1, scalar1,
      m->ll, 1);
}
