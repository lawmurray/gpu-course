#include <model.h>
#include <init.h>
#include <function.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

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
  cudaMalloc((void**)&m->theta, P*sizeof(float));
  cudaMalloc((void**)&m->dtheta, P*sizeof(float));
  cudaMalloc((void**)&m->Z, U*sizeof(float));
  cudaMalloc((void**)&m->dZ, U*sizeof(float));
  cudaMalloc((void**)&m->l, B*sizeof(float)); 
  cudaMallocHost((void**)&m->ll, sizeof(float));

  /* size */
  m->U = U;
  m->P = P;
  m->M = M;
  m->B = B;
  m->L = L;
  m->u = u;

  /* initialize parameters */
  curandGenerateNormal(gen, m->theta, P, 0.0f, 1.0f);
}

void model_term(model_t* m) {
  cudaFreeHost(m->ll);
  cudaFree(m->l);
  cudaFree(m->dZ);
  cudaFree(m->Z);
  cudaFree(m->dtheta);
  cudaFree(m->theta);

  m->ll = NULL;
  m->l = NULL;
  m->dZ = NULL;
  m->Z = NULL;
  m->dtheta = NULL;
  m->theta = NULL;
  m->U = 0;
  m->P = 0;
  m->M = 0;
  m->B = 0;
  m->L = 0;
  m->u = NULL;
}

void model_forward(model_t* m, float* X, const int B) {
  assert(B <= m->B);

  int M = m->M;
  int L = m->L;
  float* y = X + M - 1;  // last field is label

  int u = m->u[0];
  int u_prev = M - 1;

  /* first layer weights, biases and activations */
  float* W = m->theta;
  float* b = W + u*u_prev;
  float* Z = m->Z;

  cublasSetMatrix(u, B, sizeof(float), b, 0, Z, u);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, u, B, u_prev, scalar1, W, u,
      X, M, scalar1, Z, u);

  for (int l = 1; l < L; ++l) {
    u = m->u[l];
    u_prev = m->u[l - 1];

    /* rectify input layer */
    float* Z_prev = Z;
    rectify(u_prev, B, Z_prev, u_prev);

    /* next layer weights, biases and activations */
    W = b + u_prev;
    b = W + u*u_prev;
    Z = Z + u_prev;

    cublasSetMatrix(u, B, sizeof(float), b, 0, Z, u);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, u, B, u_prev, scalar1,
        W, u, Z_prev, u_prev, scalar1, Z, u);
  }
  log_likelihood(B, y, M, Z, u, m->l, 1);
  cublasSdot(handle, B, m->l, 1, scalar1, 0, m->ll);
}

void model_backward(model_t* m, float* X, const int B) {
  assert(B <= m->B);

  int U = m->U;
  int P = m->P;
  int M = m->M;
  int L = m->L;
  float* y = X + M - 1;  // last field is label

  int u = m->u[L - 1];
  int u_prev = L >= 2 ? m->u[L - 2] : M - 1;

  float* W = m->theta + P;
  float* b = NULL;
  float* Z = m->Z + U - u;
  float* Z_prev = NULL;

  float* dW = m->dtheta + P;
  float* db = NULL;
  float* dZ = m->dZ + U - u;
  float* dZ_prev = NULL;

  log_likelihood_grad(B, y, M, Z, u, dZ, u);
  for (int l = L - 1; l > 0; --l) {
    u = m->u[l];
    u_prev = m->u[l - 1];

    b = W - u;
    W = b - u*u_prev;
    Z = Z - u;
    Z_prev = Z - u_prev;

    db = dW - u;
    dW = db - u*u_prev;
    dZ = dZ - u;
    dZ_prev = dZ - u_prev;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, u_prev, B, u, scalar1, W, u,
        dZ, u, scalar0, dZ_prev, u_prev);
    rectify_grad(u_prev, B, Z_prev, u_prev, dZ_prev, u_prev);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, u, u_prev, B, scalar1, dZ,
        u, Z_prev, u_prev, scalar0, dW, u);
    cublasSgemv(handle, CUBLAS_OP_N, u, B, scalar1, dZ, u, scalar1, 0,
        scalar0, db, 1);
  }

  u = m->u[0];
  u_prev = M - 1;

  b = W - u;
  W = W - u*u_prev;
  Z = Z - u;
  assert(W == m->theta);
  assert(Z == m->Z);

  db = dW - u;
  dW = dW - u*u_prev;
  dZ = dZ - u;
  assert(dW == m->dtheta);
  assert(dZ == m->dZ);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, u, u_prev, B, scalar1, dZ, u,
      X, u_prev, scalar0, dW, u);
  cublasSgemv(handle, CUBLAS_OP_N, u, B, scalar1, dZ, u, scalar1, 0, scalar0,
      db, 1);
}
