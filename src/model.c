#include <model.h>
#include <data.h>
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
  cudaMalloc((void**)&m->Z, U*B*sizeof(float));
  cudaMalloc((void**)&m->dZ, U*B*sizeof(float));
  cudaMallocHost((void**)&m->l, sizeof(float));

  /* size */
  m->U = U;
  m->P = P;
  m->M = M;
  m->B = B;
  m->L = L;
  m->u = u;

  /* initialize */
  curandGenerateNormal(gen, m->theta, P, 0.0f, 1.0f);
}

void model_term(model_t* m) {
  cudaFreeHost(m->l);
  cudaFree(m->dZ);
  cudaFree(m->Z);
  cudaFree(m->dtheta);
  cudaFree(m->theta);

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

  int U = m->U;
  int M = m->M;
  int L = m->L;
  int u_prev = M - 1;
  int u = m->u[0];

  /* first layer weights, biases and activations */
  float* W = m->theta;
  float* b = W + u*u_prev;
  float* Z = m->Z;

  cublasSetMatrix(u, B, sizeof(float), b, 0, Z, U);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, u, B, u_prev, scalar1, W, u,
      X, M, scalar1, Z, u);

  for (int l = 1; l < L; ++l) {
    u_prev = m->u[l - 1];
    u = m->u[l];

    /* rectify input layer */
    float* Z_prev = Z;
    rectify(u_prev, B, Z_prev, u_prev);

    /* next layer weights, biases and activations */
    W = b + u_prev;
    b = W + u*u_prev;
    Z = Z + B*u_prev;

    cublasSetMatrix(u, B, sizeof(float), b, 0, Z, u);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, u, B, u_prev, scalar1,
        W, u, Z_prev, u_prev, scalar1, Z, u);
  }
}

void model_backward(model_t* m, float* X, const int B) {
  assert(B <= m->B);

  int U = m->U;
  int P = m->P;
  int M = m->M;
  int L = m->L;

  int u = m->u[L - 1];
  int u_prev = L >= 2 ? m->u[L - 2] : M - 1;

  float* W = m->theta + P;
  float* b = NULL;
  float* Z = NULL;
  float* Z_prev = m->Z + B*U - B*u;

  float* dW = m->dtheta + P;
  float* db = NULL;
  float* dZ = NULL;
  float* dZ_prev = m->dZ + B*(U - u);

  log_likelihood_grad(B, X + M - 1, M, Z_prev, u, dZ_prev, u);
  for (int l = L - 1; l > 0; --l) {
    u_prev = m->u[l - 1];
    u = m->u[l];

    Z = Z_prev;
    dZ = dZ_prev;

    b = W - u;
    W = b - u*u_prev;
    Z_prev = Z - B*u_prev;

    db = dW - u;
    dW = db - u*u_prev;
    dZ_prev = dZ - B*u_prev;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, u_prev, B, u, scalar1, W, u,
        dZ, u, scalar0, dZ_prev, u_prev);
    rectify_grad(u_prev, B, Z_prev, u_prev, dZ_prev, u_prev);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, u, u_prev, B, scalar1, dZ,
        u, Z_prev, u_prev, scalar0, dW, u);
    cublasSgemv(handle, CUBLAS_OP_N, u, B, scalar1, dZ, u, scalar1, 0,
        scalar0, db, 1);
  }

  u_prev = M - 1;
  u = m->u[0];

  Z = Z_prev;
  assert(Z == m->Z);
  dZ = dZ_prev;
  assert(dZ == m->dZ);

  b = W - u;
  W = b - u*u_prev;
  assert(W == m->theta);

  db = dW - u;
  dW = db - u*u_prev;
  assert(dW == m->dtheta);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, u, u_prev, B, scalar1, dZ, u,
      X, M, scalar0, dW, u);
  cublasSgemv(handle, CUBLAS_OP_N, u, B, scalar1, dZ, u, scalar1, 0, scalar0,
      db, 1);
}

float model_predict(model_t* m, data_t* d) {
  float* X = d->X;
  float* l = d->l;
  float* Z = m->Z;
  int N = d->N;
  int M = m->M;
  int B = m->B;
  int L = m->L;
  int U = m->U;
  int u = m->u[L - 1];

  for (int i = 0; i < N; i += B) {
    int b = (i + B < N) ? B : N - i;
    model_forward(m, X + i*M, b);
    log_likelihood(b, X + i*M + M - 1, M, Z + b*U - b*u, u, l + i, 1);
  }
  cublasSdot(handle, N, d->l, 1, scalar1, 0, m->l);
  cudaDeviceSynchronize();
  return *m->l;
}
