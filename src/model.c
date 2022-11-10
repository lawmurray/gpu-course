#include <model.h>
#include <init.h>
#include <function.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void model_init(model_t* m, const int P, const int B, const int L,
    const int* U) {
  float** W = (float**)malloc(L*sizeof(float*));
  float** dW = (float**)malloc(L*sizeof(float*));
  float** b = (float**)malloc(L*sizeof(float*));
  float** db = (float**)malloc(L*sizeof(float*));
  float** Z = (float**)malloc(L*sizeof(float*));
  float** dZ = (float**)malloc(L*sizeof(float*));
  float* ll = NULL;

  int prev = P;  // previous layer width
  for (int l = 0; l < L; ++l) {
    cudaMalloc((void**)&W[l], U[l]*prev*sizeof(float));
    cudaMalloc((void**)&dW[l], U[l]*prev*sizeof(float));
    cudaMalloc((void**)&b[l], U[l]*sizeof(float));
    cudaMalloc((void**)&db[l], U[l]*sizeof(float));
    cudaMalloc((void**)&Z[l], U[l]*B*sizeof(float));
    cudaMalloc((void**)&dZ[l], U[l]*B*sizeof(float));
    prev = U[l];
  }
  cudaMalloc((void**)&ll, B*sizeof(float)); 

  /* initialize parameters */
  curandGenerateNormal(gen, W[0], U[0]*P, 0.0f, sqrtf(1.0f/P));
  curandGenerateNormal(gen, b[0], U[0], 0.0f, 1.0f);
  for (int l = 1; l < L; ++l) {
    curandGenerateNormal(gen, W[l], U[l]*U[l - 1], 0.0f, sqrtf(1.0f/U[l - 1]));
    curandGenerateNormal(gen, b[l], U[l], 0.0f, 1.0f);
  }

  m->W = W;
  m->dW = dW;
  m->b = b;
  m->db = db;
  m->Z = Z;
  m->dZ = dZ;
  m->ll = ll;
  m->P = P;
  m->B = B;
  m->L = L;
  m->U = U;
}

void model_term(model_t* m) {
  for (int l = 0; l < m->L; ++l) {
    cudaFree(m->W[l]);
    cudaFree(m->dW[l]);
    cudaFree(m->b[l]);
    cudaFree(m->db[l]);
    cudaFree(m->Z[l]);
    cudaFree(m->dZ[l]);
  }

  free(m->W);
  free(m->dW);
  free(m->b);
  free(m->db);
  free(m->Z);
  free(m->dZ);
  free(m->ll);

  m->W = NULL;
  m->dW = NULL;
  m->b = NULL;
  m->db = NULL;
  m->Z = NULL;
  m->dZ = NULL;
  m->ll = NULL;
  m->P = 0;
  m->B = 0;
  m->L = 0;
  m->U = NULL;
}

void model_forward(model_t* m, float* X) {
  float** W = m->W;
  float** b = m->b;
  float** Z = m->Z;
  float* ll = m->ll;
  int P = m->P;
  int B = m->B;
  int L = m->L;
  const int* U = m->U;
  float* y = X + P - 1;

  cublasSetMatrix(U[0], B, sizeof(float), b[0], 0, Z[0], U[0]);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, U[0], B, P, scalar1, W[0],
      U[0], X, P, scalar1, Z[0], U[0]);
  for (int l = 1; l < L; ++l) {
    rectify(U[l - 1], B, Z[l - 1], U[l - 1]);
    cublasSetMatrix(U[l], B, sizeof(float), b[l], 0, Z[l], U[l]);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, U[l], B, U[l - 1], scalar1,
        W[l], U[l], Z[l - 1], U[l - 1], scalar1, Z[l], U[l]);
  }
  log_likelihood(B, y, P, Z[L - 1], U[L - 1], ll, 1);
}

void model_backward(model_t* m, float* y) {
  float** W = m->W;
  float** dW = m->dW;
  float** db = m->db;
  float** Z = m->Z;
  float** dZ = m->dZ;
  int P = m->P;
  int B = m->B;
  int L = m->L;
  const int* U = m->U;

  for (int l = L - 2; l >= 0; --l) {
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, U[l], B, U[l + 1], scalar1,
        W[l + 1], U[l + 1], dZ[l + 1], U[l + 1], scalar0, dZ[l], U[l]);
    rectify_grad(U[l], B, Z[l], U[l], dZ[l], U[l]);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, U[l], U[l - 1], B, scalar1,
        dZ[l], U[l], Z[l - 1], U[l - 1], scalar0, dW[l], U[l]);
    cublasSgemv(handle, CUBLAS_OP_N, U[l], B, scalar1, dZ[l], U[l], scalar1,
        0, scalar0, db[l], 1);
  }
  log_likelihood_grad(B, y, P, Z[L - 1], U[L - 1], dZ[L - 1], U[L - 1]);
}
