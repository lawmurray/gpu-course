#include <data.h>
#include <model.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(const int argc, const char *argv[]) {
  /* initialize cuBLAS */
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

  /* initialize cuRAND */
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 0);

  /* initialize scalars */
  float value0 = 0.0f;
  float value1 = 1.0f;
  float* zero;
  float* one;
  cudaMalloc((void**)&zero, sizeof(float));
  cudaMalloc((void**)&one, sizeof(float));
  cublasSetVector(1, sizeof(float), &value0, 1, zero, 1);
  cublasSetVector(1, sizeof(float), &value1, 1, one, 1);

  /* initialize data */
  data_t d;
  data_init(&d, "bikeshare.csv");

  /* initialize model */
  int P = d.P;
  int B = 64;
  int L = 3;
  int U[] = {256, 256, 2};
  model_t m;
  model_init(&m, P, B, L, U);

  /* initialize parameters */
  curandGenerateNormal(gen, m.W[0], m.U[0]*m.P, 0.0f, sqrtf(1.0f/m.P));
  curandGenerateNormal(gen, m.b[0], m.U[0], 0.0f, 1.0f);
  for (int l = 1; l < L; ++l) {
    curandGenerateNormal(gen, m.W[l], m.U[l]*m.U[l - 1], 0.0f,
        sqrtf(1.0f/m.U[l - 1]));
    curandGenerateNormal(gen, m.b[l], m.U[l], 0.0f, 1.0f);
  }

  /* forward pass */
  cublasSetMatrix(m.U[0], m.B, sizeof(float), m.b[0], 0, m.Z[0], m.U[0]);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m.U[0], m.B, m.P, one,
      m.W[0], m.U[0], d.X, m.P, one, m.Z[0], m.U[0]);
  for (int l = 1; l < L; ++l) {
    //rectify(m.Z[l - 1], m.U[l - 1], m.B);
    cublasSetMatrix(m.U[l], m.B, sizeof(float), m.b[l], 0, m.Z[l], m.U[l]);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m.U[l], m.B, m.U[l - 1],
        one, m.W[l], m.U[l], m.Z[l - 1], m.U[l - 1], one, m.Z[l], m.U[l]);
  }

  /* backward pass */
  for (int l = L - 2; l >= 0; --l) {
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m.U[l], m.B, m.U[l + 1],
        one, m.W[l + 1], m.U[l + 1], m.dZ[l + 1], m.U[l + 1], zero, m.dZ[l],
        m.U[l]);
    //rectify_grad(m.dZ[l], m.Z[l], m.U[l], m.B);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m.U[l], m.U[l - 1], m.B,
        one, m.dZ[l], m.U[l], m.Z[l - 1], m.U[l - 1], zero, m.dW[l], m.U[l]);
    cublasSgemv(handle, CUBLAS_OP_N, m.U[l], m.B, one, m.dZ[l], m.U[l], one,
        0, zero, m.db[l], 1);
  }

  /* clean up */
  model_term(&m);
  data_term(&d);
  curandDestroyGenerator(gen);
  cublasDestroy(handle);
  cudaFree(one);
  cudaFree(zero);

  return 0;
}
