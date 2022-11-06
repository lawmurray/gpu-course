#include <model.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void model_init(model_t* m, const int P, const int B, const int L,
    const int* U) {
  m->W = (float**)malloc(L*sizeof(float*));
  m->dW = (float**)malloc(L*sizeof(float*));
  m->b = (float**)malloc(L*sizeof(float*));
  m->db = (float**)malloc(L*sizeof(float*));
  m->Z = (float**)malloc(L*sizeof(float*));
  m->dZ = (float**)malloc(L*sizeof(float*));

  int prev = P;  // previous layer width
  for (int l = 0; l < L; ++l) {
    cudaMalloc((void**)&m->W[l], U[l]*prev*sizeof(float));
    cudaMalloc((void**)&m->dW[l], U[l]*prev*sizeof(float));
    cudaMalloc((void**)&m->b[l], U[l]*sizeof(float));
    cudaMalloc((void**)&m->db[l], U[l]*sizeof(float));
    cudaMalloc((void**)&m->Z[l], U[l]*B*sizeof(float));
    cudaMalloc((void**)&m->dZ[l], U[l]*B*sizeof(float));
    prev = U[l];
  }

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

  m->P = 0;
  m->B = 0;
  m->L = 0;
  m->U = NULL;
}
