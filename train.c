#include <train.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Allocate and initialize data.
 * 
 * @param d Data to initialize.
 * @param file Input CSV file.
 */
void data_init(data_t* d, const char* file) {
  float* X = NULL;  // buffer to fill
  int N = 0;  // number of lines
  int P = 0;  // number of values per line, computed from first
  
  FILE* fp = fopen(file, "r");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open file: %s\n", file);
    exit(1);
  }

  char* line = NULL;
  size_t len = 0;
  while (getline(&line, &len, fp) > 0) {
    int p = 0;
    char* token = strtok(line, ",");
    while (token) {
      X = (float*)realloc(X, (N*P + p + 1)*sizeof(float));
      X[N*P + p] = atof(token);
      ++p;
      token = strtok(NULL, ",");
    }
    ++N;

    /* set or check number of values */
    if (N == 1) {
      P = p;
    } else if (p != P) {
      fprintf(stderr, "Line %d has %d values, expecting %d\n", N, p, P);
    }
  }
  free(line);
  fclose(fp);

  cudaMalloc((void**)&d->X, N*P*sizeof(float));
  cudaMemcpy(d->X, X, N*P*sizeof(float), cudaMemcpyHostToDevice);
  free(X);

  d->N = N;
  d->P = P;
}

/**
 * Destroy and deallocate data.
 * 
 * @param d Data to destroy.
 */
void data_term(struct data_t* d) {
  cudaFree(d->X);
  
  d->X = NULL;
  d->N = 0;
  d->P = 0;
}

/**
 * Allocate and initialize model.
 *
 * @param m Model to initialize.
 * @param P Input size.
 * @param B Batch size.
 * @param L Number of layers.
 * @param U Layer widths. Should be an array of size @p L.
 */
void model_init(struct model_t* m, const int P, const int B, const int L,
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

/**
 * Destroy and deallocate model.
 * 
 * @param d Model to destroy.
 */
void model_term(struct model_t* m) {
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
