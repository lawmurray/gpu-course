#include <train.h>

#include <cuda_runtime.h>
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

  d->X = X;
  d->N = N;
  d->P = P;
}

/**
 * Destroy and deallocate data.
 * 
 * @param d Data to destroy.
 */
void data_term(struct data_t* d) {
  free(d->X);
  
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
  /* initialize data */
  data_t d;
  data_init(&d, "bikeshare.csv");

  /* initialize model */
  int P = d.P;
  int B = 64;
  int L = 4;
  int U[] = {10, 64, 64, 64, 2};
  model_t m;
  model_init(&m, P, B, L, U);

  /* clean up */
  model_term(&m);
  data_term(&d);

  return 0;
}
