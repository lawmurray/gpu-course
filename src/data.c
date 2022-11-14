#include <data.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void data_init(data_t* data, const char* file) {
  float* X = NULL;  // buffer to fill
  int M = 0;  // number of fields, computed from first record
  int N = 0;  // number of records
  
  FILE* fp = fopen(file, "r");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open file: %s\n", file);
    exit(1);
  }

  char* line = NULL;
  size_t len = 0;
  while (getline(&line, &len, fp) > 0) {
    int m = 0;
    char* token = strtok(line, ",");
    while (token) {
      X = (float*)realloc(X, (N*M + m + 1)*sizeof(float));
      X[M*N + m] = atof(token);
      ++m;
      token = strtok(NULL, ",");
    }
    ++N;

    /* set or check number of values */
    if (N == 1) {
      M = m;
    } else if (m != M) {
      fprintf(stderr, "Line %d has %d values, expecting %d\n", N, m, M);
    }
  }
  free(line);
  fclose(fp);

  cudaMallocManaged((void**)&data->X, M*N*sizeof(float), cudaMemAttachGlobal);
  cudaMemcpy(data->X, X, M*N*sizeof(float), cudaMemcpyDefault);
  free(X);
  cudaMallocManaged((void**)&data->l, N*sizeof(float), cudaMemAttachGlobal);

  data->M = M;
  data->N = N;
}

void data_term(data_t* data) {
  cudaFree(data->l);
  cudaFree(data->X);
  data->X = NULL;
  data->M = 0;
  data->N = 0;
}

void data_shuffle(data_t* data) {
  cudaDeviceSynchronize();
  int N = data->N;
  int M = data->M;
  float* X = data->X;
  for (int i = 0; i < N - 1; ++i) {
    int j = i + (lrand48() % (N - i));
    for (int k = 0; k < M; ++k) {
      float tmp = X[M*i + k];
      X[M*i + k] = X[M*j + k];
      X[M*j + k] = tmp;
    }
  }
}
