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

  cudaMalloc((void**)&data->X, M*N*sizeof(float));
  cudaMemcpy(data->X, X, M*N*sizeof(float), cudaMemcpyHostToDevice);
  free(X);

  data->M = M;
  data->N = N;
}

void data_term(data_t* data) {
  cudaFree(data->X);
  data->X = NULL;
  data->M = 0;
  data->N = 0;
}
