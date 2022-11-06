#include <data.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void data_term(data_t* d) {
  cudaFree(d->X);
  
  d->X = NULL;
  d->N = 0;
  d->P = 0;
}
