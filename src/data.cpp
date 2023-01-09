#include <data.h>
#include <function.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void data_init(data_t* data, const char* file, const real split) {
  assert(0.0f <= split && split <= 1.0f);

  real* X = NULL;  // buffer to fill
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
      X = (real*)realloc(X, (N*M + m + 1)*sizeof(real));
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

  int N_train = split*N;
  int N_test = N - N_train;

  sharedMalloc((void**)&data->X_train, M*N_train*sizeof(real));
  cudaMemcpy(data->X_train, X, M*N_train*sizeof(real), cudaMemcpyDefault);
  sharedMalloc((void**)&data->X_test, M*N_test*sizeof(real));
  cudaMemcpy(data->X_test, X + N_train, M*N_test*sizeof(real),
      cudaMemcpyDefault);
  free(X);

  sharedMalloc((void**)&data->l_train, N_train*sizeof(real));
  sharedMalloc((void**)&data->l_test, N_test*sizeof(real));

  data->N_train = N_train;
  data->N_test = N_test;
  data->M = M;
}

void data_term(data_t* data) {
  sharedFree(data->X_train);
  sharedFree(data->X_test);
  sharedFree(data->l_train);
  sharedFree(data->l_test);

  data->X_train = NULL;
  data->X_test = NULL;
  data->l_train = NULL;
  data->l_test = NULL;
  data->N_train = 0;
  data->N_test = 0;
  data->M = 0;
}

void data_shuffle(data_t* data) {
  real* X = data->X_train;
  int N = data->N_train;
  int M = data->M;

  for (int i = 0; i < N - 1; ++i) {
    int j = i + (lrand48() % (N - i));
    for (int k = 0; k < M; ++k) {
      real tmp = X[M*i + k];
      X[M*i + k] = X[M*j + k];
      X[M*j + k] = tmp;
    }
  }
}
