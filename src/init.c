#include <init.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

cublasHandle_t handle;
float* scalar0;
float* scalar1;

void cuda_init(const int seed) {
  int device = 0, value = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&value, cudaDevAttrConcurrentManagedAccess, device);
  if (!value) {
    fprintf(stderr, "warning: your device does not support concurrent managed"
        " memory, some exercises may not work as intended.\n");
  }

  /* seed random number generator */
  srand48(seed);

  /* initialize cuBLAS */
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

  /* initialize scalars */
  cudaMallocManaged((void**)&scalar0, sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&scalar1, sizeof(float), cudaMemAttachGlobal);
  *scalar0 = 0.0f;
  *scalar1 = 1.0f;
}

void cuda_term() {
  cublasDestroy(handle);
  cudaFree(scalar0);
  cudaFree(scalar1);
}
