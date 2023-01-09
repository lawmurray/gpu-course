#include <init.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

cublasHandle_t handle;
real* scalar0;
real* scalar1;

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
  sharedMalloc((void**)&scalar0, sizeof(real));
  sharedMalloc((void**)&scalar1, sizeof(real));
  *scalar0 = 0.0f;
  *scalar1 = 1.0f;
}

void cuda_term() {
  cublasDestroy(handle);
  sharedFree(scalar0);
  sharedFree(scalar1);
}
