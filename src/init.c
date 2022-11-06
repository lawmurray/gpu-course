#include <init.h>

#include <cuda_runtime.h>

void cuda_init() {
  /* initialize cuBLAS */
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

  /* initialize cuRAND */
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 0);

  /* initialize scalars */
  float value0 = 0.0f;
  float value1 = 1.0f;
  cudaMalloc((void**)&scalar0, sizeof(float));
  cudaMalloc((void**)&scalar1, sizeof(float));
  cublasSetVector(1, sizeof(float), &scalar0, 1, &value0, 1);
  cublasSetVector(1, sizeof(float), &scalar1, 1, &value1, 1);
}

void cuda_term() {
  curandDestroyGenerator(gen);
  cublasDestroy(handle);
  cudaFree(scalar0);
  cudaFree(scalar1);
}
