#include <init.h>

#include <cuda_runtime.h>

void cuda_init(const int seed) {
  /* initialize cuBLAS */
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

  /* initialize cuRAND */
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  /* initialize scalars */
  cudaMallocManaged((void**)&scalar0, sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&scalar1, sizeof(float), cudaMemAttachGlobal);
  *scalar0 = 0.0f;
  *scalar1 = 1.0f;
}

void cuda_term() {
  curandDestroyGenerator(gen);
  cublasDestroy(handle);
  cudaFree(scalar0);
  cudaFree(scalar1);
}
