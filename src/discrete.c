#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <init.h>
#include <convolve.h>

int main(const int argc, const char *argv[]) {
  /* initialize */
  int device = cuda_init(1);

  const int m = 65536;
  const int n = 65536;
  float* p;
  float* q;
  float* r;
  
  cudaMallocManaged((void**)&p, m*sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&q, n*sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&r, (m + n - 1)*sizeof(float), cudaMemAttachGlobal);

  float sum = 0.0f;
  for (int i = 0; i < m; ++i) {
    p[i] = drand48();
    sum += p[i];
  }
  for (int i = 0; i < m; ++i) {
    p[i] /= sum;
  }
  for (int i = 0; i < n; ++i) {
    q[i] = drand48();
    sum += q[i];
  }
  for (int i = 0; i < n; ++i) {
    q[i] /= sum;
  }

  cudaMemPrefetchAsync(p, m*sizeof(float), device, cudaStreamDefault);
  cudaMemPrefetchAsync(q, n*sizeof(float), device, cudaStreamDefault);
  cudaMemPrefetchAsync(r, (m + n - 1)*sizeof(float), device, cudaStreamDefault);

  /* start timer */
  struct timeval s, e;
  cudaDeviceSynchronize();
  gettimeofday(&s, NULL);

  /* end timer */
  gettimeofday(&e, NULL);
  float elapsed = (e.tv_sec - s.tv_sec) + 1.0e-6f*(e.tv_usec - s.tv_usec);
  fprintf(stderr, "elapsed %0.4fs\n", elapsed);

  /* clean up */
  cuda_term();

  return 0;
}
