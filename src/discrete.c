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

  struct timeval s, e;

  /* v0 */
  cudaDeviceSynchronize();
  gettimeofday(&s, NULL);
  convolve_v0(m, n, p, 1, q, 1, r, 1);
  cudaDeviceSynchronize();
  gettimeofday(&e, NULL);
  float elapsed_v0 = (e.tv_sec - s.tv_sec)*1.0e6 + (e.tv_usec - s.tv_usec);

  /* v1 */
  cudaDeviceSynchronize();
  gettimeofday(&s, NULL);
  convolve_v1(m, n, p, 1, q, 1, r, 1);
  cudaDeviceSynchronize();
  gettimeofday(&e, NULL);
  float elapsed_v1 = (e.tv_sec - s.tv_sec)*1.0e6 + (e.tv_usec - s.tv_usec);

  /* v2 */
  cudaDeviceSynchronize();
  gettimeofday(&s, NULL);
  convolve_v2(m, n, p, 1, q, 1, r, 1);
  cudaDeviceSynchronize();
  gettimeofday(&e, NULL);
  float elapsed_v2 = (e.tv_sec - s.tv_sec)*1.0e6 + (e.tv_usec - s.tv_usec);

  /* v3 */
  cudaDeviceSynchronize();
  gettimeofday(&s, NULL);
  convolve_v3(m, n, p, 1, q, 1, r, 1);
  cudaDeviceSynchronize();
  gettimeofday(&e, NULL);
  float elapsed_v3 = (e.tv_sec - s.tv_sec)*1.0e6 + (e.tv_usec - s.tv_usec);

  fprintf(stderr, "v0 %0.4fus\n", elapsed_v0);
  fprintf(stderr, "v1 %0.4fus\n", elapsed_v1);
  fprintf(stderr, "v2 %0.4fus\n", elapsed_v2);
  fprintf(stderr, "v3 %0.4fus\n", elapsed_v3);

  /* clean up */
  cuda_term();

  return 0;
}
