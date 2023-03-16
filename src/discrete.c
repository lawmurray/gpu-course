#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <init.h>
#include <convolve.h>

int main(const int argc, const char *argv[]) {
  /* initialize */
  int device = cuda_init(1);

  const int m = 65536, n = 65536, reps = 100;
  float *p, *q, *r_v0, *r_v1, *r_v2, *r_v3, *r_v4;
  
  cudaMallocManaged((void**)&p, m*sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&q, n*sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&r_v0, (m + n - 1)*sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&r_v1, (m + n - 1)*sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&r_v2, (m + n - 1)*sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&r_v3, (m + n - 1)*sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged((void**)&r_v4, (m + n - 1)*sizeof(float), cudaMemAttachGlobal);

  float sum_p = 0.0f;
  for (int i = 0; i < m; ++i) {
    p[i] = drand48();
    sum_p += p[i];
  }
  for (int i = 0; i < m; ++i) {
    p[i] /= sum_p;
  }

  float sum_q = 0.0f;
  for (int i = 0; i < n; ++i) {
    q[i] = drand48();
    sum_q += q[i];
  }
  for (int i = 0; i < n; ++i) {
    q[i] /= sum_q;
  }

  cudaMemPrefetchAsync(p, m*sizeof(float), device, cudaStreamDefault);
  cudaMemPrefetchAsync(q, n*sizeof(float), device, cudaStreamDefault);
  cudaMemPrefetchAsync(r_v0, (m + n - 1)*sizeof(float), device, cudaStreamDefault);
  cudaMemPrefetchAsync(r_v1, (m + n - 1)*sizeof(float), device, cudaStreamDefault);
  cudaMemPrefetchAsync(r_v2, (m + n - 1)*sizeof(float), device, cudaStreamDefault);
  cudaMemPrefetchAsync(r_v3, (m + n - 1)*sizeof(float), device, cudaStreamDefault);
  cudaMemPrefetchAsync(r_v4, (m + n - 1)*sizeof(float), device, cudaStreamDefault);

  struct timeval s, e;

  /* v0 */
  cudaDeviceSynchronize();
  gettimeofday(&s, NULL);
  for (int k = 0; k < reps; ++k) {
    convolve_v0(m, n, p, 1, q, 1, r_v0, 1);
  }
  cudaDeviceSynchronize();
  gettimeofday(&e, NULL);
  int elapsed_v0 = (e.tv_sec - s.tv_sec)*1e6 + (e.tv_usec - s.tv_usec);

  /* v1 */
  cudaDeviceSynchronize();
  gettimeofday(&s, NULL);
  for (int k = 0; k < reps; ++k) {
    convolve_v1(m, n, p, 1, q, 1, r_v1, 1);
  }
  cudaDeviceSynchronize();
  gettimeofday(&e, NULL);
  int elapsed_v1 = (e.tv_sec - s.tv_sec)*1e6 + (e.tv_usec - s.tv_usec);

  /* v2 */
  cudaDeviceSynchronize();
  gettimeofday(&s, NULL);
  for (int k = 0; k < reps; ++k) {
    convolve_v2(m, n, p, 1, q, 1, r_v2, 1);
  }
  cudaDeviceSynchronize();
  gettimeofday(&e, NULL);
  int elapsed_v2 = (e.tv_sec - s.tv_sec)*1e6 + (e.tv_usec - s.tv_usec);

  /* v3 */
  cudaDeviceSynchronize();
  gettimeofday(&s, NULL);
  for (int k = 0; k < reps; ++k) {
    convolve_v3(m, n, p, 1, q, 1, r_v3, 1);
  }
  cudaDeviceSynchronize();
  gettimeofday(&e, NULL);
  int elapsed_v3 = (e.tv_sec - s.tv_sec)*1e6 + (e.tv_usec - s.tv_usec);

  /* v4 */
  cudaDeviceSynchronize();
  gettimeofday(&s, NULL);
  for (int k = 0; k < reps; ++k) {
    convolve_v4(m, n, p, 1, q, 1, r_v4, 1);
  }
  cudaDeviceSynchronize();
  gettimeofday(&e, NULL);
  int elapsed_v4 = (e.tv_sec - s.tv_sec)*1e6 + (e.tv_usec - s.tv_usec);

  /* sum all output vectors as diagnostic, each should sum to 1 */
  cudaDeviceSynchronize();
  double sum_v0 = 0.0, sum_v1 = 0.0, sum_v2 = 0.0, sum_v3 = 0.0, sum_v4 = 0.0;
  for (int i = 0; i < m + n - 1; ++i) {
    sum_v0 += r_v0[i];
    sum_v1 += r_v1[i];
    sum_v2 += r_v2[i];
    sum_v3 += r_v3[i];
    sum_v4 += r_v4[i];
  }

  /* output results */
  fprintf(stderr, "v0    %6dus    %0.8f\n", elapsed_v0/reps, sum_v0);
  fprintf(stderr, "v1    %6dus    %0.8f\n", elapsed_v1/reps, sum_v1);
  fprintf(stderr, "v2    %6dus    %0.8f\n", elapsed_v2/reps, sum_v2);
  fprintf(stderr, "v3    %6dus    %0.8f\n", elapsed_v3/reps, sum_v3);
  fprintf(stderr, "v4    %6dus    %0.8f\n", elapsed_v4/reps, sum_v4);

  /* clean up */
  cudaFree(p);
  cudaFree(q);
  cudaFree(r_v0);
  cudaFree(r_v1);
  cudaFree(r_v2);
  cudaFree(r_v3);
  cudaFree(r_v4);
  cuda_term();

  return 0;
}
