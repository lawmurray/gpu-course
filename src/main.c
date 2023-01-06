#include <data.h>
#include <model.h>
#include <optimizer.h>
#include <init.h>
#include <function.h>

#include <stdio.h>
#include <stdlib.h>

int main(const int argc, const char *argv[]) {
  /* initialize */
  cuda_init(1);

  /* data */
  data_t d;
  data_init(&d, "bikeshare.csv", 0.8f);

  /* model */
  int B = 4096;
  int L = 3;
  int u[] = {256, 256, 2};
  model_t m;
  model_init(&m, d.M, B, L, u);

  /* optimizer */
  optimizer_t o;
  optimizer_init(&o, m.P, 1.0e-3f, 0.9f, 0.999f, 1.0e-8f);

  /* train */
  float* X_train = NULL;
  size_t bytes = d.M*d.N_train*sizeof(float);
  cudaMalloc((void**)&X_train, bytes);
  cudaEvent_t event;
  cudaEventCreate(&event);

  for (int epoch = 1; epoch <= 100; ++epoch) {
    /* copy up shuffled data */
    cudaMemcpyAsync(X_train, d.X_train, bytes, cudaMemcpyDefault,
        cudaStreamDefault);
    cudaEventRecord(event, cudaStreamDefault);

    /* train */
    for (int i = 0; i < d.N_train; i += B) {
      int b = (i + B <= d.N_train) ? B : d.N_train - i;
      model_forward(&m, X_train + i*d.M, b);
      model_backward(&m, X_train + i*d.M, b);
      optimizer_step(&o, m.theta, m.dtheta);
    }

    /* test loss */
    for (int i = 0; i < d.N_test; i += B) {
      int b = (i + B <= d.N_test) ? B : d.N_test - i;
      model_forward(&m, d.X_test + i*d.M, b);
      model_predict(&m, d.X_test + i*d.M, b);
    }

    /* shuffle data for next time */
    cudaEventSynchronize(event);
    shuffle(d.M, d.N_train, d.X_train, d.M);

    /* report */
    cudaDeviceSynchronize();
    fprintf(stderr, "epcoh %d: test loss %f\n", epoch, -*m.ll/d.N_test);
    *m.ll = 0.0f;
  }

  /* clean up */
  cudaFree(X_train);
  cudaEventDestroy(event);
  optimizer_term(&o);
  model_term(&m);
  data_term(&d);
  cuda_term();

  return 0;
}
