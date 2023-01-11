#include <init.h>
#include <data.h>
#include <model.h>
#include <optimizer.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int main(const int argc, const char *argv[]) {
  /* initialize */
  int device = cuda_init(1);

  /* data */
  data_t d;
  data_init(&d, "bikeshare.csv", 0.8f);

  /* model */
  int B = 1024;
  int L = 3;
  int u[] = {1024, 1024, 1};
  model_t m;
  model_init(&m, d.M, B, L, u);

  /* optimizer */
  optimizer_t o;
  optimizer_init(&o, m.P, 1.0e-3f, 0.9f, 0.999f, 1.0e-8f);

  /* start timer */
  struct timeval s, e;
  cudaDeviceSynchronize();
  gettimeofday(&s, NULL);

  /* train */
  float* X_train = NULL;
  size_t bytes = d.M*d.N_train*sizeof(float);
  cudaMallocManaged((void**)&X_train, bytes, cudaMemAttachGlobal);
  cudaMemcpy(X_train, d.X_train, bytes, cudaMemcpyDefault);

  for (int epoch = 1; epoch <= 100; ++epoch) {
    cudaMemPrefetchAsync(X_train, bytes, device);
    cudaMemPrefetchAsync(d.X_train, bytes, cudaCpuDeviceId);

    /* iterate over training minibatches, performing gradient updates */
    for (int i = 0; i < d.N_train; i += B) {
      int b = (i + B <= d.N_train) ? B : d.N_train - i;
      model_forward(&m, X_train + i*d.M, b);
      model_backward(&m, X_train + i*d.M, b);
      optimizer_step(&o, m.theta, m.dtheta);
    }

    /* iterate over testing minibatches, accumulating loss */
    model_loss_clear(&m);
    for (int i = 0; i < d.N_test; i += B) {
      int b = (i + B <= d.N_test) ? B : d.N_test - i;
      model_forward(&m, d.X_test + i*d.M, b);
      model_loss_accumulate(&m, d.X_test + i*d.M, b);
    }

    /* shuffle data for next time */
    data_shuffle(&d);
    float* tmp = d.X_train;
    d.X_train = X_train;
    X_train = tmp;

    /* finalize loss and report progress */
    cudaDeviceSynchronize();
    real loss = *m.ll/d.N_test;    
    gettimeofday(&e, NULL);
    real elapsed = (e.tv_sec - s.tv_sec) + 1.0e-6f*(e.tv_usec - s.tv_usec);
    fprintf(stderr, "epoch %d: test loss %f, elapsed %0.4fs\n", epoch, loss,
        elapsed);
  }

  /* clean up */
  cudaFree(X_train);
  optimizer_term(&o);
  model_term(&m);
  data_term(&d);
  cuda_term();

  return 0;
}
