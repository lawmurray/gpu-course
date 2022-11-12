#include <data.h>
#include <model.h>
#include <optimizer.h>
#include <init.h>

#include <stdio.h>

int main(const int argc, const char *argv[]) {
  /* initialize */
  cuda_init();

  /* data */
  data_t d;
  data_init(&d, "bikeshare.csv");

  /* model */
  int M = d.M;
  int B = 64;
  int L = 3;
  int u[] = {256, 256, 2};
  model_t m;
  model_init(&m, M, B, L, u);

  /* optimizer */
  optimizer_t o;
  optimizer_init(&o, m.P, 1.0e-3f, 0.9f, 0.999f, 1.0e-8f);

  /* train */
  for (int epoch = 1; epoch <= 10; ++epoch) {
    printf("epoch %d ", epoch);
    for (int i = 0; i < d.N; ++i) {
      int b = (i + B < d.N) ? B : d.N - i;
      model_forward(&m, d.X + i, b);
      model_backward(&m, d.X + i, b);
      optimizer_step(&o, m.theta, m.dtheta);
    }
    printf("loss = %f\n", 0.0f);
  }

  /* clean up */
  optimizer_term(&o);
  model_term(&m);
  data_term(&d);
  cuda_term();

  return 0;
}
