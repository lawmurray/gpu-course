#include <data.h>
#include <model.h>
#include <optimizer.h>
#include <init.h>

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
  optimizer_init(&o, 1.0e-3f, 0.9f, 0.999f, 1.0e-8f);

  /* clean up */
  optimizer_term(&o);
  model_term(&m);
  data_term(&d);
  cuda_term();

  return 0;
}
