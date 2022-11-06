#include <data.h>
#include <model.h>
#include <init.h>

int main(const int argc, const char *argv[]) {
  /* initialize cuda */
  cuda_init();

  /* initialize data */
  data_t d;
  data_init(&d, "bikeshare.csv");

  /* initialize model */
  int P = d.P;
  int B = 64;
  int L = 3;
  int U[] = {256, 256, 2};
  model_t m;
  model_init(&m, P, B, L, U);

  model_forward(&m, d.X);
  model_backward(&m, d.X);

  /* clean up */
  model_term(&m);
  data_term(&d);
  cuda_term();

  return 0;
}
