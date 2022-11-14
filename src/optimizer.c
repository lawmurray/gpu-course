#include <optimizer.h>
#include <init.h>
#include <function.h>

#include <stdlib.h>

void optimizer_init(optimizer_t* o, const int P, const float gamma,
    const float beta1, const float beta2, const float epsilon) {
  o->P = P;
  o->t = 0;
  o->gamma = gamma;
  o->beta1 = beta1;
  o->beta2 = beta2;
  o->epsilon = epsilon;

  cudaMalloc((void**)&o->m, P*sizeof(float));
  cudaMalloc((void**)&o->v, P*sizeof(float));
  cudaMemset(o->m, 0, P*sizeof(float));
  cudaMemset(o->v, 0, P*sizeof(float));
}

void optimizer_term(optimizer_t* o) {
  cudaFree(o->v);
  cudaFree(o->m);
  o->v = NULL;
  o->m = NULL;
  o->P = 0;
  o->t = 0;
}

void optimizer_step(optimizer_t* o, float* theta, float* dtheta) {
  ++o->t;
  adam(o->P, o->t, o->gamma, o->beta1, o->beta2, o->epsilon, o->m, o->v,
      theta, dtheta);
}
