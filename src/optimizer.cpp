#include <optimizer.h>
#include <init.h>
#include <function.h>

#include <stdlib.h>

void optimizer_init(optimizer_t* o, const int P, const real gamma,
    const real beta1, const real beta2, const real epsilon) {
  o->P = P;
  o->t = 0;
  o->gamma = gamma;
  o->beta1 = beta1;
  o->beta2 = beta2;
  o->epsilon = epsilon;

  sharedMalloc((void**)&o->m, P*sizeof(real));
  sharedMalloc((void**)&o->v, P*sizeof(real));
  cudaMemset(o->m, 0, P*sizeof(real));
  cudaMemset(o->v, 0, P*sizeof(real));
}

void optimizer_term(optimizer_t* o) {
  sharedFree(o->v);
  sharedFree(o->m);
  o->v = NULL;
  o->m = NULL;
  o->P = 0;
  o->t = 0;
}

void optimizer_step(optimizer_t* o, real* theta, real* dtheta) {
  ++o->t;
  adam(o->P, o->t, o->gamma, o->beta1, o->beta2, o->epsilon, o->m, o->v,
      theta, dtheta);
}
