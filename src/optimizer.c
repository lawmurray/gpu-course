#include <optimizer.h>

void optimizer_init(optimizer_t* o, const float gamma, const float beta1,
    const float beta2, const float epsilon) {
  o->gamma = gamma;
  o->beta1 = beta1;
  o->beta2 = beta2;
  o->epsilon = epsilon;
}

void optimizer_term(optimizer_t* o) {
  
}
