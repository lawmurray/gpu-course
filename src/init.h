#pragma once

#include <cublas_v2.h>
#include <curand.h>

cublasHandle_t handle;
curandGenerator_t gen;

float* scalar0;
float* scalar1;

void cuda_init(const int seed);
void cuda_term();
