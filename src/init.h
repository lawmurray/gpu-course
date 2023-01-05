#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

extern cublasHandle_t handle;
extern float* scalar0;
extern float* scalar1;

void cuda_init(const int seed);
void cuda_term();
