#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

cublasHandle_t handle;

float* scalar0;
float* scalar1;

void cuda_init(const int seed);
void cuda_term();
