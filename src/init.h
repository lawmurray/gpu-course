#pragma once

#include <config.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

extern cublasHandle_t handle;
extern real* scalar0;
extern real* scalar1;

int cuda_init(const int seed);
void cuda_term();
