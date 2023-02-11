#pragma once

#include <config.h>

extern cublasHandle_t handle;
extern float* scalar0;
extern float* scalar1;

int cuda_init(const int seed);
void cuda_term();
