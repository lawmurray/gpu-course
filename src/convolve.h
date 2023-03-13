#pragma once

void convolve_v0(const int m, const int n, const float* p, const int incp,
    const float* q, const int incq, float* r, const int incr);

void convolve_v1(const int m, const int n, const float* p, const int incp,
    const float* q, const int incq, float* r, const int incr);

void convolve_v2(const int m, const int n, const float* p, const int incp,
    const float* q, const int incq, float* r, const int incr);

void convolve_v3(const int m, const int n, const float* p, const int incp,
    const float* q, const int incq, float* r, const int incr);

void convolve_v4(const int m, const int n, const float* p, const int incp,
    const float* q, const int incq, float* r, const int incr);
