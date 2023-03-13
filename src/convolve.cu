#include <config.h>
#include <cassert>

/* nvcc compiles *.cu files as C++, whereas the remainder of our source code
 * is in C; this ensures that we get C external linkage so that we can easily
 * call the functions in this file from C code, not just C++ code */
extern "C" {

__global__ void kernel_convolve_v0(const int m, const int n, const float* p,
    const int incp, const float* q, const int incq, float* r,
    const int incr) {
  /* element of r for which thread is responsible */
  int i = threadIdx.y + blockIdx.y*blockDim.y;

  if (i < m + n - 1) {
    float result = 0.0f;
    for (int j = 0; j < n; ++j) {
      if (0 <= i - j && i - j < m) {
        result += p[(i - j)*incp]*q[j*incq];
      }
    }
    r[i] = result;
  }
}

__global__ void kernel_convolve_v1(const int m, const int n, const float* p,
    const int incp, const float* q, const int incq, float* r,
    const int incr) {
  /* element of r for which thread is responsible */
  int i = threadIdx.y + blockIdx.y*blockDim.y;
  
  if (i < m) {
    float result1 = 0.0f, result2 = 0.0f;
    for (int j = 0; j < n; ++j) {
      if (0 <= i - j) {
        result1 += p[(i - j)*incp]*q[j*incq];
      } else {
        result2 += p[(m + i - j)*incp]*q[j*incq];
      }
    }
    r[i] = result1;
    if (i < n - 1) {
      r[i + m] = result2;
    }
  }
}

__global__ void kernel_convolve_v2(const int m, const int n, const float* p,
    const int incp, const float* q, const int incq, float* r,
    const int incr) {
  /* shared memory */
  extern __shared__ float q_shared[];

  /* element of r for which thread is responsible */
  int i = threadIdx.y + blockIdx.y*blockDim.y;

  float result1 = 0.0f, result2 = 0.0f;
  for (int base = 0; base < n; base += blockDim.y) {
    int j = threadIdx.y;
    __syncthreads();
    q_shared[j] = (base + j < n) ? q[(base + j)*incq] : 0.0f;
    __syncthreads();

    if (i < m) {
      for (j = 0; j < blockDim.y; ++j) {
        if (0 <= i - base - j) {
          result1 += p[(i - base - j)*incp]*q_shared[j];
        } else {
          result2 += p[(m + i - base - j)*incp]*q_shared[j];
        }
      }
    }
  }
  if (i < m) {
    r[i] = result1;
    if (i < n - 1) {
      r[i + m] = result2;
    }
  }
}

__global__ void kernel_convolve_v3(const int m, const int n, const float* p,
    const int incp, const float* q, const int incq, float* r,
    const int incr) {
  assert(blockDim.x == warpSize && gridDim.x == 1);

  /* shared memory */
  extern __shared__ float q_shared[];

  /* element of r for which warp is responsible */
  int i = threadIdx.y + blockIdx.y*blockDim.y;

  float result1 = 0.0f, result2 = 0.0f;
  for (int base = 0; base < n; base += warpSize*blockDim.y) {
    int j = threadIdx.y*warpSize + threadIdx.x;
    __syncthreads();
    q_shared[j] = (base + j < n) ? q[(base + j)*incq] : 0.0f;
    __syncthreads();

    if (i < m) {
      for (j = threadIdx.x; j < warpSize*blockDim.y; j += warpSize) {
        if (0 <= i - base - j) {
          result1 += p[(i - base - j)*incp]*q_shared[j];
        } else {
          result2 += p[(m + i - base - j)*incp]*q_shared[j];
        }
      }
    }
  }

  /* sum across threads of warp, using butterfly sum */
  for (int k = 16; k >= 1; k /= 2) {
    result1 += __shfl_xor_sync(0xffffffff, result1, k, warpSize);
    result2 += __shfl_xor_sync(0xffffffff, result2, k, warpSize);
  }

  /* set the final result, only first thread in each warp */
  if (i < m && threadIdx.x == 0) {
    r[i] = result1;
    if (i < n - 1) {
      r[i + m] = result2;
    }
  }
}

__global__ void kernel_convolve_v4(const int m, const int n, const float* p,
    const int incp, const float* q, const int incq, float* r,
    const int incr) {
  /* shared memory */
  extern __shared__ float shared[];
  float* q_shared = shared;
  float* p_shared = q_shared + 2*blockDim.y;  // permits -ve indices

  /* element of r for which thread is responsible */
  int base_i = blockIdx.y*blockDim.y;
  int base_j = 0;
  int i = threadIdx.y;
  int j = threadIdx.y;
  int k;
  float result1 = 0.0f, result2 = 0.0f;
  while (base_j < n) {
    __syncthreads();
    k = base_j + j;
    q_shared[j] = (k < n) ? q[k*incq] : 0.0f;

    k = base_i + i;
    if (k < m) {
      if (k >= 0) {
        p_shared[i] = p[k*incp];
      } else {
        p_shared[i] = p[m + k*incp];
      }
    } else {
      p_shared[i] = 0.0f;
    }

    k = base_i + i - blockDim.y;
    if (k < m) {
      if (k >= 0) {
        p_shared[i - blockDim.y] = p[k*incp];
      } else {
        p_shared[i - blockDim.y] = p[m + k*incp];
      }
    } else {
      p_shared[i - blockDim.y] = 0.0f;
    }
    __syncthreads();

    for (k = 0; k < blockDim.y; ++k) {
      if (0 <= base_i + i - k) {
        result1 += p_shared[i - k]*q_shared[k];
      } else {
        result2 += p_shared[i - k]*q_shared[k];        
      }
    }

    base_i -= blockDim.y;
    base_j += blockDim.y;
  }

  i = threadIdx.y + blockIdx.y*blockDim.y;
  if (i < m) {
    r[i] = result1;
    if (i < n - 1) {
      r[i + m] = result2;
    }
  }
}

void convolve_v0(const int m, const int n, const float* p, const int incp,
    const float* q, const int incq, float* r, const int incr) {
  const float *p1 = p, *q1 = q;
  int incp1 = incp, incq1 = incq;
  int m1 = m, n1 = n;
  if (n > m) {
    /* swap to put largest vector on the left */
    p1 = q;
    q1 = p;
    incp1 = incq;
    incq1 = incp;
    m1 = n;
    n1 = m;
  }
  dim3 block(1, BLOCK_SIZE);
  dim3 grid(1, (m1 + n1 - 1 + block.y - 1)/block.y);
  kernel_convolve_v0<<<grid,block>>>(m1, n1, p1, incp1, q1, incq1, r, incr);
}

void convolve_v1(const int m, const int n, const float* p, const int incp,
    const float* q, const int incq, float* r, const int incr) {
  const float *p1 = p, *q1 = q;
  int incp1 = incp, incq1 = incq;
  int m1 = m, n1 = n;
  if (n > m) {
    /* swap to put largest vector on the left */
    p1 = q;
    q1 = p;
    incp1 = incq;
    incq1 = incp;
    m1 = n;
    n1 = m;
  }
  dim3 block(1, BLOCK_SIZE);
  dim3 grid(1, (m1 + block.y - 1)/block.y);
  kernel_convolve_v1<<<grid,block>>>(m1, n1, p1, incp1, q1, incq1, r, incr);
}

void convolve_v2(const int m, const int n, const float* p, const int incp,
    const float* q, const int incq, float* r, const int incr) {
  const float *p1 = p, *q1 = q;
  int incp1 = incp, incq1 = incq;
  int m1 = m, n1 = n;
  if (n > m) {
    /* swap to put largest vector on the left */
    p1 = q;
    q1 = p;
    incp1 = incq;
    incq1 = incp;
    m1 = n;
    n1 = m;
  }
  dim3 block(1, BLOCK_SIZE);
  dim3 grid(1, (m1 + block.y - 1)/block.y);
  size_t shared = block.y*sizeof(float);
  kernel_convolve_v2<<<grid,block,shared>>>(m1, n1, p1, incp1, q1, incq1, r, incr);
}

void convolve_v3(const int m, const int n, const float* p, const int incp,
    const float* q, const int incq, float* r, const int incr) {
  const float *p1 = p, *q1 = q;
  int incp1 = incp, incq1 = incq;
  int m1 = m, n1 = n;
  if (n > m) {
    /* swap to put largest vector on the left */
    p1 = q;
    q1 = p;
    incp1 = incq;
    incq1 = incp;
    m1 = n;
    n1 = m;
  }
  dim3 block(32, 16);
  dim3 grid(1, (m1 + block.y - 1)/block.y);
  size_t shared = block.x*block.y*sizeof(float);
  kernel_convolve_v3<<<grid,block,shared>>>(m1, n1, p1, incp1, q1, incq1, r, incr);
}

void convolve_v4(const int m, const int n, const float* p, const int incp,
    const float* q, const int incq, float* r, const int incr) {
  const float *p1 = p, *q1 = q;
  int incp1 = incp, incq1 = incq;
  int m1 = m, n1 = n;
  if (n > m) {
    /* swap to put largest vector on the left */
    p1 = q;
    q1 = p;
    incp1 = incq;
    incq1 = incp;
    m1 = n;
    n1 = m;
  }
  dim3 block(1, BLOCK_SIZE);
  dim3 grid(1, (m1 + block.y - 1)/block.y);
  size_t shared = 3*block.y*sizeof(float);
  kernel_convolve_v4<<<grid,block,shared>>>(m1, n1, p1, incp1, q1, incq1, r, incr);
}

}
