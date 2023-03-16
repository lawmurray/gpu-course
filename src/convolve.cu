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
  int i = threadIdx.x + blockIdx.x*blockDim.x;

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
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  
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
  extern __shared__ float q_shared[];

  /* element of r for which thread is responsible */
  int i = threadIdx.x + blockIdx.x*blockDim.x;

  float result1 = 0.0f, result2 = 0.0f;
  for (int base_j = 0; base_j < n; base_j += blockDim.x) {
    int j = threadIdx.x;
    __syncthreads();
    q_shared[j] = (base_j + j < n) ? q[(base_j + j)*incq] : 0.0f;
    __syncthreads();

    for (j = 0; j < blockDim.x; ++j) {
      if (0 <= i - base_j - j) {
        result1 += p[(i - base_j - j)*incp]*q_shared[j];
      } else {
        result2 += p[(m + i - base_j - j)*incp]*q_shared[j];
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
  extern __shared__ float shared[];
  float* q_shared = shared;
  float* p_shared = q_shared + 2*blockDim.x;  // permits -ve indices

  int i = threadIdx.x;  
  int j = threadIdx.x;
  float result1 = 0.0f, result2 = 0.0f;
  for (int base_i = blockIdx.x*blockDim.x, base_j = 0; base_j < n;
      base_i -= blockDim.x, base_j += blockDim.x) {
    __syncthreads();
    q_shared[j] = (base_j + j < n) ? q[(base_j + j)*incq] : 0.0f;
    p_shared[i] = p[((base_i + i + m) % m)*incp];
    p_shared[i - blockDim.x] = p[((base_i + i - blockDim.x + m) % m)*incp];
    __syncthreads();

    for (int k = 0; k < blockDim.x; ++k) {
      if (0 <= base_i + i - k) {
        result1 += p_shared[i - k]*q_shared[k];
      } else {
        result2 += p_shared[i - k]*q_shared[k];        
      }
    }
  }

  /* element of r for which thread is responsible */
  i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < m) {
    r[i] = result1;
    if (i < n - 1) {
      r[i + m] = result2;
    }
  }
}

__global__ void kernel_convolve_v4(const int m, const int n, const float* p,
    const int incp, const float* q, const int incq, float* r,
    const int incr) {
  assert(blockDim.x == warpSize && gridDim.x == 1);

  extern __shared__ float shared[];
  float* q_shared = shared;
  float* p_shared = q_shared + 2*warpSize*blockDim.y;  // permits -ve indices

  int i = threadIdx.y*warpSize + threadIdx.x;
  int j = threadIdx.y*warpSize + threadIdx.x;
  int l = threadIdx.y;
  float result1 = 0.0f, result2 = 0.0f;
  for (int base_i = blockIdx.y*blockDim.y, base_j = 0; base_j < n;
      base_i -= warpSize*blockDim.y, base_j += warpSize*blockDim.y) {
    __syncthreads();
    q_shared[j] = (base_j + j < n) ? q[(base_j + j)*incq] : 0.0f;
    p_shared[i] = p[((base_i + i + m) % m)*incp];
    p_shared[i - warpSize*blockDim.y] = p[((base_i + i - warpSize*blockDim.y + m) % m)*incp];
    __syncthreads();

    for (int k = threadIdx.x; k < warpSize*blockDim.y; k += warpSize) {
      if (0 <= base_i + l - k) {
        result1 += p_shared[l - k]*q_shared[k];
      } else {
        result2 += p_shared[l - k]*q_shared[k];        
      }
    }
  }

  /* sum across threads of warp, using butterfly sum */
  for (int k = 16; k >= 1; k /= 2) {
    result1 += __shfl_xor_sync(0xffffffff, result1, k, warpSize);
    result2 += __shfl_xor_sync(0xffffffff, result2, k, warpSize);
  }

  /* element of r for which warp is responsible */
  i = threadIdx.y + blockIdx.y*blockDim.y;

  /* first thread in each warp sets the final result */
  if (i < m && threadIdx.x == 0) {
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
  dim3 block(BLOCK_SIZE);
  dim3 grid((m1 + n1 - 1 + block.x - 1)/block.x);
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
  dim3 block(BLOCK_SIZE);
  dim3 grid((m1 + block.x - 1)/block.x);
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
  dim3 block(BLOCK_SIZE);
  dim3 grid((m1 + block.x - 1)/block.x);
  size_t shared = block.x*sizeof(float);
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
  dim3 block(BLOCK_SIZE);
  dim3 grid((m1 + block.x - 1)/block.x);
  size_t shared = 3*block.x*sizeof(float);
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
  dim3 block(32, 32);
  dim3 grid(1, (m1 + block.y - 1)/block.y);
  size_t shared = 3*block.x*block.y*sizeof(float);
  kernel_convolve_v4<<<grid,block,shared>>>(m1, n1, p1, incp1, q1, incq1, r, incr);
}

}
