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
  
  /* sum across elements in each thread */
  float result = 0.0f;
  for (int j = 0; j < n; ++j) {
    if (0 <= i - j && i - j < m) {
      result += p[(i - j)*incp]*q[j*incq];
    }
  }
  
  /* write element */
  r[i] = result;
}

__global__ void kernel_convolve_v1(const int m, const int n, const float* p,
    const int incp, const float* q, const int incq, float* r,
    const int incr) {
  /* first element of r for which thread is responsible */
  int i = threadIdx.y + blockIdx.y*blockDim.y;
  
  /* sum across elements in each thread */
  float result1 = 0.0f, result2 = 0.0f;
  for (int j = 0; j < n; ++j) {
    if (0 <= i - j) {
      result1 += p[(i - j)*incp]*q[j*incq];
    } else {
      result2 += p[(m + i - j)*incp]*q[j*incq];
    }
  }
  
  /* write elements */
  r[i] = result1;
  if (i < m - 1) {
    r[i + m] = result2;
  }
}

__global__ void kernel_convolve_v2(const int m, const int n, const float* p,
    const int incp, const float* q, const int incq, float* r,
    const int incr) {
  __shared__ float* q_shared;
  int i = threadIdx.y + blockIdx.y*blockDim.y;
  float result1 = 0.0f, result2 = 0.0f;
  
  for (int base = 0; base < n; base += blockDim.y) {
    /* read the next batch into shared memory */
    int j = threadIdx.y;
    q_shared[j] = base + j < n ? q[(base + j)*incq] : 0.0f;
    __syncthreads();

    /* sum across elements in each thread  */
    for (j = 0; j < blockDim.y; ++j) {
      if (0 <= i - j) {
        result1 += p[(base + i - j)*incp]*q_shared[j];
      } else {
        result2 += p[(base + m + i - j)*incp]*q_shared[j];
      }
    }
  }

  /* set the final result */
  r[i] = result1;
  if (i < m - 1) {
    r[i + m] = result2;
  }
}

__global__ void kernel_convolve_v3(const int m, const int n, const float* p,
    const int incp, const float* q, const int incq, float* r,
    const int incr) {
  assert(blockDim.x == warpSize && gridDim.x == 1);

  __shared__ float* q_shared;
  int i = blockIdx.x*blockDim.x;
  float result1 = 0.0f, result2 = 0.0f;
  
  for (int base = 0; base < n; base += warpSize*blockDim.y) {
    /* read the next batch into shared memory */
    int j = threadIdx.y*warpSize + threadIdx.x;
    q_shared[j] = base + j < n ? q[(base + j)*incq] : 0.0f;
    __syncthreads();

    /* sum across elements in each thread  */
    for (j = threadIdx.x; j < warpSize*blockDim.y; j += warpSize) {
      /* read the next batch into shared memory */
      if (0 <= i - j) {
        result1 += p[(base + i - j)*incp]*q_shared[j];
      } else {
        result2 += p[(base + m + i - j)*incp]*q_shared[j];
      }
    }
  }

  /* sum across threads of warp, using butterfly sum */
  for (int k = 16; k >= 1; k /= 2) {
    result1 += __shfl_xor_sync(0xffffffff, result1, k, warpSize);
    result2 += __shfl_xor_sync(0xffffffff, result2, k, warpSize);
  }

  /* set the final result, only first thread in each warp */
  if (threadIdx.x == 0) {
    r[i] = result1;
    if (i < m - 1) {
      r[i + m] = result2;
    }
  }
}

void convolve_v0(const int m, const int n, const float* p, const int incp,
    const float* q, const int incq, float* r, const int incr) {
  dim3 block(1, BLOCK_SIZE);
  dim3 grid(1, (m + n - 1 + block.y - 1)/block.y);
  kernel_convolve_v0<<<grid,block>>>(m, n, p, incp, q, incq, r, incr);
}

void convolve_v1(const int m, const int n, const float* p, const int incp,
    const float* q, const int incq, float* r, const int incr) {
  dim3 block(1, BLOCK_SIZE);
  dim3 grid(1, (m + block.y - 1)/block.y);
  kernel_convolve_v1<<<grid,block>>>(m, n, p, incp, q, incq, r, incr);
}

void convolve_v2(const int m, const int n, const float* p, const int incp,
    const float* q, const int incq, float* r, const int incr) {
  dim3 block(1, BLOCK_SIZE);
  dim3 grid(1, (m + block.y - 1)/block.y);
  size_t shared = block.y*sizeof(float);
  kernel_convolve_v2<<<grid,block,shared>>>(m, n, p, incp, q, incq, r, incr);
}

void convolve_v3(const int m, const int n, const float* p, const int incp,
    const float* q, const int incq, float* r, const int incr) {
  dim3 block(32, BLOCK_SIZE);
  dim3 grid(1, (m + block.y - 1)/block.y);
  size_t shared = block.y*sizeof(float);
  kernel_convolve_v3<<<grid,block,shared>>>(m, n, p, incp, q, incq, r, incr);
}

}
