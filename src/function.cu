#include <config.h>

/* nvcc compiles *.cu files as C++, whereas the remainder of our source code
 * is in C; this ensures that we get C external linkage so that we can easily
 * call the functions in this file from C code, not just C++ code */
extern "C" {

__global__ void kernel_rectify(int U, int B, real* Z, int ldZ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < U && j < B) {
    Z[i + j*ldZ] = (Z[i + j*ldZ] <= 0.0f) ? 0.0f : Z[i + j*ldZ];
    // ^ ensures that NaN propagates rather than converts to zero
  }
}

__global__ void kernel_rectify_grad(int U, int B, const real* Z, int ldZ,
    real* dZ, int lddZ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < U && j < B) {
    dZ[i + j*lddZ] = (Z[i + j*ldZ] <= 0.0f) ? 0.0f : dZ[i + j*lddZ];
  }
}

__global__ void kernel_squared_error(int B, const real* y, int incy,
    const real* m, int incm, real* l, int incl) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < B) {
    real z = y[i*incy] - m[i*incm];
    l[i*incl] = z*z;
  }
}

__global__ void kernel_squared_error_grad(int B, const real* y, int incy,
    const real* m, int incm, real* d, int incd) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < B) {
    d[i*incd] = 2.0f*(y[i*incy] - m[i*incm]);
  }
}

__global__ void kernel_adam(const int P, const int t, const real gamma,
    const real beta1, const real beta2, const real epsilon, real* m,
    real* v, real* theta, real* dtheta) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < P) {
    m[i] = beta1*m[i] + (1.0f - beta1)*-dtheta[i];
    v[i] = beta2*v[i] + (1.0f - beta2)*dtheta[i]*dtheta[i];
    real mhat = m[i]/(1.0f - powf(beta1, t));
    real vhat = v[i]/(1.0f - powf(beta2, t));
    theta[i] -= gamma*mhat/(sqrtf(vhat) + epsilon);
  }
}

void rectify(int U, int B, real* Z, int ldZ) {
  dim3 block(BLOCK_SIZE, 1);
  dim3 grid((U + block.x - 1)/block.x, (B + block.y - 1)/block.y);
  kernel_rectify<<<grid,block>>>(U, B, Z, ldZ);
}

void rectify_grad(int U, int B, const real* Z, int ldZ, real* dZ, int lddZ) {
  dim3 block(BLOCK_SIZE, 1);
  dim3 grid((U + block.x - 1)/block.x, (B + block.y - 1)/block.y);
  kernel_rectify_grad<<<grid,block>>>(U, B, Z, ldZ, dZ, lddZ);
}

void squared_error(int B, const real* y, int incy, const real* m, int incm,
    real* l, int incl) {
  dim3 block(BLOCK_SIZE);
  dim3 grid((B + block.y - 1)/block.y);
  kernel_squared_error<<<grid,block>>>(B, y, incy, m, incm, l, incl);
}

void squared_error_grad(int B, const real* y, int incy, const real* m,
    int incm, real* d, int incd) {
  dim3 block(BLOCK_SIZE);
  dim3 grid((B + block.y - 1)/block.y);
  kernel_squared_error_grad<<<grid,block>>>(B, y, incy, m, incm, d, incd);
}

void adam(const int P, const int t, const real gamma, const real beta1,
    const real beta2, const real epsilon, real* m, real* v, real* theta,
    real* dtheta) {
  dim3 block(BLOCK_SIZE);
  dim3 grid((P + block.x - 1)/block.x);
  kernel_adam<<<grid,block>>>(P, t, gamma, beta1, beta2, epsilon, m, v,
      theta, dtheta);
}

}
