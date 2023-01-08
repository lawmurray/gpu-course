static __global__ void kernel_rectify(int U, int B, float* Z, int ldZ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < U && j < B) {
    Z[i + j*ldZ] = (Z[i + j*ldZ] <= 0.0f) ? 0.0f : Z[i + j*ldZ];
    // ^ ensures that NaN propagates rather than converts to zero
  }
}

static __global__ void kernel_rectify_grad(int U, int B, const float* Z,
    int ldZ, float* dZ, int lddZ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < U && j < B) {
    dZ[i + j*lddZ] = (Z[i + j*ldZ] <= 0.0f) ? 0.0f : dZ[i + j*lddZ];
  }
}

static __global__ void kernel_squared_error(int B, const float* y, int incy,
    const float* m, int incm, float* l, int incl) {
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (j < B) {
    float z = y[j*incy] - m[j*incm];
    l[j*incl] = z*z;
  }
}

static __global__ void kernel_squared_error_grad(int B, const float* y,
    int incy, const float* m, int incm, float* d, int incd) {
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (j < B) {
    d[j*incd] = 2.0f*(y[j*incy] - m[j*incm]);
  }
}

static __global__ void kernel_adam(const int P, const int t, const float gamma,
    const float beta1, const float beta2, const float epsilon, float* m,
    float* v, float* theta, float* dtheta) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < P) {
    m[i] = beta1*m[i] + (1.0f - beta1)*-dtheta[i];
    v[i] = beta2*v[i] + (1.0f - beta2)*dtheta[i]*dtheta[i];
    float mhat = m[i]/(1.0f - powf(beta1, t));
    float vhat = v[i]/(1.0f - powf(beta2, t));
    if (vhat >= mhat*mhat) {
      theta[i] -= gamma*mhat/(sqrtf(vhat) + epsilon);
    }
  }
}

extern "C" void rectify(int U, int B, float* Z, int ldZ) {
  dim3 block(32, 8);
  dim3 grid((U + block.x - 1)/block.x, (B + block.y - 1)/block.y);
  kernel_rectify<<<grid,block>>>(U, B, Z, ldZ);
}

extern "C" void rectify_grad(int U, int B, const float* Z, int ldZ, float* dZ,
    int lddZ) {
  dim3 block(32, 8);
  dim3 grid((U + block.x - 1)/block.x, (B + block.y - 1)/block.y);
  kernel_rectify_grad<<<grid,block>>>(U, B, Z, ldZ, dZ, lddZ);
}

extern "C" void squared_error(int B, const float* y, int incy,
    const float* m, int incm, float* l, int incl) {
  dim3 block(1, 256);
  dim3 grid(1, (B + block.y - 1)/block.y);
  kernel_squared_error<<<grid,block>>>(B, y, incy, m, incm, l, incl);
}

extern "C" void squared_error_grad(int B, const float* y, int incy,
    const float* m, int incm, float* d, int incd) {
  dim3 block(1, 256);
  dim3 grid(1, (B + block.y - 1)/block.y);
  kernel_squared_error_grad<<<grid,block>>>(B, y, incy, m, incm, d, incd);
}

extern "C" void adam(const int P, const int t, const float gamma,
    const float beta1, const float beta2, const float epsilon, float* m,
    float* v, float* theta, float* dtheta) {
  dim3 block(256);
  dim3 grid((P + block.x - 1)/block.x);
  kernel_adam<<<grid,block>>>(P, t, gamma, beta1, beta2, epsilon, m, v,
      theta, dtheta);
}
