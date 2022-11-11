static const float pi = 3.14159265358979f;

__global__ void kernel_rectify(int U, int B, float* Z, int ldZ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < U && j < B) {
    Z[j*ldZ + i] = (Z[j*ldZ + i] <= 0.0f) ? 0.0f : Z[j*ldZ + i];
    // ^ ensures that NaN propagates rather than converts to zero
  }
}

__global__ void kernel_rectify_grad(int U, int B, const float* Z, int ldZ,
    float* dZ, int lddZ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < U && j < B) {
    dZ[j*lddZ + i] = (Z[j*ldZ + i] <= 0.0f) ? 0.0f : dZ[j*lddZ + i];
  }
}

__global__ void kernel_log_likelihood(int B, const float* y, int incy,
    const float* Z, int ldZ, float* ll, int incll) {
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (j < B) {
    float mu = Z[j*ldZ];
    float sigma = fabsf(Z[j*ldZ + 1]);
    float z = (y[j*incy] - mu)/sigma;
    ll[j*incll] = logf(2.0f/sqrtf(2.0f*pi)) - 0.5f*z*z - logf(sigma) -
        logf(erfcf(z/sqrtf(2.0f)));
  }
}

__global__ void kernel_log_likelihood_grad(int B, const float* y, int incy,
    const float* Z, int ldZ, float* dZ, int lddZ) {
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (j < B) {
    float mu = Z[j*ldZ];
    float sigma = fabsf(Z[j*ldZ + 1]);
    float z = (y[j*incy] - mu)/sigma;
    float sqrt2 = sqrtf(2.0f);
    float tmp = (2.0f/sqrtf(pi))*expf(-0.5f*z*z)/erfcf(z/sqrt2);
    float dmu = z/sigma - tmp/(sigma*sqrt2);
    float dsigma = z/sigma - tmp*z/(sigma*sqrt2);

    dZ[j*lddZ] = dmu;
    dZ[j*lddZ + 1] = (Z[j*ldZ + 1] > 0.0f) ? dsigma : -dsigma;
  }
}

extern "C" void rectify(int U, int B, float* Z, int ldZ) {
  dim3 block(32, 16);
  dim3 grid((U + block.x - 1)/block.x, (B + block.y - 1)/block.y);
  kernel_rectify<<<grid,block>>>(U, B, Z, ldZ);
}

extern "C" void rectify_grad(int U, int B, const float* Z, int ldZ, float* dZ,
    int lddZ) {
  dim3 block(32, 16);
  dim3 grid((U + block.x - 1)/block.x, (B + block.y - 1)/block.y);
  kernel_rectify_grad<<<grid,block>>>(U, B, Z, ldZ, dZ, lddZ);
}

extern "C" void log_likelihood(int B, const float* y, int incy,
    const float* Z, int ldZ, float* l, int incl) {
  dim3 block(2, 256);
  dim3 grid(1, (B + block.y - 1)/block.y);
  kernel_log_likelihood<<<grid,block>>>(B, y, incy, Z, ldZ, l, incl);
}

extern "C" void log_likelihood_grad(int B, const float* y, int incy,
    const float* Z, int ldZ, float* dZ, int lddZ) {
  dim3 block(2, 256);
  dim3 grid(1, (B + block.y - 1)/block.y);
  kernel_log_likelihood<<<grid,block>>>(B, y, incy, Z, ldZ, dZ, lddZ);
}
