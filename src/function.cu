static __global__ void kernel_rectify(int U, int B, float* Z, int ldZ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < U && j < B) {
    Z[j*ldZ + i] = (Z[j*ldZ + i] <= 0.0f) ? 0.0f : Z[j*ldZ + i];
    // ^ ensures that NaN propagates rather than converts to zero
  }
}

static __global__ void kernel_rectify_grad(int U, int B, const float* Z, int ldZ,
    float* dZ, int lddZ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < U && j < B) {
    dZ[j*lddZ + i] = (Z[j*ldZ + i] <= 0.0f) ? 0.0f : dZ[j*lddZ + i];
  }
}

static __global__ void kernel_log_likelihood(int B, const float* y, int incy,
    const float* Z, int ldZ, float* l, int incl) {
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (j < B) {
    float mu = fabsf(Z[j*ldZ]);
    float sigma = fabsf(Z[1 + j*ldZ]) + 1.0e-6f;
    float z = (y[j*incy] - mu)/sigma;
    float sqrt2 = sqrtf(2.0f);
    float sqrt2pi = sqrt(2.0*3.14159265358979);
    float iota = mu/sigma;
    l[j*incl] = logf(2.0f/sqrt2pi) - 0.5f*z*z - logf(sigma) -
        logf(erfcf(-iota/sqrt2));
  }
}

static __global__ void kernel_log_likelihood_grad(int B, const float* y, int incy,
    const float* Z, int ldZ, float* dZ, int lddZ) {
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (j < B) {
    float mu = fabsf(Z[j*ldZ]);
    float sigma = fabsf(Z[1 + j*ldZ]) + 1.0e-6f;
    float z = (y[j*incy] - mu)/sigma;
    float sqrt2 = sqrtf(2.0f);
    float sqrtpi = sqrt(3.14159265358979);
    float iota = mu/sigma;
    float tmp = (2.0f/sqrtpi)*expf(-0.5f*iota*iota)/(sigma*sqrt2)/erfcf(-iota/sqrt2);
    float dmu = z/sigma - tmp;
    float dsigma = (z*z - 1.0f)/sigma - tmp*iota;

    dZ[j*lddZ] = (Z[j*ldZ] >= 0.0f) ? dmu : -dmu;
    dZ[1 + j*lddZ] = (Z[1 + j*ldZ] >= 0.0f) ? dsigma : -dsigma;
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
    theta[i] = theta[i] - gamma*mhat/(sqrtf(vhat) + epsilon);
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

extern "C" void log_likelihood(int B, const float* y, int incy,
    const float* Z, int ldZ, float* l, int incl) {
  dim3 block(1, 256);
  dim3 grid(1, (B + block.y - 1)/block.y);
  kernel_log_likelihood<<<grid,block>>>(B, y, incy, Z, ldZ, l, incl);
}

extern "C" void log_likelihood_grad(int B, const float* y, int incy,
    const float* Z, int ldZ, float* dZ, int lddZ) {
  dim3 block(1, 256);
  dim3 grid(1, (B + block.y - 1)/block.y);
  kernel_log_likelihood_grad<<<grid,block>>>(B, y, incy, Z, ldZ, dZ, lddZ);
}

extern "C" void adam(const int P, const int t, const float gamma,
    const float beta1, const float beta2, const float epsilon, float* m,
    float* v, float* theta, float* dtheta) {
  dim3 block(256);
  dim3 grid((P + block.x - 1)/block.x);
  kernel_adam<<<grid,block>>>(P, t, gamma, beta1, beta2, epsilon, m, v, theta,
      dtheta);
}
