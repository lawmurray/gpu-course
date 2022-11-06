extern "C" void rectify(int U, int B, float* Z, int ldZ) {

}

extern "C" void rectify_grad(int U, int B, const float* Z, int ldZ, float* dZ,
    int lddZ) {

}

extern "C" void log_likelihood(int B, const float* y, int incy,
    const float* Z, int ldZ, float* logl) {
  
}

extern "C" void log_likelihood_grad(int B, const float* y, int incy,
    const float* Z, int ldZ, float* dZ, int lddZ) {
  
}
