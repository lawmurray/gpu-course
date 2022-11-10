/**
 * Rectify.
 * 
 * @param U Number of units.
 * @param B Batch size.
 * @param[out] Z Matrix.
 * @param ldZ Stride between columns of @p Z (lead).
 */
void rectify(int U, int B, float* Z, int ldZ);

/**
 * Gradient of rectify.
 * 
 * @param U Number of units.
 * @param B Batch size.
 * @param Z Matrix where rows index units, columns index batch members.
 * @param ldZ Stride between columns of @p Z (lead).
 * @param[in,out] dZ On input, the partial derivative with respect to $Z^+$,
 * on output the partial derivative with respect to $Z$.
 * @param lddZ Stride between columns of @p dZ (lead).
 */
void rectify_grad(int U, int B, const float* Z, int ldZ, float* dZ, int lddZ);

/**
 * Log-likelihood.
 * 
 * @param B Batch size.
 * @param y Observations.
 * @param incy Stride between elements of @p y (increment).
 * @param Z Matrix where rows index units, columns index batch members.
 * @param ldZ Stride between columns of @p Z (lead).
 * @param[out] ll Log-likelihood.
 * @param incll Stride between elements of @p ll (increment).
 */
void log_likelihood(int B, const float* y, int incy, const float* Z, int ldZ,
    float* ll, int incll);

/**
 * Gradient of log-likelihood.
 * 
 * @param B Batch size.
 * @param y Observations.
 * @param incy Stride between elements of @p y (increment).
 * @param Z Matrix where columns index batch members, and there are two rows,
 * to be converted to mean and variance.
 * @param ldZ Stride between columns of @p Z (lead).
 * @param[out] dZ Partial derivatives with respect to @p Z.
 * @param lddZ Stride between columns of @p dZ (lead).
 */
void log_likelihood_grad(int B, const float* y, int incy, const float* Z,
    int ldZ, float* dZ, int lddZ);
