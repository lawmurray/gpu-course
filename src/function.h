#pragma once

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
 * Squared error loss.
 * 
 * @param B Batch size.
 * @param y Observations.
 * @param incy Stride between elements of @p y (increment).
 * @param m Predictions.
 * @param incm Stride between elements of @p m (increment).
 * @param[out] l Log-likelihoods.
 * @param incl Stride between elements of @p l (increment).
 */
void squared_error(int B, const float* y, int incy, const float* Z, int ldZ,
    float* l, int incl);

/**
 * Gradient of squared error loss.
 * 
 * @param B Batch size.
 * @param y Observations.
 * @param incy Stride between elements of @p y (increment).
 * @param m Predictions.
 * @param incm Stride between elements of @p m (increment).
 * @param[out] d Partial derivatives with respect to @p m.
 * @param incd Stride between elements of @p d (increment).
 */
void squared_error_grad(int B, const float* y, int incy, const float* m,
    int ldm, float* d, int incd);

/**
 * Take one step of an Adam optimizer.
 */
void adam(const int P, const int t, const float gamma, const float beta1,
    const float beta2, const float epsilon, float* m, float* v, float* theta,
    float* dtheta);
