#pragma once

#include <config.h>

/**
 * Rectify.
 * 
 * @param U Number of units.
 * @param B Batch size.
 * @param[out] Z Matrix.
 * @param ldZ Stride between columns of @p Z (lead).
 */
extern "C" void rectify(int U, int B, real* Z, int ldZ);

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
extern "C" void rectify_grad(int U, int B, const real* Z, int ldZ, real* dZ,
    int lddZ);

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
extern "C" void squared_error(int B, const real* y, int incy, const real* Z,
    int ldZ, real* l, int incl);

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
extern "C" void squared_error_grad(int B, const real* y, int incy,
    const real* m, int ldm, real* d, int incd);

/**
 * Take one step of an Adam optimizer.
 */
extern "C" void adam(const int P, const int t, const real gamma,
    const real beta1, const real beta2, const real epsilon, real* m, real* v,
    real* theta, real* dtheta);
