#pragma once

#include <config.h>

/**
 * Adam optimizer.
 */
typedef struct optimizer_t {
  /**
   * First moment.
   */
  real* m;

  /**
   * Second moment.
   */
  real* v;

  /**
   * Number of parameters.
   */
  int P;

  /**
   * Number of steps taken.
   */
  int t;

  /**
   * Learning rate.
   */
  real gamma;

  /**
   * Momentum parameter.
   */
  real beta1;

  /**
   * Scaling parameter.
   */
  real beta2;

  /**
   * Small constant to improve numerical stability.
   */
  real epsilon;
} optimizer_t;

/**
 * Iniitialize Adam optimizer.
 */
void optimizer_init(optimizer_t* o, const int P, const real gamma,
    const real beta1, const real beta2, const real epsilon);

/**
 * Destroy Adam optimizer.
 */
void optimizer_term(optimizer_t* o);

/**
 * Take one step of Adam optimizer.
 */
void optimizer_step(optimizer_t* o, real* theta, real* dtheta);
