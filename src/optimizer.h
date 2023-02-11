#pragma once

#include <config.h>

/**
 * Adam optimizer.
 */
typedef struct optimizer_t {
  /**
   * First moment.
   */
  float* m;

  /**
   * Second moment.
   */
  float* v;

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
  float gamma;

  /**
   * Momentum parameter.
   */
  float beta1;

  /**
   * Scaling parameter.
   */
  float beta2;

  /**
   * Small constant to improve numerical stability.
   */
  float epsilon;
} optimizer_t;

/**
 * Iniitialize Adam optimizer.
 */
void optimizer_init(optimizer_t* o, const int P, const float gamma,
    const float beta1, const float beta2, const float epsilon);

/**
 * Destroy Adam optimizer.
 */
void optimizer_term(optimizer_t* o);

/**
 * Take one step of Adam optimizer.
 */
void optimizer_step(optimizer_t* o, float* theta, float* dtheta);
