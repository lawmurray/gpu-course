#pragma once

#include <config.h>
#include <data.h>

/**
 * Model.
 */
typedef struct model_t {
  /**
   * All parameters.
   */
  real* theta;

  /**
   * Gradient of all parameters.
   */
  real* dtheta;

  /**
   * All activations.
   */
  real* A;

  /**
   * Gradient of all activations.
   */
  real* dA;

  /**
   * Pointers into theta for individual layer weight matrices.
   */
  real** W;

  /**
   * Pointers into dtheta for gradients of individual layer weight matrices.
   */
  real** dW;

  /**
   * Pointers into theta for individual layer bias vectors.
   */
  real** b;

  /**
   * Pointers into dtheta for gradients of individual layer bias vectors.
   */
  real** db;

  /**
   * Pointers into A for layer activations.
   */
  real** Z;

  /**
   * Pointers into dA for gradients of layer activations.
   */
  real** dZ;

  /**
   * Log-likelihoods.
   */
  real* l;

  /**
   * Sum of log-likelihoods.
  */
  real* ll;

  real* ones;

  /**
   * Layer widths.
   */
  const int* u;

  /**
   * Number of units.
   */
  int U;

  /**
   * Number of parameters.
   */
  int P;

  /**
   * Number of fields in data, including features and label.
   */
  int M;

  /**
   * Maximum number of records per batch.
   */
  int B;

  /**
   * Number of layers.
   */
  int L;

} model_t;

/**
 * Allocate and initialize model.
 *
 * @param m Model to initialize.
 * @param M Number of fields.
 * @param B Number of records per batch.
 * @param L Number of layers.
 * @param u Layer widths. Should be an array of size @p L.
 */
void model_init(model_t* model, const int M, const int B, const int L,
    const int* u);

/**
 * Destroy and deallocate model.
 * 
 * @param m Model to destroy.
 */
void model_term(model_t* m);

/**
 * Perform forward pass.
 * 
 * @param m Model.
 * @param X Batch.
 * @param B Batch size.
 */
void model_forward(model_t* m, real* X, const int B);

/**
 * Perform backward pass.
 * 
 * @param m Model.
 * @param X Batch.
 * @param B Batch size.
 */
void model_backward(model_t* m, real* X, const int B);

/**
 * Reset accumulated loss to zero.
 * 
 * @param m Model.
 */
void model_loss_clear(model_t* m);

/**
 * Accumulate loss after forward pass.
 * 
 * @param m Model.
 * @param X Batch.
 * @param B Batch size.
 */
void model_loss_accumulate(model_t* m, real* X, const int B);
