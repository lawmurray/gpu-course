/**
 * Model.
 */
typedef struct model_t {
  /**
   * Layer weights.
   */
  float** W;

  /**
   * Gradients of layer weights.
   */
  float** dW;

  /**
   * Layer biases.
   */
  float** b;

  /**
   * Gradients of biases.
   */
  float** db;

  /**
   * Layer activations.
   */
  float** Z;

  /**
   * Gradients of layer activations.
   */
  float** dZ;

  /**
   * Observations.
   */
  float* y;

  /**
   * Input size.
   */
  int P;

  /**
   * Batch size.
   */
  int B;

  /**
   * Number of layers.
   */
  int L;

  /**
   * Layer widths.
   */
  const int* U;

} model_t;

/**
 * Allocate and initialize model.
 *
 * @param m Model to initialize.
 * @param P Input size.
 * @param B Batch size.
 * @param L Number of layers.
 * @param U Layer widths. Should be an array of size @p L.
 */
void model_init(model_t* m, const int P, const int B, const int L,
    const int* U);

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
 * @param X Input.
 */
void model_forward(model_t* m, float* X);

/**
 * Perform backward pass.
 * 
 * @param m Model.
 * @param y Output.
 */
void model_backward(model_t* m, float* y);
