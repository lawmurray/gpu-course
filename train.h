/**
 * Data.
 */
typedef struct data_t {
  /**
   * Buffer.
   */
  float* X;

  /**
   * Number of data points.
   */
  int N;

  /**
   * Number of features.
   */
  int P;
} data_t;

/**
 * Model.
 */
typedef struct model_t {
  /**
   * Layer weights.
   */
  float** W;

  /**
   * Layer activations.
   */
  float** Z;

  /**
   * Gradients of layer weights.
   */
  float** dW;

  /**
   * Gradients of layer activations.
   */
  float** dZ;

  /**
   * Observations.
   */
  float* y;

  /**
   * Layer widths.
   */
  const int* U;

  /**
   * Number of layers.
   */
  int L;

  /**
   * Batch size.
   */
  int B;
} struct_t;
