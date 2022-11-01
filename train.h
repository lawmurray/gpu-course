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
