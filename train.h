/**
 * Data.
 */
struct data_t {
  /**
   * Inputs.
   */
  float* X = NULL;

  /**
   * Outputs.
   */
  float* y = NULL;

  /**
   * Number of data points.
   */
  int N = 0;

  /**
   * Number of features.
   */
  int P = 0;
};

/**
 * Model.
 */
struct model_t {
  /**
   * Layer weights.
   */
  float** W = NULL;

  /**
   * Layer activations.
   */
  float** Z = NULL;

  /**
   * Gradients of layer weights.
   */
  float** dW = NULL;

  /**
   * Gradients of layer activations.
   */
  float** dZ = NULL;

  /**
   * Observations.
   */
  float* y = NULL;

  /**
   * Layer widths.
   */
  const int* U = NULL;

  /**
   * Number of layers.
   */
  int L = 0;

  /**
   * Batch size.
   */
  int B = 0;
};
