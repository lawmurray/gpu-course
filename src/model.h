/**
 * Model.
 */
typedef struct model_t {
  /**
   * Parameters.
   */
  float* theta;

  /**
   * Gradient of parameters.
   */
  float* dtheta;

  /**
   * Layer activations.
   */
  float* Z;

  /**
   * Gradient of layer activations.
   */
  float* dZ;

  /**
   * Log-likelihoods.
   */
  float* l;

  /**
   * Sum of log-likelihoods.
   */
  float* ll;

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
void model_forward(model_t* m, float* X, const int B);

/**
 * Perform backward pass.
 * 
 * @param m Model.
 * @param X Batch.
 * @param B Batch size.
 */
void model_backward(model_t* m, float* X, const int B);
