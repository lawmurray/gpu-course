/**
 * Adam optimizer.
 */
typedef struct optimizer_t {
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

  /**
   * Number of steps taken.
   */
  int t;

  /**
   * First moment.
   */
  float* m;

  /**
   * Second moment.
   */
  float* v;
} optimizer_t;

/**
 * Iniitialize Adam optimizer.
 */
void optimizer_init(optimizer_t* o, const float gamma, const float beta1,
    const float beta2, const float epsilon);

/**
 * Destroy Adam optimizer.
 */
void optimizer_term(optimizer_t* o);
