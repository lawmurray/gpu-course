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
 * Allocate and initialize data.
 * 
 * @param d Data to initialize.
 * @param file Input CSV file.
 */
void data_init(data_t* d, const char* file);

/**
 * Destroy and deallocate data.
 * 
 * @param d Data to destroy.
 */
void data_term(data_t* d);
