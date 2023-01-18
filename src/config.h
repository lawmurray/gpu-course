/**
 * @def ENABLE_SINGLE
 * 
 * Enable single precision? Set to 1 to use single precision, or 0 to use
 * double precision.
 */
#define ENABLE_SINGLE 1

/**
 * @def BLOCK_SIZE
 *
 * Preferred thread block size (maximum 1024).
 */
#define BLOCK_SIZE 256

/**
 * @def BATCH_SIZE
 *
 * Minibatch size.
 */
#define BATCH_SIZE 1024

/**
 * @def LAYER_WIDTH
 *
 * Hidden layer width.
 */
#define LAYER_WIDTH 256

/**
 * @def NEPOCHS
 *
 * Number of epochs.
 */
#define NEPOCHS 100

/**
 * Floating point type.
 */
#if ENABLE_SINGLE
typedef float real;
#else
typedef double real;
#endif

/*
 * CUBLAS function name mappings for single or double precision.
 */
#if ENABLE_SINGLE
#define ger cublasSger
#define gemv cublasSgemv
#define gemm cublasSgemm
#else
#define ger cublasDger
#define gemv cublasDgemv
#define gemm cublasDgemm
#endif
