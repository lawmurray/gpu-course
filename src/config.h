/**
 * @def ENABLE_SINGLE
 * 
 * Enable single precision? Set to 1 to use single precision, or 0 to use
 * double precision.
 */
#define ENABLE_SINGLE 1

/**
 * @def ENABLE_MANAGED
 *
 * Enable managed memory"? Set to 1 to use managed memory, or 0 to use pinned
 * host memory.
 */
#define ENABLE_MANAGED 1

/**
 * @def BLOCK_SIZE
 *
 * Preferred thread block size (maximum 1024).
 */
#define BLOCK_SIZE 256

/**
 * @def LAYER_WIDTH
 *
 * Hidden layer width.
 */
#define LAYER_WIDTH 512

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

/*
 * Memory allocation function name mappings for managed to pinned host memory.
 */
#if ENABLE_MANAGED
#define sharedMalloc cudaMallocManaged
#define sharedFree cudaFree
#else
#define sharedMalloc cudaMallocHost
#define sharedFree cudaFreeHost
#endif
