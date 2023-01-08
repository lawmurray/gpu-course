/**
 * @def ENABLE_SINGLE
 * 
 * Enable single precision?
 */
#define ENABLE_SINGLE 1

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
