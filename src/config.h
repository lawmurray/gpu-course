#pragma once

/**
 * @def ENABLE_DOUBLE
 * 
 * Enable double precision? Set to 1 to use double precision, or 0 to use
 * single precision.
 */
#define ENABLE_DOUBLE 0

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

#if ENABLE_DOUBLE
#define float double
#undef cublasSger
#undef cublasSgemv
#undef cublasSgemm
#define cublasSger cublasDger
#define cublasSgemv cublasDgemv
#define cublasSgemm cublasDgemm
#endif
