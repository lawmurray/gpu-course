#pragma once

/**
 * Data.
 */
typedef struct data_t {
  /**
   * Buffer.
   */
  float* X;

  /**
   * Log-likelihoods.
   */
  float* l;

  /**
   * Number of fields, including features and label.
   */
  int M;

  /**
   * Number of records.
   */
  int N;

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

/**
 * Shuffle the data set.
 */
void data_shuffle(data_t* d);
