#pragma once

#include <config.h>

/**
 * Data.
 */
typedef struct data_t {
  /**
   * Training data.
   */
  float* X_train;

  /**
   * Test data.
   */
  float* X_test;

  /**
   * Training data losses.
   */
  float* l_train;

  /**
   * Test data losses.
   */
  float* l_test;

  /**
   * Number of training records.
   */
  int N_train;

  /**
   * Number of test records.
   */
  int N_test;

  /**
   * Number of fields, including features and label.
   */
  int M;

} data_t;

/**
 * Allocate and initialize data.
 * 
 * @param d Data to initialize.
 * @param file Input CSV file.
 * @param split Proportion of records to randomly assign to training set.
 */
void data_init(data_t* d, const char* file, const float split);

/**
 * Destroy and deallocate data.
 * 
 * @param d Data to destroy.
 */
void data_term(data_t* d);

/**
 * Randomly shuffle training data.
 *
 * @param d Data to shuffle.
 */
void data_shuffle(data_t* d);
