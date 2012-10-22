#ifndef MTGP_PRINT_H
#define MTGP_PRINTU_H
/*
 * mtgp-print.h
 *
 * Some utility functions for Sample Programs
 *
 */
#include <stdint.h>
#include <inttypes.h>

void print_max_min(uint32_t data[], int size);
void print_float_array(const float array[], int size, int block);
void print_uint32_array(const uint32_t array[], int size, int block);
void print_double_array(const double array[], int size, int block);
void print_uint64_array(const uint64_t array[], int size, int block);

#endif
