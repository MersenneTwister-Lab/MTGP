/*
 * mtgp-print.cpp
 *
 * Some utility functions for Sample Programs
 *
 */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <mtgp-print.h>

/**
 * This function is used to compare the outputs with C program's.
 * @param array data to be printed.
 * @param size size of array.
 * @param block number of blocks.
 */
void print_float_array(const float array[], int size, int block) {
    int b = size / block;
    int j;
    int i;

    for (j = 0; j < 5; j += 5) {
	printf("%.10f %.10f %.10f %.10f %.10f\n",
	       array[j], array[j + 1],
	       array[j + 2], array[j + 3], array[j + 4]);
    }
    for (i = 1; i < block; i++) {
	for (j = -5; j < 5; j += 5) {
	    printf("%.10f %.10f %.10f %.10f %.10f\n",
		   array[b * i + j],
		   array[b * i + j + 1],
		   array[b * i + j + 2],
		   array[b * i + j + 3],
		   array[b * i + j + 4]);
	}
    }
    for (j = -5; j < 0; j += 5) {
	printf("%.10f %.10f %.10f %.10f %.10f\n",
	       array[size + j],
	       array[size + j + 1],
	       array[size + j + 2],
	       array[size + j + 3],
	       array[size + j + 4]);
    }
}

/**
 * This function is used to compare the outputs with C program's.
 * @param array data to be printed.
 * @param size size of array.
 * @param block number of blocks.
 */
void print_uint32_array(const uint32_t array[], int size, int block) {
    int b = size / block;
    int i;
    int j;

    for (j = 0; j < 5; j += 5) {
	printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32
	       " %10" PRIu32 " %10" PRIu32 "\n",
	       array[j], array[j + 1],
	       array[j + 2], array[j + 3], array[j + 4]);
    }
    for (i = 1; i < block; i++) {
	for (j = -5; j < 5; j += 5) {
	    printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32
		   " %10" PRIu32 " %10" PRIu32 "\n",
		   array[b * i + j],
		   array[b * i + j + 1],
		   array[b * i + j + 2],
		   array[b * i + j + 3],
		   array[b * i + j + 4]);
	}
    }
    for (j = -5; j < 0; j += 5) {
	printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32
	       " %10" PRIu32 " %10" PRIu32 "\n",
	       array[size + j],
	       array[size + j + 1],
	       array[size + j + 2],
	       array[size + j + 3],
	       array[size + j + 4]);
    }
}

/**
 * This function is used to compare the outputs with C program's.
 * @param array data to be printed.
 * @param size size of array.
 * @param block number of blocks.
 */
void print_double_array(const double array[], int size, int block) {
    int b = size / block;

    for (int j = 0; j < 3; j += 3) {
	printf("%.18f %.18f %.18f\n",
	       array[j], array[j + 1], array[j + 2]);
    }
    for (int i = 1; i < block; i++) {
	for (int j = -3; j < 4; j += 3) {
	    printf("%.18f %.18f %.18f\n",
		   array[b * i + j],
		   array[b * i + j + 1],
		   array[b * i + j + 2]);
	}
    }
    for (int j = -3; j < 0; j += 3) {
	printf("%.18f %.18f %.18f\n",
	       array[size + j],
	       array[size + j + 1],
	       array[size + j + 2]);
    }
}

/**
 * This function is used to compare the outputs with C program's.
 * @param array data to be printed.
 * @param size size of array.
 * @param block number of blocks.
 */
void print_uint64_array(const uint64_t array[], int size, int block) {
    int b = size / block;

    for (int j = 0; j < 3; j += 3) {
	printf("%20" PRIu64 " %20" PRIu64 " %20" PRIu64 "\n",
	       array[j], array[j + 1], array[j + 2]);
    }
    for (int i = 1; i < block; i++) {
	for (int j = -3; j < 3; j += 3) {
	    printf("%20" PRIu64 " %20" PRIu64 " %20" PRIu64 "\n",
		   array[b * i + j],
		   array[b * i + j + 1],
		   array[b * i + j + 2]);
	}
    }
    for (int j = -3; j < 0; j += 3) {
	printf("%20" PRIu64 " %20" PRIu64 " %20" PRIu64 "\n",
	       array[size + j],
	       array[size + j + 1],
	       array[size + j + 2]);
    }
}


