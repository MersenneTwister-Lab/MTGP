#ifndef UTIL_H
#define UTIL_H

#include <stdint.h>
#include <inttypes.h>
#include "test-tool.hpp"

extern "C" {
#include "mtgp32-fast.h"
}
#define MTGPDC_MEXP 11213
#define MTGPDC_N 351
#define MTGPDC_FLOOR_2P 256
#define MTGPDC_CEIL_2P 512
#define MTGPDC_PARAM_TABLE mtgp32dc_params_fast_11213
#define MEXP 11213
#define THREAD_NUM MTGPDC_FLOOR_2P
#define LARGE_SIZE (THREAD_NUM * 3)
#define BLOCK_NUM_MAX 200
#define TBL_SIZE 16

extern mtgp32_params_fast_t mtgp32dc_params_fast_11213[];

/**
 * kernel I/O
 * This structure must be initialized before first use.
 */
struct mtgp32_kernel_status_t {
    uint32_t status[MTGPDC_N];
};

void print_max_min(uint32_t data[], int size);
int get_suitable_block_num(int word_size, int thread_num, int large_size);
void make_kernel_data(mtgp32_kernel_status_t *d_status,
		      mtgp32_params_fast_t params[],
		      int block_num);
void print_float_array(const float array[], int size, int block);
void print_uint32_array(const uint32_t array[], int size, int block);
int get_suitable_block_num(int word_size, int thread_num, int large_size);

#endif
