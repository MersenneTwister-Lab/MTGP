/**
 * Sample Program for CUDA 2.3
 * written by M.Saito (saito@math.sci.hiroshima-u.ac.jp)
 *
 * MT32DC-521
 * This program generates 32-bit unsigned integers.
 * The period of generated integers is 2<sup>11213</sup>-1.
 * This also generates single precision floating point numbers.
 */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <stdlib.h>
#include "mt32dc-params521.c"
#define MEXP 521
#define BLOCK_NUM_MAX 1000
//#define TOTAL_THREAD_MAX 8192
#define THREAD_NUM 64

struct MT_STRUCT {
    int idx;
    uint32_t mat_a;
    uint32_t maskB;
    uint32_t maskC;
    uint32_t state[MTDC_N];
};
typedef struct MT_STRUCT mt_struct;

static void sgenrand_mt(uint32_t seed, mt_struct *mts) {
    int i;

    for (i = 0; i < MTDC_N; i++) {
	mts->state[i] = seed;
        seed = (UINT32_C(1812433253) * (seed  ^ (seed >> 30))) + i + 1;
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
    }
    mts->idx = MTDC_N;
}

static uint32_t genrand_mt(mt_struct *mts) {
    uint32_t x;
    int lim;
    int k;

    if (mts->idx >= MTDC_N) {
	lim = MTDC_N - MTDC_M;
	for (k = 0; k < lim; k++) {
	    x = (mts->state[k] & MTDC_UPPER_MASK)
		| (mts->state[k+1] & MTDC_LOWER_MASK);
	    mts->state[k] = mts->state[k + MTDC_M]
		^ (x >> 1) ^ (x & 1U ? mts->mat_a : 0U);
	}
	lim = MTDC_N - 1;
	for (; k < lim; k++) {
	    x = (mts->state[k] & MTDC_UPPER_MASK)
		| (mts->state[k+1] & MTDC_LOWER_MASK);
	    mts->state[k] = mts->state[k + MTDC_M - MTDC_N]
		^ (x >> 1) ^ (x & 1U ? mts->mat_a : 0U);
	}
	x = (mts->state[MTDC_N - 1] & MTDC_UPPER_MASK)
	    | (mts->state[0] & MTDC_LOWER_MASK);
	mts->state[MTDC_N - 1] = mts->state[MTDC_M - 1]
	    ^ (x >> 1) ^ (x & 1U ? mts->mat_a : 0U);
	mts->idx = 0;
    }
    x = mts->state[mts->idx];
    mts->idx += 1;
    x ^= x >> MTDC_SHIFT0;
    x ^= (x << MTDC_SHIFTB) & mts->maskB;
    x ^= (x << MTDC_SHIFTC) & mts->maskC;
    x ^= x >> MTDC_SHIFT1;

    return x;
}

static void print_uint32_array(uint32_t array[], int size, int total_thread) {
    for (int j = 0; j < total_thread; j += 5) {
	printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32
	       " %10" PRIu32 " %10" PRIu32 "\n",
	       array[j],
	       array[j + 1],
	       array[j + 2],
	       array[j + 3],
	       array[j + 4]);
    }
    for (int j = size - (total_thread / 5 + 1) * 5 ; j < 0; j += 5) {
	printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32
	       " %10" PRIu32 " %10" PRIu32 "\n",
	       array[j],
	       array[j + 1],
	       array[j + 2],
	       array[j + 3],
	       array[j + 4]);
    }
}

void init_data(mt_struct mts[],
	       mt32_params_t params[],
	       int total_thread_num) {
    for (int i = 0; i < total_thread_num; i++) {
	mts[i].mat_a = params[i].mat_a;
	mts[i].maskB = params[i].maskB;
	mts[i].maskC = params[i].maskC;
    }
    for (int i = 0; i < total_thread_num; i++) {
	sgenrand_mt(i + 1, &mts[i]);
    }
#ifdef DEBUG
    for (int idx = 0; idx < 2; idx++) {
	for (int i = 0; i < MTDC_N; i++) {
	    printf("%10"PRIu32" ", mts[idx].state[i]);
	    if (i % 5 == 4) {
		printf("\n");
	    }
	}
    printf("\n");
    }
#endif
}

static void make_uint32_random(mt_struct mts[],
			       int num_data,
			       int total_thread_num) {
    uint32_t *data;
    int size;

    size = num_data / total_thread_num;
    data = (uint32_t *)malloc(sizeof(uint32_t) * num_data);
    if (data == NULL) {
	printf("malloc failure!\n");
	exit(1);
    }
    printf("generating 32-bit unsigned random numbers.\n");
#ifdef DEBUG
    printf("total_thread_num = %d\n", total_thread_num);
    printf("THREAD_NUM = %d\n", THREAD_NUM);
    printf("num_data = %d\n", num_data);
#endif
    for (int i = 0; i < total_thread_num; i++) {
	for (int j = 0; j < size; j++) {
	    data[total_thread_num * j + i] = genrand_mt(&mts[i]);
	}
    }
    print_uint32_array(data, num_data, total_thread_num);
    printf("generated numbers: %d\n", num_data);
    free(data);
}

int main(int argc, char *argv[])
{
    // LARGE_SIZE is a multiple of 16
    int num_data = 10000000;
    int block_num;
    int num_unit;
    int r;
    int total_thread_num;
    mt_struct *mts;

    if (argc >= 2) {
	errno = 0;
	block_num = strtol(argv[1], NULL, 10);
	if (errno) {
	    printf("%s number_of_block number_of_output\n", argv[0]);
	    return 1;
	}
	if (block_num < 1 || block_num > BLOCK_NUM_MAX) {
	    printf("%s block_num should be between 1 and %d\n",
		   argv[0], BLOCK_NUM_MAX);
	    return 1;
	}
	errno = 0;
	num_data = strtol(argv[2], NULL, 10);
	if (errno) {
	    printf("%s number_of_block number_of_output\n", argv[0]);
	    return 1;
	}
	argc -= 2;
	argv += 2;
    } else {
	printf("%s number_of_block number_of_output\n", argv[0]);
	return 1;
    }
    total_thread_num = block_num * THREAD_NUM;
    num_unit = total_thread_num * MTDC_N;
    r = num_data % num_unit;
    printf("block_num = %d\n", block_num);
    printf("thread_num = %d\n", THREAD_NUM);
    printf("total_thread_num = %d\n", total_thread_num);
    printf("num_data = %d\n", num_data);
    printf("num_unit = %d\n", num_unit);
    if (r != 0) {
	num_data = num_data + num_unit - r;
    }
    printf("new num_data = %d\n", num_data);
    mts = (mt_struct *)malloc(sizeof(mt_struct) * total_thread_num);
    if (mts == NULL) {
	printf("malloc for mts failure!\n");
	return 1;
    }
    init_data(mts, MTDC_PARAM_TABLE, total_thread_num);
    make_uint32_random(mts, num_data, total_thread_num);
    free(mts);
}
