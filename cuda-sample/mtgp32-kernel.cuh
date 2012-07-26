/*
 * @file mtgp32-kernel.cuh
 *
 * @brief Sample Program for CUDA 3.2 and 4.0
 *
 * MTGP32-11213
 * This program generates 32-bit unsigned integers.
 * The period of generated integers is 2<sup>11213</sup>-1.
 *
 * This also generates single precision floating point numbers
 * uniformly distributed in the range [1, 2). (float r; 1.0 <= r < 2.0)
 */
#ifndef MTGP32_KERNEL_CUH
#define MTGP32_KERNEL_CUH

#include <cuda.h>
#include <stdint.h>
#include <inttypes.h>

#define MTGPDC_MEXP 11213
#define MTGPDC_N 351
#define MTGPDC_FLOOR_2P 256
#define MTGPDC_CEIL_2P 512
#define MTGP_TN MTGPDC_FLOOR_2P
#define MTGP_LS (MTGP_TN * 3)
#define MTGP_BN_MAX 200
#define MTGP_TS 16

extern mtgp32_params_fast_t mtgp32dc_params_fast_11213[];

/**
 * kernel I/O
 * This structure must be initialized before first use.
 */
struct mtgp32_kernel_status_t {
    uint32_t status[MTGPDC_N];
};

/*
 * Generator Parameters.
 */
__constant__ unsigned int pos_tbl[MTGP_BN_MAX];
__constant__ uint32_t param_tbl[MTGP_BN_MAX][MTGP_TS];
__constant__ uint32_t temper_tbl[MTGP_BN_MAX][MTGP_TS];
__constant__ uint32_t single_temper_tbl[MTGP_BN_MAX][MTGP_TS];
__constant__ uint32_t sh1_tbl[MTGP_BN_MAX];
__constant__ uint32_t sh2_tbl[MTGP_BN_MAX];
__constant__ uint32_t mask[1];

/**
 * Shared memory
 * The generator's internal status vector.
 */
__shared__ uint32_t status[MTGP_LS];

/**
 * The function of the recursion formula calculation.
 *
 * @param[in] X1 the farthest part of state array.
 * @param[in] X2 the second farthest part of state array.
 * @param[in] Y a part of state array.
 * @param[in] bid block id.
 * @return output
 */
__device__ uint32_t para_rec(uint32_t X1, uint32_t X2, uint32_t Y, int bid) {
    uint32_t X = (X1 & mask[0]) ^ X2;
    uint32_t MAT;

    X ^= X << sh1_tbl[bid];
    Y = X ^ (Y >> sh2_tbl[bid]);
    MAT = param_tbl[bid][Y & 0x0f];
    return Y ^ MAT;
}

/**
 * The tempering function.
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered value.
 */
__device__ uint32_t temper(uint32_t V, uint32_t T, int bid) {
    uint32_t MAT;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = temper_tbl[bid][T & 0x0f];
    return V ^ MAT;
}

/**
 * The tempering and converting function.
 * By using the preset-ted table, converting to IEEE format
 * and tempering are done simultaneously.
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered and converted value.
 */
__device__ uint32_t temper_single(uint32_t V, uint32_t T, int bid) {
    uint32_t MAT;
    uint32_t r;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = single_temper_tbl[bid][T & 0x0f];
    r = (V >> 9) ^ MAT;
    return r;
}

/**
 * Read the internal state vector from kernel I/O data, and
 * put them into shared memory.
 *
 * @param[out] status shared memory.
 * @param[in] d_status kernel I/O data
 * @param[in] bid block id
 * @param[in] tid thread id
 */
__device__ void status_read(uint32_t status[MTGP_LS],
			    const mtgp32_kernel_status_t *d_status,
			    int bid,
			    int tid) {
    status[MTGP_LS - N + tid] = d_status[bid].status[tid];
    if (tid < MTGP_N - MTGP_TN) {
	status[MTGP_LS - N + MTGP_TN + tid]
	    = d_status[bid].status[MTGP_TN + tid];
    }
    __syncthreads();
}

/**
 * Read the internal state vector from shared memory, and
 * write them into kernel I/O data.
 *
 * @param[out] d_status kernel I/O data
 * @param[in] status shared memory.
 * @param[in] bid block id
 * @param[in] tid thread id
 */
__device__ void status_write(mtgp32_kernel_status_t *d_status,
			     const uint32_t status[MTGP_LS],
			     int bid,
			     int tid) {
    d_status[bid].status[tid] = status[MTGP_LS - MTGP_N + tid];
    if (tid < MTGP_N - MTGP_TN) {
	d_status[bid].status[MTGP_TN + tid]
	    = status[4 * MTGP_TN - MTGP_N + tid];
    }
    __syncthreads();
}
#endif
