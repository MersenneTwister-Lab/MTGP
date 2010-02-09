/**
 * @file mtgp32-cuda.cu
 *
 * @brief Sample Program for CUDA 2.2
 *
 * MTGP32-11213
 * This program generates 32-bit unsigned integers.
 * The period of generated integers is 2<sup>11213</sup>-1.
 *
 * This also generates single precision floating point numbers
 * uniformly distributed in the range [1, 2). (float r; 1.0 <= r < 2.0)
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (Hiroshima University)
 *
 * Copyright (C) 2009 Mutsuo Saito, Makoto Matsumoto and
 * Hiroshima University. All rights reserved.
 *
 * The new BSD License is applied to this software, see LICENSE.txt
 */
#define __STDC_FORMAT_MACROS 1
#define __STDC_CONSTANT_MACROS 1
#include <stdio.h>
#include <cuda.h>
#include <cutil.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <stdlib.h>
extern "C" {
#include "mtgp32-fast.h"
#include "mtgp32dc-param-11213.c"
}
#define MEXP 11213
#define N MTGPDC_N
#define THREAD_NUM MTGPDC_FLOOR_2P
#define LARGE_SIZE (THREAD_NUM * 3)
//#define BLOCK_NUM 32
#define BLOCK_NUM_MAX 200
#define TBL_SIZE 16

/**
 * kernel I/O
 * This structure must be initialized before first use.
 */
struct mtgp32_kernel_status_t {
    uint32_t status[N];
};

/*
 * Generator Parameters.
 */
__constant__ uint32_t param_tbl[BLOCK_NUM_MAX][TBL_SIZE];
__constant__ uint32_t temper_tbl[BLOCK_NUM_MAX][TBL_SIZE];
__constant__ uint32_t single_temper_tbl[BLOCK_NUM_MAX][TBL_SIZE];
__constant__ uint32_t pos_tbl[BLOCK_NUM_MAX];
__constant__ uint32_t sh1_tbl[BLOCK_NUM_MAX];
__constant__ uint32_t sh2_tbl[BLOCK_NUM_MAX];
/* high_mask and low_mask should be set by make_constant(), but
 * did not work.
 */
__constant__ uint32_t mask = 0xff800000;

/**
 * Shared memory
 * The generator's internal status vector.
 */
__shared__ uint32_t status[LARGE_SIZE];

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
    uint32_t X = (X1 & mask) ^ X2;
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
__device__ void status_read(uint32_t status[LARGE_SIZE],
			    const mtgp32_kernel_status_t *d_status,
			    int bid,
			    int tid) {
    status[LARGE_SIZE - N + tid] = d_status[bid].status[tid];
    if (tid < N - THREAD_NUM) {
	status[LARGE_SIZE - N + THREAD_NUM + tid]
	    = d_status[bid].status[THREAD_NUM + tid];
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
			     const uint32_t status[LARGE_SIZE],
			     int bid,
			     int tid) {
    d_status[bid].status[tid] = status[LARGE_SIZE - N + tid];
    if (tid < N - THREAD_NUM) {
	d_status[bid].status[THREAD_NUM + tid]
	    = status[4 * THREAD_NUM - N + tid];
    }
    __syncthreads();
}

/**
 * kernel function.
 * This function generates 32-bit unsigned integers in d_data
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output
 * @param[in] size number of output data requested.
 */
__global__ void mtgp32_uint32_kernel(mtgp32_kernel_status_t* d_status,
				     uint32_t* d_data, int size) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int pos = pos_tbl[bid];
    uint32_t r;
    uint32_t o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, bid, tid);

    // main loop
    for (int i = 0; i < size; i += LARGE_SIZE) {

#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
	if ((i == 0) && (bid == 0) && (tid <= 1)) {
	    printf("status[LARGE_SIZE - N + tid]:%08x\n",
		   status[LARGE_SIZE - N + tid]);
	    printf("status[LARGE_SIZE - N + tid + 1]:%08x\n",
		   status[LARGE_SIZE - N + tid + 1]);
	    printf("status[LARGE_SIZE - N + tid + pos]:%08x\n",
		   status[LARGE_SIZE - N + tid + pos]);
	    printf("sh1:%d\n", sh1_tbl[bid]);
	    printf("sh2:%d\n", sh2_tbl[bid]);
	    printf("mask:%08x\n", mask);
	    for (int j = 0; j < 16; j++) {
		printf("tbl[%d]:%08x\n", j, param_tbl[0][j]);
	    }
	}
#endif
	r = para_rec(status[LARGE_SIZE - N + tid],
		 status[LARGE_SIZE - N + tid + 1],
		 status[LARGE_SIZE - N + tid + pos],
		 bid);
	status[tid] = r;
#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
	if ((i == 0) && (bid == 0) && (tid <= 1)) {
	    printf("status[tid]:%08x\n", status[tid]);
	}
#endif
	o = temper(r, status[LARGE_SIZE - N + tid + pos - 1], bid);
#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
	if ((i == 0) && (bid == 0) && (tid <= 1)) {
	    printf("r:%08" PRIx32 "\n", r);
	}
#endif
	d_data[size * bid + i + tid] = o;
	__syncthreads();
	r = para_rec(status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
		     status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
		     status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
		     bid);
	status[tid + THREAD_NUM] = r;
	o = temper(r,
		   status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
		   bid);
	d_data[size * bid + THREAD_NUM + i + tid] = o;
	__syncthreads();
	r = para_rec(status[2 * THREAD_NUM - N + tid],
		     status[2 * THREAD_NUM - N + tid + 1],
		     status[2 * THREAD_NUM - N + tid + pos],
		     bid);
	status[tid + 2 * THREAD_NUM] = r;
	o = temper(r, status[tid + pos - 1 + 2 * THREAD_NUM - N], bid);
	d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
	__syncthreads();
    }
    // write back status for next call
    status_write(d_status, status, bid, tid);
}

/**
 * kernel function.
 * This function generates single precision floating point numbers in d_data.
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output. IEEE single precision format.
 * @param[in] size number of output data requested.
 */
__global__ void mtgp32_single_kernel(mtgp32_kernel_status_t* d_status,
				     uint32_t* d_data, int size)
{

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int pos = pos_tbl[bid];
    uint32_t r;
    uint32_t o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, bid, tid);

    // main loop
    for (int i = 0; i < size; i += LARGE_SIZE) {
	r = para_rec(status[LARGE_SIZE - N + tid],
		     status[LARGE_SIZE - N + tid + 1],
		     status[LARGE_SIZE - N + tid + pos],
		     bid);
	status[tid] = r;
	o = temper_single(r, status[LARGE_SIZE - N + tid + pos - 1], bid);
	d_data[size * bid + i + tid] = o;
	__syncthreads();
	r = para_rec(status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
		     status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
		     status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
		     bid);
	status[tid + THREAD_NUM] = r;
	o = temper_single(
	    r,
	    status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
	    bid);
	d_data[size * bid + THREAD_NUM + i + tid] = o;
	__syncthreads();
	r = para_rec(status[2 * THREAD_NUM - N + tid],
		     status[2 * THREAD_NUM - N + tid + 1],
		     status[2 * THREAD_NUM - N + tid + pos],
		     bid);
	status[tid + 2 * THREAD_NUM] = r;
	o = temper_single(r,
			  status[tid + pos - 1 + 2 * THREAD_NUM - N],
			  bid);
	d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
	__syncthreads();
    }
    // write back status for next call
    status_write(d_status, status, bid, tid);
}

/**
 * This function sets constants in device memory.
 * @param[in] params input, MTGP32 parameters.
 */
void make_constant(const mtgp32_params_fast_t params[],
    int block_num) {
    const int size1 = sizeof(uint32_t) * block_num;
    const int size2 = sizeof(uint32_t) * block_num * TBL_SIZE;
    uint32_t *h_pos_tbl;
    uint32_t *h_sh1_tbl;
    uint32_t *h_sh2_tbl;
    uint32_t *h_param_tbl;
    uint32_t *h_temper_tbl;
    uint32_t *h_single_temper_tbl;
#if 0
    uint32_t *h_mask;
#endif
    h_pos_tbl = (uint32_t *)malloc(size1);
    h_sh1_tbl = (uint32_t *)malloc(size1);
    h_sh2_tbl = (uint32_t *)malloc(size1);
    h_param_tbl = (uint32_t *)malloc(size2);
    h_temper_tbl = (uint32_t *)malloc(size2);
    h_single_temper_tbl = (uint32_t *)malloc(size2);
#if 0
    h_mask = (uint32_t *)malloc(sizeof(uint32_t));
#endif
    if (h_pos_tbl == NULL
	|| h_sh1_tbl == NULL
	|| h_sh2_tbl == NULL
	|| h_param_tbl == NULL
	|| h_temper_tbl == NULL
	|| h_single_temper_tbl == NULL
#if 0
	|| h_mask == NULL
#endif
	) {
	printf("failure in allocating host memory for constant table.\n");
	exit(1);
    }
#if 0
    h_mask = params[0].mask;
#endif
    for (int i = 0; i < block_num; i++) {
	h_pos_tbl[i] = params[i].pos;
	h_sh1_tbl[i] = params[i].sh1;
	h_sh2_tbl[i] = params[i].sh2;
	for (int j = 0; j < TBL_SIZE; j++) {
	    h_param_tbl[i * TBL_SIZE + j] = params[i].tbl[j];
	    h_temper_tbl[i * TBL_SIZE + j] = params[i].tmp_tbl[j];
	    h_single_temper_tbl[i * TBL_SIZE + j] = params[i].flt_tmp_tbl[j];
	}
    }
    // copy from malloc area only
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(pos_tbl, h_pos_tbl, size1));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(sh1_tbl, h_sh1_tbl, size1));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(sh2_tbl, h_sh2_tbl, size1));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(param_tbl, h_param_tbl, size2));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(temper_tbl, h_temper_tbl, size2));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(single_temper_tbl,
				      h_single_temper_tbl, size2));
#if 0
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(&mask,
				      &h_mask, sizeof(uint32_t)));
#endif
    free(h_pos_tbl);
    free(h_sh1_tbl);
    free(h_sh2_tbl);
    free(h_param_tbl);
    free(h_temper_tbl);
    free(h_single_temper_tbl);
#if 0
    free(h_mask);
#endif
}

#include "mtgp-cuda-common.c"
#include "mtgp32-cuda-common.c"

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param[in] d_status kernel I/O data.
 * @param[in] num_data number of data to be generated.
 */
void make_uint32_random(mtgp32_kernel_status_t* d_status,
			int num_data,
			int block_num) {
    uint32_t* d_data;
    unsigned int timer = 0;
    uint32_t* h_data;
    cudaError_t e;
    float gputime;

    printf("generating 32-bit unsigned random numbers.\n");
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, sizeof(uint32_t) * num_data));
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    h_data = (uint32_t *) malloc(sizeof(uint32_t) * num_data);
    if (h_data == NULL) {
	printf("failure in allocating host memory for output data.\n");
	exit(1);
    }
    CUT_SAFE_CALL(cutStartTimer(timer));
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

    /* kernel call */
    mtgp32_uint32_kernel<<< block_num, THREAD_NUM>>>(
	d_status, d_data, num_data / block_num);
    cudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    CUT_SAFE_CALL(cutStopTimer(timer));
    CUDA_SAFE_CALL(
	cudaMemcpy(h_data,
		   d_data,
		   sizeof(uint32_t) * num_data,
		   cudaMemcpyDeviceToHost));
    gputime = cutGetTimerValue(timer);
    print_uint32_array(h_data, num_data, block_num);
    printf("generated numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", gputime);
    printf("Samples per second: %E \n", num_data / (gputime * 0.001));
    CUT_SAFE_CALL(cutDeleteTimer(timer));
    //free memories
    free(h_data);
    CUDA_SAFE_CALL(cudaFree(d_data));
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param[in] d_status kernel I/O data.
 * @param[in] num_data number of data to be generated.
 */
void make_single_random(mtgp32_kernel_status_t* d_status,
			int num_data,
			int block_num) {
    uint32_t* d_data;
    unsigned int timer = 0;
    float* h_data;
    cudaError_t e;
    float gputime;

    printf("generating single precision floating point random numbers.\n");
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, sizeof(uint32_t) * num_data));
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    h_data = (float *) malloc(sizeof(float) * num_data);
    if (h_data == NULL) {
	printf("failure in allocating host memory for output data.\n");
	exit(1);
    }
    CUT_SAFE_CALL(cutStartTimer(timer));
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

    /* kernel call */
    mtgp32_single_kernel<<< block_num, THREAD_NUM >>>(
	d_status, d_data, num_data / block_num);
    cudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    CUT_SAFE_CALL(cutStopTimer(timer));
    CUDA_SAFE_CALL(
	cudaMemcpy(h_data,
		   d_data,
		   sizeof(uint32_t) * num_data,
		   cudaMemcpyDeviceToHost));
    gputime = cutGetTimerValue(timer);
    print_float_array(h_data, num_data, block_num);
    printf("generated numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", gputime);
    printf("Samples per second: %E \n", num_data / (gputime * 0.001));
    CUT_SAFE_CALL(cutDeleteTimer(timer));
    //free memories
    free(h_data);
    CUDA_SAFE_CALL(cudaFree(d_data));
}

int main(int argc, char** argv)
{
    // LARGE_SIZE is a multiple of 16
    int num_data = 10000000;
    int block_num;
    int num_unit;
    int r;
    mtgp32_kernel_status_t *d_status;

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
	CUT_DEVICE_INIT(argc, argv);
	printf("%s number_of_block number_of_output\n", argv[0]);
	block_num = get_suitable_block_num(sizeof(uint32_t),
					   THREAD_NUM,
					   LARGE_SIZE);
	if (block_num <= 0) {
	    printf("can't calculate sutable number of blocks.\n");
	    return 1;
	}
	printf("the suitable number of blocks for device 0 "
	       "will be multiple of %d\n", block_num);
	return 1;
    }
    CUT_DEVICE_INIT(argc, argv);
    num_unit = LARGE_SIZE * block_num;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_status,
			      sizeof(mtgp32_kernel_status_t) * block_num));
    r = num_data % num_unit;
    if (r != 0) {
	num_data = num_data + num_unit - r;
    }
    make_constant(MTGPDC_PARAM_TABLE, block_num);
    make_kernel_data(d_status, MTGPDC_PARAM_TABLE, block_num);
    make_uint32_random(d_status, num_data, block_num);
    make_single_random(d_status, num_data, block_num);

    //finalize
    CUDA_SAFE_CALL(cudaFree(d_status));
#ifdef NEED_PROMPT
    CUT_EXIT(argc, argv);
#endif
}
