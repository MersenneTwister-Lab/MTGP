/**
 * @file mtgp64-cuda.cu
 *
 * @brief Sample Program for CUDA 2.2
 *
 * MTGP64-44497
 * This program generates 64-bit unsigned integers.
 * The period of generated integers is 2<sup>44497</sup>-1.
 *
 * This also generates double precision floating point numbers
 * uniformly distributed in the range [1, 2). (double r; 1.0 <= r < 2.0)
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
#include <cutil.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <stdlib.h>
extern "C" {
#include "mtgp64-fast.h"
}
#define MEXP 44497
#define N 696
#define THREAD_NUM 512
#define LARGE_SIZE (THREAD_NUM * 3)
#define BLOCK_NUM 32	    /* You can change this value up to 128 */
#define TBL_SIZE 16

/**
 * kernel I/O
 * This structure must be initialized before first use.
 */
struct mtgp64_kernel_status_t {
    uint64_t status[N];
};

/*
 * Generator Parameters.
 */
__constant__ uint32_t param_tbl[BLOCK_NUM][TBL_SIZE];
__constant__ uint32_t temper_tbl[BLOCK_NUM][TBL_SIZE];
__constant__ uint32_t double_temper_tbl[BLOCK_NUM][TBL_SIZE];
__constant__ uint32_t pos_tbl[BLOCK_NUM];
__constant__ uint32_t sh1_tbl[BLOCK_NUM];
__constant__ uint32_t sh2_tbl[BLOCK_NUM];
/* high_mask and low_mask should be set by make_constant(), but
 * did not work.
 */
__constant__ uint32_t high_mask = 0xffff8000;
__constant__ uint32_t low_mask =  0x00000000;

/**
 * Shared memory
 * The generator's internal status vector.
 */
__shared__ uint32_t status[2][LARGE_SIZE]; /* 512 * 3 elements, 12288 bytes. */

/**
 * The function of the recursion formula calculation.
 *
 * @param[out] RH 32-bit MSBs of output
 * @param[out] RL 32-bit LSBs of output
 * @param[in] X1H MSBs of the farthest part of state array.
 * @param[in] X1L LSBs of the farthest part of state array.
 * @param[in] X2H MSBs of the second farthest part of state array.
 * @param[in] X2L LSBs of the second farthest part of state array.
 * @param[in] YH MSBs of a part of state array.
 * @param[in] YL LSBs of a part of state array.
 * @param[in] bid block id.
 */
__device__ void para_rec(uint32_t *RH,
			 uint32_t *RL,
			 uint32_t X1H,
			 uint32_t X1L,
			 uint32_t X2H,
			 uint32_t X2L,
			 uint32_t YH,
			 uint32_t YL,
			 int bid) {
    uint32_t XH = (X1H & high_mask) ^ X2H;
    uint32_t XL = (X1L & low_mask) ^ X2L;
    uint32_t MAT;

    XH ^= XH << sh1_tbl[bid];
    XL ^= XL << sh1_tbl[bid];
    YH = XL ^ (YH >> sh2_tbl[bid]);
    YL = XH ^ (YL >> sh2_tbl[bid]);
    MAT = param_tbl[bid][YL & 0x0f];
    *RH = YH ^ MAT;
    *RL = YL;
}

/**
 * The tempering function.
 *
 * @param[in] VH MSBs of the output value should be tempered.
 * @param[in] VL LSBs of the output value should be tempered.
 * @param[in] TL LSBs of the tempering helper value.
 * @param[in] bid block id.
 * @return[in] the tempered value.
 */
__device__ uint64_t temper(uint32_t VH,
			   uint32_t VL,
			   uint32_t TL,
			   int bid) {
    uint32_t MAT;
    uint64_t r;
    TL ^= TL >> 16;
    TL ^= TL >> 8;
    MAT = temper_tbl[bid][TL & 0x0f];
    VH ^= MAT;
    r = ((uint64_t)VH << 32) | VL;
    return r;
}

/**
 * The tempering and converting function.
 * By using the preset-ted table, converting to IEEE format
 * and tempering are done simultaneously.
 *
 * @param[in] VH MSBs of the output value should be tempered.
 * @param[in] VL LSBs of the output value should be tempered.
 * @param[in] TL LSBs of the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered and converted value.
 */
__device__ uint64_t temper_double(uint32_t VH,
				  uint32_t VL,
				  uint32_t TL,
				  int bid) {
    uint32_t MAT;
    uint64_t r;
    TL ^= TL >> 16;
    TL ^= TL >> 8;
    MAT = double_temper_tbl[bid][TL & 0x0f];
    r = ((uint64_t)VH << 32) | VL;
    r = (r >> 12) ^ ((uint64_t)MAT << 32);
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
__device__ void status_read(uint32_t status[2][LARGE_SIZE],
			    const mtgp64_kernel_status_t *d_status,
			    int bid,
			    int tid) {
    uint64_t x;

    x = d_status[bid].status[tid];
    status[0][LARGE_SIZE - N + tid] = x >> 32;
    status[1][LARGE_SIZE - N + tid] = x & 0xffffffff;
    if (tid < N - THREAD_NUM) {
	x = d_status[bid].status[THREAD_NUM + tid];
	status[0][LARGE_SIZE - N + THREAD_NUM + tid] = x >> 32;
	status[1][LARGE_SIZE - N + THREAD_NUM + tid] = x & 0xffffffff;
    }
    __syncthreads();
}

/**
 * Read the internal state vector from shared memory, and
 * write them into kernel I/O data.
 *
 * @param[out] status shared memory.
 * @param[in] d_status kernel I/O data
 * @param[in] bid block id
 * @param[in] tid thread id
 */
__device__ void status_write(mtgp64_kernel_status_t *d_status,
			     const uint32_t status[2][LARGE_SIZE],
			     int bid,
			     int tid) {
    uint64_t x;

    x = (uint64_t)status[0][LARGE_SIZE - N + tid] << 32;
    x = x | status[1][LARGE_SIZE - N + tid];
    d_status[bid].status[tid] = x;
    if (tid < N - THREAD_NUM) {
	x = (uint64_t)status[0][4 * THREAD_NUM - N + tid] << 32;
	x = x | status[1][4 * THREAD_NUM - N + tid];
	d_status[bid].status[THREAD_NUM + tid] = x;
    }
    __syncthreads();
}

/**
 * kernel function.
 * This function generates 64-bit unsigned integers in d_data
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output
 * @param[in] size number of output data requested.
 */
__global__ void mtgp64_uint64_kernel(mtgp64_kernel_status_t* d_status,
				     uint64_t* d_data, int size) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int pos = pos_tbl[bid];
    uint32_t YH;
    uint32_t YL;
    uint64_t o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, bid, tid);

    // main loop
    for (int i = 0; i < size; i += LARGE_SIZE) {

#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
	if ((i == 0) && (bid == 0) && (tid <= 1)) {
	    printf("status[0][LARGE_SIZE - N + tid]:%08x\n",
		   status[0][LARGE_SIZE - N + tid]);
	    printf("status[1][LARGE_SIZE - N + tid]:%08x\n",
		   status[1][LARGE_SIZE - N + tid]);
	    printf("status[0][LARGE_SIZE - N + tid + 1]:%08x\n",
		   status[0][LARGE_SIZE - N + tid + 1]);
	    printf("status[1][LARGE_SIZE - N + tid + 1]:%08x\n",
		   status[1][LARGE_SIZE - N + tid + 1]);
	    printf("status[0][LARGE_SIZE - N + tid + pos]:%08x\n",
		   status[0][LARGE_SIZE - N + tid + pos]);
	    printf("status[1][LARGE_SIZE - N + tid + pos]:%08x\n",
		   status[1][LARGE_SIZE - N + tid + pos]);
	    printf("sh1:%d\n", sh1_tbl[bid]);
	    printf("sh2:%d\n", sh2_tbl[bid]);
	    printf("high_mask:%08x\n", high_mask);
	    printf("low_mask:%08x\n", low_mask);
	    for (int j = 0; j < 16; j++) {
		printf("tbl[%d]:%08x\n", j, param_tbl[0][j]);
	    }
	}
#endif
	para_rec(&YH,
		 &YL,
		 status[0][LARGE_SIZE - N + tid],
		 status[1][LARGE_SIZE - N + tid],
		 status[0][LARGE_SIZE - N + tid + 1],
		 status[1][LARGE_SIZE - N + tid + 1],
		 status[0][LARGE_SIZE - N + tid + pos],
		 status[1][LARGE_SIZE - N + tid + pos],
		 bid);
	status[0][tid] = YH;
	status[1][tid] = YL;
#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
	if ((i == 0) && (bid == 0) && (tid <= 1)) {
	    printf("status[0][tid]:%08x\n",	status[0][tid]);
	    printf("status[1][tid]:%08x\n",	status[1][tid]);
	}
#endif
	o = temper(YH,
		   YL,
		   status[1][LARGE_SIZE - N + tid + pos - 1],
		   bid);
#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
	if ((i == 0) && (bid == 0) && (tid <= 1)) {
	    printf("o:%016" PRIx64 "\n", o);
	}
#endif
	d_data[size * bid + i + tid] = o;
	__syncthreads();
	para_rec(&YH,
		 &YL,
		 status[0][(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
		 status[1][(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
		 status[0][(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
		 status[1][(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
		 status[0][(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
		 status[1][(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
		 bid);
	status[0][tid + THREAD_NUM] = YH;
	status[1][tid + THREAD_NUM] = YL;
	o = temper(YH,
		   YL,
		   status[1][(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
		   bid);
	d_data[size * bid + THREAD_NUM + i + tid] = o;
	__syncthreads();
	para_rec(&YH,
		 &YL,
		 status[0][2 * THREAD_NUM - N + tid],
		 status[1][2 * THREAD_NUM - N + tid],
		 status[0][2 * THREAD_NUM - N + tid + 1],
		 status[1][2 * THREAD_NUM - N + tid + 1],
		 status[0][2 * THREAD_NUM - N + tid + pos],
		 status[1][2 * THREAD_NUM - N + tid + pos],
		 bid);
	status[0][tid + 2 * THREAD_NUM] = YH;
	status[1][tid + 2 * THREAD_NUM] = YL;
	o = temper(YH,
		   YL,
		   status[1][tid + pos - 1 + 2 * THREAD_NUM - N],
		   bid);
	d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
	__syncthreads();
    }
    // write back status for next call
    status_write(d_status, status, bid, tid);
}

/**
 * kernel function.
 * This function generates double precision floating point numbers in d_data.
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output. IEEE double precision format.
 * @param[in] size number of output data requested.
 */
__global__ void mtgp64_double_kernel(mtgp64_kernel_status_t* d_status,
				     uint64_t* d_data, int size)
{

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int pos = pos_tbl[bid];
    uint32_t YH;
    uint32_t YL;
    uint64_t o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, bid, tid);

    // main loop
    for (int i = 0; i < size; i += LARGE_SIZE) {
	para_rec(&YH,
		 &YL,
		 status[0][LARGE_SIZE - N + tid],
		 status[1][LARGE_SIZE - N + tid],
		 status[0][LARGE_SIZE - N + tid + 1],
		 status[1][LARGE_SIZE - N + tid + 1],
		 status[0][LARGE_SIZE - N + tid + pos],
		 status[1][LARGE_SIZE - N + tid + pos],
		 bid);
	status[0][tid] = YH;
	status[1][tid] = YL;
	o = temper_double(YH,
			  YL,
			  status[1][LARGE_SIZE - N + tid + pos - 1],
			  bid);
	d_data[size * bid + i + tid] = o;
	__syncthreads();
	para_rec(&YH,
		 &YL,
		 status[0][(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
		 status[1][(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
		 status[0][(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
		 status[1][(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
		 status[0][(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
		 status[1][(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
		 bid);
	status[0][tid + THREAD_NUM] = YH;
	status[1][tid + THREAD_NUM] = YL;
	o = temper_double(
	    YH,
	    YL,
	    status[1][(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
	    bid);
	d_data[size * bid + THREAD_NUM + i + tid] = o;
	__syncthreads();
	para_rec(&YH,
		 &YL,
		 status[0][2 * THREAD_NUM - N + tid],
		 status[1][2 * THREAD_NUM - N + tid],
		 status[0][2 * THREAD_NUM - N + tid + 1],
		 status[1][2 * THREAD_NUM - N + tid + 1],
		 status[0][2 * THREAD_NUM - N + tid + pos],
		 status[1][2 * THREAD_NUM - N + tid + pos],
		 bid);
	status[0][tid + 2 * THREAD_NUM] = YH;
	status[1][tid + 2 * THREAD_NUM] = YL;
	o = temper_double(YH,
			  YL,
			  status[1][tid + pos - 1 + 2 * THREAD_NUM - N],
			  bid);
	d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
	__syncthreads();
    }
    // write back status for next call
    status_write(d_status, status, bid, tid);
}

/**
 * This function sets constants in device memory.
 * @param[in] params input, MTGP64 parameters.
 */
void make_constant(const mtgp64_params_fast_t params[]) {
    const int size1 = sizeof(uint32_t) * BLOCK_NUM;
    const int size2 = sizeof(uint32_t) * BLOCK_NUM * TBL_SIZE;
    uint32_t *h_pos_tbl;
    uint32_t *h_sh1_tbl;
    uint32_t *h_sh2_tbl;
    uint32_t *h_param_tbl;
    uint32_t *h_temper_tbl;
    uint32_t *h_double_temper_tbl;
#if 0
    uint32_t *h_high_mask;
    uint32_t *h_low_mask;
#endif
    h_pos_tbl = (uint32_t *)malloc(size1);
    h_sh1_tbl = (uint32_t *)malloc(size1);
    h_sh2_tbl = (uint32_t *)malloc(size1);
    h_param_tbl = (uint32_t *)malloc(size2);
    h_temper_tbl = (uint32_t *)malloc(size2);
    h_double_temper_tbl = (uint32_t *)malloc(size2);
#if 0
    h_high_mask = (uint32_t *)malloc(sizeof(uint32_t));
    h_low_mask = (uint32_t *)malloc(sizeof(uint32_t));
#endif
    if (h_pos_tbl == NULL
	|| h_sh1_tbl == NULL
	|| h_sh2_tbl == NULL
	|| h_param_tbl == NULL
	|| h_temper_tbl == NULL
	|| h_double_temper_tbl == NULL
#if 0
	|| h_high_mask == NULL
	|| h_low_mask == NULL
#endif
	) {
	printf("failure in allocating host memory for constant table.\n");
	exit(1);
    }
#if 0
    *h_high_mask = params[0].mask >> 32;
    *h_low_mask = params[0].mask & 0xffffffffU;
#endif
    for (int i = 0; i < BLOCK_NUM; i++) {
	h_pos_tbl[i] = params[i].pos;
	h_sh1_tbl[i] = params[i].sh1;
	h_sh2_tbl[i] = params[i].sh2;
	for (int j = 0; j < TBL_SIZE; j++) {
	    h_param_tbl[i * TBL_SIZE + j] = params[i].tbl[j] >> 32;
	    h_temper_tbl[i * TBL_SIZE + j] = params[i].tmp_tbl[j] >> 32;
	    h_double_temper_tbl[i * TBL_SIZE + j]
		= params[i].dbl_tmp_tbl[j] >> 32;
	}
    }
    // copy from malloc area only
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(pos_tbl, h_pos_tbl, size1));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(sh1_tbl, h_sh1_tbl, size1));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(sh2_tbl, h_sh2_tbl, size1));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(param_tbl, h_param_tbl, size2));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(temper_tbl, h_temper_tbl, size2));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(double_temper_tbl,
				      h_double_temper_tbl, size2));
#if 0
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(&high_mask,
				      &h_high_mask, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(&low_mask,
				      &h_low_mask, sizeof(uint32_t)));
#endif
    free(h_pos_tbl);
    free(h_sh1_tbl);
    free(h_sh2_tbl);
    free(h_param_tbl);
    free(h_temper_tbl);
    free(h_double_temper_tbl);
#if 0
    free(h_high_mask);
    free(h_low_mask);
#endif
}

/**
 * This function initializes kernel I/O data.
 * @param[out] d_status output kernel I/O data.
 * @param[in] params MTGP64 parameters. needed for the initialization.
 */
void make_kernel_data(mtgp64_kernel_status_t *d_status,
		     mtgp64_params_fast_t params[]) {
    mtgp64_kernel_status_t* h_status = (mtgp64_kernel_status_t *) malloc(
	sizeof(mtgp64_kernel_status_t) * BLOCK_NUM);

    if (h_status == NULL) {
	printf("failure in allocating host memory for kernel I/O data.\n");
	exit(8);
    }
    for (int i = 0; i < BLOCK_NUM; i++) {
	mtgp64_init_state(&(h_status[i].status[0]), &params[i], i + 1);
    }
#if defined(DEBUG)
    printf("h_status[0].status[0]:%016"PRIx64"\n", h_status[0].status[0]);
    printf("h_status[0].status[0]:%016"PRIx64"\n", h_status[0].status[1]);
    printf("h_status[0].status[0]:%016"PRIx64"\n", h_status[0].status[2]);
    printf("h_status[0].status[0]:%016"PRIx64"\n", h_status[0].status[3]);
#endif
    CUDA_SAFE_CALL(cudaMemcpy(d_status,
			      h_status,
			      sizeof(mtgp64_kernel_status_t) * BLOCK_NUM,
			      cudaMemcpyHostToDevice));
    free(h_status);
}

/**
 * This function is used to compare the outputs with C program's.
 * @param[in] array data to be printed.
 * @param[in] size size of array.
 * @param[in] block number of blocks.
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
 * @param[in] array data to be printed.
 * @param[in] size size of array.
 * @param[in] block number of blocks.
 */
void print_uint64_array(uint64_t array[], int size, int block) {
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

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param[in] d_status kernel I/O data.
 * @param[in] num_data number of data to be generated.
 */
void make_uint64_random(mtgp64_kernel_status_t* d_status, int num_data) {
    uint64_t* d_data;
    unsigned int timer = 0;
    uint64_t* h_data;
    cudaError_t e;
    float gputime;

    printf("generating 64-bit unsigned random numbers.\n");
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, sizeof(uint64_t) * num_data));
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    h_data = (uint64_t *) malloc(sizeof(uint64_t) * num_data);
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
    mtgp64_uint64_kernel<<< BLOCK_NUM, THREAD_NUM>>>(
	d_status, d_data, num_data / BLOCK_NUM);
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
		   sizeof(uint64_t) * num_data,
		   cudaMemcpyDeviceToHost));
    gputime = cutGetTimerValue(timer);
    print_uint64_array(h_data, num_data, BLOCK_NUM);
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
void make_double_random(mtgp64_kernel_status_t* d_status, int num_data) {
    uint64_t* d_data;
    unsigned int timer = 0;
    double* h_data;
    cudaError_t e;
    float gputime;

    printf("generating double precision floating point random numbers.\n");
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, sizeof(uint64_t) * num_data));
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    h_data = (double *) malloc(sizeof(double) * num_data);
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
    mtgp64_double_kernel<<< BLOCK_NUM, THREAD_NUM >>>(
	d_status, d_data, num_data / BLOCK_NUM);
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
		   sizeof(uint64_t) * num_data,
		   cudaMemcpyDeviceToHost));
    gputime = cutGetTimerValue(timer);
    print_double_array(h_data, num_data, BLOCK_NUM);
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
    int num_unit = LARGE_SIZE * BLOCK_NUM;
    int r;
    mtgp64_kernel_status_t* d_status;

    CUT_DEVICE_INIT(argc, argv);
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_status,
			      sizeof(mtgp64_kernel_status_t) * BLOCK_NUM));
    if (argc >= 2) {
	errno = 0;
	num_data = strtol(argv[1], NULL, 10);
	if (errno) {
	    printf("%s number_of_output\n", argv[0]);
	    return 1;
	}
    } else {
	printf("%s number_of_output\n", argv[0]);
	return 1;
    }
    r = num_data % num_unit;
    if (r != 0) {
	num_data = num_data + num_unit - r;
    }
    make_constant(mtgp64_params_fast_44497);
    make_kernel_data(d_status, mtgp64_params_fast_44497);
    make_uint64_random(d_status, num_data);
    make_double_random(d_status, num_data);

    //finalize
    CUDA_SAFE_CALL(cudaFree(d_status));
    CUT_EXIT(argc, argv);
}
