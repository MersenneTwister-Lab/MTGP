/*
 * @file mtgp32-jump-host.cuh
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
#include <stdio.h>
#include <cuda.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <stdlib.h>

#include "mtgp-util.cuh"
#include "mtgp32-jump-kernel.cuh"
#include "mtgp32-fast.h"

/**
 * This function initializes kernel I/O data.
 * @param d_status output kernel I/O data.
 * @param params MTGP32 parameters. needed for the initialization.
 */
void mtgp_host_initialize(mtgp32_kernel_status_t * d_status,
			  mtgp32_params_fast_t params[],
			  int block_num)
{
    int i;
    mtgp32_kernel_status_t* h_status
	= (mtgp32_kernel_status_t *) malloc(
	    sizeof(mtgp32_kernel_status_t) * block_num);

    if (h_status == NULL) {
	printf("failure in allocating host memory for kernel I/O data.\n");
	exit(8);
    }
    for (i = 0; i < block_num; i++) {
	mtgp32_init_state(&(h_status[i].status[0]), &params[i], i + 1);
    }
#if defined(DEBUG)
    printf("h_status[0].status[0]:%08"PRIx32"\n", h_status[0].status[0]);
    printf("h_status[0].status[1]:%08"PRIx32"\n", h_status[0].status[1]);
    printf("h_status[0].status[2]:%08"PRIx32"\n", h_status[0].status[2]);
    printf("h_status[0].status[3]:%08"PRIx32"\n", h_status[0].status[3]);
#endif
    ccudaMemcpy(d_status, h_status,
		sizeof(mtgp32_kernel_status_t) * block_num,
		cudaMemcpyHostToDevice);
    free(h_status);
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
    uint32_t *h_mask;
    h_pos_tbl = (uint32_t *)malloc(size1);
    h_sh1_tbl = (uint32_t *)malloc(size1);
    h_sh2_tbl = (uint32_t *)malloc(size1);
    h_param_tbl = (uint32_t *)malloc(size2);
    h_temper_tbl = (uint32_t *)malloc(size2);
    h_single_temper_tbl = (uint32_t *)malloc(size2);
    h_mask = (uint32_t *)malloc(sizeof(uint32_t));
    if (h_pos_tbl == NULL
	|| h_sh1_tbl == NULL
	|| h_sh2_tbl == NULL
	|| h_param_tbl == NULL
	|| h_temper_tbl == NULL
	|| h_single_temper_tbl == NULL
	|| h_mask == NULL
	) {
	printf("failure in allocating host memory for constant table.\n");
	exit(1);
    }
    h_mask[0] = params[0].mask;
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
    ccudaMemcpyToSymbol(pos_tbl, h_pos_tbl, size1);
    ccudaMemcpyToSymbol(sh1_tbl, h_sh1_tbl, size1);
    ccudaMemcpyToSymbol(sh2_tbl, h_sh2_tbl, size1);
    ccudaMemcpyToSymbol(param_tbl, h_param_tbl, size2);
    ccudaMemcpyToSymbol(temper_tbl, h_temper_tbl, size2);
    ccudaMemcpyToSymbol(single_temper_tbl, h_single_temper_tbl, size2);
    ccudaMemcpyToSymbol(mask, h_mask, sizeof(uint32_t));
    free(h_pos_tbl);
    free(h_sh1_tbl);
    free(h_sh2_tbl);
    free(h_param_tbl);
    free(h_temper_tbl);
    free(h_single_temper_tbl);
    free(h_mask);
}

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
    uint32_t* h_data;
    cudaError_t e;
    float gputime;
    cudaEvent_t start;
    cudaEvent_t end;

    printf("generating 32-bit unsigned random numbers.\n");
    ccudaMalloc((void**)&d_data, sizeof(uint32_t) * num_data);
    /* cutCreateTimer(&timer); */
    ccudaEventCreate(&start);
    ccudaEventCreate(&end);

    h_data = (uint32_t *) malloc(sizeof(uint32_t) * num_data);
    if (h_data == NULL) {
	printf("failure in allocating host memory for output data.\n");
	exit(1);
    }
    /* cutStartTimer(timer); */
    ccudaEventRecord(start, 0);
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
    /* ccutStopTimer(timer); */
    ccudaEventRecord(end, 0);
    ccudaEventSynchronize(end);
    ccudaMemcpy(h_data, d_data,
		sizeof(uint32_t) * num_data, cudaMemcpyDeviceToHost);
    /* gputime = cutGetTimerValue(timer); */
    ccudaEventElapsedTime(&gputime, start, end);
    print_uint32_array(h_data, num_data, block_num);
    printf("generated numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", gputime);
    printf("Samples per second: %E \n", num_data / (gputime * 0.001));

    /* ccutDeleteTimer(timer); */
    ccudaEventDestroy(start);
    ccudaEventDestroy(end);
    //free memories
    free(h_data);
    ccudaFree(d_data);
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
    float* h_data;
    cudaError_t e;
    float gputime;
    cudaEvent_t start;
    cudaEvent_t end;

    printf("generating single precision floating point random numbers.\n");
    ccudaMalloc((void**)&d_data, sizeof(uint32_t) * num_data);
    /* ccutCreateTimer(&timer); */
    ccudaEventCreate(&start);
    ccudaEventCreate(&end);
    h_data = (float *) malloc(sizeof(float) * num_data);
    if (h_data == NULL) {
	printf("failure in allocating host memory for output data.\n");
	exit(1);
    }
    /* ccutStartTimer(timer); */
    ccudaEventRecord(start, 0);
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
    /* ccutStopTimer(timer); */
    ccudaEventRecord(end, 0);
    ccudaEventSynchronize(end);
    ccudaMemcpy(h_data, d_data, sizeof(uint32_t) * num_data,
		cudaMemcpyDeviceToHost);
    /* gputime = cutGetTimerValue(timer); */
    ccudaEventElapsedTime(&gputime, start, end);
    print_float_array(h_data, num_data, block_num);
    printf("generated numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", gputime);
    printf("Samples per second: %E \n", num_data / (gputime * 0.001));
    /* ccutDeleteTimer(timer); */
    ccudaEventDestroy(start);
    ccudaEventDestroy(end);
    //free memories
    free(h_data);
    ccudaFree(d_data);
}

int main(int argc, char** argv)
{
    // LARGE_SIZE is a multiple of 16
    int num_data = 10000000;
    int block_num;
    int num_unit;
    int r;
    int device = 0;
    mtgp32_kernel_status_t *d_status;
    int mb, mp;

    ccudaSetDevice(device);

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
	block_num = get_suitable_block_num(device,
					   &mb,
					   &mp,
					   sizeof(uint32_t),
					   THREAD_NUM,
					   LARGE_SIZE);
	if (block_num <= 0) {
	    printf("can't calculate sutable number of blocks.\n");
	    return 1;
	}
	printf("the suitable number of blocks for device 0 "
	       "will be multiple of %d, or multiple of %d\n", block_num,
	       (mb - 1) * mp);
	return 1;
    }
    num_unit = LARGE_SIZE * block_num;
    ccudaMalloc((void**)&d_status,
			      sizeof(mtgp32_kernel_status_t) * block_num);
    r = num_data % num_unit;
    if (r != 0) {
	num_data = num_data + num_unit - r;
    }
    make_constant(MTGPDC_PARAM_TABLE, block_num);
    make_kernel_data32(d_status, MTGPDC_PARAM_TABLE, block_num);
    make_uint32_random(d_status, num_data, block_num);
    make_single_random(d_status, num_data, block_num);

    ccudaFree(d_status);
}
