/**
 * Sample host program for far jump (2<sup>256</sup> steps jump ahead)
 */
#include <stdio.h>
#include <errno.h>
#include "mtgp-util.cuh"
#include "mtgp32-jump-kernel.cuh"

#if defined(CHECK)
#include "mtgp32-fast-jump.h"
/**
 *
 */
#define BLOCK_NUM 16
static mtgp32_fast_t mtgp32[BLOCK_NUM];
static int init_check_data(int block_num, uint32_t seed)
{
    if (block_num > BLOCK_NUM) {
	printf("can't check block num > 10 %d\n", block_num);
	return 1;
    }
    for (int i = 0; i < block_num; i++) {
	int rc = mtgp32_init(&mtgp32[i],
			     &mtgp32_params_fast_11213[0],
			     seed);
	if (rc) {
	    return rc;
	}
	for (int j = 0; j < i; j++) {
	    mtgp32_fast_jump(&mtgp32[i], MTGP32_JUMP2_256);
	}
    }
    return 0;
}

static int init_check_data_array(int block_num, uint32_t seed_array[], int size)
{
    if (block_num > BLOCK_NUM) {
	printf("can't check block num > 10 %d\n", block_num);
	return 1;
    }
    for (int i = 0; i < block_num; i++) {
	int rc = mtgp32_init_by_array(&mtgp32[i],
				      &mtgp32_params_fast_11213[0],
				      seed_array,
				      size);
	if (rc) {
	    return rc;
	}
	for (int j = 0; j < i; j++) {
	    mtgp32_fast_jump(&mtgp32[i], MTGP32_JUMP2_256);
	}
    }
    return 0;
}

static void free_check_data(int block_num)
{
    for (int i = 0; i < block_num; i++) {
	mtgp32_free(&mtgp32[i]);
    }
}

static void check_data(uint32_t * h_data,
		       int num_data,
		       int block_num)
{
    int size = num_data / block_num;
    if (block_num > BLOCK_NUM) {
	printf("can't check block num > 10 %d\n", block_num);
	return;
    }
    for (int i = 0; i < block_num; i++) {
	for (int j = 0; j < size; j++) {
	    uint32_t r = mtgp32_genrand_uint32(&mtgp32[i]);
	    if (h_data[i * size + j] != r) {
		printf("mismatch i = %d, j = %d, data = %u, r = %u\n",
		       i, j, h_data[i * size + j], r);
		printf("check N.G!\n");
		return;
	    }
	}
    }
    printf("check O.K!\n");
}

static void check_status(struct mtgp32_kernel_status_t * h_status,
			 int block_num)
{
    int counter = 0;
    if (block_num > BLOCK_NUM) {
	printf("can't check block num > 10 %d\n", block_num);
	return;
    }
    int large_size = mtgp32[0].status->large_size;
    for (int i = 0; i < block_num; i++) {
	for (int j = 0; j < MTGP32_N; j++) {
	    int idx = mtgp32[i].status->idx - MTGP32_N + 1 + large_size;
	    uint32_t x = h_status[i].status[j];
	    uint32_t r = mtgp32[i].status->array[(j + idx) % large_size];
	    if (x != r) {
		printf("mismatch i = %d, j = %d, kernel = %u, C = %u\n",
		       i, j, x, r);
		printf("check N.G!\n");
		counter++;
	    }
	    if (counter > 10) {
		return;
	    }
	}
    }
    printf("check O.K!\n");
    fflush(stdout);
}
#endif

/**
 * host function.
 * This function calls initialization kernel function.
 *
 * @param[in] d_status kernel I/O data.
 * @param[in] num_data number of data to be generated.
 */
static void initialize_mtgp_kernel(mtgp32_kernel_status_t* d_status,
				   uint32_t seed,
				   int block_num)
{
    struct mtgp32_kernel_status_t* h_status;
    cudaError_t e;
    float gputime;
    cudaEvent_t start;
    cudaEvent_t end;

    printf("initializing and jumping mtgp32 kernel data.\n");
    h_status = (struct mtgp32_kernel_status_t *)
	malloc(sizeof(mtgp32_kernel_status_t) * block_num);
    if (h_status == NULL) {
	printf("failure in allocating host memory for kernel status.\n");
	exit(1);
    }
    ccudaEventCreate(&start);
    ccudaEventCreate(&end);

    ccudaEventRecord(start, 0);
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

    /* kernel call
     * CAUTION: the number of threads should be MTGP32_N for
     * initialization.
     */
    mtgp32_jump_long_seed_kernel<<<block_num, MTGP32_N>>>(d_status, seed);
    cudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    ccudaEventRecord(end, 0);
    ccudaEventSynchronize(end);
    ccudaEventElapsedTime(&gputime, start, end);
    printf("Initialization time: %f (ms)\n", gputime);
    ccudaMemcpy(h_status, d_status,
		sizeof(mtgp32_kernel_status_t) * block_num,
		cudaMemcpyDeviceToHost);
    check_status(h_status, block_num);
    ccudaEventDestroy(start);
    ccudaEventDestroy(end);
}

/**
 * host function.
 * This function calls initialization kernel function.
 *
 * @param[in] d_status kernel I/O data.
 * @param[in] num_data number of data to be generated.
 */
static void initialize_mtgp_array_kernel(mtgp32_kernel_status_t* d_status,
					 uint32_t seed_array[],
					 int seed_size,
					 int block_num)
{
    struct mtgp32_kernel_status_t* h_status;
    uint32_t *h_seed_array;
    uint32_t *d_seed_array;
    cudaError_t e;
    float gputime;
    cudaEvent_t start;
    cudaEvent_t end;

    printf("initializing and jumping mtgp32 kernel data.\n");
    h_status = (struct mtgp32_kernel_status_t *)
	malloc(sizeof(mtgp32_kernel_status_t) * block_num);
    if (h_status == NULL) {
	printf("failure in allocating host memory for kernel status.\n");
	exit(1);
    }
    h_seed_array = (uint32_t *) malloc(sizeof(uint32_t) * seed_size);
    if (h_seed_array == NULL) {
	printf("failure in allocating host memory for seed array.\n");
	exit(1);
    }
    for (int i = 0; i < seed_size; i++) {
	h_seed_array[i] = seed_array[i];
    }
    ccudaMalloc((void**)&d_seed_array, sizeof(uint32_t) * seed_size);
    ccudaMemcpy(d_seed_array, h_seed_array,
		sizeof(uint32_t) * seed_size,
		cudaMemcpyHostToDevice);
    ccudaEventCreate(&start);
    ccudaEventCreate(&end);

    ccudaEventRecord(start, 0);
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

    /* kernel call
     * CAUTION: the number of threads should be MTGP32_N for
     * initialization.
     */
    mtgp32_jump_long_array_kernel<<<block_num, MTGP32_N>>>(
	d_status, d_seed_array, seed_size);
    cudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    ccudaEventRecord(end, 0);
    ccudaEventSynchronize(end);
    ccudaEventElapsedTime(&gputime, start, end);
    printf("Initialization time (seed array): %f (ms)\n", gputime);
    ccudaMemcpy(h_status, d_status,
		sizeof(mtgp32_kernel_status_t) * block_num,
		cudaMemcpyDeviceToHost);
    check_status(h_status, block_num);
    ccudaEventDestroy(start);
    ccudaEventDestroy(end);
    ccudaFree(d_seed_array);
    free(h_seed_array);
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param[in] d_status kernel I/O data.
 * @param[in] num_data number of data to be generated.
 */
static void make_uint32_random(mtgp32_kernel_status_t* d_status,
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
    mtgp32_uint32_kernel<<<block_num, MTGP32_TN>>>(
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
#if defined(CHECK)
    check_data(h_data, num_data, block_num);
#endif
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

int main(int argc, char** argv)
{
    // LARGE_SIZE is a multiple of 16
    int num_data = 10000000;
    int block_num;
    int num_unit;
    int r;
    int device = 0;
    mtgp32_kernel_status_t *d_status;

    ccudaSetDevice(device);

    if (argc >= 2) {
	errno = 0;
	block_num = strtol(argv[1], NULL, 10);
	if (errno) {
	    printf("%s number_of_block number_of_output\n", argv[0]);
	    return 1;
	}
	if (block_num < 1) {
	    printf("%s block_num should be greater than 1\n", argv[0]);
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
    num_unit = MTGP32_LS * block_num;
    ccudaMalloc((void**)&d_status,
			      sizeof(mtgp32_kernel_status_t) * block_num);
    r = num_data % num_unit;
    if (r != 0) {
	num_data = num_data + num_unit - r;
    }
    uint32_t seed = 1;
    uint32_t rc = 0;
    rc = init_check_data(block_num, seed);
    if (rc != 0) {
	printf("init check data ERROR\n");
	return 1;
    }
    initialize_mtgp_kernel(d_status, seed, block_num);
    int count = 2;
    for (int i = 0; i < count; i++) {
	make_uint32_random(d_status, num_data, block_num);
    }
    //make_single_random(d_status, num_data, block_num);
    free_check_data(block_num);

    // another initialization
    uint32_t seed_array[] = {1,2,3,4,5,6,7};
    rc = init_check_data_array(block_num, seed_array, 7);
    if (rc != 0) {
	printf("init check data array ERROR\n");
	return 1;
    }
    initialize_mtgp_array_kernel(d_status, seed_array, 7, block_num);
    for (int i = 0; i < count; i++) {
	make_uint32_random(d_status, num_data, block_num);
    }
    free_check_data(block_num);
    ccudaFree(d_status);
    return 0;
}





