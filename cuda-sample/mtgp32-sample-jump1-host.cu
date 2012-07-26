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
static mtgp32_fast_t mtgp32[10];
static int init_check_data(int block_num, uint32_t seed)
{
    if (block_num > 10) {
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
    if (block_num > 10) {
	printf("can't check block num > 10 %d\n", block_num);
	return;
    }
    for (int i = 0; i < block_num; i++) {
	for (int j = 0; j < size; j++) {
	    uint32_t r = mtgp32_genrand_uint32(&mtgp32[i]);
	    if (h_data[i * size + j] != r) {
		printf("mismatch i = %d, j = %d, data = %x, r = %x\n",
		       i, j, h_data[i * size + j], r);
		break;
	    }
	}
    }
    printf("check O.K!\n");
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
    cudaError_t e;
    float gputime;
    cudaEvent_t start;
    cudaEvent_t end;

    printf("initializing and jumping mtgp32 kernel data.\n");
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

    ccudaEventDestroy(start);
    ccudaEventDestroy(end);
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
    uint32_t seed = 1234;
    //make_kernel_data32(d_status, MTGPDC_PARAM_TABLE, block_num);
    initialize_mtgp_kernel(d_status, seed, block_num);
#if defined(CHECK)
    init_check_data(block_num, seed);
#endif
    make_uint32_random(d_status, num_data, block_num);
    //make_single_random(d_status, num_data, block_num);
#if defined(CHECK)
    free_check_data(block_num);
#endif
    ccudaFree(d_status);
    return 0;
}





