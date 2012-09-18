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
#define JUMP3_162 "4e022b6965b54c1482e7b06800d7ab878156be8c27310501682c643c8030dfd29d1434deccbdb862130c09c21bf81cd5b0b3d9fc17d916ee6b6ef81098ac47ded1618183c3dd821b28e0d278f7c1c9fb0e2ffdf88c6c54d4e1f422b5a1d10a3e2c5829f66b3cda9d39a855346dbf627e05cf4fc866476b70807f64d87dc5f4a9e9036cfcbb9268aa1d16e294e59aed724d78a4c9b9c8d346b7346bddd68ab3355ca06e61152af12161d8737cfd2dcba50399a20a392a3c91e39421071f374d9d94831c49d289a00e4450dbaf3f90e1acd44b0bb855fb2d265429783a8b15051f63e12561fc77aceb9368448d8a3c65c0cf476a76608ffd2d829e7ff0be12422a66e0a074509cdf0c36ce7f97f003c475fcd0d9f0926b88d2f27d13dfd96b615b9a26040a54e836533574d5dfceba6848bd4c4d484e0267081a48433f3bed529a390ea14e82ff18bba8c7964325bbc414dcb1cb755ad6eb2e8539e10dd392ceb3575de88f54f2fb8170e05d1689e35b1b7eedae7865aeefd97da2ce8b2f78a8b1f8efaa4d65ec46b56ad38a57d914197fb945286dcfc96bdcccb3df026ece7f829c1e6910d4d3ed7f993c7954baf4f42850aeaff0755e15ae767e920fce4fef6e3b3fd3faba0712907250fc2b3eca8c4f7c84aeaaa3c0448807813e228a1c5ef30c2c79657545c7e6402be304ef3e5bd9d7c73d72579bbe0c1e9c934d7fd152921c742f1988443f731ac4554a4166c3ad51fd0c8261bdd8b1eaa5ccc464069fc285647e091db00184fccc3a8e153a20ad843e314dbea62d09c2476e5933bc8cec8af06d202b8bbd814326d70a92f230dacc0669a95a18df339fb5b2d8f14ca33f16b82291ab76a21078b91153d51b60b4c465cae36631bdb2772247f9429b56e53681f42f07f03769c24990e2038498d609f3d7aabdc4066fc1e5342e597ed7cd4ac0f35726465a9ee063f266ead0723d6ea0f5de4b877cab6e00f001189fd0a17de510390dcf31bf224d1644ddec3473e5d044a3befa0244d4c8c0684cd03d964dc35442f56e0eb0ebf4ffbe1574dcd964c767ab084e28f84e04111fb807f10d0dd09e276a8144cf74688cb5307252480739e9b01a16730814e218fe0efff07cfcae739ef56c151b8d918152db36a38d5b18754e94754ba467c91595030255813286020e53cfd1f081a4365584b24ab467d86a9303cc743ac02fb380447e84accb89181c671b001a9984bf3415106308ee16c7f8764c5bff729838abc7fee3217b4a2587511587b8694f91f3502b8500c095a8ba7e9ddf59efd90d82bebdb39bb19de94987a6d7681a11a66e44d2dea21f5520ea267b44172c074d9d227507a1b036367ec5aed86c622bb83b2b6218e41ad0cb2a8b1aeeb8796de185c873861e5f3a740169ec63a3c401af071d072a62874a660e8740a4f1978b8d53299099469ff4372a6e189e4518998231653f25e5ae40271238689a520384ad6a12307f467926d6a1d47cabee5d86acf85aa8cbfda915fc1ba2617182e40613fbfd9b828055e945ede5bad64961254523f7c1aa64ee35cad67bfe73981c7ccf0932e9d07cec230e694aab5e3e2e78771fb4a53ba6b43942cf09d407cf5aab2e9f136ff766912d20018099f6dd4f86dd650748dcda04bca6229dcbdd1ddefe7360664d29645a319a3afd7e33beeff8a60a9f044aa0c68af4b3f3bcd997f09c671a7ba41c490ec986fb30db2bfba937e35625dc131e580997db3483812e337fe382bbac08c92cc5d5ab4db7b06d8ecb35f0179b1901caaa7b3609531582fbac0d669bc1e6585e4dade37a44af384fd2ffb0937f8f60aa7cb7b790f4c0c724b210389807137aa8136ef8a7f826f0e4cf1430e73d1feec28047c889146943f6accade0280ebcfdd3db5ed35fa6153646cd926e3811bc1c339f756c4ac090c82ffe3f9caacf6cf112d1de3d12be4bf0bee95095a8f4d924ae4289d989c5299372"
#define MAX_BLOCK_NUM 20
static mtgp32_fast_t mtgp32[MAX_BLOCK_NUM];
static int init_check_data(int block_num, uint32_t seed)
{
    if (block_num > MAX_BLOCK_NUM) {
	printf("can't check block num > %d %d\n", MAX_BLOCK_NUM, block_num);
	return 1;
    }
    for (int i = 0; i < block_num; i++) {
	int rc = mtgp32_init(&mtgp32[i],
			     &mtgp32_params_fast_11213[0],
			     seed);
	if (rc) {
	    return rc;
	}
	if (i == 0) {
	    continue;
	}
	for (int j = 0; j < i; j++) {
	    mtgp32_fast_jump(&mtgp32[i], JUMP3_162);
	}
    }
    return 0;
}

static int init_check_data_array(int block_num, uint32_t seed_array[], int size)
{
    if (block_num > MAX_BLOCK_NUM) {
	printf("can't check block num > %d %d\n", MAX_BLOCK_NUM, block_num);
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
	    mtgp32_fast_jump(&mtgp32[i], JUMP3_162);
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
    if (block_num > MAX_BLOCK_NUM) {
	printf("can't check block num > %d %d\n", MAX_BLOCK_NUM, block_num);
	return;
    }
    for (int i = 0; i < block_num; i++) {
	for (int j = 0; j < size; j++) {
	    uint32_t r = mtgp32_genrand_uint32(&mtgp32[i]);
	    if (h_data[i * size + j] != r) {
		printf("mismatch i = %d, j = %d, data = %u, r = %u\n",
		       i, j, h_data[i * size + j], r);
		printf("check_data check N.G!\n");
		return;
	    }
	}
    }
    printf("check_data check O.K!\n");
}

static void check_status(struct mtgp32_kernel_status_t * h_status,
			 int block_num)
{
    int counter = 0;
    if (block_num > MAX_BLOCK_NUM) {
	printf("can't check block num > %d %d\n", MAX_BLOCK_NUM, block_num);
	return;
    }
    int large_size = mtgp32[0].status->large_size;
    for (int i = 0; i < block_num; i++) {
	for (int j = 0; j < MTGP32_N; j++) {
	    int idx = mtgp32[i].status->idx - MTGP32_N + 1 + large_size;
	    uint32_t x = h_status[i].status[j];
	    uint32_t r = mtgp32[i].status->array[(j + idx) % large_size];
	    if (j == 0) {
		x = x & mtgp32[i].params.mask;
		r = r & mtgp32[i].params.mask;
	    }
	    if (x != r) {
		printf("mismatch i = %d, j = %d, kernel = %x, C = %x\n",
		       i, j, x, r);
		printf("idx1 = %d, idx2 = %d, idx3 = %d, large_size = %d\n",
		       mtgp32[i].status->idx, idx, (j + idx) % large_size,
		       large_size);
		printf("check_status check N.G!\n");
		counter++;
	    }
	    if (counter > 10) {
		return;
	    }
	}
    }
    if (counter == 0) {
	printf("check_status check O.K!\n");
    }
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





