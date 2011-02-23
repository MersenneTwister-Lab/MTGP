#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <util.h>

// S1, S2, S3, and M are all constants, and z is part of the
// private per-thread generator state.
__device__ unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M)
{
    unsigned b=(((z << S1) ^ z) >> S2);
    return z = (((z & M) << S3) ^ b);
}

// A and C are constants
__device__ unsigned LCGStep(unsigned &z, unsigned A, unsigned C)
{
    return z=(A*z+C);
}

__device__ uint32_t HybridTaus(unsigned& z1,
			    unsigned& z2,
			    unsigned& z3,
			    unsigned& z4)
{
    // Combined period is lcm(p1,p2,p3,p4)~ 2^121
    //return 2.3283064365387e-10 * (              // Periods
    return (              // Periods
	TausStep(z1, 13, 19, 12, 4294967294UL) ^  // p1=2^31-1
	TausStep(z2, 2, 25, 4, 4294967288UL) ^    // p2=2^30-1
	TausStep(z3, 3, 11, 17, 4294967280UL) ^   // p3=2^28-1
	LCGStep(z4, 1664525, 1013904223UL)        // p4=2^32
	);
}

__global__ void hybrid_kernel(uint32_t* d_data, int size)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    unsigned z1 = bid * 13 + tid + 1;
    unsigned z2 = tid * 29 + bid + 9;
    unsigned z3 = bid * 7 + tid * 97 + 8;
    unsigned z4 = bid * 19937 + tid * 607 + 2;

    for (int i = 0; i < size; i += blockDim.x) {
	d_data[size * bid + i + tid] = HybridTaus(z1, z2, z3, z4);
    }
}

__global__ void hybrid_reduce(uint32_t* d_data, int size)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    unsigned z1 = bid * 13 + tid + 1;
    unsigned z2 = tid * 29 + bid + 9;
    unsigned z3 = bid * 7 + tid * 97 + 8;
    unsigned z4 = bid * 19937 + tid * 607 + 2;
    unsigned xsum = 0;

    for (int i = 0; i < size; i += blockDim.x) {
	xsum ^= HybridTaus(z1, z2, z3, z4);
    }
    d_data[blockDim.x * bid + tid] = xsum;
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param[in] num_data number of data to be generated.
 */
void make_hybrid_random(int num_data,
			int block_num) {
    uint32_t* d_data;
    uint32_t* h_data;
    cudaError_t e;

    printf("generating uint32_t random numbers.\n");
    ccudaMalloc((void**)&d_data, sizeof(uint32_t) * num_data);
    /* cutCreateTimer(&timer); */
    float elapsed_time_ms=0.0f;
    cudaEvent_t start, stop;
    ccudaEventCreate(&start);
    ccudaEventCreate(&stop);
    h_data = (uint32_t *) malloc(sizeof(uint32_t) * num_data);
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
    hybrid_kernel<<< block_num, THREAD_NUM>>>(d_data, num_data / block_num);
    ccudaEventRecord(stop, 0);
    ccudaEventSynchronize(stop);
    ccudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    /* CUT_SAFE_CALL(cutStopTimer(timer)); */
    ccudaMemcpy(h_data,
		d_data,
		sizeof(uint32_t) * num_data,
		cudaMemcpyDeviceToHost);
    /* gputime = cutGetTimerValue(timer);*/
    ccudaEventElapsedTime(&elapsed_time_ms, start, stop);

    print_uint32_array(h_data, num_data, block_num);
    printf("generated numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", elapsed_time_ms);
    printf("Samples per second: %E \n", num_data / (elapsed_time_ms * 0.001));
    /* CUT_SAFE_CALL(cutDeleteTimer(timer));*/
    ccudaEventDestroy(start);
    ccudaEventDestroy(stop);
    //free memories
    free(h_data);
    ccudaFree(d_data);
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param[in] num_data number of data to be generated.
 */
void make_hybrid_reduce(int num_data,
			int block_num) {
    uint32_t* d_data;
    uint32_t* h_data;
    cudaError_t e;

    printf("generating uint32_t random numbers.\n");
    ccudaMalloc((void**)&d_data, sizeof(uint32_t) * block_num * THREAD_NUM);
    /* CUT_SAFE_CALL(cutCreateTimer(&timer)); */
    float elapsed_time_ms=0.0f;
    cudaEvent_t start, stop;
    ccudaEventCreate(&start);
    ccudaEventCreate(&stop);

    h_data = (uint32_t *) malloc(sizeof(uint32_t) * block_num * THREAD_NUM);
    if (h_data == NULL) {
	printf("failure in allocating host memory for output data.\n");
	exit(1);
    }
    /* CUT_SAFE_CALL(cutStartTimer(timer)); */
    ccudaEventRecord(start, 0);

    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

    /* kernel call */
    hybrid_reduce<<< block_num, THREAD_NUM>>>(d_data, num_data / block_num);
    ccudaEventRecord(stop, 0);
    ccudaEventSynchronize(stop);
    ccudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    /* CUT_SAFE_CALL(cutStopTimer(timer)); */
    ccudaEventElapsedTime(&elapsed_time_ms, start, stop);
    ccudaMemcpy(h_data, d_data, sizeof(uint32_t) * block_num * THREAD_NUM,
		cudaMemcpyDeviceToHost);
    /* gputime = cutGetTimerValue(timer); */
    printf("generated numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", elapsed_time_ms);
    printf("Samples per second: %E \n", num_data / (elapsed_time_ms * 0.001));
    /* CUT_SAFE_CALL(cutDeleteTimer(timer)); */
    ccudaEventDestroy(start);
    ccudaEventDestroy(stop);
    //free memories
    free(h_data);
    ccudaFree(d_data);
}

int main(int argc, char** argv)
{
    int num_data = 1;
    int block_num;
    int num_unit;
    int r;

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
    num_unit = THREAD_NUM * block_num;
    r = num_data % num_unit;
    if (r != 0) {
	num_data = num_data + num_unit - r;
    }
    make_hybrid_random(num_data, block_num);
    make_hybrid_reduce(num_data, block_num);
}
