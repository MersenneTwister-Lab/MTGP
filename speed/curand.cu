/**
 * Sample Program for CUDA 3.2
 * written by M.Saito (saito@math.sci.hiroshima-u.ac.jp)
 *
 * This sample uses texture reference.
 * The generation speed of PRNG using texture is faster than using
 * constant tabel on Geforce GTX 260.
 *
 * MTGP32-11213
 * This program generates 32-bit unsigned integers.
 * The period of generated integers is 2<sup>11213</sup>-1.
 * This also generates single precision floating point numbers.
 */
#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <errno.h>
#include <stdlib.h>
#include <util.h>

__global__ void init_kernel(curandState * d_status) {
    const int globalid = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state = d_status[globalid];
    curand_init(1234, globalid, 0, &state);
    d_status[globalid] = state;
}

/**
 * kernel function.
 * This function generates 32-bit unsigned integers in d_data
 *
 * @params[in,out] d_status kernel I/O data
 * @params[out] d_data output
 * @params[in] size number of output data requested.
 */
__global__ void random_kernel(curandState * d_status,
			      float * d_data,
			      int size) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int globalid = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state = d_status[globalid];

    // main loop
    for (int i = 0; i < size; i += blockDim.x) {
	d_data[size * bid + i + tid] = curand_uniform(&state);
    }
}

/**
 * kernel function.
 * This function generates 32-bit unsigned integers in d_data
 *
 * @params[in,out] d_status kernel I/O data
 * @params[out] d_data output
 * @params[in] size number of output data requested.
 */
__global__ void reduce_kernel(curandState * d_status,
			      float * d_data,
			      int size) {
    const int globalid = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state = d_status[globalid];
    float sum = 0;

    for (int i = 0; i < size; i += blockDim.x) {
	sum += curand_uniform(&state);
    }
    d_data[globalid] = sum;
}


/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param d_status kernel I/O data.
 * @param num_data number of data to be generated.
 */
void make_init(curandState * d_status, int block_num, int thread_num) {
    cudaError_t e;

    printf("initializing curand.\n");
    /* cutCreateTimer(&timer); */
    float elapsed_time_ms=0.0f;
    cudaEvent_t start, stop;
    ccudaEventCreate(&start);
    ccudaEventCreate(&stop);

    ccudaEventRecord(start, 0);
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

    /* kernel call */
    init_kernel<<< block_num, thread_num>>>(d_status);
    ccudaEventRecord(stop, 0);
    ccudaEventSynchronize(stop);

    ccudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    /* cutStopTimer(timer); */
    ccudaEventElapsedTime(&elapsed_time_ms, start, stop);

    printf("initializing\n");
    printf("Processing time: %f (ms)\n", elapsed_time_ms);
    ccudaEventDestroy(start);
    ccudaEventDestroy(stop);
}

void make_float(curandState* d_status, int num_data,
		 int block_num, int thread_num) {
    float* d_data;
    float* h_data;
    cudaError_t e;

    printf("generating float random numbers.\n");
    ccudaMalloc((void**)&d_data, sizeof(float) * num_data);

    /* cutCreateTimer(&timer); */
    float elapsed_time_ms=0.0f;
    cudaEvent_t start, stop;
    ccudaEventCreate(&start);
    ccudaEventCreate(&stop);

    h_data = (float *) malloc(sizeof(float) * num_data);
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
    random_kernel<<< block_num, THREAD_NUM>>>(
	d_status, d_data, num_data / block_num);
    ccudaEventRecord(stop, 0);
    ccudaEventSynchronize(stop);

    ccudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    /* cutStopTimer(timer); */
    ccudaEventElapsedTime(&elapsed_time_ms, start, stop);
    ccudaMemcpy(h_data, d_data,
	       sizeof(float) * num_data,
	       cudaMemcpyDeviceToHost);

    /* gputime = cutGetTimerValue(timer);*/
    print_float_array(h_data, num_data, block_num);
    printf("generated numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", elapsed_time_ms);
    printf("Samples per second: %E \n", num_data / elapsed_time_ms);
    /* cutDeleteTimer(timer); */
    ccudaEventDestroy(start);
    ccudaEventDestroy(stop);

    //free memories
    free(h_data);
    ccudaFree(d_data);
}

void make_reduce(curandState* d_status, int num_data,
		 int block_num, int thread_num) {
    float* d_data;
    float* h_data;
    cudaError_t e;

    printf("generating float random numbers.\n");
    ccudaMalloc((void**)&d_data, sizeof(float) * block_num * thread_num);

    /* cutCreateTimer(&timer); */
    float elapsed_time_ms=0.0f;
    cudaEvent_t start, stop;
    ccudaEventCreate(&start);
    ccudaEventCreate(&stop);

    h_data = (float *) malloc(sizeof(float) *block_num * thread_num);
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
    reduce_kernel<<< block_num, thread_num>>>(
	d_status, d_data, num_data / block_num);
    ccudaEventRecord(stop, 0);
    ccudaEventSynchronize(stop);

    ccudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    /* cutStopTimer(timer); */
    ccudaEventElapsedTime(&elapsed_time_ms, start, stop);
    ccudaMemcpy(h_data, d_data,
	       sizeof(float) * block_num * thread_num,
	       cudaMemcpyDeviceToHost);

    /* gputime = cutGetTimerValue(timer);*/
    printf("generated numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", elapsed_time_ms);
    printf("Samples per second: %E \n", num_data / elapsed_time_ms);
    /* cutDeleteTimer(timer); */
    ccudaEventDestroy(start);
    ccudaEventDestroy(stop);

    //free memories
    free(h_data);
    ccudaFree(d_data);
}


int main(int argc, char *argv[])
{
    // LARGE_SIZE is a multiple of 16
    int num_data = 10000000;
    int block_num;
    int thread_num = THREAD_NUM;
    int num_unit;
    int r;
    curandState *d_status;

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

    num_unit = block_num * thread_num;
    ccudaMalloc((void**)&d_status,
	       sizeof(curandState) * block_num * thread_num);
    r = num_data % num_unit;
    if (r != 0) {
	num_data = num_data + num_unit - r;
    }
    make_init(d_status, block_num, thread_num);
    make_float(d_status, num_data, block_num, thread_num);
    make_reduce(d_status, num_data, block_num, thread_num);

    //finalize
    ccudaFree(d_status);
}
