/**
 * Sample Program for CUDA 2.3
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
#include "mt32dc-params521.c"
}
#define MEXP 521
#define BLOCK_NUM_MAX 1000
#define THREAD_NUM 64

texture<uint32_t, 1, cudaReadModeElementType> tex_param_ref;

/**
 * The function of the recursion formula calculation.
 *
 * @param[in] X1 the farthest part of state array.
 * @param[in] X2 the second farthest part of state array.
 * @param[in] Y a part of state array.
 * @param[in] bid block id.
 * @return output
 */
#if 0
__device__ uint32_t para_rec(uint32_t X1, uint32_t X2, uint32_t Y,
			     int total_id) {
    uint32_t X = (X1 & MTDC_UPPER_MASK) | (X2 & MTDC_LOWER_MASK);

    X = (X >> 1) ^ Y ^ tex1Dfetch(tex_param_ref, total_id * 4 + (X & 1));

    return X;
}
#endif
/**
 * The tempering function.
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered value.
 */
#if 0
__device__ uint32_t temper(uint32_t x,  uint32_t maskB, uint32_t maskC) {

    x ^= x >> MTDC_SHIFT0;
    x ^= (x << MTDC_SHIFTB) & maskB;
    x ^= (x << MTDC_SHIFTC) & maskC;
    x ^= x >> MTDC_SHIFT1;
    return x;
}
#endif
/**
 * Read the internal state vector from kernel I/O data, and
 * put them into shared memory.
 *
 * @param[out] status shared memory.
 * @param[in] d_status kernel I/O data
 * @param[in] bid block id
 * @param[in] tid thread id
 */
#if 0
__device__ void status_read(uint32_t status[MTDC_N],
			    const uint32_t *d_status,
			    int total_id,
			    int total_thread_num) {
    for (int i = 0; i < MTDC_N; i++) {
	status[i] = d_status[i * total_thread_num + total_id];
    }
}
#endif
/**
 * Read the internal state vector from shared memory, and
 * write them into kernel I/O data.
 *
 * @param[out] d_status kernel I/O data
 * @param[in] status shared memory.
 * @param[in] bid block id
 * @param[in] tid thread id
 */
#if 0
__device__ void status_write(uint32_t *d_status,
			     const uint32_t status[],
			     int total_id,
			     int total_thread_num) {
    for (int i = 0; i < MTDC_N; i++) {
	d_status[i * total_thread_num + total_id] = status[i];
    }
}
#endif
__device__ uint32_t get_tex_params(int idx) {
    return tex1Dfetch(tex_param_ref, idx);
}
/**
 * kernel function.
 * This function generates 32-bit unsigned integers in d_data
 *
 * @params[in,out] d_status kernel I/O data
 * @params[out] d_data output
 * @params[in] size number of output data requested.
 */
__global__ void mt32_uint32_kernel(uint32_t* d_status,
				   uint32_t* d_data,
				   int size) {
    const int total_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_thread_num = gridDim.x * blockDim.x;
    uint32_t x;
    uint32_t mat_a = get_tex_params(total_id * 4 + 1);
    uint32_t maskB = get_tex_params(total_id * 4 + 2);
    uint32_t maskC = get_tex_params(total_id * 4 + 3);
    uint32_t status[16];
    uint32_t st0;
    int p;

    // copy status data from global memory to shared memory.
    //status_read(status, d_status, total_id, total_thread_num);
    st0 = d_status[total_id];
    for (int i = 1; i < MTDC_N; i++) {
	status[i] = d_status[total_thread_num * i + total_id];
    }

    p = 0;
    for (int i = 0; i < size; i++) {
 	x = (st0 & MTDC_UPPER_MASK) | (status[p] & MTDC_LOWER_MASK);
	x = (x >> 1) ^ status[(p + MTDC_M - 1) & 0x0f]
	    ^ (((x & 1) == 1) ? mat_a : 0);
	st0 = status[p];
	status[p] = x;
	x ^= x >> MTDC_SHIFT0;
	x ^= (x << MTDC_SHIFTB) & maskB;
	x ^= (x << MTDC_SHIFTC) & maskC;
	x ^= x >> MTDC_SHIFT1;
	d_data[total_thread_num * i + total_id] = x;
	p = (p + 1) & 0x0f;
    }
    // write back status for next call
    //status_write(d_status, status, total_id, total_thread_num);
    d_status[total_id] = st0;
#pragma unroll 1
    for (int i = 1; i < MTDC_N; i++) {
	d_status[total_thread_num * i + total_id] = status[(i + p - 1) & 0x0f];
    }
}

/*
 * Sample Program for CUDA 2.3
 * written by M.Saito (saito@math.sci.hiroshima-u.ac.jp)
 *
 * This sample uses texture reference.
 * The generation speed of PRNG using texture is faster than using
 * constant tabel on Geforce GTX 260.
 *
 * MTGP32-11213
 * This program generates 32-bit unsigned integers.
 * The period of generated integers is 2<sup>23209</sup>-1.
 * This also generates single precision floating point numbers.
 */

int get_suitable_block_num(int word_size, int thread_num, int large_size) {
    cudaDeviceProp dev;
    CUdevice cuDevice;
    int max_thread_dev;
    int max_block, max_block_mem, max_block_dev;
    int major, minor, ver;

    CUDA_SAFE_CALL(cudaGetDeviceProperties(&dev, 0));
    cuDeviceGet(&cuDevice, 0);
    cuDeviceComputeCapability(&major, &minor, cuDevice);
    max_block_mem = dev.sharedMemPerBlock / (large_size * word_size);
    if (major == 9999 && minor == 9999) {
	return -1;
    }
    ver = major * 100 + minor;
    if (ver <= 101) {
	max_thread_dev = 768;
    } else if (ver <= 103) {
	max_thread_dev = 1024;
    } else {
	max_thread_dev = 1024;
    }
    max_block_dev = max_thread_dev / thread_num;
    if (max_block_mem < max_block_dev) {
	max_block = max_block_mem;
    } else {
	max_block = max_block_dev;
    }
    return max_block * dev.multiProcessorCount;
}

/**
 * This function is used to compare the outputs with C program's.
 * @param array data to be printed.
 * @param size size of array.
 * @param block number of blocks.
 */
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

void mt32_init_state(uint32_t *status, int total_thread_id,
		     int total_thread_num, uint32_t seed) {
    int i;
    for (i = 0; i < MTDC_N; i++) {
	status[i * total_thread_num + total_thread_id] = seed;
	seed = (UINT32_C(1812433253) * (seed ^ (seed >> 30))) + i + 1;
    }
#ifdef DEBUG
    if (total_thread_id == 0 || total_thread_id == 1) {
	printf("state after initialization\n");
	for(i = 0; i < MTDC_N; i++) {
	    printf("%10"PRIu32" ",
		   status[i * total_thread_num + total_thread_id]);
	    if (i % 5 == 4) {
		printf("\n");
	    }
	}
	printf("\n");
    }
#endif
}

/**
 * This function initializes kernel I/O data.
 * @param d_status output kernel I/O data.
 * @param params MTGP32 parameters. needed for the initialization.
 */
void init_kernel_data(uint32_t *d_status,
		      mt32_params_t params[],
		      int total_thread_num) {
    uint32_t* h_status = (uint32_t *) malloc(
	sizeof(uint32_t) * MTDC_N * total_thread_num);

    if (h_status == NULL) {
	printf("failure in allocating host memory for kernel I/O data.\n");
	exit(8);
    }
    for (int i = 0; i < total_thread_num; i++) {
	mt32_init_state(h_status, i, total_thread_num, i + 1);
    }
    CUDA_SAFE_CALL(cudaMemcpy(d_status,
			      h_status,
			      sizeof(uint32_t) * MTDC_N * total_thread_num,
			      cudaMemcpyHostToDevice));
    free(h_status);
}

/**
 * This function sets texture lookup table.
 * @param params input, MTGP32 parameters.
 * @param d_texture_tbl device memory used for texture bind
 * @param block_num block number used for kernel call
 */
void make_texture(const mt32_params_t params[],
		  uint32_t *d_texture_tbl,
		  int total_thread_num) {
    const int size = sizeof(uint32_t) * total_thread_num;
    uint32_t *h_texture_tbl;

    h_texture_tbl = (uint32_t *)malloc(size * 4);
    if (h_texture_tbl == NULL) {
	printf("failure in allocating host memory for constant table.\n");
	exit(1);
    }
    for (int i = 0; i < total_thread_num; i++) {
	h_texture_tbl[i * 4] = 0;
	h_texture_tbl[i * 4 + 1] = params[i].mat_a;
	h_texture_tbl[i * 4 + 2] = params[i].maskB;
	h_texture_tbl[i * 4 + 3] = params[i].maskC;
    }
    CUDA_SAFE_CALL(cudaMemcpy(d_texture_tbl, h_texture_tbl, size * 4,
			      cudaMemcpyHostToDevice));
    tex_param_ref.filterMode = cudaFilterModePoint;
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_param_ref, d_texture_tbl, size * 4));
    free(h_texture_tbl);
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param d_status kernel I/O data.
 * @param num_data number of data to be generated.
 */
void make_uint32_random(uint32_t* d_status,
			int num_data,
			int total_thread_num) {
    uint32_t* d_data;
    unsigned int timer = 0;
    uint32_t* h_data;
    cudaError_t e;
    float gputime;
    dim3 block;
    dim3 thread;

    printf("generating 32-bit unsigned random numbers.\n");
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, sizeof(uint32_t) * num_data));
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    h_data = (uint32_t *) malloc(sizeof(uint32_t) * num_data);
    if (h_data == NULL) {
	printf("failure in allocating host memory for output data.\n");
	exit(1);
    }
#ifdef DEBUG
    printf("total_thread_num = %d\n", total_thread_num);
    printf("THREAD_NUM = %d\n", THREAD_NUM);
    printf("num_data = %d\n", num_data);
#endif
    block.x = total_thread_num / THREAD_NUM;
    block.y = 1;
    block.z = 1;
    thread.x = THREAD_NUM;
    thread.y = 1;
    thread.z = 1;
    CUT_SAFE_CALL(cutStartTimer(timer));
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

    /* kernel call */
    mt32_uint32_kernel<<< block, thread >>>(d_status, d_data, num_data / total_thread_num);
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
    print_uint32_array(h_data, num_data, total_thread_num);
    printf("generated numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", gputime);
    printf("Samples per second: %E \n", num_data / (gputime * 0.001));
    CUT_SAFE_CALL(cutDeleteTimer(timer));
    //free memories
    free(h_data);
    CUDA_SAFE_CALL(cudaFree(d_data));
}

int main(int argc, char *argv[])
{
    // LARGE_SIZE is a multiple of 16
    int num_data = 10000000;
    int block_num;
    int num_unit;
    int r;
    int total_thread_num;
    uint32_t *d_status;
    uint32_t *d_texture;

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
					   MTDC_N);
	if (block_num <= 0) {
	    printf("can't calculate sutable number of blocks.\n");
	    return 1;
	}
	printf("the suitable number of blocks for device 0 "
	       "will be multiple of %d\n", block_num);
	return 1;
    }
    CUT_DEVICE_INIT(argc, argv);

    total_thread_num = block_num * THREAD_NUM;
    num_unit = total_thread_num * MTDC_N;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_status,
			      sizeof(uint32_t) * MTDC_N * total_thread_num));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_texture,
			      sizeof(uint32_t) * total_thread_num * 4));
    r = num_data % num_unit;
    if (r != 0) {
	num_data = num_data + num_unit - r;
    }
    //make_constant_param(MTDC_PARAM_TABLE, total_thread_num);
    make_texture(MTDC_PARAM_TABLE, d_texture, total_thread_num);
    init_kernel_data(d_status, MTDC_PARAM_TABLE, total_thread_num);
    make_uint32_random(d_status, num_data, total_thread_num);

    //finalize
    CUDA_SAFE_CALL(cudaFree(d_status));
    CUDA_SAFE_CALL(cudaFree(d_texture));
#ifdef NEED_PROMPT
    CUT_EXIT(argc, argv);
#endif
}
