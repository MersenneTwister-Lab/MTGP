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
#define BLOCK_NUM_MAX 200
#define TOTAL_THREAD_MAX 8000

__constant__ uint32_t maskB[TOTAL_THREAD_MAX];
__constant__ uint32_t maskC[TOTAL_THREAD_MAX];

/**
 * kernel I/O
 * This structure must be initialized before first use.
 */
struct mt32_kernel_status_t {
    uint32_t status[MTDC_N];
};

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
__device__ uint32_t para_rec(uint32_t X1, uint32_t X2, uint32_t Y,
			     uint32_t mat[2]) {
    uint32_t X = (X1 & MTDC_UPPER_MASK) | (X2 & MTDC_LOWER_MASK);

    X = (X >> 1) ^ Y ^ mat[X & 1];
    return X;
}

/**
 * The tempering function.
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered value.
 */
__device__ uint32_t temper(uint32_t x,  uint32_t maskB, uint32_t maskC) {

    x ^= x >> MTDC_SHIFT0;
    x ^= (x << MTDC_SHIFTB) & maskB;
    x ^= (x << MTDC_SHIFTC) & maskC;
    x ^= x >> MTDC_SHIFT1;
    return x;
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
__device__ void status_read(uint32_t status[MTDC_N],
			    const mt32_kernel_status_t *d_status,
			    int total_id) {
    int i;
    for (i = 0; i < MTDC_N; i++) {
	status[i] = d_status[total_id].status[i];
    }
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
__device__ void status_write(mt32_kernel_status_t *d_status,
			     const uint32_t status[],
			     int total_id) {
    int i;
    for (i = 0; i < MTDC_N; i++) {
	d_status[total_id].status[i] = status[i];
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
__global__ void mt32_uint32_kernel(mt32_kernel_status_t* d_status,
				   uint32_t* d_data, int size) {
    const int total_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t mat[2];
    uint32_t r;
    uint32_t o;
    mat[0] = 0;
    mat[1] = tex1Dfetch(tex_param_ref, total_id);
    uint32_t status[MTDC_N];

    // copy status data from global memory to shared memory.
    status_read(status, d_status, total_id);

    // main loop
    for (int i = 0; i < size; i += MTDC_N) {
	int j;
	for (j = 0; j < MTDC_N - MTDC_M; j++) {
	    r = para_rec(status[j], status[j + 1], status[j + MTDC_M], mat);
	    status[j] = r;
	    o = temper(r, maskB[total_id], maskC[total_id]);
	    d_data[size * total_id + i + j] = o;
	}
	for (; j < MTDC_N - 1; j++) {
	    r = para_rec(status[j], status[j + 1],
			 status[j + MTDC_M - MTDC_N],
			 mat);
	    status[j] = r;
	    o = temper(r, maskB[total_id], maskC[total_id]);
	    d_data[size * total_id + i + j] = o;
	}
	r = para_rec(status[MTDC_N - 1],
		     status[0],
		     status[MTDC_M - 1],
		     mat);
	status[j] = r;
	o = temper(r, maskB[total_id], maskC[total_id]);
	d_data[size * total_id + i + MTDC_N -1] = o;
    }
    // write back status for next call
    status_write(d_status, status, total_id);
}

#include "mtgp-cuda-common.c"

/**
 * This function is used to compare the outputs with C program's.
 * @param array data to be printed.
 * @param size size of array.
 * @param block number of blocks.
 */
void print_uint32_array(uint32_t array[], int size, int total_thread) {
    int b = size / total_thread;

    for (int j = 0; j < 5; j += 5) {
	printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32
	       " %10" PRIu32 " %10" PRIu32 "\n",
	       array[j], array[j + 1],
	       array[j + 2], array[j + 3], array[j + 4]);
    }
    for (int i = 1; i < total_thread; i++) {
	for (int j = -5; j < 5; j += 5) {
	    printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32
		   " %10" PRIu32 " %10" PRIu32 "\n",
		   array[b * i + j],
		   array[b * i + j + 1],
		   array[b * i + j + 2],
		   array[b * i + j + 3],
		   array[b * i + j + 4]);
	}
    }
    for (int j = -5; j < 0; j += 5) {
	printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32
	       " %10" PRIu32 " %10" PRIu32 "\n",
	       array[size + j],
	       array[size + j + 1],
	       array[size + j + 2],
	       array[size + j + 3],
	       array[size + j + 4]);
    }
}

void mt32_init_state(uint32_t status[0], uint32_t seed) {
    int i;
    for (i = 0; i < MTDC_N; i++) {
	status[i] = seed;
	seed = (UINT32_C(1812433253) * (seed ^ (seed >> 30))) + i + 1;
    }
}

/**
 * This function initializes kernel I/O data.
 * @param d_status output kernel I/O data.
 * @param params MTGP32 parameters. needed for the initialization.
 */
void init_kernel_data(mt32_kernel_status_t *d_status,
		      mt32_params_t params[],
		      int total_thread_num) {
    mt32_kernel_status_t* h_status = (mt32_kernel_status_t *) malloc(
	sizeof(mt32_kernel_status_t) * total_thread_num);

    if (h_status == NULL) {
	printf("failure in allocating host memory for kernel I/O data.\n");
	exit(8);
    }
    for (int i = 0; i < total_thread_num; i++) {
	mt32_init_state(&(h_status[i].status[0]), i + 1);
    }
    CUDA_SAFE_CALL(cudaMemcpy(d_status,
			      h_status,
			      sizeof(mt32_kernel_status_t) * total_thread_num,
			      cudaMemcpyHostToDevice));
    free(h_status);
}

/**
 * This function sets constants in device memory.
 * @param params input, MTGP32 parameters.
 */
void make_constant_param(const mt32_params_t params[],
			 int total_thread_num) {
    const int size1 = sizeof(uint32_t) * total_thread_num;
    uint32_t *h_maskB_tbl;
    uint32_t *h_maskC_tbl;
    h_maskB_tbl = (uint32_t *)malloc(size1);
    h_maskC_tbl = (uint32_t *)malloc(size1);
    if (h_maskB_tbl == NULL || h_maskB_tbl == NULL) {
	printf("failure in allocating host memory for constant table.\n");
	exit(1);
    }
    for (int i = 0; i < total_thread_num; i++) {
	h_maskB_tbl[i] = params[i].maskB;
	h_maskC_tbl[i] = params[i].maskC;
    }
    // copy from malloc area only
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(maskB, h_maskB_tbl, size1));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(maskC, h_maskC_tbl, size1));
    free(h_maskB_tbl);
    free(h_maskC_tbl);
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

    h_texture_tbl = (uint32_t *)malloc(size);
    if (h_texture_tbl == NULL) {
	printf("failure in allocating host memory for constant table.\n");
	exit(1);
    }
    for (int i = 0; i < total_thread_num; i++) {
	h_texture_tbl[i] = params[i].mat_a;
    }
    CUDA_SAFE_CALL(cudaMemcpy(d_texture_tbl, h_texture_tbl, size,
			      cudaMemcpyHostToDevice));
    tex_param_ref.filterMode = cudaFilterModePoint;
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_param_ref, d_texture_tbl, size));
    free(h_texture_tbl);
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param d_status kernel I/O data.
 * @param num_data number of data to be generated.
 */
void make_uint32_random(mt32_kernel_status_t* d_status,
			int num_data,
			int total_thread_num) {
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
    mt32_uint32_kernel<<< total_thread_num / 256, 256 >>>(
	d_status, d_data, num_data / total_thread_num);
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
    mt32_kernel_status_t *d_status;
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
					   256,
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

    total_thread_num = block_num * 256;
    num_unit = total_thread_num * 256 * MTDC_N;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_status,
			      sizeof(mt32_kernel_status_t) * total_thread_num));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_texture,
			      sizeof(uint32_t) * total_thread_num));
    r = num_data % num_unit;
    if (r != 0) {
	num_data = num_data + num_unit - r;
    }
    make_constant_param(MTDC_PARAM_TABLE, total_thread_num);
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
