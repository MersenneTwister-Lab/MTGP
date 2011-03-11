#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <util.h>

#include <stdint.h>

const int thread_num = 256;
///////////////////////////////////////////////////////////////////////////////
// Public constants

const unsigned WarpStandard_K=32;
const unsigned WarpStandard_REG_COUNT=3;
const unsigned WarpStandard_STATE_WORDS=32;

const uint32_t WarpStandard_TEST_DATA[WarpStandard_STATE_WORDS]={
    0x8cf35fea, 0xe1dd819e, 0x4a7d0a8e, 0xe0c05911, 0xfd053b8d,
    0x30643089, 0x6f6ac111, 0xc4869595, 0x9416b7be, 0xe6d329e8,
    0x5af0f5bf, 0xc5c742b5, 0x7197e922, 0x71aa35b4, 0x2070b9d1,
    0x2bb34804, 0x7754a517, 0xe725315e, 0x7f9dd497, 0x043b58bf,
    0x83ffa33d, 0x2532905a, 0xbdfe0c8a, 0x16f68671, 0x0d14da2e,
    0x847efd5f, 0x1edeec64, 0x1bebdf9b, 0xf74d4ff3, 0xd404774b,
    0x8ee32599, 0xefe0c405
};

///////////////////////////////////////////////////////////////////////////////
// Private constants

const char *WarpStandard_name="WarpRNG[CorrelatedU32Rng;k=32;g=16;rs=0;w=32;n=1024;hash=deac2e12ec6e615]";
const char *WarpStandard_post_processing="addtaps";
__device__ const unsigned WarpStandard_Q[2][32]={
  {29,24,5,23,14,26,11,31,9,3,1,28,0,2,22,20,18,15,
   27,13,10,16,8,17,25,12,19,30,7,6,4,21},
  {5,14,28,24,19,13,0,17,11,20,7,10,6,15,2,9,8,23,4,
   30,12,25,3,21,26,27,31,18,22,16,29,1}
};
const unsigned WarpStandard_Z0=2;
__device__ const unsigned WarpStandard_Z1[32]={
  0,1,0,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1};

const unsigned WarpStandard_SHMEM_WORDS=32;
//const unsigned WarpStandard_GMEM_WORDS=0;

////////////////////////////////////////////////////////////////////////////////
// Public functions

__device__ void WarpStandard_LoadState(const unsigned *seed,
				       unsigned *regs, unsigned *shmem)
{
  unsigned offset=threadIdx.x % 32;  unsigned base=threadIdx.x-offset;
  // setup constants
  regs[0]=WarpStandard_Z1[offset];
  regs[1]=base + WarpStandard_Q[0][offset];
  regs[2]=base + WarpStandard_Q[1][offset];
  // Setup state
  unsigned stateOff=blockDim.x * blockIdx.x * 1 + threadIdx.x * 1;
  shmem[threadIdx.x]=seed[stateOff];
}

__device__ void WarpStandard_SaveState(const unsigned *regs,
				       const unsigned *shmem, unsigned *seed)
{
  unsigned stateOff=blockDim.x * blockIdx.x * 1 + threadIdx.x * 1;
  seed[stateOff] = shmem[threadIdx.x];
}

__device__ unsigned WarpStandard_Generate(unsigned *regs, unsigned *shmem)
{
#if __DEVICE_EMULATION__
    __syncthreads();
#endif
  unsigned t0=shmem[regs[1]], t1=shmem[regs[2]];
  unsigned res=(t0<<WarpStandard_Z0) ^ (t1>>regs[0]);
#if __DEVICE_EMULATION__
    __syncthreads();
#endif
  shmem[threadIdx.x]=res;
  return t0+t1;
}


extern __shared__ unsigned shmem[];

__global__ void warp_kernel(uint32_t* d_data, int size,
			    const uint32_t seed[])
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    unsigned rngRegs[WarpStandard_REG_COUNT];

    WarpStandard_LoadState(seed, rngRegs, shmem);

    for (int i = 0; i < size; i += blockDim.x) {
	d_data[size * bid + i + tid]
	    = WarpStandard_Generate(rngRegs, shmem);
    }
}

__global__ void float_kernel(float* d_data, int size,
			    const uint32_t seed[])
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    unsigned rngRegs[WarpStandard_REG_COUNT];
    uint32_t r;

    WarpStandard_LoadState(seed, rngRegs, shmem);

    for (int i = 0; i < size; i += blockDim.x) {
	r = WarpStandard_Generate(rngRegs, shmem);
#if defined(FLOAT_MASK)
	r = (r >> 9) | 0x3f800000U;
	d_data[size * bid + i + tid] = __int_as_float(r) - 1.0f;
#else
	d_data[size * bid + i + tid] = 2.3283064365387e-10 * r;
#endif
    }
}

__global__ void warp_reduce_kernel(uint32_t* d_data, int size,
				   const uint32_t seed[])
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    unsigned rngRegs[WarpStandard_REG_COUNT];
    unsigned xsum = 0;

    WarpStandard_LoadState(seed, rngRegs, shmem);

    for (int i = 0; i < size; i += blockDim.x) {
	xsum ^= WarpStandard_Generate(rngRegs, shmem);
    }
    d_data[blockDim.x * bid + tid] = xsum;
}

__global__ void float_reduce(float * d_data, int size,
			     const uint32_t seed[])
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    unsigned rngRegs[WarpStandard_REG_COUNT];
    float sum = 0;
    uint32_t r;

    WarpStandard_LoadState(seed, rngRegs, shmem);

    for (int i = 0; i < size; i += blockDim.x) {
	r = WarpStandard_Generate(rngRegs, shmem);
#if defined(FLOAT_MASK)
	r = (r >> 9) | 0x3f800000U;
	sum += __int_as_float(r) - 1.0f;
#else
	sum += 2.3283064365387e-10 * r;
#endif
    }
    d_data[blockDim.x * bid + tid] = sum;
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param[in] num_data number of data to be generated.
 */
void make_warp_random(int num_data,
		      int block_num) {
    uint32_t* d_data;
    uint32_t* d_seed;
    uint32_t* h_data;
    uint32_t* h_seed;
    cudaError_t e;
    unsigned rngsPerBlock = thread_num / WarpStandard_K;
    unsigned sharedMemBytesPerBlock
	= rngsPerBlock * WarpStandard_SHMEM_WORDS * 4;
    int seed_size = rngsPerBlock * block_num * 32;
    uint32_t tmp;

    printf("generating unsigned random numbers.\n");
    ccudaMalloc((void**)&d_data, sizeof(uint32_t) * num_data);
    ccudaMalloc((void**)&d_seed, sizeof(uint32_t) * seed_size);
    /* cutCreateTimer(&timer); */
    float elapsed_time_ms=0.0f;
    cudaEvent_t start, stop;
    ccudaEventCreate(&start);
    ccudaEventCreate(&stop);

    h_data = (uint32_t *) malloc(sizeof(uint32_t) * num_data);
    h_seed = (uint32_t *) malloc(sizeof(uint32_t) * seed_size);
    if (h_data == NULL || h_seed == NULL) {
	printf("failure in allocating host memory for output data.\n");
	exit(1);
    }
    memcpy(h_seed, WarpStandard_TEST_DATA, sizeof(uint32_t) * 32);
    tmp = h_seed[17];
    for (int i = 0; i < seed_size - 35; i++) {
	tmp = (tmp >> 11) * h_seed[i] + i;
	h_seed[i + 32] = tmp ^ (tmp << 3);
    }
    ccudaMemcpy(d_seed, h_seed, sizeof(uint32_t) * seed_size,
		cudaMemcpyHostToDevice);
    /* cutStartTimer(timer); */
    ccudaEventRecord(start, 0);
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

    /* kernel call */
    warp_kernel<<< block_num, thread_num, sharedMemBytesPerBlock>>>
	(d_data, num_data / block_num, d_seed);
    ccudaEventRecord(stop, 0);
    ccudaEventSynchronize(stop);
    ccudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    /* cutStopTimer(timer); */

    ccudaMemcpy(h_data, d_data,
	       sizeof(uint32_t) * num_data, cudaMemcpyDeviceToHost);
    ccudaEventElapsedTime(&elapsed_time_ms, start, stop);
    print_uint32_array(h_data, num_data, block_num);
    printf("generated numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", elapsed_time_ms);
    printf("Samples per second: %E \n", num_data / (elapsed_time_ms * 0.001));
    /* cutDeleteTimer(timer); */
    ccudaEventDestroy(start);
    ccudaEventDestroy(stop);

    //free memories
    free(h_data);
    free(h_seed);
    ccudaFree(d_data);
    ccudaFree(d_seed);
}

void make_float_random(int num_data,
		      int block_num) {
    float* d_data;
    uint32_t* d_seed;
    float* h_data;
    uint32_t* h_seed;
    cudaError_t e;
    unsigned rngsPerBlock = thread_num / WarpStandard_K;
    unsigned sharedMemBytesPerBlock
	= rngsPerBlock * WarpStandard_SHMEM_WORDS * 4;
    int seed_size = rngsPerBlock * block_num * 32;
    uint32_t tmp;

    printf("generating unsigned random numbers.\n");
    ccudaMalloc((void**)&d_data, sizeof(float) * num_data);
    ccudaMalloc((void**)&d_seed, sizeof(uint32_t) * seed_size);
    /* cutCreateTimer(&timer); */
    float elapsed_time_ms=0.0f;
    cudaEvent_t start, stop;
    ccudaEventCreate(&start);
    ccudaEventCreate(&stop);

    h_data = (float *) malloc(sizeof(float) * num_data);
    h_seed = (uint32_t *) malloc(sizeof(uint32_t) * seed_size);
    if (h_data == NULL || h_seed == NULL) {
	printf("failure in allocating host memory for output data.\n");
	exit(1);
    }
    memcpy(h_seed, WarpStandard_TEST_DATA, sizeof(uint32_t) * 32);
    tmp = h_seed[17];
    for (int i = 0; i < seed_size - 35; i++) {
	tmp = (tmp >> 11) * h_seed[i] + i;
	h_seed[i + 32] = tmp ^ (tmp << 3);
    }
    ccudaMemcpy(d_seed, h_seed, sizeof(uint32_t) * seed_size,
		cudaMemcpyHostToDevice);
    /* cutStartTimer(timer); */
    ccudaEventRecord(start, 0);
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

    /* kernel call */
    float_kernel<<< block_num, thread_num, sharedMemBytesPerBlock>>>
	(d_data, num_data / block_num, d_seed);
    ccudaEventRecord(stop, 0);
    ccudaEventSynchronize(stop);
    ccudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    /* cutStopTimer(timer); */

    ccudaMemcpy(h_data, d_data,
	       sizeof(float) * num_data, cudaMemcpyDeviceToHost);
    ccudaEventElapsedTime(&elapsed_time_ms, start, stop);
    print_float_array(h_data, num_data, block_num);
    printf("generated numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", elapsed_time_ms);
    printf("Samples per second: %E \n", num_data / (elapsed_time_ms * 0.001));
    /* cutDeleteTimer(timer); */
    ccudaEventDestroy(start);
    ccudaEventDestroy(stop);

    //free memories
    free(h_data);
    free(h_seed);
    ccudaFree(d_data);
    ccudaFree(d_seed);
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param[in] num_data number of data to be generated.
 */
void make_warp_reduced(int num_data,
		       int block_num) {
    uint32_t* d_data;
    uint32_t* d_seed;
    uint32_t* h_data;
    uint32_t* h_seed;
    cudaError_t e;
    unsigned rngsPerBlock = thread_num / WarpStandard_K;
    unsigned sharedMemBytesPerBlock
	= rngsPerBlock * WarpStandard_SHMEM_WORDS * 4;
    int seed_size = rngsPerBlock * block_num * 32;
    int all_threads = block_num * thread_num;
    uint32_t tmp;

    printf("generating unsigned random numbers.\n");
    cudaMalloc((void**)&d_data, sizeof(uint32_t) * all_threads);
    cudaMalloc((void**)&d_seed, sizeof(uint32_t) * seed_size);
    /* cutCreateTimer(&timer); */
    float elapsed_time_ms=0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_data = (uint32_t *) malloc(sizeof(uint32_t) * all_threads);
    h_seed = (uint32_t *) malloc(sizeof(uint32_t) * seed_size);
    if (h_data == NULL || h_seed == NULL) {
	printf("failure in allocating host memory for output data.\n");
	exit(1);
    }
    memcpy(h_seed, WarpStandard_TEST_DATA, sizeof(uint32_t) * 32);
    tmp = h_seed[17];
    for (int i = 0; i < seed_size -35; i++) {
	tmp = (tmp >> 11) * h_seed[i] + i;
	h_seed[i + 32] = tmp ^ (tmp << 3);
    }
    cudaMemcpy(d_seed, h_seed, sizeof(uint32_t) * seed_size,
	       cudaMemcpyHostToDevice);
    /* cutStartTimer(timer); */
    cudaEventRecord(start, 0);
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

    /* kernel call */
    warp_reduce_kernel<<< block_num, thread_num, sharedMemBytesPerBlock>>>
	(d_data, num_data / block_num, d_seed);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    /* cutStopTimer(timer); */
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    cudaMemcpy(h_data, d_data, sizeof(uint32_t) * all_threads,
	       cudaMemcpyDeviceToHost);
    printf("reduce generation numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", elapsed_time_ms);
    printf("Samples per second: %E \n", num_data / (elapsed_time_ms * 0.001));
    /* cutDeleteTimer(timer); */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //free memories
    free(h_data);
    free(h_seed);
    ccudaFree(d_data);
    ccudaFree(d_seed);
}
/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param[in] num_data number of data to be generated.
 */
void make_float_reduced(int num_data,
			int block_num) {
    float* d_data;
    uint32_t* d_seed;
    float* h_data;
    uint32_t* h_seed;
    cudaError_t e;
    unsigned rngsPerBlock = thread_num / WarpStandard_K;
    unsigned sharedMemBytesPerBlock
	= rngsPerBlock * WarpStandard_SHMEM_WORDS * 4;
    int seed_size = rngsPerBlock * block_num * 32;
    int all_threads = block_num * thread_num;
    uint32_t tmp;

    printf("generating unsigned random numbers.\n");
    cudaMalloc((void**)&d_data, sizeof(float) * all_threads);
    cudaMalloc((void**)&d_seed, sizeof(uint32_t) * seed_size);
    /* cutCreateTimer(&timer); */
    float elapsed_time_ms=0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_data = (float *) malloc(sizeof(float) * all_threads);
    h_seed = (uint32_t *) malloc(sizeof(uint32_t) * seed_size);
    if (h_data == NULL || h_seed == NULL) {
	printf("failure in allocating host memory for output data.\n");
	exit(1);
    }
    memcpy(h_seed, WarpStandard_TEST_DATA, sizeof(uint32_t) * 32);
    tmp = h_seed[17];
    for (int i = 0; i < seed_size -35; i++) {
	tmp = (tmp >> 11) * h_seed[i] + i;
	h_seed[i + 32] = tmp ^ (tmp << 3);
    }
    cudaMemcpy(d_seed, h_seed, sizeof(uint32_t) * seed_size,
	       cudaMemcpyHostToDevice);
    /* cutStartTimer(timer); */
    cudaEventRecord(start, 0);
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

    /* kernel call */
    float_reduce<<< block_num, thread_num, sharedMemBytesPerBlock>>>
	(d_data, num_data / block_num, d_seed);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);
    }
    /* cutStopTimer(timer); */
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    cudaMemcpy(h_data, d_data, sizeof(uint32_t) * all_threads,
	       cudaMemcpyDeviceToHost);
    printf("reduce generation numbers: %d\n", num_data);
    printf("Processing time: %f (ms)\n", elapsed_time_ms);
    printf("Samples per second: %E \n", num_data / (elapsed_time_ms * 0.001));
    /* cutDeleteTimer(timer); */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //free memories
    free(h_data);
    free(h_seed);
    ccudaFree(d_data);
    ccudaFree(d_seed);
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
    num_unit = thread_num * block_num;
    r = num_data % num_unit;
    if (r != 0) {
	num_data = num_data + num_unit - r;
    }
    make_float_random(num_data, block_num);
    make_float_reduced(num_data, block_num);
}
