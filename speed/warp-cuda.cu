#include <cuda.h>
#include <cutil.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <util.h>

#define BLOCK_NUM_MAX 1000
#define THREAD_NUM 256
#include <stdint.h>

/////////////////////////////////////////////////////////////////////////////////////
// Public constants

const unsigned WarpStandard_K=32;
const unsigned WarpStandard_REG_COUNT=3;
const unsigned WarpStandard_STATE_WORDS=32;

const uint32_t WarpStandard_TEST_DATA[WarpStandard_STATE_WORDS]={
	0x8cf35fea, 0xe1dd819e, 0x4a7d0a8e, 0xe0c05911, 0xfd053b8d, 0x30643089, 0x6f6ac111, 0xc4869595, 0x9416b7be, 0xe6d329e8, 0x5af0f5bf, 0xc5c742b5, 0x7197e922, 0x71aa35b4, 0x2070b9d1, 0x2bb34804, 0x7754a517, 0xe725315e, 0x7f9dd497, 0x043b58bf, 0x83ffa33d, 0x2532905a, 0xbdfe0c8a, 0x16f68671, 0x0d14da2e, 0x847efd5f, 0x1edeec64, 0x1bebdf9b, 0xf74d4ff3, 0xd404774b, 0x8ee32599, 0xefe0c405
};

//////////////////////////////////////////////////////////////////////////////////////
// Private constants

const char *WarpStandard_name="WarpRNG[CorrelatedU32Rng;k=32;g=16;rs=0;w=32;n=1024;hash=deac2e12ec6e615]";
const char *WarpStandard_post_processing="addtaps";
const unsigned WarpStandard_N=1024;
const unsigned WarpStandard_W=32;
const unsigned WarpStandard_G=16;
const unsigned WarpStandard_SR=0;
__device__ const unsigned WarpStandard_Q[2][32]={
  {29,24,5,23,14,26,11,31,9,3,1,28,0,2,22,20,18,15,27,13,10,16,8,17,25,12,19,30,7,6,4,21},
  {5,14,28,24,19,13,0,17,11,20,7,10,6,15,2,9,8,23,4,30,12,25,3,21,26,27,31,18,22,16,29,1}
};
const unsigned WarpStandard_Z0=2;
__device__ const unsigned WarpStandard_Z1[32]={
  0,1,0,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1};

const unsigned WarpStandard_SHMEM_WORDS=32;
const unsigned WarpStandard_GMEM_WORDS=0;

////////////////////////////////////////////////////////////////////////////////////////
// Public functions

__device__ void WarpStandard_LoadState(const unsigned *seed, unsigned *regs, unsigned *shmem)
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

__device__ void WarpStandard_SaveState(const unsigned *regs, const unsigned *shmem, unsigned *seed)
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
};


extern __shared__ unsigned shmem[];

__global__ void warp_kernel(uint32_t* d_data, int size, const uint32_t seed[])
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

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param[in] num_data number of data to be generated.
 */
void make_warp_random(int num_data,
		      int block_num) {
    uint32_t* d_data;
    unsigned int timer = 0;
    uint32_t* h_data;
    cudaError_t e;
    float gputime;
    unsigned rngsPerBlock = THREAD_NUM / WarpStandard_K;
    unsigned sharedMemBytesPerBlock
	= rngsPerBlock * WarpStandard_SHMEM_WORDS * 4;

    printf("generating unsigned random numbers.\n");
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
    warp_kernel<<< block_num, THREAD_NUM, sharedMemBytesPerBlock>>>
	(d_data, num_data / block_num, WarpStandard_TEST_DATA);
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
    print_uint_array(h_data, num_data, block_num);
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
	CUT_DEVICE_INIT(argc, argv);
	printf("%s number_of_block number_of_output\n", argv[0]);
	return 1;
    }
    CUT_DEVICE_INIT(argc, argv);
    num_unit = THREAD_NUM * block_num;
    r = num_data % num_unit;
    if (r != 0) {
	num_data = num_data + num_unit - r;
    }
    make_warp_random(num_data, block_num);

    //finalize
#ifdef NEED_PROMPT
    CUT_EXIT(argc, argv);
#endif
}
