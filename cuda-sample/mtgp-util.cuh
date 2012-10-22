#ifndef MTGP_UTIL_CUH
#define MTGP_UTIL_CUH
/*
 * mtgp-util.h
 *
 * Some utility functions for Sample Programs
 *
 */
#include <cuda.h>
#include <stdint.h>
#include <inttypes.h>
#if defined(DEBUG)
#define DENTER(x) printf("entering %s\n", x)
#define DEXIT(x) printf("exiting %s\n", x)
#else
#define DENTER(x)
#define DEXIT(x)
#endif
static inline int get_suitable_block_num(int device, int *max_block_num,
					 int *mp_num, int word_size,
					 int thread_num, int large_size);
template<class T>
static inline int ccudaMemcpyToSymbol(const T &symbol,
				      const void * src,
				      size_t count);

static inline void exception_maker(cudaError rc, const char * funcname)
{
    using namespace std;
    if (rc != cudaSuccess) {
	const char * message = cudaGetErrorString(rc);
	fprintf(stderr, "In %s Error(%d):%s\n", funcname, rc, message);
	throw message;
    }
}

static inline int ccudaGetDeviceCount(int * num)
{
    DENTER("ccudaGetDeviceCount");
    cudaError rc = cudaGetDeviceCount(num);
    exception_maker(rc, "ccudaGetDeviceCount");
    return cudaSuccess;
}

static inline int ccudaSetDevice(int dev)
{
    DENTER("ccudaSetDevice");
    cudaError rc = cudaSetDevice(dev);
    exception_maker(rc, "ccudaSetDevice");
    return cudaSuccess;
}

static inline int ccudaMalloc(void **devPtr, size_t size)
{
    DENTER("ccudaMalloc");
    cudaError rc = cudaMalloc((void **)(void*)devPtr, size);
    exception_maker(rc, "ccudaMalloc");
    return cudaSuccess;
}

static inline int ccudaFree(void *devPtr)
{
    DENTER("ccudaFree");
    cudaError rc = cudaFree(devPtr);
    exception_maker(rc, "ccudaFree");
    return cudaSuccess;
}

static inline int ccudaMemcpy(void *dest, void *src, size_t size,
			      enum cudaMemcpyKind kind)
{
    DENTER("ccudaMemcpy");
    cudaError rc = cudaMemcpy(dest, src, size, kind);
    exception_maker(rc, "ccudaMemcpy");
    return cudaSuccess;
}

static inline int ccudaEventCreate(cudaEvent_t * event)
{
    DENTER("ccudaEventCreate");
    cudaError rc = cudaEventCreate(event);
    exception_maker(rc, "ccudaEventCreate");
    return cudaSuccess;
}

static inline int ccudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    DENTER("ccudaEventRecord");
    cudaError rc = cudaEventRecord(event, stream);
    exception_maker(rc, "ccudaEventRecord");
    return cudaSuccess;
}

static inline int ccudaEventSynchronize(cudaEvent_t event)
{
    DENTER("ccudaEventSynchronize");
    cudaError rc = cudaEventSynchronize(event);
    exception_maker(rc, "ccudaEventSynchronize");
    return cudaSuccess;
}

static inline int ccudaThreadSynchronize()
{
    DENTER("ccudaThreadSynchronize");
    cudaError rc = cudaThreadSynchronize();
    exception_maker(rc, "ccudaThreadSynchronize");
    return cudaSuccess;
}

static inline int ccudaEventElapsedTime(float * ms,
				 cudaEvent_t start, cudaEvent_t end)
{
    DENTER("ccudaEventElapsedTIme");
    cudaError rc = cudaEventElapsedTime(ms, start, end);
    exception_maker(rc, "ccudaEventElapsedTime");
    return cudaSuccess;
}

static inline int ccudaEventDestroy(cudaEvent_t event)
{
    DENTER("ccudaEventDestroy");
    cudaError rc = cudaEventDestroy(event);
    exception_maker(rc, "ccudaEventDestroy");
    return cudaSuccess;
}

template<class T>
static inline int ccudaMemcpyToSymbol(const T &symbol,
			       const void * src,
			       size_t count)
{
    DENTER("ccudaMemcpyToSymbol");
    cudaError rc = cudaMemcpyToSymbol((const void *)&symbol,
					src, count, 0, cudaMemcpyHostToDevice);
    exception_maker(rc, "ccudaMemcpyToSymbol");
    return cudaSuccess;
}

static inline int ccudaGetDeviceProperties(struct cudaDeviceProp * prop, int device)
{
    DENTER("ccudaGetDeviceProperties");
    cudaError rc = cudaGetDeviceProperties(prop, device);
    exception_maker(rc, "ccudaGetDeviceProperties");
    return cudaSuccess;
}

template<class T, int dim, enum cudaTextureReadMode readMode>
static inline int ccudaBindTexture(size_t * offset,
			    const struct texture< T, dim, readMode > & texref,
			    const void * devPtr,
			    size_t size = UINT_MAX)
{
    DENTER("ccudaTextureReadMode");
    cudaError rc = cudaBindTexture(offset, texref, devPtr, size);
    exception_maker(rc, "ccudaBIndTexture");
    return cudaSuccess;
}

static inline int get_suitable_block_num(int device,
					 int *max_block_num,
					 int *mp_num,
					 int word_size,
					 int thread_num,
					 int large_size)
{
    DENTER("get_suitable_block");
    cudaDeviceProp dev;
    CUdevice cuDevice;
    int max_thread_dev;
    int max_block, max_block_mem, max_block_dev;
    int major, minor, ver;
    //int regs, max_block_regs;

    ccudaGetDeviceProperties(&dev, device);
    cuDeviceGet(&cuDevice, device);
    cuDeviceComputeCapability(&major, &minor, cuDevice);
    //cudaFuncGetAttributes()
#if 0
    if (word_size == 4) {
	regs = 14;
    } else {
	regs = 16;
    }
    max_block_regs = dev.regsPerBlock / (regs * thread_num);
#endif
    max_block_mem = dev.sharedMemPerBlock / (large_size * word_size + 16);
    if (major == 9999 && minor == 9999) {
	return -1;
    }
    ver = major * 100 + minor;
    if (ver <= 101) {
	max_thread_dev = 768;
    } else if (ver <= 103) {
	max_thread_dev = 1024;
    } else if (ver <= 200) {
	max_thread_dev = 1536;
    } else {
	max_thread_dev = 1536;
    }
    max_block_dev = max_thread_dev / thread_num;
    if (max_block_mem < max_block_dev) {
	max_block = max_block_mem;
    } else {
	max_block = max_block_dev;
    }
#if 0
    if (max_block_regs < max_block) {
	max_block = max_block_regs;
    }
#endif
    *max_block_num = max_block;
    *mp_num = dev.multiProcessorCount;
    return max_block * dev.multiProcessorCount;
}

#endif
