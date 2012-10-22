#ifndef MTGP_UTIL_HPP
#define MTGP_UTIL_HPP
/*
 * @file mtgp-util.hpp
 *
 * Some utility functions for Sample Programs
 *
 */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <cuda.h>
#include <cuda_runtime.h>

int get_suitable_block_num(int device, int *max_block_num,
			   int *mp_num, int word_size,
			   int thread_num, int large_size);
void print_max_min(uint32_t data[], int size);
void print_float_array(const float array[], int size, int block);
void print_uint32_array(const uint32_t array[], int size, int block);
void print_double_array(const double array[], int size, int block);
void print_uint64_array(const uint64_t array[], int size, int block);

static inline void exception_maker(cudaError_t rc, const char * funcname);

static inline void exception_maker(cudaError_t rc, const char * funcname)
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
    cudaError_t rc = cudaGetDeviceCount(num);
    exception_maker(rc, "ccudaGetDeviceCount");
    return CUDA_SUCCESS;
}

static inline int ccudaSetDevice(int dev)
{
    cudaError_t rc = cudaSetDevice(dev);
    exception_maker(rc, "ccudaSetDevice");
    return CUDA_SUCCESS;
}

static inline int ccudaMalloc(void **devPtr, size_t size)
{
    cudaError_t rc = cudaMalloc((void **)(void*)devPtr, size);
    exception_maker(rc, "ccudaMalloc");
    return CUDA_SUCCESS;
}

static inline int ccudaFree(void *devPtr)
{
    cudaError_t rc = cudaFree(devPtr);
    exception_maker(rc, "ccudaFree");
    return CUDA_SUCCESS;
}

static inline int ccudaMemcpy(void *dest, void *src, size_t size,
		      enum cudaMemcpyKind kind)
{
    cudaError_t rc = cudaMemcpy(dest, src, size, kind);
    exception_maker(rc, "ccudaMemcpy");
    return CUDA_SUCCESS;
}

static inline int ccudaEventCreate(cudaEvent_t * event)
{
    cudaError_t rc = cudaEventCreate(event);
    exception_maker(rc, "ccudaEventCreate");
    return CUDA_SUCCESS;
}

static inline int ccudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    cudaError_t rc = cudaEventRecord(event, stream);
    exception_maker(rc, "ccudaEventRecord");
    return CUDA_SUCCESS;
}

static inline int ccudaEventSynchronize(cudaEvent_t event)
{
    cudaError_t rc = cudaEventSynchronize(event);
    exception_maker(rc, "ccudaEventSynchronize");
    return CUDA_SUCCESS;
}

static inline int ccudaThreadSynchronize()
{
    cudaError_t rc = cudaThreadSynchronize();
    exception_maker(rc, "ccudaThreadSynchronize");
    return CUDA_SUCCESS;
}

static inline int ccudaEventElapsedTime(float * ms,
				 cudaEvent_t start, cudaEvent_t end)
{
    cudaError_t rc = cudaEventElapsedTime(ms, start, end);
    exception_maker(rc, "ccudaEventElapsedTime");
    return CUDA_SUCCESS;
}

static inline int ccudaEventDestroy(cudaEvent_t event)
{
    cudaError_t rc = cudaEventDestroy(event);
    exception_maker(rc, "ccudaEventDestroy");
    return CUDA_SUCCESS;
}

static inline int ccudaMemcpyToSymbol(const void * symbol,
			       const void * src,
			       size_t count,
			       size_t offset = 0,
			       enum cudaMemcpyKind kind
			       = cudaMemcpyHostToDevice)
{
    cudaError_t rc = cudaMemcpyToSymbol((const char *)symbol,
					src, count, offset, kind);
    exception_maker(rc, "ccudaMemcpyToSymbol");
    return CUDA_SUCCESS;
}

static inline int ccudaGetDeviceProperties(struct cudaDeviceProp * prop,
					   int device)
{
    cudaError_t rc = cudaGetDeviceProperties(prop, device);
    exception_maker(rc, "ccudaGetDeviceProperties");
    return CUDA_SUCCESS;
}

#if 0
template<class T, int dim, enum cudaTextureReadMode readMode>
static inline int ccudaBindTexture(size_t * offset,
				   const struct texture< T, dim, readMode > & texref,
				   const void * devPtr,
				   size_t size = UINT_MAX)
{

    cudaError_t rc = cudaBindTexture(offset, texref, devPtr, size);
    exception_maker(rc, "ccudaBIndTexture");
    return CUDA_SUCCESS;
}
#endif
#endif
