#ifndef _TEST_TOOL_HPP_
#define _TEST_TOOL_HPP_

#include <iostream>
#include <iomanip>
#include <cuda.h>

inline void exception_maker(cudaError_t rc, const char * funcname)
{
    using namespace std;
    if (rc != cudaSuccess) {
	const char * message = cudaGetErrorString(rc);
	cerr << "In " << funcname << " Error:" << message << endl;
	throw message;
    }
}

inline int ccudaMalloc(void **devPtr, size_t size)
{
    cudaError_t rc = cudaMalloc((void **)(void*)devPtr, size);
    exception_maker(rc, "ccudaMalloc");
    return cudaSuccess;
}

inline int ccudaFree(void *devPtr)
{
    cudaError_t rc = cudaFree(devPtr);
    exception_maker(rc, "ccudaFree");
    return cudaSuccess;
}

inline int ccudaMemcpy(void *dest, void *src, size_t size,
		      enum cudaMemcpyKind kind)
{
    cudaError_t rc = cudaMemcpy(dest, src, size, kind);
    exception_maker(rc, "ccudaMemcpy");
    return cudaSuccess;
}

inline int ccudaEventCreate(cudaEvent_t * event)
{
    cudaError_t rc = cudaEventCreate(event);
    exception_maker(rc, "ccudaEventCreate");
    return cudaSuccess;
}

inline int ccudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    cudaError_t rc = cudaEventRecord(event, stream);
    exception_maker(rc, "ccudaEventRecord");
    return cudaSuccess;
}

inline int ccudaEventSynchronize(cudaEvent_t event)
{
    cudaError_t rc = cudaEventSynchronize(event);
    exception_maker(rc, "ccudaEventSynchronize");
    return cudaSuccess;
}

inline int ccudaThreadSynchronize()
{
    cudaError_t rc = cudaThreadSynchronize();
    exception_maker(rc, "ccudaThreadSynchronize");
    return cudaSuccess;
}

inline int ccudaEventElapsedTime(float * ms,
				 cudaEvent_t start, cudaEvent_t end)
{
    cudaError_t rc = cudaEventElapsedTime(ms, start, end);
    exception_maker(rc, "ccudaEventElapsedTime");
    return cudaSuccess;
}

inline int ccudaEventDestroy(cudaEvent_t event)
{
    cudaError_t rc = cudaEventDestroy(event);
    exception_maker(rc, "ccudaEventDestroy");
    return cudaSuccess;
}

inline int ccudaMemcpyToSymbol(const void * symbol,
			       const void * src,
			       size_t count,
			       size_t offset = 0,
			       enum cudaMemcpyKind kind
			       = cudaMemcpyHostToDevice)
{
    cudaError_t rc = cudaMemcpyToSymbol((const char *)symbol,
					src, count, offset, kind);
    exception_maker(rc, "ccudaMemcpyToSymbol");
    return cudaSuccess;
}

template<class T, int dim, enum cudaTextureReadMode readMode>
inline int ccudaBindTexture(size_t * offset,
			    const struct texture< T, dim, readMode > & texref,
			    const void * devPtr,
			    size_t size = UINT_MAX)
{

    cudaError_t rc = cudaBindTexture(offset, texref, devPtr, size);
    exception_maker(rc, "ccudaBIndTexture");
    return cudaSuccess;
}

#endif
