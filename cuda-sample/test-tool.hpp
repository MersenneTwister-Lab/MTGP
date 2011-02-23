#include <cuda.h>

inline void exception_maker(cudaError_t rc) {
    if (rc != cudaSuccess) {
	throw cudaGetErrorString(rc);
    }
}

inline int ccudaMalloc(void **devPtr, size_t size) {
    cudaError_t rc = cudaMalloc((void **)(void*)devPtr, size);
    exception_maker(rc);
    return cudaSuccess;
}

inline int ccudaMemcpy(void *dest, void *src, size_t size,
		      enum cudaMemcpyKind kind) {
    cudaError_t rc = cudaMemcpy(dest, src, size, kind);
    exception_maker(rc);
    return cudaSuccess;
}

inline int ccudaEventCreate(cudaEvent_t * event) {
    cudaError_t rc = cudaEventCreate(&start);
    exception_maker(rc);
    return cudaSuccess;
}

inline int ccudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    cudaError_t rc = cudaEventRecord(event, stream);
    exception_maker(rc);
    return cudaSuccess;
}

inline int ccudaEventSynchronize(cudaEvent_t event) {
    cudaError_t rc = cudaEventSynchronize(stop);
    exception_maker(rc);
    return cudaSuccess;
}
