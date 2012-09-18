/*
 * @file mtgp32-jump-kernel.cuh
 *
 * @brief Sample Program for CUDA 3.2 and 4.0
 *
 * This program changes internal state of MTGP to jumped state.
 */
#ifndef MTGP32_JUMP_KERNEL_CUH
#define MTGP32_JUMP_KERNEL_CUH

#include <cuda.h>
#include <stdint.h>
#include <inttypes.h>

#define MTGP32_MEXP 11213
#define MTGP32_N 351
#define MTGP32_FLOOR_2P 256
#define MTGP32_CEIL_2P 512
#define MTGP32_TN MTGP32_FLOOR_2P
#define MTGP32_LS (MTGP32_TN * 3)
#define MTGP32_TS 16

/**
 * kernel I/O
 * This structure must be initialized before first use.
 */
struct mtgp32_kernel_status_t {
    uint32_t status[MTGP32_N];
};

/*
 * Generator Parameters.
 */
/* No.1 delta:686 weight:1659 */
__constant__ unsigned int mtgp32_pos = 84;
__constant__ uint32_t mtgp32_sh1 = 12;
__constant__ uint32_t mtgp32_sh2 = 4;
__constant__ uint32_t mtgp32_param_tbl[MTGP32_TS]
= {0x00000000, 0x71588353, 0xdfa887c1, 0xaef00492,
   0x4ba66c6e, 0x3afeef3d, 0x940eebaf, 0xe55668fc,
   0xa53da0ae, 0xd46523fd, 0x7a95276f, 0x0bcda43c,
   0xee9bccc0, 0x9fc34f93, 0x31334b01, 0x406bc852};
__constant__ uint32_t mtgp32_temper_tbl[MTGP32_TS]
= {0x00000000, 0x200040bb, 0x1082c61e, 0x308286a5,
   0x10021c03, 0x30025cb8, 0x0080da1d, 0x20809aa6,
   0x0003f0b9, 0x2003b002, 0x108136a7, 0x3081761c,
   0x1001ecba, 0x3001ac01, 0x00832aa4, 0x20836a1f};
__constant__ uint32_t mtgp32_single_temper_tbl[MTGP32_TS]
= {0x3f800000, 0x3f900020, 0x3f884163, 0x3f984143,
   0x3f88010e, 0x3f98012e, 0x3f80406d, 0x3f90404d,
   0x3f8001f8, 0x3f9001d8, 0x3f88409b, 0x3f9840bb,
   0x3f8800f6, 0x3f9800d6, 0x3f804195, 0x3f9041b5};
__constant__ uint32_t mtgp32_mask = 0xfff80000;
__constant__ uint32_t mtgp32_non_zero = 0x4d544750;

/* jump polynomial for 3^162 steps jump */
#include "mtgp32-jump-table.cuh"

/**
 * Shared memory
 * The generator's internal status vector.
 */
__shared__ uint32_t mtgp32_status[MTGP32_LS];
__shared__ uint32_t mtgp32_jwork[MTGP32_N];
__shared__ uint32_t mtgp32_jstatus[MTGP32_N];

/**
 * The function of the recursion formula calculation.
 *
 * @param[in] X1 the farthest part of state array.
 * @param[in] X2 the second farthest part of state array.
 * @param[in] Y a part of state array.
 * @param[in] bid block id.
 * @return output
 */
__device__ uint32_t para_rec(uint32_t X1, uint32_t X2, uint32_t Y) {
    uint32_t X = (X1 & mtgp32_mask) ^ X2;
    uint32_t MAT;

    X ^= X << mtgp32_sh1;
    Y = X ^ (Y >> mtgp32_sh2);
    MAT = mtgp32_param_tbl[Y & 0x0f];
    return Y ^ MAT;
}

/**
 * The tempering function.
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered value.
 */
__device__ uint32_t temper(uint32_t V, uint32_t T) {
    uint32_t MAT;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = mtgp32_temper_tbl[T & 0x0f];
    return V ^ MAT;
}

/**
 * The tempering and converting function.
 * By using the preset-ted table, converting to IEEE format
 * and tempering are done simultaneously.
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered and converted value.
 */
__device__ uint32_t temper_single(uint32_t V, uint32_t T) {
    uint32_t MAT;
    uint32_t r;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = mtgp32_single_temper_tbl[T & 0x0f];
    r = (V >> 9) ^ MAT;
    return r;
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
__device__ void status_read(uint32_t status[MTGP32_LS],
			    const mtgp32_kernel_status_t *d_status,
			    int bid,
			    int tid) {
    status[MTGP32_LS - MTGP32_N + tid] = d_status[bid].status[tid];
    if (tid < MTGP32_N - MTGP32_TN) {
	status[MTGP32_LS - MTGP32_N + MTGP32_TN + tid]
	    = d_status[bid].status[MTGP32_TN + tid];
    }
    __syncthreads();
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
__device__ void status_write(mtgp32_kernel_status_t *d_status,
			     const uint32_t status[MTGP32_LS],
			     int bid,
			     int tid) {
    d_status[bid].status[tid] = status[MTGP32_LS - MTGP32_N + tid];
    if (tid < MTGP32_N - MTGP32_TN) {
	d_status[bid].status[MTGP32_TN + tid]
	    = status[4 * MTGP32_TN - MTGP32_N + tid];
    }
    __syncthreads();
}

/**
 * device function.
 * This function changes internal state of MTGP to jumped state.
 * threads per block should be MTGP32_N.
 *
 * @param[in,out] d_status kernel I/O data
 * @param[in] jump_poly jump polynomial
 */
__device__ void mtgp32_jump(uint32_t work[],
			    uint32_t mtgp32_status[],
			    uint32_t jump_poly[]) {
    const int tid = threadIdx.x;
    int index = 0;
    uint32_t r;
    int pos = mtgp32_pos;
    work[tid] = 0;

    // jump
    for (int i = 0; i < MTGP32_N; i++) {
	uint32_t bits = jump_poly[i];
	for (int j = 0; j < 32; j++) {
	    if ((bits & 1) != 0) {
		//add
		work[tid] ^= mtgp32_status[(tid + index) % MTGP32_N];
		__syncthreads();
	    }
	    //next_state
	    if (tid == 0) {
		r = para_rec(mtgp32_status[(tid + index) % MTGP32_N],
			     mtgp32_status[(tid + index + 1) % MTGP32_N],
			     mtgp32_status[(tid + index + pos) % MTGP32_N]);
		mtgp32_status[(tid + index) % MTGP32_N] = r;
	    }
	    index = (index + 1) % MTGP32_N;
	    __syncthreads();
	    bits = bits >> 1;
	}
    }
}

/**
 * kernel function.
 * This function changes internal state of MTGP to jumped state.
 * threads per block should be MTGP32_N.
 *
 * @param[in,out] d_status kernel I/O data
 * @param[in] jump_poly jump polynomial
 */
__global__ void mtgp32_jump_kernel(mtgp32_kernel_status_t* d_status,
				   uint32_t jump_poly[], int count) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    // copy status data from global memory to shared memory.
    mtgp32_jstatus[tid] = d_status[bid].status[tid];
    __syncthreads();
    // jump
    for (int i = 0; i < count; i++) {
	mtgp32_jump(mtgp32_jwork, mtgp32_jstatus, jump_poly);
	__syncthreads();
	mtgp32_jstatus[tid] = mtgp32_jwork[tid];
	__syncthreads();
    }
    __syncthreads();
    d_status[bid].status[tid] = mtgp32_jstatus[tid];
}

/**
 * This function represents a function used in the initialization
 * by mtgp32_init_by_array() and mtgp32_init_by_str().
 * @param[in] x 32-bit integer
 * @return 32-bit integer
 */
__device__ uint32_t mtgp32_ini_func1(uint32_t x) {
    return (x ^ (x >> 27)) * UINT32_C(1664525);
}

/**
 * This function represents a function used in the initialization
 * by mtgp32_init_by_array() and mtgp32_init_by_str().
 * @param[in] x 32-bit integer
 * @return 32-bit integer
 */
__device__ uint32_t mtgp32_ini_func2(uint32_t x) {
    return (x ^ (x >> 27)) * UINT32_C(1566083941);
}

/**
 * This function initializes the internal state array with a 32-bit
 * integer seed.
 * @param[in] seed a 32-bit integer used as the seed.
 */
__device__ void mtgp32_init_state(uint32_t seed) {
    int i;
    uint32_t hidden_seed;
    uint32_t tmp;
    const int tid = threadIdx.x;
    hidden_seed = mtgp32_param_tbl[4] ^ (mtgp32_param_tbl[8] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    tmp &= 0xff;
    tmp |= tmp << 8;
    tmp |= tmp << 16;
    //memset(mtgp32_jstatus, tmp & 0xff, sizeof(uint32_t) * size);
    mtgp32_jstatus[tid] = tmp;
    __syncthreads();
    if (tid == 0) {
	mtgp32_jstatus[0] = seed;
	mtgp32_jstatus[1] = hidden_seed;
	for (i = 1; i < MTGP32_N; i++) {
	    mtgp32_jstatus[i] ^= UINT32_C(1812433253)
		* (mtgp32_jstatus[i - 1]
		   ^ (mtgp32_jstatus[i - 1] >> 30))
		+ i;
	}
    }
}

/**
 * This function allocates and initializes the internal state array
 * with a 32-bit integer array. The allocated memory should be freed by
 * calling mtgp32_free(). \b para should be one of the elements in
 * the parameter table (mtgp32-param-ref.c).
 *
 * @param[in] seed_array a 32-bit integer array used as a seed.
 * @param[in] length length of the seed_array.
 */
__device__ void mtgp32_init_by_array(uint32_t *seed_array, int length) {
    int i, j, count;
    uint32_t r;
    int lag;
    int mid;
    int size = MTGP32_N;
    uint32_t hidden_seed;
    uint32_t tmp;
    const int tid = threadIdx.x;

//    st = mtgp32_jstatus;
    if (size >= 623) {
	lag = 11;
    } else if (size >= 68) {
	lag = 7;
    } else if (size >= 39) {
	lag = 5;
    } else {
	lag = 3;
    }
    mid = (size - lag) / 2;

    hidden_seed = mtgp32_param_tbl[4] ^ (mtgp32_param_tbl[8] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;

    tmp &= 0xff;
    tmp |= tmp << 8;
    tmp |= tmp << 16;
    mtgp32_jstatus[tid] = tmp;
    __syncthreads();
    if (tid != 0) {
	return;
    }
    mtgp32_jstatus[0] = hidden_seed;
    if (length + 1 > size) {
	count = length + 1;
    } else {
	count = size;
    }
    r = mtgp32_ini_func1(mtgp32_jstatus[0]
			 ^ mtgp32_jstatus[mid]
			 ^ mtgp32_jstatus[size - 1]);
    mtgp32_jstatus[mid] += r;
    r += length;
    mtgp32_jstatus[(mid + lag) % size] += r;
    mtgp32_jstatus[0] = r;
    i = 1;
    count--;
    for (i = 1, j = 0; (j < count) && (j < length); j++) {
	r = mtgp32_ini_func1(mtgp32_jstatus[i]
			     ^ mtgp32_jstatus[(i + mid) % size]
			     ^ mtgp32_jstatus[(i + size - 1) % size]);
	mtgp32_jstatus[(i + mid) % size] += r;
	r += seed_array[j] + i;
	mtgp32_jstatus[(i + mid + lag) % size] += r;
	mtgp32_jstatus[i] = r;
	i = (i + 1) % size;
    }
    for (; j < count; j++) {
	r = mtgp32_ini_func1(mtgp32_jstatus[i]
			     ^ mtgp32_jstatus[(i + mid) % size]
			     ^ mtgp32_jstatus[(i + size - 1) % size]);
	mtgp32_jstatus[(i + mid) % size] += r;
	r += i;
	mtgp32_jstatus[(i + mid + lag) % size] += r;
	mtgp32_jstatus[i] = r;
	i = (i + 1) % size;
    }
    for (j = 0; j < size; j++) {
	r = mtgp32_ini_func2(mtgp32_jstatus[i]
			     + mtgp32_jstatus[(i + mid) % size]
			     + mtgp32_jstatus[(i + size - 1) % size]);
	mtgp32_jstatus[(i + mid) % size] ^= r;
	r -= i;
	mtgp32_jstatus[(i + mid + lag) % size] ^= r;
	mtgp32_jstatus[i] = r;
	i = (i + 1) % size;
    }
    if (mtgp32_jstatus[size - 1] == 0) {
	mtgp32_jstatus[size - 1] = mtgp32_non_zero;
    }
}

/**
 *
 *
 */
__device__ void mtgp32_table_jump(int bid, int tid, uint32_t jump_table[][MTGP32_N + 1])
{
    for (int i = 0; i < 32; i++) {
	if ((bid & (1 << i)) == 0) {
	    continue;
	}
	if (i % 2 == 0) {
	    mtgp32_jump(mtgp32_jwork, mtgp32_jstatus, jump_table[i / 2]);
	    __syncthreads();
	    mtgp32_jstatus[tid] = mtgp32_jwork[tid];
	    __syncthreads();
	} else {
	    mtgp32_jump(mtgp32_jwork, mtgp32_jstatus, jump_table[i / 2]);
	    __syncthreads();
	    mtgp32_jump(mtgp32_jstatus, mtgp32_jwork, jump_table[i / 2]);
	    __syncthreads();
	}
    }
}

/**
 * kernel function.
 * This function changes internal state of MTGP to jumped state.
 * threads per block should be MTGP32_N.
 *
 * @param[in,out] d_status kernel I/O data
 */
__global__ void mtgp32_jump_long_seed_kernel(mtgp32_kernel_status_t* d_status,
					     uint32_t seed)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    mtgp32_init_state(seed);
    __syncthreads();

    // jump
    mtgp32_table_jump(bid, tid, mtgp32_jump_table);
    d_status[bid].status[tid] = mtgp32_jstatus[tid];
    __syncthreads();
}

/**
 * kernel function.
 * This function changes internal state of MTGP to jumped state.
 * threads per block should be MTGP32_N.
 *
 * @param[in,out] d_status kernel I/O data
 */
__global__ void mtgp32_jump_long_array_kernel(mtgp32_kernel_status_t * d_status,
					      uint32_t * seed_array, int length)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    mtgp32_init_by_array(seed_array, length);
    __syncthreads();

    mtgp32_table_jump(bid, tid, mtgp32_jump_table);
    d_status[bid].status[tid] = mtgp32_jstatus[tid];
}

/**
 * kernel function.
 * This function changes internal state of MTGP to jumped state.
 * threads per block should be MTGP32_N.
 *
 * @param[in,out] d_status kernel I/O data
 */
__global__ void mtgp32_jump_seed_kernel(mtgp32_kernel_status_t* d_status,
					uint32_t jump_table[][MTGP32_N + 1],
					uint32_t seed)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    mtgp32_init_state(seed);
    __syncthreads();

    // jump
    mtgp32_table_jump(bid, tid, jump_table);
    d_status[bid].status[tid] = mtgp32_jstatus[tid];
    __syncthreads();
}

/**
 * kernel function.
 * This function changes internal state of MTGP to jumped state.
 * threads per block should be MTGP32_N.
 *
 * @param[in,out] d_status kernel I/O data
 */
__global__ void mtgp32_jump_array_kernel(mtgp32_kernel_status_t * d_status,
					 uint32_t jump_table[][MTGP32_N + 1],
					 uint32_t * seed_array,
					 int length)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    mtgp32_init_by_array(seed_array, length);
    __syncthreads();

    mtgp32_table_jump(bid, tid, jump_table);
    d_status[bid].status[tid] = mtgp32_jstatus[tid];
}

/**
 * kernel function.
 * This function generates 32-bit unsigned integers in d_data
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output
 * @param[in] size number of output data requested.
 */
__global__ void mtgp32_uint32_kernel(mtgp32_kernel_status_t* d_status,
				     uint32_t* d_data, int size) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int pos = mtgp32_pos;
    uint32_t * status = mtgp32_status;
    uint32_t r;
    uint32_t o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, bid, tid);

    // main loop
    for (int i = 0; i < size; i += MTGP32_LS) {
	r = para_rec(status[MTGP32_LS - MTGP32_N + tid],
		 status[MTGP32_LS - MTGP32_N + tid + 1],
		 status[MTGP32_LS - MTGP32_N + tid + pos]);
	status[tid] = r;
	o = temper(r, status[MTGP32_LS - MTGP32_N + tid + pos - 1]);
	d_data[size * bid + i + tid] = o;
	__syncthreads();
	r = para_rec(status[(4 * MTGP32_TN - MTGP32_N + tid) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + tid + 1) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + tid + pos)
			    % MTGP32_LS]);
	status[tid + MTGP32_TN] = r;
	o = temper(r,
		   status[(4 * MTGP32_TN - MTGP32_N + tid + pos - 1)
			  % MTGP32_LS]);
	d_data[size * bid + MTGP32_TN + i + tid] = o;
	__syncthreads();
	r = para_rec(status[2 * MTGP32_TN - MTGP32_N + tid],
		     status[2 * MTGP32_TN - MTGP32_N + tid + 1],
		     status[2 * MTGP32_TN - MTGP32_N + tid + pos]);
	status[tid + 2 * MTGP32_TN] = r;
	o = temper(r, status[tid + pos - 1 + 2 * MTGP32_TN - MTGP32_N]);
	d_data[size * bid + 2 * MTGP32_TN + i + tid] = o;
	__syncthreads();
    }
    // write back status for next call
    status_write(d_status, status, bid, tid);
}

/**
 * kernel function.
 * This function generates single precision floating point numbers in d_data.
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output. IEEE single precision format.
 * @param[in] size number of output data requested.
 */
__global__ void mtgp32_single_kernel(mtgp32_kernel_status_t* d_status,
				     uint32_t* d_data, int size)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int pos = mtgp32_pos;
    uint32_t * status = mtgp32_status;
    uint32_t r;
    uint32_t o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, bid, tid);

    // main loop
    for (int i = 0; i < size; i += MTGP32_LS) {
	r = para_rec(status[MTGP32_LS - MTGP32_N + tid],
		     status[MTGP32_LS - MTGP32_N + tid + 1],
		     status[MTGP32_LS - MTGP32_N + tid + pos]);
	status[tid] = r;
	o = temper_single(r, status[MTGP32_LS - MTGP32_N + tid + pos - 1]);
	d_data[size * bid + i + tid] = o;
	__syncthreads();
	r = para_rec(status[(4 * MTGP32_TN - MTGP32_N + tid) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + tid + 1) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + tid + pos)
			    % MTGP32_LS]);
	status[tid + MTGP32_TN] = r;
	o = temper_single(
	    r,
	    status[(4 * MTGP32_TN - MTGP32_N + tid + pos - 1) % MTGP32_LS]);
	d_data[size * bid + MTGP32_TN + i + tid] = o;
	__syncthreads();
	r = para_rec(status[2 * MTGP32_TN - MTGP32_N + tid],
		     status[2 * MTGP32_TN - MTGP32_N + tid + 1],
		     status[2 * MTGP32_TN - MTGP32_N + tid + pos]);
	status[tid + 2 * MTGP32_TN] = r;
	o = temper_single(r,
			  status[tid + pos - 1 + 2 * MTGP32_TN - MTGP32_N]);
	d_data[size * bid + 2 * MTGP32_TN + i + tid] = o;
	__syncthreads();
    }
    // write back status for next call
    status_write(d_status, status, bid, tid);
}
#endif
