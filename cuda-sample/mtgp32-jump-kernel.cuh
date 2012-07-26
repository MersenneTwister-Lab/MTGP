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
#define MTGP32_JUMP2_256 "931879d2d427b2e2f79edea1ece351638d73bee8f0abc183f4bdd827fcfe6a026689bca36beec17eed784dfc6855069d322627421a6964db1a85abddb82d8eb30842b0963787baab453b8a637ea8fb46481677a4168837e1cd5f95ca6400da721a3049ff1e01033d5434007c8ad40cc30d3a22c0c0f606be6b3aa1507c6be942c257b092b46d03d1a6633c9d297292c8bfb70205e14d02f4068b434d8474a046c455f3523eec0ff154eaf0841fa1b1aaa7bb861ce859a4b607e5704ffdcf6a65b7ad71d5d41ba454bee8847d8340bb8663a67911ac729d4a9328b14cf794ac6c41858e9b54e1e541ea1ea8f6a8dcaaea1122e33b8e4cdab8d12d3fc028428075a0d03298226c51c116b74606f9c73e12e9f3ac8be6b63ecd88776c471c8a7f43f3bd095cf1d0716e2b24c3836f9d7ae65de449554fe81dcb346af787f4f9590591088609487025e4128a56c805a4d9a5defd61776824c08dd8305a790beb6fad80cbc20d6db8f9ea52b421656507223a41738a09292199dafcb673d854f06c0ef5dd542e8a185fffa7a604df4d9532882d8b735f1cc2ee7a2a24d5770c86c4a86eae43d582833b209676616a1a0681e8fe208445409269b9c6229c64d22f8ef11037ec1d30e8b76f0d1003d1c6caf72f51dd96e1b02f8ddbc8ce9d6549d379fa8cabd0e911dc9f48fc3b3937b9f40471fde677804d22de561b3641d709b7eb365af249d09352aa4b9e2838be43d3686da5ab75eb56c1df71e71fd9203a20d21df36aadbcffc3922c0bc29370c0721d824c667aa8594881572aae1920a50abad4487bc9f1fed1c882050bd3efc598b5cdbe4b2628cef56dba810f48f2803ca7dc703597c8487f0af903dc85ddfecd07fe91e46c944cf627579e6478b8031a467ebfacfe25ee4f41e730ac93ab8526141881d1890c6dad6b652da40bde4cddb4416abc2f45f7b8b29cf34278368c1dc178d4e6d41d5a27d17cbf6401fe8a80d5ddb20df07abdc1933d9075ca563ecc121a5372cfecdf73588cd5af54fa9e2df81d59c8d53ebfa17fa2720e23653414b8695eacdb08b3411893b0af7897adc8fd86e78c3ec2982c39aec30ac3f026a451204aa53f841da8ed2098a14bad8cb99332f9e1241142fec6a3030d96696c6e3b1fa05077786a066d65922eca5058e68eba17778489acc3f9f85767980e81fef7b22509fae89080a70430e1f70826ba929472a71d696ba1f4f47a20f64bda0a8d06fbe13ffa53823a9719b63ca116f0f7f7c5470f96cf0c75a2f8068f2cf666e629955293265363115abaff2d4ed79c0b137bbc03431b84ca3d62bdbb6ea81a00babf2f16ff1cfdfc5e4a8095d6995c37c55dc9bba9c1e6da314e44c48da8a2d371e20d7ca64c83f5e895cd1b7a3174edaaa42b77d9e344dacb6be68c2e25d20d4a3128302bf24d935635038d1b95489cabcf910709f274e60d8d4d9c2e9b7c4bf997ca76f0a5cc0ac6f27cf4e36ce20d0bce535755b9019f4e18d30dd82fe7f6eafae8090833e3a7b1131f0aa9509c6deee47147c7d7fccfc221367da8ff33a610e8e7a3c11f0071a329b02c8e74e759bc316e6b07ab1ae151b5cbc963e737f0089311face9e854cf9694bbb05048d0d0b876521873dd63737dc7de876955f35a98bb730143adc66882f211ac262266315d2842ff4e32469a28afaec4f9f8ce97b8e1c217f265ea27e1ab0fcd92e41f723dbc9b64081193c2815cb8fb96672231fea53dd5c5d52eb4bdf77846cee1b8a1ff68649c6a312cc9dc360d58dc418bbe44895998bb5abb355bf200d45a5a1cd434f35a528c3a027d744645ecd982f183a830c875271f11604eb293c7a3003e296335380d040230461c85a8958e09b4b27b539dafd0f585fae231d909a0a97f47d2a2007e84f859da0c9438d118fbea3057e42fec7135ecdd217d94676faae92a74c25cb16ddc91de1e05fdb7c99b8af57121502f07c00d6a248a"

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

/* jump polynomial for 2^256 step jump */
__constant__ char mtgp32_jump2_256[] = MTGP32_JUMP2_256;


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
			    char* jump_poly) {
    const int tid = threadIdx.x;
    int index = 0;
    uint32_t r;
    int pos = mtgp32_pos;
    work[tid] = 0;

    // jump
    for (int i = 0; jump_poly[i] != '\0'; i++) {
	char bits = jump_poly[i];
	if (bits >= 'a' && bits <= 'f') {
	    bits = bits - 'a' + 10;
	} else {
	    bits = bits - '0';
	}
	bits = bits & 0x0f;
	for (int j = 0; j < 4; j++) {
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
				   char* jump_poly, int count) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    // copy status data from global memory to shared memory.
    mtgp32_jstatus[tid] = d_status[bid].status[tid];
    __syncthreads();
    // jump
    if (tid == 0) {
	for (int i = 0; i < count; i++) {
	    mtgp32_jump(mtgp32_jwork, mtgp32_jstatus, jump_poly);
	    __syncthreads();
	    mtgp32_jstatus[tid] = mtgp32_jwork[tid];
	    __syncthreads();
	}
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
    tmp <<= 8;
    tmp <<= 16;
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
    tmp <<= 8;
    tmp <<= 16;
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
    //int pos = mtgp32_pos;
    //int index = 0;
    //uint32_t r;

    // copy status data from global memory to shared memory.
    //mtgp32_jstatus[tid] = d_status[bid].status[tid];
    //__syncthreads();
    mtgp32_init_state(seed);
    __syncthreads();

    // jump
    for (int i = 0; i < bid; i++) {
	mtgp32_jump(mtgp32_jwork, mtgp32_jstatus, mtgp32_jump2_256);
	__syncthreads();
	mtgp32_jstatus[tid] = mtgp32_jwork[tid];
    }
    __syncthreads();
    d_status[bid].status[tid] = mtgp32_jstatus[tid];
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

    // copy status data from global memory to shared memory.
//    mtgp32_jstatus[tid] = d_status[bid].status[tid];
//    __syncthreads();
    mtgp32_init_by_array(seed_array, length);
    __syncthreads();

    // jump
    for (int i = 0; i < bid; i++) {
	mtgp32_jump(mtgp32_jwork, mtgp32_jstatus, mtgp32_jump2_256);
	__syncthreads();
	mtgp32_jstatus[tid] = mtgp32_jwork[tid];
    }
    __syncthreads();
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

#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
	if ((i == 0) && (bid == 0) && (tid <= 1)) {
	    printf("status[MTGP32_LS - MTGP32_N + tid]:%08x\n",
		   status[MTGP32_LS - MTGP32_N + tid]);
	    printf("status[MTGP32_LS - MTGP32_N + tid + 1]:%08x\n",
		   status[MTGP32_LS - MTGP32_N + tid + 1]);
	    printf("status[MTGP32_LS - MTGP32_N + tid + pos]:%08x\n",
		   status[MTGP32_LS - MTGP32_N + tid + pos]);
	    printf("sh1:%d\n", sh1_tbl[bid]);
	    printf("sh2:%d\n", sh2_tbl[bid]);
	    printf("mask:%08x\n", mask[0]);
	    for (int j = 0; j < 16; j++) {
		printf("tbl[%d]:%08x\n", j, param_tbl[0][j]);
	    }
	}
#endif
	r = para_rec(status[MTGP32_LS - MTGP32_N + tid],
		 status[MTGP32_LS - MTGP32_N + tid + 1],
		 status[MTGP32_LS - MTGP32_N + tid + pos]);
	status[tid] = r;
#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
	if ((i == 0) && (bid == 0) && (tid <= 1)) {
	    printf("status[tid]:%08x\n", status[tid]);
	}
#endif
	o = temper(r, status[MTGP32_LS - MTGP32_N + tid + pos - 1]);
#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
	if ((i == 0) && (bid == 0) && (tid <= 1)) {
	    printf("r:%08" PRIx32 "\n", r);
	}
#endif
	d_data[size * bid + i + tid] = o;
	__syncthreads();
	r = para_rec(status[(4 * MTGP32_TN - MTGP32_N + tid) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + tid + 1) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + tid + pos) % MTGP32_LS]);
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
		     status[(4 * MTGP32_TN - MTGP32_N + tid + pos) % MTGP32_LS]);
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
