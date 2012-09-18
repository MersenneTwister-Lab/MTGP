/*
 * @file mtgp32-jump.cl
 *
 * @brief Sample Program for openCL 1.1
 *
 * This program changes internal state of MTGP to jumped state.
 */

/*
 * Generator Parameters.
 */

/* No.1 delta:686 weight:1659 */
#define MTGP32_MEXP 11213
#define MTGP32_N 351
#define MTGP32_FLOOR_2P 256
#define MTGP32_CEIL_2P 512
#define MTGP32_TN MTGP32_FLOOR_2P
#define MTGP32_LS (MTGP32_TN * 3)
#define MTGP32_TS 16

__constant int mtgp32_pos = 84;
__constant uint mtgp32_sh1 = 12;
__constant uint mtgp32_sh2 = 4;
__constant uint mtgp32_param_tbl[MTGP32_TS]
= {0x00000000, 0x71588353, 0xdfa887c1, 0xaef00492,
   0x4ba66c6e, 0x3afeef3d, 0x940eebaf, 0xe55668fc,
   0xa53da0ae, 0xd46523fd, 0x7a95276f, 0x0bcda43c,
   0xee9bccc0, 0x9fc34f93, 0x31334b01, 0x406bc852};
__constant uint mtgp32_temper_tbl[MTGP32_TS]
= {0x00000000, 0x200040bb, 0x1082c61e, 0x308286a5,
   0x10021c03, 0x30025cb8, 0x0080da1d, 0x20809aa6,
   0x0003f0b9, 0x2003b002, 0x108136a7, 0x3081761c,
   0x1001ecba, 0x3001ac01, 0x00832aa4, 0x20836a1f};
__constant uint mtgp32_single_temper_tbl[MTGP32_TS]
= {0x3f800000, 0x3f900020, 0x3f884163, 0x3f984143,
   0x3f88010e, 0x3f98012e, 0x3f80406d, 0x3f90404d,
   0x3f8001f8, 0x3f9001d8, 0x3f88409b, 0x3f9840bb,
   0x3f8800f6, 0x3f9800d6, 0x3f804195, 0x3f9041b5};
__constant uint mtgp32_mask = 0xfff80000;
__constant uint mtgp32_non_zero = 0x4d544750;

/* jump polynomial for 3^162 steps jump */

/**
 * The function of the recursion formula calculation.
 *
 * @param[in] X1 the farthest part of state array.
 * @param[in] X2 the second farthest part of state array.
 * @param[in] Y a part of state array.
 * @return output
 */

static inline uint para_rec(uint X1, uint X2, uint Y)
{
    uint X = (X1 & mtgp32_mask) ^ X2;
    uint MAT;

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
 * @return the tempered value.
 */
static inline uint temper(uint V, uint T)
{
    uint MAT;

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
 * @return the tempered and converted value.
 */

static inline uint temper_single(uint V, uint T)
{
    uint MAT;
    uint r;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = mtgp32_single_temper_tbl[T & 0x0f];
    r = (V >> 9) ^ MAT;
    return r;
}

/**
 * Read the internal state vector from kernel I/O data, and
 * put them into local memory.
 *
 * @param[out] status shared memory.
 * @param[in] d_status kernel I/O data
 * @param[in] gid block id
 * @param[in] lid thread id
 */
static inline void status_read(__local uint status[MTGP32_LS],
			       __global uint *d_status,
			       int gid,
			       int lid)
{
    status[MTGP32_LS - MTGP32_N + lid]
	= d_status[gid * MTGP32_N + lid];
    if (lid < MTGP32_N - MTGP32_TN) {
	status[MTGP32_LS - MTGP32_N + MTGP32_TN + lid]
	    = d_status[gid * MTGP32_N + MTGP32_TN + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

/**
 * Read the internal state vector from local memory, and
 * write them into kernel I/O data.
 *
 * @param[out] d_status kernel I/O data
 * @param[in] status shared memory.
 * @param[in] gid block id
 * @param[in] lid thread id
 */
static inline void status_write(__global uint *d_status,
				__local uint status[MTGP32_LS],
				int gid,
				int lid)
 {
    d_status[gid * MTGP32_N + lid]
	= status[MTGP32_LS - MTGP32_N + lid];
    if (lid < MTGP32_N - MTGP32_TN) {
	d_status[gid * MTGP32_N + MTGP32_TN + lid]
	    = status[4 * MTGP32_TN - MTGP32_N + lid];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/**
 * This function represents a function used in the initialization
 * by mtgp32_init_by_array() and mtgp32_init_by_str().
 * @param[in] x 32-bit integer
 * @return 32-bit integer
 */
static inline uint mtgp32_ini_func1(uint x)
{
    return (x ^ (x >> 27)) * 1664525U;
}

/**
 * This function represents a function used in the initialization
 * by mtgp32_init_by_array() and mtgp32_init_by_str().
 * @param[in] x 32-bit integer
 * @return 32-bit integer
 */
static inline uint mtgp32_ini_func2(uint x)
{
    return (x ^ (x >> 27)) * 1566083941U;
}

/**
 * This function initializes the internal state array with a 32-bit
 * integer seed.
 * @param[in] seed a 32-bit integer used as the seed.
 */
static inline void mtgp32_init_state(__local uint status[],
				     uint seed)
{
    int i;
    uint hidden_seed;
    uint tmp;
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    hidden_seed = mtgp32_param_tbl[4] ^ (mtgp32_param_tbl[8] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    tmp &= 0xff;
    tmp |= tmp << 8;
    tmp |= tmp << 16;
    //memset(mtgp32_status, tmp & 0xff, sizeof(uint) * size);
    status[lid] = tmp;
    if ((local_size < MTGP32_N) && (lid < MTGP32_N - MTGP32_TN)) {
	status[MTGP32_TN + lid] = tmp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
	status[0] = seed;
	status[1] = hidden_seed;
	for (i = 1; i < MTGP32_N; i++) {
	    status[i] ^= i + 1812433253U * (status[i - 1]
					    ^ (status[i - 1] >> 30));
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
static inline void mtgp32_init_by_array(__local uint status[],
					__global uint *seed_array,
					int length)
{
    int i, j, count;
    uint r;
    int lag;
    int mid;
    int size = MTGP32_N;
    uint hidden_seed;
    uint tmp;
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

//    st = mtgp32_status;
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
    status[lid] = tmp;

    if ((local_size < MTGP32_N) && (lid < MTGP32_N - MTGP32_TN)) {
	status[MTGP32_TN + lid] = tmp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid != 0) {
	return;
    }
    status[0] = hidden_seed;
    if (length + 1 > size) {
	count = length + 1;
    } else {
	count = size;
    }
    r = mtgp32_ini_func1(status[0]
			 ^ status[mid]
			 ^ status[size - 1]);
    status[mid] += r;
    r += length;
    status[(mid + lag) % size] += r;
    status[0] = r;
    i = 1;
    count--;
    for (i = 1, j = 0; (j < count) && (j < length); j++) {
	r = mtgp32_ini_func1(status[i]
			     ^ status[(i + mid) % size]
			     ^ status[(i + size - 1) % size]);
	status[(i + mid) % size] += r;
	r += seed_array[j] + i;
	status[(i + mid + lag) % size] += r;
	status[i] = r;
	i = (i + 1) % size;
    }
    for (; j < count; j++) {
	r = mtgp32_ini_func1(status[i]
			     ^ status[(i + mid) % size]
			     ^ status[(i + size - 1) % size]);
	status[(i + mid) % size] += r;
	r += i;
	status[(i + mid + lag) % size] += r;
	status[i] = r;
	i = (i + 1) % size;
    }
    for (j = 0; j < size; j++) {
	r = mtgp32_ini_func2(status[i]
			     + status[(i + mid) % size]
			     + status[(i + size - 1) % size]);
	status[(i + mid) % size] ^= r;
	r -= i;
	status[(i + mid + lag) % size] ^= r;
	status[i] = r;
	i = (i + 1) % size;
    }
    if (status[size - 1] == 0) {
	status[size - 1] = mtgp32_non_zero;
    }
}

/**
 * device function.
 * This function changes internal state of MTGP to jumped state.
 * threads per block should be MTGP32_N.
 *
 * @param[in,out] d_status kernel I/O data
 * @param[in] jump_poly jump polynomial
 */
static inline void mtgp32_jump(int gid,
			       int lid,
			       __local uint work[],
			       __local uint status[],
			       __global uint jump_poly[])
{
    int index = 0;
    uint r;
    int pos = mtgp32_pos;
    uint bits;
    int i, j;
    const int local_size = get_local_size(0);

    work[lid] = 0;
    if ((local_size < MTGP32_N) && (lid < MTGP32_N - MTGP32_TN)) {
	work[MTGP32_TN + lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // jump
    for (i = 0; i < MTGP32_N; i++) {
	bits = jump_poly[i];
	for (j = 0; j < 32; j++) {
	    if ((bits & 1) != 0) {
		//add
		work[lid] ^= status[(lid + index) % MTGP32_N];
		if ((local_size < MTGP32_N) && (lid < MTGP32_N - MTGP32_TN)) {
		    work[MTGP32_TN + lid]
			^= status[(MTGP32_TN + lid + index) % MTGP32_N];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	    }
	    //next_state
	    if (lid == 0) {
		r = para_rec(status[(lid + index) % MTGP32_N],
			     status[(lid + index + 1) % MTGP32_N],
			     status[(lid + index + pos) % MTGP32_N]);
		status[(lid + index) % MTGP32_N] = r;
	    }
	    barrier(CLK_LOCAL_MEM_FENCE);
	    index = (index + 1) % MTGP32_N;
	    bits = bits >> 1;
	}
    }
}

/**
 *
 *
 */
static inline void mtgp32_table_jump(int gid,
				     int lid,
				     __global uint * jump_table,
				     __local uint * work,
				     __local uint * status)
{
    int mask = 1;
    int i;
    int idx;
    const int local_size = get_local_size(0);

    for (i = 0; (i < 32) && (mask <= gid); i++) {
	idx = i / 2;
	if ((gid & mask) != 0) {
	    if (i % 2 == 0) {
		mtgp32_jump(gid, lid, work, status,
			    &jump_table[idx * MTGP32_N]);
		barrier(CLK_LOCAL_MEM_FENCE);
		status[lid] = work[lid];
		if ((local_size < MTGP32_N) && (lid < MTGP32_N - MTGP32_TN)) {
		    status[MTGP32_TN + lid] = work[MTGP32_TN + lid];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	    } else {
		mtgp32_jump(gid, lid, work, status,
			    &jump_table[idx * MTGP32_N]);
		barrier(CLK_LOCAL_MEM_FENCE);
		mtgp32_jump(gid, lid, status, work,
			    &jump_table[idx * MTGP32_N]);
		barrier(CLK_LOCAL_MEM_FENCE);
	    }
	}
	mask = mask << 1;
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
__kernel void mtgp32_jump_kernel(__global uint * d_status,
				 __global uint jump_poly[],
				 int count)
{

    __local uint work[MTGP32_N];
    __local uint status[MTGP32_N];

    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    // copy status data from global memory to shared memory.
    status[lid] = d_status[gid * MTGP32_N + lid];
    if ((local_size < MTGP32_N) && (lid < MTGP32_N - MTGP32_TN)) {
	status[MTGP32_TN + lid] = d_status[gid * MTGP32_N + MTGP32_TN + lid];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // jump
    for (int i = 0; i < count; i++) {
	mtgp32_jump(gid, lid, work, status, jump_poly);
	barrier(CLK_LOCAL_MEM_FENCE);
	status[lid] = work[lid];
	if (local_size < MTGP32_N) {
	    if (lid < MTGP32_N - MTGP32_TN) {
		status[MTGP32_TN + lid] ^= work[MTGP32_TN + lid];
	    }
	}
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    d_status[gid * MTGP32_N + lid] = status[lid];
    if ((local_size < MTGP32_N) && (lid < MTGP32_N - MTGP32_TN)) {
	d_status[gid * MTGP32_N + MTGP32_TN + lid]
	    = status[MTGP32_TN + lid];
    }
}

/**
 * kernel function.
 * This function changes internal state of MTGP to jumped state.
 * threads per block should be MTGP32_N.
 *
 * @param[in,out] d_status kernel I/O data
 */
__kernel void mtgp32_jump_long_seed_kernel(__global uint * d_status,
					   uint seed,
					   __global uint * jump_table)
{
    __local uint work[MTGP32_N];
    __local uint status[MTGP32_N];

    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    // initialize
    mtgp32_init_state(status, seed);
    barrier(CLK_LOCAL_MEM_FENCE);

    // jump
    mtgp32_table_jump(gid, lid, jump_table, work, status);
    d_status[gid * MTGP32_N + lid] = status[lid];
    if ((local_size < MTGP32_N) && (lid < MTGP32_N - MTGP32_TN)) {
	d_status[gid * MTGP32_N + MTGP32_TN + lid]
	    = status[MTGP32_TN + lid];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/**
 * kernel function.
 * This function changes internal state of MTGP to jumped state.
 * threads per block should be MTGP32_N.
 *
 * @param[in,out] d_status kernel I/O data
 */
__kernel void mtgp32_jump_long_array_kernel(__global uint * d_status,
					    __global uint * seed_array,
					    int length,
					    __global uint * jump_table)
{
    __local uint work[MTGP32_N];
    __local uint status[MTGP32_N];
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    // initialize
    mtgp32_init_by_array(status, seed_array, length);
    barrier(CLK_LOCAL_MEM_FENCE);

    // jump
    mtgp32_table_jump(gid, lid, jump_table, work, status);
    d_status[gid * MTGP32_N + lid] = status[lid];
    if ((local_size < MTGP32_N) && (lid < MTGP32_N - MTGP32_TN)) {
	d_status[gid * MTGP32_N + MTGP32_TN + lid]
	    = status[MTGP32_TN + lid];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/**
 * kernel function.
 * This function changes internal state of MTGP to jumped state.
 * threads per block should be MTGP32_N.
 *
 * @param[in,out] d_status kernel I/O data
 */
__kernel void mtgp32_jump_seed_kernel(__global uint * d_status,
				      __global uint jump_table[],
				      uint seed)
{
    __local uint work[MTGP32_N];
    __local uint status[MTGP32_N];
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    // initialize
    mtgp32_init_state(status, seed);
    barrier(CLK_LOCAL_MEM_FENCE);

    // jump
    mtgp32_table_jump(gid, lid, jump_table, work, status);
    d_status[gid * MTGP32_N + lid] = status[lid];
    if ((local_size < MTGP32_N) && (lid < MTGP32_N - MTGP32_TN)) {
	d_status[gid * MTGP32_N + MTGP32_TN + lid] = status[MTGP32_TN + lid];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/**
 * kernel function.
 * This function changes internal state of MTGP to jumped state.
 * threads per block should be MTGP32_N.
 *
 * @param[in,out] d_status kernel I/O data
 */
__kernel void mtgp32_jump_array_kernel(__global uint * d_status,
				       __global uint jump_table[],
				       __global uint * seed_array,
				       int length)
{
    __local uint work[MTGP32_N];
    __local uint status[MTGP32_N];
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    // initialize
    mtgp32_init_by_array(status, seed_array, length);
    barrier(CLK_LOCAL_MEM_FENCE);

    // jump
    mtgp32_table_jump(gid, lid, jump_table, work, status);
    d_status[gid * MTGP32_N + lid] = status[lid];
    if ((local_size < MTGP32_N) && (lid < MTGP32_N - MTGP32_TN)) {
	d_status[gid * MTGP32_N + MTGP32_TN + lid] = status[MTGP32_TN + lid];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/**
 * kernel function.
 * This function generates 32-bit unsigned integers in d_data
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output
 * @param[in] size number of output data requested.
 */
__kernel void mtgp32_uint32_kernel(__global uint * d_status,
				   __global uint * d_data,
				   int size)
{
    __local uint status[MTGP32_LS];
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    int pos = mtgp32_pos;
    uint r;
    uint o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, gid, lid);

    // main loop
    for (int i = 0; i < size; i += MTGP32_LS) {
	r = para_rec(status[MTGP32_LS - MTGP32_N + lid],
		 status[MTGP32_LS - MTGP32_N + lid + 1],
		 status[MTGP32_LS - MTGP32_N + lid + pos]);
	status[lid] = r;
	o = temper(r, status[MTGP32_LS - MTGP32_N + lid + pos - 1]);
	d_data[size * gid + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(status[(4 * MTGP32_TN - MTGP32_N + lid) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + lid + 1) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + lid + pos)
			    % MTGP32_LS]);
	status[lid + MTGP32_TN] = r;
	o = temper(r,
		   status[(4 * MTGP32_TN - MTGP32_N + lid + pos - 1)
			  % MTGP32_LS]);
	d_data[size * gid + MTGP32_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(status[2 * MTGP32_TN - MTGP32_N + lid],
		     status[2 * MTGP32_TN - MTGP32_N + lid + 1],
		     status[2 * MTGP32_TN - MTGP32_N + lid + pos]);
	status[lid + 2 * MTGP32_TN] = r;
	o = temper(r, status[lid + pos - 1 + 2 * MTGP32_TN - MTGP32_N]);
	d_data[size * gid + 2 * MTGP32_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write back status for next call
    status_write(d_status, status, gid, lid);
}

/**
 * kernel function.
 * This function generates single precision floating point numbers in d_data.
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output. IEEE single precision format.
 * @param[in] size number of output data requested.
 */
__kernel void mtgp32_single_kernel(__global uint * d_status,
				   __global uint* d_data,
				   int size)
{
    __local uint status[MTGP32_LS];
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    int pos = mtgp32_pos;
    uint r;
    uint o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, gid, lid);

    // main loop
    for (int i = 0; i < size; i += MTGP32_LS) {
	r = para_rec(status[MTGP32_LS - MTGP32_N + lid],
		     status[MTGP32_LS - MTGP32_N + lid + 1],
		     status[MTGP32_LS - MTGP32_N + lid + pos]);
	status[lid] = r;
	o = temper_single(r, status[MTGP32_LS - MTGP32_N + lid + pos - 1]);
	d_data[size * gid + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(status[(4 * MTGP32_TN - MTGP32_N + lid) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + lid + 1) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + lid + pos)
			    % MTGP32_LS]);
	status[lid + MTGP32_TN] = r;
	o = temper_single(
	    r,
	    status[(4 * MTGP32_TN - MTGP32_N + lid + pos - 1) % MTGP32_LS]);
	d_data[size * gid + MTGP32_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(status[2 * MTGP32_TN - MTGP32_N + lid],
		     status[2 * MTGP32_TN - MTGP32_N + lid + 1],
		     status[2 * MTGP32_TN - MTGP32_N + lid + pos]);
	status[lid + 2 * MTGP32_TN] = r;
	o = temper_single(r,
			  status[lid + pos - 1 + 2 * MTGP32_TN - MTGP32_N]);
	d_data[size * gid + 2 * MTGP32_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write back status for next call
    status_write(d_status, status, gid, lid);
}

