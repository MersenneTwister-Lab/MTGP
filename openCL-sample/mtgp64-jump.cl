/*
 * @file mtgp64-jump.cl
 *
 * @brief Sample Program for openCL 1.2
 *
 * This program changes internal state of MTGP to jumped state.
 */

/*
 * Generator Parameters.
 */
/* No.1 delta:686 weight:1659 */
#define MTGP64_MEXP 11213
#define MTGP64_N 176
#define MTGP64_FLOOR_2P 128
#define MTGP64_CEIL_2P 256
#define MTGP64_TN MTGP64_FLOOR_2P
#define MTGP64_LS (MTGP64_TN * 3)
#define MTGP64_TS 16

__constant int mtgp64_pos = 45;
__constant uint mtgp64_sh1 = 11;
__constant uint mtgp64_sh2 = 4;
__constant ulong mtgp64_param_tbl[MTGP64_TS]
= {0x0000000000000000UL,
   0xacd0c7eb00000000UL,
   0x63f8aada00000000UL,
   0xcf286d3100000000UL,
   0x966a000000000000UL,
   0x3abac7eb00000000UL,
   0xf592aada00000000UL,
   0x59426d3100000000UL,
   0x0000367100000000UL,
   0xacd0f19a00000000UL,
   0x63f89cab00000000UL,
   0xcf285b4000000000UL,
   0x966a367100000000UL,
   0x3abaf19a00000000UL,
   0xf5929cab00000000UL,
   0x59425b4000000000UL};
__constant ulong mtgp64_temper_tbl[MTGP64_TS]
= {0x0000000000000000UL,
   0x2000000000000000UL,
   0xc000000000000000UL,
   0xe000000000000000UL,
   0x0901000000000000UL,
   0x2901000000000000UL,
   0xc901000000000000UL,
   0xe901000000000000UL,
   0x401bf00000000000UL,
   0x601bf00000000000UL,
   0x801bf00000000000UL,
   0xa01bf00000000000UL,
   0x491af00000000000UL,
   0x691af00000000000UL,
   0x891af00000000000UL,
   0xa91af00000000000UL};
__constant ulong mtgp64_double_temper_tbl[MTGP64_TS]
= {0x3ff0000000000000UL,
   0x3ff2000000000000UL,
   0x3ffc000000000000UL,
   0x3ffe000000000000UL,
   0x3ff0901000000000UL,
   0x3ff2901000000000UL,
   0x3ffc901000000000UL,
   0x3ffe901000000000UL,
   0x3ff401bf00000000UL,
   0x3ff601bf00000000UL,
   0x3ff801bf00000000UL,
   0x3ffa01bf00000000UL,
   0x3ff491af00000000UL,
   0x3ff691af00000000UL,
   0x3ff891af00000000UL,
   0x3ffa91af00000000UL};
__constant ulong mtgp64_mask = 0xfff8000000000000UL;
__constant ulong mtgp64_non_zero = 0x4d544750;

/* =========================
   declarations
   ========================= */
static inline ulong para_rec(ulong X1, ulong X2, ulong Y);
static inline ulong temper(ulong V, ulong T);
static inline ulong temper_double(ulong V, ulong T);
static inline void status_read(__local ulong * status,
			       __global ulong * d_status,
			       int gid,
			       int lid);
static inline void status_write(__global ulong * d_status,
				__local ulong * status,
				int gid,
				int lid);
static inline ulong mtgp64_ini_func1(ulong x);
static inline ulong mtgp64_ini_func2(ulong x);
static inline void mtgp64_init_state(__local ulong * status,
				     ulong seed);
static inline void mtgp64_init_by_array(__local ulong * status,
					__global ulong * seed_array,
					int length);
static inline void mtgp64_jump(int gid,
			       int lid,
			       __local ulong * work,
			       __local ulong * status,
			       __global uint * jump_poly);
static inline void mtgp64_table_jump(int gid,
				     int lid,
				     __global uint * jump_table,
				     __local ulong * work,
				     __local ulong * status);

/* ================================ */
/* mtgp64 sample kernel code        */
/* ================================ */
/**
 * kernel function.
 * This function changes internal state of MTGP to jumped state.
 * threads per block should be MTGP64_N.
 *
 * @param[in,out] d_status kernel I/O data
 * @param[in] jump_poly jump polynomial
 */
__kernel void mtgp64_jump_kernel(__global ulong * d_status,
				 __global uint * jump_poly)
{
    __local ulong work[MTGP64_N];
    __local ulong status[MTGP64_N];

    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    // copy status data from global memory to local memory.
    status[lid] = d_status[gid * MTGP64_N + lid];
    if ((local_size < MTGP64_N) && (lid < MTGP64_N - MTGP64_TN)) {
	status[MTGP64_TN + lid] = d_status[gid * MTGP64_N + MTGP64_TN + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // jump
    mtgp64_jump(gid, lid, work, status, jump_poly);
    barrier(CLK_LOCAL_MEM_FENCE);
    status[lid] = work[lid];
    if (local_size < MTGP64_N) {
	if (lid < MTGP64_N - MTGP64_TN) {
	    status[MTGP64_TN + lid] = work[MTGP64_TN + lid];
	}
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // copy status data from local memory to global memory
    d_status[gid * MTGP64_N + lid] = status[lid];
    if ((local_size < MTGP64_N) && (lid < MTGP64_N - MTGP64_TN)) {
	d_status[gid * MTGP64_N + MTGP64_TN + lid] = status[MTGP64_TN + lid];
    }
}

/**
 * kernel function.
 * This function changes internal state of MTGP to jumped state.
 * threads per block should be MTGP64_N.
 *
 * @param[in,out] d_status kernel I/O data
 */
__kernel void mtgp64_jump_seed_kernel(__global ulong * d_status,
				      ulong seed,
				      __global uint * jump_table)
{
    __local ulong work[MTGP64_N];
    __local ulong status[MTGP64_N];

    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    // initialize
    mtgp64_init_state(status, seed);
    barrier(CLK_LOCAL_MEM_FENCE);

    // jump
    mtgp64_table_jump(gid, lid, jump_table, work, status);
    d_status[gid * MTGP64_N + lid] = status[lid];
    if ((local_size < MTGP64_N) && (lid < MTGP64_N - MTGP64_TN)) {
	d_status[gid * MTGP64_N + MTGP64_TN + lid] = status[MTGP64_TN + lid];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/**
 * kernel function.
 * This function changes internal state of MTGP to jumped state.
 * threads per block should be MTGP64_N.
 *
 * @param[in,out] d_status kernel I/O data
 */
__kernel void mtgp64_jump_array_kernel(__global ulong * d_status,
				       __global ulong * seed_array,
				       int length,
				       __global uint * jump_table)
{
    __local ulong work[MTGP64_N];
    __local ulong status[MTGP64_N];

    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    // initialize
    mtgp64_init_by_array(status, seed_array, length);
    barrier(CLK_LOCAL_MEM_FENCE);

    // jump
    mtgp64_table_jump(gid, lid, jump_table, work, status);
    d_status[gid * MTGP64_N + lid] = status[lid];
    if ((local_size < MTGP64_N) && (lid < MTGP64_N - MTGP64_TN)) {
	d_status[gid * MTGP64_N + MTGP64_TN + lid] = status[MTGP64_TN + lid];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/**
 * kernel function.
 * This function generates 64-bit unsigned integers in d_data
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output
 * @param[in] size number of output data requested.
 */
__kernel void mtgp64_uint64_kernel(__global ulong * d_status,
				   __global ulong * d_data,
				   int size)
{
    __local ulong status[MTGP64_LS];
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    int pos = mtgp64_pos;
    ulong r;
    ulong o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, gid, lid);

    // main loop
    for (int i = 0; i < size; i += MTGP64_LS) {
	r = para_rec(status[MTGP64_LS - MTGP64_N + lid],
		 status[MTGP64_LS - MTGP64_N + lid + 1],
		 status[MTGP64_LS - MTGP64_N + lid + pos]);
	status[lid] = r;
	o = temper(r, status[MTGP64_LS - MTGP64_N + lid + pos - 1]);
	d_data[size * gid + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(status[(4 * MTGP64_TN - MTGP64_N + lid) % MTGP64_LS],
		     status[(4 * MTGP64_TN - MTGP64_N + lid + 1) % MTGP64_LS],
		     status[(4 * MTGP64_TN - MTGP64_N + lid + pos)
			    % MTGP64_LS]);
	status[lid + MTGP64_TN] = r;
	o = temper(r,
		   status[(4 * MTGP64_TN - MTGP64_N + lid + pos - 1)
			  % MTGP64_LS]);
	d_data[size * gid + MTGP64_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(status[2 * MTGP64_TN - MTGP64_N + lid],
		     status[2 * MTGP64_TN - MTGP64_N + lid + 1],
		     status[2 * MTGP64_TN - MTGP64_N + lid + pos]);
	status[lid + 2 * MTGP64_TN] = r;
	o = temper(r, status[lid + pos - 1 + 2 * MTGP64_TN - MTGP64_N]);
	d_data[size * gid + 2 * MTGP64_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write back status for next call
    status_write(d_status, status, gid, lid);
}

/**
 * kernel function.
 * This function generates double precision floating point numbers in d_data.
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output. IEEE double precision format.
 * @param[in] size number of output data requested.
 */
__kernel void mtgp64_double12_kernel(__global ulong * d_status,
				     __global ulong * d_data,
				     int size)
{
    __local ulong status[MTGP64_LS];
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    int pos = mtgp64_pos;
    ulong r;
    ulong o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, gid, lid);

    // main loop
    for (int i = 0; i < size; i += MTGP64_LS) {
	r = para_rec(status[MTGP64_LS - MTGP64_N + lid],
		     status[MTGP64_LS - MTGP64_N + lid + 1],
		     status[MTGP64_LS - MTGP64_N + lid + pos]);
	status[lid] = r;
	o = temper_double(r, status[MTGP64_LS - MTGP64_N + lid + pos - 1]);
	d_data[size * gid + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(status[(4 * MTGP64_TN - MTGP64_N + lid) % MTGP64_LS],
		     status[(4 * MTGP64_TN - MTGP64_N + lid + 1) % MTGP64_LS],
		     status[(4 * MTGP64_TN - MTGP64_N + lid + pos)
			    % MTGP64_LS]);
	status[lid + MTGP64_TN] = r;
	o = temper_double(
	    r,
	    status[(4 * MTGP64_TN - MTGP64_N + lid + pos - 1) % MTGP64_LS]);
	d_data[size * gid + MTGP64_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(status[2 * MTGP64_TN - MTGP64_N + lid],
		     status[2 * MTGP64_TN - MTGP64_N + lid + 1],
		     status[2 * MTGP64_TN - MTGP64_N + lid + pos]);
	status[lid + 2 * MTGP64_TN] = r;
	o = temper_double(r,
			  status[lid + pos - 1 + 2 * MTGP64_TN - MTGP64_N]);
	d_data[size * gid + 2 * MTGP64_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write back status for next call
    status_write(d_status, status, gid, lid);
}

#if defined(HAVE_DOUBLE)
#pragma OPENCL_EXTENSION cl_khr_fp64 : enable
/**
 * kernel function.
 * This function generates double precision floating point numbers in d_data.
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output. IEEE double precision format.
 * @param[in] size number of output data requested.
 */
__kernel void mtgp64_double01_kernel(__global ulong * d_status,
				     __global double * d_data,
				     int size)
{
    __local ulong status[MTGP64_LS];
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    int pos = mtgp64_pos;
    ulong r;
    ulong o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, gid, lid);

    // main loop
    for (int i = 0; i < size; i += MTGP64_LS) {
	r = para_rec(status[MTGP64_LS - MTGP64_N + lid],
		     status[MTGP64_LS - MTGP64_N + lid + 1],
		     status[MTGP64_LS - MTGP64_N + lid + pos]);
	status[lid] = r;
	o = temper_double(r, status[MTGP64_LS - MTGP64_N + lid + pos - 1]);
	d_data[size * gid + i + lid] = as_double(o) - 1.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(status[(4 * MTGP64_TN - MTGP64_N + lid) % MTGP64_LS],
		     status[(4 * MTGP64_TN - MTGP64_N + lid + 1) % MTGP64_LS],
		     status[(4 * MTGP64_TN - MTGP64_N + lid + pos)
			    % MTGP64_LS]);
	status[lid + MTGP64_TN] = r;
	o = temper_double(
	    r,
	    status[(4 * MTGP64_TN - MTGP64_N + lid + pos - 1) % MTGP64_LS]);
	d_data[size * gid + MTGP64_TN + i + lid] = as_double(o) - 1.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(status[2 * MTGP64_TN - MTGP64_N + lid],
		     status[2 * MTGP64_TN - MTGP64_N + lid + 1],
		     status[2 * MTGP64_TN - MTGP64_N + lid + pos]);
	status[lid + 2 * MTGP64_TN] = r;
	o = temper_double(r,
			  status[lid + pos - 1 + 2 * MTGP64_TN - MTGP64_N]);
	d_data[size * gid + 2 * MTGP64_TN + i + lid] = as_double(o) - 1.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write back status for next call
    status_write(d_status, status, gid, lid);
}
#endif

/* ================================ */
/* mtgp64 sample device function    */
/* ================================ */
/**
 * The function of the recursion formula calculation.
 *
 * @param[in] X1 the farthest part of state array.
 * @param[in] X2 the second farthest part of state array.
 * @param[in] Y a part of state array.
 * @return output
 */
static inline ulong para_rec(ulong X1, ulong X2, ulong Y)
{
    ulong X;
    ulong R;
    uint XH;
    uint XL;
    uint YH;
    uint YL;

    X = (X1 & mtgp64_mask) ^ X2;
    XH = (uint)(X >> 32);
    XL = (uint)(X & 0xffffffffU);
    YH = (uint)(Y >> 32);
    YL = (uint)(Y & 0xffffffffU);
    XH ^= XH << mtgp64_sh1;
    XL ^= XL << mtgp64_sh1;
    YH = XL ^ (YH >> mtgp64_sh2);
    YL = XH ^ (YL >> mtgp64_sh2);
    R = ((ulong)YH << 32) | YL;
    R ^= mtgp64_param_tbl[YL & 0x0f];
    return R;
}

/**
 * The tempering function.
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @return the tempered value.
 */
static inline ulong temper(ulong V, ulong T)
{
    ulong MAT;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = mtgp64_temper_tbl[T & 0x0f];
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

static inline ulong temper_double(ulong V, ulong T)
{
    ulong MAT;
    ulong r;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = mtgp64_double_temper_tbl[T & 0x0f];
    r = (V >> 12) ^ MAT;
    return r;
}

/**
 * Read the internal state vector from kernel I/O data, and
 * put them into local memory.
 *
 * @param[out] status in local memory.
 * @param[in] d_status kernel I/O data
 * @param[in] gid block id
 * @param[in] lid thread id
 */
static inline void status_read(__local ulong * status,
			       __global ulong * d_status,
			       int gid,
			       int lid)
{
    status[MTGP64_LS - MTGP64_N + lid]
	= d_status[gid * MTGP64_N + lid];
    if (lid < MTGP64_N - MTGP64_TN) {
	status[MTGP64_LS - MTGP64_N + MTGP64_TN + lid]
	    = d_status[gid * MTGP64_N + MTGP64_TN + lid];
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
static inline void status_write(__global ulong * d_status,
				__local ulong * status,
				int gid,
				int lid)
{
    d_status[gid * MTGP64_N + lid]
	= status[MTGP64_LS - MTGP64_N + lid];
    if (lid < MTGP64_N - MTGP64_TN) {
	d_status[gid * MTGP64_N + MTGP64_TN + lid]
	    = status[4 * MTGP64_TN - MTGP64_N + lid];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/**
 * This function represents a function used in the initialization
 * by mtgp64_init_by_array() and mtgp64_init_by_str().
 * @param[in] x 64-bit integer
 * @return 64-bit integer
 */
static inline ulong mtgp64_ini_func1(ulong x)
{
    return (x ^ (x >> 59)) * 2173292883993UL;
}

/**
 * This function represents a function used in the initialization
 * by mtgp64_init_by_array() and mtgp64_init_by_str().
 * @param[in] x 64-bit integer
 * @return 64-bit integer
 */
static inline ulong mtgp64_ini_func2(ulong x)
{
    return (x ^ (x >> 59)) * 58885565329898161UL;
}

/**
 * This function initializes the internal state array with a 64-bit
 * integer seed.
 * @param[in] seed a 64-bit integer used as the seed.
 */
static inline void mtgp64_init_state(__local ulong * status,
				     ulong seed)
{
    int i;
    ulong hidden_seed;
    ulong tmp;
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    hidden_seed = mtgp64_param_tbl[4] ^ (mtgp64_param_tbl[8] << 16);
    tmp = hidden_seed >> 32;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    tmp &= 0xff;
    tmp |= tmp << 8;
    tmp |= tmp << 16;
    tmp |= tmp << 32;
    //memset(mtgp64_status, tmp & 0xff, sizeof(ulong) * size);
    status[lid] = tmp;
    if ((local_size < MTGP64_N) && (lid < MTGP64_N - MTGP64_TN)) {
	status[MTGP64_TN + lid] = tmp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
	status[0] = seed;
	status[1] = hidden_seed;
	for (i = 1; i < MTGP64_N; i++) {
	    status[i] ^= i + 6364136223846793005UL
		* (status[i - 1] ^ (status[i - 1] >> 62));
	}
    }
}

/**
 * This function allocates and initializes the internal state array
 * with a 64-bit integer array. The allocated memory should be freed by
 * calling mtgp64_free(). \b para should be one of the elements in
 * the parameter table (mtgp64-param-ref.c).
 *
 * @param[in] seed_array a 64-bit integer array used as a seed.
 * @param[in] length length of the seed_array.
 */
static inline void mtgp64_init_by_array(__local ulong * status,
					__global ulong * seed_array,
					int length)
{
    int i, j, count;
    ulong r;
    int lag;
    int mid;
    int size = MTGP64_N;
    ulong hidden_seed;
    ulong tmp;
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

//    st = mtgp64_status;
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

    hidden_seed = mtgp64_param_tbl[4] ^ (mtgp64_param_tbl[8] << 16);
    tmp = hidden_seed >> 32;
    tmp += tmp >> 16;
    tmp += tmp >> 8;

    tmp &= 0xff;
    tmp |= tmp << 8;
    tmp |= tmp << 16;
    tmp |= tmp << 32;
    status[lid] = tmp;

    if ((local_size < MTGP64_N) && (lid < MTGP64_N - MTGP64_TN)) {
	status[MTGP64_TN + lid] = tmp;
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
    r = mtgp64_ini_func1(status[0]
			 ^ status[mid]
			 ^ status[size - 1]);
    status[mid] += r;
    r += length;
    status[(mid + lag) % size] += r;
    status[0] = r;
    i = 1;
    count--;
    for (i = 1, j = 0; (j < count) && (j < length); j++) {
	r = mtgp64_ini_func1(status[i]
			     ^ status[(i + mid) % size]
			     ^ status[(i + size - 1) % size]);
	status[(i + mid) % size] += r;
	r += seed_array[j] + i;
	status[(i + mid + lag) % size] += r;
	status[i] = r;
	i = (i + 1) % size;
    }
    for (; j < count; j++) {
	r = mtgp64_ini_func1(status[i]
			     ^ status[(i + mid) % size]
			     ^ status[(i + size - 1) % size]);
	status[(i + mid) % size] += r;
	r += i;
	status[(i + mid + lag) % size] += r;
	status[i] = r;
	i = (i + 1) % size;
    }
    for (j = 0; j < size; j++) {
	r = mtgp64_ini_func2(status[i]
			     + status[(i + mid) % size]
			     + status[(i + size - 1) % size]);
	status[(i + mid) % size] ^= r;
	r -= i;
	status[(i + mid + lag) % size] ^= r;
	status[i] = r;
	i = (i + 1) % size;
    }
    if (status[size - 1] == 0) {
	status[size - 1] = mtgp64_non_zero;
    }
}

/**
 * device function.
 * This function changes internal state of MTGP to jumped state.
 * threads per block should be MTGP64_N.
 *
 * @param[in,out] d_status kernel I/O data
 * @param[in] jump_poly jump polynomial
 */
static inline void mtgp64_jump(int gid,
			       int lid,
			       __local ulong * work,
			       __local ulong * status,
			       __global uint * jump_poly)
{
    int index = 0;
    ulong r;
    int pos = mtgp64_pos;
    uint bits;
    int i, j;
    const int local_size = get_local_size(0);

    work[lid] = 0;
    if ((local_size < MTGP64_N) && (lid < MTGP64_N - MTGP64_TN)) {
	work[MTGP64_TN + lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // jump
    for (i = 0; i < MTGP64_N; i++) {
	bits = jump_poly[i];
	for (j = 0; j < 32; j++) {
	    if ((bits & 1) != 0) {
		//add
		work[lid] ^= status[(lid + index) % MTGP64_N];
		if ((local_size < MTGP64_N) && (lid < MTGP64_N - MTGP64_TN)) {
		    work[MTGP64_TN + lid]
			^= status[(MTGP64_TN + lid + index) % MTGP64_N];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	    }
	    //next_state
	    if (lid == 0) {
		r = para_rec(status[(lid + index) % MTGP64_N],
			     status[(lid + index + 1) % MTGP64_N],
			     status[(lid + index + pos) % MTGP64_N]);
		status[(lid + index) % MTGP64_N] = r;
	    }
	    barrier(CLK_LOCAL_MEM_FENCE);
	    index = (index + 1) % MTGP64_N;
	    bits = bits >> 1;
	}
    }
}

/**
 * device function.
 * This function changes internal state of MTGP to jumped state
 * using jump table.
 *
 * jump_table contains n * N step jump data.
 * jump_table[0]      N step jump
 * jump_table[1]  4 * N step jump
 * jump_table[2] 16 * N step jump
 * jump_table[3] 64 * N step jump
 * @param[in] gid group id
 * @param[in] lid local id
 * @param[in] jump_table constant table for jump
 * @param[in, out] work working area
 * @param[in, out] status mtgp status
 */
static inline void mtgp64_table_jump(int gid,
				     int lid,
				     __global uint * jump_table,
				     __local ulong * work,
				     __local ulong * status)
{
    int mask = 1;
    int i;
    int idx;
    const int local_size = get_local_size(0);

    for (i = 0; (i < 64) && (mask <= gid); i++) {
	idx = i / 2;
	if ((gid & mask) != 0) {
	    if (i % 2 == 0) {
		mtgp64_jump(gid, lid, work, status,
			    &jump_table[idx * MTGP64_N]);
		barrier(CLK_LOCAL_MEM_FENCE);
		status[lid] = work[lid];
		if ((local_size < MTGP64_N) && (lid < MTGP64_N - MTGP64_TN)) {
		    status[MTGP64_TN + lid] = work[MTGP64_TN + lid];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	    } else {
		mtgp64_jump(gid, lid, work, status,
			    &jump_table[idx * MTGP64_N]);
		barrier(CLK_LOCAL_MEM_FENCE);
		mtgp64_jump(gid, lid, status, work,
			    &jump_table[idx * MTGP64_N]);
		barrier(CLK_LOCAL_MEM_FENCE);
	    }
	}
	mask = mask << 1;
    }
}

