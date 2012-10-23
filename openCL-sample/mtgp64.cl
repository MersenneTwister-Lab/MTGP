/*
 * @file mtgp64.cl
 *
 * @brief MTGP Sample Program for openCL 1.1
 * 1 parameter for 1 generator
 * MEXP = 11213
 */

/*
 * Generator Parameters.
 */

#define MTGP64_MEXP 11213
#define MTGP64_N 176
#define MTGP64_FLOOR_2P 128
#define MTGP64_CEIL_2P 256
#define MTGP64_TN MTGP64_FLOOR_2P
#define MTGP64_LS (MTGP64_TN * 3)
#define MTGP64_TS 16

/* =========================
   declarations
   ========================= */
struct MTGP64_T {
    __local ulong * status;
    __constant ulong * param_tbl;
    __constant ulong * temper_tbl;
    __constant ulong * double_temper_tbl;
    uint pos;
    uint sh1;
    uint sh2;
};
typedef struct MTGP64_T mtgp64_t;

__constant ulong mtgp64_mask = 0xfff8000000000000UL;
__constant ulong mtgp64_non_zero = 0x4d544750;

static inline ulong para_rec(mtgp64_t * mtgp, ulong X1, ulong X2, ulong Y);
static inline ulong temper(mtgp64_t * mtgp, ulong V, ulong T);
static inline ulong temper_double(mtgp64_t * mtgp, ulong V, ulong T);
static inline void status_read(__local ulong  * status,
			       __global ulong * d_status,
			       int gid,
			       int lid);
static inline void status_write(__global ulong * d_status,
				__local ulong * status,
				int gid,
				int lid);
static inline ulong mtgp64_ini_func1(ulong x);
static inline ulong mtgp64_ini_func2(ulong x);
static inline void mtgp64_init_state(mtgp64_t * mtgp, ulong seed);
static inline void mtgp64_init_by_array(mtgp64_t * mtgp,
					__global ulong *seed_array,
					int length);
/* ================================ */
/* mtgp64 sample kernel code        */
/* ================================ */
/**
 * This function sets up initial state by seed.
 * kernel function.
 *
 * @param[in] param_tbl recursion parameters
 * @param[in] temper_tbl tempering parameters
 * @param[in] double_temper_tbl tempering parameters for double
 * @param[in] pos_tbl pic-up positions
 * @param[in] sh1_tbl shift parameters
 * @param[in] sh2_tbl shift parameters
 * @param[out] d_status kernel I/O data
 * @param[in] seed initializing seed
 */
__kernel void mtgp64_init_seed_kernel(
    __constant ulong * param_tbl,
    __constant ulong * temper_tbl,
    __constant ulong * double_temper_tbl,
    __constant uint * pos_tbl,
    __constant uint * sh1_tbl,
    __constant uint * sh2_tbl,
    __global ulong * d_status,
    ulong seed)
{
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    __local ulong status[MTGP64_N];
    mtgp64_t mtgp;
    mtgp.status = status;
    mtgp.param_tbl = &param_tbl[MTGP64_TS * gid];
    mtgp.temper_tbl = &temper_tbl[MTGP64_TS * gid];
    mtgp.double_temper_tbl = &double_temper_tbl[MTGP64_TS * gid];
    mtgp.pos = pos_tbl[gid];
    mtgp.sh1 = sh1_tbl[gid];
    mtgp.sh2 = sh2_tbl[gid];

    // initialize
    mtgp64_init_state(&mtgp, seed + gid);
    barrier(CLK_LOCAL_MEM_FENCE);

    d_status[gid * MTGP64_N + lid] = status[lid];
    if ((local_size < MTGP64_N) && (lid < MTGP64_N - MTGP64_TN)) {
	d_status[gid * MTGP64_N + MTGP64_TN + lid] = status[MTGP64_TN + lid];
    }
}

/**
 * This function sets up initial state by seed.
 * kernel function.
 *
 * @param[in] param_tbl recursion parameters
 * @param[in] temper_tbl tempering parameters
 * @param[in] double_temper_tbl tempering parameters for double
 * @param[in] pos_tbl pic-up positions
 * @param[in] sh1_tbl shift parameters
 * @param[in] sh2_tbl shift parameters
 * @param[out] d_status kernel I/O data
 * @param[in] seed_array initializing seeds
 * @param[in] length length of seed_array
 */
__kernel void mtgp64_init_array_kernel(
    __constant ulong * param_tbl,
    __constant ulong * temper_tbl,
    __constant ulong * double_temper_tbl,
    __constant uint * pos_tbl,
    __constant uint * sh1_tbl,
    __constant uint * sh2_tbl,
    __global ulong * d_status,
    __global ulong * seed_array,
    int length)
{
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    __local ulong status[MTGP64_N];
    mtgp64_t mtgp;
    mtgp.status = status;
    mtgp.param_tbl = &param_tbl[MTGP64_TS * gid];
    mtgp.temper_tbl = &temper_tbl[MTGP64_TS * gid];
    mtgp.double_temper_tbl = &double_temper_tbl[MTGP64_TS * gid];
    mtgp.pos = pos_tbl[gid];
    mtgp.sh1 = sh1_tbl[gid];
    mtgp.sh2 = sh2_tbl[gid];

    // initialize
    mtgp64_init_by_array(&mtgp, seed_array, length);
    barrier(CLK_LOCAL_MEM_FENCE);

    d_status[gid * MTGP64_N + lid] = status[lid];
    if ((local_size < MTGP64_N) && (lid < MTGP64_N - MTGP64_TN)) {
	d_status[gid * MTGP64_N + MTGP64_TN + lid] = status[MTGP64_TN + lid];
    }
}

/**
 * kernel function.
 * This function generates 64-bit unsigned integers in d_data
 *
 * @param[in] param_tbl recursion parameters
 * @param[in] temper_tbl tempering parameters
 * @param[in] double_temper_tbl tempering parameters for double
 * @param[in] pos_tbl pic-up positions
 * @param[in] sh1_tbl shift parameters
 * @param[in] sh2_tbl shift parameters
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output
 * @param[in] size number of output data requested.
 */
__kernel void mtgp64_uint64_kernel(
    __constant ulong * param_tbl,
    __constant ulong * temper_tbl,
    __constant ulong * double_temper_tbl,
    __constant uint * pos_tbl,
    __constant uint * sh1_tbl,
    __constant uint * sh2_tbl,
    __global ulong * d_status,
    __global ulong * d_data,
    int size)
{
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    __local ulong status[MTGP64_LS];
    mtgp64_t mtgp;
    ulong r;
    ulong o;

    mtgp.status = status;
    mtgp.param_tbl = &param_tbl[MTGP64_TS * gid];
    mtgp.temper_tbl = &temper_tbl[MTGP64_TS * gid];
    mtgp.double_temper_tbl = &double_temper_tbl[MTGP64_TS * gid];
    mtgp.pos = pos_tbl[gid];
    mtgp.sh1 = sh1_tbl[gid];
    mtgp.sh2 = sh2_tbl[gid];

    int pos = mtgp.pos;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, gid, lid);

    // main loop
    for (int i = 0; i < size; i += MTGP64_LS) {
	r = para_rec(&mtgp,
		     status[MTGP64_LS - MTGP64_N + lid],
		     status[MTGP64_LS - MTGP64_N + lid + 1],
		     status[MTGP64_LS - MTGP64_N + lid + pos]);
	status[lid] = r;
	o = temper(&mtgp, r, status[MTGP64_LS - MTGP64_N + lid + pos - 1]);
	d_data[size * gid + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(&mtgp,
		     status[(4 * MTGP64_TN - MTGP64_N + lid) % MTGP64_LS],
		     status[(4 * MTGP64_TN - MTGP64_N + lid + 1) % MTGP64_LS],
		     status[(4 * MTGP64_TN - MTGP64_N + lid + pos)
			    % MTGP64_LS]);
	status[lid + MTGP64_TN] = r;
	o = temper(&mtgp,
		   r,
		   status[(4 * MTGP64_TN - MTGP64_N + lid + pos - 1)
			  % MTGP64_LS]);
	d_data[size * gid + MTGP64_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(&mtgp,
		     status[2 * MTGP64_TN - MTGP64_N + lid],
		     status[2 * MTGP64_TN - MTGP64_N + lid + 1],
		     status[2 * MTGP64_TN - MTGP64_N + lid + pos]);
	status[lid + 2 * MTGP64_TN] = r;
	o = temper(&mtgp, r, status[lid + pos - 1 + 2 * MTGP64_TN - MTGP64_N]);
	d_data[size * gid + 2 * MTGP64_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write back status for next call
    status_write(d_status, status, gid, lid);
}

/**
 * This kernel function generates double precision floating point numbers
 * in the range [1, 2) in d_data.
 *
 * @param[in] param_tbl recursion parameters
 * @param[in] temper_tbl tempering parameters
 * @param[in] double_temper_tbl tempering parameters for double
 * @param[in] pos_tbl pic-up positions
 * @param[in] sh1_tbl shift parameters
 * @param[in] sh2_tbl shift parameters
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output. IEEE double precision format.
 * @param[in] size number of output data requested.
 */
__kernel void mtgp64_double12_kernel(
    __constant ulong * param_tbl,
    __constant ulong * temper_tbl,
    __constant ulong * double_temper_tbl,
    __constant uint * pos_tbl,
    __constant uint * sh1_tbl,
    __constant uint * sh2_tbl,
    __global ulong * d_status,
    __global ulong * d_data,
    int size)
{
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    __local ulong status[MTGP64_LS];
    mtgp64_t mtgp;
    ulong r;
    ulong o;

    mtgp.status = status;
    mtgp.param_tbl = &param_tbl[MTGP64_TS * gid];
    mtgp.temper_tbl = &temper_tbl[MTGP64_TS * gid];
    mtgp.double_temper_tbl = &double_temper_tbl[MTGP64_TS * gid];
    mtgp.pos = pos_tbl[gid];
    mtgp.sh1 = sh1_tbl[gid];
    mtgp.sh2 = sh2_tbl[gid];

    int pos = mtgp.pos;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, gid, lid);

    // main loop
    for (int i = 0; i < size; i += MTGP64_LS) {
	r = para_rec(&mtgp,
		     status[MTGP64_LS - MTGP64_N + lid],
		     status[MTGP64_LS - MTGP64_N + lid + 1],
		     status[MTGP64_LS - MTGP64_N + lid + pos]);
	status[lid] = r;
	o = temper_double(&mtgp,
			  r,
			  status[MTGP64_LS - MTGP64_N + lid + pos - 1]);
	d_data[size * gid + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(&mtgp,
		     status[(4 * MTGP64_TN - MTGP64_N + lid) % MTGP64_LS],
		     status[(4 * MTGP64_TN - MTGP64_N + lid + 1) % MTGP64_LS],
		     status[(4 * MTGP64_TN - MTGP64_N + lid + pos)
			    % MTGP64_LS]);
	status[lid + MTGP64_TN] = r;
	o = temper_double(
	    &mtgp,
	    r,
	    status[(4 * MTGP64_TN - MTGP64_N + lid + pos - 1) % MTGP64_LS]);
	d_data[size * gid + MTGP64_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(&mtgp,
		     status[2 * MTGP64_TN - MTGP64_N + lid],
		     status[2 * MTGP64_TN - MTGP64_N + lid + 1],
		     status[2 * MTGP64_TN - MTGP64_N + lid + pos]);
	status[lid + 2 * MTGP64_TN] = r;
	o = temper_double(&mtgp,
			  r,
			  status[lid + pos - 1 + 2 * MTGP64_TN - MTGP64_N]);
	d_data[size * gid + 2 * MTGP64_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write back status for next call
    status_write(d_status, status, gid, lid);
}

#if defined(HAVE_DOUBLE)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
/**
 * This kernel function generates double precision floating point numbers
 * in the range [0, 1) in d_data.
 *
 * @param[in] param_tbl recursion parameters
 * @param[in] temper_tbl tempering parameters
 * @param[in] double_temper_tbl tempering parameters for double
 * @param[in] pos_tbl pic-up positions
 * @param[in] sh1_tbl shift parameters
 * @param[in] sh2_tbl shift parameters
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output. IEEE double precision format.
 * @param[in] size number of output data requested.
 */
__kernel void mtgp64_double01_kernel(
    __constant ulong * param_tbl,
    __constant ulong * temper_tbl,
    __constant ulong * double_temper_tbl,
    __constant uint * pos_tbl,
    __constant uint * sh1_tbl,
    __constant uint * sh2_tbl,
    __global ulong * d_status,
    __global double * d_data,
    int size)
{
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    __local ulong status[MTGP64_LS];
    mtgp64_t mtgp;
    ulong r;
    ulong o;

    mtgp.status = status;
    mtgp.param_tbl = &param_tbl[MTGP64_TS * gid];
    mtgp.temper_tbl = &temper_tbl[MTGP64_TS * gid];
    mtgp.double_temper_tbl = &double_temper_tbl[MTGP64_TS * gid];
    mtgp.pos = pos_tbl[gid];
    mtgp.sh1 = sh1_tbl[gid];
    mtgp.sh2 = sh2_tbl[gid];

    int pos = mtgp.pos;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, gid, lid);

    // main loop
    for (int i = 0; i < size; i += MTGP64_LS) {
	r = para_rec(&mtgp,
		     status[MTGP64_LS - MTGP64_N + lid],
		     status[MTGP64_LS - MTGP64_N + lid + 1],
		     status[MTGP64_LS - MTGP64_N + lid + pos]);
	status[lid] = r;
	o = temper_double(&mtgp,
			  r,
			  status[MTGP64_LS - MTGP64_N + lid + pos - 1]);
	d_data[size * gid + i + lid] = as_double(o) - 1.0;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(&mtgp,
		     status[(4 * MTGP64_TN - MTGP64_N + lid) % MTGP64_LS],
		     status[(4 * MTGP64_TN - MTGP64_N + lid + 1) % MTGP64_LS],
		     status[(4 * MTGP64_TN - MTGP64_N + lid + pos)
			    % MTGP64_LS]);
	status[lid + MTGP64_TN] = r;
	o = temper_double(
	    &mtgp,
	    r,
	    status[(4 * MTGP64_TN - MTGP64_N + lid + pos - 1) % MTGP64_LS]);
	d_data[size * gid + MTGP64_TN + i + lid] = as_double(o) - 1.0;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(&mtgp,
		     status[2 * MTGP64_TN - MTGP64_N + lid],
		     status[2 * MTGP64_TN - MTGP64_N + lid + 1],
		     status[2 * MTGP64_TN - MTGP64_N + lid + pos]);
	status[lid + 2 * MTGP64_TN] = r;
	o = temper_double(&mtgp,
			  r,
			  status[lid + pos - 1 + 2 * MTGP64_TN - MTGP64_N]);
	d_data[size * gid + 2 * MTGP64_TN + i + lid] = as_double(o) - 1.0;
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
 * @param[in] mtgp mtgp64 structure
 * @param[in] X1 the farthest part of state array.
 * @param[in] X2 the second farthest part of state array.
 * @param[in] Y a part of state array.
 * @return output
 */
static inline ulong para_rec(mtgp64_t * mtgp, ulong X1, ulong X2, ulong Y)
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
    XH ^= XH << mtgp->sh1;
    XL ^= XL << mtgp->sh1;
    YH = XL ^ (YH >> mtgp->sh2);
    YL = XH ^ (YL >> mtgp->sh2);
    R = ((ulong)YH << 32) | YL;
    R ^= mtgp->param_tbl[YL & 0x0f];
    return R;
}

/**
 * The tempering function.
 *
 * @param[in] mtgp mtgp64 structure
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @return the tempered value.
 */
static inline ulong temper(mtgp64_t * mtgp, ulong V, ulong T)
{
    ulong MAT;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = mtgp->temper_tbl[T & 0x0f];
    return V ^ MAT;
}

/**
 * The tempering and converting function.
 * By using the preset-ted table, converting to IEEE format
 * and tempering are done simultaneously.
 *
 * @param[in] mtgp mtgp64 structure
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @return the tempered and converted value.
 */
static inline ulong temper_double(mtgp64_t * mtgp, ulong V, ulong T)
{
    ulong MAT;
    ulong R;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = mtgp->double_temper_tbl[T & 0x0f];
    R = (V >> 12) ^ MAT;
    return R;
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
static inline void status_read(__local ulong  * status,
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
 * @param[in] mtgp mtgp64 structure
 * @param[in] seed a 64-bit integer used as the seed.
 */
static inline void mtgp64_init_state(mtgp64_t * mtgp, ulong seed)
{
    int i;
    ulong hidden_seed;
    ulong tmp;
    __local ulong * status = mtgp->status;
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    hidden_seed = mtgp->param_tbl[4] ^ (mtgp->param_tbl[8] << 16);
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

    if (lid == 0) {
	status[0] = seed;
	for (i = 1; i < MTGP64_N; i++) {
	    status[i] = hidden_seed
		^ (i + 6364136223846793005UL
		   * (status[i - 1] ^ (status[i - 1] >> 62)));
	    hidden_seed = tmp;
	}
    }
}

/**
 * This function allocates and initializes the internal state array
 * with a 64-bit integer array. The allocated memory should be freed by
 * calling mtgp64_free(). \b para should be one of the elements in
 * the parameter table (mtgp64-param-ref.c).
 *
 * @param[in] mtgp mtgp64 structure
 * @param[in] seed_array a 64-bit integer array used as a seed.
 * @param[in] length length of the seed_array.
 */
static inline void mtgp64_init_by_array(mtgp64_t * mtgp,
					__global ulong *seed_array,
					int length)
{
    int i, j, count;
    ulong r;
    int lag;
    int mid;
    int size = MTGP64_N;
    ulong hidden_seed;
    ulong tmp;
    __local ulong * status = mtgp->status;
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

    hidden_seed = mtgp->param_tbl[4] ^ (mtgp->param_tbl[8] << 16);
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

