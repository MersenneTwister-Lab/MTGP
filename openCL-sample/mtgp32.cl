/*
 * @file mtgp32.cl
 *
 * @brief MTGP Sample Program for openCL 1.1
 * 1 parameter for 1 generator
 */

/*
 * Generator Parameters.
 */

#define MTGP32_MEXP 11213
#define MTGP32_N 351
#define MTGP32_FLOOR_2P 256
#define MTGP32_CEIL_2P 512
#define MTGP32_TN MTGP32_FLOOR_2P
#define MTGP32_LS (MTGP32_TN * 3)
#define MTGP32_TS 16

/* =========================
   declarations
   ========================= */
struct MTGP32_T {
    __local uint * status;
    __constant uint * param_tbl;
    __constant uint * temper_tbl;
    __constant uint * single_temper_tbl;
    uint pos;
    uint sh1;
    uint sh2;
};
typedef struct MTGP32_T mtgp32_t;

__constant uint mtgp32_mask = 0xfff80000;
__constant uint mtgp32_non_zero = 0x4d544750;

static inline uint para_rec(mtgp32_t * mtgp, uint X1, uint X2, uint Y);
static inline uint temper(mtgp32_t * mtgp, uint V, uint T);
static inline uint temper_single(mtgp32_t * mtgp, uint V, uint T);
static inline void status_read(__local uint  * status,
			       __global uint * d_status,
			       int gid,
			       int lid);
static inline void status_write(__global uint * d_status,
				__local uint * status,
				int gid,
				int lid);
static inline uint mtgp32_ini_func1(uint x);
static inline uint mtgp32_ini_func2(uint x);
static inline void mtgp32_init_state(mtgp32_t * mtgp, uint seed);
static inline void mtgp32_init_by_array(mtgp32_t * mtgp,
					__global uint *seed_array,
					int length);
/* ================================ */
/* mtgp32 sample kernel code        */
/* ================================ */
/**
 * This function sets up initial state by seed.
 * kernel function.
 *
 * @param[in] param_tbl recursion parameters
 * @param[in] temper_tbl tempering parameters
 * @param[in] single_temper_tbl tempering parameters for float
 * @param[in] pos_tbl pic-up positions
 * @param[in] sh1_tbl shift parameters
 * @param[in] sh2_tbl shift parameters
 * @param[out] d_status kernel I/O data
 * @param[in] seed initializing seed
 */
__kernel void mtgp32_init_seed_kernel(
    __constant uint * param_tbl,
    __constant uint * temper_tbl,
    __constant uint * single_temper_tbl,
    __constant uint * pos_tbl,
    __constant uint * sh1_tbl,
    __constant uint * sh2_tbl,
    __global uint * d_status,
    uint seed)
{
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    __local uint status[MTGP32_N];
    mtgp32_t mtgp;
    mtgp.status = status;
    mtgp.param_tbl = &param_tbl[MTGP32_TS * gid];
    mtgp.temper_tbl = &temper_tbl[MTGP32_TS * gid];
    mtgp.single_temper_tbl = &single_temper_tbl[MTGP32_TS * gid];
    mtgp.pos = pos_tbl[gid];
    mtgp.sh1 = sh1_tbl[gid];
    mtgp.sh2 = sh2_tbl[gid];

    // initialize
    mtgp32_init_state(&mtgp, seed + gid);
    barrier(CLK_LOCAL_MEM_FENCE);

    d_status[gid * MTGP32_N + lid] = status[lid];
    if ((local_size < MTGP32_N) && (lid < MTGP32_N - MTGP32_TN)) {
	d_status[gid * MTGP32_N + MTGP32_TN + lid] = status[MTGP32_TN + lid];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/**
 * This function sets up initial state by seed.
 * kernel function.
 *
 * @param[in] param_tbl recursion parameters
 * @param[in] temper_tbl tempering parameters
 * @param[in] single_temper_tbl tempering parameters for float
 * @param[in] pos_tbl pic-up positions
 * @param[in] sh1_tbl shift parameters
 * @param[in] sh2_tbl shift parameters
 * @param[out] d_status kernel I/O data
 * @param[in] seed_array initializing seeds
 * @param[in] length length of seed_array
 */
__kernel void mtgp32_init_array_kernel(
    __constant uint * param_tbl,
    __constant uint * temper_tbl,
    __constant uint * single_temper_tbl,
    __constant uint * pos_tbl,
    __constant uint * sh1_tbl,
    __constant uint * sh2_tbl,
    __global uint * d_status,
    __global uint * seed_array,
    int length)
{
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    __local uint status[MTGP32_N];
    mtgp32_t mtgp;
    mtgp.status = status;
    mtgp.param_tbl = &param_tbl[MTGP32_TS * gid];
    mtgp.temper_tbl = &temper_tbl[MTGP32_TS * gid];
    mtgp.single_temper_tbl = &single_temper_tbl[MTGP32_TS * gid];
    mtgp.pos = pos_tbl[gid];
    mtgp.sh1 = sh1_tbl[gid];
    mtgp.sh2 = sh2_tbl[gid];

    // initialize
    mtgp32_init_by_array(&mtgp, seed_array, length);
    barrier(CLK_LOCAL_MEM_FENCE);

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
 * @param[in] param_tbl recursion parameters
 * @param[in] temper_tbl tempering parameters
 * @param[in] single_temper_tbl tempering parameters for float
 * @param[in] pos_tbl pic-up positions
 * @param[in] sh1_tbl shift parameters
 * @param[in] sh2_tbl shift parameters
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output
 * @param[in] size number of output data requested.
 */
__kernel void mtgp32_uint32_kernel(
    __constant uint * param_tbl,
    __constant uint * temper_tbl,
    __constant uint * single_temper_tbl,
    __constant uint * pos_tbl,
    __constant uint * sh1_tbl,
    __constant uint * sh2_tbl,
    __global uint * d_status,
    __global uint * d_data,
    int size)
{
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    __local uint status[MTGP32_LS];
    mtgp32_t mtgp;
    uint r;
    uint o;

    mtgp.status = status;
    mtgp.param_tbl = &param_tbl[MTGP32_TS * gid];
    mtgp.temper_tbl = &temper_tbl[MTGP32_TS * gid];
    mtgp.single_temper_tbl = &single_temper_tbl[MTGP32_TS * gid];
    mtgp.pos = pos_tbl[gid];
    mtgp.sh1 = sh1_tbl[gid];
    mtgp.sh2 = sh2_tbl[gid];

    int pos = mtgp.pos;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, gid, lid);

    // main loop
    for (int i = 0; i < size; i += MTGP32_LS) {
	r = para_rec(&mtgp,
		     status[MTGP32_LS - MTGP32_N + lid],
		     status[MTGP32_LS - MTGP32_N + lid + 1],
		     status[MTGP32_LS - MTGP32_N + lid + pos]);
	status[lid] = r;
	o = temper(&mtgp, r, status[MTGP32_LS - MTGP32_N + lid + pos - 1]);
	d_data[size * gid + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(&mtgp,
		     status[(4 * MTGP32_TN - MTGP32_N + lid) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + lid + 1) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + lid + pos)
			    % MTGP32_LS]);
	status[lid + MTGP32_TN] = r;
	o = temper(&mtgp,
		   r,
		   status[(4 * MTGP32_TN - MTGP32_N + lid + pos - 1)
			  % MTGP32_LS]);
	d_data[size * gid + MTGP32_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(&mtgp,
		     status[2 * MTGP32_TN - MTGP32_N + lid],
		     status[2 * MTGP32_TN - MTGP32_N + lid + 1],
		     status[2 * MTGP32_TN - MTGP32_N + lid + pos]);
	status[lid + 2 * MTGP32_TN] = r;
	o = temper(&mtgp, r, status[lid + pos - 1 + 2 * MTGP32_TN - MTGP32_N]);
	d_data[size * gid + 2 * MTGP32_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write back status for next call
    status_write(d_status, status, gid, lid);
}

/**
 * This kernel function generates single precision floating point numbers
 * in the range [1, 2) in d_data.
 *
 * @param[in] param_tbl recursion parameters
 * @param[in] temper_tbl tempering parameters
 * @param[in] single_temper_tbl tempering parameters for float
 * @param[in] pos_tbl pic-up positions
 * @param[in] sh1_tbl shift parameters
 * @param[in] sh2_tbl shift parameters
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output. IEEE single precision format.
 * @param[in] size number of output data requested.
 */
__kernel void mtgp32_single12_kernel(
    __constant uint * param_tbl,
    __constant uint * temper_tbl,
    __constant uint * single_temper_tbl,
    __constant uint * pos_tbl,
    __constant uint * sh1_tbl,
    __constant uint * sh2_tbl,
    __global uint * d_status,
    __global uint* d_data,
    int size)
{
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    __local uint status[MTGP32_LS];
    mtgp32_t mtgp;
    uint r;
    uint o;

    mtgp.status = status;
    mtgp.param_tbl = &param_tbl[MTGP32_TS * gid];
    mtgp.temper_tbl = &temper_tbl[MTGP32_TS * gid];
    mtgp.single_temper_tbl = &single_temper_tbl[MTGP32_TS * gid];
    mtgp.pos = pos_tbl[gid];
    mtgp.sh1 = sh1_tbl[gid];
    mtgp.sh2 = sh2_tbl[gid];

    int pos = mtgp.pos;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, gid, lid);

    // main loop
    for (int i = 0; i < size; i += MTGP32_LS) {
	r = para_rec(&mtgp,
		     status[MTGP32_LS - MTGP32_N + lid],
		     status[MTGP32_LS - MTGP32_N + lid + 1],
		     status[MTGP32_LS - MTGP32_N + lid + pos]);
	status[lid] = r;
	o = temper_single(&mtgp,
			  r,
			  status[MTGP32_LS - MTGP32_N + lid + pos - 1]);
	d_data[size * gid + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(&mtgp,
		     status[(4 * MTGP32_TN - MTGP32_N + lid) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + lid + 1) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + lid + pos)
			    % MTGP32_LS]);
	status[lid + MTGP32_TN] = r;
	o = temper_single(
	    &mtgp,
	    r,
	    status[(4 * MTGP32_TN - MTGP32_N + lid + pos - 1) % MTGP32_LS]);
	d_data[size * gid + MTGP32_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(&mtgp,
		     status[2 * MTGP32_TN - MTGP32_N + lid],
		     status[2 * MTGP32_TN - MTGP32_N + lid + 1],
		     status[2 * MTGP32_TN - MTGP32_N + lid + pos]);
	status[lid + 2 * MTGP32_TN] = r;
	o = temper_single(&mtgp,
			  r,
			  status[lid + pos - 1 + 2 * MTGP32_TN - MTGP32_N]);
	d_data[size * gid + 2 * MTGP32_TN + i + lid] = o;
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write back status for next call
    status_write(d_status, status, gid, lid);
}

/**
 * This kernel function generates single precision floating point numbers
 * in the range [0, 1) in d_data.
 *
 * @param[in] param_tbl recursion parameters
 * @param[in] temper_tbl tempering parameters
 * @param[in] single_temper_tbl tempering parameters for float
 * @param[in] pos_tbl pic-up positions
 * @param[in] sh1_tbl shift parameters
 * @param[in] sh2_tbl shift parameters
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output. IEEE single precision format.
 * @param[in] size number of output data requested.
 */
__kernel void mtgp32_single01_kernel(
    __constant uint * param_tbl,
    __constant uint * temper_tbl,
    __constant uint * single_temper_tbl,
    __constant uint * pos_tbl,
    __constant uint * sh1_tbl,
    __constant uint * sh2_tbl,
    __global uint * d_status,
    __global float* d_data,
    int size)
{
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    __local uint status[MTGP32_LS];
    mtgp32_t mtgp;
    uint r;
    uint o;

    mtgp.status = status;
    mtgp.param_tbl = &param_tbl[MTGP32_TS * gid];
    mtgp.temper_tbl = &temper_tbl[MTGP32_TS * gid];
    mtgp.single_temper_tbl = &single_temper_tbl[MTGP32_TS * gid];
    mtgp.pos = pos_tbl[gid];
    mtgp.sh1 = sh1_tbl[gid];
    mtgp.sh2 = sh2_tbl[gid];

    int pos = mtgp.pos;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, gid, lid);

    // main loop
    for (int i = 0; i < size; i += MTGP32_LS) {
	r = para_rec(&mtgp,
		     status[MTGP32_LS - MTGP32_N + lid],
		     status[MTGP32_LS - MTGP32_N + lid + 1],
		     status[MTGP32_LS - MTGP32_N + lid + pos]);
	status[lid] = r;
	o = temper_single(&mtgp,
			  r,
			  status[MTGP32_LS - MTGP32_N + lid + pos - 1]);
	d_data[size * gid + i + lid] = as_float(o) - 1.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(&mtgp,
		     status[(4 * MTGP32_TN - MTGP32_N + lid) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + lid + 1) % MTGP32_LS],
		     status[(4 * MTGP32_TN - MTGP32_N + lid + pos)
			    % MTGP32_LS]);
	status[lid + MTGP32_TN] = r;
	o = temper_single(
	    &mtgp,
	    r,
	    status[(4 * MTGP32_TN - MTGP32_N + lid + pos - 1) % MTGP32_LS]);
	d_data[size * gid + MTGP32_TN + i + lid] = as_float(o) - 1.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	r = para_rec(&mtgp,
		     status[2 * MTGP32_TN - MTGP32_N + lid],
		     status[2 * MTGP32_TN - MTGP32_N + lid + 1],
		     status[2 * MTGP32_TN - MTGP32_N + lid + pos]);
	status[lid + 2 * MTGP32_TN] = r;
	o = temper_single(&mtgp,
			  r,
			  status[lid + pos - 1 + 2 * MTGP32_TN - MTGP32_N]);
	d_data[size * gid + 2 * MTGP32_TN + i + lid] = as_float(o) - 1.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write back status for next call
    status_write(d_status, status, gid, lid);
}

/* ================================ */
/* mtgp32 sample device function    */
/* ================================ */
/**
 * The function of the recursion formula calculation.
 *
 * @param[in] mtgp mtgp32 structure
 * @param[in] X1 the farthest part of state array.
 * @param[in] X2 the second farthest part of state array.
 * @param[in] Y a part of state array.
 * @return output
 */
static inline uint para_rec(mtgp32_t * mtgp, uint X1, uint X2, uint Y)
{
    uint X = (X1 & mtgp32_mask) ^ X2;
    uint MAT;

    X ^= X << mtgp->sh1;
    Y = X ^ (Y >> mtgp->sh2);
    MAT = mtgp->param_tbl[Y & 0x0f];
    return Y ^ MAT;
}

/**
 * The tempering function.
 *
 * @param[in] mtgp mtgp32 structure
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @return the tempered value.
 */
static inline uint temper(mtgp32_t * mtgp, uint V, uint T)
{
    uint MAT;

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
 * @param[in] mtgp mtgp32 structure
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @return the tempered and converted value.
 */
static inline uint temper_single(mtgp32_t * mtgp, uint V, uint T)
{
    uint MAT;
    uint r;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = mtgp->single_temper_tbl[T & 0x0f];
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
static inline void status_read(__local uint  * status,
			       __global uint * d_status,
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
static inline void status_write(__global uint * d_status,
				__local uint * status,
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
 * @param[in] mtgp mtgp32 structure
 * @param[in] seed a 32-bit integer used as the seed.
 */
static inline void mtgp32_init_state(mtgp32_t * mtgp, uint seed)
{
    int i;
    uint hidden_seed;
    uint tmp;
    __local uint * status = mtgp->status;
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    hidden_seed = mtgp->param_tbl[4] ^ (mtgp->param_tbl[8] << 16);
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
    mem_fence(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
	status[0] = seed;
	for (i = 1; i < MTGP32_N; i++) {
	    status[i] = hidden_seed ^ (i + 1812433253U * (status[i - 1]
							  ^ (status[i - 1] >> 30)));
	    hidden_seed = tmp;
	}
    }
}

/**
 * This function allocates and initializes the internal state array
 * with a 32-bit integer array. The allocated memory should be freed by
 * calling mtgp32_free(). \b para should be one of the elements in
 * the parameter table (mtgp32-param-ref.c).
 *
 * @param[in] mtgp mtgp32 structure
 * @param[in] seed_array a 32-bit integer array used as a seed.
 * @param[in] length length of the seed_array.
 */
static inline void mtgp32_init_by_array(mtgp32_t * mtgp,
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
    __local uint * status = mtgp->status;
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

    hidden_seed = mtgp->param_tbl[4] ^ (mtgp->param_tbl[8] << 16);
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
    mem_fence(CLK_LOCAL_MEM_FENCE);
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

