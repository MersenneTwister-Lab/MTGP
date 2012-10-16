/**
 * Sample host program to generate a sequence
 * using jump and parallel generation.
 *
 * 1. make initial state from a seed
 * 2. calculate initial position for each work group.
 *    (This step is time comsuming)
 * 3. Loop:
 *   3.1 generate sub-sequences parallel
 *   3.2 jump for next loop
 */

#include "opencl_tools.hpp"
#include <cstddef>
#include <cfloat>
#include <ctime>
#include <NTL/GF2X.h>
#include <NTL/ZZ.h>

typedef uint32_t uint;
#include "mtgp64-calc-poly.hpp"
#include "mtgp-calc-jump.hpp"
#include "mtgp64-fast-jump.h"
#include "mtgp64-sample-common.h"
#include "parse_opt.h"

using namespace std;
using namespace cl;
using namespace NTL;

/* ================== */
/* OpenCL information */
/* ================== */
std::vector<cl::Platform> platforms;
std::vector<cl::Device> devices;
cl::Context context;
std::string programBuffer;
cl::Program program;
cl::Program::Sources source;
cl::CommandQueue queue;
std::string errorMessage;

/* ========================= */
/* Sample global variables
/* ========================= */
/**
 * max size of jump table
 * 2^(2*MAX_JUMP_TABLE-1) work groups will be supported
 * currently max 2048 work groups are supported
 */
#define MAX_JUMP_TABLE 6
static mtgp64_fast_t mtgp64;
static bool thread_max = false;
/* small size for check */
static const int jump_step = MTGP64_LS * 10;
static ZZ jump;
static uint32_t jump_poly[MTGP64_JTS];
static uint32_t jump_initial[MTGP64_JTS * MAX_JUMP_TABLE];


/* =========================
   declaration
   ========================= */
static int test(int argc, char * argv[]);
static void make_jump_table(int group_num);
static void initialize_by_seed(options& opt,
			       Buffer& status_buffer,
			       int group,
			       uint64_t seed);
static void initialize_by_array(options& opt,
				Buffer& status_buffer,
				int group,
				uint64_t seed_array[],
				int seed_size);
static void status_jump(Buffer& status_buffer, int group);
static void generate_uint64(int group_num,
			    Buffer& status_buffer,
			    int data_size);
static void generate_double12(int group_num,
			      Buffer& status_buffer,
			      int data_size);
static void generate_double01(int group_num,
			      Buffer& status_buffer,
			      int data_size);
static int init_check_data(mtgp64_fast_t * mtgp64,
			   uint64_t seed);
static int init_check_data_array(mtgp64_fast_t * mtgp64,
				 uint64_t seed_array[],
				 int size);
static void free_check_data(mtgp64_fast_t * mtgp64);
static void check_data(uint64_t * h_data, int num_data);
static void check_double12(double * h_data, int num_data);
static void check_double01(double * h_data, int num_data);

/* =========================
   mtgp64 sample code
   ========================= */
/**
 * main
 * catch errors
 *@param argc number of arguments
 *@param argv array of arguments
 *@return 0 normal, -1 error
 */
int main(int argc, char * argv[]) {
    try {
	return test(argc, argv);
    } catch (cl::Error e) {
	cerr << "Error Code:" << e.err() << endl;
	cerr << errorMessage << endl;
	cerr << e.what() << endl;
    }
}

/**
 * sample main
 *@param argc number of arguments
 *@param argv array of arguments
 *@return 0 normal, -1 error
 */
static int test(int argc, char * argv[]) {
#if defined(DEBUG)
    cout << "test start" << endl;
#endif
    options opt;
    if (!parse_opt(opt, argc, argv)) {
	return -1;
    }
    // OpenCL setup
#if defined(DEBUG)
    cout << "openCL setup start" << endl;
#endif
    platforms = getPlatforms();
    devices = getDevices();
    context = getContext();
#if defined(APPLE) || defined(__MACOSX) || defined(__APPLE__)
    source = getSource("mtgp64-jump.cli");
#else
    source = getSource("mtgp64-jump.cl");
#endif
    const char * compile_option = "";
    bool double_extension = false;
    if (hasDoubleExtension()) {
	double_extension = true;
	compile_option = "-DHAVE_DOUBLE";
    }
    program = getProgram(compile_option);
    queue = getCommandQueue();
#if defined(DEBUG)
    cout << "openCL setup end" << endl;
#endif

    int max_group_size = getMaxGroupSize();
    if (opt.group_num > max_group_size) {
	cout << "group_num greater than max value("
	     << max_group_size << ")"
	     << endl;
	return -1;
    }
    int max_size = getMaxWorkItemSize(0);
    if (MTGP64_TN > max_size) {
	cout << "workitem size is greater than max value("
	     << dec << max_size << ")"
	     << "current:" << dec << MTGP64_N << endl;
	return -1;
    }
    if (MTGP64_N > max_size) {
	thread_max = true;
    }
    int local_mem_size = getLocalMemSize();
    if (local_mem_size < sizeof(uint64_t) * MTGP64_N * 2) {
	cout << "local memory size is smaller than min value("
	     << dec << sizeof(uint64_t) * MTGP64_N * 2
	     << ") current:"
	     << dec << local_mem_size << endl;
	return -1;
    }
    Buffer status_buffer(context,
			 CL_MEM_READ_WRITE,
			 sizeof(uint64_t) * MTGP64_N * opt.group_num);

    make_jump_table(opt.group_num);
    int data_count = opt.data_count;
    int data_unit = jump_step * opt.group_num;

    // initialize by seed
    // generate uint64_t
    init_check_data(&mtgp64, 1234);
    initialize_by_seed(opt, status_buffer, opt.group_num, 1234);
    while (data_count > 0) {
	generate_uint64(opt.group_num, status_buffer, data_unit);
	status_jump(status_buffer, opt.group_num);
	data_count -= data_unit;
    }
    free_check_data(&mtgp64);

    // initialize by array
    // generate double
    uint64_t seed_array[5] = {1, 2, 3, 4, 5};
    init_check_data_array(&mtgp64, seed_array, 5);
    initialize_by_array(opt, status_buffer, opt.group_num,
			seed_array, 5);
    data_count = opt.data_count;

    while (data_count > 0) {
	generate_double12(opt.group_num, status_buffer, data_unit);
	status_jump(status_buffer, opt.group_num);
	data_count -= data_unit;
	if (data_count > 0 && double_extension) {
	    generate_double01(opt.group_num, status_buffer, data_unit);
	    status_jump(status_buffer, opt.group_num);
	    data_count -= data_unit;
	}
    }
    free_check_data(&mtgp64);
    return 0;
}

/**
 * prepare jump polynomial.
 * this step may be pre-computed in practical use.
 * @param group_num number of work groups
 */
static void make_jump_table(int group_num)
{
#if defined(DEBUG)
    cout << "make_jump_table start" << endl;
#endif
    mtgp64_fast_t dummy;
    int rc = mtgp64_init(&dummy, &mtgp64dc_params_fast_11213[0], 1);
    if (rc) {
	cerr << "init error" << endl;
	throw cl::Error(rc, "mtgp64 init error");
    }
    GF2X poly;
    clock_t start = clock();
    mtgp64_calc_characteristic(poly, &dummy);
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    cout << "calc_characteristic: " << dec << time << "ms" << endl;
    ZZ step;
    step = jump_step;
    start = clock();
    for (int i = 0; i < MAX_JUMP_TABLE; i++) {
	calc_jump(&jump_initial[i * MTGP64_JTS],
		  MTGP64_JTS,
		  step,
		  poly);
	step *= 4;
    }
    step = jump_step;
    step *= group_num - 1;
    calc_jump(jump_poly, MTGP64_JTS, step, poly);
    end = clock();
    time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    cout << "make jump table: " << dec << time << "ms" << endl;
#if defined(DEBUG)
    cout << "step:" << dec << step << endl;
    cout << "jump_poly[0]:" << hex << jump_poly[0] << endl;
    cout << "jump_poly[1]:" << hex << jump_poly[1] << endl;
    cout << "jump_initial[0]:" << hex << jump_initial[0 * MTGP64_JTS] << endl;
    cout << "jump_initial[1]:" << hex << jump_initial[1 * MTGP64_JTS] << endl;
    cout << "jump_initial[2]:" << hex << jump_initial[2 * MTGP64_JTS] << endl;
#endif
#if defined(DEBUG)
    cout << "make_jump_table end" << endl;
#endif
}

/**
 * initialize mtgp status in device global memory
 * using seed and fixed jump.
 * jump step is fixed to 3^162.
 *@param opt command line option
 *@param status_buffer mtgp status in device global memory
 *@param group number of group
 *@param seed seed for initialization
 */
static void initialize_by_seed(options& opt,
			       Buffer& status_buffer,
			       int group,
			       uint64_t seed)
{
#if defined(DEBUG)
    cout << "initialize_by_seed start" << endl;
#endif
    // jump table
    Buffer jump_table_buffer(context,
			     CL_MEM_READ_WRITE,
			     MTGP64_JTS * MAX_JUMP_TABLE * sizeof(uint32_t));
    queue.enqueueWriteBuffer(jump_table_buffer,
			     CL_TRUE,
			     0,
			     MTGP64_JTS * MAX_JUMP_TABLE * sizeof(uint32_t),
			     jump_initial);

    Kernel init_kernel(program, "mtgp64_jump_seed_kernel");
#if defined(DEBUG)
    cout << "arg0 start" << endl;
#endif
    init_kernel.setArg(0, status_buffer);
    init_kernel.setArg(1, seed);
    init_kernel.setArg(2, jump_table_buffer);
#if defined(DEBUG)
    cout << "arg2 end" << endl;
#endif
    int local_item = MTGP64_N;
    if (thread_max) {
	local_item = MTGP64_TN;
    }
    NDRange global(group * local_item);
    NDRange local(local_item);
    Event event;
#if defined(DEBUG)
    cout << "global:" << dec << group * local_item << endl;
    cout << "group:" << dec << group << endl;
    cout << "local:" << dec << local_item << endl;
#endif
    queue.enqueueNDRangeKernel(init_kernel,
			       NullRange,
			       global,
			       local,
			       NULL,
			       &event);
    double time = get_time(event);
    cout << "initializing time = " << time * 1000 << "ms" << endl;
#if 0
    uint status[group * MTGP64_N];
    queue.enqueueReadBuffer(status_buffer,
			    CL_TRUE,
			    0,
			    sizeof(uint64_t) * MTGP64_N * group,
			    status);
#if defined(DEBUG)
    cout << "status[0]:" << hex << status[0] << endl;
    cout << "status[MTGP64_N - 1]:" << hex << status[MTGP64_N - 1] << endl;
    cout << "status[MTGP64_N]:" << hex << status[MTGP64_N] << endl;
    cout << "status[MTGP64_N + 1]:" << hex << status[MTGP64_N + 1] << endl;
#endif
    check_status(status, group);
#endif
#if defined(DEBUG)
    cout << "initialize_by_seed end" << endl;
#endif
}

/**
 * initialize mtgp status in device global memory
 * using an array of seeds and jump.
 *@param opt command line option
 *@param status_buffer mtgp status in device global memory
 *@param group number of group
 *@param seed_array seeds for initialization
 *@param seed_size size of seed_array
 */
static void initialize_by_array(options& opt,
				Buffer& status_buffer,
				int group,
				uint64_t seed_array[],
				int seed_size)
{
#if defined(DEBUG)
    cout << "initialize_by_array start" << endl;
#endif
    // jump table
    Buffer jump_table_buffer(context,
			     CL_MEM_READ_WRITE,
			     MTGP64_JTS * MAX_JUMP_TABLE * sizeof(uint32_t));
    queue.enqueueWriteBuffer(jump_table_buffer,
			     CL_TRUE,
			     0,
			     MTGP64_JTS * MAX_JUMP_TABLE * sizeof(uint32_t),
			     jump_initial);

    Buffer seed_array_buffer(context,
			     CL_MEM_READ_WRITE,
			     seed_size * sizeof(uint64_t));
    queue.enqueueWriteBuffer(seed_array_buffer,
			     CL_TRUE,
			     0,
			     seed_size * sizeof(uint64_t),
			     seed_array);
    Kernel init_kernel(program, "mtgp64_jump_array_kernel");
    init_kernel.setArg(0, status_buffer);
    init_kernel.setArg(1, seed_array_buffer);
    init_kernel.setArg(2, seed_size);
    init_kernel.setArg(3, jump_table_buffer);
    int local_item = MTGP64_N;
    if (thread_max) {
	local_item = MTGP64_TN;
    }
    NDRange global(group * local_item);
    NDRange local(local_item);
    Event event;
    queue.enqueueNDRangeKernel(init_kernel,
			       NullRange,
			       global,
			       local,
			       NULL,
			       &event);
    double time = get_time(event);
#if 0
    uint status[group * MTGP64_N];
    queue.enqueueReadBuffer(status_buffer,
			    CL_TRUE,
			    0,
			    sizeof(uint64_t) * MTGP64_N * group,
			    status);
    check_status(status, group);
#endif
    cout << "initializing time = " << time * 1000 << "ms" << endl;
#if defined(DEBUG)
    cout << "initialize_by_array end" << endl;
#endif
}

/**
 * jump mtgp status in device global memory
 *@param status_buffer mtgp status in device global memory
 *@param group number of group
 */
static void status_jump(Buffer& status_buffer, int group)
{
#if defined(DEBUG)
    cout << "jump start" << endl;
#endif
    // jump table
    Buffer jump_table_buffer(context,
			     CL_MEM_READ_WRITE,
			     MTGP64_JTS * sizeof(uint32_t));
    queue.enqueueWriteBuffer(jump_table_buffer,
			     CL_TRUE,
			     0,
			     MTGP64_JTS * sizeof(uint32_t),
			     jump_poly);

    Kernel init_kernel(program, "mtgp64_jump_kernel");
    init_kernel.setArg(0, status_buffer);
    init_kernel.setArg(1, jump_table_buffer);
    int local_item = MTGP64_N;
    if (thread_max) {
	local_item = MTGP64_TN;
    }
    NDRange global(group * local_item);
    NDRange local(local_item);
    Event event;
#if defined(DEBUG)
    cout << "global:" << dec << group * local_item << endl;
    cout << "group:" << dec << group << endl;
    cout << "local:" << dec << local_item << endl;
#endif
    queue.enqueueNDRangeKernel(init_kernel,
			       NullRange,
			       global,
			       local,
			       NULL,
			       &event);
    double time = get_time(event);
    cout << "jump time = " << time * 1000 << "ms" << endl;
#if defined(DEBUG)
    cout << "jump end" << endl;
#endif
}

/**
 * generate 64 bit unsigned random numbers in device global memory
 *@param group_num number of groups for execution
 *@param status_buffer mtgp status in device global memory
 *@param data_size number of data to generate
 */
static void generate_uint64(int group_num,
			    Buffer& status_buffer,
			    int data_size)
{
#if defined(DEBUG)
    cout << "generate_uint64 start" << endl;
    cout << "data_size:" << dec << data_size << endl;
#endif
    int item_num = MTGP64_TN * group_num;
    int min_size = MTGP64_LS * group_num;
    if (data_size % min_size != 0) {
	data_size = (data_size / min_size + 1) * min_size;
    }
#if defined(DEBUG)
    cout << "data_size:" << dec << data_size << endl;
#endif
    Kernel uint_kernel(program, "mtgp64_uint64_kernel");
    Buffer output_buffer(context,
			 CL_MEM_READ_WRITE,
			 data_size * sizeof(uint64_t));
    uint_kernel.setArg(0, status_buffer);
    uint_kernel.setArg(1, output_buffer);
    uint_kernel.setArg(2, data_size / group_num);
    NDRange global(item_num);
    NDRange local(MTGP64_TN);
    Event generate_event;
#if defined(DEBUG)
    cout << "generate_uint64 enque kernel start" << endl;
#endif
    queue.enqueueNDRangeKernel(uint_kernel,
			       NullRange,
			       global,
			       local,
			       NULL,
			       &generate_event);
#if defined(DEBUG)
    cout << "generate_uint64 enque kernel end" << endl;
#endif
    uint64_t * output = new uint64_t[data_size];
#if defined(DEBUG)
    cout << "generate_uint64 event wait start" << endl;
#endif
    generate_event.wait();
#if defined(DEBUG)
    cout << "generate_uint64 event wait end" << endl;
#endif
#if defined(DEBUG)
    cout << "generate_uint64 readbuffer start" << endl;
#endif
    queue.enqueueReadBuffer(output_buffer,
			    CL_TRUE,
			    0,
			    data_size * sizeof(uint64_t),
			    &output[0]);
#if defined(DEBUG)
    cout << "generate_uint64 readbuffer end" << endl;
#endif
    check_data(output, data_size);
    print_uint64(&output[0], data_size, item_num);
    double time = get_time(generate_event);
    cout << "generate time:" << time * 1000 << "ms" << endl;
    delete[] output;
#if defined(DEBUG)
    cout << "generate_uint64 end" << endl;
#endif
}

/**
 * generate double precision floating point numbers in the range [1, 2)
 * in device global memory
 *@param group_num number of groups for execution
 *@param status_buffer mtgp status in device global memory
 *@param data_size number of data to generate
 */
static void generate_double12(int group_num,
			      Buffer& status_buffer,
			      int data_size)
{
    int item_num = MTGP64_TN * group_num;
    int min_size = MTGP64_LS * group_num;
    if (data_size % min_size != 0) {
	data_size = (data_size / min_size + 1) * min_size;
    }
    Kernel double_kernel(program, "mtgp64_double12_kernel");
    Buffer output_buffer(context,
			 CL_MEM_READ_WRITE,
			 data_size * sizeof(double));
    double_kernel.setArg(0, status_buffer);
    double_kernel.setArg(1, output_buffer);
    double_kernel.setArg(2, data_size / group_num);
    NDRange global(item_num);
    NDRange local(MTGP64_TN);
    Event generate_event;
    queue.enqueueNDRangeKernel(double_kernel,
			       NullRange,
			       global,
			       local,
			       NULL,
			       &generate_event);
    double * output = new double[data_size];
    generate_event.wait();
    queue.enqueueReadBuffer(output_buffer,
			    CL_TRUE,
			    0,
			    data_size * sizeof(double),
			    &output[0]);
    check_double12(output, data_size);
    print_double(output, data_size, item_num);
    double time = get_time(generate_event);
    delete[] output;
    cout << "generate time:" << time * 1000 << "ms" << endl;
}

/**
 * generate double precision floating point numbers in the range [0, 1)
 * in device global memory
 *@param group_num number of groups for execution
 *@param status_buffer mtgp status in device global memory
 *@param data_size number of data to generate
 */
static void generate_double01(int group_num,
			      Buffer& status_buffer,
			      int data_size)
{
    int item_num = MTGP64_TN * group_num;
    int min_size = MTGP64_LS * group_num;
    if (data_size % min_size != 0) {
	data_size = (data_size / min_size + 1) * min_size;
    }
    Kernel double_kernel(program, "mtgp64_double01_kernel");
    Buffer output_buffer(context,
			 CL_MEM_READ_WRITE,
			 data_size * sizeof(double));
    double_kernel.setArg(0, status_buffer);
    double_kernel.setArg(1, output_buffer);
    double_kernel.setArg(2, data_size / group_num);
    NDRange global(item_num);
    NDRange local(MTGP64_TN);
    Event generate_event;
    queue.enqueueNDRangeKernel(double_kernel,
			       NullRange,
			       global,
			       local,
			       NULL,
			       &generate_event);
    double * output = new double[data_size];
    generate_event.wait();
    queue.enqueueReadBuffer(output_buffer,
			    CL_TRUE,
			    0,
			    data_size * sizeof(double),
			    &output[0]);
    check_double01(output, data_size);
    print_double(output, data_size, item_num);
    double time = get_time(generate_event);
    delete[] output;
    cout << "generate time:" << time * 1000 << "ms" << endl;
}

/* ==============
 * check programs
 * ==============*/
static int init_check_data(mtgp64_fast_t * mtgp64,
			   uint64_t seed)
{
#if defined(DEBUG)
    cout << "init_check_data start" << endl;
#endif
    int rc = mtgp64_init(mtgp64,
			 &mtgp64dc_params_fast_11213[0],
			 seed);
    if (rc) {
	return rc;
    }
#if defined(DEBUG)
    cout << "init_check_data end" << endl;
#endif
    return 0;
}

static int init_check_data_array(mtgp64_fast_t * mtgp64,
				 uint64_t seed_array[],
				 int size)
{
#if defined(DEBUG)
    cout << "init_check_data_array start" << endl;
#endif
    int rc = mtgp64_init_by_array(mtgp64,
				  &mtgp64dc_params_fast_11213[0],
				  seed_array,
				  size);
    if (rc) {
	return rc;
    }
#if defined(DEBUG)
    cout << "init_check_data_array end" << endl;
#endif
    return 0;
}

static void free_check_data(mtgp64_fast_t * mtgp64)
{
#if defined(DEBUG)
    cout << "free_check_data start" << endl;
#endif
    mtgp64_free(mtgp64);
#if defined(DEBUG)
    cout << "free_check_data end" << endl;
#endif
}

static void check_data(uint64_t * h_data, int num_data)
{
#if defined(DEBUG)
    cout << "check_data start" << endl;
    cout << "num_data:" << dec << num_data << endl;
#endif
    bool error = false;
    bool disp_flg = true;
    int count = 0;
    for (int j = 0; j < num_data; j++) {
	uint64_t r = mtgp64_genrand_uint64(&mtgp64);
	if ((h_data[j] != r) && disp_flg) {
	    cout << "mismatch"
		 << " j = " << dec << j
		 << " data = " << hex << h_data[j]
		 << " r = " << hex << r << endl;
	    cout << "check_data check N.G!" << endl;
	    count++;
	    error = true;
	}
	if (count > 10) {
	    disp_flg = false;
	}
    }
    if (!error) {
	cout << "check_data check O.K!" << endl;
    }
#if defined(DEBUG)
    cout << "check_data end" << endl;
#endif
}

static void check_double12(double * h_data, int num_data)
{
#if defined(DEBUG)
    cout << "check_double start" << endl;
#endif
    bool error = false;
    bool disp_flg = true;
    int count = 0;
    for (int j = 0; j < num_data; j++) {
	double r =  mtgp64_genrand_close1_open2(&mtgp64);
	if (!(-FLT_EPSILON <= h_data[j] - r &&
	     h_data[j] - r <= FLT_EPSILON)
	    && disp_flg) {
	    cout << "mismatch"
		 << " j = " << dec << j
		 << " data = " << dec << h_data[j]
		 << " r = " << dec << r << endl;
	    cout << "check_data check N.G!" << endl;
	    count++;
	    error = true;
	}
	if (count > 10) {
	    disp_flg = false;
	}
    }
    if (!error) {
	cout << "check_double check O.K!" << endl;
    }
#if defined(DEBUG)
    cout << "check_double end" << endl;
#endif
}

static void check_double01(double * h_data, int num_data)
{
#if defined(DEBUG)
    cout << "check_double start" << endl;
#endif
    bool error = false;
    bool disp_flg = true;
    int count = 0;
    for (int j = 0; j < num_data; j++) {
	double r =  mtgp64_genrand_close_open(&mtgp64);
	if (!(-FLT_EPSILON <= h_data[j] - r &&
	     h_data[j] - r <= FLT_EPSILON)
	    && disp_flg) {
	    cout << "mismatch"
		 << " j = " << dec << j
		 << " data = " << dec << h_data[j]
		 << " r = " << dec << r << endl;
	    cout << "check_data check N.G!" << endl;
	    count++;
	    error = true;
	}
	if (count > 10) {
	    disp_flg = false;
	}
    }
    if (!error) {
	cout << "check_double check O.K!" << endl;
    }
#if defined(DEBUG)
    cout << "check_double end" << endl;
#endif
}


