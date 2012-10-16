/**
 * Sample host program for far jump
 * 3<sup>162</sup> steps jump ahead
 */
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define __CL_ENABLE_EXCEPTIONS

#include "opencl_tools.hpp"

#include <cstddef>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <float.h>

typedef uint32_t uint;
#include "mtgp32-fast-jump.h"
#include "mtgp32-sample-common.h"
#include "mtgp32-jump-string.h"
#include "mtgp32-jump-table.h"
#include "parse_opt.h"

using namespace std;
using namespace cl;

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

#define MAX_GROUP_NUM 20
static mtgp32_fast_t mtgp32[MAX_GROUP_NUM];
static bool thread_max = false;

/* =========================
   declaration
   ========================= */
static int init_check_data(mtgp32_fast_t mtgp32[],
			   int group_num,
			   uint32_t seed);
static int init_check_data_array(mtgp32_fast_t mtgp32[],
				 int group_num,
				 uint32_t seed_array[],
				 int size);
static void free_check_data(mtgp32_fast_t mtgp32[], int group_num);
static void check_data(uint32_t * h_data,
		       int num_data,
		       int group_num);
static void check_float12(float * h_data,
		       int num_data,
		       int group_num);
static void check_float01(float * h_data,
		       int num_data,
		       int group_num);
static void check_status(uint * h_status,
			 int group_num);
static int test(int argc, char * argv[]);
static void initialize_by_seed(Buffer& status_buffer,
			       int group,
			       uint32_t seed);
static void initialize_by_array(Buffer& status_buffer,
				int group,
				uint32_t seed_array[],
				int seed_size);
static void generate_uint32(int group_num,
			    Buffer& status_buffer,
			    int data_size);
static void generate_single12(int group_num,
			      Buffer& status_buffer,
			      int data_size);
static void generate_single01(int group_num,
			      Buffer& status_buffer,
			      int data_size);
/* =========================
   mtgp32 sample code
   ========================= */

/**
 * main
 * catch errors
 *@param argc number of arguments
 *@param argv array of arguments
 *@return 0 normal, -1 error
 */
int main(int argc, char * argv[])
{
    try {
	return test(argc, argv);
    } catch (Error e) {
	cerr << "Error Code:" << e.err() << endl;
	cerr << e.what() << endl;
	return -1;
    }
    return 0;
}

/**
 * sample main
 *@param argc number of arguments
 *@param argv array of arguments
 *@return 0 normal, -1 error
 */
static int test(int argc, char * argv[])
{
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
    source = getSource("mtgp32-jump.cli");
#else
    source = getSource("mtgp32-jump.cl");
#endif
    program = getProgram();
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
    if (MTGP32_TN > max_size) {
	cout << "workitem size is greater than max value("
	     << dec << max_size << ")"
	     << "current:" << dec << MTGP32_N << endl;
	return -1;
    }
    if (MTGP32_N > max_size) {
	thread_max = true;
    }
    int local_mem_size = getLocalMemSize();
    if (local_mem_size < sizeof(uint32_t) * MTGP32_N * 2) {
	cout << "local memory size not sufficient:"
	     << dec << local_mem_size << endl;
	return -1;
    }
    Buffer status_buffer(context,
			 CL_MEM_READ_WRITE,
			 sizeof(uint32_t) * MTGP32_N * opt.group_num);

    // initialize by seed
    // generate uint32_t
    init_check_data(mtgp32, opt.group_num, 1234);
    initialize_by_seed(status_buffer, opt.group_num, 1234);
    for (int i = 0; i < 2; i++) {
	generate_uint32(opt.group_num, status_buffer, opt.data_count );
    }
    free_check_data(mtgp32, opt.group_num);

    // initialize by array
    // generate single float
    uint32_t seed_array[5] = {1, 2, 3, 4, 5};
    init_check_data_array(mtgp32, opt.group_num, seed_array, 5);
    initialize_by_array(status_buffer, opt.group_num,
			seed_array, 5);
    for (int i = 0; i < 1; i++) {
	generate_single12(opt.group_num, status_buffer, opt.data_count);
	generate_single01(opt.group_num, status_buffer, opt.data_count);
    }
    free_check_data(mtgp32, opt.group_num);
    return 0;
}

/**
 * initialize mtgp status in device global memory
 * using seed and fixed jump.
 * jump step is fixed to 3^162.
 *@param status_buffer mtgp status in device global memory
 *@param group number of group
 *@param seed seed for initialization
 */
static void initialize_by_seed(Buffer& status_buffer,
			       int group,
			       uint32_t seed)
{
#if defined(DEBUG)
    cout << "initialize_by_seed start" << endl;
#endif
    // jump table
    Buffer jump_table_buffer(context,
			     CL_MEM_READ_WRITE,
			     MTGP32_N * 6 * sizeof(uint32_t));
    queue.enqueueWriteBuffer(jump_table_buffer,
			     CL_TRUE,
			     0,
			     MTGP32_N * 6 * sizeof(uint32_t),
			     mtgp32_jump_table);

    Kernel init_kernel(program, "mtgp32_jump_seed_kernel");
#if defined(DEBUG)
    cout << "arg0 start" << endl;
#endif
    init_kernel.setArg(0, status_buffer);
    init_kernel.setArg(1, seed);
    init_kernel.setArg(2, jump_table_buffer);
#if defined(DEBUG)
    cout << "arg2 end" << endl;
#endif
    int local_item = MTGP32_N;
    if (thread_max) {
	local_item = MTGP32_TN;
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
    uint status[group * MTGP32_N];
    queue.enqueueReadBuffer(status_buffer,
			    CL_TRUE,
			    0,
			    sizeof(uint32_t) * MTGP32_N * group,
			    status);
    cout << "initializing time = " << time * 1000 << "ms" << endl;
#if defined(DEBUG)
    cout << "status[0]:" << hex << status[0] << endl;
    cout << "status[MTGP32_N - 1]:" << hex << status[MTGP32_N - 1] << endl;
    cout << "status[MTGP32_N]:" << hex << status[MTGP32_N] << endl;
    cout << "status[MTGP32_N + 1]:" << hex << status[MTGP32_N + 1] << endl;
#endif
    check_status(status, group);
#if defined(DEBUG)
    cout << "initialize_by_seed end" << endl;
#endif
}

/**
 * initialize mtgp status in device global memory
 * using an array of seeds and fixed jump.
 * jump step is fixed to 3^162.
 *@param status_buffer mtgp status in device global memory
 *@param group number of group
 *@param seed_array seeds for initialization
 *@param seed_size size of seed_array
 */
static void initialize_by_array(Buffer& status_buffer,
				int group,
				uint32_t seed_array[],
				int seed_size)
{
#if defined(DEBUG)
    cout << "initialize_by_array start" << endl;
#endif
    // jump table
    Buffer jump_table_buffer(context,
			     CL_MEM_READ_WRITE,
			     MTGP32_N * 6 * sizeof(uint32_t));
    queue.enqueueWriteBuffer(jump_table_buffer,
			     CL_TRUE,
			     0,
			     MTGP32_N * 6 * sizeof(uint32_t),
			     mtgp32_jump_table);

    Buffer seed_array_buffer(context,
			     CL_MEM_READ_WRITE,
			     seed_size * sizeof(uint32_t));
    queue.enqueueWriteBuffer(seed_array_buffer,
			     CL_TRUE,
			     0,
			     seed_size * sizeof(uint32_t),
			     seed_array);
    Kernel init_kernel(program, "mtgp32_jump_array_kernel");
    init_kernel.setArg(0, status_buffer);
    init_kernel.setArg(1, seed_array_buffer);
    init_kernel.setArg(2, seed_size);
    init_kernel.setArg(3, jump_table_buffer);
    int local_item = MTGP32_N;
    if (thread_max) {
	local_item = MTGP32_TN;
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
    uint status[group * MTGP32_N];
    queue.enqueueReadBuffer(status_buffer,
			    CL_TRUE,
			    0,
			    sizeof(uint32_t) * MTGP32_N * group,
			    status);
    cout << "initializing time = " << time * 1000 << "ms" << endl;
    check_status(status, group);
#if defined(DEBUG)
    cout << "initialize_by_array end" << endl;
#endif
}

/**
 * generate 32 bit unsigned random numbers in device global memory
 *@param group_num number of groups for execution
 *@param status_buffer mtgp status in device global memory
 *@param data_size number of data to generate
 */
static void generate_uint32(int group_num,
			    Buffer& status_buffer,
			    int data_size)
{
#if defined(DEBUG)
    cout << "generate_uint32 start" << endl;
#endif
    int item_num = MTGP32_TN * group_num;
    int min_size = MTGP32_LS * group_num;
    if (data_size % min_size != 0) {
	data_size = (data_size / min_size + 1) * min_size;
    }
    Kernel uint_kernel(program, "mtgp32_uint32_kernel");
    Buffer output_buffer(context,
			 CL_MEM_READ_WRITE,
			 data_size * sizeof(uint32_t));
    uint_kernel.setArg(0, status_buffer);
    uint_kernel.setArg(1, output_buffer);
    uint_kernel.setArg(2, data_size / group_num);
    NDRange global(item_num);
    NDRange local(MTGP32_TN);
    Event generate_event;
#if defined(DEBUG)
    cout << "generate_uint32 enque kernel start" << endl;
#endif
    queue.enqueueNDRangeKernel(uint_kernel,
			       NullRange,
			       global,
			       local,
			       NULL,
			       &generate_event);
#if defined(DEBUG)
    cout << "generate_uint32 enque kernel end" << endl;
#endif
    uint32_t * output = new uint32_t[data_size];
#if defined(DEBUG)
    cout << "generate_uint32 event wait start" << endl;
#endif
    generate_event.wait();
#if defined(DEBUG)
    cout << "generate_uint32 event wait end" << endl;
#endif
#if defined(DEBUG)
    cout << "generate_uint32 readbuffer start" << endl;
#endif
    queue.enqueueReadBuffer(output_buffer,
			    CL_TRUE,
			    0,
			    data_size * sizeof(uint32_t),
			    &output[0]);
#if defined(DEBUG)
    cout << "generate_uint32 readbuffer end" << endl;
#endif
    check_data(output, data_size, group_num);
    print_uint32(&output[0], data_size, item_num);
    double time = get_time(generate_event);
    cout << "generate time:" << time * 1000 << "ms" << endl;
    delete[] output;
#if defined(DEBUG)
    cout << "generate_uint32 end" << endl;
#endif
}

/**
 * generate single precision floating point numbers in the range [1, 2)
 * in device global memory
 *@param group_num number of groups for execution
 *@param status_buffer mtgp status in device global memory
 *@param data_size number of data to generate
 */
static void generate_single12(int group_num,
			      Buffer& status_buffer,
			      int data_size)
{
    int item_num = MTGP32_TN * group_num;
    int min_size = MTGP32_LS * group_num;
    if (data_size % min_size != 0) {
	data_size = (data_size / min_size + 1) * min_size;
    }
    Kernel single_kernel(program, "mtgp32_single12_kernel");
    Buffer output_buffer(context,
			 CL_MEM_READ_WRITE,
			 data_size * sizeof(float));
    single_kernel.setArg(0, status_buffer);
    single_kernel.setArg(1, output_buffer);
    single_kernel.setArg(2, data_size / group_num);
    NDRange global(item_num);
    NDRange local(MTGP32_TN);
    Event generate_event;
    queue.enqueueNDRangeKernel(single_kernel,
			       NullRange,
			       global,
			       local,
			       NULL,
			       &generate_event);
    float * output = new float[data_size];
    generate_event.wait();
    queue.enqueueReadBuffer(output_buffer,
			    CL_TRUE,
			    0,
			    data_size * sizeof(float),
			    &output[0]);
    check_float12(output, data_size, group_num);
    print_float(output, data_size, item_num);
    double time = get_time(generate_event);
    delete[] output;
    cout << "generate time:" << time * 1000 << "ms" << endl;
}

/**
 * generate single precision floating point numbers in the range [0, 1)
 * in device global memory
 *@param group_num number of groups for execution
 *@param status_buffer mtgp status in device global memory
 *@param data_size number of data to generate
 */
static void generate_single01(int group_num,
			      Buffer& status_buffer,
			      int data_size)
{
    int item_num = MTGP32_TN * group_num;
    int min_size = MTGP32_LS * group_num;
    if (data_size % min_size != 0) {
	data_size = (data_size / min_size + 1) * min_size;
    }
    Kernel single_kernel(program, "mtgp32_single01_kernel");
    Buffer output_buffer(context,
			 CL_MEM_READ_WRITE,
			 data_size * sizeof(float));
    single_kernel.setArg(0, status_buffer);
    single_kernel.setArg(1, output_buffer);
    single_kernel.setArg(2, data_size / group_num);
    NDRange global(item_num);
    NDRange local(MTGP32_TN);
    Event generate_event;
    queue.enqueueNDRangeKernel(single_kernel,
			       NullRange,
			       global,
			       local,
			       NULL,
			       &generate_event);
    float * output = new float[data_size];
    generate_event.wait();
    queue.enqueueReadBuffer(output_buffer,
			    CL_TRUE,
			    0,
			    data_size * sizeof(float),
			    &output[0]);
    check_float01(output, data_size, group_num);
    print_float(output, data_size, item_num);
    double time = get_time(generate_event);
    delete[] output;
    cout << "generate time:" << time * 1000 << "ms" << endl;
}

/* ==============
 * check programs
 * ==============*/
static int init_check_data(mtgp32_fast_t mtgp32[],
			   int group_num,
			   uint32_t seed)
{
#if defined(DEBUG)
    cout << "init_check_data start" << endl;
#endif
    if (group_num > MAX_GROUP_NUM) {
	cout << "can't check group number > " << dec << MAX_GROUP_NUM
	     << group_num << endl;
	return 1;
    }
    for (int i = 0; i < group_num; i++) {
	int rc = mtgp32_init(&mtgp32[i],
			     &mtgp32_params_fast_11213[0],
			     seed);
	if (rc) {
	    return rc;
	}
	if (i == 0) {
	    continue;
	}
	for (int j = 0; j < i; j++) {
	    mtgp32_fast_jump(&mtgp32[i], mtgp32_jump_string);
	}
    }
#if defined(DEBUG)
    cout << "init_check_data end" << endl;
#endif
    return 0;
}

static int init_check_data_array(mtgp32_fast_t mtgp32[],
				 int group_num,
				 uint32_t seed_array[],
				 int size)
{
#if defined(DEBUG)
    cout << "init_check_data_array start" << endl;
#endif
    if (group_num > MAX_GROUP_NUM) {
	cout << "can't check group number > " << dec << MAX_GROUP_NUM
	     << group_num << endl;
	return 1;
    }
    for (int i = 0; i < group_num; i++) {
	int rc = mtgp32_init_by_array(&mtgp32[i],
				      &mtgp32_params_fast_11213[0],
				      seed_array,
				      size);
	if (rc) {
	    return rc;
	}
	for (int j = 0; j < i; j++) {
	    mtgp32_fast_jump(&mtgp32[i], mtgp32_jump_string);
	}
    }
#if defined(DEBUG)
    cout << "init_check_data_array end" << endl;
#endif
    return 0;
}

static void free_check_data(mtgp32_fast_t mtgp32[], int group_num)
{
#if defined(DEBUG)
    cout << "free_check_data start" << endl;
#endif
    for (int i = 0; i < group_num; i++) {
	mtgp32_free(&mtgp32[i]);
    }
#if defined(DEBUG)
    cout << "free_check_data end" << endl;
#endif
}

static void check_data(uint32_t * h_data,
		       int num_data,
		       int group_num)
{
#if defined(DEBUG)
    cout << "check_data start" << endl;
#endif
    int size = num_data / group_num;
#if defined(DEBUG)
    cout << "size = " << dec << size << endl;
#endif
    if (group_num > MAX_GROUP_NUM) {
	cout << "can't check group number > " << dec << MAX_GROUP_NUM
	     << " current:" << group_num << endl;
	return;
    }
    bool error = false;
    for (int i = 0; i < group_num; i++) {
	bool disp_flg = true;
	int count = 0;
	for (int j = 0; j < size; j++) {
	    uint32_t r = mtgp32_genrand_uint32(&mtgp32[i]);
	    if ((h_data[i * size + j] != r) && disp_flg) {
		cout << "mismatch i = " << dec << i
		     << " j = " << dec << j
		     << " data = " << hex << h_data[i * size + j]
		     << " r = " << hex << r << endl;
		cout << "check_data check N.G!" << endl;
		count++;
		error = true;
	    }
	    if (count > 10) {
		disp_flg = false;
	    }
	}
    }
    if (!error) {
	cout << "check_data check O.K!" << endl;
    }
#if defined(DEBUG)
    cout << "check_data end" << endl;
#endif
}

static void check_float12(float * h_data,
			  int num_data,
			  int group_num)
{
#if defined(DEBUG)
    cout << "check_data start" << endl;
#endif
    int size = num_data / group_num;
#if defined(DEBUG)
    cout << "size = " << dec << size << endl;
#endif
    if (group_num > MAX_GROUP_NUM) {
	cout << "can't check group number > " << dec << MAX_GROUP_NUM
	     << " current:" << group_num << endl;
	return;
    }
    bool error = false;
    for (int i = 0; i < group_num; i++) {
	bool disp_flg = true;
	int count = 0;
	for (int j = 0; j < size; j++) {
	    float r = mtgp32_genrand_close1_open2(&mtgp32[i]);
	    float d = h_data[i * size + j];
	    bool ok = (-FLT_EPSILON <= (r - d))
		&& ((r - d) <= FLT_EPSILON);
	    if (!ok && disp_flg) {
		cout << "mismatch i = " << dec << i
		     << " j = " << dec << j
		     << " data = " << hex << h_data[i * size + j]
		     << " r = " << hex << r << endl;
		cout << "check_data check N.G!" << endl;
		count++;
		error = true;
	    }
	    if (count > 10) {
		disp_flg = false;
	    }
	}
    }
    if (!error) {
	cout << "check_data check O.K!" << endl;
    }
#if defined(DEBUG)
    cout << "check_data end" << endl;
#endif
}

static void check_float01(float * h_data,
			  int num_data,
			  int group_num)
{
#if defined(DEBUG)
    cout << "check_data start" << endl;
#endif
    int size = num_data / group_num;
#if defined(DEBUG)
    cout << "size = " << dec << size << endl;
#endif
    if (group_num > MAX_GROUP_NUM) {
	cout << "can't check group number > " << dec << MAX_GROUP_NUM
	     << " current:" << group_num << endl;
	return;
    }
    bool error = false;
    for (int i = 0; i < group_num; i++) {
	bool disp_flg = true;
	int count = 0;
	for (int j = 0; j < size; j++) {
	    float r = mtgp32_genrand_close_open(&mtgp32[i]);
	    float d = h_data[i * size + j];
	    bool ok = (-FLT_EPSILON <= (r - d))
		&& ((r - d) <= FLT_EPSILON);
	    if (!ok && disp_flg) {
		cout << "mismatch i = " << dec << i
		     << " j = " << dec << j
		     << " data = " << hex << h_data[i * size + j]
		     << " r = " << hex << r << endl;
		cout << "check_data check N.G!" << endl;
		count++;
		error = true;
	    }
	    if (count > 10) {
		disp_flg = false;
	    }
	}
    }
    if (!error) {
	cout << "check_data check O.K!" << endl;
    }
#if defined(DEBUG)
    cout << "check_data end" << endl;
#endif
}

static void check_status(uint * h_status,
			 int group_num)
{
#if defined(DEBUG)
    cout << "check_status start" << endl;
#endif
    int counter = 0;
    if (group_num > MAX_GROUP_NUM) {
	cout << "can't check group number > " << dec << MAX_GROUP_NUM
	     << " current:" << group_num << endl;
	return;
    }
    int large_size = mtgp32[0].status->large_size;
    for (int i = 0; i < group_num; i++) {
	for (int j = 0; j < MTGP32_N; j++) {
	    int idx = mtgp32[i].status->idx - MTGP32_N + 1 + large_size;
	    uint32_t x = h_status[i * MTGP32_N + j];
	    uint32_t r = mtgp32[i].status->array[(j + idx) % large_size];
	    if (j == 0) {
		x = x & mtgp32[i].params.mask;
		r = r & mtgp32[i].params.mask;
	    }
	    if (x != r) {
		cout << "mismatch i = " << dec << i
		     << " j = " << dec << j
		     << " device = " << hex << x
		     << " host = " << hex << r << endl;
		cout << "check_status check N.G!" << endl;
		counter++;
	    }
	    if (counter > 10) {
		return;
	    }
	}
    }
    if (counter == 0) {
	cout << "check_status check O.K!" << endl;
    }
#if defined(DEBUG)
    cout << "check_status end" << endl;
#endif
}

