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
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define __CL_ENABLE_EXCEPTIONS

#if defined(APPLE) || defined(__MACOSX) || defined(__APPLE__)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

//#include <exception>
#include <cstddef>
#include <cfloat>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <NTL/GF2X.h>
#include <NTL/ZZ.h>

typedef uint32_t uint;
#include "mtgp32-calc-poly.hpp"
#include "mtgp-calc-jump.hpp"
#include "mtgp32-fast-jump.h"
#include "mtgp32-sample-jump-common.h"
#include "parse_opt.h"

using namespace std;
using namespace cl;
using namespace NTL;

/* ================== */
/* OpenCL information */
/* ================== */
static std::vector<cl::Platform> platforms;
static std::vector<cl::Device> devices;
static cl::Context context;
static std::string programBuffer;
static cl::Program program;
static cl::Program::Sources source;
static cl::CommandQueue queue;
static std::string errorMessage;

/* ========================= */
/* Sample global variables
/* ========================= */
/**
 * max size of jump table
 * 2^(2*MAX_JUMP_TABLE-1) work groups will be supported
 * currently max 2048 work groups are supported
 */
#define MAX_JUMP_TABLE 6
static mtgp32_fast_t mtgp32;
static bool thread_max = false;
/* small size for check */
static const int jump_step = MTGP32_LS * 10;
static ZZ jump;
static uint32_t jump_poly[MTGP32_N];
static uint32_t jump_initial[MTGP32_N * MAX_JUMP_TABLE];

/* ========================= */
/* OpenCL interface function */
/* ========================= */
static void getPlatforms()
{
#if defined(DEBUG)
    cout << "start get platform" << endl;
#endif
    errorMessage = "getPlatform failed";
    cl_int err = Platform::get(&platforms);
    if(err != CL_SUCCESS)
    {
        cout << "getPlatform failed" << err << endl;
	return;
    }
#if defined(DEBUG)
    cout << "vendor:" << endl;
    for (int i = 0; i < platforms.size(); i++) {
	cout << platforms[i].getInfo<CL_PLATFORM_VENDOR>() << endl;
    }
#endif
    errorMessage = "";
#if defined(DEBUG)
    cout << "end get platform" << endl;
#endif
}

static void getDevices()
{
#if defined(DEBUG)
    cout << "start get devices" << endl;
#endif
    cl_int err;
    errorMessage = "getDevices failed";
    err = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if(err != CL_SUCCESS)
    {
        cout << "getDevices failed err:" << err << endl;
	return;
    }
    errorMessage = "";
#if defined(DEBUG)
    cout << "end get devices" << endl;
#endif
}

static void getContext()
{
#if defined(DEBUG)
    cout << "start get context" << endl;
#endif
    errorMessage = "create context failed";
    Context local_context(devices);
    context = local_context;
    errorMessage = "";
#if defined(DEBUG)
    cout << "end get context" << endl;
#endif
}

static void readFile(const char * filename)
{
#if defined(DEBUG)
    cout << "start read file" << endl;
#endif
    ifstream ifs;
    errorMessage = "read file failed";
    ifs.open(filename, fstream::in | fstream::binary);
    if (ifs) {
        ifs.seekg(0, std::fstream::end);
        ::size_t size = (::size_t)ifs.tellg();
        ifs.seekg(0, std::fstream::beg);
	char * buf = new char[size + 1];
	ifs.read(buf, size);
        ifs.close();
        buf[size] = '\0';
	programBuffer = buf;
	delete[] buf;
    }
#if defined(DEBUG) && 0
    cout << "source:" << endl;
    cout << programBuffer << endl;
#endif
    errorMessage = "create sources failed";
    cl::Program::Sources local_source(1,
				      make_pair(programBuffer.c_str(),
						programBuffer.size()));
    source = local_source;
    errorMessage = "";
#if defined(DEBUG)
    cout << "end read file" << endl;
#endif
}

static void getProgram()
{
#if defined(DEBUG)
    cout << "start get program" << endl;
#endif
    cl_int err;
    errorMessage = "create program failed";
    Program local_program = Program(context, source, &err);
    if (err != CL_SUCCESS) {
	cout << "get program err:" << err << endl;
	return;
    }
#if defined(DEBUG)
    cout << "start build" << endl;
#endif
    const char * option = "";
    errorMessage = "program build failed";
    try {
	err = local_program.build(devices, option);
    } catch (cl::Error e) {
	if (e.err() != CL_SUCCESS) {
	    if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
		std::string str
		    = local_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
		cout << "compile error:" << endl;
		cout << str << endl;
	    } else {
		cout << "build error but not program failure err:"
		     << dec << e.err()
		     << " " << e.what() << endl;
	    }
	}
	throw e;
    }
    program = local_program;
    errorMessage = "";
#if defined(DEBUG)
    cout << "end get program" << endl;
#endif
}

static void getCommandQueue()
{
#if defined(DEBUG)
    cout << "start get command queue" << endl;
#endif
    cl_int err;
    errorMessage = "create command queue failed";
    CommandQueue local_queue(context,
			     devices[0],
			     CL_QUEUE_PROFILING_ENABLE,
			     &err);
    if (err != CL_SUCCESS) {
	cout << "command queue create failure err:"
	     << dec << err << endl;
	return;
    }
    queue = local_queue;
    errorMessage = "";
#if defined(DEBUG)
    cout << "end get command queue" << endl;
#endif
}

static int getMaxGroupSize()
{
#if defined(DEBUG)
    cout << "start get max group size" << endl;
#endif
    ::size_t size;
    cl_int err;
    errorMessage = "device getinfo(CL_DEVICE_MAX_WORK_GROUP_SIZE) failed";
    err = devices[0].getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);
    if (err != CL_SUCCESS) {
	cout << "device getinfo(CL_DEVICE_MAX_WORK_GROUP_SIZE) err:"
	     << dec << err << endl;
    }
    errorMessage = "";
#if defined(DEBUG)
    cout << "end get max group size" << endl;
#endif
    return size;
}

static cl_ulong getLocalMemSize()
{
#if defined(DEBUG)
    cout << "start get local mem size" << endl;
#endif
    cl_int err;
    cl_ulong size;
    errorMessage = "device getinfo(CL_DEVICE_LOCAL_MEM_SIZE) failed";
    err = devices[0].getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
    if (err != CL_SUCCESS) {
	cout << "device getinfo(CL_DEVICE_LOCAL_MEM_SIZE) err:"
	     << dec << err << endl;
    }
    errorMessage = "";
#if defined(DEBUG)
    cout << "end get local mem size" << endl;
#endif
    return size;
}

static int getMaxWorkItemSize(int dim)
{
#if defined(DEBUG)
    cout << "start get max work item size" << endl;
#endif
    std::vector<std::size_t> vec;
    cl_int err;
    errorMessage = "device getinfo(CL_DEVICE_MAX_WORK_ITEM_SIZE) failed";
    err = devices[0].getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &vec);
    if (err != CL_SUCCESS) {
	cout << "device getinfo(CL_DEVICE_MAX_WORK_ITEM_SIZES) err:"
	     << dec << err << endl;
    }
    errorMessage = "";
#if defined(DEBUG)
    cout << "end get max work item size :" << dec << vec[dim] << endl;
#endif
    return vec[dim];
}


/* ========================= */
/* mtgp32 sample code        */
/* ========================= */

/**
 * prepare jump polynomial.
 *
 * this step may be pre-computed in practical use.
 */
static void make_jump_table(int group_num)
{
#if defined(DEBUG)
    cout << "make_jump_table start" << endl;
#endif
    mtgp32_fast_t dummy;
    int rc = mtgp32_init(&dummy, &mtgp32_params_fast_11213[0], 1);
    if (rc) {
	cerr << "init error" << endl;
	throw cl::Error(rc, "mtgp32 init error");
    }
    GF2X poly;
    clock_t start = clock();
    calc_characteristic(poly, &dummy);
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    cout << "calc_characteristic: " << dec << time << "ms" << endl;
    ZZ step;
    step = jump_step;
    start = clock();
    for (int i = 0; i < MAX_JUMP_TABLE; i++) {
	calc_jump(&jump_initial[i * MTGP32_N],
		  MTGP32_N,
		  step,
		  poly);
	step *= 4;
    }
    step = jump_step;
    step *= group_num - 1;
    calc_jump(jump_poly, MTGP32_N, step, poly);
    end = clock();
    time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    cout << "make jump table: " << dec << time << "ms" << endl;
#if defined(DEBUG)
    cout << "step:" << dec << step << endl;
    cout << "jump_poly[0]:" << hex << jump_poly[0] << endl;
    cout << "jump_poly[1]:" << hex << jump_poly[1] << endl;
    cout << "jump_initial[0]:" << hex << jump_initial[0 * MTGP32_N] << endl;
    cout << "jump_initial[1]:" << hex << jump_initial[1 * MTGP32_N] << endl;
    cout << "jump_initial[2]:" << hex << jump_initial[2 * MTGP32_N] << endl;
#endif
#if defined(DEBUG)
    cout << "make_jump_table end" << endl;
#endif
}

static int init_check_data(mtgp32_fast_t * mtgp32,
			   uint32_t seed)
{
#if defined(DEBUG)
    cout << "init_check_data start" << endl;
#endif
    int rc = mtgp32_init(mtgp32,
			 &mtgp32_params_fast_11213[0],
			 seed);
    if (rc) {
	return rc;
    }
#if defined(DEBUG)
    cout << "init_check_data end" << endl;
#endif
    return 0;
}

static int init_check_data_array(mtgp32_fast_t * mtgp32,
				 uint32_t seed_array[],
				 int size)
{
#if defined(DEBUG)
    cout << "init_check_data_array start" << endl;
#endif
    int rc = mtgp32_init_by_array(mtgp32,
				  &mtgp32_params_fast_11213[0],
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

static void free_check_data(mtgp32_fast_t * mtgp32)
{
#if defined(DEBUG)
    cout << "free_check_data start" << endl;
#endif
    mtgp32_free(mtgp32);
#if defined(DEBUG)
    cout << "free_check_data end" << endl;
#endif
}

static void check_data(uint32_t * h_data, int num_data)
{
#if defined(DEBUG)
    cout << "check_data start" << endl;
#endif
    bool error = false;
    bool disp_flg = true;
    int count = 0;
    for (int j = 0; j < num_data; j++) {
	uint32_t r = mtgp32_genrand_uint32(&mtgp32);
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

static void check_single(float * h_data, int num_data)
{
#if defined(DEBUG)
    cout << "check_single start" << endl;
#endif
    bool error = false;
    bool disp_flg = true;
    int count = 0;
    for (int j = 0; j < num_data; j++) {
	float r =  mtgp32_genrand_close1_open2(&mtgp32);
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
	cout << "check_single check O.K!" << endl;
    }
#if defined(DEBUG)
    cout << "check_single end" << endl;
#endif
}

double get_time(Event& event)
{
    event.wait();
    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    return (end - start) * 1.0e-9;
}

void initialize_by_seed(options& opt,
			Buffer& status_buffer,
			int group,
			uint32_t seed)
{
#if defined(DEBUG)
    cout << "initialize_by_seed start" << endl;
#endif
    // jump table
    Buffer jump_table_buffer(context,
			     CL_MEM_READ_WRITE,
			     MTGP32_N * MAX_JUMP_TABLE * sizeof(uint32_t));
    queue.enqueueWriteBuffer(jump_table_buffer,
			     CL_TRUE,
			     0,
			     MTGP32_N * MAX_JUMP_TABLE * sizeof(uint32_t),
			     jump_initial);

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
    cout << "initializing time = " << time * 1000 << "ms" << endl;
#if 0
    uint status[group * MTGP32_N];
    queue.enqueueReadBuffer(status_buffer,
			    CL_TRUE,
			    0,
			    sizeof(uint32_t) * MTGP32_N * group,
			    status);
#if defined(DEBUG)
    cout << "status[0]:" << hex << status[0] << endl;
    cout << "status[MTGP32_N - 1]:" << hex << status[MTGP32_N - 1] << endl;
    cout << "status[MTGP32_N]:" << hex << status[MTGP32_N] << endl;
    cout << "status[MTGP32_N + 1]:" << hex << status[MTGP32_N + 1] << endl;
#endif
    check_status(status, group);
#endif
#if defined(DEBUG)
    cout << "initialize_by_seed end" << endl;
#endif
}

void initialize_by_array(options& opt,
			 Buffer& status_buffer,
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
			     MTGP32_N * MAX_JUMP_TABLE * sizeof(uint32_t));
    queue.enqueueWriteBuffer(jump_table_buffer,
			     CL_TRUE,
			     0,
			     MTGP32_N * MAX_JUMP_TABLE * sizeof(uint32_t),
			     jump_initial);

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
#if 0
    uint status[group * MTGP32_N];
    queue.enqueueReadBuffer(status_buffer,
			    CL_TRUE,
			    0,
			    sizeof(uint32_t) * MTGP32_N * group,
			    status);
    cout << "initializing time = " << time * 1000 << "ms" << endl;
    check_status(status, group);
#endif
#if defined(DEBUG)
    cout << "initialize_by_array end" << endl;
#endif
}

void status_jump(options& opt, Buffer& status_buffer, int group)
{
#if defined(DEBUG)
    cout << "jump start" << endl;
#endif
    // jump table
    Buffer jump_table_buffer(context,
			     CL_MEM_READ_WRITE,
			     MTGP32_N * sizeof(uint32_t));
    queue.enqueueWriteBuffer(jump_table_buffer,
			     CL_TRUE,
			     0,
			     MTGP32_N * sizeof(uint32_t),
			     jump_poly);

    Kernel init_kernel(program, "mtgp32_jump_kernel");
    init_kernel.setArg(0, status_buffer);
    init_kernel.setArg(1, jump_table_buffer);
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
    cout << "jump time = " << time * 1000 << "ms" << endl;
#if defined(DEBUG)
    cout << "jump end" << endl;
#endif
}

void print_uint32(uint32_t data[], int size, int item_num)
{
    int max_seq = 10;
    int max_item = 6;
    if (size / item_num < max_seq) {
	max_seq = size / item_num;
    }
    if (item_num < max_item) {
	max_item = item_num;
    }
    for (int i = 0; i < max_seq; i++) {
	for (int j = 0; j < max_item; j++) {
	    cout << setw(10) << dec << data[item_num * i + j] << " ";
	}
	cout << endl;
    }
}

void print_float(float data[], int size, int item_num)
{
    int max_seq = 10;
    int max_item = 6;
    if (size / item_num < max_seq) {
	max_seq = size / item_num;
    }
    if (item_num < max_item) {
	max_item = item_num;
    }
    for (int i = 0; i < max_seq; i++) {
	for (int j = 0; j < max_item; j++) {
	    cout << setprecision(9) << setw(10)
		 << dec << left << setfill('0')
		 << data[item_num * i + j] << " ";
	}
	cout << endl;
    }
}

void generate_uint32(int group_num,
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
    check_data(output, data_size);
    print_uint32(&output[0], data_size, item_num);
    double time = get_time(generate_event);
    cout << "generate time:" << time * 1000 << "ms" << endl;
    delete[] output;
#if defined(DEBUG)
    cout << "generate_uint32 end" << endl;
#endif
}

void generate_single(int group_num,
		     Buffer& tiny_buffer,
		     int data_size)
{
    int item_num = MTGP32_TN * group_num;
    int min_size = MTGP32_LS * group_num;
    if (data_size % min_size != 0) {
	data_size = (data_size / min_size + 1) * min_size;
    }
    Kernel single_kernel(program, "mtgp32_single_kernel");
    Buffer output_buffer(context,
			 CL_MEM_READ_WRITE,
			 data_size * sizeof(float));
    single_kernel.setArg(0, tiny_buffer);
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
    check_single(output, data_size);
    print_float(&output[0], data_size, item_num);
    double time = get_time(generate_event);
    delete[] output;
    cout << "generate time:" << time * 1000 << "ms" << endl;
}

int test(int argc, char * argv[]) {
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
    getPlatforms();
    getDevices();
    getContext();
#if defined(APPLE) || defined(__MACOSX) || defined(__APPLE__)
    readFile("mtgp32-jump.cli");
#else
    readFile("mtgp32-jump.cl");
#endif
    getProgram();
    getCommandQueue();
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
	cout << "local memory size is smaller than min value("
	     << dec << sizeof(uint32_t) * MTGP32_N * 2
	     << ") current:"
	     << dec << local_mem_size << endl;
	return -1;
    }
    Buffer status_buffer(context,
			 CL_MEM_READ_WRITE,
			 sizeof(uint32_t) * MTGP32_N * opt.group_num);

    make_jump_table(opt.group_num);
    int data_count = opt.data_count;
    int data_unit = jump_step * opt.group_num;

    // initialize by seed
    // generate uint32_t
    init_check_data(&mtgp32, 1234);
    initialize_by_seed(opt, status_buffer, opt.group_num, 1234);
    while (data_count > 0) {
	generate_uint32(opt.group_num, status_buffer, data_unit);
	status_jump(opt, status_buffer, opt.group_num);
	data_count -= data_unit;
    }
    free_check_data(&mtgp32);

    // initialize by array
    // generate single float
    uint32_t seed_array[5] = {1, 2, 3, 4, 5};
    init_check_data_array(&mtgp32, seed_array, 5);
    initialize_by_array(opt, status_buffer, opt.group_num,
			seed_array, 5);
    data_count = opt.data_count;
    while (data_count > 0) {
	generate_single(opt.group_num, status_buffer, data_unit);
	status_jump(opt, status_buffer, opt.group_num);
	data_count -= data_unit;
    }
    free_check_data(&mtgp32);
    return 0;
}

int main(int argc, char * argv[]) {
    try {
	return test(argc, argv);
    } catch (cl::Error e) {
	cerr << "Error Code:" << e.err() << endl;
	cerr << errorMessage << endl;
	cerr << e.what() << endl;
    }
}
