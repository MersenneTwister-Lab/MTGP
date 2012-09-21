/**
 * Sample host program for
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

typedef uint32_t uint;
#include "mtgp32-fast.h"
#include "mtgp32-sample-common.h"
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

static mtgp32_fast_t * mtgp32;
static bool thread_max = false;
struct buffers_t {
    Buffer status;
    Buffer rec;
    Buffer tmp;
    Buffer flt;
    Buffer pos;
    Buffer sh1;
    Buffer sh2;
};

/* ========================= */
/* mtgp32 sample code        */
/* ========================= */
static int init_check_data(mtgp32_fast_t mtgp32[],
			   int group_num,
			   uint32_t seed)
{
#if defined(DEBUG)
    cout << "init_check_data start" << endl;
#endif
    for (int i = 0; i < group_num; i++) {
	int rc = mtgp32_init(&mtgp32[i],
			     &mtgp32_params_fast_11213[i],
			     seed + i);
	if (rc) {
	    return rc;
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
    for (int i = 0; i < group_num; i++) {
	int rc = mtgp32_init_by_array(&mtgp32[i],
				      &mtgp32_params_fast_11213[i],
				      seed_array,
				      size);
	if (rc) {
	    return rc;
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

static void check_status(uint * h_status,
			 int group_num)
{
#if defined(DEBUG)
    cout << "check_status start" << endl;
#endif
    int counter = 0;
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

static Buffer get_rec_buff(mtgp32_params_fast_t * params,
			   int group_num) {
#if defined(DEBUG)
    cout << "get_rec_buff start" << endl;
#endif
    // recursion table
    uint32_t * rec_tbl = new uint32_t[MTGP32_TS * group_num];
    for (int i = 0; i < group_num; i++) {
	for (int j = 0; j < MTGP32_TS; j++) {
	    rec_tbl[i * MTGP32_TS + j] = params[i].tbl[j];
	}
    }
    Buffer recursion_buffer(context,
			    CL_MEM_READ_ONLY,
			    MTGP32_TS * group_num * sizeof(uint32_t));
    queue.enqueueWriteBuffer(recursion_buffer,
			     CL_TRUE,
			     0,
			     MTGP32_TS * group_num * sizeof(uint32_t),
			     rec_tbl);
    delete[] rec_tbl;
#if defined(DEBUG)
    cout << "get_rec_buff end" << endl;
#endif
    return recursion_buffer;
}

static Buffer get_tmp_buff(mtgp32_params_fast_t * params,
			   int group_num) {
#if defined(DEBUG)
    cout << "get_tmp_buff start" << endl;
#endif
    // temper table
    uint32_t * tmp_tbl = new uint32_t[MTGP32_TS * group_num];
    for (int i = 0; i < group_num; i++) {
	for (int j = 0; j < MTGP32_TS; j++) {
	    tmp_tbl[i * MTGP32_TS + j] = params[i].tmp_tbl[j];
	}
    }
    Buffer temper_buffer(context,
			 CL_MEM_READ_ONLY,
			 MTGP32_TS * group_num * sizeof(uint32_t));
    queue.enqueueWriteBuffer(temper_buffer,
			     CL_TRUE,
			     0,
			     MTGP32_TS * group_num * sizeof(uint32_t),
			     tmp_tbl);
    delete[] tmp_tbl;
#if defined(DEBUG)
    cout << "get_tmp_buff end" << endl;
#endif
    return temper_buffer;
}

static Buffer get_flt_tmp_buff(mtgp32_params_fast_t * params,
			       int group_num) {
#if defined(DEBUG)
    cout << "get_flt_tmp_buff start" << endl;
#endif
    // temper table
    uint32_t * tmp_tbl = new uint32_t[MTGP32_TS * group_num];
    for (int i = 0; i < group_num; i++) {
	for (int j = 0; j < MTGP32_TS; j++) {
	    tmp_tbl[i * MTGP32_TS + j] = params[i].flt_tmp_tbl[j];
	}
    }
    Buffer temper_buffer(context,
			 CL_MEM_READ_ONLY,
			 MTGP32_TS * group_num * sizeof(uint32_t));
    queue.enqueueWriteBuffer(temper_buffer,
			     CL_TRUE,
			     0,
			     MTGP32_TS * group_num * sizeof(uint32_t),
			     tmp_tbl);
    delete[] tmp_tbl;
#if defined(DEBUG)
    cout << "get_flt_tmp_buff end" << endl;
#endif
    return temper_buffer;
}

static Buffer get_pos_buff(mtgp32_params_fast_t * params,
			   int group_num) {
#if defined(DEBUG)
    cout << "get_pos_buff start" << endl;
#endif
    // temper table
    uint32_t * pos_tbl = new uint32_t[group_num];
    for (int i = 0; i < group_num; i++) {
	pos_tbl[i] = params[i].pos;
    }
    Buffer pos_buffer(context,
		      CL_MEM_READ_ONLY,
		      group_num * sizeof(uint32_t));
    queue.enqueueWriteBuffer(pos_buffer,
			     CL_TRUE,
			     0,
			     group_num * sizeof(uint32_t),
			     pos_tbl);
    delete[] pos_tbl;
#if defined(DEBUG)
    cout << "get_pos_buff end" << endl;
#endif
    return pos_buffer;
}

static Buffer get_sh1_buff(mtgp32_params_fast_t * params,
			   int group_num) {
#if defined(DEBUG)
    cout << "get_sh1_buff start" << endl;
#endif
    // temper table
    uint32_t * sh1_tbl = new uint32_t[group_num];
    for (int i = 0; i < group_num; i++) {
	sh1_tbl[i] = params[i].sh1;
    }
    Buffer sh1_buffer(context,
		      CL_MEM_READ_ONLY,
		      group_num * sizeof(uint32_t));
    queue.enqueueWriteBuffer(sh1_buffer,
			     CL_TRUE,
			     0,
			     group_num * sizeof(uint32_t),
			     sh1_tbl);
    delete[] sh1_tbl;
#if defined(DEBUG)
    cout << "get_sh1_buff end" << endl;
#endif
    return sh1_buffer;
}

static Buffer get_sh2_buff(mtgp32_params_fast_t * params,
			   int group_num) {
#if defined(DEBUG)
    cout << "get_sh2_buff start" << endl;
#endif
    // temper table
    uint32_t * sh2_tbl = new uint32_t[group_num];
    for (int i = 0; i < group_num; i++) {
	sh2_tbl[i] = params[i].sh2;
    }
    Buffer sh2_buffer(context,
		      CL_MEM_READ_ONLY,
		      group_num * sizeof(uint32_t));
    queue.enqueueWriteBuffer(sh2_buffer,
			     CL_TRUE,
			     0,
			     group_num * sizeof(uint32_t),
			     sh2_tbl);
    delete[] sh2_tbl;
#if defined(DEBUG)
    cout << "get_sh2_buff end" << endl;
#endif
    return sh2_buffer;
}

void initialize_by_seed(options& opt,
			buffers_t mtgp_buffers,
			int group,
			uint32_t seed)
{
#if defined(DEBUG)
    cout << "initialize_by_seed start" << endl;
#endif
    Kernel init_kernel(program, "mtgp32_init_seed_kernel");
    init_kernel.setArg(0, mtgp_buffers.rec);
    init_kernel.setArg(1, mtgp_buffers.tmp);
    init_kernel.setArg(2, mtgp_buffers.flt);
    init_kernel.setArg(3, mtgp_buffers.pos);
    init_kernel.setArg(4, mtgp_buffers.sh1);
    init_kernel.setArg(5, mtgp_buffers.sh2);
    init_kernel.setArg(6, mtgp_buffers.status);
    init_kernel.setArg(7, seed);
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
    queue.enqueueReadBuffer(mtgp_buffers.status,
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

void initialize_by_array(options& opt,
			 buffers_t& mtgp_buffers,
			 int group,
			 uint32_t seed_array[],
			 int seed_size)
{
#if defined(DEBUG)
    cout << "initialize_by_array start" << endl;
#endif
    Buffer seed_array_buffer(context,
			     CL_MEM_READ_WRITE,
			     seed_size * sizeof(uint32_t));
    queue.enqueueWriteBuffer(seed_array_buffer,
			     CL_TRUE,
			     0,
			     seed_size * sizeof(uint32_t),
			     seed_array);
    Kernel init_kernel(program, "mtgp32_init_array_kernel");
    init_kernel.setArg(0, mtgp_buffers.rec);
    init_kernel.setArg(1, mtgp_buffers.tmp);
    init_kernel.setArg(2, mtgp_buffers.flt);
    init_kernel.setArg(3, mtgp_buffers.pos);
    init_kernel.setArg(4, mtgp_buffers.sh1);
    init_kernel.setArg(5, mtgp_buffers.sh2);
    init_kernel.setArg(6, mtgp_buffers.status);
    init_kernel.setArg(7, seed_array_buffer);
    init_kernel.setArg(8, seed_size);
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
    queue.enqueueReadBuffer(mtgp_buffers.status,
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

void generate_uint32(int group_num,
		     buffers_t& mtgp_buffers,
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
    uint_kernel.setArg(0, mtgp_buffers.rec);
    uint_kernel.setArg(1, mtgp_buffers.tmp);
    uint_kernel.setArg(2, mtgp_buffers.flt);
    uint_kernel.setArg(3, mtgp_buffers.pos);
    uint_kernel.setArg(4, mtgp_buffers.sh1);
    uint_kernel.setArg(5, mtgp_buffers.sh2);
    uint_kernel.setArg(6, mtgp_buffers.status);
    uint_kernel.setArg(7, output_buffer);
    uint_kernel.setArg(8, data_size / group_num);
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
    uint32_t * output = new uint32_t[data_size];
    generate_event.wait();
    queue.enqueueReadBuffer(output_buffer,
			    CL_TRUE,
			    0,
			    data_size * sizeof(uint32_t),
			    output);
    check_data(output, data_size, group_num);
    print_uint32(output, data_size, item_num);
    double time = get_time(generate_event);
    cout << "generate time:" << time * 1000 << "ms" << endl;
    delete[] output;
#if defined(DEBUG)
    cout << "generate_uint32 end" << endl;
#endif
}

void generate_single(int group_num,
		     buffers_t& mtgp_buffers,
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
    single_kernel.setArg(0, mtgp_buffers.rec);
    single_kernel.setArg(1, mtgp_buffers.tmp);
    single_kernel.setArg(2, mtgp_buffers.flt);
    single_kernel.setArg(3, mtgp_buffers.pos);
    single_kernel.setArg(4, mtgp_buffers.sh1);
    single_kernel.setArg(5, mtgp_buffers.sh2);
    single_kernel.setArg(6, mtgp_buffers.status);
    single_kernel.setArg(7, output_buffer);
    single_kernel.setArg(8, data_size / group_num);
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
    platforms = getPlatforms();
    devices = getDevices();
    context = getContext();
#if defined(APPLE) || defined(__MACOSX) || defined(__APPLE__)
    source = getSource("mtgp32.cli");
#else
    source = getSource("mtgp32.cl");
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
    buffers_t mtgp_buffers;
    mtgp_buffers.status = status_buffer;
    mtgp_buffers.rec = get_rec_buff(mtgp32_params_fast_11213, opt.group_num);
    mtgp_buffers.tmp = get_tmp_buff(mtgp32_params_fast_11213, opt.group_num);
    mtgp_buffers.flt = get_flt_tmp_buff(mtgp32_params_fast_11213,
					opt.group_num);
    mtgp_buffers.pos = get_pos_buff(mtgp32_params_fast_11213, opt.group_num);
    mtgp_buffers.sh1 = get_sh1_buff(mtgp32_params_fast_11213, opt.group_num);
    mtgp_buffers.sh2 = get_sh2_buff(mtgp32_params_fast_11213, opt.group_num);
    // initialize by seed
    // generate uint32_t
    mtgp32 = new mtgp32_fast_t[opt.group_num];
    init_check_data(mtgp32, opt.group_num, 1234);
    initialize_by_seed(opt, mtgp_buffers, opt.group_num, 1234);
    for (int i = 0; i < 2; i++) {
	generate_uint32(opt.group_num, mtgp_buffers, opt.data_count);
    }
    free_check_data(mtgp32, opt.group_num);

    // initialize by array
    // generate single float
    uint32_t seed_array[5] = {1, 2, 3, 4, 5};
    init_check_data_array(mtgp32, opt.group_num, seed_array, 5);
    initialize_by_array(opt, mtgp_buffers, opt.group_num,
			seed_array, 5);
    for (int i = 0; i < 2; i++) {
	generate_single(opt.group_num, mtgp_buffers, opt.data_count);
    }
    free_check_data(mtgp32, opt.group_num);
    delete[] mtgp32;
    return 0;
}

int main(int argc, char * argv[]) {
    try {
	return test(argc, argv);
    } catch (Error e) {
	cerr << "Error Code:" << e.err() << endl;
	cerr << e.what() << endl;
    }
}
