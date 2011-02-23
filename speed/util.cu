#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>
#include <util.h>
#include <stdlib.h>
#include <cuda.h>


/**
 * This function initializes kernel I/O data.
 * @param d_status output kernel I/O data.
 * @param params MTGP32 parameters. needed for the initialization.
 */
void make_kernel_data(mtgp32_kernel_status_t * d_status,
		      mtgp32_params_fast_t params[],
		      int block_num) {
    int i;
    mtgp32_kernel_status_t* h_status = (mtgp32_kernel_status_t *) malloc(
	sizeof(mtgp32_kernel_status_t) * block_num);

    if (h_status == NULL) {
	printf("failure in allocating host memory for kernel I/O data.\n");
	exit(8);
    }
    for (i = 0; i < block_num; i++) {
	mtgp32_init_state(&(h_status[i].status[0]), &params[i], i + 1);
    }
#if defined(DEBUG)
    printf("h_status[0].status[0]:%08"PRIx32"\n", h_status[0].status[0]);
    printf("h_status[0].status[1]:%08"PRIx32"\n", h_status[0].status[1]);
    printf("h_status[0].status[2]:%08"PRIx32"\n", h_status[0].status[2]);
    printf("h_status[0].status[3]:%08"PRIx32"\n", h_status[0].status[3]);
#endif
    cudaMemcpy(d_status, h_status,
	       sizeof(mtgp32_kernel_status_t) * block_num,
	       cudaMemcpyHostToDevice);
    free(h_status);
}

/**
 * This function is used to compare the outputs with C program's.
 * @param array data to be printed.
 * @param size size of array.
 * @param block number of blocks.
 */
void print_float_array(const float array[], int size, int block) {
    int b = size / block;
    int i;
    int j;

    for (j = 0; j < 5; j += 5) {
	printf("%.10f %.10f %.10f %.10f %.10f\n",
	       array[j], array[j + 1],
	       array[j + 2], array[j + 3], array[j + 4]);
    }
    for (i = 1; i < block; i++) {
	for (j = -5; j < 5; j += 5) {
	    printf("%.10f %.10f %.10f %.10f %.10f\n",
		   array[b * i + j],
		   array[b * i + j + 1],
		   array[b * i + j + 2],
		   array[b * i + j + 3],
		   array[b * i + j + 4]);
	}
    }
    for (j = -5; j < 0; j += 5) {
	printf("%.10f %.10f %.10f %.10f %.10f\n",
	       array[size + j],
	       array[size + j + 1],
	       array[size + j + 2],
	       array[size + j + 3],
	       array[size + j + 4]);
    }
}

/**
 * This function is used to compare the outputs with C program's.
 * @param array data to be printed.
 * @param size size of array.
 * @param block number of blocks.
 */
void print_uint32_array(const uint32_t array[], int size, int block) {
    int b = size / block;
    int i;
    int j;

    for (j = 0; j < 5; j += 5) {
	printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32 " %10"
	       PRIu32 " %10" PRIu32 "\n",
	       array[j], array[j + 1],
	       array[j + 2], array[j + 3], array[j + 4]);
    }
    for (i = 1; i < block; i++) {
	for (j = -5; j < 5; j += 5) {
	    printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32 " %10"
		   PRIu32 " %10" PRIu32 "\n",
		   array[b * i + j],
		   array[b * i + j + 1],
		   array[b * i + j + 2],
		   array[b * i + j + 3],
		   array[b * i + j + 4]);
	}
    }
    for (j = -5; j < 0; j += 5) {
	printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32 " %10" PRIu32
	       " %10" PRIu32 "\n",
	       array[size + j],
	       array[size + j + 1],
	       array[size + j + 2],
	       array[size + j + 3],
	       array[size + j + 4]);
    }
}

/*
 * Sample Program for CUDA 2.3
 * written by M.Saito (saito@math.sci.hiroshima-u.ac.jp)
 *
 * This sample uses texture reference.
 * The generation speed of PRNG using texture is faster than using
 * constant tabel on Geforce GTX 260.
 *
 * MTGP32-11213
 * This program generates 32-bit unsigned integers.
 * The period of generated integers is 2<sup>23209</sup>-1.
 * This also generates single precision floating point numbers.
 */

int get_suitable_block_num(int word_size, int thread_num, int large_size) {
    cudaDeviceProp dev;
    CUdevice cuDevice;
    int max_thread_dev;
    int max_block, max_block_mem, max_block_dev;
    int major, minor, ver;

    cudaGetDeviceProperties(&dev, 0);
    cuDeviceGet(&cuDevice, 0);
    cuDeviceComputeCapability(&major, &minor, cuDevice);
    max_block_mem = dev.sharedMemPerBlock / (large_size * word_size);
    if (major == 9999 && minor == 9999) {
	return -1;
    }
    ver = major * 100 + minor;
    if (ver <= 101) {
	max_thread_dev = 768;
    } else if (ver <= 103) {
	max_thread_dev = 1024;
    } else {
	max_thread_dev = 1024;
    }
    max_block_dev = max_thread_dev / thread_num;
    if (max_block_mem < max_block_dev) {
	max_block = max_block_mem;
    } else {
	max_block = max_block_dev;
    }
    return max_block * dev.multiProcessorCount;
}

void print_max_min(uint32_t data[], int size) {
    uint32_t max = 0;
    uint32_t min = 0xffffffffU;
    for (int i = 0; i < size; i++) {
	if (data[i] > max) {
	    max = data[i];
	}
	if (data[i] < min) {
	    min = data[i];
	}
    }
    printf("max:%d min:%d\n", max, min);

}
