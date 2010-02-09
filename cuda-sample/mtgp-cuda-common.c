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

    CUDA_SAFE_CALL(cudaGetDeviceProperties(&dev, 0));
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
