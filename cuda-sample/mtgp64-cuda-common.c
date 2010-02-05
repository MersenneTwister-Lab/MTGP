/*
 * Sample Program for CUDA 2.3
 * written by M.Saito (saito@math.sci.hiroshima-u.ac.jp)
 *
 * This sample uses texture reference.
 * The generation speed of PRNG using texture is faster than using
 * constant tabel on Geforce GTX 260.
 *
 */

/**
 * This function initializes kernel I/O data.
 * @param d_status output kernel I/O data.
 * @param params MTGP64 parameters. needed for the initialization.
 */
void make_kernel_data(mtgp64_kernel_status_t *d_status,
		      mtgp64_params_fast_t params[],
		      int block_num) {
    mtgp64_kernel_status_t* h_status = (mtgp64_kernel_status_t *) malloc(
	sizeof(mtgp64_kernel_status_t) * block_num);

    if (h_status == NULL) {
	printf("failure in allocating host memory for kernel I/O data.\n");
	exit(8);
    }
    for (int i = 0; i < block_num; i++) {
	mtgp64_init_state(&(h_status[i].status[0]), &params[i], i + 1);
    }
#if defined(DEBUG)
    printf("h_status[0].status[0]:%016"PRIx64"\n", h_status[0].status[0]);
    printf("h_status[0].status[0]:%016"PRIx64"\n", h_status[0].status[1]);
    printf("h_status[0].status[0]:%016"PRIx64"\n", h_status[0].status[2]);
    printf("h_status[0].status[0]:%016"PRIx64"\n", h_status[0].status[3]);
#endif
    CUDA_SAFE_CALL(cudaMemcpy(d_status,
			      h_status,
			      sizeof(mtgp64_kernel_status_t) * block_num,
			      cudaMemcpyHostToDevice));
    free(h_status);
}

/**
 * This function is used to compare the outputs with C program's.
 * @param array data to be printed.
 * @param size size of array.
 * @param block number of blocks.
 */
void print_double_array(const double array[], int size, int block) {
    int b = size / block;

    for (int j = 0; j < 3; j += 3) {
	printf("%.18f %.18f %.18f\n",
	       array[j], array[j + 1], array[j + 2]);
    }
    for (int i = 1; i < block; i++) {
	for (int j = -3; j < 4; j += 3) {
	    printf("%.18f %.18f %.18f\n",
		   array[b * i + j],
		   array[b * i + j + 1],
		   array[b * i + j + 2]);
	}
    }
    for (int j = -3; j < 0; j += 3) {
	printf("%.18f %.18f %.18f\n",
	       array[size + j],
	       array[size + j + 1],
	       array[size + j + 2]);
    }
}

/**
 * This function is used to compare the outputs with C program's.
 * @param array data to be printed.
 * @param size size of array.
 * @param block number of blocks.
 */
void print_uint64_array(uint64_t array[], int size, int block) {
    int b = size / block;

    for (int j = 0; j < 3; j += 3) {
	printf("%20" PRIu64 " %20" PRIu64 " %20" PRIu64 "\n",
	       array[j], array[j + 1], array[j + 2]);
    }
    for (int i = 1; i < block; i++) {
	for (int j = -3; j < 3; j += 3) {
	    printf("%20" PRIu64 " %20" PRIu64 " %20" PRIu64 "\n",
		   array[b * i + j],
		   array[b * i + j + 1],
		   array[b * i + j + 2]);
	}
    }
    for (int j = -3; j < 0; j += 3) {
	printf("%20" PRIu64 " %20" PRIu64 " %20" PRIu64 "\n",
	       array[size + j],
	       array[size + j + 1],
	       array[size + j + 2]);
    }
}
