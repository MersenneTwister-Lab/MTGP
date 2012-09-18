#ifndef MTGP32_SAMPLE_JUMP2_KERNEL_CU
#define MTGP32_SAMPLE_JUMP2_KERNEL_CU
/**
 * @file mtgp32-jump-table.cuh
 *
 * @brief data for long jump.
 *
 * mtgp32_jump_table[0] jump    1 x 3<sup>162</sup> steps
 * mtgp32_jump_table[1] jump    4 x 3<sup>162</sup> steps
 * mtgp32_jump_table[2] jump   16 x 3<sup>162</sup> steps
 * mtgp32_jump_table[3] jump   64 x 3<sup>162</sup> steps
 * mtgp32_jump_table[4] jump  256 x 3<sup>162</sup> steps
 * mtgp32_jump_table[5] jump 1024 x 3<sup>162</sup> steps
 */

/**
 * JUMP_STEP calculated from global memory size
 */
__constant__ uint32_t mtgp32_sample_jump2[MTGP32_N+1];

/**
 * mtgp32_jump_table[0] jump    1 x JUMP_STEP steps
 * mtgp32_jump_table[1] jump    4 x JUMP_STEP steps
 * mtgp32_jump_table[2] jump   16 x JUMP_STEP steps
 * mtgp32_jump_table[3] jump   64 x JUMP_STEP steps
 * mtgp32_jump_table[4] jump  256 x JUMP_STEP steps
 * mtgp32_jump_table[5] jump 1024 x JUMP_STEP steps
 */
__constant__ uint32_t mtgp32_sample_jump2_table[6][MTGP32_N+1];

#endif
