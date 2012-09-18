#ifndef MTGP32_JUMP_TABLE_H
#define MTGP32_JUMP_TABLE_H
/**
 * @file mtgp32-jump-table.h
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
#include <stdint.h>
extern const uint32_t mtgp32_jump_table[];

#endif
