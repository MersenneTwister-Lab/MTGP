#ifndef MTGP32_QUICK_FIX_HPP
#define MTGP32_QUICK_FIX_HPP
#include <string>
#include "mtgp32-fast.h"

void qf_calc_characteristic(std::string& poly, mtgp32_fast_t * mtgp32);
void qf_calc_jump(uint32_t h_large_jump_array[],
		  int size,
		  long large_jump_step,
		  std::string& poly);
#endif
