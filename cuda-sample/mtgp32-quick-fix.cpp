/*
 * I can't use nvcc with NTL.h in my environment.
 *
 */
#include "mtgp32-quick-fix.hpp"
#include "mtgp32-calc-poly.hpp"
#include "mtgp-calc-jump.hpp"

using namespace NTL;
using namespace std;

void qf_calc_characteristic(string& poly, mtgp32_fast_t * mtgp32)
{
    calc_characteristic(poly, mtgp32);
}

/**
 * calculate the jump polynomial.
 * SFMT generates 4 32-bit integers from one internal state.
 * @param jump_str output string which represents jump polynomial.
 * @param step jump step of internal state
 * @param characteristic polynomial
 */
void qf_calc_jump(uint32_t jump_array[],
		  int size,
		  long jump_step,
		  string& poly)
{
    GF2X characteristic;
    GF2X jump;
    ZZ step;
    step = jump_step;
    stringtopoly(characteristic, poly);
    calc_jump(jump_array, size, step, characteristic);
}
