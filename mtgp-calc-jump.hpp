#ifndef MTGP_CALC_JUMP_HPP
#define MTGP_CALC_JUMP_HPP
/**
 * @file calc_jump.hpp
 *
 * @brief functions for calculating jump polynomial.
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (The University of Tokyo)
 *
 * Copyright (C) 2012 Mutsuo Saito, Makoto Matsumoto,
 * Hiroshima University and The University of Tokyo.
 * All rights reserved.
 *
 * The 3-clause BSD License is applied to this software, see
 * LICENSE.txt
 */
#include <iostream>
#include <iomanip>
#include <sstream>
#include <NTL/GF2X.h>

static inline void polytostring(std::string& x, NTL::GF2X& polynomial)
{
    using namespace NTL;
    using namespace std;

    long degree = deg(polynomial);
    int buff;
    stringstream ss;
    for (int i = 0; i <= degree; i+=4) {
	buff = 0;
	for (int j = 0; j < 4; j++) {
	    if (IsOne(coeff(polynomial, i + j))) {
		buff |= 1 << j;
	    } else {
		buff &= (0x0f ^ (1 << j));
	    }
	}
	ss << hex << buff;
    }
    ss << flush;
    x = ss.str();
}

static inline void polytoarray(uint32_t array[],
			       int size,
			       NTL::GF2X& polynomial)
{
    using namespace NTL;
    using namespace std;

    long degree = deg(polynomial);
    uint32_t buff;
    int index = 0;
    if (size == 0) {
	return;
    } else if (size < degree / 32 + 2) {
	array[0] = 0;
	return;
    }
    for (int i = 0; i <= degree; i+=32) {
	buff = 0;
	for (int j = 0; j < 32; j++) {
	    if (IsOne(coeff(polynomial, i + j))) {
		buff |= 1 << j;
	    } else {
		buff &= (0xffffffffU ^ (1 << j));
	    }
	}
	array[index] = buff;
	index++;
    }
    array[index] = 0;
}

/**
 * converts string to polynomial
 * @param str string
 * @param poly output polynomial
 */
static inline void stringtopoly(NTL::GF2X& poly, std::string& str)
{
    using namespace NTL;
    using namespace std;

    stringstream ss(str);
    char c;
    long p = 0;
    clear(poly);
    while(ss) {
	ss >> c;
	if (!ss) {
	    break;
	}
	if (c >= 'a') {
	    c = c - 'a' + 10;
	} else {
	    c = c - '0';
	}
	for (int j = 0; j < 4; j++) {
	    if (c & (1 << j)) {
		SetCoeff(poly, p, 1);
	    } else {
		SetCoeff(poly, p, 0);
	    }
	    p++;
	}
    }
}

/**
 * calculate the jump polynomial.
 * SFMT generates 4 32-bit integers from one internal state.
 * @param jump_str output string which represents jump polynomial.
 * @param step jump step of internal state
 * @param characteristic polynomial
 */
static inline void calc_jump(uint32_t array[],
			     int size,
			     NTL::ZZ& step,
			     NTL::GF2X& characteristic)
{
    using namespace NTL;
    using namespace std;
    GF2X jump;
    PowerXMod(jump, step, characteristic);
    polytoarray(array, size, jump);
}

/**
 * calculate the jump polynomial.
 * SFMT generates 4 32-bit integers from one internal state.
 * @param jump_str output string which represents jump polynomial.
 * @param step jump step of internal state
 * @param characteristic polynomial
 */
static inline void calc_jump(std::string& jump_str,
			     NTL::ZZ& step,
			     NTL::GF2X& characteristic)
{
    using namespace NTL;
    using namespace std;
    GF2X jump;
    PowerXMod(jump, step, characteristic);
    polytostring(jump_str, jump);
}

#endif
