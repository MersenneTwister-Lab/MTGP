#ifndef MTGP_CALC_JUMP_HPP
#define MTGP_CALC_JUMP_HPP
/**
 * @file mtgp-calc-jump.hpp
 *
 * @brief functions for calculating jump polynomial.
 *
 * calculate jump polynomial from jump step and the
 * characteristic polynomial using PowerXMod.<br/>
 * jump polynomial p is calculated<br/>
 *     p = X<sup>s</sup> mod q,<br/>
 * where s is jump step and q is characteristic polynomial.
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

/**
 * convert polynomial to string
 * @param[out] x output string
 * @param[in] polynomial polynomial
 */
static inline void polytostring(std::string& x, NTL::GF2X& polynomial)
{
    using namespace NTL;
    using namespace std;

    long degree = deg(polynomial);
    int buff = 0;
    int index = 0;
    stringstream ss;
    for (int i = 0; i <= degree; i++) {
	if (index < i / 4) {
	    ss << hex << buff;
	    buff = 0;
	    index = i / 4;
	}
	if (IsOne(coeff(polynomial, i))) {
	    buff |= 1 << (i % 4);
	}
    }
    if (buff != 0) {
	ss << hex << buff;
    }
    ss << flush;
    x = ss.str();
}

/**
 * Convert polynomial to 32-bit unsigned array.
 * Resulted array is used in kernel functions of cuda and OpenCL.
 * @param[out] array array format polynomial
 * @param[in] size max size of array
 * @param polynomial polynomial
 */
static inline void polytoarray(uint32_t array[],
			       int size,
			       NTL::GF2X& polynomial)
{
    using namespace NTL;
    using namespace std;

    long degree = deg(polynomial);
    if (size == 0) {
	return;
    } else if (size < degree / 32 + 1) {
	array[0] = 0;
	return;
    }
    for (int i = 0; i < size; i++) {
	array[i] = 0;
    }
    for (int i = 0; i <= degree; i++) {
	int index = i / 32;
	int pos = i % 32;
	if (IsOne(coeff(polynomial, i))) {
	    array[index] |= 1 << pos;
	}
    }
}

/**
 * converts string to polynomial
 * @param[in] str string
 * @param[out] poly output polynomial
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
 * @param[out] array output string which represents jump polynomial.
 * @param[in] size max size of array
 * @param[in] step jump step of internal state
 * @param[in] characteristic characteristic polynomial
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
 * @param[out] jump_str output string which represents jump polynomial.
 * @param[in] step jump step of internal state
 * @param[in] characteristic characteristic polynomial
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
