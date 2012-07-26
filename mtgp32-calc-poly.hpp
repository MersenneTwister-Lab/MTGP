#ifndef MTGP32_CALC_POLY_HPP
#define MTGP32_CALC_POLY_HPP
/**
 * @file mtgp32_calc_poly.hpp
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
#include <string>
#include <NTL/GF2X.h>
#include "mtgp32-fast.h"

void calc_characteristic(NTL::GF2X& poly, mtgp32_fast_t * mtgp32);
void calc_characteristic(std::string& str, mtgp32_fast_t * mtgp32);

#endif
