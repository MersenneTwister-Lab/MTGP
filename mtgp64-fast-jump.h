#pragma once
#ifndef MTGP64_FAST_JUMP_H
#define MTGP64_FAST_JUMP_H
/**
 * @file mtgp64_fast_jump.h
 *
 * @brief do jump using jump polynomial.
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

#include "mtgp32-fast.h"

#if defined(__cplusplus)
extern "C" {
#endif
    void mtgp64_fast_jump(mtgp64_fast_t * mtgp64, const char * jump_string);
#if defined(__cplusplus)
}
#endif /* C++ */

#endif /* include fence */
