/**
 * @file mtgp32_fast_jump.c
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

#include <assert.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "mtgp32-fast.h"
#include "mtgp32-fast-jump.h"

#if defined(__cplusplus)
extern "C" {
#endif

    inline static void add(mtgp32_fast_t *dest, mtgp32_fast_t *src) {
	int dp = dest->status->idx;
	int sp = src->status->idx;
	int large_size = src->status->large_size;
	int diff = (sp - dp + large_size) % large_size;
	int p;
	int i;
	for (i = 0; i < large_size - diff; i++) {
	    p = i + diff;
	    dest->status->array[i] ^= src->status->array[p];
	}
	for (; i < large_size; i++) {
	    p = i + diff - large_size;
	    dest->status->array[i] ^= src->status->array[p];
	}
    }

/**
 * jump ahead using jump_string
 * @param dsfmt dSFMT internal state input and output.
 * @param jump_string string which represents jump polynomial.
 */
    void mtgp32_fast_jump(mtgp32_fast_t * mtgp32, const char * jump_string) {
	mtgp32_fast_t work;
	int bits;
	mtgp32_params_fast_t params = mtgp32->params;
	mtgp32_init(&work, &params, 0);
	memset(&work.status->array[0], 0,
	       sizeof(uint32_t) * work.status->large_size);
	work.status->idx = work.status->large_size - 1;

	for (int i = 0; jump_string[i] != '\0'; i++) {
	    bits = jump_string[i];
	    assert(isxdigit(bits));
	    bits = tolower(bits);
	    if (bits >= 'a' && bits <= 'f') {
		bits = bits - 'a' + 10;
	    } else {
		bits = bits - '0';
	    }
	    bits = bits & 0x0f;
	    for (int j = 0; j < 4; j++) {
		if ((bits & 1) != 0) {
		    add(&work, mtgp32);
		}
		mtgp32_next_state(mtgp32);
		bits = bits >> 1;
	    }
	}
	copy(mtgp32, &work);
    }

#if defined(__cplusplus)
}
#endif
