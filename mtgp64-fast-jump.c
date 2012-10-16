/**
 * @file mtgp64-fast-jump.c
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
#include "mtgp64-fast.h"
#include "mtgp64-fast-jump.h"

/**
 * add internal states as F2-vector.
 * @param[in,out] dest mtgp generator
 * @param[in] src mtgp generator
 */
inline static void add(mtgp64_fast_t *dest, mtgp64_fast_t *src) {
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
 * @param mtgp64 MTGP internal state input and output.
 * @param jump_string string which represents jump polynomial.
 */
void mtgp64_fast_jump(mtgp64_fast_t * mtgp64, const char * jump_string) {
    mtgp64_fast_t work;
    int bits;
    mtgp64_params_fast_t params = mtgp64->params;
    mtgp64_init(&work, &params, 0);
    memset(&work.status->array[0], 0,
	   sizeof(uint64_t) * work.status->large_size);
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
		add(&work, mtgp64);
	    }
	    mtgp64_next_state(mtgp64);
	    bits = bits >> 1;
	}
    }
    mtgp64_copy(mtgp64, &work);
}
