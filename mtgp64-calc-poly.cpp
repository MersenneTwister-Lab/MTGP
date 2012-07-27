/**
 * @file mtgp64-calc-poly.cpp
 *
 * @brief calculate characteristic polynomial for 64bit mtgp.
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (Hiroshima University)
 *
 * Copyright (c) 2012 Mutsuo Saito, Makoto Matsumoto, Hiroshima
 * University and University of Tokyo. All rights reserved.
 *
 * The new BSD License is applied to this software, see LICENSE.txt
 */
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <string.h>
#include <string>
#include <errno.h>
#include <NTL/GF2X.h>
#include <NTL/vec_GF2.h>
#include <NTL/GF2XFactoring.h>
#include "mtgp64-calc-poly.hpp"
#include "mtgp-calc-jump.hpp"
#include "mtgp64-fast.h"

using namespace std;
using namespace NTL;

void mtgp64_calc_characteristic(GF2X& poly, mtgp64_fast_t * mtgp64)
{
    vec_GF2 seq;
    int mexp = mtgp64->params.mexp;
    seq.SetLength(2 * mexp);
    for (int i = 0; i < 2 * mexp; i++) {
	seq[i] = mtgp64_genrand_uint64(mtgp64) & 1;
    }
    MinPolySeq(poly, seq, mexp);
}

void mtgp64_calc_characteristic(string& str, mtgp64_fast_t * mtgp64)
{
    GF2X poly;
    mtgp64_calc_characteristic(poly, mtgp64);
    polytostring(str, poly);
}


#if defined(MAIN)
int main(int argc, char *argv[]) {
    int mexp;
    int no;
    uint32_t seed = 1;
    mtgp64_params_fast_t *params;
    mtgp64_fast_t mtgp64;
    int rc;

    if (argc <= 2) {
	printf("%s: mexp no.\n", argv[0]);
	return 1;
    }
    mexp = strtol(argv[1], NULL, 10);
    if (errno) {
	printf("%s: mexp no.\n", argv[0]);
	return 2;
    }
    no = strtol(argv[2], NULL, 10);
    if (errno) {
	printf("%s: mexp no.\n", argv[0]);
	return 3;
    }
    switch (mexp) {
    case 23209:
	params = mtgp64_params_fast_23209;
	break;
    case 44497:
	params = mtgp64_params_fast_44497;
	break;
    case 110503:
	params = mtgp64_params_fast_110503;
	break;
    default:
	printf("%s: mexp no.\n", argv[0]);
	printf("mexp shuould be 23209, 44497 or 110503\n");
	return 4;
    }
    if (no >= 128 || no < 0) {
	printf("%s: mexp no.\n", argv[0]);
	printf("no must be between 0 and 127\n");
	return 5;
    }
    params += no;
    rc = mtgp64_init(&mtgp64, params, seed);
    if (rc) {
	printf("failure in mtgp64_init\n");
	return -1;
    }
    mtgp64_print_idstring(&mtgp64, stdout);
    string s;
    mtgp64_calc_characteristic(s, &mtgp64);
    printf("%s\n", s.c_str());
    mtgp64_free(&mtgp64);
    return 0;
}
#endif
