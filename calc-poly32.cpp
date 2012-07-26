/**
 * @file calc-poly32.cpp
 *
 * @brief calculate characteristic polynomial for 32bit mtgp.
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
#include "mtgp32-calc-poly.hpp"
#include "mtgp-calc-jump.hpp"
#include "mtgp32-fast.h"

using namespace std;
using namespace NTL;

void calc_characteristic(GF2X& poly, mtgp32_fast_t * mtgp32)
{
    vec_GF2 seq;
    int mexp = mtgp32->params.mexp;
    seq.SetLength(2 * mexp);
    for (int i = 0; i < 2 * mexp; i++) {
	seq[i] = mtgp32_genrand_uint32(mtgp32) & 1;
    }
    MinPolySeq(poly, seq, mexp);
}

void calc_characteristic(string& str, mtgp32_fast_t * mtgp32)
{
    GF2X poly;
    calc_characteristic(poly, mtgp32);
    polytostring(str, poly);
}


#if defined(MAIN)
int main(int argc, char *argv[]) {
    int mexp;
    int no;
    uint32_t seed = 1;
    mtgp32_params_fast_t *params;
    mtgp32_fast_t mtgp32;
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
    case 11213:
	params = mtgp32_params_fast_11213;
	break;
    case 23209:
	params = mtgp32_params_fast_23209;
	break;
    case 44497:
	params = mtgp32_params_fast_44497;
	break;
    default:
	printf("%s: mexp no.\n", argv[0]);
	printf("mexp shuould be 11213, 23209 or 44497\n");
	return 4;
    }
    if (no >= 128 || no < 0) {
	printf("%s: mexp no.\n", argv[0]);
	printf("no must be between 0 and 127\n");
	return 5;
    }
    params += no;
    rc = mtgp32_init(&mtgp32, params, seed);
    if (rc) {
	printf("failure in mtgp32_init\n");
	return -1;
    }
    mtgp32_print_idstring(&mtgp32, stdout);
    string s;
    calc_characteristic(s, &mtgp32);
    printf("%s\n", s.c_str());
    mtgp32_free(&mtgp32);
    return 0;
}
#endif
