/**
 * @file test-jump32.cpp
 *
 * @brief test jump function.
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
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <inttypes.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#include "mtgp32-calc-poly.hpp"
#include "mtgp-calc-jump.hpp"
#include "mtgp32-fast-jump.h"
#include <NTL/GF2X.h>
#include <NTL/vec_GF2.h>
#include <NTL/ZZ.h>

using namespace NTL;
using namespace std;

static void test(mtgp32_fast_t * mtgp, GF2X& poly);
static int check(mtgp32_fast_t *a, mtgp32_fast_t *b);
static void print_state(mtgp32_fast_t * a, mtgp32_fast_t * b);
static void print_sequence(mtgp32_fast_t * a, mtgp32_fast_t * b);
static void speed(mtgp32_fast_t * mtgp, GF2X& characteristic);

int main(int argc, char * argv[]) {
    if (argc <= 3) {
	printf("%s -s|-c mexp no.\n", argv[0]);
	return -1;
    }
    errno = 0;
    GF2X characteristic;
    mtgp32_params_fast_t *params;
    mtgp32_fast_t mtgp;
    uint32_t seed = 0;
    //    int rc;
    int mexp = strtol(argv[2], NULL, 10);
    int no = strtol(argv[3], NULL, 10);
    if (no < 0 || no > 127) {
	cout << "error in no" << endl;
	return 1;
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
        return 2;
    }
    params += no;
    mtgp32_init(&mtgp, params, seed);
    calc_characteristic(characteristic, &mtgp);
    if (argv[1][1] == 's') {
	speed(&mtgp, characteristic);
    } else {
	test(&mtgp, characteristic);
    }
    mtgp32_free(&mtgp);
    return 0;
}

static void speed(mtgp32_fast_t * mtgp, GF2X& characteristic)
{
    long step = 10000;
    int exp = 4;
    ZZ test_count;
    string jump_string;
    clock_t start;
    double elapsed1;
    double elapsed2;

    test_count = step;
    for (int i = 0; i < 10; i++) {
	start = clock();
	calc_jump(jump_string, test_count, characteristic);
	elapsed1 = clock() - start;
	elapsed1 = elapsed1 * 1000 / CLOCKS_PER_SEC;
	cout << "mexp "
	     << setw(5)
	     << mtgp->params.mexp
	     << " jump 10^"
	     << setfill('0') << setw(2)
	     << exp
	     << " steps calc_jump:"
	     << setfill(' ') << setiosflags(ios::fixed)
	     << setw(6) << setprecision(3)
	     << elapsed1
	     << "ms"
	     << endl;
	start = clock();

	for (int j = 0; j < 10; j++) {
	    mtgp32_fast_jump(mtgp, jump_string.c_str());
	}
	elapsed2 = clock() - start;
	elapsed2 = elapsed2 * 1000 / 10 / CLOCKS_PER_SEC;
	cout << "mexp "
	     << setw(5)
	     << mtgp->params.mexp
	     << " jump 10^"
	     << setfill('0') << setw(2)
	     << exp
	     << " steps MTGP_jump:"
	     << setfill(' ') << setiosflags(ios::fixed)
	     << setw(6) << setprecision(3)
	     << elapsed2
	     << "ms"
	     << endl;
	test_count *= 100;
	exp += 2;
    }
}

static int check(mtgp32_fast_t *a, mtgp32_fast_t *b)
{
    int check = 0;
    for (int i = 0; i < 100; i++) {
	uint32_t x = mtgp32_genrand_uint32(a);
	uint32_t y = mtgp32_genrand_uint32(b);
	if (x != y) {
	    print_state(a, b);
	    print_sequence(a, b);
	    check = 1;
	    break;
	}
    }
    if (check == 0) {
      cout << "OK!" << endl;
    } else {
      cout << "NG!" << endl;
    }
    return check;
}

static void print_state(mtgp32_fast_t *a, mtgp32_fast_t * b)
{
  int large_size = a->status->large_size;
  cout << "idx = " << dec << a->status->idx
       << "   " << dec << b->status->idx
       << endl;
    for (int i = 0; (i < 10) && (i < large_size); i++) {
      cout << setfill('0') << setw(8) << hex
	   << a->status->array[(i + a->status->idx) % large_size];
      cout << " ";
      cout << setfill('0') << setw(8) << hex
	   << b->status->array[(i + b->status->idx) % large_size];
      cout << endl;
    }
}

static void print_sequence(mtgp32_fast_t *a, mtgp32_fast_t * b)
{
    for (int i = 0; i < 25; i++) {
	uint32_t c, d;
	c = mtgp32_genrand_uint32(a);
	d = mtgp32_genrand_uint32(b);
	cout << setfill('0') << setw(8) << hex << c;
	cout << " " << setfill('0') << setw(8) << hex << d;
	cout << endl;
    }
}

static void test(mtgp32_fast_t * mtgp, GF2X& characteristic)
{
    mtgp32_fast_t new_mtgp_z;
    mtgp32_fast_t * new_mtgp = &new_mtgp_z;
//    uint32_t seed[] = {1, 998102, 1234, 0, 5};
//    uint32_t seed[20] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    long steps[] = {1, 2, mtgp->status->size + 1,
		    mtgp->status->size * 32 - 1,
		    mtgp->status->size * 32 + 1,
		    3003,
		    200004,
		    10000005};
    int steps_size = sizeof(steps) / sizeof(long);
    ZZ test_count;
    string jump_string;
    mtgp32_params_fast_t params = mtgp->params;
    mtgp32_init(new_mtgp, &params, 0);
    mtgp32_genrand_uint32(mtgp);
    mtgp32_genrand_uint32(mtgp);
    mtgp32_genrand_uint32(mtgp);
    /* plus jump */
    for (int index = 0; index < steps_size; index++) {
//	mtgp_init_gen_rand(mtgp, seed[index]);
	test_count = steps[index];
	cout << "mexp " << dec << mtgp->params.mexp << " jump "
	     << test_count << " steps" << endl;
//	*new_mtgp = *mtgp;
	mtgp32_copy(new_mtgp, mtgp);
	for (long i = 0; i < steps[index]; i++) {
	    mtgp32_genrand_uint32(mtgp);
	}
	calc_jump(jump_string, test_count, characteristic);
#if defined(DEBUG)
	cout << "jump string:" << jump_string << endl;
	cout << "before jump:" << endl;
	print_state(new_mtgp, mtgp);
#endif
	mtgp32_fast_jump(new_mtgp, jump_string.c_str());
#if defined(DEBUG)
	cout << "after jump:" << endl;
	print_state(new_mtgp, mtgp);
#endif
	if (check(new_mtgp, mtgp)) {
	    return;
	}
    }
}
