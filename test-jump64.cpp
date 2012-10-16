/**
 * @file test-jump64.cpp
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
#include "mtgp64-calc-poly.hpp"
#include "mtgp-calc-jump.hpp"
#include "mtgp64-fast-jump.h"
#include <NTL/GF2X.h>
#include <NTL/vec_GF2.h>
#include <NTL/ZZ.h>

using namespace NTL;
using namespace std;

static void test(mtgp64_fast_t * mtgp, GF2X& poly);
static int check(mtgp64_fast_t *a, mtgp64_fast_t *b);
static void print_state(mtgp64_fast_t * a, mtgp64_fast_t * b);
static void print_sequence(mtgp64_fast_t * a, mtgp64_fast_t * b);
static void speed(mtgp64_fast_t * mtgp, GF2X& characteristic);

/**
 * test main
 * @param[in] argc number of arguments.
 * @param[in] argv an array of arguments.
 * @return 0 if normal, others abnormal.
 */
int main(int argc, char * argv[]) {
    if (argc <= 3) {
	printf("%s -s|-c mexp no.\n", argv[0]);
	return -1;
    }
    errno = 0;
    GF2X characteristic;
    mtgp64_params_fast_t *params;
    mtgp64_fast_t mtgp;
    uint32_t seed = 0;
    //    int rc;
    int mexp = strtol(argv[2], NULL, 10);
    int no = strtol(argv[3], NULL, 10);
    if (no < 0 || no > 127) {
	cout << "error in no" << endl;
	return 1;
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
        return 2;
    }
    params += no;
    mtgp64_init(&mtgp, params, seed);
    mtgp64_calc_characteristic(characteristic, &mtgp);
    if (argv[1][1] == 's') {
	speed(&mtgp, characteristic);
    } else {
	test(&mtgp, characteristic);
    }
    mtgp64_free(&mtgp);
    return 0;
}

/**
 * check speed
 * @param[in] mtgp generator
 * @param[in] characteristic characteristic polynomial
 */
static void speed(mtgp64_fast_t * mtgp, GF2X& characteristic)
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
	    mtgp64_fast_jump(mtgp, jump_string.c_str());
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

/**
 * equality check
 * @param[in] a mtgp generator
 * @param[in] b mtgp generator
 * @return 0 if equal
 */
static int check(mtgp64_fast_t *a, mtgp64_fast_t *b)
{
    int check = 0;
    for (int i = 0; i < 100; i++) {
	uint64_t x = mtgp64_genrand_uint64(a);
	uint64_t y = mtgp64_genrand_uint64(b);
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

/**
 * print internal state of two mtgp generators for checking by human eyes.
 * @param[in] a mtgp generator
 * @param[in] b mtgp generator
 */
static void print_state(mtgp64_fast_t *a, mtgp64_fast_t * b)
{
  int large_size = a->status->large_size;
  cout << "idx = " << dec << a->status->idx
       << "   " << dec << b->status->idx
       << endl;
    for (int i = 0; (i < 10) && (i < large_size); i++) {
      cout << setfill('0') << setw(16) << hex
	   << a->status->array[(i + a->status->idx) % large_size];
      cout << " ";
      cout << setfill('0') << setw(16) << hex
	   << b->status->array[(i + b->status->idx) % large_size];
      cout << endl;
    }
}

/**
 * print output sequences of two mtgp generators for checking by human eyes.
 * @param[in] a mtgp generator
 * @param[in] b mtgp generator
 */
static void print_sequence(mtgp64_fast_t *a, mtgp64_fast_t * b)
{
    for (int i = 0; i < 25; i++) {
	uint64_t c, d;
	c = mtgp64_genrand_uint64(a);
	d = mtgp64_genrand_uint64(b);
	cout << setfill('0') << setw(16) << hex << c;
	cout << " " << setfill('0') << setw(16) << hex << d;
	cout << endl;
    }
}

/**
 * sanity check
 * @param[in] mtgp generator
 * @param[in] characteristic characteristic polynomial
 */
static void test(mtgp64_fast_t * mtgp, GF2X& characteristic)
{
    mtgp64_fast_t new_mtgp_z;
    mtgp64_fast_t * new_mtgp = &new_mtgp_z;
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
    mtgp64_params_fast_t params = mtgp->params;
    mtgp64_init(new_mtgp, &params, 0);
    mtgp64_genrand_uint64(mtgp);
    mtgp64_genrand_uint64(mtgp);
    mtgp64_genrand_uint64(mtgp);
    /* plus jump */
    for (int index = 0; index < steps_size; index++) {
//	mtgp_init_gen_rand(mtgp, seed[index]);
	test_count = steps[index];
	cout << "mexp " << dec << mtgp->params.mexp << " jump "
	     << test_count << " steps" << endl;
//	*new_mtgp = *mtgp;
	mtgp64_copy(new_mtgp, mtgp);
	for (long i = 0; i < steps[index]; i++) {
	    mtgp64_genrand_uint64(mtgp);
	}
	calc_jump(jump_string, test_count, characteristic);
#if defined(DEBUG)
	cout << "jump string:" << jump_string << endl;
	cout << "before jump:" << endl;
	print_state(new_mtgp, mtgp);
#endif
	mtgp64_fast_jump(new_mtgp, jump_string.c_str());
#if defined(DEBUG)
	cout << "after jump:" << endl;
	print_state(new_mtgp, mtgp);
#endif
	if (check(new_mtgp, mtgp)) {
	    return;
	}
    }
}
