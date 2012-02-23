/**
 */
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <time.h>
extern "C" {
#include "mtgp32-fast.h"
#include "mtgp64-fast.h"
}

using namespace NTL;
using namespace std;

int bit_size;
int mexp;
int no;
uint64_t jump_step;

§ﬁ§¿≈”√Ê
static void u32_apply_jump(int mexp, int no, int seed);
static void u64_apply_jump(int mexp, int no, int seed);
static void read_poly(uint64_t *array, FILE *fp);

int main(int argc, char *argv[]) {
    uint32_t seed = 1;

    if (argc <= 4) {
	cout << argv[0] << ": bit_size mexp no. jump_step" << endl;
	return 1;
    }
    bit_size = strtol(argv[1], NULL, 10);
    if (errno) {
	cout << argv[0] << ": bit_size error." << endl;
	return 1;
    }
    if (bit_size != 32 && bit_size != 64) {
	cout << argv[0] << ": bit_size error. bit size is 32 or 64" << endl;
	return 1;
    }
    mexp = strtol(argv[2], NULL, 10);
    if (errno) {
	cout << argv[0] << ": mexp error" << endl;
	return 1;
    }
    no = strtol(argv[3], NULL, 10);
    if (errno) {
	cout << argv[0] << ": no. error \n" << endl;
	return 3;
    }
    jump_step = strtoull(argv[4], NULL, 10);
    if (errno) {
	cout << argv[0] << ": jump_step error.\n" << endl;
	return 3;
    }
    if (bit_size == 32) {
	u32_print_poly(mexp, no, seed);
    } else {
	u64_print_poly(mexp, no, seed);
    }
    return 0;
}

static void print_poly(GF2X& poly)
{
    int size = mexp / 64 + 1;
    uint64_t array[size];
    for (int i = 0; i < size; i++) {
	array[i] = 0;
    }
    for (int i = 0; i <= mexp; i++) {
	if (IsOne(coeff(poly, i))) {
	    uint64_t mask = UINT64_C(1) << (i % 64);
	    int idx = i / 64;
	    array[idx] |= mask;
	}
    }
    for (int i = 0; i < size; i++) {
	printf("0x%016"PRIx64", ", array[i]);
	if (i % 4 == 3) {
	    printf("\n");
	}
    }
    printf("\n");
}

static void calc_minpoly(vec_GF2& vec, GF2X& minpoly) {
    MinPolySeq(minpoly, vec, mexp);
#if 0
    printf("minpoly for %d bit, mexp=%d, no=%d is\n",
	   bit_size, mexp, no);
    print_poly(minpoly);
#endif
}

static void print_jump_poly(GF2X& minpoly, uint64_t step)
{
    GF2XModulus modpoly(minpoly);
    GF2X jump_poly;
    ZZ jump;
    jump = (uint32_t)(step >> 32);
    jump <<= 32;
    jump += (uint32_t)(step & 0xffffffffU);
    clock_t start;
    clock_t ellipsed;
    start = clock();
    PowerXMod(jump_poly, jump, modpoly);
    ellipsed = clock() - start;
    printf("jump_poly for %"PRIx64" steps is\n", jump_step);
    print_poly(jump_poly);
    printf("ellipsed time to calculate jump_poly = %fms\n",
	   (double)ellipsed / CLOCKS_PER_SEC * 1000);
}

static void u32_print_poly(int mexp, int no, int seed) {
    mtgp32_params_fast_t *params;
    mtgp32_fast_t mtgp32;
    vec_GF2 vec;
    GF2X poly;

    int rc;

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
	cout << "mexp shuould be 11213, 23209 or 44497" << endl;
	exit(1);
    }
    if (no >= 128 || no < 0) {
	cout << "No. must be between 0 and 127" << endl;
	exit(1);
    }
    params += no;
    rc = mtgp32_init(&mtgp32, params, seed);
    if (rc) {
	cout << "failure in mtgp32_init" << endl;
	exit(1);
    }
    mtgp32_print_idstring(&mtgp32, stdout);
    vec.SetLength(2 * mexp);
    for (int i = 0; i < 2 * mexp; i++) {
	vec[i] = mtgp32_genrand_uint32(&mtgp32) & 1;
    }
    calc_minpoly(vec, poly);
    print_jump_poly(poly, jump_step);
    mtgp32_free(&mtgp32);
}

static void u64_print_poly(int mexp, int no, int seed) {
    mtgp64_params_fast_t *params;
    mtgp64_fast_t mtgp64;
    vec_GF2 vec;
    GF2X poly;
    int rc;

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
	cout << "mexp shuould be 11213, 23209 or 44497" << endl;
	exit(1);
    }
    if (no >= 128 || no < 0) {
	cout << "No. must be between 0 and 127" << endl;
	exit(1);
    }
    params += no;
    rc = mtgp64_init(&mtgp64, params, seed);
    if (rc) {
	cout << "failure in mtgp64_init." << endl;
	exit(1);
    }
    mtgp64_print_idstring(&mtgp64, stdout);
    vec.SetLength(2 * mexp);
    for (int i = 0; i < 2 * mexp; i++) {
	vec[i] = mtgp64_genrand_uint64(&mtgp64) & 1;
    }
    calc_minpoly(vec, poly);
    print_jump_poly(poly, jump_step);
    mtgp64_free(&mtgp64);
}
