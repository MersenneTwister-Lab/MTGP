#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <NTL/vec_GF2.h>
#include <NTL/GF2X.h>
#include <NTL/GF2Xfactoring.h>

const uint32_t seed[32]={
    0x8cf35fea, 0xe1dd819e, 0x4a7d0a8e, 0xe0c05911, 0xfd053b8d,
    0x30643089, 0x6f6ac111, 0xc4869595, 0x9416b7be, 0xe6d329e8,
    0x5af0f5bf, 0xc5c742b5, 0x7197e922, 0x71aa35b4, 0x2070b9d1,
    0x2bb34804, 0x7754a517, 0xe725315e, 0x7f9dd497, 0x043b58bf,
    0x83ffa33d, 0x2532905a, 0xbdfe0c8a, 0x16f68671, 0x0d14da2e,
    0x847efd5f, 0x1edeec64, 0x1bebdf9b, 0xf74d4ff3, 0xd404774b,
    0x8ee32599, 0xefe0c405
};

const int pos1[32] = {29,24,5,23,14,26,11,31,9,3,1,28,0,2,22,20,
		      18,15,27,13,10,16,8,17,25,12,19,30,7,6,4,21};
const int pos2[32] = {5,14,28,24,19,13,0,17,11,20,7,10,6,15,2,9,8,
		      23,4,30,12,25,3,21,26,27,31,18,22,16,29,1};
const int shift1 = 2;
const int shift2[32] = {0,1,0,1,1,1,0,0,1,0,0,1,0,0,1,0,
		       0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1};
void check_polynomial(uint32_t status[]);

static void generate(uint32_t status[], uint32_t out[])
{
    uint32_t t0;
    uint32_t t1;
    uint32_t res[32];
    for (int i = 0; i < 32; i++) {
	t0 = status[pos1[i]];
	t1 = status[pos2[i]];
	res[i] = (t0 << shift1) ^ (t1 >> shift2[i]);
#if defined(CORRE)
	out[i] = res[i];
#else
	out[i] = t0 + t1;
#endif
    }
    for (int i = 0; i < 32; i++) {
	status[i] = res[i];
    }
}

int main(int argc, char **argv) {
    uint32_t status[32];
    uint32_t out[32];
    memcpy(status, seed, sizeof(uint32_t) * 32);
    for (int i = 0; i < 100; i++) {
	generate(status, out);
	for (int j = 0; j < 32; j++) {
	    printf("%10u ", out[j]);
	    if (j % 4 == 3) {
		printf("\n");
	    }
	}
    }
    check_polynomial(status);
    return 0;
}

void check_polynomial(uint32_t status[]) {
    using namespace NTL;
    using namespace std;

    uint32_t out[32];
    vec_GF2 vec;
    int mexp = 1024;
    vec.SetLength(mexp * 2);
    GF2X minpoly;
    for (int k = 0; k < 10; k++) {
    for (int i = 0; i < 32; i++) {
	for (int j = 0; j < mexp * 2; j++) {
	    generate(status, out);
	    vec[j] = (status[k] >> i) & 1;
	}
	MinPolySeq(minpoly, vec, mexp);
	cout << dec << k << ":";
	cout << dec << i << ":";
	cout << "deg " << dec << deg(minpoly);
	if (IterIrredTest(minpoly)) {
	    cout << " Irreducible" << endl;
	} else {
	    cout << " Reducible" << endl;
	}
    }
    }
}
