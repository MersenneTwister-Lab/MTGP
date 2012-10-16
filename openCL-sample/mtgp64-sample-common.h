#ifndef MTGP64_SAMPLE_COMMON_H
#define MTGP64_SAMPLE_COMMON_H

#include <stdint.h>
#include <inttypes.h>
#include <iostream>
#include <iomanip>

#define MTGP64_MEXP 11213
#define MTGP64_N 176
#define MTGP64_FLOOR_2P 128
#define MTGP64_CEIL_2P 256
#define MTGP64_TN MTGP64_FLOOR_2P
#define MTGP64_LS (MTGP64_TN * 3)
#define MTGP64_TS 16
#define MTGP64_JTS (MTGP64_MEXP / 32 + 1)

extern "C" {
#include "mtgp64-fast.h"
    extern mtgp64_params_fast_t mtgp64dc_params_fast_11213[];
}

static inline void print_uint64(uint64_t data[], int size, int item_num)
{
    using namespace std;

    int max_seq = 10;
    int max_item = 3;
    if (size / item_num < max_seq) {
	max_seq = size / item_num;
    }
    if (item_num < max_item) {
	max_item = item_num;
    }
    for (int i = 0; i < max_seq; i++) {
	for (int j = 0; j < max_item; j++) {
	    cout << setw(20) << dec << data[item_num * i + j] << " ";
	}
	cout << endl;
    }
}

static inline void print_double(double data[], int size, int item_num)
{
    using namespace std;

    int max_seq = 10;
    int max_item = 3;
    if (size / item_num < max_seq) {
	max_seq = size / item_num;
    }
    if (item_num < max_item) {
	max_item = item_num;
    }
    for (int i = 0; i < max_seq; i++) {
	for (int j = 0; j < max_item; j++) {
	    cout << setprecision(18) << setw(24)
		 << dec << left << setfill(' ')
		 << data[item_num * i + j] << " ";
	}
	cout << endl;
    }
    cout << setprecision(10);
}

#endif
