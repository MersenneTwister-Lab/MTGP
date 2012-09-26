#ifndef MTGP32_SAMPLE_COMMON_H
#define MTGP32_SAMPLE_COMMON_H

#include <stdint.h>
#include <inttypes.h>
#include <iostream>
#include <iomanip>

#define MTGP32_MEXP 11213
#define MTGP32_N 351
#define MTGP32_FLOOR_2P 256
#define MTGP32_CEIL_2P 512
#define MTGP32_TN MTGP32_FLOOR_2P
#define MTGP32_LS (MTGP32_TN * 3)
#define MTGP32_TS 16

static inline void print_uint32(uint32_t data[], int size, int item_num)
{
    using namespace std;

    int max_seq = 10;
    int max_item = 6;
    if (size / item_num < max_seq) {
	max_seq = size / item_num;
    }
    if (item_num < max_item) {
	max_item = item_num;
    }
    for (int i = 0; i < max_seq; i++) {
	for (int j = 0; j < max_item; j++) {
	    cout << setw(10) << dec << data[item_num * i + j] << " ";
	}
	cout << endl;
    }
}

static inline void print_float(float data[], int size, int item_num)
{
    using namespace std;

    int max_seq = 10;
    int max_item = 6;
    if (size / item_num < max_seq) {
	max_seq = size / item_num;
    }
    if (item_num < max_item) {
	max_item = item_num;
    }
    for (int i = 0; i < max_seq; i++) {
	for (int j = 0; j < max_item; j++) {
	    cout << setprecision(9) << setw(12)
		 << dec << left << setfill(' ')
		 << data[item_num * i + j] << " ";
	}
	cout << endl;
    }
}

#endif
