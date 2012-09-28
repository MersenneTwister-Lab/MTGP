#include <stdint.h>
#include "mtgp64-fast.h"
#define MTGPDC_MEXP 11213
#define MTGPDC_N 176
#define MTGPDC_FLOOR_2P 128
#define MTGPDC_CEIL_2P 256
#define MTGPDC_PARAM_TABLE mtgp64dc_params_fast_11213

mtgp64_params_fast_t mtgp64dc_params_fast_11213[]
 = {
    {
        /* No.0 delta:3250 weight:1987 */
        11213,
        45,
        11,
        4,
        {UINT64_C(0x0000000000000000),
         UINT64_C(0xacd0c7eb00000000),
         UINT64_C(0x63f8aada00000000),
         UINT64_C(0xcf286d3100000000),
         UINT64_C(0x966a000000000000),
         UINT64_C(0x3abac7eb00000000),
         UINT64_C(0xf592aada00000000),
         UINT64_C(0x59426d3100000000),
         UINT64_C(0x0000367100000000),
         UINT64_C(0xacd0f19a00000000),
         UINT64_C(0x63f89cab00000000),
         UINT64_C(0xcf285b4000000000),
         UINT64_C(0x966a367100000000),
         UINT64_C(0x3abaf19a00000000),
         UINT64_C(0xf5929cab00000000),
         UINT64_C(0x59425b4000000000)},
        {UINT64_C(0x0000000000000000),
         UINT64_C(0x2000000000000000),
         UINT64_C(0xc000000000000000),
         UINT64_C(0xe000000000000000),
         UINT64_C(0x0901000000000000),
         UINT64_C(0x2901000000000000),
         UINT64_C(0xc901000000000000),
         UINT64_C(0xe901000000000000),
         UINT64_C(0x401bf00000000000),
         UINT64_C(0x601bf00000000000),
         UINT64_C(0x801bf00000000000),
         UINT64_C(0xa01bf00000000000),
         UINT64_C(0x491af00000000000),
         UINT64_C(0x691af00000000000),
         UINT64_C(0x891af00000000000),
         UINT64_C(0xa91af00000000000)},
        {UINT64_C(0x3ff0000000000000),
         UINT64_C(0x3ff2000000000000),
         UINT64_C(0x3ffc000000000000),
         UINT64_C(0x3ffe000000000000),
         UINT64_C(0x3ff0901000000000),
         UINT64_C(0x3ff2901000000000),
         UINT64_C(0x3ffc901000000000),
         UINT64_C(0x3ffe901000000000),
         UINT64_C(0x3ff401bf00000000),
         UINT64_C(0x3ff601bf00000000),
         UINT64_C(0x3ff801bf00000000),
         UINT64_C(0x3ffa01bf00000000),
         UINT64_C(0x3ff491af00000000),
         UINT64_C(0x3ff691af00000000),
         UINT64_C(0x3ff891af00000000),
         UINT64_C(0x3ffa91af00000000)},
        UINT64_C(0xfff8000000000000),
        {0x0e,0x4c,0x44,0x8c,0xba,0xcf,0xa0,0xf8,0xec,0x70,
         0xc5,0x1d,0x24,0xb9,0xff,0xce,0xe7,0x48,0x43,0xf0,0x00}
    }
};
const int mtgpdc_params_11213_num  = 1;
