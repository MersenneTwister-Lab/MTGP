#ifndef MTGP32_SAMPLE_JUMP_COMMON_H
#define MTGP32_SAMPLE_JUMP_COMMON_H

#define MTGP32_MEXP 11213
#define MTGP32_N 351
#define MTGP32_FLOOR_2P 256
#define MTGP32_CEIL_2P 512
#define MTGP32_TN MTGP32_FLOOR_2P
#define MTGP32_LS (MTGP32_TN * 3)
#define MTGP32_TS 16

#if defined(__cplusplus)
extern "C" {
#endif
#if 0
/**
 * kernel I/O
 * This structure must be initialized before first use.
 */
struct MTGP32_KERNEL_STATUS_T {
    uint status[MTGP32_N];
};

typedef struct MTGP32_KERNEL_STATUS_T mtgp32_status_t;
#endif
#if defined(__cplusplus)
}
#endif
#endif
