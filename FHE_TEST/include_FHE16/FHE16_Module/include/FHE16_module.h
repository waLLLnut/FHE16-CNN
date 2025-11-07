#ifndef FHE16_MODULE_H
#define FHE16_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ntt/PRE_TABLE_IDX.h"
#include "ntt/ntt16bit_GPU_A5000.h"
#include "ntt/ntt16bit_armv8_2_a_neon.h"
#include "ntt/ntt16bit_avx1_x86_64.h"
#include "ntt/ntt16bit_sse4_1_x86_64.h"
#include "ntt/ntt16bit_x86_64_avx2.h"
#include "ntt/ntt16bit_x86_64_avx512.h"
#include "ntt/ntt32bit_x86_64.h"

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // FHE16_MODULE_H
