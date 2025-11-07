#ifndef NTT_32bit_64represent_EF
#define NTT_32bit_64represent_EF

#ifndef __cplusplus
#include<stdint.h>
#endif

#if CPUTYPE == 1 // IF LINUX
	#include<stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include<param.h>

// Cooley-Tukey NTT 
void ntt_32bit_64represent(uint64_t *roots,uint64_t *x, int step);
// Gentleman-Sande INTT 

void intt_32bit_64represent(uint64_t inv,uint64_t invpr, uint64_t *iroots, uint64_t *x, int step);
// PRODUCT 
//
void polymulNTT_32bit_64represent( uint64_t *out, uint64_t *a, uint64_t *b, uint64_t *roots, uint64_t inv, uint64_t invpr, uint64_t * iroots);

#ifdef __cplusplus
}
#endif






#endif
