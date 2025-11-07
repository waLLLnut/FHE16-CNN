#ifndef ADDER_HEADER_EF
#define ADDER_HEADER_EF

#ifndef __cplusplus
#include<stdint.h>
#endif


#include<CMAKEPARAM.h>

#if CPUTYPE == 1 // IF LINUX
	#include<stdint.h>
	#include<stdbool.h>

#endif

#if AVXTYPE == 2
	#include<immintrin.h>
#elif AVXTYPE == 3
	#include<immintrin.h>
#elif AVXTYPE == 5

#endif




#include<param.h>
#include<stdio.h>
#include<stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif


extern int FHE16_ADDER_ARR64[64];
extern int FHE16_ADDER_ARR128[128];
extern int FHE16_ADDER_ARR256[256];
extern int FHE16_ADDER_ARR512[512];





#ifdef __cplusplus
}
#endif



#endif
