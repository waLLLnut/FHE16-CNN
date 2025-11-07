#ifndef RANDOM_H_EF
#define RANDOM_H_EF



#include<CMAKEPARAM.h>
#include <param.h>

#if ((CPUTYPE == 1) || (CPUTYPE == 3))// IF LINUX
	#include <stdint.h>
#endif



#ifdef __cplusplus
extern "C" {
#endif

uint64_t uniform_sampling_Q(uint64_t Q);



#ifdef __cplusplus
}
#endif
#endif // HEADER END
