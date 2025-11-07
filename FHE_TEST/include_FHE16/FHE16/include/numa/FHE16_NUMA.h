#ifndef FHE16_NUMA_H
#define FHE16_NUMA_H

#include <pthread.h>

#ifndef __cplusplus
#include<stdint.h>
#endif


#include<CMAKEPARAM.h>

#if CPUTYPE == 1 // IF LINUX
    #include<stdint.h>
    #include<stdbool.h>

#endif


#ifdef __cplusplus
extern "C" {
#endif


void aligned_numa_alloc(size_t size, int node, int align, void **add_orign, void **aligned_add);



#ifdef __cplusplus
}
#endif



#endif
