#ifndef FHE16_THREADDATA_H
#define FHE16_THREADDATA_H

#include<atomic>
#include<pthread.h>
//#include<BINFHE.hpp>
#include<Core.hpp>
#include<BinOperationCstyle.hpp>


struct alignas(64) PaddedAtomic {
	std::atomic_int val;
};

// DH & Shin Plz !!!
//struct PrefixAdderData{
struct alignas(64) PrefixAdderData{
    int32_t **x                 ;
    int32_t **y                 ;
	int32_t **z                 ;
	int32_t **tmp1              ;
    int32_t **tmp2              ;
    int32_t ***p                ;
    int32_t ***g                ;
    int32_t **res               ;
	PaddedAtomic *g_depth_cnta  ;
    PaddedAtomic *p_depth_cnta  ;
    PaddedAtomic *PROCESS       ;
    //PaddedAtomic *PROCESS2      ;
	FHE16BOOTParam **BOOTParams ;
	int16_t	*** ZeroCheck		; // for until implementing trivial operation 
	
	int32_t **** MUL_VERSION	;
    int16_t	* active_rows		; 
    PaddedAtomic ***MUL_LOCK     ;
  
	int idx						;
    int received_idx			;
    int nbits                   ;
    int DEPTH                   ;
    int thread_num				;
    int N						;
    bool subtract               ;
    char padding[28];
};


#endif



