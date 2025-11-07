#ifndef FHE16_THREADOPERATION_H
#define FHE16_THREADOPERATION_H

#include<atomic>
#include<pthread.h>
//#include<BINFHE.hpp>
#include<Core.hpp>
#include<BinOperationCstyle.hpp>

////////////////////////// Version Care
void *PrefixAdder_V1(void *arg);




////////////////////////////// We using .....
//void *PrefixAdder(void *arg) = &PrefixAdder_V1;
void *PrefixAdder(void *arg) { return PrefixAdder_V1(arg); }



void *PrefixAdder3(void *arg);
void *PrefixAND(void *arg);
void *PrefixRELU(void *arg);


void *PrefixOR(void *arg);
void *PrefixABS(void *arg);


void *SMULL_THREAD(void *arg);
void *SMULL_DADDA(void *arg);
void *SMULL_DADDA_Constant(void *arg);
void *PrefixAdder_Constant(void *arg);





void *Compare(void *arg);
void *MAXorMIN(void* arg);
void *EQ(void *arg);
void *SELECT(void *arg);
void *GATE_VEC(void *arg);

#endif



