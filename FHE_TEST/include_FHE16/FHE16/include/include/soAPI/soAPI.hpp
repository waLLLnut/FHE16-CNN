#ifndef FHE16_SOAPI_H
#define FHE16_SOAPI_H

#include<poly_class.hpp>
#include<lwe.hpp>
#include<FHE16Param.hpp>
#include<CMAKEPARAM.h>

#if NUMA_FLAG == 1
#include <numa.h>
#endif 
#include <unistd.h>
#include <ThreadData.hpp>
extern EFHEs::EFHE_BIN_Param_List	* G_FHE16_PARAM ;
extern PrefixAdderData				* G_FHE16_ADDER_DATA;

//extern NTTTable16           * G_FHE16_BKNTT ;
extern int32_t              * G_FHE16_sk    ;


void FHE16_LoadEval();
int *FHE16_GenEval();
void FHE16_DeleteEval();
//int32_t *FHE16_ENC(int msg, int bit);

int32_t *FHE16_ENC(int msg, int bit, int32_t* &tmp_SK, int32_t* &tmp_E);
int32_t *FHE16_ENC(int msg, int bit);
int32_t *FHE16_ENCInt(int msg, int bit, int32_t* &tmp_SK, int32_t* &tmp_E);
int32_t *FHE16_ENCInt(int msg, int bit);
int32_t *FHE16_ENCInt(int *msg, int bit, int32_t* &tmp_SK, int32_t* &tmp_E);
int32_t *FHE16_ENCInt(int *msg, int bit);



int FHE16_DEC(int32_t *CT, int32_t *sk, int bits, int &E);
int64_t FHE16_DECInt(int32_t *CT, int32_t *sk);
int32_t *FHE16_DECIntVec(int32_t *CT, int32_t *sk);



int32_t *FHE16_PREFIX_FLAG(int32_t *CT1, int32_t *CT2, bool FLAG);
int32_t *FHE16_COMPARE(int32_t *CT1, int32_t *CT2, bool FLAG);
int32_t *FHE16_MAXorMIN(int32_t *CT1, int32_t *CT2, bool FLAG);



int32_t *FHE16_ADD(int32_t *CT1, int32_t *CT2);
int32_t *FHE16_ADD3(int32_t *CT1, int32_t*CT2, int32_t *CT3);
int32_t *FHE16_SUB(int32_t * CT1, int32_t *CT2);


int32_t *FHE16_LE(int32_t *CT1, int32_t *CT2);
int32_t *FHE16_LT(int32_t *CT1, int32_t *CT2);
int32_t *FHE16_GE(int32_t *CT1, int32_t *CT2);
int32_t *FHE16_GT(int32_t *CT1, int32_t *CT2);
int32_t *FHE16_MAX(int32_t * CT1, int32_t *CT2);
int32_t *FHE16_MIN(int32_t * CT1, int32_t *CT2);
int32_t *FHE16_ANDVEC(int32_t * CT1, int32_t *CT2);
int32_t *FHE16_ORVEC(int32_t * CT1, int32_t *CT2);
int32_t *FHE16_XORVEC(int32_t * CT1, int32_t *CT2);
int32_t *FHE16_SELECT(int32_t *CT_select, int32_t *CT1, int32_t *CT2);
int32_t *FHE16_SMULL(int32_t *CT1, int32_t *CT2);
int32_t *FHE16_SDIV(int32_t *CT1, int32_t *CT2, int32_t *CT_REM, int32_t *IsZero);
int32_t *FHE16_RELU(int32_t *CT);


///////////////////////// CONSTANT
int32_t *FHE16_SMULL_CONSTANT(int32_t *CT, int32_t *Constant);
int32_t *FHE16_SMULL_CONSTANT(int32_t *CT, int64_t Constant);
int32_t *FHE16_SMULL_CONSTANT(int32_t *CT, int Constant);


int32_t *FHE16_ADD_CONSTANT(int32_t *CT, int32_t *Constant);
int32_t *FHE16_ADD_CONSTANT(int32_t *CT, int64_t Constant);
int32_t *FHE16_ADD_CONSTANT(int32_t *CT, int Constant);











int32_t *FHE16_LSHIFTL(int32_t * CT, int k);
int32_t *FHE16_LSHIFTR(int32_t * CT, int k);
int32_t *FHE16_ASHIFTR(int32_t * CT, int k);


int32_t *FHE16_ROTATEL(int32_t * CT, int k);
int32_t *FHE16_ROTATER(int32_t * CT, int k);



int32_t *FHE16_ADD_POWTWO(int32_t *CT1, int pow);
int32_t *FHE16_SUB_POWTWO(int32_t *CT1, int pow);
int32_t *FHE16_NEG(int32_t *CT1);
int32_t *FHE16_ABS(int32_t *CT1);



int32_t *FHE16_EQ(int32_t *CT1, int32_t *CT2);
int32_t *FHE16_NEQ(int32_t *CT1, int32_t *CT2);




//int32_t **





int32_t FHE16_LZC_Plain(int x);





#endif // End header
