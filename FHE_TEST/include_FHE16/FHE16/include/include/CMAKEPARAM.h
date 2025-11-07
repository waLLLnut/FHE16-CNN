#ifndef  CMAKEPARAM_H
#define CMAKEPARAM_H

#define PVM 1


/*************** SYSTEM Parameter *************/
#define CPUTYPE									1
#define SYSTEMTYPE							Linux




/*** Binary Configuration *****/

#define PVM                     1
#define NumQ_BIN                1
#define QbitSystem_BIN          32
#define Q1_RLWE_BIN             12289  
#define Q2_RLWE_BIN             18433  
#define QBIT_BK_BIN             28
#define N_LWE_BIN               650
#define N_RLWE_BIN              512
#define K_RLWE_BIN              2
#define QBIT_KS_BIN             14
#define GB_BK_BIN               4
#define GB_BK_LEN_BIN           6
#define GB_KS_BIN               10
#define GB_KS_LEN_BIN           2
#define INCOMPLETE_DEPTH_BIN    0



/*** DEBUG Configuration ********/
#define EFHE_DEBUG              0
#define COMPARE_OPENFHE         
#define MONTGOFLAG				
//#define INCOMPLETE_TWEAK		






/********** ASM configuration ******************/
//#define AVX128_16bit              
#define AVXTYPE                     2
#endif
