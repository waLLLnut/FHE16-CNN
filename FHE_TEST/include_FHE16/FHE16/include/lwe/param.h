#ifndef PARAMS_H
#define PARAMS_H

//#ifndef __cplusplus
//#include<stdint.h>
//#endif


#ifdef __cplusplus
extern "C" {
#endif



#include<CMAKEPARAM.h>
#include<structures.h>

//#include <immintrin.h>



#ifndef EF_N
#define EF_N 1024
#endif


/*32bit clear upper bits*/
#define CLEAR_HIGH_BIT = 4294967295 // 2^32 - 1

/*********************
 * Q related values
***********************/


    








#ifndef EF_Q
    /*
    #define EF_Q 167772161
    #define EF_Q_opt_bit 32
    #define EF_Q_bit 28
    #define EF_Q_bit_approx 25 // -> Q = Q_Delta*2^{Q_bit_approx} +1
    #define EF_Q_Delta 5
    #define INV3 55924054
    #define INV9 74565405
    #define INV15 44739243
    #define MONT_MU 167772159
    #define MONT_MASK 268435455
    #define MONT_MASK_APPROX 33554431
    #define MONT_MASK_APPROX_ADD_ONE 33554432
    #define BARRET_MU 858993454
    #define BARRET_DIV1_BIT 26
    #define BARRET_DIV2_BIT 31
    #define DEGREE (1<<LGN)
    #define WL 32  // sizeof(uint32_t)
    #define ONE 268435456
    #define R2MODP 6710889   // R^2 mod q
    #define ONE_INV 104857600

    #define LGN 10          // Degree n=2^LGN
    #define L 256             // 2nq < 2^{31-L}
    */

    #define LGN 10          // Degree n=2^LGN
    
    #define EF_Q_32bit 12289
    #define EF_Q_32bit_Square 151019521
    #define EF_Q_32bit_opt_bit 32
    #define EF_Q_32bit_bit 14
    #define EF_Q_32bit_bit_approx 25 // -> Q = Q_Delta*2^{Q_bit_approx} +1
    #define EF_Q_32bit_Delta 5
    #define INV3_32bit 8193
    #define INV9_32bit 2731
    #define INV15_32bit 9012
    #define MONT_MU_32bit 4143984639
    #define MONT_MASK_32bit 294967295
    #define MONT_MASK_32bit_APPROX 294967295
    #define MONT_MASK_APPROX_ADD_ONE 294967296
    #define R2MODP 0x1620   // R^2 mod q
    #define BARRET_MU_32bit 1431539266
    #define BARRET_DIV1_BIT_32bit 12
    #define BARRET_DIV2_BIT_32bit 32
    #define DEGREE (1<<LGN)
    #define WL_32bit 32  // sizeof(uint32_t)
    //#define ONE_32bit 0x2AC8
    #define ONE_32bit 10952
    #define ONE_INV_32bit 11857
    #define L_INTT_32bit 1             // 2nq < 2^{31-L}
    



    /*
    #define EF_Q_16bit 641
    #define EF_Q_16bit_Square 410881
    #define EF_Q_16bit_opt_bit 16
    #define EF_Q_16bit_bit 10
    #define EF_Q_16bit_bit_approx 0 // -> Q = Q_Delta*2^{Q_bit_approx} +1
    #define EF_Q_16bit_Delta 0
    #define INV3_16bit 0
    #define INV9_16bit 0
    #define INV15_16bit 0
   //#define R2MODP 0x1620   // R^2 mod q
    #define BARRET_MU_16bit 26173
    #define BARRET_DIV1_BIT_16bit 8
    #define BARRET_DIV2_BIT_16bit 16
    #define DEGREE (1<<LGN)
    #define WL_16bit 16  // sizeof(uint32_t)
    //#define ONE_32bit 0x2AC8
    #define MONT_MU_16bit 49791
    #define MONT_MASK_16bit 0
    #define MONT_MASK_16bit_APPROX 0
    #define ONE_16bit 154
    #define ONE_INV_16bit 487
    #define L_INTT_16bit 1             // 2nq < 2^{31-L}
    */
    
    #define EF_Q_16bit 12289
    #define EF_Q_16bit_Square 410881
    #define EF_Q_16bit_opt_bit 16
    #define EF_Q_16bit_bit 10
    #define EF_Q_16bit_bit_approx 0 // -> Q = Q_Delta*2^{Q_bit_approx} +1
    #define EF_Q_16bit_Delta 0
    #define INV3_16bit 0
    #define INV9_16bit 0
    #define INV15_16bit 0
   //#define R2MODP 0x1620   // R^2 mod q
    #define BARRET_MU_16bit 26173
    #define BARRET_DIV1_BIT_16bit 8
    #define BARRET_DIV2_BIT_16bit 16
    #define DEGREE (1<<LGN)
    #define WL_16bit 16  // sizeof(uint32_t)
    //#define ONE_32bit 0x2AC8
    #define MONT_MU_16bit 12287
    #define MONT_MASK_16bit 0
    #define MONT_MASK_16bit_APPROX 0
    #define ONE_16bit 4091
    #define ONE_INV_16bit 0
    #define L_INTT_16bit 1             // 2nq < 2^{31-L}
   





    //#define ND 349496
    //#define R 268435456
#endif

/*
#define LGN 8          // Degree n=2^LGN
#define ND 0xF7002FFF   // 1/(R-q) mod R
//#define ONE 0x2AC8      // R mod q
#define ONE 1           // ONE 
#define R2MODP 0x1620   // R^2 mod q
#define L 1             // 2nq < 2^{31-L}
#define XLIM 0x2AA9C    // available excess 2^{31}/q   
#define PRIME 12289

#define DEGREE (1<<LGN)
//#define WL (8*sizeof(int32_t))
#define WL 32
*/
/*
#define LGN 8          // Degree n=2^LGN
#define ND 0xF7002FFF   // 1/(R-q) mod R
//#define ONE 0x2AC8      // R mod q
#define ONE 1           // ONE 
#define R2MODP 0x1620   // R^2 mod q
#define L 1             // 2nq < 2^{31-L}
#define XLIM 0x2AA9C    // available excess 2^{31}/q   
#define PRIME 12289

#define DEGREE (1<<LGN)
//#define WL (8*sizeof(int32_t))
#define WL 32
*/



#ifdef __cplusplus
}
#endif

#endif // End header
