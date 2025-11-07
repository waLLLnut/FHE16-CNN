#ifndef PRE_TABLE_IDX_H
#define PRE_TABLE_IDX_H


// QDATA IDX 
// QDATA[0] => Computing table about Q
// QDATA{1] => Sort of NTT
// QDATA[2] => Sort of INTT
// QDATA[3] => Multiplication values


#define QDATA_Q                         0
#define QDATA_QBIT                      1
#define QDATA_QTWOPOW                   2
#define QDATA_QDEPTH                    3
#define QDATA_WIDTH						4


#define QDATA_Q_MONT                    5
#define QDATA_Q_MONT_ONE                6
#define QDATA_Q_MONT_SQUARE             7
#define QDATA_Q_MONT_INV                8


#define QDATA_Q_MONT_SIGN               9
#define QDATA_Q_MONT_SIGN_ONE           10
#define QDATA_Q_MONT_SIGN_SQUARE        11
#define QDATA_Q_MONT_SIGN_INV           12


#define QDATA_Q_BARR_SHORT              13
#define QDATA_Q_BARR_SHORT_SIGN         14
#define QDATA_Q_BARR_LONG               15
#define QDATA_Q_BARR_LONG_SIGN          16

#define QDATA_NINV                      17
#define QDATA_NINV_SHOUP                18

#define QDATA_NINVPR                    19
#define QDATA_NINVPR_SHOUP              20

#define NTT_END_IDX                     21
#define MUL_END_IDX                     22
#define INTT_END_IDX                    23

#define GADGETQ0_INV_MUL                24
#define GADGETQ0_INV_MULSHOUP           25
#define QDATA_N							26


#define QDATA_LEN                       27


/* N depth w value => 2^N-1 */

#define QDATA_Q_1024_16bit_AVX2_PACKING 64  // 2^6 - 1 = 53, 7,8,9,10's values are saved as packed.  


#endif // MACRO_PARAMS_H
