#ifndef NTT16BIT_AVX1_X86_64_H
#define NTT16BIT_AVX1_X86_64_H

#ifdef __cplusplus
extern "C" {
#endif

extern void asm_shuffle_test_avx1(int16_t *x, int16_t *y);



/******************** PERMULTE *********************************/
/*
extern void asm_permute_depth1_1024_avx2(int16_t *x);
extern void asm_unpermute_depth1_1024_avx2(int16_t *x);
extern void asm_unpermute_depth2_1024_avx2(int16_t *x);
extern void asm_unpermute_depth3_1024_avx2(int16_t *x);
*/





/******************* 16bit USING ASM functions **************************/

/** NTT **/    
/*
extern void asm_ntt0_16bit_1024_x86_64_avx2(int16_t *x, const uint16_t* datas);
extern void asm_ntt0_16bit_1024_x86_64_avx2(int16_t *x, const uint16_t* datas);
*/



//* INTT **/    
/*
extern void asm_intt0_16bit_1024_x86_64_avx2(int16_t *x, const uint16_t* datas);
*/

// Mont Multiplicaton
//extern void asm_MulMontSafe0_16bit_1024_x86_64_avx2(int16_t *to, int16_t *from, const uint16_t* datas);

/********************************** 15bit 1024 **************************************/
////// ADDTION

//extern void asm_add_1024_avx2(int16_t *to, int16_t *f1, int16_t *F2, int16_t *Q);



//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NTT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
/*
extern void asm_ntt0_shoup_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_ntt1_shoup_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_ntt2_shoup_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_ntt3_shoup_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_ntt4_shoup_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);

extern void asm_ntt0_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_ntt1_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_ntt2_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_ntt3_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_ntt4_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
	
extern void asm_ntt2_mont_tweak_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_ntt3_mont_tweak_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_ntt4_mont_tweak_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);


extern void asm_intt0_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_intt1_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_intt2_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_intt3_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_intt4_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);


extern void asm_intt2_mont_tweak_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_intt3_mont_tweak_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_intt4_mont_tweak_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);

*/



//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! INTT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
/*
extern void asm_intt0_shoup_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,    const int16_t* INFO);
extern void asm_intt1_shoup_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,    const int16_t* INFO);
extern void asm_intt2_shoup_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,    const int16_t* INFO);
extern void asm_intt3_shoup_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,    const int16_t* INFO);
extern void asm_intt4_shoup_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,    const int16_t* INFO);

*/




//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MUL  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

// (x,x,y can be callable)
/*
extern void asm_mul0_mont_15bit_1024_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int16_t* INFO);
extern void asm_mul1_mont_15bit_1024_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int16_t* INFO);
extern void asm_mul2_mont_15bit_1024_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int16_t* INFO);
extern void asm_mul3_mont_15bit_1024_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int16_t* INFO);


//extern void asm_mul2_mont_TWEAK_15bit_1024_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t* INFO);
//extern void asm_mul3_mont_TWEAK_15bit_1024_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t* INFO);


//asm_mul2_mont_TWEAK_15bit_1024_avx2

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GADGET !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
extern void asm_decom_qnum2_base8_rm8_len3_15bit_1024(int16_t *from, int16_t * to, const int16_t *INFO, const int16_t *GADGET);




////////////////////////////////////////////////////////////////////// 512 /////////////////////////////////////////////////////////
extern void asm_unpermute_depth1_512_avx2(int16_t *x);
extern void asm_unpermute_depth2_512_avx2(int16_t *x);
extern void asm_unpermute_depth3_512_avx2(int16_t *x);



extern void asm_ntt0_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_ntt1_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_ntt2_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_ntt3_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_ntt4_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);

extern void asm_intt0_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_intt1_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_intt2_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);
extern void asm_intt3_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int16_t* INFO);

extern void asm_mul0_mont_15bit_512_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int16_t* INFO);
extern void asm_mul1_mont_15bit_512_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int16_t* INFO);
extern void asm_mul2_mont_15bit_512_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int16_t* INFO);
extern void asm_mul3_mont_15bit_512_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int16_t* INFO);




extern void asm_decom_qnum2_base8_rm8_len3_15bit_512(int16_t *from, int16_t * to, const int16_t *INFO, const int16_t *GADGET);


/////////////////TEST

extern void asm_shuffle_cannonical2_avx2(int16_t * x);
extern void asm_shuffle_cannonical4_avx2(int16_t * x);
extern void asm_shuffle_cannonical8_avx2(int16_t * x);
extern void asm_shuffle_cannonical16_avx2(int16_t * x);

extern void asm_wshift(int16_t * x);
*/



/**************************************************************************/


#ifdef __cplusplus
}
#endif

#endif // End header
