#ifndef NTT16BIT_ARMV8_2_NEON_H
#define NTT16BIT_ARMV8_2_NEON_H

#ifdef __cplusplus
extern "C" {
#endif


//extern void asm_muladd(int16_t *res, const int16_t * x, const int16_t *y, const int16_t *z);

extern void asm_shuffle_test_neon(int16_t *x, int16_t *y);
extern void asm_reduction_test_neon(int16_t *x, int16_t *y, int16_t *z, int16_t *q, int16_t *mont);




/******************** PERMULTE *********************************/

//extern void asm_permute_depth1_1024_avx2(int16_t *x);
extern void asm_unpermute_depth1_1024_neon(int16_t *x);
extern void asm_unpermute_depth2_1024_neon(int16_t *x);
extern void asm_unpermute_depth3_1024_neon(int16_t *x);

/********************************** 15bit 1024 **************************************/

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NTT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

extern void asm_ntt0_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt1_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt2_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt3_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt4_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);

extern void asm_ntt0_to_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x, int16_t* y,     const int32_t* INFO);
extern void asm_ntt1_to_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x, int16_t *y,     const int32_t* INFO);
extern void asm_ntt2_to_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x, int16_t *y,     const int32_t* INFO);
extern void asm_ntt3_to_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x, int16_t *y,    const int32_t* INFO);
extern void asm_ntt4_to_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x, int16_t *y,    const int32_t* INFO);




extern void asm_intt0_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt1_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt2_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt3_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt4_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);

extern void asm_intt0_to_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt1_to_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt2_to_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt3_to_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt4_to_mont_15bit_1024_neon(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);



//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MUL  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

// (x,x,y can be callable)

extern void asm_mul0_mont_15bit_1024_neon(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul1_mont_15bit_1024_neon(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul2_mont_15bit_1024_neon(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul3_mont_15bit_1024_neon(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GADGET !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
extern void asm_decom_qnum2_base8_rm8_len3_15bit_1024_neon(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET);

extern void asm_decom_qnum2_base8_rm8_len3_neon(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);



////////////////////////////////////////////////////////////////////// 512 /////////////////////////////////////////////////////////
extern void asm_unpermute_depth1_512_neon(int16_t *x);
extern void asm_unpermute_depth2_512_neon(int16_t *x);
extern void asm_unpermute_depth3_512_neon(int16_t *x);



extern void asm_ntt0_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt1_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt2_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt3_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt4_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);

extern void asm_ntt0_to_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,  int16_t *y, const int32_t* INFO);
extern void asm_ntt1_to_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,  int16_t *y, const int32_t* INFO);
extern void asm_ntt2_to_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_ntt3_to_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_ntt4_to_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);


extern void asm_intt0_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt1_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt2_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt3_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);


extern void asm_intt0_to_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x, int16_t *y, const int32_t* INFO);
extern void asm_intt1_to_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_intt2_to_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x, int16_t *y, const int32_t* INFO);
extern void asm_intt3_to_mont_15bit_512_neon(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);




extern void asm_mul0_mont_15bit_512_neon(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul1_mont_15bit_512_neon(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul2_mont_15bit_512_neon(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul3_mont_15bit_512_neon(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);




extern void asm_decom_qnum2_base8_rm8_len3_15bit_512_neon(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET);


/////////////////TEST
/*
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
