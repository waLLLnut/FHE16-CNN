#ifndef NTT16BIT_X86_64_AVX512_H
#define NTT16BIT_X86_64_AVX512_H

#ifdef __cplusplus
extern "C" {
#endif


//extern void asm_muladd(int16_t *res, const int16_t * x, const int16_t *y, const int16_t *z);

extern void asm_shuffle_test_avx512(int16_t *x, int16_t *y);



/******************** PERMULTE *********************************/

//extern void asm_permute_depth1_1024_avx2(int16_t *x);
extern void asm_unpermute_depth1_1024_avx512(int16_t *x);
extern void asm_unpermute_depth2_1024_avx512(int16_t *x);
extern void asm_unpermute_depth3_1024_avx512(int16_t *x);




/************************************ 15bit 4096 ********************************************/

extern void asm_ntt0_to_mont_15bit_4096_avx512(const int16_t *NTTINFO, int16_t * x, int16_t* y,     const int32_t* INFO);
extern void asm_ntt1_to_mont_15bit_4096_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y,     const int32_t* INFO);
extern void asm_ntt2_to_mont_15bit_4096_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y,     const int32_t* INFO);
extern void asm_ntt3_to_mont_15bit_4096_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y,    const int32_t* INFO);
extern void asm_ntt4_to_mont_15bit_4096_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y,    const int32_t* INFO);

extern void asm_intt0_to_mont_15bit_4096_avx512(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt1_to_mont_15bit_4096_avx512(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt2_to_mont_15bit_4096_avx512(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt3_to_mont_15bit_4096_avx512(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt4_to_mont_15bit_4096_avx512(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MUL  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

// (x,x,y can be callable)

extern void asm_mul0_mont_15bit_4096_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul1_mont_15bit_4096_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul2_mont_15bit_4096_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul3_mont_15bit_4096_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);

extern void asm_fma0_mont_15bit_4096_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma1_mont_15bit_4096_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma2_mont_15bit_4096_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma3_mont_15bit_4096_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);




/*********************************** 15bit 2048 ******************************************/

extern void asm_ntt0_to_mont_15bit_2048_avx512(const int16_t *NTTINFO, int16_t * x, int16_t* y,     const int32_t* INFO);
extern void asm_ntt1_to_mont_15bit_2048_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y,     const int32_t* INFO);
extern void asm_ntt2_to_mont_15bit_2048_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y,     const int32_t* INFO);
extern void asm_ntt3_to_mont_15bit_2048_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y,    const int32_t* INFO);
extern void asm_ntt4_to_mont_15bit_2048_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y,    const int32_t* INFO);

extern void asm_intt0_to_mont_15bit_2048_avx512(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt1_to_mont_15bit_2048_avx512(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt2_to_mont_15bit_2048_avx512(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt3_to_mont_15bit_2048_avx512(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt4_to_mont_15bit_2048_avx512(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MUL  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

// (x,x,y can be callable)

extern void asm_mul0_mont_15bit_2048_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul1_mont_15bit_2048_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul2_mont_15bit_2048_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul3_mont_15bit_2048_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);

extern void asm_fma0_mont_15bit_2048_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma1_mont_15bit_2048_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma2_mont_15bit_2048_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma3_mont_15bit_2048_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);


/********************************** 15bit 1024 **************************************/


extern void asm_ntt0_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt1_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt2_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt3_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt4_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);

extern void asm_ntt0_to_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x, int16_t* y,     const int32_t* INFO);
extern void asm_ntt1_to_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y,     const int32_t* INFO);
extern void asm_ntt2_to_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y,     const int32_t* INFO);
extern void asm_ntt3_to_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y,    const int32_t* INFO);
extern void asm_ntt4_to_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y,    const int32_t* INFO);




extern void asm_intt0_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt1_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt2_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt3_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt4_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);

extern void asm_intt0_to_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt1_to_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt2_to_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt3_to_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);
extern void asm_intt4_to_mont_15bit_1024_avx512(const int16_t *NTTINFO, int16_t * x,     int16_t *y, const int32_t* INFO);



//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MUL  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

// (x,x,y can be callable)

extern void asm_mul0_mont_15bit_1024_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul1_mont_15bit_1024_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul2_mont_15bit_1024_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul3_mont_15bit_1024_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);

extern void asm_fma0_mont_15bit_1024_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma1_mont_15bit_1024_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma2_mont_15bit_1024_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma3_mont_15bit_1024_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);


///////////////////////////////////////// FMA

extern void asm_fma0_mont_15bit_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len, int N);
extern void asm_fma1_mont_15bit_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len, int N);
extern void asm_fma2_mont_15bit_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len, int N);
extern void asm_fma3_mont_15bit_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len, int N);





////////////////////////////////////////////////////////////////////// 512 /////////////////////////////////////////////////////////
extern void asm_unpermute_depth1_512_avx512(int16_t *x);
extern void asm_unpermute_depth2_512_avx512(int16_t *x);
extern void asm_unpermute_depth3_512_avx512(int16_t *x);



extern void asm_ntt0_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt1_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt2_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt3_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt4_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);



extern void asm_ntt0_to_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,  int16_t *y, const int32_t* INFO);
extern void asm_ntt1_to_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,  int16_t *y, const int32_t* INFO);
extern void asm_ntt2_to_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_ntt3_to_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_ntt4_to_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);


extern void asm_ntt0_to_mont_14bit_512_avx512(const int16_t *NTTINFO, int16_t * x,  int16_t *y, const int32_t* INFO);
extern void asm_ntt1_to_mont_14bit_512_avx512(const int16_t *NTTINFO, int16_t * x,  int16_t *y, const int32_t* INFO);
extern void asm_ntt2_to_mont_14bit_512_avx512(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_ntt3_to_mont_14bit_512_avx512(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_ntt4_to_mont_14bit_512_avx512(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);



extern void asm_intt0_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt1_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt2_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt3_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);


extern void asm_intt0_to_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y, const int32_t* INFO);
extern void asm_intt1_to_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_intt2_to_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y, const int32_t* INFO);
extern void asm_intt3_to_mont_15bit_512_avx512(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);

extern void asm_intt0_to_mont_14bit_512_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y, const int32_t* INFO);
extern void asm_intt1_to_mont_14bit_512_avx512(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_intt2_to_mont_14bit_512_avx512(const int16_t *NTTINFO, int16_t * x, int16_t *y, const int32_t* INFO);
extern void asm_intt3_to_mont_14bit_512_avx512(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);




extern void asm_mul0_mont_15bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul1_mont_15bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul2_mont_15bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul3_mont_15bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);

extern void asm_mul0_mont_14bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul1_mont_14bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul2_mont_14bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul3_mont_14bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);




extern void asm_mul0_mont_14bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul1_mont_14bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul2_mont_14bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul3_mont_14bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);



extern void asm_fma0_mont_15bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma1_mont_15bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma2_mont_15bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma3_mont_15bit_512_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);



//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GADGET !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
extern void asm_decom_qnum2_base8_rm8_len3_15bit_1024_avx512(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET);
extern void asm_decom_qnum2_base8_rm8_len3_15bit_512_avx512(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET);

extern void asm_decom_mont_qnum2_base8_rm8_len3_15bit_512_avx512(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET);



extern void asm_decom_qnum2_base8_rm8_len3_avx512(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);

extern void asm_decom_mont_qnum2_base8_rm8_len3_avx512(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);
extern void asm_decom_mont_qnum2_base9_rm10_len2_avx512(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);
extern void asm_decom_mont_qnum2_base11_rm17_len1_avx512(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);


extern void asm_decomFloat_mont_qnum2_base8_rm8_len3_avx512(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);
extern void asm_decomFloat_mont_qnum2_base9_rm10_len2_avx512(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);
extern void asm_decomFloat_mont_qnum2_base11_rm17_len1_avx512(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);



extern void asmFloat_test_avx512(int16_t * x, float *y);
extern void asmMulh_32bit_test_avx512(int32_t * x, int32_t *y, int32_t *res1, int32_t *res2);

void C_asm_decom_mont_qnum2_base8_rm8_len3_16bit_avx512(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim, int Ks);
void C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx512(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim, int Ks);
void C_asm_decom_mont_qnum2_base11_rm17_len1_16bit_avx512(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim, int Ks);


/////////////////TEST
/*
extern void asm_shuffle_cannonical2_avx2(int16_t * x);
extern void asm_shuffle_cannonical4_avx2(int16_t * x);
extern void asm_shuffle_cannonical8_avx2(int16_t * x);
extern void asm_shuffle_cannonical16_avx2(int16_t * x);

extern void asm_wshift(int16_t * x);
*/

/************************ Vec Mat Mul ******************************/
void C_asm_mul0_mont_VecMat_DEPTH0_14bit_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int out_len, int in_len);

void C_asm_mul0_mont_VecMat_DEPTH0_14bit_2_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int out_len, int in_len);
void C_asm_mul0_mont_VecMat_DEPTH0_14bit_3_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int out_len, int in_len);
void C_asm_mul0_mont_VecMat_DEPTH0_14bit_4_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int out_len, int in_len);
void C_asm_mul0_mont_VecMat_DEPTH0_14bit_5_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int out_len, int in_len);
void C_asm_mul0_mont_VecMat_DEPTH0_14bit_6_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int out_len, int in_len);
void C_asm_mul0_mont_VecMat_DEPTH0_14bit_7_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int out_len, int in_len);
void C_asm_mul0_mont_VecMat_DEPTH0_14bit_8_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int out_len, int in_len);

void C_asm_mul0_mont_VecMat_DEPTH0_14bit_2_avx512_VNNI(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int out_len, int in_len);
void C_asm_mul0_mont_VecMat_DEPTH0_14bit_3_avx512_VNNI(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int out_len, int in_len);
void C_asm_mul0_mont_VecMat_DEPTH0_14bit_4_avx512_VNNI(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int out_len, int in_len);
void C_asm_mul0_mont_VecMat_DEPTH0_14bit_5_avx512_VNNI(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int out_len, int in_len);
void C_asm_mul0_mont_VecMat_DEPTH0_14bit_6_avx512_VNNI(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int out_len, int in_len);
void C_asm_mul0_mont_VecMat_DEPTH0_14bit_7_avx512_VNNI(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int out_len, int in_len);
void C_asm_mul0_mont_VecMat_DEPTH0_14bit_8_avx512_VNNI(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int out_len, int in_len);








/******************************* C code *******************************************/
void C_asm_mul0_mont_14bit_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);




void C_asm_mul0_mont_15bit_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
void C_asm_mul1_mont_15bit_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
void C_asm_mul2_mont_15bit_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
void C_asm_mul3_mont_15bit_avx512(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);


#ifdef __cplusplus
}
#endif

#endif // End header
