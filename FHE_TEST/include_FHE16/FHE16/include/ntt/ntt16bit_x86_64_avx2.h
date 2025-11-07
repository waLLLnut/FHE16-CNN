#ifndef NTT16BIT_X86_64_AVX2_H
#define NTT16BIT_X86_64_AVX2_H

#ifdef __cplusplus
extern "C" {
#endif



/******************** PERMULTE *********************************/
extern void asm_permute_depth1_1024_avx2(int16_t *x);

extern void asm_unpermute_depth1_1024_avx2(int16_t *x);
extern void asm_unpermute_depth2_1024_avx2(int16_t *x);
extern void asm_unpermute_depth3_1024_avx2(int16_t *x);


/*************************** 4096 **********************************/


extern void asm_ntt0_to_mont_15bit_4096_avx2(const int16_t *NTTINFO, int16_t * x, int16_t *y,   const int32_t* INFO);
extern void asm_ntt1_to_mont_15bit_4096_avx2(const int16_t *NTTINFO, int16_t * x, int16_t *y,   const int32_t* INFO);
extern void asm_ntt2_to_mont_15bit_4096_avx2(const int16_t *NTTINFO, int16_t * x, int16_t *y,   const int32_t* INFO);
extern void asm_ntt3_to_mont_15bit_4096_avx2(const int16_t *NTTINFO, int16_t * x, int16_t *y,   const int32_t* INFO);
extern void asm_ntt4_to_mont_15bit_4096_avx2(const int16_t *NTTINFO, int16_t * x, int16_t *y,	const int32_t* INFO);
	
extern void asm_intt0_to_mont_15bit_4096_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y,	const int32_t* INFO);
extern void asm_intt1_to_mont_15bit_4096_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_intt2_to_mont_15bit_4096_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_intt3_to_mont_15bit_4096_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_intt4_to_mont_15bit_4096_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MUL  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

// (x,x,y can be callable)
extern void asm_mul0_mont_15bit_4096_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul1_mont_15bit_4096_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul2_mont_15bit_4096_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul3_mont_15bit_4096_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);



extern void asm_fma0_mont_15bit_4096_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma1_mont_15bit_4096_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma2_mont_15bit_4096_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma3_mont_15bit_4096_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);


/********************** 2048 ***********************************/

extern void asm_ntt0_to_mont_15bit_2048_avx2(const int16_t *NTTINFO, int16_t * x, int16_t *y,   const int32_t* INFO);
extern void asm_ntt1_to_mont_15bit_2048_avx2(const int16_t *NTTINFO, int16_t * x, int16_t *y,   const int32_t* INFO);
extern void asm_ntt2_to_mont_15bit_2048_avx2(const int16_t *NTTINFO, int16_t * x, int16_t *y,   const int32_t* INFO);
extern void asm_ntt3_to_mont_15bit_2048_avx2(const int16_t *NTTINFO, int16_t * x, int16_t *y,   const int32_t* INFO);
extern void asm_ntt4_to_mont_15bit_2048_avx2(const int16_t *NTTINFO, int16_t * x, int16_t *y,	const int32_t* INFO);
	
extern void asm_intt0_to_mont_15bit_2048_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y,	const int32_t* INFO);
extern void asm_intt1_to_mont_15bit_2048_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_intt2_to_mont_15bit_2048_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_intt3_to_mont_15bit_2048_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_intt4_to_mont_15bit_2048_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MUL  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

// (x,x,y can be callable)
extern void asm_mul0_mont_15bit_2048_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul1_mont_15bit_2048_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul2_mont_15bit_2048_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul3_mont_15bit_2048_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);


extern void asm_fma0_mont_15bit_2048_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma1_mont_15bit_2048_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma2_mont_15bit_2048_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma3_mont_15bit_2048_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);


/*******************  1024 **************************/

extern void asm_ntt0_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt1_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt2_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt3_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt4_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
	
extern void asm_ntt0_to_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x, int16_t *y,   const int32_t* INFO);
extern void asm_ntt1_to_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x, int16_t *y,   const int32_t* INFO);
extern void asm_ntt2_to_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x, int16_t *y,   const int32_t* INFO);
extern void asm_ntt3_to_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x, int16_t *y,   const int32_t* INFO);
extern void asm_ntt4_to_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x, int16_t *y,	const int32_t* INFO);
	

extern void asm_intt0_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt1_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt2_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt3_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt4_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);

extern void asm_intt0_to_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y,	const int32_t* INFO);
extern void asm_intt1_to_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_intt2_to_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_intt3_to_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_intt4_to_mont_15bit_1024_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MUL  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

// (x,x,y can be callable)
extern void asm_mul0_mont_15bit_1024_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul1_mont_15bit_1024_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul2_mont_15bit_1024_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul3_mont_15bit_1024_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);



extern void asm_fma0_mont_15bit_1024_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma1_mont_15bit_1024_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma2_mont_15bit_1024_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma3_mont_15bit_1024_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);




////////////////////////////////////////////////////////////////////// 512 /////////////////////////////////////////////////////////
extern void asm_unpermute_depth1_512_avx2(int16_t *x);
extern void asm_unpermute_depth2_512_avx2(int16_t *x);
extern void asm_unpermute_depth3_512_avx2(int16_t *x);



extern void asm_ntt0_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt1_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt2_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt3_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_ntt4_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);

extern void asm_ntt0_to_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_ntt1_to_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_ntt2_to_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,    int16_t *y,	const int32_t* INFO);
extern void asm_ntt3_to_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,    int16_t *y,	const int32_t* INFO);
extern void asm_ntt4_to_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,    int16_t *y,	const int32_t* INFO);




extern void asm_intt0_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt1_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt2_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);
extern void asm_intt3_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,     const int32_t* INFO);


extern void asm_intt0_to_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_intt1_to_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,	int16_t *y, const int32_t* INFO);
extern void asm_intt2_to_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,    int16_t *y,	const int32_t* INFO);
extern void asm_intt3_to_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,    int16_t *y,	const int32_t* INFO);
extern void asm_intt4_to_mont_15bit_512_avx2(const int16_t *NTTINFO, int16_t * x,    int16_t *y,	const int32_t* INFO);





extern void asm_mul0_mont_15bit_512_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul1_mont_15bit_512_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul2_mont_15bit_512_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
extern void asm_mul3_mont_15bit_512_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);


extern void asm_fma0_mont_15bit_512_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma1_mont_15bit_512_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma2_mont_15bit_512_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);
extern void asm_fma3_mont_15bit_512_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len);





//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GADGET !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


// C
void C_asm_decom_mont_qnum2_base8_rm8_len3_16bit_avx2(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim, int Ks);
void C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx2(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim, int Ks);
void C_asm_decom_mont_qnum2_base11_rm17_len1_16bit_avx2(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim, int Ks);
	





//extern void asm_decom_qnum2_base8_rm8_len3_15bit_1024(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);





extern void asm_decom_qnum2_base8_rm8_len3(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);
extern void asm_decom_mont_qnum2_base8_rm8_len3(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);
extern void asm_decom_mont_qnum2_base9_rm10_len2(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);
extern void asm_decom_mont_qnum2_base11_rm17_len1(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);
	






extern void asm_decomFloat_mont_qnum2_base8_rm8_len3_avx2(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);
extern void asm_decomFloat_mont_qnum2_base9_rm10_len2_avx2(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);
extern void asm_decomFloat_mont_qnum2_base11_rm17_len1_avx2(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);
	
extern void asm_decomInt_mont_qnum2_base8_rm8_len3_avx2(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);
extern void asm_decomInt_mont_qnum2_base9_rm10_len2_avx2(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);
extern void asm_decomInt_mont_qnum2_base11_rm17_len1_avx2(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);
	

/////////////////////////////////////// FMA TOTAL //////////////////////////////////////////////
extern void asm_fma0_mont_15bit_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len, int N);
extern void asm_fma1_mont_15bit_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len, int N);
extern void asm_fma2_mont_15bit_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len, int N);
extern void asm_fma3_mont_15bit_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO, int len, int N);



//asm_decom_qnum2_base8_rm8_len3

//extern void asm_decom_qnum2_base8_rm8_len3_15bit_512(int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim);


/////////////////TEST

extern void asm_shuffle_cannonical2_avx2(int16_t * x);
extern void asm_shuffle_cannonical4_avx2(int16_t * x);
extern void asm_shuffle_cannonical8_avx2(int16_t * x);
extern void asm_shuffle_cannonical16_avx2(int16_t * x);

extern void asm_wshift(int16_t * x);
extern void asmFloat_test(int16_t * x, float *y);

extern void asmMulh_32bit_test(int32_t * x, int32_t *y, int32_t *res1, int32_t *res2);






/***************************************** C *********************************/

void C_asm_mul0_mont_15bit_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
void C_asm_mul1_mont_15bit_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
void C_asm_mul2_mont_15bit_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);
void C_asm_mul3_mont_15bit_avx2(int16_t *res, int16_t * x, int16_t *y, const int16_t *MULINFO, const int32_t* INFO);








/*

//extern void asm_x86_64_ntt32bit(uint32_t *x, uint32_t *TB, uint32_t N, uint32_t Q, uint32_t Mont_mu);
//extern void volatile __attribute__((optimize("O1"))) asm_ntt_16bit_1024_x86_64_avx2(int16_t *x, const volatile int16_t* datas);
extern void asm_ntt_16bit_1024_x86_64_avx2(int16_t *x, const uint16_t* datas);

// Prepare
extern void asm_ntt_inplace_16bit_1024_x86_64_avx2(int16_t *x, const uint16_t* datas);
extern void asm_ntt_inplace_prepare_16bit_1024_x86_64_avx2(int16_t *x, int16_t *y, const uint16_t* datas);

// Multiplication
extern void asm_inplace_ntt_mul_intt_16bit_1024_x86_64_avx2(int16_t *x, int16_t *y, const uint16_t* datas);
extern void asm_inplace_incomplete1_school_ntt_mul_intt_16bit_1024_x86_64_avx2(int16_t *x, int16_t *y, const uint16_t *datas);
extern void asm_inplace_incomplete2_school_ntt_mul_intt_16bit_1024_x86_64_avx2(int16_t *x, int16_t *y, const uint16_t *datas);
extern void asm_inplace_incomplete3_school_ntt_mul_intt_16bit_1024_x86_64_avx2(int16_t *x, int16_t *y, const uint16_t *datas);


// Linearity
extern void asm_inplace_ntt_linear_intt_16bit_1024_x86_64_avx2(int16_t *x, const int16_t *y, const uint16_t* datas);


//extern void asm_inplace_ntt_mul_intt_16bit_1024_x86_64_avx2(int16_t *x, const uint16_t *y, const uint16_t* datas);
extern void asm_MontTest(int16_t *x, int16_t *y, const uint16_t* datas);


extern void asm_SignedMontTest(int16_t *x, int16_t *y, int16_t *q, int16_t *mu);
extern void asm_SignedBarretTest(int16_t *x, int16_t *q, int16_t *mu);
extern void asm_SignedShoupTest(int16_t *x, int16_t *y,  int16_t *y_prime, int16_t * q);






extern void asm_Perm_Test(uint16_t *x , uint16_t *y);
extern void asm_BarretTest(int16_t *x, const uint16_t *y);
*/
/* Input register rdi , rsi, rdx, rcx, r8, r9
 * uint32_t *x      : edi 
 * uint32_t *tb     : esi 
 * uint32_t N       : edx
 * uint32_t Q       : ecx
 * uint32_t Mont_mu : r8d
 */


#ifdef __cplusplus
}
#endif

#endif // End header
