#ifndef MULTIPLICATION_MACRO_H_EF
#define MULTIPLICATION_MACRO_H_EF

#include<CMAKEPARAM.h>

#ifndef __cplusplus
#include<stdint.h>
#endif

#if CPUTYPE == 1 // IF LINUX
	#include<stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


#if AVXTYPE == 3
	#include <immintrin.h> 
#endif
#include <param.h>
#include <reduction.h>
#include <arithmetic.h>
#include <ntt.h>



// ASM _TEST!!
#if AVXTYPE == 2 // IF LINUX
	#include<ntt16bit_x86_64_avx2.h>
#elif AVXTYPE == 3
	#include<ntt16bit_x86_64_avx512.h>

#elif AVXTYPE == 5
	#include<ntt16bit_armv8_2_a_neon.h>
#endif

#if	AVXTYPE == 5
	#define NTT32(NTTINFO, x, INFO, N)								asm_ntt0_to_mont_31bit_1024_neon(NTTINFO, x, x, INFO)
	#define NTTTo32(NTTINFO, x, y, INFO, N)							asm_ntt0_to_mont_31bit_1024_neon(NTTINFO, x, y, INFO)
	#define INTT32(NTTINFO, x, INFO, N)								asm_intt0_to_mont_31bit_1024_neon(NTTINFO, x, x, INFO)
	#define INTTTo32(NTTINFO, x, y, INFO, N)						asm_intt0_to_mont_31bit_1024_neon(NTTINFO, x, y, INFO)
	#define PointMulMont_32bit(out, x, y, MULINFO, INFO, N)			asm_mul0_mont_32bit_1024_neon(out, x, y, MULINFO, INFO)
	

#else
	#define NTT32(NTTINFO, x, INFO, N)								ntt_to_32bit(NTTINFO, x, x, INFO)
	#define NTTTo32(NTTINFO, x, y, INFO, N)							ntt_to_32bit(NTTINFO, x, y, INFO)
	#define INTT32(NTTINFO, x, INFO, N)								intt_to_32bit(NTTINFO, x, x, INFO)
	#define INTTTo32(NTTINFO, x, y, INFO, N)						intt_to_32bit(NTTINFO, x, y, INFO)
	#define PointMulMont_32bit(out, x, y, MULINFO, INFO, N)			Mul0MontSchool_32bit(out, x, y, MULINFO, INFO, N)

#endif


#if AVXTYPE == 0
	// 16bit
	/*
	#define NTT16(NTTINFO, x, INFO, N)							ntt_to_16bit3(NTTINFO, x, x, INFO, N)
	#define NTTTo16(NTTINFO, x, y, INFO, N)						ntt_to_16bit3(NTTINFO, x, y, INFO, N)
	#define INTT16(NTTINFO, x, INFO, N)							intt_to_16bit3(NTTINFO, x, x,INFO, N)
	#define INTTTo16(NTTINFO, x, y, INFO, N)					intt_to_16bit3(NTTINFO, x, y, INFO, N)
	*/
	
	#define NTT16(NTTINFO, x, INFO, N)							ntt_to_16bit4_unperm(NTTINFO, x, x, INFO)
	#define NTTTo16(NTTINFO, x, y, INFO, N)						ntt_to_16bit4_unperm(NTTINFO, x, y, INFO)
	#define INTT16(NTTINFO, x, INFO, N)							intt_to_16bit4_unperm(NTTINFO, x, x,INFO)
	#define INTTTo16(NTTINFO, x, y, INFO, N)					intt_to_16bit4_unperm(NTTINFO, x, y, INFO)
	
	#if INCOMPLETE_DEPTH_BIN == 0	
	//////////// 32bit
		#define PointMulMont_32bit(out, x, y, MULINFO, INFO, N)     Mul0MontSchool_32bit(out, x, y, MULINFO, INFO)
		#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)     Mul0MontSchool_16bit(out, x, y, MULINFO, INFO)

	#elif INCOMPLETE_DEPTH_BIN == 1	
		#define PointMulMont_32bit(out, x, y, MULINFO, INFO, N)     Mul1MontSchool_32bit(out, x, y, MULINFO, INFO)
		#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)     Mul1MontSchool_16bit(out, x, y, MULINFO, INFO)

	#elif INCOMPLETE_DEPTH_BIN == 2	
		#define PointMulMont_32bit(out, x, y, MULINFO, INFO, N)     Mul2MontSchool_32bit(out, x, y, MULINFO, INFO)
		#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)     Mul_any_MontSchool_16bit(out, x, y, MULINFO, INFO, 2)

	#elif INCOMPLETE_DEPTH_BIN == 3	
		#define PointMulMont_32bit(out, x, y, MULINFO, INFO, N)     Mul3MontSchool_32bit(out, x, y, MULINFO, INFO)
		#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)     Mul_any_MontSchool_16bit(out, x, y, MULINFO, INFO, 3)
	#endif	

#else
	
	// NTT & MUL & INTT
	// Multiplication 
	#if INCOMPLETE_DEPTH_BIN == 0	
		//////////// 32bit
		#define PointMulMont_32bit(out, x, y, MULINFO, INFO, N)     Mul0MontSchool_32bit(out, x, y, MULINFO, INFO)
		#if AVXTYPE == 5
			#if N_RLWE_BIN == 4096
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt0_to_mont_15bit_4096_neon(NTTINFO, x, y, INFO)
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt0_to_mont_15bit_4096_neon(NTTINFO, x, x,  INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt0_to_mont_15bit_4096_neon(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt0_to_mont_15bit_4096_neon(NTTINFO, x, x, INFO)
				#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul0_mont_15bit_4096_neon(out, x, y, MULINFO, INFO)
			#elif N_RLWE_BIN == 2048
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt0_to_mont_15bit_2048_neon(NTTINFO, x, y, INFO)
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt0_to_mont_15bit_2048_neon(NTTINFO, x, x,  INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt0_to_mont_15bit_2048_neon(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt0_to_mont_15bit_2048_neon(NTTINFO, x, x, INFO)
				#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul0_mont_15bit_2048_neon(out, x, y, MULINFO, INFO)
			#elif N_RLWE_BIN == 1024
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt0_to_mont_15bit_1024_neon(NTTINFO, x, y, INFO)
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt0_to_mont_15bit_1024_neon(NTTINFO, x, x,  INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt0_to_mont_15bit_1024_neon(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt0_to_mont_15bit_1024_neon(NTTINFO, x, x, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul0_mont_15bit_1024_neon(out, x, y, MULINFO, INFO)
				
				#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		C_asm_mul0_mont_15bit_neon(out, x, y, MULINFO, INFO)
			
			#elif N_RLWE_BIN == 512

				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt0_to_mont_15bit_512_neon(NTTINFO, x, y, INFO)
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt0_to_mont_15bit_512_neon(NTTINFO, x, x,  INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt0_to_mont_15bit_512_neon(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt0_to_mont_15bit_512_neon(NTTINFO, x, x, INFO)
				#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul0_mont_15bit_512_neon(out, x, y, MULINFO, INFO)
		
			#endif
			//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma0_mont_15bit_1024_neon(out, x, y, MULINFO, INFO, len, N*len)


		// 16bit
		#elif AVXTYPE == 3  
			
			#if N_RLWE_BIN == 4096
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt0_to_mont_15bit_4096_avx512(NTTINFO, x, y, INFO)
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt0_to_mont_15bit_4096_avx512(NTTINFO, x, x,  INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt0_to_mont_15bit_4096_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt0_to_mont_15bit_4096_avx512(NTTINFO, x, x, INFO)
			#elif N_RLWE_BIN == 2048
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt0_to_mont_15bit_2048_avx512(NTTINFO, x, y, INFO)
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt0_to_mont_15bit_2048_avx512(NTTINFO, x, x,  INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt0_to_mont_15bit_2048_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt0_to_mont_15bit_2048_avx512(NTTINFO, x, x, INFO)
			#elif N_RLWE_BIN == 1024
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt0_to_mont_15bit_1024_avx512(NTTINFO, x, y, INFO)
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt0_to_mont_15bit_1024_avx512(NTTINFO, x, x,  INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt0_to_mont_15bit_1024_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt0_to_mont_15bit_1024_avx512(NTTINFO, x, x, INFO)
			
			#elif N_RLWE_BIN == 512
				
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt0_to_mont_15bit_512_avx512(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt0_to_mont_15bit_512_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt0_to_mont_15bit_512_avx512(NTTINFO, x, x,INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt0_to_mont_15bit_512_avx512(NTTINFO, x, y, INFO)
			
				/*
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt0_to_mont_14bit_512_avx512(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt0_to_mont_14bit_512_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt0_to_mont_14bit_512_avx512(NTTINFO, x, x,INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt0_to_mont_14bit_512_avx512(NTTINFO, x, y, INFO)
				#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul0_mont_14bit_512_avx512(out, x, y, MULINFO, INFO)
				*/

				//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma0_mont_15bit_512_avx512(out, x, y, MULINFO, INFO, len)
				//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma0_mont_15bit_avx512(out, x, y, MULINFO, INFO, len*N, N)
	


			#endif
			#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma0_mont_15bit_avx512(out, x, y, MULINFO, INFO, len*N, N)
			#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		C_asm_mul0_mont_15bit_avx512(out, x, y, MULINFO, INFO)
			
		#elif AVXTYPE == 2  
				
			#if N_RLWE_BIN == 4096
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt0_mont_15bit_4096_avx2(NTTINFO, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt0_to_mont_15bit_4096_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt0_mont_15bit_4096_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt0_to_mont_15bit_4096_avx2(NTTINFO, x, y, INFO)
				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma0_mont_15bit_4096_avx2(out, x, y, MULINFO, INFO, len)

			#elif N_RLWE_BIN == 2048
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt0_mont_15bit_2048_avx2(NTTINFO, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt0_to_mont_15bit_2048_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt0_mont_15bit_2048_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt0_to_mont_15bit_2048_avx2(NTTINFO, x, y, INFO)
				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma0_mont_15bit_2048_avx2(out, x, y, MULINFO, INFO, len)

			#elif N_RLWE_BIN == 1024
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt0_mont_15bit_1024_avx2(NTTINFO, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt0_to_mont_15bit_1024_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt0_mont_15bit_1024_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt0_to_mont_15bit_1024_avx2(NTTINFO, x, y, INFO)
				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma0_mont_15bit_1024_avx2(out, x, y, MULINFO, INFO, len)

			#elif  N_RLWE_BIN == 512
				
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt0_to_mont_15bit_512_avx2(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt0_to_mont_15bit_512_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt0_mont_15bit_512_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt0_to_mont_15bit_512_avx2(NTTINFO, x, y, INFO)
			
				/*
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt0_to_mont_14bit_512_avx2(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt0_to_mont_14bit_512_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt0_mont_14bit_512_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt0_to_mont_14bit_512_avx2(NTTINFO, x, y, INFO)
				#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul0_mont_14bit_512_avx2(out, x, y, MULINFO, INFO)
				*/

				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma0_mont_15bit_512_avx2(out, x, y, MULINFO, INFO, len)


			#endif
			//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma0_mont_15bit_avx2(out, x, y, MULINFO, INFO, len*N, N)
			//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma0_mont_15bit_1024_avx2(out, x, y, MULINFO, INFO, len)
			#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		C_asm_mul0_mont_15bit_avx2(out, x, y, MULINFO, INFO)
				

		#elif AVXTYPE == 1
		#else
			printf("STH WRONG !!");
			abort();
		#endif

	#elif  INCOMPLETE_DEPTH_BIN == 1
		#define PointMulMont_32bit(out, x, y, MULINFO, INFO, N)     Mul1MontSchool_32bit(out, x, y, MULINFO, INFO)

		#if AVXTYPE == 3  
			
			#if N_RLWE_BIN == 4096
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt1_to_mont_15bit_4096_avx512(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt1_to_mont_15bit_4096_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt1_to_mont_15bit_4096_avx512(NTTINFO, x, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt1_to_mont_15bit_4096_avx512(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul1_mont_15bit_4096_avx512(out, x, y, MULINFO, INFO)
				//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma1_mont_15bit_4096_avx512(out, x, y, MULINFO, INFO, len)

			#elif N_RLWE_BIN == 2048
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt1_to_mont_15bit_2048_avx512(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt1_to_mont_15bit_2048_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt1_to_mont_15bit_2048_avx512(NTTINFO, x, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt1_to_mont_15bit_2048_avx512(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul1_mont_15bit_2048_avx512(out, x, y, MULINFO, INFO)
		

				//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma1_mont_15bit_2048_avx512(out, x, y, MULINFO, INFO, len)
			#elif N_RLWE_BIN == 1024
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt1_to_mont_15bit_1024_avx512(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt1_to_mont_15bit_1024_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt1_to_mont_15bit_1024_avx512(NTTINFO, x, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt1_to_mont_15bit_1024_avx512(NTTINFO, x, y, INFO)
				#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul1_mont_15bit_1024_avx512(out, x, y, MULINFO, INFO)
				//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma1_mont_15bit_1024_avx512(out, x, y, MULINFO, INFO, len)


			#elif N_RLWE_BIN == 512
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt1_to_mont_15bit_512_avx512(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt1_to_mont_15bit_512_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt1_to_mont_15bit_512_avx512(NTTINFO, x, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt1_to_mont_15bit_512_avx512(NTTINFO, x, y, INFO)
				#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul1_mont_15bit_512_avx512(out, x, y, MULINFO, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		C_asm_mul1_mont_15bit_avx512(out, x, y, MULINFO, INFO)
				
				//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma1_mont_15bit_512_avx512(out, x, y, MULINFO, INFO, len)
			#endif
			#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma1_mont_15bit_avx512(out, x, y, MULINFO, INFO, len*N, N)
			#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		C_asm_mul1_mont_15bit_avx512(out, x, y, MULINFO, INFO)
	


		#elif AVXTYPE == 2  
			#if N_RLWE_BIN == 4096
				#define NTT16(NTTINFO, x, INFO, N)								asm_ntt1_mont_15bit_4096_avx2(NTTINFO, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)							asm_ntt1_to_mont_15bit_4096_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)								asm_intt1_mont_15bit_4096_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)						asm_intt1_to_mont_15bit_4096_avx2(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)			asm_mul1_mont_15bit_4096_avx2(out, x, y, MULINFO, INFO)
				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma1_mont_15bit_4096_avx2(out, x, y, MULINFO, INFO, len)

			#elif N_RLWE_BIN == 2048
				#define NTT16(NTTINFO, x, INFO, N)								asm_ntt1_mont_15bit_2048_avx2(NTTINFO, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)							asm_ntt1_to_mont_15bit_2048_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)								asm_intt1_mont_15bit_2048_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)						asm_intt1_to_mont_15bit_2048_avx2(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)			asm_mul1_mont_15bit_2048_avx2(out, x, y, MULINFO, INFO)
				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma1_mont_15bit_2048_avx2(out, x, y, MULINFO, INFO, len)

			#elif N_RLWE_BIN == 1024
				#define NTT16(NTTINFO, x, INFO, N)								asm_ntt1_mont_15bit_1024_avx2(NTTINFO, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)							asm_ntt1_to_mont_15bit_1024_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)								asm_intt1_mont_15bit_1024_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)						asm_intt1_to_mont_15bit_1024_avx2(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)			asm_mul1_mont_15bit_1024_avx2(out, x, y, MULINFO, INFO)
				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma1_mont_15bit_1024_avx2(out, x, y, MULINFO, INFO, len)


			#elif  N_RLWE_BIN == 512
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt1_mont_15bit_512_avx2(NTTINFO, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt1_to_mont_15bit_512_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt1_mont_15bit_512_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt1_to_mont_15bit_512_avx2(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul1_mont_15bit_512_avx2(out, x, y, MULINFO, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		C_asm_mul1_mont_15bit_avx2(out, x, y, MULINFO, INFO)
				
				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma1_mont_15bit_512_avx2(out, x, y, MULINFO, INFO, len)

			#endif
			//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma1_mont_15bit_avx2(out, x, y, MULINFO, INFO, len*N, N)
			#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		C_asm_mul1_mont_15bit_avx2(out, x, y, MULINFO, INFO)
			

		#elif AVXTYPE == 1

		#endif

	#elif  INCOMPLETE_DEPTH_BIN == 2
		//#define PointMulMont_32bit(out, x, y, MULINFO, INFO, N)     Mul2MontSchool_32bit(out, x, y, MULINFO, INFO)
		#define PointMulMont_32bit(out, x, y, MULINFO, INFO, N)     Mul1MontSchool_32bit(out, x, y, MULINFO, INFO)

		#if AVXTYPE == 3  
			#if N_RLWE_BIN == 4096
				#define NTT16(NTTINFO, x, INFO, N)								asm_ntt2_to_mont_15bit_4096_avx512(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)							asm_ntt2_to_mont_15bit_4096_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)								asm_intt2_to_mont_15bit_4096_avx512(NTTINFO, x, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)						asm_intt2_to_mont_15bit_4096_avx512(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)			asm_mul2_mont_15bit_4096_avx512(out, x, y, MULINFO, INFO)
				//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma2_mont_15bit_4096_avx512(out, x, y, MULINFO, INFO, len)

			#elif N_RLWE_BIN == 2048
				#define NTT16(NTTINFO, x, INFO, N)								asm_ntt2_to_mont_15bit_2048_avx512(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)							asm_ntt2_to_mont_15bit_2048_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)								asm_intt2_to_mont_15bit_2048_avx512(NTTINFO, x, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)						asm_intt2_to_mont_15bit_2048_avx512(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)			asm_mul2_mont_15bit_2048_avx512(out, x, y, MULINFO, INFO)
				//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma2_mont_15bit_2048_avx512(out, x, y, MULINFO, INFO, len)

			#elif N_RLWE_BIN == 1024
				#define NTT16(NTTINFO, x, INFO, N)								asm_ntt2_to_mont_15bit_1024_avx512(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)							asm_ntt2_to_mont_15bit_1024_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)								asm_intt2_to_mont_15bit_1024_avx512(NTTINFO, x, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)						asm_intt2_to_mont_15bit_1024_avx512(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)			asm_mul2_mont_15bit_1024_avx512(out, x, y, MULINFO, INFO)
				//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma2_mont_15bit_1024_avx512(out, x, y, MULINFO, INFO, len)


			#elif N_RLWE_BIN == 512
				#define NTT16(NTTINFO, x, INFO, N)								asm_ntt2_to_mont_15bit_512_avx512(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)							asm_ntt2_to_mont_15bit_512_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)								asm_intt2_to_mont_15bit_512_avx512(NTTINFO, x, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)						asm_intt2_to_mont_15bit_512_avx512(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)			asm_mul2_mont_15bit_512_avx512(out, x, y, MULINFO, INFO)
				//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma2_mont_15bit_512_avx512(out, x, y, MULINFO, INFO, len)


			#endif
			#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma2_mont_15bit_avx512(out, x, y, MULINFO, INFO, len*N, N)
			#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)			C_asm_mul2_mont_15bit_avx512(out, x, y, MULINFO, INFO)
		
		#elif AVXTYPE == 2  
			
			#if N_RLWE_BIN == 4096
				#define NTT16(NTTINFO, x, INFO, N)								asm_ntt2_mont_15bit_4096_avx2(NTTINFO, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)							asm_ntt2_to_mont_15bit_4096_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)								asm_intt2_mont_15bit_4096_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)						asm_intt2_to_mont_15bit_4096_avx2(NTTINFO, x, y, INFO)
				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma2_mont_15bit_4096_avx2(out, x, y, MULINFO, INFO, len)


			#elif N_RLWE_BIN == 2048
				#define NTT16(NTTINFO, x, INFO, N)								asm_ntt2_mont_15bit_2048_avx2(NTTINFO, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)							asm_ntt2_to_mont_15bit_2048_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)								asm_intt2_mont_15bit_2048_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)						asm_intt2_to_mont_15bit_2048_avx2(NTTINFO, x, y, INFO)
				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma2_mont_15bit_2048_avx2(out, x, y, MULINFO, INFO, len)


			#elif N_RLWE_BIN == 1024
				#define NTT16(NTTINFO, x, INFO, N)								asm_ntt2_mont_15bit_1024_avx2(NTTINFO, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)							asm_ntt2_to_mont_15bit_1024_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)								asm_intt2_mont_15bit_1024_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)						asm_intt2_to_mont_15bit_1024_avx2(NTTINFO, x, y, INFO)
				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma2_mont_15bit_1024_avx2(out, x, y, MULINFO, INFO, len)


			#elif  N_RLWE_BIN == 512
				#define NTT16(NTTINFO, x, INFO, N)								asm_ntt2_mont_15bit_512_avx2(NTTINFO, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)							asm_ntt2_to_mont_15bit_512_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)								asm_intt2_mont_15bit_512_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)						asm_intt2_to_mont_15bit_512_avx2(NTTINFO, x, y, INFO)
				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma2_mont_15bit_512_avx2(out, x, y, MULINFO, INFO, len)
			
			#endif
			//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma2_mont_15bit_avx2(out, x, y, MULINFO, INFO, len*N, N)
			#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)			C_asm_mul2_mont_15bit_avx2(out, x, y, MULINFO, INFO)
	

		#elif AVXTYPE == 1

		#else
		
		#endif

	#elif  INCOMPLETE_DEPTH_BIN == 3
		#define PointMulMont_32bit(out, x, y, MULINFO, INFO, N)     Mul3MontSchool_32bit(out, x, y, MULINFO, INFO)
		#if AVXTYPE == 3  
			
			#if N_RLWE_BIN == 4096
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt3_to_mont_15bit_4096_avx512(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt3_to_mont_15bit_4096_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt3_to_mont_15bit_4096_avx512(NTTINFO, x, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt3_to_mont_15bit_4096_avx512(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul3_mont_15bit_4096_avx512(out, x, y, MULINFO, INFO)
				//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma3_mont_15bit_4096_avx512(out, x, y, MULINFO, INFO, len)

			#elif N_RLWE_BIN == 2048
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt3_to_mont_15bit_2048_avx512(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt3_to_mont_15bit_2048_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt3_to_mont_15bit_2048_avx512(NTTINFO, x, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt3_to_mont_15bit_2048_avx512(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul3_mont_15bit_2048_avx512(out, x, y, MULINFO, INFO)
				//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma3_mont_15bit_2048_avx512(out, x, y, MULINFO, INFO, len)

			#elif N_RLWE_BIN == 1024
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt3_to_mont_15bit_1024_avx512(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt3_to_mont_15bit_1024_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt3_to_mont_15bit_1024_avx512(NTTINFO, x, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt3_to_mont_15bit_1024_avx512(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul3_mont_15bit_1024_avx512(out, x, y, MULINFO, INFO)
				//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma3_mont_15bit_1024_avx512(out, x, y, MULINFO, INFO, len)


			#elif N_RLWE_BIN == 512
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt3_to_mont_15bit_512_avx512(NTTINFO, x, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt3_to_mont_15bit_512_avx512(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt3_to_mont_15bit_512_avx512(NTTINFO, x, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt3_to_mont_15bit_512_avx512(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul3_mont_15bit_512_avx512(out, x, y, MULINFO, INFO)
				//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma3_mont_15bit_512_avx512(out, x, y, MULINFO, INFO, len)


			#endif
			//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma3_mont_15bit_avx512(out, x, y, MULINFO, INFO, len*N, N)
			#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		C_asm_mul3_mont_15bit_avx512(out, x, y, MULINFO, INFO)
			
		#elif AVXTYPE == 2  
			#if N_RLWE_BIN == 4096
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt3_mont_15bit_4096_avx2(NTTINFO, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt3_to_mont_15bit_4096_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt3_mont_15bit_4096_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt3_to_mont_15bit_4096_avx2(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul3_mont_15bit_4096_avx2(out, x, y, MULINFO, INFO)
				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma3_mont_15bit_4096_avx2(out, x, y, MULINFO, INFO, len)

			#elif N_RLWE_BIN == 2048
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt3_mont_15bit_2048_avx2(NTTINFO, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt3_to_mont_15bit_2048_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt3_mont_15bit_2048_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt3_to_mont_15bit_2048_avx2(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul3_mont_15bit_2048_avx2(out, x, y, MULINFO, INFO)
				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma3_mont_15bit_2048_avx2(out, x, y, MULINFO, INFO, len)

			#elif N_RLWE_BIN == 1024
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt3_mont_15bit_1024_avx2(NTTINFO, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt3_to_mont_15bit_1024_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt3_mont_15bit_1024_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt3_to_mont_15bit_1024_avx2(NTTINFO, x, y, INFO)
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul3_mont_15bit_1024_avx2(out, x, y, MULINFO, INFO)
				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma3_mont_15bit_1024_avx2(out, x, y, MULINFO, INFO, len)


			#elif  N_RLWE_BIN == 512
				#define NTT16(NTTINFO, x, INFO, N)							asm_ntt3_mont_15bit_512_avx2(NTTINFO, x, INFO)
				#define NTTTo16(NTTINFO, x, y, INFO, N)						asm_ntt3_to_mont_15bit_512_avx2(NTTINFO, x, y, INFO)
				#define INTT16(NTTINFO, x, INFO, N)							asm_intt3_mont_15bit_512_avx2(NTTINFO, x, INFO)
				#define INTTTo16(NTTINFO, x, y, INFO, N)					asm_intt3_to_mont_15bit_512_avx2(NTTINFO, x, y, INFO)
				
				//#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)		asm_mul3_mont_15bit_512_avx2(out, x, y, MULINFO, INFO)
	
				#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma3_mont_15bit_512_avx2(out, x, y, MULINFO, INFO, len)
				
			#endif
			//#define FMAMont_clear_16bit(out, x, y, MULINFO, INFO, len, N)	asm_fma3_mont_15bit_avx2(out, x, y, MULINFO, INFO, len*N, N)

			#define PointMulMont_16bit(out, x, y, MULINFO, INFO, N)			C_asm_mul3_mont_15bit_avx2(out, x, y, MULINFO, INFO)
	


		#elif AVXTYPE == 1
		#else
		#endif
	#else
	  /*
		INCOMPLETE_DEPTH_BIN == 4
		#define PointMulMont1(a, b, c, d, e, f, g, h)  Mul4MontSchool_32bit(a, b, c, d, e, f, g, h)
	   
		#define NTT16(NTTINFO, x, INFO, N)     asm_ntt4_shoup_15bit_1024_avx2(NTTINFO, x, INFO)
		#define INTT16(NTTINFO, x, INFO, N)    asm_intt4_shoup_15bit_1024_avx2(NTTINFO, x, INFO)
	 */
	  printf("STH WRING !!!!!!!");
	  abort();
	#endif
#endif


//////// GADGET VALUE ////////////////////////////
#if AVXTYPE == 5
	
	// 16bit
	/*
	#define DecomA(from, to, INFO, GADGET, dim) asm_decom_mont_qnum2_base8_rm8_len3_neon(from, to, INFO, GADGET, dim)
	#define DecomB(from, to, INFO, GADGET, dim) asm_decom_mont_qnum2_base8_rm8_len3_neon(from, to, INFO, GADGET, dim)
	*/

	/*
	#define DecomA(from, to, INFO, GADGET, dim) asm_decom_mont_qnum2_base9_rm10_len2_neon(from, to, INFO, GADGET, dim)
	#define DecomB(from, to, INFO, GADGET, dim) asm_decom_mont_qnum2_base9_rm10_len2_neon(from, to, INFO, GADGET, dim)
	*/
	
	/*
	#define DecomA(from, to, INFO, GADGET, dim) asm_decom_mont_qnum2_base9_rm10_len2_neon(from, to, INFO, GADGET, dim)
	#define DecomB(from, to, INFO, GADGET, dim) asm_decom_mont_qnum2_base11_rm17_len1_neon(from, to, INFO, GADGET, dim)
	*/

	///// CTYPE
	/*
	#define DecomA(from, to, INFO, GADGET, dim, KBK) C_asm_decom_mont_qnum2_base8_rm8_len3_16bit_neon(from, to, INFO, GADGET, dim, KBK)
	#define DecomB(from, to, INFO, GADGET, dim, KBK) C_asm_decom_mont_qnum2_base8_rm8_len3_16bit_neon(from, to, INFO, GADGET, dim, KBK)
	*/


	/*
	#define DecomA(from, to, INFO, GADGET, dim, KBK) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_neon(from, to, INFO, GADGET, dim, KBK)
	#define DecomB(from, to, INFO, GADGET, dim, KBK) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_neon(from, to, INFO, GADGET, dim, KBK)
	*/
	


	  /*
	#define DecomA(from, to, INFO, GADGET, dim, KBK) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_neon(from, to, INFO, GADGET, dim, KBK)
	#define DecomB(from, to, INFO, GADGET, dim, KBK) C_asm_decom_mont_qnum2_base11_rm17_len1_16bit_neon(from, to, INFO, GADGET, dim, KBK)

*/
/*
	#define DecomA(from, to, INFO, GADGET, dim, KBK) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_neon(from, to, INFO, GADGET, dim, KBK)
	#define DecomB(from, to, INFO, GADGET, dim, KBK) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_neon(from, to, INFO, GADGET, dim, KBK)
*/
	
#define DecomA(from, to, INFO, GADGET, dim, KBK) C_asm_decom_mont_qnum2_base8_rm8_len3_16bit_neon(from, to, INFO, GADGET, dim, KBK)
	#define DecomB(from, to, INFO, GADGET, dim, KBK) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_neon(from, to, INFO, GADGET, dim, KBK)

	/*
	#define DecomA(from, to, INFO, GADGET, dim) DecompTwoPowRemoveCRT2_Base9_Remove10_Len2_16bit(from, to, INFO, GADGET, dim)
	#define DecomB(from, to, INFO, GADGET, dim) DecompTwoPowRemoveCRT2_Base11_Remove17_Len1_16bit(from, to, INFO, GADGET, dim)
	*/

	
	// 32bit
	//#define DecomA(from, to, INFO, GADGET, dim) DecompTwoPowRemoveCRT2_Base9_Remove10_Len2_16bit(from, to, INFO, GADGET, dim)
	//#define DecomB(from, to, INFO, GADGET, dim) DecompTwoPowRemoveCRT2_Base9_Remove10_Len2_16bit(from, to, INFO, GADGET, dim)





#elif AVXTYPE == 3
	
	/*
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base8_rm8_len3_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base8_rm8_len3_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	*/
	
	/*
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base8_rm8_len3_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base8_rm8_len3_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	*/


	/*
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	*/
	
	/*
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	*/


	/*  
	#define DecomA(from, to, INFO, GADGET, dim) asm_decomFloat_mont_qnum2_base9_rm10_len2_16bit_avx512(from, to, INFO, GADGET, dim)
	#define DecomB(from, to, INFO, GADGET, dim) asm_decomFloat_mont_qnum2_base9_rm10_len2_16bit_avx512(from, to, INFO, GADGET, dim)
	*/
	
	
	/*
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base11_rm17_len1_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	*/
	

	/*	
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base8_rm8_len3_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base8_rm8_len3_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	*/

	
	/*
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base11_rm17_len1_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	*/
	
	  /*
#define DecomA(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	*/

	//#define DecomA(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base8_rm8_len3_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx512(from, to, INFO, GADGET, dim, dimlen)
	
#elif AVXTYPE == 2
	/*
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base8_rm8_len3(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base8_rm8_len3(from, to, INFO, GADGET, dim, dimlen)
	*/
	/*
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base8_rm8_len3_16bit_avx2(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base8_rm8_len3_16bit_avx2(from, to, INFO, GADGET, dim, dimlen)
	*/	
	
	// Param2
	/*
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base9_rm10_len2(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base9_rm10_len2(from, to, INFO, GADGET, dim, dimlen)
	*/
	/*
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx2(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx2(from, to, INFO, GADGET, dim, dimlen)
	*/


	/*
	#define DecomA(from, to, INFO, GADGET, dim) asm_decomFloat_mont_qnum2_base9_rm10_len2_avx2(from, to, INFO, GADGET, dim)
	#define DecomB(from, to, INFO, GADGET, dim) asm_decomFloat_mont_qnum2_base9_rm10_len2_avx2(from, to, INFO, GADGET, dim)
	*/
	
	
	// Param2
	/*
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base9_rm10_len2(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) asm_decom_mont_qnum2_base11_rm17_len1(from, to, INFO, GADGET, dim, dimlen)
	*/
	
	/*
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base8_rm8_len3_16bit_avx2(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base8_rm8_len3_16bit_avx2(from, to, INFO, GADGET, dim, dimlen)
	*/
	  
	/*
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx2(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base11_rm17_len1_16bit_avx2(from, to, INFO, GADGET, dim, dimlen)
	*/
	
	/*  
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx2(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx2(from, to, INFO, GADGET, dim, dimlen)
	*/

	#define DecomA(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base8_rm8_len3_16bit_avx2(from, to, INFO, GADGET, dim, dimlen)
	//#define DecomA(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx2(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) C_asm_decom_mont_qnum2_base9_rm10_len2_16bit_avx2(from, to, INFO, GADGET, dim, dimlen)





#elif AVXTYPE == 1
#elif AVXTYPE == 0
	/*
	#define DecomA(from, to, INFO, GADGET, dim) DecompTwoPowRemoveCRT2_Base8_Remove8_Len3_16bit(from, to, INFO, GADGET, dim)
	#define DecomB(from, to, INFO, GADGET, dim) DecompTwoPowRemoveCRT2_Base8_Remove8_Len3_16bit(from, to, INFO, GADGET, dim)
	*/
	
	
	/*	 
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) DecompTwoPowRemoveCRT2_Base9_Remove10_Len2_16bit(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) DecompTwoPowRemoveCRT2_Base9_Remove10_Len2_16bit(from, to, INFO, GADGET, dim, dimlen)
	*/
 
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) DecompTwoPowRemoveCRT2_Base8_Remove8_Len3_16bit(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) DecompTwoPowRemoveCRT2_Base8_Remove8_Len3_16bit(from, to, INFO, GADGET, dim, dimlen)
	//#define DecomB(from, to, INFO, GADGET, dim, dimlen) DecompTwoPowRemoveCRT2_Base9_Remove10_Len2_16bit(from, to, INFO, GADGET, dim, dimlen)
	


	/*
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) DecompTwoPowRemoveCRT2_Base9_Remove11_Len2_16bit(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) DecompTwoPowRemoveCRT2_Base9_Remove11_Len2_16bit(from, to, INFO, GADGET, dim, dimlen)
	*/
	


	/*
	#define DecomA(from, to, INFO, GADGET, dim) DecompFloatTwoPowRemoveCRT2_Base9_Remove10_Len2_16bit(from, to, INFO, GADGET, dim)
	#define DecomB(from, to, INFO, GADGET, dim) DecompFloatTwoPowRemoveCRT2_Base9_Remove10_Len2_16bit(from, to, INFO, GADGET, dim)
	*/

	
	/*
	#define DecomA(from, to, INFO, GADGET, dim, dimlen) DecompTwoPowRemoveCRT2_Base9_Remove10_Len2_16bit(from, to, INFO, GADGET, dim, dimlen)
	#define DecomB(from, to, INFO, GADGET, dim, dimlen) DecompTwoPowRemoveCRT2_Base11_Remove17_Len1_16bit(from, to, INFO, GADGET, dim, dimlen)
	*/


#endif

#define DecomA_32bit(from, to, INFO, GADGET, dim) DecompTwoPowRemoveCRT2_num2q_32bit(from, to, INFO, GADGET, dim)
#define DecomB_32bit(from, to, INFO, GADGET, dim) DecompTwoPowRemoveCRT2_num2q_32bit(from, to, INFO, GADGET, dim)



extern void asm_unperm2_mont_15bit_1024_avx512(int16_t * x);
extern void asm_unperm2_mont_15bit_1024_avx2(int16_t * x);
extern void asm_unperm3_mont_15bit_1024_avx512(int16_t * x);
extern void asm_unperm3_mont_15bit_1024_avx2(int16_t * x);




#ifdef __cplusplus
}
#endif
#endif // HEADER END
