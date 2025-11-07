#ifndef AUTOMORPHISM_EF
#define AUTOMORPHISM_EF

#ifndef __cplusplus
#include<stdint.h>
#endif


#include<CMAKEPARAM.h>
#include<math.h>

#if CPUTYPE == 1 // IF LINUX
	#include<stdint.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif

#include<param.h>

inline void Automorphism_Coeff_16bit(int16_t *x, int16_t *tmps, int16_t *Q_arr, int idx, int N, int q_num) {
	
	int16_t M = N << 1;
	int16_t N_log = round(log2(N));

	int16_t M_mask = M - 1;
	int16_t N_mask = N - 1;
	for (int jj = 0; jj < q_num; jj++) {
		int16_t Q = Q_arr[jj];
		for (int ii = 0; ii < N; ii++) {
			int16_t to = (ii * idx) & M_mask;
			bool Negative (to >= N);	
			tmps[(to & N_mask) + jj*N] = Negative ? (Q - x[ii+ jj*N]) :  (x[ii + jj*N]);
		}
	}

	// Re write
	for (int ii = 0; ii < N*q_num; ii++) {
		x[ii] = tmps[ii];
	}
	return ;

}


inline void Automorphism_Coeff_32bit(int32_t *x, int32_t *tmps, int32_t *Q_arr, int idx, int N, int q_num) {
	
	int32_t M = N << 1;
	int32_t N_log = round(log2(N));

	int32_t M_mask = M - 1;
	int32_t N_mask = N - 1;
	for (int jj = 0; jj < q_num; jj++) {
		int32_t Q = Q_arr[jj];
		for (int ii = 0; ii < N; ii++) {
			int32_t to = (ii * idx) & M_mask;
			bool Negative (to >= N);	
			tmps[(to & N_mask) + jj*N] = Negative ? (Q - x[ii+ jj*N]) :  (x[ii + jj*N]);
		}
	}

	// Re write
	for (int ii = 0; ii < N*q_num; ii++) {
		x[ii] = tmps[ii];
	}
	return ;

}





#ifdef __cplusplus
}
#endif



#endif
