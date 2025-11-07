#ifndef FHE16_PARAM_H
#define FHE16_PARAM_H

#include<poly_class.hpp>
#include<lwe.hpp>
#include<BINFHE.hpp>
#include<CMAKEPARAM.h>

/*
struct NTTTable16;
struct NTTTable32;
struct NTTTable64;
class EFHE_BIN_Param_List;
struct NTTTable16Struct;
*/


//struct alignas(64) FHE16Params{
 struct FHE16Params{
        double _sigma_lwe;
	    double _sigma_ks;
		double _sigma_bs;
		int64_t _Q_TOT;
		int _n_lwe;
		int _n_lwe_cache;
		int _q_num;		
		// LWE Ciphertext
        int _q_lwe;
        int _q_lwe_bit;
		int _scaling_lwe; 
		
        // KS parameter
        int _q_ks;
        int _q_ks_bit;
        int _n_ks;
		int _base_ks;
        int _base_rm_ks;
		int _gadget_len_ks;	
		int _base_bk_a;
        int _base_rm_bk_a;
        int _base_len_bk_a;
		int _base_bk_b;
        int _base_rm_bk_b;
		int _base_len_bk_b;
		int16_t _q_bk16[5];
        int _q_bk_bit16 = 0;
		int	_n_bk;
		int _k_bk;
		bool _KS_FIRST;
		bool _PACKED_KS;
		
		// PubKey
		int _PK_row;
		int _PK_col;
		int _PK_Q;
		char PADDING[44];

};
	
extern FHE16Params GINX16bit_128b;
//extern BIN_EV_MEHOD;

// 
struct alignas(64) FHE16BOOTParam{
//struct FHE16BOOTParam{
	int32_t *PK_raw_32bit;
	int16_t *KS_raw_16bit;
	int16_t *KS_raw_16bit_start;


	int16_t *BK_raw_16bit;
	int16_t *BK_raw_16bit_start;

	int16_t *ACC;
	int16_t *ACC_start;
	int16_t *ROT;
	int16_t *ROT_start;
	int16_t *MEMORY_HANDLE;
	int16_t *MEMORY_HANDLE_start;
	FHE16Params *PARAM;
	FHE16Params *PARAM_start;
	NTTTable16Struct *st;
	NTTTable16Struct *st_start;
	char PADDING[8];
};
 

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif




#endif // End header
