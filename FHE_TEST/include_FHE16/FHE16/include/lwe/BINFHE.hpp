#ifndef EFHE_BIN2_H
#define EFHE_BIN2_H

#include<poly_class.hpp>
#include<lwe.hpp>
#include<FHE16Param.hpp>
#include<CMAKEPARAM.h>
#include <numa.h>
#include <Core.hpp>
#include<BinOperationCstyle.hpp>


extern int get_core_id();
enum BIN_EV_METHOD : uint8_t {
        GINX_32bit,
        GINX_16bit,
		LMKCDEY_32bit,
		LMKCDEY_16bit
    };



namespace EFHEs {
	
    // Predefined in poly_class
    class Poly32;
    class Poly16;
	class LWECiphertext;

    class EvaluationKey32 {
        private:

            const EFHE_BIN_Param_List * _param = nullptr;        
            

            /***************** 32BIT method pointers *****************/
            
            int32_t * _BK_raw = nullptr;
            Poly32 **** _BK = nullptr;
	    
            int32_t * _KS_raw = nullptr;
            //LWECiphertext * _KS = nullptr;
            
			int32_t **** _KS = nullptr;		
				
			/// ** numa ** ///
			void **_BK_raw_NUMA_32bit			= nullptr;
			void **_BK_raw_NUMA_32bit_Aligned	= nullptr;
			void **_KS_raw_NUMA_32bit			= nullptr;
			void **_KS_raw_NUMA_32bit_Aligned	= nullptr;
			void **_R_KS_raw_NUMA_32bit			= nullptr;
			void **_R_KS_raw_NUMA_32bit_Aligned	= nullptr;


			/// ** numa Boot alocation
			void * _Boot_CT_Before				= nullptr;
			void ** _Boot_CT_Before_NUMA		= nullptr;
			void ** _Boot_CT_Before_Aligned		= nullptr;
			int _Boot_CT_Before_16bit_size		= -1;
			void * _BL_Memory					= nullptr;
			void ** _BL_Memory_NUMA				= nullptr;
			void ** _BL_Memory_Aligned			= nullptr;
			int _BL_Memory_16bit_size			= -1;
			void * _KS_Memory					= nullptr;
			void ** _KS_Memory_NUMA				= nullptr;
			void ** _KS_Memory_Aligned			= nullptr;
			int _KS_Memory_16bit_size			= -1;
	




        

			// ACCPolynomials
            Poly32 * _ACC_BIN = nullptr;
						
			Poly32 *_Rot_Poly = nullptr;
			int32_t *_Rot_Poly_raw = nullptr;


            // For test
            int32_t * _sk = nullptr;
            Poly32*  _sk_bk = nullptr;

		    
			/***************** 16BIT Encryption Pointer *****************/
        	
            int32_t * _PK_raw_32bit = nullptr;
			


			/***************** 16BIT method pointers *****************/
        	
            int16_t * _BK_raw_16bit = nullptr;
            Poly16 **** _BK_16bit = nullptr;
			int16_t * _R_KS_raw_16bit = nullptr;
            Poly16 ** _R_KS_16bit = nullptr;
            
			/// ** numa ** ///
			void *** _BK_raw_NUMA_16bit			= nullptr;
			void *** _BK_raw_NUMA_16bit_Aligned	= nullptr;
			void *** _KS_raw_NUMA_16bit			= nullptr;
			void *** _KS_raw_NUMA_16bit_Aligned	= nullptr;
			void **_R_KS_raw_NUMA_16bit			= nullptr;
			void **_R_KS_raw_NUMA_16bit_Aligned	= nullptr;

			int _BK_raw_NUMA_16bit_data_size =0;
			int _KS_raw_NUMA_16bit_data_size =0;
			int _R_KS_raw_NUMA_16bit_data_size =0;



			


            int16_t * _KS_raw_16bit = nullptr;
            //LWECiphertext * _KS = nullptr;
            int16_t ***** _KS_16bit = nullptr;
           

            // ACCPolynomials
            Poly16 * _ACC_BIN_16bit = nullptr;
						
			Poly16 *_Rot_Poly_16bit = nullptr;
			int16_t *_Rot_Poly_raw_16bit = nullptr;


        
            // For test
            int16_t * _sk_16bit = nullptr;
            Poly16	*_sk_bk_16bit = nullptr;

            /***********************   ENDs     **********************/
			int _LWE_sk_make_size	= 2;	// Two is binary & three is ternary
			int _LWE_sk_BK_size		= 2;	// it controls LWE secret key 
			// Bootstrapping MEthod
            BIN_EV_METHOD METHOD = GINX_16bit; 


			/****************** LMKCDEY param ***************/
			int _LMKCDEY_window	= 10;
			
			
			
			Poly32 *** _Auto_Poly32 = nullptr;
			int32_t *_Auto_Poly32_raw = nullptr;
			Poly16 ***_Auto_Poly16 = nullptr;
			int16_t *_Auto_Poly16_raw = nullptr;
			
			Poly16 *** _BK_LMKCDEY_16bit	=	 nullptr;
			int16_t *_BK_LMKCDEY_16bit_raw	=	nullptr;
			Poly32 *** _BK_LMKCDEY_32bit	=	 nullptr;
			int32_t *_BK_LMKCDEY_32bit_raw	=	nullptr;
			int	* _five_pow_array		=	 nullptr;
			int	* _five_pow_array_inv	=	 nullptr;
	
			FHE16BOOTParam * _BOOTParam = nullptr;	// should be deleted
			FHE16BOOTParam** _BOOTThreadParam = nullptr;	// should be deleted


       public:
            explicit EvaluationKey32(const EFHE_BIN_Param_List *param, int32_t* &sk, BIN_EV_METHOD METHOD);
            
            
			FHE16BOOTParam* GetBOOTParam() const {
				return _BOOTParam;
			};
			FHE16BOOTParam** GetBOOTThreadParam() const {
				return _BOOTThreadParam;
			};
			void MakeBOOTParam(
				int16_t *KS_raw_16bit, 
				int16_t *BK_raw_16bit, 
				int32_t *PK_raw_32bit,
				int16_t *ACC,
				int16_t *ROT,
				FHE16Params *PARAM, 
				NTTTable16Struct *st,
				FHE16BOOTParam* &BOOTPARAM,
				FHE16BOOTParam** &BOOTThreadPARAM,
				int LWE_sk_size
				);
			
            const EFHE_BIN_Param_List * GetParam() const { return _param;};
			


            void GenGINXEv( int32_t *sk, Poly32 *tmpsk2);
        	void GenKSEv( Poly32 *tmp_sk, int32_t *sk, int64_t *sk_sign);
            void GenRotPoly();
            void GenRotPoly_16bit();
          
			void GenACC();
            //void MakeBKsk_32bit(Poly32* &tmp_sk, Poly32* &tmp_sk2);
            void MakeBKsk_32bit(Poly32* &tmp_sk, Poly32* &tmp_sk2, int64_t *sk_bk_sign);
            int32_t *GetACC(int idx) const { return _ACC_BIN[idx].GetPoly();};

            void GenGINXEv_16bit( int32_t *sk, Poly16 *tmpsk2);
            void GenLMKCDEYEv_16bit(int32_t * sk, Poly16 * tmp_sk2);
			void GenLMKCDEYEv_32bit(int32_t * sk, Poly32 * tmp_sk2);
			void GenAutoPoly_16bit(Poly16 *tmp_sk, Poly16 *tmp_sk_mont);
			void GenAutoPoly_32bit(Poly32 *tmp_sk, Poly32 *tmp_sk_mont);

			void GenFivePowArr();
			void GenKSEv_16bit( Poly16 *tmp_sk, int32_t *sk, int64_t *sk_sign);
			void GenPackedKSEv_16bit( Poly16 *tmp_sk, int32_t *sk, int64_t *sk_sign);
			void GenACC_16bit();
			void GenACC_LMKCDEY_16bit();
        	void GenACC_LMKCDEY_32bit();
            int16_t *GetACC_16bit(int idx) const { return _ACC_BIN_16bit[idx].GetPoly();};
			void MakeBKsk_16bit(Poly16* &tmp_sk, Poly16* &tmp_sk2, int64_t *BK_sk_sign);
            //void MakeBKsk_16bit(Poly16* &tmp_sk, Poly16* &tmp_sk2);
            
			int16_t *GetKSRaw16() const {
				/*
				if (numa_available() == -1) {
					return _KS_raw_16bit;
				} else {
					return (int16_t *)(_KS_raw_NUMA_16bit_Aligned[numa_node_of_cpu(sched_getcpu())][0]);
				}   */
				return _BOOTThreadParam[get_core_id()]->KS_raw_16bit_start;
			};
	

			int16_t *GetBKRaw16() const {
				/*
				if (numa_available() == -1) {
					return _BK_raw_16bit;
				} else {
					return (int16_t *)(_BK_raw_NUMA_16bit_Aligned[
						numa_node_of_cpu(sched_getcpu())][0]);
				}*/
				return _BOOTThreadParam[get_core_id()]->BK_raw_16bit_start;
			};
			int32_t *GetPKRaw32() const {
				return	_PK_raw_32bit;
			};

			int16_t *GetRot16() const {return  _Rot_Poly_raw_16bit;};
			// It will disappear
			Poly16 *GetSK_BK() const { return _sk_bk_16bit;};
		

			~EvaluationKey32();
            
			// LMCDKEY
			void BootstrappingRaw_LMKCDEY( int16_t *a_and_b,	int32_t *ACC);
			void BootstrappingRawCRT_16bit_LMKCDEY(int16_t *a_and_b,	int16_t *ACC);
			//GINX,using
            void BootstrappingRaw(				int16_t *a_and_b,	int32_t *ACC);
			void BootstrappingRawCRT_16bit(		int16_t *a_and_b,	int16_t *ACC);
 			void BootstrappingRawBin(			int16_t *a_and_b,	int32_t *ACC);
 			void BootstrappingRawCRTBin_16bit( int16_t *a_and_b,	int16_t *ACC);
                


            void Bootstrapping(             int32_t *a_and_b,	int32_t *ACC, int32_t *result);
        void BootstrappingRawCRT(       int32_t *a_and_b,	int32_t *ACC, int32_t *result);
        void BootstrappingRawK2CRT_16bit( int32_t *a_and_b, int16_t *ACC, int16_t *result);
            


            void BootstrappingClass(        int32_t *a_and_b,	int32_t *result);

			///// PARALLEL
			void BootstrappingParallel(				int32_t *a_and_b,	int32_t *ACC, int32_t *result, int Num);
            void BootstrappingRawParallel(			int32_t *a_and_b,	int32_t *ACC, int32_t *result, int Num);
            void BootstrappingRawCRTParallel(		int32_t *a_and_b,	int32_t *ACC, int32_t *result, int Num);
            void BootstrappingRawCRT_SIMD_16bit( int32_t *a_and_b,	int16_t *ACC, int16_t *result, int Num);
            
			/////// THREAD 
            void BootstrappingRawCRT_16bit_accelerate(	int32_t *a_and_b,	int16_t *ACC, int16_t *result );
 			void BootstrappingRaw_accelerate(			int32_t *a_and_b,	int32_t *ACC, int32_t *result );
			void BootstrappingRaw_accelerate2(			int32_t *a_and_b,	int32_t *ACC, int32_t *result );
    


			///////// Tensor Decomposition
            void BootstrappingRawCRT_SIMD_16bit_2_4_2( int32_t *a_and_b,	int16_t *ACC, int16_t *result, int Num);
			void BootstrappingRawCRT_SIMD_16bit_3_4_2( int32_t *a_and_b,	int16_t *ACC, int16_t *result, int Num);
 			void BootstrappingRawCRT_SIMD_16bit_4_4_2( int32_t *a_and_b,	int16_t *ACC, int16_t *result, int Num);
  			void BootstrappingRawCRT_SIMD_16bit_5_4_2( int32_t *a_and_b,	int16_t *ACC, int16_t *result, int Num);
			void BootstrappingRawCRT_SIMD_16bit_3_4_3( int32_t *a_and_b,	int16_t *ACC, int16_t *result, int Num);
			void BootstrappingRawCRT_SIMD_16bit_4_4_3( int32_t *a_and_b,	int16_t *ACC, int16_t *result, int Num);
			











			void KeySwitchRawCRT_16bit_KS16(int16_t* val, int LOC);			  
			void KeySwitchRawCRT_16bit_KS16_Input32(int32_t* val, int LOC);			  
			void KeySwitchRawCRT_16bit_PackedKS16_Input32(int32_t* val, int LOC);			  
			

			void KeySwitch_KS16(Poly32* val, const int32_t b, LWECiphertext &res, int LOC);
			void KeySwitchRaw_KS16(int32_t* val, int LOC);
            void KeySwitchRawCRT_KS16(int32_t* val, int32_t *tmpVec, LWECiphertext &res, int LOC);
			void KeySwitchRawCRT_16bit_KS16_2(int16_t* val, void * KS_MEMORY, int LOC);			  
			void KeySwitchRawCRT_SIMD_16bit_KS16(int16_t* val, LWECiphertext *res, int LOC, int WINDOW);



			///////
			int GetLWESupportSize() const { return _LWE_sk_make_size;};
            
			int GetBootBeforeSize16bit() const {return _Boot_CT_Before_16bit_size;};
            int GetBLMemorySize16bit() const {return _BL_Memory_16bit_size;};
            void *GetBLMemory() const {return _BL_Memory_Aligned[0];};
            int GetKSMemorySize16bit() const {return _KS_Memory_16bit_size;};
			
	};



    // PARAM LIST

    class EFHE_BIN_Param_List{
        private:
    
		 bool _KS_FIRST = true;
		 //bool _KS_FIRST = false;
		
		 bool _PACKED_KS	= false;
		 //bool _PACKED_KS	= true;


        // LWE Parameter1
        //uint32_t _n_lwe = N_LWE_BIN;
        //int _n_lwe          = 503;
        //int _n_lwe			= 537;
		// int _n_lwe			= 512;
		//int _n_lwe				= 503;
		//int _n_lwe				= 811;
		int _n_lwe			= 585;
		//int _n_lwe			= 746;
        double _sigma_lwe	=	3.19;
		int _n_lwe_avx1_16bit		= ceil(((double)(_n_lwe+1)) / 8)  * 8;
		int _n_lwe_avx2_16bit		= ceil(((double)(_n_lwe+1)) / 16) * 16;
		int _n_lwe_avx512_16bit		= ceil(((double)(_n_lwe+1)) / 32) * 32;


		//int _n_lwe          = 10;
        int _q_num          = NumQ_BIN;
		
		// LWE Ciphertext
        int _q_lwe          = ((uint32_t)1) << 14;
        int _q_lwe_bit      = std::ceil(std::log2(_q_lwe));
		int _scaling_lwe	= _q_lwe >> 2; 
		
        // KS parameter
        int _q_ks           = _q_lwe;
        int _q_ks_bit       = std::ceil(std::log2(_q_ks));
        int _n_ks           = _n_lwe;
        double _sigma_ks	=	3.14;
		//double _sigma_ks	=	0;


		//int _base_ks        = 4;
        //int _base_rm_ks		= 6;
		//int _base_ks        = 2;
        //int _base_rm_ks		= 0;
		int _base_ks        = 5;
        int _base_rm_ks		= 0;
		
		int _gadget_len_ks  = (int)(ceil((double)(_q_ks_bit - _base_rm_ks) / (double)_base_ks));


        // BK Gadget base
        
		/*
		int _base_bk_a      = 8;
        int _base_rm_bk_a   = 8;
        int _base_bk_b      = 8;
        int _base_rm_bk_b   = 8;
		*/
		
		
		
		int _base_bk_a      = 9;
        int _base_rm_bk_a   = 10;
        int _base_bk_b      = 9;
        int _base_rm_bk_b   = 10;
		

		
		/*
		int _base_bk_a      = 8;
        int _base_rm_bk_a   = 8;
        int _base_bk_b      = 9;
        int _base_rm_bk_b   = 10;
		*/
	
		
		/*
		int _base_bk_a      = 9;
        int _base_rm_bk_a   = 10;
        int _base_bk_b      = 11;
        int _base_rm_bk_b   = 17;
		*/



		/*
		int _base_bk_a      = 9;
        int _base_rm_bk_a   = 11;
        int _base_bk_b      = 9;
        int _base_rm_bk_b   = 11;
		*/
	
			/************* 32bit *******************/

        int32_t  *_q_bk         = nullptr;
        // q_bk_bit is required ?? 
        int64_t  _Q_TOT         = 0;
        int _q_bk_bit       = std::ceil(std::log2(_Q_TOT));
        
        int _gadget_len_bk_a;
        int _gadget_len_bk_b;
 
        /************* 16bit *******************/
        /*
        int16_t _q_bk16[2]      = {12289, 18433};
        int _q_bk_bit16         = std::ceil(std::log2(226523137));
        int _gadget_len_bk_a16  = (uint32_t)(ceil( (double)(_q_bk_bit16 - _base_rm_bk_a) / (double)_base_bk_a));
        int _gadget_len_bk_b16  = (uint32_t)(ceil( (double)(_q_bk_bit16 - _base_rm_bk_b) / (double)_base_bk_b));
        */
        int16_t *_q_bk16 = nullptr;
        int _q_bk_bit16 = 0;
        //double _sigma_bk	=	3.19;
		double _sigma_bk	=	3.59;
		//double _sigma_bk	=	6.85;
		//double _sigma_bk	=	0;
	// GADGET DECOM Funs
		//int16_t *from, int16_t * to, const int32_t *INFO, const int16_t *GADGET, int dim)
		void (*_GadgetABoot)(int16_t*, int16_t*, const int32_t*, const int16_t*, int)	= nullptr;
		void (*_GadgetBBoot)(int16_t*, int16_t*, const int32_t*, const int16_t*, int)	= nullptr;
		void (*_GadgetARot)(int16_t*, int16_t*, const int32_t*, const int16_t*, int)	= nullptr;	
		void (*_GadgetAPack)(int16_t*, int16_t*, const int32_t*, const int16_t*, int)	= nullptr;
		void (*_GadgetAMul)(int16_t*, int16_t*, const int32_t*, const int16_t*, int)	= nullptr;







		//int _gadget_len_bk_a16 = 0;
        //int _gadget_len_bk_b16 = 0;
		    
           
        /************* 64bit *******************/
        int      _n_bk          = N_RLWE_BIN;
        int      _k_bk          = K_RLWE_BIN;

        EvaluationKey32 * _ev       = nullptr; // Dangling pointer ! be careful       
        
        // NTTTable
        NTTTable32 *_BK_ntt_param   = nullptr;
        NTTTable16 *_BK_ntt_param16 = nullptr;
        NTTTable16Struct *_st		= nullptr;
		


        BIN_EV_METHOD _METHOD;
		FHE16Params * _FHE16PARAM		= nullptr;
    public:
        explicit EFHE_BIN_Param_List(int32_t* &sk); // Default...
        explicit EFHE_BIN_Param_List(int32_t* &sk, BIN_EV_METHOD METHOD);
        explicit EFHE_BIN_Param_List(int32_t* &sk, BIN_EV_METHOD METHOD, FHE16Params *PARAM);
        explicit EFHE_BIN_Param_List( uint32_t N, uint32_t K, uint32_t Q);
        ~EFHE_BIN_Param_List();
        void Constructing(int32_t* &sk, BIN_EV_METHOD METHOD, FHE16Params *PARAM);
		void SetGadgetFuns();    

		
		EvaluationKey32* GetEV() const { return _ev;};
        NTTTable16Struct* GetST() const { return _st;};
		FHE16Params*	GetFHE16PARAM()	const { return _FHE16PARAM;};
		// Plane LWE parameter
		bool		GetKSFirst()	const { return _KS_FIRST;};
		bool		GetKSPacked()	const { return _PACKED_KS;};
		int32_t		GetQLWE()       const;
        int32_t		GetQbitLWE()    const;
        int32_t		GetNLWE()       const;
        int32_t		GetScalingLWE()	const;
		double		GetSigmaLWE()	const;
		// BK parameter
        int32_t    *GetQBK() const;
        int16_t    *GetQBK16bit() const;
        int64_t     GetQBKTOT() const;
        uint32_t    GetNBK() const;
        int32_t     GetBaseBKA() const { return _base_bk_a;};
        int32_t     GetBaseRMBKA() const { return _base_rm_bk_a;};
        int32_t     GetGadgetLenBKA() const {return _gadget_len_bk_a;};
        int32_t     GetBaseBKB() const { return _base_bk_b;};
        int32_t     GetBaseRMBKB() const { return _base_rm_bk_b;};
        int32_t     GetGadgetLenBKB() const {return _gadget_len_bk_b;};
		double		GetSigmaBK()	const;
	
		// KS parameter
		uint32_t    GetKBK() const;
        uint32_t    GetNKS() const { return _n_ks;};
        uint32_t    GetQKS() const { return _q_ks;};
        uint32_t    GetBaseKS() const { return _base_ks;};
		uint32_t	GetBaseRMKS() const { return _base_rm_ks;};
        uint32_t    GetGadgetLenKS() const { return _gadget_len_ks;};
        int32_t     GetQbitBK() const { return _q_bk_bit;};
    	double		GetSigmaKS()	const;
	        
		int			GetNLWE_AVX1_16bit() const {return _n_lwe_avx1_16bit;};
		int			GetNLWE_AVX2_16bit() const {return _n_lwe_avx2_16bit;};
		int			GetNLWE_AVX512_16bit() const {return _n_lwe_avx512_16bit;};



        int32_t *   GenSK();
        void GenEv(int32_t * sk);  
        

        // Prepare Bootstrapping
        //void BeforeBoot(const LWECiphertext &c1, const LWECiphertext &c2, LWECiphertext & res, int32_t offset, bool neg) const;
        void BeforeBoot(int16_t *tmp_val) const;
        void BeforeBoot_KSFirst(int32_t *tmp_val) const;
        void BeforeBoot_KSFirst_2OUT(int32_t *tmp_val, int32_t *tmpval2) const;
        void BeforeBoot_PackedKSFirst(int32_t *tmp_val) const;
        
		void BeforeBootSIMD(const LWECiphertext *c1, const LWECiphertext *c2, LWECiphertext* & res, int32_t offset, bool neg, int window) const;
		
		void BeforeBootAccelerate(const LWECiphertext &c1, const LWECiphertext &c2, LWECiphertext &res, int32_t offset, bool neg) const;
		void Bootstrapping2_1(void *inputs, void *BL_MEMORY, void * KS_MEMORY) const; 

		

		void BeforeBoot_TD(const LWECiphertext *c1, const LWECiphertext *c2, LWECiphertext* & res, const int32_t offset, const bool neg, int window, int Type) const;

        // Operation
        void TWOOUT_AND_OR(const LWECiphertext &c1, const LWECiphertext &c2, LWECiphertext & res, LWECiphertext &res2) const;
		 

		void AND(const LWECiphertext &c1, const LWECiphertext &c2, LWECiphertext & res) const;
		void AND_parallel(const LWECiphertext* c1, const LWECiphertext* c2, LWECiphertext* res, int window) const;
		void AND_TD(const LWECiphertext* c1, const LWECiphertext* c2, LWECiphertext* res, int WINDOW, int TYPE) const; 
		
		void AND_accelerate(const LWECiphertext &c1, const LWECiphertext &c2, LWECiphertext &res) const;
	


		void NAND(const LWECiphertext &c1, const LWECiphertext &c2, LWECiphertext & res) const;
        void OR(const LWECiphertext &c1, const LWECiphertext &c2, LWECiphertext & res) const;
        void XOR(const LWECiphertext &c1, const LWECiphertext &c2, LWECiphertext & res) const;
		void NOT(const LWECiphertext &c1, LWECiphertext & res) const;
        

		// Primitive gate operation
		void XOR3(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, LWECiphertext & res) const;
   		void XOR4(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, const LWECiphertext &c4,  LWECiphertext & res) const;
		void XOR5(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, const LWECiphertext &c4, const LWECiphertext &c5, LWECiphertext & res) const;
		void XOR6(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, const LWECiphertext &c4, const LWECiphertext &c5, const LWECiphertext &c6, LWECiphertext & res) const;
		void XOR7(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, const LWECiphertext &c4, const LWECiphertext &c5, const LWECiphertext &c6, const LWECiphertext &c7, LWECiphertext & res) const;



		void MAJ3(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, LWECiphertext & res) const;
 		void EQ3(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, LWECiphertext & res) const;
 		void AND_XOR6(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, LWECiphertext & res) const;
	 	void OR_XOR6(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, LWECiphertext & res) const;
		

	

        
        const NTTTable32 * GetBKNTTParam() const {
            return _BK_ntt_param;
        }
        const NTTTable16 * GetBKNTTParam16() const {
            return _BK_ntt_param16;
        }

		auto GetFuncGadgetABoot() const { return _GadgetABoot;};
		auto GetFuncGadgetBBoot() const { return _GadgetBBoot;};
		auto GetFuncGadgetARot() const { return _GadgetARot;};
		auto GetFuncGadgetAPack() const { return _GadgetAPack;};
		auto GetFuncGadgetAMul() const { return _GadgetAMul;};


    };

};
#endif // End header
