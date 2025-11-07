#ifndef NTTTABLE_HPP_EF
#define NTTTABLE_HPP_EF


#include<CMAKEPARAM.h>
#include <param.h>
#include <structures.h>
#include <iostream>
#include <BINFHE.hpp>
#include <PRE_TABLE_IDX.h>
#include <stdint.h>

using namespace std;

	typedef enum class Q_DEPTH_INFO{
		DEPTH0_1BIT_LESS,
		DEPTH1_1BIT_LESS,
		DEPTH2_1BIT_LESS,
		DEPTH3_1BIT_LESS,
		DEPTH4_1BIT_LESS,
		
		DEPTH0_2BIT_LESS,
		DEPTH1_2BIT_LESS,
		DEPTH2_2BIT_LESS,
		DEPTH3_2BIT_LESS,
		DEPTH4_2BIT_LESS,

		DEPTH0_3BIT_LESS,
		DEPTH1_3BIT_LESS,
		DEPTH2_3BIT_LESS,
		DEPTH3_3BIT_LESS,
		DEPTH4_3BIT_LESS,

		DEPTH0_4BIT_LESS,
		DEPTH1_4BIT_LESS,
		DEPTH2_4BIT_LESS,
		DEPTH3_4BIT_LESS,
		DEPTH4_4BIT_LESS
	};



struct alignas(64) NTTTable16Struct{
//struct NTTTable16Struct{
	// FUNCTIONS
	void (**_MUL_MONT_IN_NTT)(int16_t*, int16_t*, int16_t*, const int16_t*, const int32_t*)		= nullptr;  // 8 			
	void (**_NTT_TO_MONT)(const int16_t*, int16_t*, int16_t*,	const int32_t*)					= nullptr;	// 16	
	void (**_INTT_TO_MONT)(const int16_t*, int16_t*, int16_t*,	const int32_t*)					= nullptr;	// 24	
	void (**_ADD)(const int16_t*, const int16_t*, int16_t*,	int16_t, int16_t, int)				= nullptr;	// 32	
	void (**_SUB)(const int16_t*, const int16_t*, int16_t*,	int16_t, int16_t, int)				= nullptr;	// 40		           
    
	void (* DECOMA)  (int16_t*, int16_t *, const int32_t*, const int32_t*, int, int, int, int)		= nullptr;
	void (* DECOMB)  (int16_t*, int16_t *, const int32_t*, const int32_t*, int, int, int, int)		= nullptr;
	void (** _VECMATMUL) (int16_t *, int16_t *, int16_t *, const int16_t *, const int32_t *, int, int) = nullptr;
	//C_asm_mul0_mont_VecMat_DEPTH0_14bit_avx512(ACC_tmp_arr, Gadget_raw, BK_raw_16bit + BK_cnt, QNTTINFO_local, QINFO_local, NBK*(KBK+1), NBK * (KBK * glBKA + glBKB));




	//(int16_t * from, int16_t *to, const int32_t * INFO, const int32_t *GADGET, int dim, int rm_len, int base_len)
	int64_t *    _crt_basis			= nullptr;	// 48
    int64_t *	_QINFO_start_arr	= nullptr;	// 56
	int32_t *   _QINFO				= nullptr;  // 64
    int16_t *   _QNTTINFO			= nullptr;  // 72
	Q_DEPTH_INFO *	_Q_DEPTH_INFO	= nullptr;  // 80
	int64_t     _Q_TOT = 0; // 88
	int         _N;  // 92
    int			_N_log; // 96
    int			_Q_num; // 100
	int          _Q_TOT_bit = 0; // 104
		
	//////// Will be Deprecated
	int			_QNTT_START_idx  	= 0; //108
    int			_QINTT_START_idx  	= 0; // 112
	int	        _QMUL_START_idx		= 0; // 116
	int			_QGADGET_START_idx 	= 0;  // 120
		          
	////////////////////
	int			_Gadget_END_idx		=0; // 124
	char _padding[128 + 64  - 140 - 8];
};



/************************** C style construction funcs ***********************************/

void NTTTable16Const(NTTTable16Struct* &st, int16_t *Q, int * Depth, int N, int q_num);
void NTTTable16Decon(NTTTable16Struct* &st);

/************************** C style construction funcs ENDS ***********************************/


namespace EFHEs{

    // Predefined in poly_class
    class EFHE_BIN_Param_List;
    class Poly32;
    class Poly16;
    class Poly64;
    class NTTTable32 {
        private:
 			// FUNCTIONS
			void (**_MUL_MONT_IN_NTT)(int32_t*, int32_t*, int32_t*, int)					= nullptr;			
			void (**_NTT_TO_MONT)(int64_t*, int64_t*, int64_t*, int)						= nullptr;			
			void (**_INTT_TO_MONT)(int64_t*, int64_t*, int64_t*, int)						= nullptr;			
			void (**_ADD)(const int32_t*, const int32_t*, int32_t*,	int32_t, int32_t, int)	= nullptr;			
			void (**_SUB)(const int32_t*, const int32_t*, int32_t*,	int32_t, int32_t, int)	= nullptr;			




			//void (**_MUL_MONT_IN_NTT)()		= nullptr;			


          
            int64_t *    _crt_basis			= nullptr;
            int64_t *	_QINFO_start_arr	= nullptr;
			int32_t	*   _QINFO				= nullptr;
            int32_t	*   _QNTTINFO			= nullptr;
			Q_DEPTH_INFO *	_Q_DEPTH_INFO	= nullptr;
 

			int64_t     _Q_TOT = 0;
          
			int         _N;
            int			_N_log;
            
            // Q setting
            int			_Q_num;
            //int16_t     *_Q = nullptr;
			int          _Q_TOT_bit = 0;

			// WILL be Deprecated
			//int			_QNTT_START_idx  	= 0;
            //int			_QINTT_START_idx  	= 0;
			//int	        _QMUL_START_idx		= 0;
			//int			_QGADGET_START_idx 	= 0;
			///////////////////////////
			int			_Gadget_END_idx		= 0;
			bool		_MONTGO_TWEAK		= true;
				
        public:
            explicit NTTTable32(); // Default ...
            explicit NTTTable32(int32_t *Q, int * Depth, int N, int q_num);
            ~NTTTable32();

            /* We will merge into this ARRAY */
            
            void MAKE_QDATA(int32_t *Q,			int *mul_depth, int N, int q_num);
			void MAKE_QDATAMontgo(int32_t *Q,	int *mul_depth, int N, int q_num, bool MONTGO_NTT = true);
			void MakePolyFuncs(int qnum);
	
            int     reverse(int v, int N_log);
            int     spmd(int x, int n, int m);
            
            int     invers(int x,int y);
            int		sqrmp(int x,int m);
            int64_t xGCD(int64_t a,int64_t b, int64_t &x, int64_t &y);
            int64_t INV64(int64_t a,int64_t Q);
            
            int64_t GetQtot() const {return _Q_TOT;};
            int     GetQnum() const { return _Q_num;};
			int     GetN() const { return _N;};
			
            
            /////////////// Related Q /////////////////////
            //int64_t	*GetQBarretLong() const { return _Q_barret_long;};
            //int32_t *GetQBarret() const { return _Q_barret;};
	        
            int32_t GetQ(int idx) const             { return _QINFO[ QDATA_Q                    + idx*QDATA_LEN]; };
            int32_t GetQMontgo(int idx) const       { return _QINFO[ QDATA_Q_MONT_SIGN          + idx*QDATA_LEN]; }; 
            int32_t GetOneMontgo(int idx) const     { return _QINFO[ QDATA_Q_MONT_SIGN_ONE      + idx*QDATA_LEN]; };
			int32_t GetSquareMontgo(int idx) const  { return _QINFO[ QDATA_Q_MONT_SIGN_SQUARE   + idx*QDATA_LEN]; };
			int32_t GetOneMontgoInv(int idx) const  { return _QINFO[ QDATA_Q_MONT_SIGN_INV      + idx*QDATA_LEN]; };
      		int32_t GetNTTDepth(int idx) const      { return _QINFO[ QDATA_QDEPTH               + idx*QDATA_LEN]; };
			;			


			auto GetFuncMulMontInNTT() const { return _MUL_MONT_IN_NTT;};        
			auto GetFuncMulMontInNTTAtIdx(int idx) const { return _MUL_MONT_IN_NTT[idx];};        
			
			   
            //int32_t GetQ(int idx) const             { return    (_QINFO + _Gadget_END_idx)[ QDATA_Q                    + _QINFO_start_arr(idx)]; };
            //int32_t GetQMontgo(int idx) const       { return    (_QINFO + _Gadget_END_idx)[ QDATA_Q_MONT_SIGN          + _QINFO_start_arr(idx)]; };  
            //int32_t GetOneMontgo(int idx) const     { return    (_QINFO + _Gadget_END_idx)[ QDATA_Q_MONT_SIGN_ONE      + _QINFO_start_arr(idx)]; };
            //int32_t GetSquareMontgo(int idx) const  { return    (_QINFO + _Gadget_END_idx)[ QDATA_Q_MONT_SIGN_SQUARE   + _QINFO_start_arr(idx)]; };
            //int32_t GetOneMontgoInv(int idx) const  { return    (_QINFO + _Gadget_END_idx)[ QDATA_Q_MONT_SIGN_INV      + _QINFO_start_arr(idx)]; };
            //int32_t GetNTTDepth(int idx) const      { return    (_QINFO + _Gadget_END_idx)[ QDATA_QDEPTH               + _QINFO_start_arr(idx)]; };


			const int32_t *GetNTTINFO(int idx)		const {return _QNTTINFO + _Gadget_END_idx + _QINFO_start_arr[idx] + 0 ;}	;
			const int32_t *GetMULNTTINFO(int idx)	const {return _QNTTINFO + _Gadget_END_idx + _QINFO_start_arr[idx] + 0;}	;
    	    const int32_t *GetINTTINFO(int idx)		const {return _QNTTINFO + _Gadget_END_idx + + _QINFO_start_arr[idx] + 0;}	;
    	    //const int32_t *GetGADGETINFO(int idx)	const {return _QNTTINFO + _Gadget_END_idx + + _QINFO_start_arr(idx) + 0; }	;
		

            int64_t *GetCRTbasis() const { return _crt_basis;}
            int32_t FastAddWithRangeZeroToQ(const int32_t a, const int32_t b, int32_t Q) const;
            int32_t FastSubWithRangeZeroToQ(const int32_t a, const int32_t b, int32_t Q) const;
			int32_t FastAddWithRangeQOverTwoToQOverTwo(const int32_t a, const int32_t b, int32_t Q) const;
	        int32_t BarretMulWithRangeZeroToQ(const int32_t a, const int32_t b, int32_t Q, int32_t shift, int32_t longs) const;
            
			/*
            int16_t Int2Montgo16bit(int16_t a) const;
            
            int16_t Mul16bit_Montgo(int16_t a, int16_t b, int16_t Q, int16_t montgo, int16_t one_inv) const;
            int16_t Mul16bit_Barret(const int32_t c, int16_t Q, int16_t shift, int16_t longs) const;
            int16_t bit16_Montgo(int32_t P, int16_t Q, int16_t montgo, int16_t one_inv) const;
            */
			// NTT functionality
			void init_ntt_16bit();
            void ReductionTest();
            // Multiplication method
            

            // Get QNTTTable
            const int32_t * GetNTTDATA()    const { return _QNTTINFO;}
            int32_t * GetDATA()       const { return _QINFO;}
            int64_t * GetSTARTIDX()       const { return _QINFO_start_arr;}
             
			int32_t * GetNTTDATAIdx(int idx) const { 
				return _QNTTINFO + _Gadget_END_idx + _QINFO_start_arr[idx]; 
            };
            int32_t * GetDATA(int idx)       const { return _QINFO +  idx * QDATA_LEN;};

			//Gadget Functionality
			void GadgetDecompTwoPowRemoveOne(const int32_t * from, int32_t * to, int dim, int poly_len, int base, int gl_len) const;
	};



    class NTTTable16 {
        private:
			// FUNCTIONS
			void (**_MUL_MONT_IN_NTT)(int16_t*, int16_t*, int16_t*, const int16_t*, const int32_t*)		= nullptr;			
			void (**_NTT_TO_MONT)(const int16_t*, int16_t*, int16_t*,	const int32_t*)					= nullptr;			
			void (**_INTT_TO_MONT)(const int16_t*, int16_t*, int16_t*,	const int32_t*)					= nullptr;			
			void (**_ADD)(const int16_t*, const int16_t*, int16_t*,	int16_t, int16_t, int)				= nullptr;			
			void (**_SUB)(const int16_t*, const int16_t*, int16_t*,	int16_t, int16_t, int)				= nullptr;			
			//void (**_MUL_MONT_IN_NTT)()		= nullptr;			
           
            int64_t *    _crt_basis			= nullptr;
            int64_t *	_QINFO_start_arr	= nullptr;
		
			int32_t *   _QINFO				= nullptr;
            int16_t *   _QNTTINFO			= nullptr;
			Q_DEPTH_INFO *	_Q_DEPTH_INFO	= nullptr;
			
 


			int64_t     _Q_TOT = 0;
        
			int         _N;
            int			_N_log;
            
            // Q setting
            int			_Q_num;
            //int16_t     *_Q = nullptr;
           int          _Q_TOT_bit = 0;
		
			//////// Will be Deprecated
			int			_QNTT_START_idx  	= 0;
            int			_QINTT_START_idx  	= 0;
			int	        _QMUL_START_idx		= 0;
			int			_QGADGET_START_idx 	= 0;
		          
			////////////////////
			int			_Gadget_END_idx		=0;

        public:
            explicit NTTTable16(); // Default ...
            explicit NTTTable16(int16_t *Q, int *Depth, int N, int q_num);
            explicit NTTTable16( NTTTable16Struct *st );
            ~NTTTable16();

            /* We will merge into this ARRAY */
            
            void MAKE_QDATA(int16_t *Q, int *mul_depth, int N, int q_num);
			void MAKE_QDATAMontgo(int16_t *Q, int *mul_depth, int N, int q_num, bool MONTGO_NTT = true);
			void MakePolyFuncs(int qnum);
		




            int     reverse(int v, int N_log);
            int     spmd(int x, int n, int m);
            
            int     invers(int x,int y);
            int sqrmp(int x,int m);
            int64_t xGCD(int64_t a,int64_t b, int64_t &x, int64_t &y);
            int64_t INV64(int64_t a,int64_t Q);
            
            int64_t GetQtot() const {return _Q_TOT;};
            int     GetQnum() const { return _Q_num;};
			int     GetN() const { return _N;};
			
            
            /////////////// Related Q /////////////////////
            //int64_t	*GetQBarretLong() const { return _Q_barret_long;};
            //int32_t *GetQBarret() const { return _Q_barret;};
	        
            int16_t GetQ(int idx) const             { return _QINFO[ QDATA_Q                    + idx*QDATA_LEN]; };
            int16_t GetQMontgo(int idx) const       { return _QINFO[ QDATA_Q_MONT_SIGN          + idx*QDATA_LEN]; }; 
            int16_t GetOneMontgo(int idx) const     { return _QINFO[ QDATA_Q_MONT_SIGN_ONE      + idx*QDATA_LEN]; };
			int16_t GetSquareMontgo(int idx) const  { return _QINFO[ QDATA_Q_MONT_SIGN_SQUARE   + idx*QDATA_LEN]; };
			int16_t GetOneMontgoInv(int idx) const  { return _QINFO[ QDATA_Q_MONT_SIGN_INV      + idx*QDATA_LEN]; };
      		int16_t GetNTTDepth(int idx) const      { return _QINFO[ QDATA_QDEPTH               + idx*QDATA_LEN]; };
            
			const int16_t *GetNTTINFO()		const {return _QNTTINFO + _QNTT_START_idx;}	;
			const int16_t *GetMULNTTINFO()	const {return _QNTTINFO + _QMUL_START_idx;}	;
    	    const int16_t *GetINTTINFO()	const {return _QNTTINFO + _QINTT_START_idx;}	;
    	    const int16_t *GetGADGETINFO()	const {return _QNTTINFO + _QGADGET_START_idx;}	;

			auto GetFuncMulMontInNTT()				const { return _MUL_MONT_IN_NTT;};        
			auto GetFuncMulMontInNTTAtIdx(int idx)	const { return _MUL_MONT_IN_NTT[idx];};        
			auto GetFuncNTTToMont()					const { return _NTT_TO_MONT; };        
			auto GetFuncINTTToMont()				const { return _INTT_TO_MONT;};        
			auto GetFuncADD()						const { return _ADD;};        
			auto GetFuncSUB()						const { return _SUB;};        
		
		


		
	//uint32_t *GetBarretShift() const { return _barret_shift;};
           //int16_t **GetRoot() const { return _roots;}
            //int16_t **GetIRoot() const { return _iroots;}
            //int16_t **GetinRoot() const { return _in_roots;}
            //int16_t GetINV(int idx) const { return _inv;}
            //int16_t GetINVPR( int idx) const { return _invpr;}
            
            int64_t *GetCRTbasis() const { return _crt_basis;}
            int16_t FastAddWithRangeZeroToQ(const int16_t a, const int16_t b, int16_t Q) const;
            int16_t FastSubWithRangeZeroToQ(const int16_t a, const int16_t b, int16_t Q) const;
			int16_t FastAddWithRangeQOverTwoToQOverTwo(const int16_t a, const int16_t b, int16_t Q) const;
	        int16_t BarretMulWithRangeZeroToQ(const int16_t a, const int16_t b, int16_t Q, int16_t shift, int16_t longs) const;
            /*
            int16_t Int2Montgo16bit(int16_t a) const;
            
            int16_t Mul16bit_Montgo(int16_t a, int16_t b, int16_t Q, int16_t montgo, int16_t one_inv) const;
            int16_t Mul16bit_Barret(const int32_t c, int16_t Q, int16_t shift, int16_t longs) const;
            int16_t bit16_Montgo(int32_t P, int16_t Q, int16_t montgo, int16_t one_inv) const;
            */
			// NTT functionality
			void init_ntt_16bit();
            void ReductionTest();
            // Multiplication method
            

            // Get QNTTTable
            const int16_t * GetNTTDATA()    const { return _QNTTINFO;}
            int32_t * GetDATA()       const { return _QINFO;}
            int64_t * GetSTARTIDX()       const { return _QINFO_start_arr;}
          
            // Not SAFE

			int16_t * GetNTTDATAIdx(int idx) const { 
				return _QNTTINFO + _Gadget_END_idx  + _QINFO_start_arr[idx]; 
            };
			int32_t * GetDATA(int idx)       const { return _QINFO + idx*QDATA_LEN;};

            void Mul0MontSchool(int16_t *res, int16_t * x, int16_t *y) const;
            void Mul1MontSchool(int16_t *res, int16_t * x, int16_t *y) const;
            void Mul2MontSchool(int16_t *res, int16_t * x, int16_t *y) const;
            void Mul3MontSchool(int16_t *res, int16_t * x, int16_t *y) const;
            void Mul4MontSchool(int16_t *res, int16_t * x, int16_t *y) const;
            void Mul5MontSchool(int16_t *res, int16_t * x, int16_t *y) const;
            void Mul0BarrSchool(int16_t *res, int16_t * x, int16_t *y) const;
            void Mul1BarrSchool(int16_t *res, int16_t * x, int16_t *y) const;
            void Mul2BarrSchool(int16_t *res, int16_t * x, int16_t *y) const;
            void Mul3BarrSchool(int16_t *res, int16_t * x, int16_t *y) const;
            void Mul4BarrSchool(int16_t *res, int16_t * x, int16_t *y) const;
            void Mul5BarrSchool(int16_t *res, int16_t * x, int16_t *y) const;
            
			//Gadget Functionality
			void GadgetDecompTwoPowRemoveOne(const int16_t * from, int16_t * to, int dim, int poly_len, int base, int gl_len) const;
	};




    class NTTTable64 {
        private:
		    
			// FUNCTIONS
			void (**_MUL_MONT_IN_NTT)(int64_t*, int64_t*, int64_t*, int)					= nullptr;			
			void (**_NTT_TO_MONT)(int64_t*, int64_t*, int64_t*, int)						= nullptr;			
			void (**_INTT_TO_MONT)(int64_t*, int64_t*, int64_t*, int)						= nullptr;			
			void (**_ADD)(const int64_t*, const int64_t*, int64_t*,	int64_t, int64_t, int)	= nullptr;			
			void (**_SUB)(const int64_t*, const int64_t*, int64_t*,	int64_t, int64_t, int)	= nullptr;			





			int64_t **   _roots      = nullptr;
            int64_t **   _iroots     = nullptr;
            int64_t **   _in_roots   = nullptr;
                
            int64_t *    _inv        = nullptr; 
            int64_t *    _invpr      = nullptr;
            int64_t *    _crt_basis  = nullptr;
                
            int64_t     _N;
            int64_t     _N_log;
            
            // Q setting
            int64_t     _Q_num;
            int64_t     *_Q = nullptr;
            int64_t     _Q_TOT = 0;
            int64_t     _Q_TOT_bit = 0;

            //int32_t     *_Q_barret = nullptr;
            //int64_t		*_Q_barret_long = nullptr; // check
            int64_t     *_Q_montgo = nullptr; 
            int64_t     *_One_montgo = nullptr;
            int64_t		*_One_montgo_inv = nullptr;
            int64_t     *_One_square_montgo = nullptr;
            //uint32_t    *_barret_shift = nullptr;
            int64_t    *_NTT_depth = nullptr; 
            //INCOMPLETE_DEPTH_BIN;
           
        public:
            explicit NTTTable64(); // Default ...
            explicit NTTTable64(int64_t *Q, int N, int q_num);
            ~NTTTable64();
            uint32_t reverse(uint32_t v, int N_log);
            int64_t spmd(int64_t x,int64_t n,int64_t m);
            int64_t invers(int64_t x,int64_t y);
            int64_t sqrmp(int64_t x,int64_t m);
            int64_t xGCD(int64_t a,int64_t b, int64_t &x, int64_t &y);
            int64_t INV64(int64_t a,int64_t Q);
            int64_t *GetQ() const { return _Q;};
		    int64_t GetQtot() const {return _Q_TOT;};
            int     GetQnum() const { return _Q_num;};
			int     GetN() const { return _N;};
			//int64_t	*GetQBarretLong() const { return _Q_barret_long;};
            //int32_t *GetQBarret() const { return _Q_barret;};
	        int64_t *GetQMontgo() const { return  _Q_montgo;}; 
            int64_t *GetOneMontgo() const { return _One_montgo;};
			int64_t *GetOneSquareMontgo() const { return _One_square_montgo;};
			int64_t *GetOneMontgoInv() const { return _One_montgo_inv;};
            //uint32_t *GetBarretShift() const { return _barret_shift;};
     		int64_t *GetNTTDepth() const { return _NTT_depth;};
            int64_t **GetRoot() const { return _roots;}
            int64_t **GetIRoot() const { return _iroots;}
            int64_t **GetinRoot() const { return _in_roots;}
            int64_t *GetINV() const { return _inv;}
            int64_t *GetINVPR() const { return _invpr;}
            int64_t *GetCRTbasis() const { return _crt_basis;}
            int64_t FastAddWithRangeZeroToQ(const int64_t a, const int64_t b, int64_t Q) const;
            int64_t FastSubWithRangeZeroToQ(const int64_t a, const int64_t b, int64_t Q) const;
			int64_t FastAddWithRangeQOverTwoToQOverTwo(const int64_t a, const int64_t b, int64_t Q) const;
	        int64_t BarretMulWithRangeZeroToQ(const int64_t a, const int64_t b, int64_t Q, int64_t shift, int64_t longs) const;
            int64_t Int2Montgo64bit(int64_t a) const;
            
            int64_t Mul64bit_Montgo(int64_t a, int64_t b, int64_t Q, int64_t montgo, int64_t one_inv) const;
            int64_t Mul64bit_Barret(const int64_t c, int64_t Q, int64_t shift, int64_t longs) const;
           
            int64_t bit32_Montgo(int64_t P, int64_t Q, int64_t montgo, int64_t one_inv) const;
            
			// NTT functionality
			void init_ntt_64bit();

           
			//Gadget Functionality
			void GadgetDecompTwoPowRemoveOne(const int64_t * from, int64_t * to, int32_t dim, int32_t poly_len, int32_t base, int32_t gl_len) const;
	};




};




#endif // HEADER END
