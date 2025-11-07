#ifndef FHE16_BINOPERATIONCSTYLE_H
#define FHE16_BINOPERATIONCSTYLE_H

#include<BINFHE.hpp>


#include<poly_class.hpp>
#include<lwe.hpp>
#include<FHE16Param.hpp>
#include<CMAKEPARAM.h>
#include <numa.h>


enum BIN_EV_METHOD : std::uint8_t;
//extern  BIN_EV_METHOD;
//enum BIN_EV_METHOD;
/*
    enum BIN_EV_METHOD {
        GINX_32bit,
        GINX_16bit,
        LMKCDEY_32bit,
        LMKCDEY_16bit
    };  
*/

void C_BeforeBoot_KSFirst(
			int32_t *IN_RES1,
			int16_t *KS_raw_16bit, 
			int16_t *BK_raw_16bit, 
			int16_t *ACC,
            int16_t *ROT,
			int16_t *MEMORY_HANDLE, 
			FHE16Params *PARAM, 
			NTTTable16Struct *st,
			BIN_EV_METHOD METHOD
			) ;
        
	void C_FHE16_OR(const int32_t *c1, const int32_t *c2, int32_t *res,
		FHE16BOOTParam* BOOTParams,
		BIN_EV_METHOD METHOD);
  /*
	void C_FHE16_AND(const int32_t *c1, const int32_t *c2, int32_t *res,
			
			int16_t *KS_raw_16bit, 
			int16_t *BK_raw_16bit, 
			int16_t *ACC,
            int16_t *ROT,
			int16_t *MEMORY_HANDLE, 
			FHE16Params *PARAM, 
			NTTTable16Struct *st,
			
			FHE16BOOTParam* BOOTParams,
			BIN_EV_METHOD METHOD);
  	
*/
	void C_FHE16_AND(const int32_t *c1, const int32_t *c2, int32_t *res,
			FHE16BOOTParam* BOOTParams,
			BIN_EV_METHOD METHOD);
  	


	void C_FHE16_XOR(const int32_t *c1, const int32_t *c2, int32_t *res,
			FHE16BOOTParam* BOOTParams,
			BIN_EV_METHOD METHOD);
 
	void C_FHE16_XOR3(const int32_t *c1, const int32_t *c2, const int32_t *c3, int32_t *res,
			FHE16BOOTParam* BOOTParams,
			BIN_EV_METHOD METHOD);
  	
	void C_FHE16_XOR4(const int32_t *c1, const int32_t *c2, const int32_t *c3, 
			const int32_t *c4,
			int32_t *res,
			FHE16BOOTParam* BOOTParams,
			BIN_EV_METHOD METHOD);
  

	void C_FHE16_XOR5(const int32_t *c1, const int32_t *c2, const int32_t *c3, 
			const int32_t *c4, const int32_t *c5,
			int32_t *res,
			FHE16BOOTParam* BOOTParams,
			BIN_EV_METHOD METHOD);
  
	void C_FHE16_XOR6(const int32_t *c1, const int32_t *c2, const int32_t *c3, 
			const int32_t *c4, const int32_t *c5, const int32_t *c6,
			int32_t *res,
			FHE16BOOTParam* BOOTParams,
			BIN_EV_METHOD METHOD);
  
	void C_FHE16_XOR7(const int32_t *c1, const int32_t *c2, const int32_t *c3, 
			const int32_t *c4, const int32_t *c5, const int32_t *c6, const int32_t *c7,
			int32_t *res,
			FHE16BOOTParam* BOOTParams,
			BIN_EV_METHOD METHOD);
  


	void C_FHE16_AND_XOR(const int32_t *c1, const int32_t *c2, const int32_t *c3, int32_t *res,
			FHE16BOOTParam* BOOTParams,
			BIN_EV_METHOD METHOD);
  	
	void C_FHE16_OR_XOR(const int32_t *c1, const int32_t *c2, const int32_t *c3, int32_t *res,
			FHE16BOOTParam* BOOTParams,
			BIN_EV_METHOD METHOD);
  	

	void C_FHE16_MAJ3(const int32_t *c1, const int32_t *c2, const int32_t *c3, int32_t *res,
			FHE16BOOTParam* BOOTParams,
			BIN_EV_METHOD METHOD);
  	
	void C_FHE16_EQ3(const int32_t *c1, const int32_t *c2, const int32_t *c3, int32_t *res,
			FHE16BOOTParam* BOOTParams,
			BIN_EV_METHOD METHOD);
  	


	void C_FHE16_NOT(const int32_t *c1, int32_t *res,
			FHE16BOOTParam* BOOTParams);
  





	/*
	void C_BeforeBoot(int16_t *tmp_val) const;
    void C_BeforeBoot_KSFirst(int32_t *tmp_val) const;
    void C_BeforeBoot_KSFirst_2OUT(int32_t *tmp_val, int32_t *tmpval2) const;
    void C_BeforeBoot_PackedKSFirst(int32_t *tmp_val) const;
        
	void C_BeforeBootSIMD(const LWECiphertext *c1, const LWECiphertext *c2, LWECiphertext* & res, int32_t offset, bool neg, int window) const;
		
	void C_BeforeBootAccelerate(const LWECiphertext &c1, const LWECiphertext &c2, LWECiphertext &res, int32_t offset, bool neg) const;
	void C_Bootstrapping2_1(void *inputs, void *BL_MEMORY, void * KS_MEMORY) const; 
	void C_BeforeBoot_TD(const LWECiphertext *c1, const LWECiphertext *c2, LWECiphertext* & res, const int32_t offset, const bool neg, int window, int Type) const;

        // Operation
    void C_TWOOUT_AND_OR(const LWECiphertext &c1, const LWECiphertext &c2, LWECiphertext & res, LWECiphertext &res2) const;
		 

	void C_AND(const LWECiphertext &c1, const LWECiphertext &c2, LWECiphertext & res) const;
	void C_AND_parallel(const LWECiphertext* c1, const LWECiphertext* c2, LWECiphertext* res, int window) const;
	void C_AND_TD(const LWECiphertext* c1, const LWECiphertext* c2, LWECiphertext* res, int WINDOW, int TYPE) const; 
		
	void C_AND_accelerate(const LWECiphertext &c1, const LWECiphertext &c2, LWECiphertext &res) const;
	void C_NAND(const LWECiphertext &c1, const LWECiphertext &c2, LWECiphertext & res) const;
    void C_OR(const LWECiphertext &c1, const LWECiphertext &c2, LWECiphertext & res) const;
    void C_XOR(const LWECiphertext &c1, const LWECiphertext &c2, LWECiphertext & res) const;
	void C_NOT(const LWECiphertext &c1, LWECiphertext & res) const;
        

		// Primitive gate operation
	void C_XOR3(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, LWECiphertext & res) const;
   	void C_XOR4(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, const LWECiphertext &c4,  LWECiphertext & res) const;
	void C_XOR5(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, const LWECiphertext &c4, const LWECiphertext &c5, LWECiphertext & res) const;
	void C_XOR6(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, const LWECiphertext &c4, const LWECiphertext &c5, const LWECiphertext &c6, LWECiphertext & res) const;
	void C_XOR7(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, const LWECiphertext &c4, const LWECiphertext &c5, const LWECiphertext &c6, const LWECiphertext &c7, LWECiphertext & res) const;



	void C_MAJ3(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, LWECiphertext & res) const;
 	void C_EQ3(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, LWECiphertext & res) const;
 	void C_AND_XOR6(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, LWECiphertext & res) const;
	void C_OR_XOR6(const LWECiphertext &c1, const LWECiphertext &c2, const LWECiphertext &c3, LWECiphertext & res) const;
		

*/
#endif // End header
