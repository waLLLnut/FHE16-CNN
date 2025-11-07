#ifndef ARITHMETIC_H_EF
#define ARITHMETIC_H_EF

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

#include <param.h>
#include <reduction.h>



//#include <structures.h>
    
    void test_ar();

    //asm_Montgo_Exact
    
    /*Maybe Later*/
    extern void asm_Montgo_Exact(uint32_t *out, uint32_t x, uint32_t y);
    extern void asm_Mod_Exact(uint32_t *out, uint32_t x, uint32_t y);
    extern void asm_Barret(uint32_t *out, uint32_t x, uint32_t y);
    extern void asm_Montgo_fast2(uint32_t *out, uint32_t x, uint32_t y);
    

    void power2(int *out, int num, int power );
    int gcd( int a, int b );
    //void SchoolBook_32bit(uint32_t *out, uint32_t *a, uint32_t *b, uint32_t N);
   


	/*
	void Mul0MontSchool_32bit(int32_t *out, int32_t *a, int32_t *b, int32_t *root, int32_t N, int32_t Q, int32_t mont, int32_t One_mont_inv); 
    void Mul1MontSchool_32bit(int32_t *out, int32_t *a, int32_t *b, int32_t *root, int32_t N, int32_t Q, int32_t mont, int32_t One_mont_inv);



    void Mul2MontSchool_32bit(int32_t *out, int32_t *a, int32_t *b, int32_t *root, int32_t N, int32_t Q, int32_t mont, int32_t One_mont_inv);
    void Mul3MontSchool_32bit(int32_t *out, int32_t *a, int32_t *b, int32_t *root, int32_t N, int32_t Q, int32_t mont, int32_t One_mont_inv);
    void Mul4MontSchool_32bit(int32_t *out, int32_t *a, int32_t *b, int32_t *root, int32_t N, int32_t Q, int32_t mont, int32_t One_mont_inv);
    void Mul5MontSchool_32bit(int32_t *out, int32_t *a, int32_t *b, int32_t *root, int32_t N, int32_t Q, int32_t mont, int32_t One_mont_inv);
    
    
    void Mul1MontKara_32bit(int32_t *out, int32_t *a, int32_t *b, int32_t *root, int32_t N, int32_t Q, int32_t mont, int32_t One_mont_inv);
    void Mul2MontKara_32bit(int32_t *out, int32_t *a, int32_t *b, int32_t *root, int32_t N, int32_t Q, int32_t mont, int32_t One_mont_inv);
    void Mul3MontKara_32bit(int32_t *out, int32_t *a, int32_t *b, int32_t *root, int32_t N, int32_t Q, int32_t mont, int32_t One_mont_inv);
    
    void Mul3MontToom_32bit(int32_t *out, int32_t *a, int32_t *b, int32_t *root, int32_t N, int32_t Q, int32_t mont, int32_t One_mont_inv);
	*/


void Mul0MontSchool_32bit(int32_t *out, int32_t *a, int32_t *b, const int32_t *root, const int32_t *INFO);
void Mul1MontSchool_32bit(int32_t *out, int32_t *a, int32_t *b, const int32_t *root, const int32_t *INFO);
void Mul2MontSchool_32bit(int32_t *out, int32_t *a, int32_t *b, const int32_t *root, const int32_t *INFO);
void Mul3MontSchool_32bit(int32_t *out, int32_t *a, int32_t *b, const int32_t *root, const int32_t *INFO);









    /*Step Check */
    
    //void Karachuba_32bit_deg2_Montgo(uint32_t *out, uint32_t *a, uint32_t *b, uint32_t *root, uint32_t N);
    //void SchoolBook_32bit_deg2_Montgo(uint32_t *out, uint32_t *a, uint32_t *b, uint32_t *root, uint32_t N);
  
    /*DEGREE 4 CHECK*/
    //void SchoolBook_32bit_deg4_Montgo(uint32_t *out, uint32_t *a, uint32_t *b, uint32_t *root, uint32_t N);
    //void Karachuba_32bit_deg4_Montgo(uint32_t *out, uint32_t *a, uint32_t *b, uint32_t *root, uint32_t N);
    //void Karachuba1_arr4(uint32_t *out, uint32_t *a, uint32_t *b, uint32_t *root, uint32_t N);
    //void ToomCook_32bit_deg4_Montgo(uint32_t *out, uint32_t *a, uint32_t *b, uint32_t *root, uint32_t N);

    /* DEGREE 8 CHECK*/
    //void SchoolBook_32bit_deg8_Montgo(uint32_t *out, uint32_t *a, uint32_t *b, uint32_t *root, uint32_t N);
    //void Karachuba_32bit_deg8_Montgo(uint32_t *out, uint32_t *a, uint32_t *b, uint32_t *root, uint32_t N);
 
    /* DEGREE 16 CHECK*/
    //void SchoolBook_32bit_deg16_Montgo(uint32_t *out, uint32_t *a, uint32_t *b, uint32_t *root, uint32_t N);

    /* DEGREE 32 CHECK*/
    //void SchoolBook_32bit_deg32_Montgo(uint32_t *out, uint32_t *a, uint32_t *b, uint32_t *root, uint32_t N);





/*
#define Int2Montgo32bit(a, out) \
    Mul32bit_Barret(a, ONE_32bit, out) \
    IfandMinus32bit((int32_t) out, EF_Q_32bit)\
    out = (uint32_t) Int2Montgo32bit_tmp;
  
#define Montgo2Int32bit(a, out) \
    Mul32bit_Barret(a, ONE_INV_32bit, out) \
    tmp = (int32_t) out; \
    IfandMinus32bit(tmp, EF_Q_32bit)\
    out = (uint32_t) tmp;
    
*/

//#define Mul32bit_Montgo(a, b, out) \
//uint64_t P = ((uint64_t)a * (uint64_t) b); \
//out = ((((uint64_t)((uint32_t)P*(uint32_t)MONT_MU_32bit)) * EF_Q_32bit) +P) >> 32;


//Montgo_32bit((((uint64_t)a) * ((uint64_t)b)), out);

/*
#define Montgo_32bit(P, out) \
out = ((((uint64_t)((uint32_t)P*(uint32_t)MONT_MU_32bit)) * EF_Q_32bit) +P) >> 32;
*/


/*
uint32_t inline Montgo_32bit(uint64_t P) {
return  (uint32_t)(((((uint64_t)((uint32_t)P*(uint32_t)MONT_MU_32bit)) * EF_Q_32bit) +P) >> 32);
}
*/
uint32_t inline Mul32bit_Montgo(uint32_t a, uint32_t b) {
    uint64_t P = ((uint64_t)a * (uint64_t) b); 
    return (uint32_t)(((((uint64_t)((uint32_t)P*(uint32_t)MONT_MU_32bit)) * EF_Q_32bit) +P) >> 32);
}
uint32_t inline Mul32bit_Barret(uint32_t a, uint32_t b) {
    uint64_t P = ((uint64_t)a * (uint64_t) b); 
    return ((uint32_t)(P -((((P >> BARRET_DIV1_BIT_32bit) * BARRET_MU_32bit) >> BARRET_DIV2_BIT_32bit)* EF_Q_32bit)));
}


#define Barret_32bit(P, out) \
out = ((uint32_t)((P) -(((((P) >> BARRET_DIV1_BIT_32bit) * BARRET_MU_32bit) >> BARRET_DIV2_BIT_32bit)* EF_Q_32bit)));


#define IfandMinus32bit(val, q) \
val= (val) - (q); \
val= (val) + (((val)>>(31))&(q)) 

#define IfandMinus64bit(val, q) \
val= (val) - (q); \
val= (val) + (((val)>>(63))&(q)) 


uint32_t inline Int2Montgo32bit(uint32_t a){
    uint32_t out1 = Mul32bit_Barret(a, ONE_32bit);
    int32_t out2 = (int32_t) out1;
    IfandMinus32bit(out2, EF_Q_32bit);
    return (uint32_t) out2;
}
uint32_t inline Montgo2Int32bit(uint32_t a) {
    uint32_t out = Mul32bit_Barret(a, ONE_INV_32bit);
    int32_t tmp = (int32_t) out; 
    IfandMinus32bit(tmp, EF_Q_32bit);
    return (uint32_t) tmp;
}






/*
#define Mul32bit_Barret(a, b, out) \
uint64_t P = ((uint64_t)a * (uint64_t) b); \
out = ((uint32_t)(P -((((P >> BARRET_DIV1_BIT_32bit) * BARRET_MU_32bit) >> BARRET_DIV2_BIT_32bit)* EF_Q_32bit)));
*/
 

/****************************** 16bit **********************************/

void Mul0MontSchool_16bit(int16_t *out, int16_t *a, int16_t *b, const int16_t *root, const int32_t *INFO);
void Mul1MontSchool_16bit(int16_t *out, int16_t *a, int16_t *b, const int16_t *root, const int32_t *INFO);
void Mul_any_MontSchool_16bit(int16_t *out, int16_t *a, int16_t *b, const int16_t *root, const int32_t *INFO, int DEPTH);



#ifdef __cplusplus
}
#endif
#endif // HEADER END
