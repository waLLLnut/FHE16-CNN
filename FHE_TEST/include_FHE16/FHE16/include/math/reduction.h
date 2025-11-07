#ifndef REDUCTION_EF_1
#define REDUCTION_EF_1

#ifndef __cplusplus
#include<stdint.h>
#endif


#include<CMAKEPARAM.h>

#if CPUTYPE == 1 // IF LINUX
	#include<stdint.h>
	#include<stdbool.h>

#endif

#if AVXTYPE == 2
	#include<immintrin.h>
#elif AVXTYPE == 3
	#include<immintrin.h>
#elif AVXTYPE == 5

#endif




#include<param.h>
#include<stdio.h>
#include<stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif





/******************************** 32bit and 16bit vector reduction ***********************************/



inline void FastAddWithRangeZeroToQVec_16bit(const int16_t *a, const int16_t *b, int16_t *c, int16_t Q, int16_t Qne, int N) {
    for (int ii = 0; ii < N; ii++) { 
		int16_t tmp = ((a[ii] - Q) + b[ii]);
		c[ii] = tmp + ( ( (tmp) >>(15) )&(Q));
	}
}


inline void FastAddWithRangeQneToQVec_16bit(const int16_t *a, const int16_t *b, int16_t *c,  int16_t Q, int16_t Q_ne, int N) {
    for (int ii = 0; ii < N; ii++) { 
		int16_t tmp = (a[ii] + b[ii]);
		int16_t flag	=	 tmp >> 15;
		c[ii] = tmp +  ((Q & flag) ^ (Q_ne &(~flag)));
		//return c;
	}
}

inline void FastSubWithRangeZeroToQVec_16bit(const int16_t *a, const int16_t *b, int16_t *c, int16_t Q, int16_t Qne, int N) {
    for (int ii = 0; ii < N; ii++) { 
		int16_t tmp = ((a[ii] ) - b[ii]);
		c[ii] = tmp + ( ( (tmp) >>(15) )&(Q));
	}
}


inline void FastSubWithRangeQneToQVec_16bit(const int16_t *a, const int16_t *b, int16_t *c,  int16_t Q, int16_t Q_ne, int N) {
    for (int ii = 0; ii < N; ii++) { 
		int16_t tmp = (a[ii] - b[ii]);
		int16_t flag	=	 tmp >> 15;
		c[ii] = tmp +  ((Q & flag) ^ (Q_ne &(~flag)));
		//return c;
	}
}





inline int16_t FastAddWithRangeZeroToQVec_32bit(const int32_t a, const int32_t b, int32_t *c,  int32_t Q, int32_t Qne, int N) {
    //int32_t c = ((a - Q) + b);
    //return (c + ( ( (c) >>(31) )&(Q)));
}


inline int16_t FastAddWithRangeQneToQVec_32bit(const int32_t a, const int32_t b, int32_t *c, int32_t Q, int32_t Q_ne, int N) {
    /*
	int16_t c = (a + b);
    int16_t	flag	=	 c >> 31;
	c +=  ((Q & flag) ^ (Q_ne &(~flag)));
	return c;
	*/
}





/********************************** 32bit *******************************************/

inline int32_t FastAddWithRangeZeroToQ_32bit(const int32_t a, const int32_t b, int32_t Q) {
    int32_t c = ((a - Q) + b);
    return (c + (((c)>>(31))&(Q)));
}


inline void FastAddPolyWithRangeZeroToQ_32bit(int32_t *res, const int32_t *a, const int32_t *b, int32_t Q, int32_t N) {
    for (int ii = 0; ii < N; ii++) {
       res[ii] = a[ii]+ b[ii] - Q;
       res[ii] += (res[ii] >> 31) & Q;
    }
}


inline int32_t FastSubWithRangeZeroToQ_32bit(const int32_t a, const int32_t b, int32_t Q) {
    int32_t c = a - b;
    return (c + (((c)>>(31))&(Q)));
}

inline int32_t MulBarrett_32bit(const int32_t a, const int32_t b, int32_t Q, int32_t shift, int32_t longs) {
    int64_t c = (int64_t) a * (int64_t) b;
    int32_t d = (int32_t ) (c - ( (int64_t) Q) * (((( 
        (c ) >> (shift - 2)) * longs) >>(shift + 3))+1) );
    return d + (((d)>>(31))&(Q));     
}    

inline int32_t MulMontgo_32bit(int32_t a, int32_t b, int32_t Q, int32_t montgo, int32_t One_inv) {
    int64_t P = ((int64_t)(a) * (int64_t)(b));
    int32_t P1 = (int32_t) (P >> 32);
    uint32_t P0 = (uint32_t) (P); // mod 2^32
    int32_t P0_tmp = P0 * montgo;
    // Why....
    P1 = (P1) - ((int32_t) (        (  ( (int64_t)( (P0_tmp)) ) * ((int64_t)(Q)) )   >> 32 ));
    return P1 +( (P1 >> 31) & Q);
};

inline int32_t Montgo_32bit(int64_t P, int32_t Q, int32_t montgo, int32_t One_inv) {
    int32_t P1 = (int32_t) (P >> 32);
    uint32_t P0 = (uint32_t) (P); // mod 2^32
    int32_t P0_tmp = P0 * montgo;

    P1 = (P1) - ((int32_t) (        (  ( (int64_t)( (P0_tmp) ) ) * ((int64_t)(Q)) )   >> 32 ));
    return P1 +( (P1 >> 31) & Q);
};


inline int32_t Int2Montgo_32bit(int32_t a, int32_t One_square, int32_t Q, int32_t Q_montgo, int32_t One_montgo_inv) { 
    return MulMontgo_32bit(a, One_square, Q, Q_montgo, One_montgo_inv);
};

inline int32_t Montgo2Int_32bit(int32_t a, int32_t One_square, int32_t Q, int32_t Q_montgo, int32_t One_montgo_inv) { 
    return Montgo_32bit(a, Q, Q_montgo, One_montgo_inv);
};


/**************************************** 16bit *******************************************/

inline int16_t FastAddWithRangeZeroToQ_16bit(const int16_t a, const int16_t b, int16_t Q) {
    int16_t c = ((a - Q) + b);
    return (c + ( ( (c) >>(15) )&(Q)));
}


#if AVXTYPE == 3

__attribute__((always_inline)) inline  void FastAddWithRangeQneToQ_16bitVec(const int16_t *x, const int16_t *y, int16_t *res, int16_t Q,int N) {

	__m512i Q_vec	= _mm512_set1_epi16(Q);
	__m512i ZERO	= _mm512_setzero_si512();
	
	__m512i flagA1;
	__m512i flagA2;
	__m512i flagA3;
	__m512i flagA4;

	__m512i flagB1;
	__m512i flagB2;
	__m512i flagB3;
	__m512i flagB4;


	__mmask32 mask1;
	__mmask32 mask2;
	__mmask32 mask3;
	__mmask32 mask4;
	
	__m512i a1;
	__m512i a2;
	__m512i a3;
	__m512i a4;

	__m512i b1;
	__m512i b2;
	__m512i b3;
	__m512i b4;
	
	__m512i c1;
	__m512i c2;
	__m512i c3;
	__m512i c4;
	
	for (int ii = 0; ii < N; ii+= 32*4) {
		a1 = _mm512_load_si512(x+ii+32*0);
		a2 = _mm512_load_si512(x+ii+32*1);
		a3 = _mm512_load_si512(x+ii+32*2);
		a4 = _mm512_load_si512(x+ii+32*3);
		
		b1 = _mm512_load_si512(y+ii+32*0);
		b2 = _mm512_load_si512(y+ii+32*1);
		b3 = _mm512_load_si512(y+ii+32*2);
		b4 = _mm512_load_si512(y+ii+32*3);

		// 1. c = a + b
		c1 = _mm512_add_epi16(a1, b1);
		c2 = _mm512_add_epi16(a2, b2);
		c3 = _mm512_add_epi16(a3, b3);
		c4 = _mm512_add_epi16(a4, b4);

		mask1	= _mm512_cmp_epi16_mask(c1,ZERO, _MM_CMPINT_LT);
		mask2	= _mm512_cmp_epi16_mask(c2,ZERO, _MM_CMPINT_LT);
		mask3	= _mm512_cmp_epi16_mask(c3,ZERO, _MM_CMPINT_LT);
		mask4	= _mm512_cmp_epi16_mask(c4,ZERO, _MM_CMPINT_LT);

		flagA1			=	_mm512_add_epi16(c1, Q_vec);
		flagA2			=	_mm512_add_epi16(c2, Q_vec);
		flagA3			=	_mm512_add_epi16(c3, Q_vec);
		flagA4			=	_mm512_add_epi16(c4, Q_vec);
	

		flagB1			=	_mm512_sub_epi16(c1, Q_vec);
		flagB2			=	_mm512_sub_epi16(c2, Q_vec);
		flagB3			=	_mm512_sub_epi16(c3, Q_vec);
		flagB4			=	_mm512_sub_epi16(c4, Q_vec);
		
		c1 = _mm512_mask_blend_epi16(mask1, flagB1, flagA1);
		c2 = _mm512_mask_blend_epi16(mask2, flagB2, flagA2);
		c3 = _mm512_mask_blend_epi16(mask3, flagB3, flagA3);
		c4 = _mm512_mask_blend_epi16(mask4, flagB4, flagA4);

		_mm512_store_si512(res + ii+ 32*0, c1);
		_mm512_store_si512(res + ii+ 32*1, c2);
		_mm512_store_si512(res + ii+ 32*2, c3);
		_mm512_store_si512(res + ii+ 32*3, c4);




	}
}

#endif

inline int16_t FastAddWithRangeQneToQ_16bit(const int16_t a, const int16_t b, int16_t Q, int16_t Q_ne) {
    int16_t c = (a + b);
    int16_t	flag	=	 c >> 15;
	c +=  ((Q & flag) ^ (Q_ne &(~flag)));
	return c;
}






inline void FastAddPolyWithRangeZeroToQ_16bit(int16_t *res, const int16_t *a, const int16_t *b, int16_t Q, int16_t N) {
    for (int ii = 0; ii < N; ii++) {
       res[ii] = ((a[ii] - Q) + (b[ii]));
       res[ii] += (res[ii] >> 15) & Q;
    }
}


inline int16_t FastSubWithRangeZeroToQ_16bit(const int16_t a, const int16_t b, int16_t Q) {
    int16_t c = (a - b);
    return (c +  (  ( (c) >>(15)) &(Q) ));
}




//////////////////////////////////////// CHECK OK /////////////////////////////////////


/*********** Requirement ***********
* q: 0 < q< 2^15 odd number
* a = -2^15 q \leq a \leq 2^15 q
* a=a_1 2^16+a_0 / 0 leq a_0 <\beta

************************************/
inline int16_t MulMontgo_16bit(int16_t a, int16_t b, int16_t Q, int16_t montgo, int16_t One_inv) {
    int32_t P = ((int32_t)(a) * (int32_t)(b));
    int16_t P1 = (int16_t) (P >> 16);
    uint16_t P0 = (uint16_t) (P); // mod 2^32  
    int16_t P0_prod = P0 * montgo;
    P1 = (P1) - ((int16_t) (        (  ( (int32_t)( P0_prod) ) * ((int32_t)(Q)) )   >> 16 ));
    P1 += (P1 >> 15) & Q; 
   
    return P1;
    //return P1 +( (P1 >> 15) & Q);
}


inline int16_t MulMontgoOne_16bit(int16_t a, int16_t Q, int16_t montgo, int16_t One_inv) {
	int16_t P1 = (int16_t) (a >> 15);
   
	int16_t P0_prod = a * montgo;
    P1 = (P1) - ((int16_t) (        (  ( (int32_t)( P0_prod) ) * ((int32_t)(Q)) )   >> 16 )) ;
    P1 += (P1 >> 15) & Q; 
   
    return P1;
}




/*
inline int16_t MulPreBarret_16bit(int16_t a, int16_t b, int16_t b_prime, int16_t Q, int16_t montgo, int16_t One_inv) {
    int16_t P = ((int16_t)(a) * b;
    


	int16_t P1 = (int16_t) (P >> 16);
    uint16_t P0 = (uint16_t) (P); // mod 2^32  
    int16_t P0_prod = P0 * montgo;
    P1 = (P1) - ((int16_t) (        (  ( (int32_t)( P0_prod) ) * ((int32_t)(Q)) )   >> 16 ));
    P1 += (P1 >> 15) & Q; 
   
    return P1;
    //return P1 +( (P1 >> 15) & Q);
}
*/



inline int16_t MulMontgoUnsigned_16bit(uint16_t a, uint16_t b, int16_t Q, uint16_t montgo, uint16_t One_inv) {
    uint32_t P = ((uint32_t)(a) * (uint32_t)(b));
    uint16_t P1 = (uint16_t) (P >> 16);
    uint16_t P0 = (uint16_t) (P); // mod 2^32  
    uint16_t P0_prod = P0 * montgo;
    
    
    int16_t P3 = ((int16_t)(P1) - ((int16_t) (        (  ( (int32_t)( P0_prod) ) * ((int32_t)(Q)) )   >> 16 )));
    P3 += (P3 >> 15) & Q; 
   
    return P3;
    //return P1 +( (P1 >> 15) & Q);
}



inline int16_t Montgo_16bit(int32_t P, int16_t Q, int16_t montgo, int16_t One_inv) {
    int16_t P1 = (int16_t) (P >> 16);
    uint16_t P0 = (uint16_t) (P); // mod 2^32  
    int16_t P0_prod = P0 * montgo;
    
    P1 = (P1) - ((int16_t) (        (  ( (int32_t)( P0_prod) ) * ((int32_t)(Q)) )   >> 16 ));
    P1 += (P1 >> 15) & Q; 
   
    return P1;
    
};


inline int16_t Int2Montgo_16bit(int16_t a, int16_t One_square, int16_t Q, int16_t Q_montgo, int16_t One_montgo_inv) { 
    
    return MulMontgo_16bit(a, One_square, Q, Q_montgo, One_montgo_inv);
    //return a;
};

inline int16_t Montgo2Int_16bit(int16_t a, int16_t Q, int16_t Q_montgo, int16_t One_montgo_inv) { 
    return Montgo_16bit(a, Q, Q_montgo, One_montgo_inv);
    //return a;
};


inline int16_t MulBarrett_16bit(int16_t a, int16_t b, int16_t Q, int16_t barr, int16_t Q_bit) {
    printf("!!!! CAUTION THIS HAVE a BUG !!!!!!!!!!!!!!!!!!!!!!!! \n ");
    fflush(stdin);
    fflush(stdout);


    //wait(1);
    abort();
    /*
    int32_t c = (int32_t) a * (int32_t) b;
    int32_t c1 = c >> (Q_bit - 1);
    int32_t c2 = c1 * barr;
    int16_t c3 = c2 >> 16;
    int16_t c4 = c - Q * (c3 + 1);
    c4 += (c4 >> 15) & Q; 
    */
    //return c4;
    return 0;    
}    

// Product val should be less than 15+ Qbit
inline int16_t ShortBarrett_16bit(int16_t a, int16_t Q, int16_t barr, int16_t Q_bit) {
  
    int32_t t = ((int32_t) a) * ((int32_t) barr);
    int16_t t1 = (int16_t)(t >> (14 + Q_bit)); // 
    
    t1 *= Q;
    
    return a - t1;    
}   

inline int16_t ShortBarrett2_16bit(int16_t a, int16_t Q, int16_t barr, int16_t Q_bit) {
  
    int32_t t = ((int32_t) a) * ((int32_t) barr);
    int16_t t1 = (int16_t)(t >> (16 + Q_bit - 2)); //  // Lower ..
    
    t1 *= Q;
    
    return a - t1;    
}   




// Product val should be less than 15+ Qbit
inline int16_t ShortUBarrett_16bit(uint16_t a, uint16_t Q, uint16_t barr, int16_t Q_bit) {
  
    uint32_t t = ((uint32_t) a) * ((uint32_t) barr);
    int16_t t1 = (int16_t)(t >> (15 + Q_bit)); // 
    
    t1 *= Q;
    
    return (int16_t)(a - t1);    
}

inline int16_t ShoupMul_16bit(int16_t a, int16_t w, int16_t w_prime, int16_t Q) {
    
	uint16_t w_prime_u = (uint16_t) w_prime;
    uint32_t val = ((uint32_t) a) * ((uint32_t) w_prime_u);
	//INTEL, No carry version
	/*
	int16_t val2 = (int16_t)((uint16_t) (val >> 16));
	int16_t C = (a * w) - (Q *val2);
	C += (C>> 15) & Q;
    C -= Q;
	C += (C>> 15) & Q;
	*/
	// INTEL. Carry version


	int16_t carry = (val & ((1l << 15))) >> 15; // 16'th values
	int16_t val2 = (int16_t)((uint16_t) (val >> 16)) + carry;
	int16_t C = (a * w) - (Q *val2);
	C += (C>> 15) & Q;
  
	return C;

}

inline int16_t ShoupMulARM_16bit(int16_t a, int16_t w, int16_t w_prime, int16_t Q) {
    //uint16_t w_prime_u = (auint16_t) w_prime;
	/*
    int32_t val = ((int32_t) a) * ((int32_t) w_prime);
	int16_t val2 = (int16_t)((int16_t) ((val<<1) >> 16));
	*/

	int32_t val = ((int32_t) a) * ((int32_t) w_prime) * 2;
	
	//int16_t val2 = (int16_t)((int16_t) (( val >> 16) - ((val << 16) >> 31)));
	//int16_t val2 = (int16_t)((int16_t) (( val >> 16)   ));
	
	// Get 16bit 
	int16_t carry = (val & ((1l << 15))) >> 15; // 16'th values
	//printf("carry is %d \n", carry);


	int16_t val2 = (int16_t)((int16_t) (( val >> 16)  + carry ));
	int16_t C = (a * w) - (Q *val2);
    
	C += (C>> 15) & Q;

	return C;

}






 /*

inline int16_t SMontgoS_16bit(int32_t P, int16_t Q, int16_t montgo, int16_t One_inv) {
    
   // int16_t P1 = (int16_t) (P >> 16);
    //uint16_t P0 = (int16_t) (P); // mod 2^32
    // Why....
    //P1 = (P1) - ((int16_t) (        (  ( (int32_t)( (P0 *  montgo)) ) * ((int32_t)(Q)) )   >> 16 ));
    //return P1 +( (P1 >> 15) & Q);
    

    return (int16_t) (((uint64_t)(  ((int64_t) P) + ((int64_t) Q) *((int64_t) Q) *((int64_t) Q) )  ) %((uint64_t) Q));
};


inline int16_t MulMontgo_16bit(int16_t a, int16_t b, int16_t Q, int16_t montgo, int16_t One_inv) {
   
    uint32_t a_tmp = (uint32_t)(((uint64_t)(((int64_t) a) + ((int64_t) Q) *((int64_t) Q))) %((uint64_t) Q));
    uint32_t b_tmp = (uint32_t)(((uint64_t)(((int64_t) b) + ((int64_t) Q) *((int64_t) Q))) %((uint64_t) Q));

    return (int16_t)(a_tmp * b_tmp % ((uint32_t) Q));
};

inline int16_t Montgo_16bit(int32_t P, int16_t Q, int16_t montgo, int16_t One_inv) {
    
    //int16_t P1 = (int16_t) (P >> 16);
    //uint16_t P0 = (int16_t) (P); // mod 2^32
    // Why....
    //P1 = (P1) - ((int16_t) (        (  ( (int32_t)( (P0 *  montgo)) ) * ((int32_t)(Q)) )   >> 16 ));
    //return P1 +( (P1 >> 15) & Q);
    

    return (int16_t) (((uint64_t)(  ((int64_t) P) + ((int64_t) Q) *((int64_t) Q) *((int64_t) Q) )  ) %((uint64_t) Q));
};


inline int16_t Int2Montgo_16bit(int16_t a, int16_t One_square, int16_t Q, int16_t Q_montgo, int16_t One_montgo_inv) { 
    
    //return MulMontgo_16bit(a, One_square, Q, Q_montgo, One_montgo_inv);
    return a;
};

inline int16_t Montgo2Int_16bit(int16_t a, int16_t Q, int16_t Q_montgo, int16_t One_montgo_inv) { 
    //return Montgo_16bit(a, Q, Q_montgo, One_montgo_inv);
    return a;
};

*/



#ifdef __cplusplus
}
#endif



#endif
