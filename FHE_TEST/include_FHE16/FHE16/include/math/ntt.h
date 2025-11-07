#ifndef NTT_CH_EF
#define NTT_CH_EF

#ifndef __cplusplus
#include<stdint.h>
#endif


#include<CMAKEPARAM.h>

#if CPUTYPE == 1 // IF LINUX
	#include<stdint.h>
#endif




#ifdef __cplusplus
extern "C" {
#endif

#include<param.h>



/* precompute roots of unity and its powers */
/*
int invers(int x,int y);
uint32_t reverse(uint32_t v);
int spmd(int x,int n,int m);
int sqrmp(int x,int m);
*/

//uint32_t nres(uint32_t x);

//void init_ntt_32bit(uint32_t *roots, uint32_t *iroots);

// Cooley-Tukey NTT 
//void ntt_32bit(const int32_t *roots, int32_t *x, int step, int N, int Q, int montgo, int One_inv);
// Gentleman-Sande INTT 

//void intt2_32bit(const int32_t *iroots, int32_t *x, int32_t step,  int32_t inv, int32_t invpr,  int32_t N, int32_t Q, int32_t montgo, int32_t One_inv);

void ntt_32bit(const int32_t	*QNTTINFO,	int32_t *x, const int32_t *QINFO);

void ntt_to_32bit(const int32_t	*QNTTINFO,	int32_t *x,int32_t *y, const int32_t *QINFO);


void intt_32bit(const int32_t		*QINTTINFO, int32_t *x,	const int32_t *QINFO);
void intt_to_32bit(const int32_t	*QINTTINFO, int32_t *x,	int32_t * y, const int32_t *QINFO);




/*
// 16bit
void ntt_16bit( int16_t *roots, int16_t *x, int16_t less_step, int16_t N, int16_t Q, int16_t montgo, int16_t One_inv);
void intt_16bit(const int16_t *iroots, int16_t *x, int16_t step,  int16_t inv, int16_t invpr,  int16_t N, int16_t Q, int16_t montgo, int16_t One_inv);
*/
 
void ntt_16bit3(const int16_t *QNTTINFO, int16_t *x,    const int32_t *QINFO);
void ntt_to_16bit3(const int16_t *QNTTINFO, int16_t *x, int16_t*y,  const int32_t *QINFO);



void intt_16bit3(const int16_t *QINTTINFO, int16_t *x,  const int32_t *QINFO);
void intt_to_16bit3(const int16_t *QINTTINFO, int16_t *x,  int16_t *y,  const int32_t *QINFO);


/// New Incomplete

void ntt_to_16bit4_unperm(const int16_t *QNTTINFO, int16_t *x, int16_t*y,  const int32_t *QINFO);

void intt_to_16bit4_unperm(const int16_t *QNTTINFO, int16_t *x, int16_t*y,  const int32_t *QINFO);





#ifdef __cplusplus
}
#endif






#endif
