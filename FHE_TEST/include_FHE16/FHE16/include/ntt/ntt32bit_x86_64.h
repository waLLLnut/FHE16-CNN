#ifndef ntt32bit_x86_64_H
#define ntt32bit_x86_64_H

#ifdef __cplusplus
extern "C" {
#endif

//extern void asm_x86_64_ntt32bit(uint32_t *x, uint32_t *TB, uint32_t N, uint32_t Q, uint32_t Mont_mu);
//extern void asm_ntt_16bit_1024_x86_64_avx2(int32_t *x);


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
