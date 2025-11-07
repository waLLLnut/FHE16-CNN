#ifndef GADGET_EF
#define GADGET_EF

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


// With 32 bit version
void DecompIn2Out216to32(int16_t * from, int16_t *to, const int32_t * INFO, const int32_t *GADGET, int dim, int rm_len, int base_len, int vec_len);

void DecompIn2Out316to32(int16_t * from, int16_t *to, const int32_t * INFO, const int32_t *GADGET, int dim, int rm_len, int base_len, int vec_len);

void DecompIn2Out116to32(int16_t * from, int16_t *to, const int32_t * INFO, const int32_t *GADGET, int dim, int rm_len, int base_len, int vec_len);






// NO CRT version   
void inline DecompTwoPowRemoveOne(const int32_t * from, int32_t * to, int32_t dim, int32_t poly_len,  int32_t base, int32_t gl_len, int32_t rm, int32_t Q) {
        int32_t tmp2;
        int32_t modular = (int32_t)( ((int)1<< base)  - 1);
        int32_t shift_val = base;
		int32_t Q_over_2 = Q >> 1;

        int32_t idx1 = 0;
        int32_t idx2 = 0;
        int32_t idx3 = 0;
		
		int rm_shift = 32 - rm;
        int base_shift = 32 - shift_val;

		for (int idx_num_poly = 0; idx_num_poly < poly_len; idx_num_poly++)  {
			for (int el = 0; el < dim; el++) {
				int32_t aa = from[idx1 + el];
    			//aa = aa - ( ((~(aa - Q_over_2)) >> 31) & Q);


				// Signed
				/*
				if (aa >- Q_over_2) { aa -= Q;}
				idx2 = 0;
				shift_val = base;
				int32_t rms = (aa << (rm_shift)) >> (rm_shift);
				aa -= rms;
				aa >>= rm;
				
				for (int idx_gl = 0; idx_gl < gl_len; idx_gl++){
                   // Check
                    to[el + idx2 + idx3] =  (aa << base_shift)>> base_shift;
					aa -= to[el + idx2 + idx3];
					
					aa >>= shift_val;
					idx2 += dim;
                }*/
				// Unsigned, noremove
				idx2 = 0;
				for (int idx_gl = 0; idx_gl < gl_len; idx_gl++){
                   // Check
                    to[el + idx2 + idx3] =  aa & modular;
					aa >>= base;
					idx2 += dim;
                }
	
			}
            
			idx1 += dim;
            idx3 = idx1 * gl_len;
        }
			
    }

 // NO CRT version   
void inline DecompTwoPowRemoveOne_16bit(const int16_t * from, int16_t * to, int32_t dim, int32_t poly_len,  int32_t base, int32_t gl_len) {
        int16_t tmp2;
        int16_t modular = (int32_t)( ((int)1<< base)  - 1);
        int32_t shift_val;

        int32_t idx1 = 0;
        int32_t idx2 = 0;
        int32_t idx3 = 0;
        for (int idx_num_poly = 0; idx_num_poly < poly_len; idx_num_poly++)  {
            shift_val = base;
            idx2 = 0;
            for (int idx_gl = 0; idx_gl < gl_len; idx_gl++){
                for (int el = 0; el < dim; el++) {
                    // Check
                    tmp2 = (from[idx1 + el] >> (shift_val));
                    to[el + idx2 + idx3] = (int32_t)(tmp2 & modular);
                }
                // index recalculation
                shift_val +=base;
                idx2 += dim;
            }
            idx1 += dim;
            idx3 = idx1 * gl_len;
        }
    }

  


void inline SignedDecompTwoPowRemoveOneCRT_32bit(const int32_t * from, int32_t * to, int32_t dim, int32_t poly_len,  int32_t base, int32_t gl_len, int rm, int32_t *Q, int32_t q_len, int64_t * crt_basis) {
        int64_t tmp;
        int32_t tmp2;
        int32_t tmp3;
        
        int32_t modular = (int32_t)( ((int)1<< base)  - 1);
        int32_t shift_val;
        int64_t Q_tot = 1;
        for (int jj = 0; jj < q_len; jj++) {
            Q_tot *= Q[jj];
        }



        // Signed
        int32_t modular2 = (int32_t)( ((int)1<< (base-1))  - 1);
        int32_t shift_val2 = base - 1;

        int32_t LeftShift       = 32 - base;
        int     LeftShiftFirst  = 32 - rm;

        int32_t Q_over_2 = (int32_t) floor((double) Q_tot / 2.0);

        int32_t idx1 = 0;
        int32_t idx2 = 0;
        int32_t idx3 = 0;
        int32_t idx4 = 0;
        for (int idx_num_poly = 0; idx_num_poly < poly_len; idx_num_poly++)  {
            shift_val = base;
            idx2 = 0;
          

            for (int el = 0; el < dim; el++) {
                // My el, reconstruct
                tmp = 0;
                for (int jj = 0; jj < q_len; jj++) {
                    tmp += crt_basis[jj] * ((int64_t) from[idx1 + el + dim*jj]);
                }
                tmp2  = (int32_t)(tmp % Q_tot);
                //tmp2 = from[idx1 + el];
                
                // Conversion -q/2 ~ q/2
                tmp2 = tmp2 - ( ((~(tmp2 - Q_over_2)) >> 31) & Q_tot);
                
                // Remove one variable
                tmp3 = (tmp2 << LeftShiftFirst ) >> LeftShiftFirst;
                tmp2 = (tmp2 - tmp3) >> rm;
                //tmp2 >>= base;
                idx2 = 0;
          
                for (int idx_gl = 0; idx_gl < gl_len; idx_gl++){
                    tmp3 = (tmp2 << LeftShift ) >> LeftShift;
                    tmp2 = (tmp2 - tmp3) >> base;

                    for (int tt = 0; tt < q_len; tt++) {
                        to[el + idx2 +  idx3 + tt*dim] = tmp3;
                    }
                    idx2 += dim * q_len;
                }
            }
            idx1 += (dim)*q_len;
            idx3 = idx1 * gl_len;
        }
    }



void inline SignedDecompTwoPowRemoveOne_32bit(const int32_t * from, int32_t * to, int32_t dim, int32_t poly_len,  int32_t base, int32_t gl_len, int rm, int32_t Q) {
        int64_t tmp;
        int32_t tmp2;
        int32_t tmp3;
        
        int32_t modular = (int32_t)( ((int)1<< base)  - 1);
        int32_t shift_val;
      

        // Signed
        int32_t modular2 = (int32_t)( ((int)1<< (base-1))  - 1);
        int32_t shift_val2 = base - 1;

        int32_t LeftShift       = 32 - base;
        int     LeftShiftFirst  = 32 - rm;

        int32_t Q_over_2 = (int32_t) floor((double) Q / 2.0);

        int32_t idx1 = 0;
        int32_t idx2 = 0;
        int32_t idx3 = 0;
        int32_t idx4 = 0;
        for (int idx_num_poly = 0; idx_num_poly < poly_len; idx_num_poly++)  {
            shift_val = base;
            idx2 = 0;
          
            for (int el = 0; el < dim; el++) {
                // My el, reconstruct
                tmp2 = from[idx1 + el];
                
                // Conversion -q/2 ~ q/2
                tmp2 = tmp2 - ( ((~(tmp2 - Q_over_2)) >> 31) & Q);
                
                // Remove one variable
                tmp3 = (tmp2 << LeftShiftFirst ) >> LeftShiftFirst;
                tmp2 = (tmp2 - tmp3) >> rm;
                //tmp2 >>= base;
                idx2 = 0;
          
                for (int idx_gl = 0; idx_gl < gl_len; idx_gl++){
                    tmp3 = (tmp2 << LeftShift ) >> LeftShift;
                    tmp2 = (tmp2 - tmp3) >> base;

                    to[el + idx2 +  idx3] = tmp3;
                    idx2 += dim;
                }
            }
            idx1 += (dim);
            idx3 = idx1 * gl_len;
        }
    }


/*************************** 16bit **************************************/

void inline SignedDecompTwoPowRemoveOneCRT_16bit(const int16_t * from, int16_t * to, int dim, int poly_len,  int base, int gl_len, int rm, int16_t *Q, int q_len, int64_t * crt_basis) {
        int64_t tmp;
        int32_t tmp2;
        int32_t tmp3;
        
        int32_t modular = (int32_t)( ((int)1<< base)  - 1);
        int32_t shift_val;
        int64_t Q_tot = 1;
        for (int jj = 0; jj < q_len; jj++) {
            Q_tot *= Q[jj];
        }



        // Signed
        int32_t modular2 = (int32_t)( ((int)1<< (base-1))  - 1);
        int32_t shift_val2 = base - 1;

        int32_t LeftShift       = 32 - base;
        int     LeftShiftFirst  = 32 - rm;

        int32_t Q_over_2 = (int32_t) floor((double) Q_tot / 2.0);

        int32_t idx1 = 0;
        int32_t idx2 = 0;
        int32_t idx3 = 0;
        int32_t idx4 = 0;
        for (int idx_num_poly = 0; idx_num_poly < poly_len; idx_num_poly++)  {
            shift_val = base;
            idx2 = 0;
          

            for (int el = 0; el < dim; el++) {
                // My el, reconstruct
                tmp = 0;
                for (int jj = 0; jj < q_len; jj++) {
                    tmp += crt_basis[jj] * ((int64_t) from[idx1 + el + dim*jj]);
                }
                tmp2  = (int32_t)(tmp % Q_tot);
                //tmp2 = from[idx1 + el];
                
                // Conversion -q/2 ~ q/2
                tmp2 = tmp2 - ( ((~(tmp2 - Q_over_2)) >> 31) & Q_tot);
                
                // Remove one variable
                tmp3 = (tmp2 << LeftShiftFirst ) >> LeftShiftFirst;
                tmp2 = (tmp2 - tmp3) >> rm;
                //tmp2 >>= base;
                idx2 = 0;
          
                for (int idx_gl = 0; idx_gl < gl_len; idx_gl++){
                    tmp3 = (tmp2 << LeftShift ) >> LeftShift;
                    tmp2 = (tmp2 - tmp3) >> base;
                    if (tmp3 >= 0) {
                        for (int tt = 0; tt < q_len; tt++) {
                            to[el + idx2 +  idx3 + tt*dim] = tmp3;
                        }
                    } else {
                        for (int tt = 0; tt < q_len; tt++) {
                            to[el + idx2 +  idx3 + tt*dim] = Q[tt] + tmp3;
                        }
                    
                    }
                    idx2 += dim * q_len;
                }
            }
            idx1 += (dim)*q_len;
            idx3 = idx1 * gl_len;
        }
    }



void inline SignedDecompTwoPowRemoveOne_16bit(const int32_t * from, int16_t * to, int dim, int poly_len,  int base, int gl_len, int rm, int16_t Q) {
        int64_t tmp;
        int32_t tmp2;
        int32_t tmp3;
        
        int32_t modular = (int32_t)( ((int)1<< base)  - 1);
        int32_t shift_val;
      

        // Signed
        int32_t modular2 = (int32_t)( ((int)1<< (base-1))  - 1);
        int32_t shift_val2 = base - 1;

        int32_t LeftShift       = 32 - base;
        int     LeftShiftFirst  = 32 - rm;

        int32_t Q_over_2 = (int32_t) floor((double) Q / 2.0);

        int32_t idx1 = 0;
        int32_t idx2 = 0;
        int32_t idx3 = 0;
        int32_t idx4 = 0;
        for (int idx_num_poly = 0; idx_num_poly < poly_len; idx_num_poly++)  {
            shift_val = base;
            idx2 = 0;
          
            for (int el = 0; el < dim; el++) {
                // My el, reconstruct
                tmp2 = from[idx1 + el];
                
                // Conversion -q/2 ~ q/2
                tmp2 = tmp2 - ( ((~(tmp2 - Q_over_2)) >> 31) & Q);
                
                // Remove one variable
                tmp3 = (tmp2 << LeftShiftFirst ) >> LeftShiftFirst;
                tmp2 = (tmp2 - tmp3) >> rm;
                //tmp2 >>= base;
                idx2 = 0;
          
                for (int idx_gl = 0; idx_gl < gl_len; idx_gl++){
                    tmp3 = (tmp2 << LeftShift ) >> LeftShift;
                    tmp2 = (tmp2 - tmp3) >> base;

                    to[el + idx2 +  idx3] = tmp3;
                    idx2 += dim;
                }
            }
            idx1 += (dim);
            idx3 = idx1 * gl_len;
        }
    }




void DecompTwoPowRemoveCRT2_Base8_Remove8_Len3_16bit(	int16_t * from, int16_t * to, const int32_t * INFO, const int16_t *GADGET, int dim, int Ks);
void DecompTwoPowRemoveCRT2_Base9_Remove10_Len2_16bit(	int16_t * from, int16_t * to, const int32_t * INFO, const int16_t *GADGET, int dim, int Ks);
void DecompTwoPowRemoveCRT2_Base9_Remove11_Len2_16bit(	int16_t * from, int16_t * to, const int32_t * INFO, const int16_t *GADGET, int dim, int Ks);
void DecompTwoPowRemoveCRT2_Base11_Remove17_Len1_16bit(	int16_t * from, int16_t * to, const int32_t * INFO, const int16_t *GADGET, int dim, int Ks);

void DecompFloatTwoPowRemoveCRT2_Base8_Remove8_Len3_16bit(	int16_t * from, int16_t * to, const int32_t * INFO, const int16_t *GADGET, int dim);
void DecompFloatTwoPowRemoveCRT2_Base9_Remove10_Len2_16bit(	int16_t * from, int16_t * to, const int32_t * INFO, const int16_t *GADGET, int dim, int Ks);
void DecompFloatTwoPowRemoveCRT2_Base11_Remove17_Len1_16bit(	int16_t * from, int16_t * to, const int32_t * INFO, const int16_t *GADGET, int dim);





void DecompTwoPowRemoveCRT2_num2q_16bit_asm(	int16_t * from, int16_t * to, const int32_t * INFO, const int16_t *GADGET, int dim, int Ks);





void DecompTwoPowRemoveCRT2_num2q_32bit(	int32_t * from, int32_t * to, const int32_t * INFO, const int32_t *GADGET, int dim);




void inline DecompTwoPowRemoveOneCRT1_32bit(int32_t * from, int32_t * to, int32_t dim, int32_t poly_len,  int32_t base, int32_t rm_base, int32_t gl_len) {
        int32_t tmp2;
        int32_t modular = (int32_t)( ((int)1<< base)  - 1);
        int32_t shift_val;

        int32_t idx1 = 0;
        int32_t idx2 = 0;
        int32_t idx3 = 0;
        for (int idx_num_poly = 0; idx_num_poly < poly_len; idx_num_poly++)  {
            shift_val = rm_base;
            idx2 = 0;
            for (int idx_gl = 0; idx_gl < gl_len; idx_gl++){
				for (int el = 0; el < dim; el++) {
					tmp2 = (from[idx1 + el] >> (shift_val));
					to[el + idx2 + idx3] = (int32_t)(tmp2 & modular);
				}
				
				// index recalculation
                shift_val +=base;
                idx2 += dim;
            }
            idx1 += dim;
            idx3 = idx1 * gl_len;
        }
    }

// Ver2
void inline DecompTwoPow_base8_rm10_len2_32bit(int32_t * from, int32_t * to, int32_t dim, int32_t poly_len,  int32_t base, int32_t gl_len) {
        int32_t tmp2;
        int32_t modular = (int32_t)( ((int)1<< base)  - 1);
        int32_t shift_val;

        int32_t idx1 = 0;
        int32_t idx2 = 0;
        int32_t idx3 = 0;
        for (int idx_num_poly = 0; idx_num_poly < poly_len; idx_num_poly++)  {
            shift_val = base;
            idx2 = 0;
            for (int idx_gl = 0; idx_gl < gl_len; idx_gl++){
                for (int el = 0; el < dim; el++) {
                    // Check
                    tmp2 = (from[idx1 + el] >> (shift_val));
                    to[el + idx2 + idx3] = (int32_t)(tmp2 & modular);
                }
                // index recalculation
                shift_val +=base;
                idx2 += dim;
            }
            idx1 += dim;
            idx3 = idx1 * gl_len;
        }
    }




#ifdef __cplusplus
}
#endif



#endif
