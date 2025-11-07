#ifndef RLWE_H
#define RLWE_H


#include <random>
#include<BINFHE.hpp>
#include<lwe.hpp>

namespace EFHEs{


/********************** NOT FINISHED !!!!!!!!!!!!! 24.1.12 *****************/

class RLWECiphertext {
    private:
        const EFHE_Param_List * _Param = nullptr;
        uint32_t _N;
        uint32_t _Q;
        uint32_t *_a = nullptr;
        uint32_t *_b = nullptr;
        
        uint32_t *_a_NTT = nullptr;
        uint32_t *_b_NTT = nullptr;
        
        bool    _NTT_sync = false;
        bool    _is_NTT   = false;

    public:
        explicit RLWECiphertext(const EFHE_Param_List * Param): _Param(Param) {
            
            
            /*
            _N = Param->GetNRLWE1();
            //_Q = Param->GetQRLWE1();
            _a = (uint32_t *)malloc(sizeof(uint32_t)*_N);
            _b = (uint32_t *)malloc(sizeof(uint32_t)*_N);
            */

        }
        ~RLWECiphertext() {
            if (_a != nullptr) free(_a);
            if (_b != nullptr) free(_b);
            _Param = nullptr;
        }
        
        void MakeTrivialCiphertext() {
            std::random_device rd; 
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint32_t> dis_Q(_Q);
            std::uniform_int_distribution<uint32_t> dis_E(7);
        
            for (int ii = 0; ii < _N; ii++) {
                _a[ii] = dis_Q(gen);
                _b[ii] = dis_E(gen);
                if (_b[ii] >= 4) {
                    _b[ii] = _Q - _b[ii];
                }
            }
        }

        void Enc(uint32_t *sk, uint32_t *m, uint32_t Delta) {
            
            this->MakeTrivialCiphertext();
            int32_t tmp;
            for (int jj =0; jj <_N; jj++) {
                for (int ii = 0; ii <= jj; ii++) {
                    _b[jj] += (_a[ii] * sk[ii]);
                    _b[jj] %= _Q;
                }
                
                for (int ii = jj+1; ii < _N; ii++) {
                    tmp = ((int32_t)_b[jj]) - ((int32_t)(_a[ii] * sk[_N - ii]));
                    if (tmp < 0) {
                        tmp += _Q;
                    }
                    _b[jj] = tmp;
                }
                _b[jj] += m[jj]*Delta;
                _b[jj] %= _Q;
            }
        }
        void Phase(uint32_t *sk, uint32_t *m) {
            int32_t tmp;
            for (int jj = 0; jj < _N; jj++) {
                tmp = (int32_t) _b[jj];
                for (int ii = 0; ii <= jj; ii++) {
                    tmp  -= (int32_t)(_a[ii] * sk[ii]);
                    if (tmp < 0) {
                        tmp += _Q;
                    }
                }
                m[jj] = tmp;
                for (int ii = jj+1; ii < _N; ii++) {
                    m[jj] += _a[ii] * sk[_N - ii];
                }
            }
        }
    };
};


#endif // End header
