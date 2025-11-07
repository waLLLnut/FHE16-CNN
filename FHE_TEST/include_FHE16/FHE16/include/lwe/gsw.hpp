#ifndef RLWE_H
#define RLWE_H


#include <random>
#include<LWECryptosystem.hpp>
#include<rlwe.hpp>


namespace EFHEs{
    
    class GSWCiphertext {
    private:
        const EFHE_Param_List * _Param = nullptr;
        uint32_t _N;
        uint32_t _Q;
        uint32_t _l;
        uint32_t _Delta;

        RLWECiphertext *_RLWE = nullptr;
       
        bool    _NTT_sync = false;
        bool    _is_NTT   = false;

    public:
        explicit GSWCiphertext(const EFHE_Param_List * Param, int l): _Param(Param), _l(l) {
            _N = Param->GetN();
            _Q = Param->GetQ();
            _a = (uint32_t *)malloc(sizeof(uint32_t)*_N);
            _b = (uint32_t *)malloc(sizeof(uint32_t)*_N);
        }
        ~GSWCiphertext() {
            free(_a);
            free(_b);
            _Param = nullptr;
        }
        
        void MakeTrivialCiphertext() {
            std::random_device rd; 
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint32_t> dis_Q(_Q);
            std::uniform_int_distribution<uint32_t> dis_E(7);
        
            for (int ii = ; ii < _N; ii++) {
                _a[ii] = dis_Q(gen);
                _b = dis_E(gen);
                if (_b >= 4) {
                    _b = - b;
                }

            }
                 
        }
        void Enc(uint32_t *sk, uint32_t *m, uint32_t Delta) {
            
            this->MakeTrivialCiphertext();
            int32_t tmp;
            for (int jj =0; jj <_N; jj++) {
                for (int ii = 0; ii <= jj; ii++) {
                    _b[jj] += (_a[ii] * sk[ii]);
                    _b %= _Q;
                }
                
                for (int ii = jj+1; ii < _N; ii++) {
                    tmp = ((int32_t)_b[jj]) - ((int32_t)(_a[ii] * sk[_N - ii]));
                    if (tmp < 0) {
                        tmp += _Q;
                    }
                    _b[jj] = tmp;
                }
                b[jj] += m[jj]*Delta;
                b[jj] %= _Q;
            }
        }
        void Phase(uint32_t *sk, uint32_t *m) {
            int32_t tmp;
            for (int jj = 0; jj < _N; jj++) {
                tmp = (int32_t) _b;
                for (int ii = 0; ii <= jj; ii++) {
                    tmp  -= (int32_t)(_a[ii] * sk[ii]);
                    if (tmp < 0) {
                        tmp += Q;
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
