#ifndef FHE16LWE_H
#define FHE16LWE_H


#include <random>
#include<BINFHE.hpp>


namespace EFHEs{
// predefine



class EFHE_Param_List;

class LWECiphertext {
    private:
        
        // Global Info
        const EFHE_BIN_Param_List * _Param = nullptr;
        

		// 16bit Q Info // Not ...
		NTTTable16* Q16bitINFO = nullptr;
        NTTTable32* Q32bitINFO = nullptr;
        NTTTable64* Q64bitINFO = nullptr;
       
        int64_t *_a = nullptr;
        int64_t _b;
		int64_t _Q;
    	int64_t *_Q_arr = nullptr;
		int64_t _scaling;
        int32_t _N;

		int32_t _type;
    	int		_Qnum = 1;
		bool	_QisPowerOfTwo = true;
		
      

    public:
        explicit LWECiphertext();
        explicit LWECiphertext(const EFHE_BIN_Param_List * Param);
        explicit LWECiphertext(const EFHE_BIN_Param_List * Param, int32_t * sk, int m);
        explicit LWECiphertext(int N, int Q, int scaling, int32_t * sk, int m);
        explicit LWECiphertext(int N, int Q, int scaling);
        
		explicit LWECiphertext(int N, int Q, int scaling, int32_t * sk, int m, bool IsPowerOfTwo);
    	explicit LWECiphertext(int N, int Q, int scaling, int m, bool IsPowerOfTwo);
        explicit LWECiphertext(int N, int Q, int scaling, bool IsPowerOfTwo);
      
		explicit LWECiphertext(LWECiphertext &&ref) noexcept;
        LWECiphertext(const LWECiphertext &ref);
         //LWECiphertext(LWECiphertext &&ref);
        
        ~LWECiphertext() noexcept;
        const LWECiphertext &operator=(const LWECiphertext &ref);
        const LWECiphertext &operator=(LWECiphertext &&ref) noexcept;
        // Operation
        friend std::ostream &operator<<(std::ostream &os, const LWECiphertext &ref);
        // Operation
        friend std::ostream &operator<<(std::ostream &os, const LWECiphertext &&ref);
        
		// Copy construct
	//	
		// Operation, This is what our gonna do
		LWECiphertext operator==(const LWECiphertext &ref);   
		LWECiphertext operator==(LWECiphertext &&ref);
		LWECiphertext operator!=(const LWECiphertext &ref);   
		LWECiphertext operator!=(LWECiphertext &&ref);
		void operator&=(const LWECiphertext &ref);   
		//void operator&=(LWECiphertext &&ref); no needs   
		LWECiphertext operator&(const LWECiphertext &ref);   
		LWECiphertext operator&(LWECiphertext &&ref);   

		void operator|=(const LWECiphertext &ref);   
		LWECiphertext operator|(const LWECiphertext &ref);   
		LWECiphertext operator|(LWECiphertext &&ref);   

		void operator+=(const LWECiphertext &ref);   
		LWECiphertext operator+(const LWECiphertext &ref);   
		LWECiphertext operator+(LWECiphertext &&ref);   

		LWECiphertext operator!();   
		LWECiphertext operator~();   




		void NOT() noexcept;
    	

		//GF12289 operator==(const GF12289 &&ref);



        // Functionaliy
        void MakeTrivialCiphertext();
        void ClearElement();

        void Enc(int32_t *sk, int m, int32_t Delta);
        void Enc(int32_t *sk, int m);

        void Phase(int32_t *sk, int32_t &m);       
        void Dec(int32_t *sk, int32_t &m);
        void Dec(int32_t *sk, int32_t &m, double &e);
        int32_t Dec(int32_t *sk);


        int32_t GetAat32bit(int idx) const;
        int32_t GetB() const { return (int32_t)_b;};
        void SetAat32bit(int32_t idx, int32_t val);
        void SetB(int32_t val) { _b=val;};
        int32_t GetQ() const; 
        uint32_t GetN() const; 
        uint32_t GetScaling() const; 
        const EFHE_BIN_Param_List * GetParam() const {return _Param;};
 
        void AddB(int32_t val);
        void AddAat(int idx, int32_t val);
		void SubB(int32_t val);
        void SubAat(int idx, int32_t val);


		const int32_t * Get32bitAB_readable_arr() const;
		int32_t * Get32bitAB_writable_arr();
		void Replacing32bitAB_arr(int32_t *ARR);
		void SetBFromArray();
		void SetArrayFromB();
	

	};

};


#endif // End header
