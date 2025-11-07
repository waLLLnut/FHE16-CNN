#ifndef POLY_CLASS_HPP_EF
#define POLY_CLASS_HPP_EF


#include<CMAKEPARAM.h>
#include <param.h>
#include <structures.h>
#include <iostream>
#include <BINFHE.hpp>
#include <ntttable.hpp>


/*
struct NTTTable16;
struct NTTTable32;
struct NTTTable64;
*/

using namespace std;

namespace EFHEs{

    /*
    // DepthChecking Funs
    #if INCOMPLETE_DEPTH_BIN == 0
        #define PointMulMont(a, b, c)  Mul0MontSchool(a,b,c)
        #define PointMulBarr(a, b, c)  Mul0BarrSchool(a,b,c)
    #elif  INCOMPLETE_DEPTH_BIN == 1
        #define PointMulMont(a, b, c)  Mul1MontSchool(a,b,c)
        #define PointMulBarr(a, b, c)  Mul1BarrSchool(a,b,c)
    #elif  INCOMPLETE_DEPTH_BIN == 2
        #define PointMulMont(a, b, c)  Mul2MontSchool(a,b,c)
        #define PointMulBarr(a, b, c)  Mul2BarrSchool(a,b,c)
    #elif  INCOMPLETE_DEPTH_BIN == 3
        #define PointMulMont(a, b, c)  Mul3MontSchool(a,b,c)
        #define PointMulBarr(a, b, c)  Mul3BarrSchool(a,b,c)
    #elif  INCOMPLETE_DEPTH_BIN == 4
        #define PointMulMont(a, b, c)  Mul4MontSchool(a,b,c)
        #define PointMulBarr(a, b, c)  Mul4BarrSchool(a,b,c)
    #elif  INCOMPLETE_DEPTH_BIN == 5
        #define PointMulMont(a, b, c)  Mul5MontSchool(a,b,c)
        #define PointMulBarr(a, b, c)  Mul5BarrSchool(a,b,c)
    #endif
    */

    // Predefined in poly_class
    class EFHE_Param_List;
    //class NTTTable32;
    
    class Poly32 {
        private:
            //const EFHE_Param_List *_Param = nullptr;
            const NTTTable32 * _Param_NTT = nullptr;
            int32_t *   _poly       = nullptr;
           
            bool        _IsMont     = false;
            bool        _IsNTT      = false;
          
        public:
            Poly32();
            explicit Poly32(const NTTTable32 * _Param_NTT );
            explicit Poly32(const NTTTable32 * _Param_NTT, int32_t *poly, bool isNTT, bool isMont);
            Poly32(Poly32 &&ref) noexcept;
            Poly32(const Poly32 &ref);
           

 
            int32_t * GetPoly();
            //int32_t * GetNTTPoly();
            const NTTTable32 * GetNTTTable() const;
            int32_t GetN() const;
            int32_t GetQ(int idx) const;
            int32_t GetQnum() const;
            

			//int32_t SlowModQ(int32_t x, uint32_t Q);
            //int32_t SlowMulQ(int32_t x, int32_t y, uint32_t Q);
            
            void PRINT() const; 
           
			void Automorphism( int idx);
			void Reduction();
            
            
            // Operation
            friend std::ostream &operator<<(std::ostream &os, const Poly32 &ref);
            // Operation
            friend std::ostream &operator<<(std::ostream &os, const Poly32 &&ref); 



            // Copy construct
            const Poly32 &operator=(const Poly32 &ref);
            // Rvalue
            const Poly32 &operator=(Poly32 &&ref) noexcept;

            // Addition
            void operator+=(const Poly32 &ref);
            void operator+=(int32_t x);
            // Operation
            Poly32 operator+(const Poly32 &ref);
            Poly32 operator+(Poly32 &&ref);
            Poly32 operator+(int32_t x);


            // Substraction
            void operator-=(const Poly32 &ref); 
            void operator-=(int32_t x);

            // Operation
            Poly32 operator-(Poly32 &&ref);
            Poly32 operator-(const Poly32 &ref);
            Poly32 operator-(int32_t x);

            // Multiplicatio
            void operator*=(const Poly32 &ref); 
            void operator*=(int32_t x);           
            // Operation
            Poly32 operator*(Poly32 &&ref);
            Poly32 operator*(const Poly32 &ref);
            Poly32 operator*(int32_t x);

            //Get Element
            int32_t operator[](int32_t idx);

            ~Poly32() noexcept;

            void ClearElement();
            void GenUniformPoly(); 
            void GenTernaryPoly(); 
            void GenDGPoly( double sigma); 
          
			void GenUniformPolyNTT(); 
            void Decompose(int base, Poly32 * res, int len) const;
            void AddAt(int idx, int32_t val);
			void AddBigAt(int idx, int64_t val);
            void SubAt(int idx, int32_t val);
            void SetAt(int idx, int32_t val);
            void SetAtCRT(int idx1, int idx2, int32_t val);
            void SetAtll(int idx, int32_t val);
            void RotCanonical(int idx);                   
            void ReduceQ(int next_q);

            // NTT
            void NTT();
            void ForceNTT();
            void INTT();
            void ForceINTT();
		   
            void ForceMont();
            void ForceIMont();
            void Mont();
            void IMont();
			
			void MUL_NTT_MONT(int32_t *res, int32_t *x, int32_t *y, const NTTTable32 *Param_x, const NTTTable32 *Param_y, int N, int q_num); 


            // Auto
            void Aut(int idx);
		};


    class Poly16 {
        private:
            const NTTTable16 * _Param_NTT = nullptr;
			// Mul function pointers
			


			int16_t *   _poly       = nullptr;
           
            bool        _IsMont     = false;
            bool        _IsNTT      = false;
          
        public:
            Poly16();
            explicit Poly16(const NTTTable16 * _Param_NTT );
            explicit Poly16(const NTTTable16 * _Param_NTT, int16_t *poly, bool isNTT, bool isMont);
            Poly16(Poly16 &&ref) noexcept;
            Poly16(const Poly16 &ref);
            
            int16_t * GetPoly();
            int16_t * GetNTTPoly();
            const NTTTable16 * GetNTTTable() const;
            
			int16_t GetN() const;
            int16_t GetQ(int idx) const ;
            int16_t GetQnum() const;
           	
			void Automorphism( int idx);


            void PRINT() const; 
           
            
            // Operation
            friend std::ostream &operator<<(std::ostream &os, const Poly16 &ref);
            // Operation
            friend std::ostream &operator<<(std::ostream &os, const Poly16 &&ref); 



            // Copy construct
            const Poly16 &operator=(const Poly16 &ref);
            // Rvalue
            const Poly16 &operator=(Poly16 &&ref) noexcept;

            // Addition
            void operator+=(const Poly16 &ref);
			void operator+=(int16_t x);
			Poly16 operator+(const Poly16 &ref);
            Poly16 operator+(Poly16 &&ref); 
			Poly16 operator+(int16_t x);
        
            // Subtract
            void operator-=(const Poly16 &ref); 
            void operator-=(int16_t x);
			Poly16 operator-(const Poly16 &ref);
            Poly16 operator-(Poly16 &&ref);
            Poly16 operator-(int16_t x);

             // Multiplicatio
            void	operator*=(const Poly16 &ref); 
            void	operator*=(int16_t x);           
            // Operation
            Poly16 operator*(const Poly16 &ref);
            Poly16 operator*(Poly16 &&ref);
            Poly16 operator*(int16_t x);


            //Get Element
            int16_t operator[](int idx);

            ~Poly16() ;

            void ClearElement();
            void GenUniformPoly(); 
            void GenTernaryPoly(); 
            void GenDGPoly( double sigma); 
            void GenUniformPolyNTT(); 
            void Decompose(int base, Poly16 * res, int len) const;
            void AddAt(int idx, int16_t val);
            void AddAt(int idx, int64_t val);
            void AddBigAt(int idx, int64_t val);
            void SubAt(int idx, int16_t val);
            void SubAt(int idx, int64_t val);
            void SetAt(int idx, int64_t val);
            void SetAtAll(int idx, int16_t val);
            void SetAtCRT(int idx1, int idx2, int16_t val);
            void RotCanonical(int idx);                   
            void ReduceQ(int next_q);
			void Reduction();
            // NTT
            void NTT();
            void ForceNTT();
            void INTT();
            void ForceINTT();
		   
			void NTTTweak();
            void INTTTweak();
               
            
            void ForceMont();
            void ForceIMont();
            void Mont();
            void IMont();
			void CenteringCoeff();
			
			void MUL_NTT_MONT(int16_t *res, int16_t *x, int16_t *y, const NTTTable16 *Param_x, const NTTTable16 *Param_y, int N, int q_num); 


		};

    class Poly64 {
        private:
            //const EFHE_Param_List *_Param = nullptr;
            const NTTTable64 * _Param_NTT = nullptr;
            int64_t *   _poly       = nullptr;
           
            bool        _IsMont     = false;
            bool        _IsNTT      = false;
          
        public:
            Poly64();
            explicit Poly64(const NTTTable64 * _Param_NTT );
            explicit Poly64(const NTTTable64 * _Param_NTT, int64_t *poly, bool isNTT, bool isMont);
            Poly64(Poly64 &&ref);
            Poly64(const Poly64 &ref);
            
            int64_t * GetPoly();
            int64_t * GetNTTPoly();
            const NTTTable64 * GetNTTTable() const;
            //int64_t GetN() const { return _Param_NTT->GetN();};
            //int64_t GetQ() const { return _Param_NTT->GetQ();};
            //int64_t GetQnum() const { return _Param_NTT->GetQnum();};
           

			void Automorphism( int idx);
			void Reduction();
            

            void PRINT() const; 
          
            // Operation
            friend std::ostream &operator<<(std::ostream &os, const Poly64 &ref);
            // Operation
            friend std::ostream &operator<<(std::ostream &os, const Poly64 &&ref); 



            // Copy construct
            const Poly64 &operator=(const Poly64 &ref);
            // Rvalue
            const Poly64 &operator=(Poly64 &&ref);

            // Addition
            const Poly64 &operator+=(const Poly64 &ref);
            const Poly64 &operator+=(int64_t x);
            // Operation
            Poly64 operator+(Poly64 &ref);
            Poly64 operator+(const Poly64 &&ref);
            Poly64 operator+(int64_t x);


            // Substraction
            const Poly64 &operator-=(const Poly64 &ref); 
            const Poly64 &operator-=(int64_t x);

            // Operation
            Poly64 operator-(Poly64 &ref);
            Poly64 operator-(const Poly64 &&ref);
            Poly64 operator-(int64_t x);

            // Multiplicatio
            const Poly64 &operator*=(const Poly64 &ref); 
            const Poly64 &operator*=(int64_t x);           
            // Operation
            Poly64 operator*(Poly64 &ref);
            Poly64 operator*(int64_t x);

            //Get Element
            int64_t operator[](int idx);

            ~Poly64() ;

            void ClearElement();
            void GenUniformPoly(); 
            void GenTernaryPoly(); 
            void GenDGPoly( double sigma); 
          
			void GenUniformPolyNTT(); 
            void Decompose(int base, Poly64 * res, int len) const;
            void AddAt(int idx, int64_t val);
            void SubAt(int idx, int64_t val);
            void SetAt(int idx, int64_t val);
            void SetAtCRT(int idx1, int idx2, int64_t val);
			void SetAtAll(int idx, int64_t val);
          
            void RotCanonical(int idx);                   
            void ReduceQ(int next_q);

            // NTT
            void NTT();
            void ForceNTT();
            void INTT();
            void ForceINTT();
		       
            void ForceMont();
            void ForceIMont();
            void Mont();
            void IMont();



		};


};




#endif // HEADER END
