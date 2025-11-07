#include <iostream>
#include <chrono>
#include <BinOperationCstyle.hpp>
#include <soAPI.hpp>

using namespace std;

int main() {
    using clock = std::chrono::steady_clock;
    using ms = std::chrono::duration<double, std::milli>;

    int msg_bit = 32;
    int m1 = 126;
    int m2 = -723;
    int cval = 7;

    int32_t *sk = FHE16_GenEval();

    // =============================
    // 암호화
    // =============================
    auto t_enc_start = clock::now();
    int32_t *CT1 = FHE16_ENCInt(m1, msg_bit);
    auto t_enc_end = clock::now();
    int32_t *CT2 = FHE16_ENCInt(m2, msg_bit);

    // =============================
    // ADD
    // =============================
    auto t_add_start = clock::now();
    int32_t *CT_ADD = FHE16_ADD(CT1, CT2);
    auto t_add_end = clock::now();

    // =============================
    // SUB
    // =============================
    auto t_sub_start = clock::now();
    int32_t *CT_SUB = FHE16_SUB(CT1, CT2);
    auto t_sub_end = clock::now();

    // =============================
    // ADD3
    // =============================
    auto t_add3_start = clock::now();
    int32_t *CT_ADD3 = FHE16_ADD3(CT1, CT2, CT1);
    auto t_add3_end = clock::now();

    // =============================
    // SMULL_CONSTANT
    // =============================
    auto t_smull_start = clock::now();
    int32_t *CT_MULC = FHE16_SMULL_CONSTANT(CT1, cval);
    auto t_smull_end = clock::now();

    // =============================
    // ADD_CONSTANT
    // =============================
    auto t_addc_start = clock::now();
    int32_t *CT_ADDC = FHE16_ADD_CONSTANT(CT1, cval);
    auto t_addc_end = clock::now();

    // =============================
    // RELU
    // =============================
    auto t_relu_start = clock::now();
    int32_t *CT_RELU = FHE16_RELU(CT2);
    auto t_relu_end = clock::now();

    // =============================
    // MAX
    // =============================
    auto t_max_start = clock::now();
    int32_t *CT_MAX = FHE16_MAX(CT1, CT2);
    auto t_max_end = clock::now();

    // =============================
    // 결과 출력
    // =============================
    cout.setf(ios::fixed);
    cout.precision(3);

    cout << "FHE16_ENCInt        : " << ms(t_enc_end - t_enc_start).count() << " ms" << endl;
    cout << "FHE16_ADD           : " << ms(t_add_end - t_add_start).count() << " ms" << endl;
    cout << "FHE16_SUB           : " << ms(t_sub_end - t_sub_start).count() << " ms" << endl;
    cout << "FHE16_ADD3          : " << ms(t_add3_end - t_add3_start).count() << " ms" << endl;
    cout << "FHE16_SMULL_CONSTANT: " << ms(t_smull_end - t_smull_start).count() << " ms" << endl;
    cout << "FHE16_ADD_CONSTANT  : " << ms(t_addc_end - t_addc_start).count() << " ms" << endl;
    cout << "FHE16_RELU          : " << ms(t_relu_end - t_relu_start).count() << " ms" << endl;
    cout << "FHE16_MAX           : " << ms(t_max_end - t_max_start).count() << " ms" << endl;


    return 0;
}

