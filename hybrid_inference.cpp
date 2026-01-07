/**
 * FHE16-CNN Hybrid Inference
 *
 * n개 레이어 중 마지막 a개만 FHE로 실행
 * --fhe-layers=N 옵션으로 조절 (기본값: 5 = 전체 FHE)
 *
 * 레이어 구조:
 *   Layer 1: Encryption (입력 암호화)
 *   Layer 2: Conv3x3 + Bias
 *   Layer 3: ReLU
 *   Layer 4: SumPool
 *   Layer 5: FC (Fully Connected)
 *
 * 예시:
 *   --fhe-layers=5 : 전체 FHE (기본)
 *   --fhe-layers=2 : FC만 FHE (Pool 출력에서 암호화)
 *   --fhe-layers=1 : Decrypt만 FHE (FC 출력에서 암호화)
 *   --fhe-layers=0 : 평문 연산만 (테스트용)
 *
 * Copyright (c) 2025 waLLLnut
 */

#include <iostream>
#include <vector>
#include <array>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <climits>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <csignal>
#include <BinOperationCstyle.hpp>
#include <soAPI.hpp>

using namespace std;

using Cipher = int32_t*;
static constexpr int MSG_BIT = 32;

// --- Parameter settings ---
static constexpr int IN_H = 28, IN_W = 28;
static constexpr int K = 3, KH = 3, KW = 3, STR = 3;
static constexpr int OUT_H = 9, OUT_W = 9;
static constexpr int PH = 3, PW = 3;
static constexpr int POOL_H = 3, POOL_W = 3;
static constexpr int FLAT = K * POOL_H * POOL_W;
static constexpr int NUM_CLASSES = 10;

// --- Global FHE layers setting ---
static int g_fhe_layers = 5; // 기본값: 전체 FHE

// ---------- Optimization functions ----------
Cipher get_zero() {
    static Cipher Z = nullptr;
    if (!Z) Z = FHE16_ENCInt(0, MSG_BIT);
    return Z;
}

inline Cipher mul_scalar_opt(Cipher c, int32_t w) {
    if (w == 0) return get_zero();
    if (w == 1) return c;
    if (w == -1) return FHE16_SUB(get_zero(), c);
    return FHE16_SMULL_CONSTANT(c, w);
}

inline Cipher add3(Cipher a, Cipher b, Cipher c) { return FHE16_ADD3(a, b, c); }

// ---------- MNIST/Weights loaders ----------
vector<int32_t> read_csv_line_ints(const string& path) {
    ifstream f(path);
    if (!f) { cerr << "[ERR] open " << path << "\n"; abort(); }
    string line; getline(f, line);
    vector<int32_t> vals; string tok; stringstream ss(line);
    while (getline(ss, tok, ',')) {
        size_t s = 0; while (s < tok.size() && isspace((unsigned char)tok[s]))++s;
        size_t e = tok.size(); while (e > s && isspace((unsigned char)tok[e - 1]))--e;
        if (e > s) vals.push_back((int32_t)stol(tok.substr(s, e - s)));
    }
    return vals;
}

bool load_mnist_csv(const string& path, uint8_t img[IN_H][IN_W]) {
    ifstream fin(path);
    if (!fin) return false;
    string line; int r = 0;
    while (r < IN_H && getline(fin, line)) {
        stringstream ss(line); string tok; int c = 0;
        while (c < IN_W && getline(ss, tok, ',')) {
            int v = stoi(tok);
            if (v < 0) v = 0; else if (v > 255) v = 255;
            img[r][c] = (uint8_t)v; ++c;
        } ++r;
    }
    return r == IN_H;
}

struct Weights {
    int32_t conv[K][KH][KW];
    int32_t bias_conv[K];
    int32_t fc[NUM_CLASSES][FLAT];
    int32_t bias_fc[NUM_CLASSES];
};

Weights load_weights(const string& conv_w, const string& conv_b,
                     const string& fc_w, const string& fc_b) {
    Weights W{};
    auto cw = read_csv_line_ints(conv_w);
    size_t idx = 0;
    for (int k = 0; k < K; ++k)
        for (int dy = 0; dy < KH; ++dy)
            for (int dx = 0; dx < KW; ++dx)
                W.conv[k][dy][dx] = cw[idx++];
    auto cb = read_csv_line_ints(conv_b);
    for (int k = 0; k < K; ++k) W.bias_conv[k] = cb[k];
    auto fw = read_csv_line_ints(fc_w);
    idx = 0;
    for (int c = 0; c < NUM_CLASSES; ++c)
        for (int i = 0; i < FLAT; ++i)
            W.fc[c][i] = fw[idx++];
    auto fb = read_csv_line_ints(fc_b);
    for (int c = 0; c < NUM_CLASSES; ++c) W.bias_fc[c] = fb[c];
    return W;
}

// ============================================================
// PLAIN (평문) 연산 함수들
// ============================================================

array<array<int32_t, IN_W>, IN_H> to_plain_image(const uint8_t img[IN_H][IN_W]) {
    array<array<int32_t, IN_W>, IN_H> x{};
    for (int y = 0; y < IN_H; ++y)
        for (int z = 0; z < IN_W; ++z)
            x[y][z] = (int32_t)img[y][z];
    return x;
}

array<array<array<int32_t, OUT_W>, OUT_H>, K>
plain_conv3x3(const array<array<int32_t, IN_W>, IN_H>& x, const Weights& W) {
    array<array<array<int32_t, OUT_W>, OUT_H>, K> y{};
    for (int k = 0; k < K; ++k)
        for (int oy = 0; oy < OUT_H; ++oy)
            for (int ox = 0; ox < OUT_W; ++ox) {
                int acc = 0;
                int iy = oy * STR, ix = ox * STR;
                for (int dy = 0; dy < KH; ++dy)
                    for (int dx = 0; dx < KW; ++dx)
                        acc += x[iy + dy][ix + dx] * W.conv[k][dy][dx];
                acc += W.bias_conv[k];
                y[k][oy][ox] = acc;
            }
    return y;
}

void plain_relu_inplace(array<array<array<int32_t, OUT_W>, OUT_H>, K>& feat) {
    for (int k = 0; k < K; ++k)
        for (int y = 0; y < OUT_H; ++y)
            for (int x = 0; x < OUT_W; ++x)
                feat[k][y][x] = max(0, feat[k][y][x]);
}

array<array<array<int32_t, POOL_W>, POOL_H>, K>
plain_sum_pool3x3(const array<array<array<int32_t, OUT_W>, OUT_H>, K>& x) {
    array<array<array<int32_t, POOL_W>, POOL_H>, K> y{};
    for (int k = 0; k < K; ++k)
        for (int py = 0; py < POOL_H; ++py)
            for (int px = 0; px < POOL_W; ++px) {
                int sy = py * PH, sx = px * PW;
                int acc = 0;
                for (int dy = 0; dy < PH; ++dy)
                    for (int dx = 0; dx < PW; ++dx)
                        acc += x[k][sy + dy][sx + dx];
                y[k][py][px] = acc;
            }
    return y;
}

vector<int32_t> plain_flatten(const array<array<array<int32_t, POOL_W>, POOL_H>, K>& x) {
    vector<int32_t> v;
    v.reserve(FLAT);
    for (int k = 0; k < K; ++k)
        for (int py = 0; py < POOL_H; ++py)
            for (int px = 0; px < POOL_W; ++px)
                v.push_back(x[k][py][px]);
    return v;
}

array<int32_t, NUM_CLASSES>
plain_fc(const vector<int32_t>& flat, const Weights& W) {
    array<int32_t, NUM_CLASSES> out{};
    for (int c = 0; c < NUM_CLASSES; ++c) {
        long long acc = 0;
        for (int i = 0; i < FLAT; ++i)
            acc += 1LL * flat[i] * W.fc[c][i];
        acc += W.bias_fc[c];
        out[c] = (int32_t)acc;
    }
    return out;
}

// ============================================================
// FHE (암호문) 연산 함수들
// ============================================================

array<array<Cipher, IN_W>, IN_H> encrypt_image(const uint8_t img[IN_H][IN_W]) {
    array<array<Cipher, IN_W>, IN_H> ct{};
    for (int y = 0; y < IN_H; ++y)
        for (int x = 0; x < IN_W; ++x)
            ct[y][x] = FHE16_ENCInt((int)img[y][x], MSG_BIT);
    return ct;
}

array<array<array<Cipher, OUT_W>, OUT_H>, K>
fhe_conv3x3(const array<array<Cipher, IN_W>, IN_H>& x, const Weights& W) {
    array<array<array<Cipher, OUT_W>, OUT_H>, K> y{};
    for (int k = 0; k < K; ++k)
        for (int oy = 0; oy < OUT_H; ++oy)
            for (int ox = 0; ox < OUT_W; ++ox) {
                const int iy = oy * STR, ix = ox * STR;
                if (iy + KH > IN_H || ix + KW > IN_W) continue;
                Cipher m00 = mul_scalar_opt(x[iy + 0][ix + 0], W.conv[k][0][0]);
                Cipher m01 = mul_scalar_opt(x[iy + 0][ix + 1], W.conv[k][0][1]);
                Cipher m02 = mul_scalar_opt(x[iy + 0][ix + 2], W.conv[k][0][2]);
                Cipher m10 = mul_scalar_opt(x[iy + 1][ix + 0], W.conv[k][1][0]);
                Cipher m11 = mul_scalar_opt(x[iy + 1][ix + 1], W.conv[k][1][1]);
                Cipher m12 = mul_scalar_opt(x[iy + 1][ix + 2], W.conv[k][1][2]);
                Cipher m20 = mul_scalar_opt(x[iy + 2][ix + 0], W.conv[k][2][0]);
                Cipher m21 = mul_scalar_opt(x[iy + 2][ix + 1], W.conv[k][2][1]);
                Cipher m22 = mul_scalar_opt(x[iy + 2][ix + 2], W.conv[k][2][2]);
                Cipher r0 = add3(m00, m01, m02);
                Cipher r1 = add3(m10, m11, m12);
                Cipher r2 = add3(m20, m21, m22);
                Cipher acc = add3(r0, r1, r2);
                if (W.bias_conv[k] != 0)
                    acc = FHE16_ADD_CONSTANT(acc, W.bias_conv[k]);
                y[k][oy][ox] = acc;
            }
    return y;
}

void fhe_relu_inplace(array<array<array<Cipher, OUT_W>, OUT_H>, K>& feat) {
    for (int k = 0; k < K; ++k)
        for (int y = 0; y < OUT_H; ++y)
            for (int x = 0; x < OUT_W; ++x)
                feat[k][y][x] = FHE16_RELU(feat[k][y][x]);
}

array<array<array<Cipher, POOL_W>, POOL_H>, K>
fhe_sum_pool3x3(const array<array<array<Cipher, OUT_W>, OUT_H>, K>& x) {
    array<array<array<Cipher, POOL_W>, POOL_H>, K> y{};
    for (int k = 0; k < K; ++k)
        for (int py = 0; py < POOL_H; ++py)
            for (int px = 0; px < POOL_W; ++px) {
                int sy = py * PH, sx = px * PW;
                if (sy + PH > OUT_H || sx + PW > OUT_W) continue;
                Cipher a1 = add3(x[k][sy + 0][sx + 0], x[k][sy + 0][sx + 1], x[k][sy + 0][sx + 2]);
                Cipher a2 = add3(x[k][sy + 1][sx + 0], x[k][sy + 1][sx + 1], x[k][sy + 1][sx + 2]);
                Cipher a3 = add3(x[k][sy + 2][sx + 0], x[k][sy + 2][sx + 1], x[k][sy + 2][sx + 2]);
                y[k][py][px] = add3(a1, a2, a3);
            }
    return y;
}

vector<Cipher> fhe_flatten(const array<array<array<Cipher, POOL_W>, POOL_H>, K>& x) {
    vector<Cipher> v;
    v.reserve(FLAT);
    for (int k = 0; k < K; ++k)
        for (int y = 0; y < POOL_H; ++y)
            for (int xw = 0; xw < POOL_W; ++xw)
                v.push_back(x[k][y][xw]);
    return v;
}

// Plain vector -> Cipher vector 변환
vector<Cipher> encrypt_flat_vector(const vector<int32_t>& plain) {
    vector<Cipher> ct;
    ct.reserve(plain.size());
    for (auto v : plain)
        ct.push_back(FHE16_ENCInt(v, MSG_BIT));
    return ct;
}

array<Cipher, NUM_CLASSES>
fhe_fc(const vector<Cipher>& flat, const Weights& W) {
    array<Cipher, NUM_CLASSES> out{};
    for (int c = 0; c < NUM_CLASSES; ++c) {
        vector<Cipher> partial;
        int i = 0;
        for (; i + 2 < FLAT; i += 3) {
            Cipher t1 = mul_scalar_opt(flat[i], W.fc[c][i]);
            Cipher t2 = mul_scalar_opt(flat[i + 1], W.fc[c][i + 1]);
            Cipher t3 = mul_scalar_opt(flat[i + 2], W.fc[c][i + 2]);
            partial.push_back(add3(t1, t2, t3));
        }
        if (i < FLAT) {
            Cipher rem = get_zero();
            for (; i < FLAT; ++i)
                rem = FHE16_ADD(rem, mul_scalar_opt(flat[i], W.fc[c][i]));
            partial.push_back(rem);
        }
        while (partial.size() > 1) {
            vector<Cipher> next;
            size_t j = 0;
            for (; j + 2 < partial.size(); j += 3)
                next.push_back(add3(partial[j], partial[j + 1], partial[j + 2]));
            if (j < partial.size()) {
                Cipher rem = get_zero();
                for (; j < partial.size(); ++j) rem = FHE16_ADD(rem, partial[j]);
                next.push_back(rem);
            }
            partial.swap(next);
        }
        Cipher acc = partial.front();
        if (W.bias_fc[c] != 0)
            acc = FHE16_ADD_CONSTANT(acc, W.bias_fc[c]);
        out[c] = acc;
    }
    return out;
}

// ============================================================
// HYBRID INFERENCE (핵심 함수)
// ============================================================

/**
 * hybrid_inference: n개 레이어 중 마지막 fhe_layers개만 FHE로 실행
 *
 * fhe_layers 값에 따른 동작:
 *   5: 전체 FHE (Enc → Conv → ReLU → Pool → FC → Dec)
 *   4: Conv부터 FHE (Plain입력 → Enc → Conv → ReLU → Pool → FC → Dec)
 *   3: ReLU부터 FHE (Plain Conv → Enc → ReLU → Pool → FC → Dec)
 *   2: Pool부터 FHE (Plain Conv,ReLU → Enc → Pool → FC → Dec)
 *   1: FC만 FHE (Plain Conv,ReLU,Pool → Enc → FC → Dec)
 *   0: 전체 평문 (테스트용)
 */
int hybrid_inference(const uint8_t img[IN_H][IN_W], const Weights& W,
                     int32_t* sk, int fhe_layers, bool verbose = false) {
    using clock = chrono::steady_clock;
    auto sec = [](auto d) { return chrono::duration<double>(d).count(); };

    auto t_start = clock::now();

    if (fhe_layers == 0) {
        // 전체 평문 연산
        if (verbose) cout << "[Hybrid] Mode: Full Plain (fhe_layers=0)\n";
        auto x = to_plain_image(img);
        auto conv = plain_conv3x3(x, W);
        plain_relu_inplace(conv);
        auto pool = plain_sum_pool3x3(conv);
        auto flat = plain_flatten(pool);
        auto logits = plain_fc(flat, W);

        int pred = -1, best = INT32_MIN;
        for (int c = 0; c < NUM_CLASSES; ++c)
            if (logits[c] > best) { best = logits[c]; pred = c; }

        if (verbose) cout << "[Hybrid] Total: " << sec(clock::now() - t_start) << " s\n";
        return pred;
    }

    if (fhe_layers >= 5) {
        // 전체 FHE
        if (verbose) cout << "[Hybrid] Mode: Full FHE (fhe_layers=5)\n";

        auto t0 = clock::now();
        auto ct_in = encrypt_image(img);
        if (verbose) cout << "  Encrypt: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        auto conv_out = fhe_conv3x3(ct_in, W);
        if (verbose) cout << "  Conv: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        fhe_relu_inplace(conv_out);
        if (verbose) cout << "  ReLU: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        auto pool_out = fhe_sum_pool3x3(conv_out);
        if (verbose) cout << "  Pool: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        auto flat = fhe_flatten(pool_out);
        auto logits = fhe_fc(flat, W);
        if (verbose) cout << "  FC: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        int pred = -1; int32_t best = INT32_MIN;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            int32_t v = FHE16_DECInt(logits[c], sk);
            if (v > best) { best = v; pred = c; }
        }
        if (verbose) cout << "  Decrypt: " << sec(clock::now() - t0) << " s\n";
        if (verbose) cout << "[Hybrid] Total: " << sec(clock::now() - t_start) << " s\n";
        return pred;
    }

    // 하이브리드 모드: 앞쪽 레이어는 평문, 뒤쪽만 FHE
    if (verbose) cout << "[Hybrid] Mode: Hybrid (fhe_layers=" << fhe_layers << ")\n";

    // Step 1: 평문으로 처리할 부분
    auto plain_img = to_plain_image(img);

    array<array<array<int32_t, OUT_W>, OUT_H>, K> plain_conv;
    array<array<array<int32_t, POOL_W>, POOL_H>, K> plain_pool;
    vector<int32_t> plain_flat;
    array<int32_t, NUM_CLASSES> plain_logits;

    // fhe_layers < 5: Conv 평문
    if (fhe_layers < 5) {
        auto t0 = clock::now();
        plain_conv = plain_conv3x3(plain_img, W);
        if (verbose) cout << "  Plain Conv: " << sec(clock::now() - t0) << " s\n";
    }

    // fhe_layers < 4: ReLU 평문
    if (fhe_layers < 4) {
        auto t0 = clock::now();
        plain_relu_inplace(plain_conv);
        if (verbose) cout << "  Plain ReLU: " << sec(clock::now() - t0) << " s\n";
    }

    // fhe_layers < 3: Pool 평문
    if (fhe_layers < 3) {
        auto t0 = clock::now();
        plain_pool = plain_sum_pool3x3(plain_conv);
        if (verbose) cout << "  Plain Pool: " << sec(clock::now() - t0) << " s\n";
    }

    // fhe_layers < 2: Flatten 평문
    if (fhe_layers < 2) {
        plain_flat = plain_flatten(plain_pool);
    }

    // fhe_layers < 1: FC도 평문 (이미 위에서 처리됨)
    if (fhe_layers < 1) {
        plain_logits = plain_fc(plain_flat, W);
        int pred = -1, best = INT32_MIN;
        for (int c = 0; c < NUM_CLASSES; ++c)
            if (plain_logits[c] > best) { best = plain_logits[c]; pred = c; }
        return pred;
    }

    // Step 2: FHE로 처리할 부분
    if (fhe_layers == 1) {
        // FC만 FHE
        auto t0 = clock::now();
        auto ct_flat = encrypt_flat_vector(plain_flat);
        if (verbose) cout << "  Encrypt flat: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        auto logits = fhe_fc(ct_flat, W);
        if (verbose) cout << "  FHE FC: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        int pred = -1; int32_t best = INT32_MIN;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            int32_t v = FHE16_DECInt(logits[c], sk);
            if (v > best) { best = v; pred = c; }
        }
        if (verbose) cout << "  Decrypt: " << sec(clock::now() - t0) << " s\n";
        if (verbose) cout << "[Hybrid] Total: " << sec(clock::now() - t_start) << " s\n";
        return pred;
    }

    if (fhe_layers == 2) {
        // Pool + FC를 FHE
        // Conv 출력을 암호화
        auto t0 = clock::now();
        array<array<array<Cipher, OUT_W>, OUT_H>, K> ct_conv{};
        for (int k = 0; k < K; ++k)
            for (int y = 0; y < OUT_H; ++y)
                for (int x = 0; x < OUT_W; ++x)
                    ct_conv[k][y][x] = FHE16_ENCInt(plain_conv[k][y][x], MSG_BIT);
        if (verbose) cout << "  Encrypt conv out: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        auto pool_out = fhe_sum_pool3x3(ct_conv);
        if (verbose) cout << "  FHE Pool: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        auto flat = fhe_flatten(pool_out);
        auto logits = fhe_fc(flat, W);
        if (verbose) cout << "  FHE FC: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        int pred = -1; int32_t best = INT32_MIN;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            int32_t v = FHE16_DECInt(logits[c], sk);
            if (v > best) { best = v; pred = c; }
        }
        if (verbose) cout << "  Decrypt: " << sec(clock::now() - t0) << " s\n";
        if (verbose) cout << "[Hybrid] Total: " << sec(clock::now() - t_start) << " s\n";
        return pred;
    }

    if (fhe_layers == 3) {
        // ReLU + Pool + FC를 FHE
        auto t0 = clock::now();
        array<array<array<Cipher, OUT_W>, OUT_H>, K> ct_conv{};
        for (int k = 0; k < K; ++k)
            for (int y = 0; y < OUT_H; ++y)
                for (int x = 0; x < OUT_W; ++x)
                    ct_conv[k][y][x] = FHE16_ENCInt(plain_conv[k][y][x], MSG_BIT);
        if (verbose) cout << "  Encrypt conv out: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        fhe_relu_inplace(ct_conv);
        if (verbose) cout << "  FHE ReLU: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        auto pool_out = fhe_sum_pool3x3(ct_conv);
        if (verbose) cout << "  FHE Pool: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        auto flat = fhe_flatten(pool_out);
        auto logits = fhe_fc(flat, W);
        if (verbose) cout << "  FHE FC: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        int pred = -1; int32_t best = INT32_MIN;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            int32_t v = FHE16_DECInt(logits[c], sk);
            if (v > best) { best = v; pred = c; }
        }
        if (verbose) cout << "  Decrypt: " << sec(clock::now() - t0) << " s\n";
        if (verbose) cout << "[Hybrid] Total: " << sec(clock::now() - t_start) << " s\n";
        return pred;
    }

    if (fhe_layers == 4) {
        // Conv + ReLU + Pool + FC를 FHE (입력만 평문)
        auto t0 = clock::now();
        array<array<Cipher, IN_W>, IN_H> ct_in{};
        for (int y = 0; y < IN_H; ++y)
            for (int x = 0; x < IN_W; ++x)
                ct_in[y][x] = FHE16_ENCInt(plain_img[y][x], MSG_BIT);
        if (verbose) cout << "  Encrypt input: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        auto conv_out = fhe_conv3x3(ct_in, W);
        if (verbose) cout << "  FHE Conv: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        fhe_relu_inplace(conv_out);
        if (verbose) cout << "  FHE ReLU: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        auto pool_out = fhe_sum_pool3x3(conv_out);
        if (verbose) cout << "  FHE Pool: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        auto flat = fhe_flatten(pool_out);
        auto logits = fhe_fc(flat, W);
        if (verbose) cout << "  FHE FC: " << sec(clock::now() - t0) << " s\n";

        t0 = clock::now();
        int pred = -1; int32_t best = INT32_MIN;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            int32_t v = FHE16_DECInt(logits[c], sk);
            if (v > best) { best = v; pred = c; }
        }
        if (verbose) cout << "  Decrypt: " << sec(clock::now() - t0) << " s\n";
        if (verbose) cout << "[Hybrid] Total: " << sec(clock::now() - t_start) << " s\n";
        return pred;
    }

    return -1;
}

// ---------- MAIN ----------
int main(int argc, char** argv) {
    signal(SIGSEGV, [](int) { cerr << "[FATAL] Segmentation fault\n"; _Exit(1); });
    cout.setf(ios::fixed); cout << setprecision(3);

    // Parse arguments
    string img_csv = "../mnist_batch/mnist_0.csv";
    int fhe_layers = 5;
    bool verbose = true;

    for (int a = 1; a < argc; ++a) {
        if (strncmp(argv[a], "--fhe-layers=", 13) == 0)
            fhe_layers = atoi(argv[a] + 13);
        else if (strncmp(argv[a], "--input=", 8) == 0)
            img_csv = argv[a] + 8;
        else if (strcmp(argv[a], "--quiet") == 0 || strcmp(argv[a], "-q") == 0)
            verbose = false;
    }

    cout << "==========================================\n";
    cout << " FHE16-CNN Hybrid Inference\n";
    cout << " --fhe-layers=" << fhe_layers << " (0=plain, 5=full FHE)\n";
    cout << " --input=" << img_csv << "\n";
    cout << "==========================================\n";

    // Load weights
    string conv_w = "../export_csv/conv_w.csv";
    string conv_b = "../export_csv/conv_b.csv";
    string fc_w = "../export_csv/fc_w.csv";
    string fc_b = "../export_csv/fc_b.csv";

    int32_t* sk = FHE16_GenEval();
    auto W = load_weights(conv_w, conv_b, fc_w, fc_b);

    // Load image
    uint8_t img[IN_H][IN_W];
    if (!load_mnist_csv(img_csv, img)) {
        cerr << "[ERR] Failed to load " << img_csv << "\n";
        return 1;
    }

    // Load label
    string label_path = img_csv.substr(0, img_csv.find_last_of('.')) + ".label";
    int true_label = -1;
    { ifstream f(label_path); if (f) f >> true_label; }

    // Run inference
    int pred = hybrid_inference(img, W, sk, fhe_layers, verbose);

    cout << "==========================================\n";
    cout << " True label : " << true_label << "\n";
    cout << " Predicted  : " << pred << "\n";
    cout << " Match      : " << (pred == true_label ? "YES" : "NO") << "\n";
    cout << "==========================================\n";

    return 0;
}
