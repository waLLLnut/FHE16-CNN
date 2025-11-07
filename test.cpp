#include <iostream>
#include <vector>
#include <array>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <climits>
#include <iomanip>
#include <csignal>

#include <BinOperationCstyle.hpp>
#include <soAPI.hpp>
using namespace std;

using Cipher = int32_t*;
static constexpr int MSG_BIT = 32; // Can be adjusted to prevent potential overflow.

// --- Parameter settings  ---
static constexpr int IN_H=28, IN_W=28; // MNIST size
static constexpr int K=3, KH=3, KW=3, STR=3; // convolution filter size
static constexpr int OUT_H=9, OUT_W=9; // convolution output size
static constexpr int PH=3, PW=3; // Sumpool size
static constexpr int POOL_H=3, POOL_W=3; // Sumpool output size
static constexpr int FLAT=K*POOL_H*POOL_W; // Flattened vector size
static constexpr int NUM_CLASSES=10; // Class(0~9)

// ----------------- Optimization functions -----------------
// For efficiency, Enc(0) is cached in the global variable Z. 
Cipher get_zero() {
    static Cipher Z=nullptr;
    if(!Z) Z=FHE16_ENCInt(0,MSG_BIT);
    return Z;
}

//Scalar Multiplication Optimization (if the scalar is 0, return Enc(0), if 1, return itself, if -1, return using substraction)
inline Cipher mul_scalar_opt(Cipher c,int32_t w){
    if(w==0) return get_zero();
    if(w==1) return c;
    if(w==-1) return FHE16_SUB(get_zero(),c);
    return FHE16_SMULL_CONSTANT(c,w);
}
// Make fuction more intuitively
inline Cipher add3(Cipher a,Cipher b,Cipher c){return FHE16_ADD3(a,b,c);}

// ----------------- MNIST sample loaders -----------------
vector<int32_t> read_csv_line_ints(const string& path){
    ifstream f(path);
    if(!f){cerr<<"[ERR] open "<<path<<"\n"; abort();}
    string line; getline(f,line);
    vector<int32_t> vals; string tok; stringstream ss(line);
    while(getline(ss,tok,',')){
        size_t s=0; while(s<tok.size()&&isspace((unsigned char)tok[s]))++s;
        size_t e=tok.size(); while(e>s&&isspace((unsigned char)tok[e-1]))--e;
        if(e>s) vals.push_back((int32_t)stol(tok.substr(s,e-s)));
    }
    return vals;
}
bool load_mnist_csv(const string& path,uint8_t img[IN_H][IN_W]){
    ifstream fin(path);
    if(!fin) return false;
    string line; int r=0;
    while(r<IN_H && getline(fin,line)){
        stringstream ss(line); string tok; int c=0;
        while(c<IN_W && getline(ss,tok,',')){
            int v=stoi(tok);
            if(v<0)v=0;else if(v>255)v=255;
            img[r][c]=(uint8_t)v; ++c;
        } ++r;
    } return r==IN_H;
}

// ----------------- CNN Weights loaders -----------------
struct Weights{
    int32_t conv[K][KH][KW];
    int32_t bias_conv[K];
    int32_t fc[NUM_CLASSES][FLAT];
    int32_t bias_fc[NUM_CLASSES];
};
Weights load_weights(const string& conv_w,const string& conv_b,
                     const string& fc_w,const string& fc_b){
    Weights W{};
    auto cw=read_csv_line_ints(conv_w);
    size_t idx=0;
    for(int k=0;k<K;++k)
        for(int dy=0;dy<KH;++dy)
            for(int dx=0;dx<KW;++dx)
                W.conv[k][dy][dx]=cw[idx++];
    auto cb=read_csv_line_ints(conv_b);
    for(int k=0;k<K;++k) W.bias_conv[k]=cb[k];
    auto fw=read_csv_line_ints(fc_w);
    idx=0;
    for(int c=0;c<NUM_CLASSES;++c)
        for(int i=0;i<FLAT;++i)
            W.fc[c][i]=fw[idx++];
    auto fb=read_csv_line_ints(fc_b);
    for(int c=0;c<NUM_CLASSES;++c) W.bias_fc[c]=fb[c];
    return W;
}

// ----------------- Encryption -----------------
array<array<Cipher,IN_W>,IN_H> encrypt_image(const uint8_t img[IN_H][IN_W]){
    array<array<Cipher,IN_W>,IN_H> ct{};
    for(int y=0;y<IN_H;++y)
        for(int x=0;x<IN_W;++x)
            ct[y][x]=FHE16_ENCInt((int)img[y][x],MSG_BIT);
    return ct;
}

// ----------------- Convolution layer(optimized ADD3) -----------------
array<array<array<Cipher,OUT_W>,OUT_H>,K>
conv3x3s3_bias_add3x4(const array<array<Cipher,IN_W>,IN_H>& x,const Weights& W){
    array<array<array<Cipher,OUT_W>,OUT_H>,K> y{};

    for(int k=0;k<K;++k){
        for(int oy=0;oy<OUT_H;++oy){
            for(int ox=0;ox<OUT_W;++ox){
                const int iy=oy*STR, ix=ox*STR;
                if(iy+KH>IN_H || ix+KW>IN_W) continue;
                Cipher m00=mul_scalar_opt(x[iy+0][ix+0],W.conv[k][0][0]);
                Cipher m01=mul_scalar_opt(x[iy+0][ix+1],W.conv[k][0][1]);
                Cipher m02=mul_scalar_opt(x[iy+0][ix+2],W.conv[k][0][2]);
                Cipher m10=mul_scalar_opt(x[iy+1][ix+0],W.conv[k][1][0]);
                Cipher m11=mul_scalar_opt(x[iy+1][ix+1],W.conv[k][1][1]);
                Cipher m12=mul_scalar_opt(x[iy+1][ix+2],W.conv[k][1][2]);
                Cipher m20=mul_scalar_opt(x[iy+2][ix+0],W.conv[k][2][0]);
                Cipher m21=mul_scalar_opt(x[iy+2][ix+1],W.conv[k][2][1]);
                Cipher m22=mul_scalar_opt(x[iy+2][ix+2],W.conv[k][2][2]);
                Cipher r0=add3(m00,m01,m02);
                Cipher r1=add3(m10,m11,m12);
                Cipher r2=add3(m20,m21,m22);
                Cipher acc=add3(r0,r1,r2);

                int32_t bias = W.bias_conv[k];
                if (bias != 0)
                    acc = FHE16_ADD_CONSTANT(acc, bias);

                y[k][oy][ox]=acc;
            }
        }
    }
    return y;
}

// ----------------- ReLU -----------------
void relu_inplace(array<array<array<Cipher,OUT_W>,OUT_H>,K>& feat){
    for(int k=0;k<K;++k)
        for(int y=0;y<OUT_H;++y)
            for(int x=0;x<OUT_W;++x)
                feat[k][y][x]=FHE16_RELU(feat[k][y][x]);
}

// ----------------- SumPooling layer(optimized ADD3) -----------------
array<array<array<Cipher,POOL_W>,POOL_H>,K>
sum_pool3x3_add3x4(const array<array<array<Cipher,OUT_W>,OUT_H>,K>& x){
    array<array<array<Cipher,POOL_W>,POOL_H>,K> y{};
    for(int k=0;k<K;++k)
        for(int py=0;py<POOL_H;++py)
            for(int px=0;px<POOL_W;++px){
                int sy=py*PH, sx=px*PW;
                if(sy+PH>OUT_H || sx+PW>OUT_W) continue;
                Cipher a1=add3(x[k][sy+0][sx+0],x[k][sy+0][sx+1],x[k][sy+0][sx+2]);
                Cipher a2=add3(x[k][sy+1][sx+0],x[k][sy+1][sx+1],x[k][sy+1][sx+2]);
                Cipher a3=add3(x[k][sy+2][sx+0],x[k][sy+2][sx+1],x[k][sy+2][sx+2]);
                y[k][py][px]=add3(a1,a2,a3);
            }
    return y;
}

// ----------------- Flattening -----------------
vector<Cipher> flatten_3x3x3(const array<array<array<Cipher,POOL_W>,POOL_H>,K>& x){
    vector<Cipher> v; v.reserve(FLAT);
    for(int k=0;k<K;++k)
        for(int y=0;y<POOL_H;++y)
            for(int xw=0;xw<POOL_W;++xw)
                v.push_back(x[k][y][xw]);
    return v;
}

// ----------------- FC layer -----------------
array<Cipher,NUM_CLASSES>
fc10_add3_schedule(const vector<Cipher>& flat,const Weights& W){
    array<Cipher,NUM_CLASSES> out{};

    for(int c=0;c<NUM_CLASSES;++c){
        vector<Cipher> partial;
        int i=0;
        for(;i+2<FLAT;i+=3){
            Cipher t1=mul_scalar_opt(flat[i],W.fc[c][i]);
            Cipher t2=mul_scalar_opt(flat[i+1],W.fc[c][i+1]);
            Cipher t3=mul_scalar_opt(flat[i+2],W.fc[c][i+2]);
            partial.push_back(add3(t1,t2,t3));
        }
        if(i<FLAT){
            Cipher rem=get_zero();
            for(;i<FLAT;++i)
                rem=FHE16_ADD(rem,mul_scalar_opt(flat[i],W.fc[c][i]));
            partial.push_back(rem);
        }
        while(partial.size()>1){
            vector<Cipher> next;
            size_t j=0;
            for(;j+2<partial.size();j+=3)
                next.push_back(add3(partial[j],partial[j+1],partial[j+2]));
            if(j<partial.size()){
                Cipher rem=get_zero();
                for(;j<partial.size();++j) rem=FHE16_ADD(rem,partial[j]);
                next.push_back(rem);
            }
            partial.swap(next);
        }
        Cipher acc=partial.front();

        int32_t bias = W.bias_fc[c];
        if (bias != 0)
            acc = FHE16_ADD_CONSTANT(acc, bias);

        out[c]=acc;
    }
    return out;
}

// ----------------- MAIN -----------------
int main(int argc,char** argv){
    using clock=chrono::steady_clock;
    auto sec=[](auto d){return chrono::duration<double>(d).count();};
    cout.setf(ios::fixed); cout<<setprecision(3);

    // File load
    string conv_w="../export_csv/conv_w.csv";
    string conv_b="../export_csv/conv_b.csv";
    string fc_w="../export_csv/fc_w.csv";
    string fc_b="../export_csv/fc_b.csv";
    string img_csv="../mnist_batch/mnist_0.csv";

    string label_path=img_csv.substr(0,img_csv.find_last_of('.'))+".label";
    int true_label=-1; {ifstream fin(label_path); if(fin) fin>>true_label;}

    int32_t* sk=FHE16_GenEval();
    auto W=load_weights(conv_w,conv_b,fc_w,fc_b);
    uint8_t img[IN_H][IN_W];
    if(!load_mnist_csv(img_csv,img)){cerr<<"[ERR] failed to load "<<img_csv<<"\n";return 1;}

    cout<<"[INFO] Single-sample inference\n";

    // inference

    auto total_start=clock::now();

    auto t0=clock::now(); auto ct_in=encrypt_image(img); auto t1=clock::now();
    cout<<"Encryption: "<<sec(t1-t0)<<" s\n";

    auto c0=clock::now(); auto conv_out=conv3x3s3_bias_add3x4(ct_in,W); auto c1=clock::now();
    cout<<"Conv: "<<sec(c1-c0)<<" s\n";

    auto r0=clock::now(); relu_inplace(conv_out); auto r1=clock::now();
    cout<<"ReLU: "<<sec(r1-r0)<<" s\n";

    auto p0=clock::now(); auto pool_out=sum_pool3x3_add3x4(conv_out); auto p1=clock::now();
    cout<<"SumPool: "<<sec(p1-p0)<<" s\n";

    auto f0=clock::now(); auto flat_out=flatten_3x3x3(pool_out); auto f1=clock::now();
    cout<<"Flatten: "<<sec(f1-f0)<<" s\n";

    auto F0=clock::now(); auto logits=fc10_add3_schedule(flat_out,W); auto F1=clock::now();
    cout<<"FC: "<<sec(F1-F0)<<" s\n";

    auto A0=clock::now();
    int pred=-1; int32_t best=INT32_MIN;
    for(int c=0;c<NUM_CLASSES;++c){
        int32_t v=FHE16_DECInt(logits[c],sk);
        if(v>best){best=v; pred=c;}
    }
    auto A1=clock::now();
    cout<<"Argmax(decrypt): "<<sec(A1-A0)<<" s\n";

    auto total_end=clock::now();

    cout<<"===========================================\n";
    cout<<"True label : "<<true_label<<"\n";
    cout<<"Predicted  : "<<pred<<" (logit="<<best<<")\n";
    cout<<"Total time : "<<sec(total_end-total_start)<<" s\n";
    cout<<"===========================================\n";

    return 0;
}

