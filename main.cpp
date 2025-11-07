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
static constexpr int MSG_BIT = 20; // Can be adjusted to prevent potential overflow.    

// --- Parameter settings  ---
static constexpr int IN_H=28, IN_W=28; // MNIST size
static constexpr int K=3, KH=3, KW=3, STR=3; // convolution filter size
static constexpr int OUT_H=9, OUT_W=9; // convolution output size
static constexpr int PH=3, PW=3; // Sumpool size
static constexpr int POOL_H=3, POOL_W=3; // Sumpool output size
static constexpr int FLAT=K*POOL_H*POOL_W; // Flattened vector size
static constexpr int NUM_CLASSES=10; // Class(0~9)

// ---------- Optimization functions ----------
Cipher get_zero() {
    static Cipher Z = nullptr;
    if (!Z) Z = FHE16_ENCInt(0, MSG_BIT);
    return Z;
}
//Scalar Multiplication Optimization (if the scalar is 0, return Enc(0), if 1, return itself, if -1, return using substraction)
inline Cipher mul_scalar_opt(Cipher c, int32_t w){
    if(w==0) return get_zero();
    if(w==1) return c;
    if(w==-1) return FHE16_SUB(get_zero(),c);
    return FHE16_SMULL_CONSTANT(c,w);
}
// Make fuction more intuitively
inline Cipher add3(Cipher a,Cipher b,Cipher c){return FHE16_ADD3(a,b,c);}

// ---------- MNIST sample loaders ----------
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

// ---------- CNN Weights loaders ----------
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

// ---------- Encryption ----------
array<array<Cipher,IN_W>,IN_H> encrypt_image(const uint8_t img[IN_H][IN_W]){
    array<array<Cipher,IN_W>,IN_H> ct{};
    for(int y=0;y<IN_H;++y)
        for(int x=0;x<IN_W;++x)
            ct[y][x]=FHE16_ENCInt((int)img[y][x],MSG_BIT);
    return ct;
}

// ---------- Convolution layer(optimized ADD3) ----------
array<array<array<Cipher,OUT_W>,OUT_H>,K>
conv3x3s3_bias_add3x4(const array<array<Cipher,IN_W>,IN_H>& x,const Weights& W){
    array<array<array<Cipher,OUT_W>,OUT_H>,K> y{};
    for(int k=0;k<K;++k)
        for(int oy=0;oy<OUT_H;++oy)
            for(int ox=0;ox<OUT_W;++ox){
                const int iy=oy*STR, ix=ox*STR;
                if(iy+KH>IN_H || ix+KW>IN_W) continue;
                Cipher acc=get_zero();
                for(int dy=0;dy<KH;++dy)
                    for(int dx=0;dx<KW;++dx)
                        acc=FHE16_ADD(acc, mul_scalar_opt(x[iy+dy][ix+dx],W.conv[k][dy][dx]));
                if(W.bias_conv[k]!=0) acc=FHE16_ADD_CONSTANT(acc,W.bias_conv[k]);
                y[k][oy][ox]=acc;
            }
    return y;
}

// ---------- ReLU ----------
void relu_inplace(array<array<array<Cipher,OUT_W>,OUT_H>,K>& feat){
    for(int k=0;k<K;++k)
        for(int y=0;y<OUT_H;++y)
            for(int x=0;x<OUT_W;++x)
                feat[k][y][x]=FHE16_RELU(feat[k][y][x]);
}

// ---------- SumPooling layer(optimized ADD3) ----------
array<array<array<Cipher,POOL_W>,POOL_H>,K>
sum_pool3x3_add3x4(const array<array<array<Cipher,OUT_W>,OUT_H>,K>& x){
    array<array<array<Cipher,POOL_W>,POOL_H>,K> y{};
    for(int k=0;k<K;++k)
        for(int py=0;py<POOL_H;++py)
            for(int px=0;px<POOL_W;++px){
                int sy=py*PH, sx=px*PW;
                if(sy+PH>OUT_H || sx+PW>OUT_W) continue;
                Cipher acc=get_zero();
                for(int dy=0;dy<PH;++dy)
                    for(int dx=0;dx<PW;++dx)
                        acc=FHE16_ADD(acc, x[k][sy+dy][sx+dx]);
                y[k][py][px]=acc;
            }
    return y;
}

// ---------- Flattening ----------
vector<Cipher> flatten_3x3x3(const array<array<array<Cipher,POOL_W>,POOL_H>,K>& x){
    vector<Cipher> v; v.reserve(FLAT);
    for(int k=0;k<K;++k)
        for(int y=0;y<POOL_H;++y)
            for(int xw=0;xw<POOL_W;++xw)
                v.push_back(x[k][y][xw]);
    return v;
}

// ---------- FC layer ----------
array<Cipher,NUM_CLASSES>
fc10_add3_schedule(const vector<Cipher>& flat,const Weights& W){
    array<Cipher,NUM_CLASSES> out{};
    for(int c=0;c<NUM_CLASSES;++c){
        Cipher acc=get_zero();
        for(int i=0;i<FLAT;++i)
            acc=FHE16_ADD(acc, mul_scalar_opt(flat[i],W.fc[c][i]));
        if(W.bias_fc[c]!=0) acc=FHE16_ADD_CONSTANT(acc,W.bias_fc[c]);
        out[c]=acc;
    }
    return out;
}

// ---------- Plain value inference (for comparison) ----------
int plain_predict(const uint8_t img[IN_H][IN_W], const Weights& W){
    array<array<int32_t,IN_W>,IN_H> x{};
    for(int y=0;y<IN_H;++y)
        for(int z=0;z<IN_W;++z)
            x[y][z]=(int32_t)img[y][z];
    array<array<array<int32_t,OUT_W>,OUT_H>,K> y{};
    for(int k=0;k<K;++k)
        for(int oy=0;oy<OUT_H;++oy)
            for(int ox=0;ox<OUT_W;++ox){
                int acc=0; int iy=oy*STR, ix=ox*STR;
                for(int dy=0;dy<KH;++dy)
                    for(int dx=0;dx<KW;++dx)
                        acc+=x[iy+dy][ix+dx]*W.conv[k][dy][dx];
                acc+=W.bias_conv[k];
                y[k][oy][ox]=max(0,acc);
            }
    array<array<array<int32_t,POOL_W>,POOL_H>,K> p{};
    for(int k=0;k<K;++k)
        for(int py=0;py<POOL_H;++py)
            for(int px=0;px<POOL_W;++px){
                int sy=py*PH,sx=px*PW;int acc=0;
                for(int dy=0;dy<PH;++dy)
                    for(int dx=0;dx<PW;++dx)
                        acc+=y[k][sy+dy][sx+dx];
                p[k][py][px]=acc;
            }
    vector<int32_t> f;f.reserve(FLAT);
    for(int k=0;k<K;++k)
        for(int py=0;py<POOL_H;++py)
            for(int px=0;px<POOL_W;++px)
                f.push_back(p[k][py][px]);
    array<int32_t,NUM_CLASSES> out{};
    for(int c=0;c<NUM_CLASSES;++c){
        long long acc=0;
        for(int i=0;i<FLAT;++i)
            acc+=1LL*f[i]*W.fc[c][i];
        acc+=W.bias_fc[c];
        out[c]=(int32_t)acc;
    }
    int pred=-1,best=INT32_MIN;
    for(int c=0;c<NUM_CLASSES;++c)
        if(out[c]>best){best=out[c];pred=c;}
    return pred;
}

// ---------- MAIN ----------
int main(int argc,char** argv){
    signal(SIGSEGV,[](int){cerr<<"[FATAL] Segmentation fault\n";_Exit(1);});
    using clock=chrono::steady_clock;
    auto sec=[](auto d){return chrono::duration<double>(d).count();};
    cout.setf(ios::fixed);cout<<setprecision(3);

    // File load
    string conv_w="../export_csv/conv_w.csv";
    string conv_b="../export_csv/conv_b.csv";
    string fc_w="../export_csv/fc_w.csv";
    string fc_b="../export_csv/fc_b.csv";
    int32_t* sk=FHE16_GenEval();
    auto W=load_weights(conv_w,conv_b,fc_w,fc_b);

    int batch_index=0;
    for(int a=1;a<argc;++a)
        if(strncmp(argv[a],"--batch=",8)==0)
            batch_index=atoi(argv[a]+8);

    int correct=0;
    int diff=0;
    double batch_enc=0,batch_conv=0,batch_relu=0,batch_pool=0,batch_flat=0,batch_fc=0,batch_arg=0;
    auto start=clock::now();

    for(int i=batch_index*100;i<(batch_index+1)*100;++i){
        string fname="../mnist_batch/mnist_"+to_string(i)+".csv";
        string label_path="../mnist_batch/mnist_"+to_string(i)+".label";
        int label=-1;{ifstream f(label_path);if(f)f>>label;}
        uint8_t img[IN_H][IN_W];
        if(!load_mnist_csv(fname,img)){cerr<<"[ERR] "<<fname<<"\n";continue;}

        // Plain inference
        int plain_pred=plain_predict(img,W);

        // FHE inference
        auto t0=clock::now();auto ct_in=encrypt_image(img);auto t1=clock::now();
        batch_enc+=sec(t1-t0);
        auto c0=clock::now();auto conv_out=conv3x3s3_bias_add3x4(ct_in,W);auto c1=clock::now();
        batch_conv+=sec(c1-c0);
        auto r0=clock::now();relu_inplace(conv_out);auto r1=clock::now();
        batch_relu+=sec(r1-r0);
        auto p0=clock::now();auto pool_out=sum_pool3x3_add3x4(conv_out);auto p1=clock::now();
        batch_pool+=sec(p1-p0);
        auto f0=clock::now();auto flat_out=flatten_3x3x3(pool_out);auto f1=clock::now();
        batch_flat+=sec(f1-f0);
        auto F0=clock::now();auto logits=fc10_add3_schedule(flat_out,W);auto F1=clock::now();
        batch_fc+=sec(F1-F0);
        auto A0=clock::now();
        int fhe_pred=-1;int32_t best=INT32_MIN;
        for(int c=0;c<NUM_CLASSES;++c){
            int32_t v=FHE16_DECInt(logits[c],sk);
            if(v>best){best=v;fhe_pred=c;}
        }
        auto A1=clock::now();
        batch_arg+=sec(A1-A0);

        if(fhe_pred==label) correct++;
        if(fhe_pred!=plain_pred) diff++;
    }

    auto end=clock::now();
    double total=sec(end-start);

    cout<<"[Batch "<<batch_index<<"] Accuracy="<<correct<<"/100 ("<<(double)correct
        << "%) Diff="<<diff<<"/100 Time="<<total<<" s\n";

    ofstream log("accuracy_log.txt",ios::app);
    log<<"[Batch "<<batch_index<<"] Accuracy="<<correct<<"/100 ("<<(double)correct
       << "%) Diff="<<diff<<"/100 Time="<<total<<" s\n";
    log.close();

    return 0;
}

