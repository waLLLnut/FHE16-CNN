#ifndef GPUAPI_CUH
#define GPUAPI_CUH

#ifdef __cplusplus
extern "C" {
#endif
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>


#include <stdio.h>
#ifdef USE_CUDA
#include "gpu.hpp"
#endif


#define checkCudaErrors(err) { if (err != CUDA_SUCCESS) { std::cerr << "CUDA Error: " << err << std::endl; std::exit(EXIT_FAILURE); }}

std::vector<char> readPTXFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file - " << filename << std::endl;
        abort();
    }   
    return std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}



void printCudaVersion() {
    std::cout << "CUDA Compiled version: " << __CUDACC_VER__ << std::endl;
    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout << "CUDA Runtime version: " << runtime_ver << std::endl
    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    std::cout << "CUDA Driver version: " << driver_ver << std::endl;
};

#ifdef __cplusplus
}
#endif

#endif // End header
