cmake_minimum_required(VERSION 3.10)
project(FHE16Example CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 소스 파일
add_executable(main main.cpp)

# NUMA
find_library(NUMA_LIB numa REQUIRED)
# 헤더 경로 (libFHE16.a에서 제공하는 헤더가 /usr/include/FHE16에 있다고 가정)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include_FHE16/FHE16/include)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include_FHE16/FHE16/include/soAPI)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include_FHE16/FHE16/include/math)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include_FHE16/FHE16/include/lwe)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include_FHE16/FHE16/include/thread)

target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include_FHE16/FHE16_Module/include/ntt/)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include_FHE16/FHE16_Module/include/bootstrapping/)

# 정적 라이브러리 직접 링크
target_link_libraries(main PRIVATE ${CMAKE_SOURCE_DIR}/lib/libFHE16.so)
target_link_libraries(main PRIVATE ${CMAKE_SOURCE_DIR}/lib/libFHE16_Module.so)
target_link_libraries(main PRIVATE ${NUMA_LIB})


