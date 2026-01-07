cmake_minimum_required(VERSION 3.10)
project(FHE16Example CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 공통 헤더 경로와 라이브러리 경로를 변수로 묶기
set(FHE16_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/FHE_TEST/include_FHE16/FHE16/include
    ${CMAKE_SOURCE_DIR}/FHE_TEST/include_FHE16/FHE16/include/soAPI
    ${CMAKE_SOURCE_DIR}/FHE_TEST/include_FHE16/FHE16/include/math
    ${CMAKE_SOURCE_DIR}/FHE_TEST/include_FHE16/FHE16/include/lwe
    ${CMAKE_SOURCE_DIR}/FHE_TEST/include_FHE16/FHE16/include/thread
    ${CMAKE_SOURCE_DIR}/FHE_TEST/include_FHE16/FHE16_Module/include/ntt/
    ${CMAKE_SOURCE_DIR}/FHE_TEST/include_FHE16/FHE16_Module/include/bootstrapping/
)

set(FHE16_LIBS
    ${CMAKE_SOURCE_DIR}/FHE_TEST/lib/libFHE16.so
    ${CMAKE_SOURCE_DIR}/FHE_TEST/lib/libFHE16_Module.so
)

find_library(NUMA_LIB numa REQUIRED)

# === main 실행파일 ===
add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_definitions(-DSOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_include_directories(main PRIVATE ${FHE16_INCLUDE_DIRS})
target_link_libraries(main PRIVATE ${FHE16_LIBS} ${NUMA_LIB})

# === test 실행파일 추가 ===
add_executable(test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
target_include_directories(test PRIVATE ${FHE16_INCLUDE_DIRS})
target_link_libraries(test PRIVATE ${FHE16_LIBS} ${NUMA_LIB})

# === hybrid 실행파일 추가 (마지막 N개 레이어만 FHE) ===
add_executable(hybrid ${CMAKE_CURRENT_SOURCE_DIR}/hybrid_inference.cpp)
target_include_directories(hybrid PRIVATE ${FHE16_INCLUDE_DIRS})
target_link_libraries(hybrid PRIVATE ${FHE16_LIBS} ${NUMA_LIB})

