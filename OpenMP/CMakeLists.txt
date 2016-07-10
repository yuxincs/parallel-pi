cmake_minimum_required(VERSION 2.8)
project(OpenMP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(APPLE)
    set(CMAKE_CXX_COMPILER "clang-omp")
endif()

set(SOURCE_FILES main.cpp)
add_executable(OpenMP ${SOURCE_FILES})