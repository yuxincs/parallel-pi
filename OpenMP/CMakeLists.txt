cmake_minimum_required(VERSION 2.8)
project(OpenMP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")

if(APPLE)
    set(CMAKE_CXX_COMPILER "gcc-6")
endif()

set(SOURCE_FILES main.cpp)
add_executable(OpenMP ${SOURCE_FILES})