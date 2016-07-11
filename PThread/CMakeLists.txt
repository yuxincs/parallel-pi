cmake_minimum_required(VERSION 2.8)
project(PThread)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(PThread ${SOURCE_FILES})