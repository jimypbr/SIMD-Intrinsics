cmake_minimum_required(VERSION 3.1)
project(vec_intrin)

#set(CMAKE_CXX_COMPILER g++)
#set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native -std=c++11 ")

set(SOURCE_FILES main.cpp)
add_executable(vec_intrin ${SOURCE_FILES})