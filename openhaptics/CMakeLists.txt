cmake_minimum_required(VERSION 3.0.0)
project(youngy VERSION 0.1.0)

find_package (Threads)

add_executable(youngy main.cpp)

target_link_libraries(youngy HD HDU)
target_link_libraries (youngy ${CMAKE_THREAD_LIBS_INIT})