cmake_minimum_required(VERSION 2.6)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp  -pthread")

set (SOURCES main.cpp)


FIND_PACKAGE( Boost REQUIRED serialization system   )
FIND_PACKAGE(OpenMP)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

add_executable(randforest ${SOURCES})
target_link_libraries(randforest randforestlib ${Boost_LIBRARIES})
