cmake_minimum_required(VERSION 2.6)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

IF(WIN32)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
ELSE()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11  -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2 -std=c++11 -fopenmp  -pthread")
ENDIF()
set(Boost_USE_STATIC_LIBS ON)

file(GLOB_RECURSE HEADERS "../include/*.h")
file(GLOB_RECURSE SOURCES "*.cpp")

include_directories(../include/
    ${BOOST_INCLUDEDIR})
add_library(randforestlib STATIC ${SOURCES} ${HEADERS})
