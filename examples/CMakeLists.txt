cmake_minimum_required(VERSION 3.10)

option(Boost_DEBUG "Set to ON to enable debug output from FindBoost." OFF)
option(Boost_DETAILED_FAILURE_MSG "Set to ON to get detailed information" OFF)

set(Boost_NO_SYSTEM_PATHS ON)
set(Boost_USE_STATIC_LIBS TRUE)
set(Boost_USE_MULTITHREADED TRUE)
set(Boost_LIB_PREFIX lib)
set(Boost_ARCHITECTURE -x64)
set(Boost_COMPILER -mgw8)
set(BOOST_ROOT ${PROJECT_SOURCE_DIR}/libs/boost)
set(BOOST_LIBRARYDIR ${PROJECT_SOURCE_DIR}/libs/boost/stage/lib)
set(BOOST_INCLUDEDIR ${PROJECT_SOURCE_DIR}/libs/boost)

find_package(Boost REQUIRED COMPONENTS filesystem system iostreams)

add_executable(demo demo.cpp)

set_property(TARGET demo PROPERTY CXX_STANDARD 17)

target_link_libraries(demo f-framework-lib ${Boost_LIBRARIES})

target_include_directories(demo
    PRIVATE ${PROJECT_SOURCE_DIR}/libs/gnuplot-iostream
    PRIVATE ${Boost_INCLUDE_DIR}
)
