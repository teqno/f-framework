cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(f-framework-demo VERSION 1.0)

include_directories(
    PUBLIC ${f-framework-demo_SOURCE_DIR}/../libs
    PUBLIC ${f-framework-demo_SOURCE_DIR}/../headers
)

add_executable(f-framework-demo test.cpp)

link_libraries(${f-framework-demo_SOURCE_DIR}/../build/libf-framework.a)