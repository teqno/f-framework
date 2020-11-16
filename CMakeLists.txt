cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(f-framework VERSION 1.0)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
#add_subdirectory(libs/Catch2)