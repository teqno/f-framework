set(CATCH_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/libs/Catch2/single_include)

file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

find_package(Catch2 REQUIRED PATHS "${PROJECT_SOURCE_DIR}/libs/Catch2")

include(ParseAndAddCatchTests)

add_library(testsMainLib STATIC testsMain.cpp)
target_include_directories(testsMainLib PRIVATE ${CATCH_INCLUDE_DIR})

foreach(testSrc ${TEST_SRCS})
    get_filename_component(testName ${testSrc} NAME_WE)
    
    add_executable(${testName} ${testSrc})

    target_link_libraries(${testName} f-framework-lib testsMainLib)   
    target_include_directories(${testName} PRIVATE ${CATCH_INCLUDE_DIR})

    ParseAndAddCatchTests(${testName})
endforeach(testSrc)

