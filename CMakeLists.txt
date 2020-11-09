cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(f-framework VERSION 1.0)

add_library(f-framework STATIC
    src/activations.cpp
    src/core.cpp
    src/Layer.cpp
    src/loss.cpp
    src/Network.cpp
    src/Neuron.cpp
    src/propagation.cpp
    src/random.cpp
)

set_target_properties(f-framework PROPERTIES PUBLIC_HEADER include/f-framework.h)

target_include_directories(f-framework
    PRIVATE src
    PRIVATE headers
    PRIVATE libs
)

include(GNUInstallDirs)

install(TARGETS f-framework
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)