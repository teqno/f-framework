add_library(f-framework-lib STATIC
    activations.cpp
    core.cpp
    Layer.cpp
    loss.cpp
    Network.cpp
    Neuron.cpp
    propagation.cpp
    random.cpp
)

set_target_properties(f-framework-lib PROPERTIES PUBLIC_HEADER include/f-framework.h)

target_include_directories(f-framework-lib
    PUBLIC ${PROJECT_SOURCE_DIR}/headers
    PUBLIC ${PROJECT_SOURCE_DIR}/libs
)

set_target_properties(f-framework-lib PROPERTIES PUBLIC_HEADER include/f-framework.h)

include(GNUInstallDirs)

install(TARGETS f-framework-lib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)