add_library(f-framework-lib STATIC
    activations.cpp
    core.cpp
    Layer.cpp
    loss.cpp
    Network.cpp
    Neuron.cpp
    propagation.cpp
    utils.cpp
)

set_property(TARGET f-framework-lib PROPERTY CXX_STANDARD 17)

target_include_directories(f-framework-lib
    PUBLIC ${PROJECT_SOURCE_DIR}/headers
    PUBLIC ${PROJECT_SOURCE_DIR}/libs
)

target_compile_options(f-framework-lib PRIVATE -Werror -Wall -Wextra)

set_target_properties(f-framework-lib PROPERTIES UNITY_BUILD ON)
