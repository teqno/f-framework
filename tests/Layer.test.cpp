#include "catch2/catch.hpp"
#include <Eigen/Dense>
#include "Layer.h"

using namespace Catch::literals;
TEST_CASE("Layer", "[Layer]")
{
    std::srand(1); // makes sure that same random values are generated each test run
    int layer_size = 3;
    int input_size = 5;

    SECTION("Layer parameters dimensions")
    {
        Layer layer = Layer(layer_size, input_size, ACTIVATION_FUNCTION::LINEAR);

        DataTypes::LayerHyperParameters params = layer.getParams();

        
    }
}