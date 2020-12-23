// #include "catch2/catch.hpp"
// #include <Eigen/Dense>
// #include "Neuron.h"

// using namespace Catch::literals;
// TEST_CASE("Neuron", "[Neuron]")
// {
//     std::srand(1); // makes sure that same random values are generated each test run
//     int input_size = 5;

//     Neuron neuron = Neuron(input_size);

//     SECTION("Parameters dimensions")
//     {
//         DataTypes::HyperParameters parameters = neuron.getParameters();

//         int actual = parameters.w.rows();
//         int expected = 5;

//         CHECK(actual == expected);
//     }

//     SECTION("W parameter value")
//     {
//         DataTypes::HyperParameters parameters = neuron.getParameters();

//         Eigen::VectorXd actual = parameters.w;
//         Eigen::VectorXd expected(input_size);
//         expected << 0.001251,
//                     0.563585,
//                     0.193304,
//                     0.808741,
//                     0.585009;

//         INFO("Actual: " << actual);
//         INFO("Expected: " << expected);

//         REQUIRE(actual.isApprox(expected, 0.000001));
//     }

//     SECTION("B parameter value")
//     {
//         DataTypes::HyperParameters parameters = neuron.getParameters();

//         double actual = parameters.b;
//         double expected = 0;
        
//         INFO("Actual: " << actual);
//         INFO("Expected: " << expected);

//         REQUIRE(actual == Approx(expected));
//     }

//     SECTION("W parameter modify value")
//     {
//         Eigen::VectorXd newW(input_size);
//         newW << 0.1,
//                 0.2,
//                 0.3,
//                 0.4,
//                 0.5;
//         neuron.setW(newW);

//         Eigen::VectorXd actual = neuron.getW();
//         Eigen::VectorXd expected = newW;

//         INFO("Actual: " << actual);
//         INFO("Expected: " << expected);

//         REQUIRE(actual.isApprox(expected, 0.000001));
//     }

//     SECTION("B parameter modify value")
//     {
//         double newB = 0.5;
//         neuron.setB(newB);

//         double actual = neuron.getB();
//         double expected = newB;

//         INFO("Actual: " << actual);
//         INFO("Expected: " << expected);

//         REQUIRE(actual == Approx(expected));
//     }
// }