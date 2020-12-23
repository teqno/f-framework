// #include "catch2/catch.hpp"
// #include <Eigen/Dense>
// #include "propagation.h"
// #include "utils.h"

// using namespace Catch::literals;
// TEST_CASE("Preactivation function", "[preactivation]")
// {
//     Eigen::VectorXd x(3);
//     Eigen::VectorXd w(3);

//     x << 1, 2, 3;
//     w << 4, 5, 6;

//     double b = 7;

//     double actual = preactivation(x, w, b);
//     double expected = 1 * 4 + 2 * 5 + 3 * 6 + 7;

//     CHECK(actual == expected);
// }

// TEST_CASE("Activation function", "[activation]")
// {
//     Eigen::VectorXd z(3);
//     z << -10, 0, 10;

//     SECTION("Linear")
//     {
//         Eigen::VectorXd actual = activation(z, ACTIVATION_FUNCTION::LINEAR);
//         Eigen::VectorXd expected(3);
//         expected << -10, 0, 10;

//         INFO("Actual: " << actual);
//         INFO("Expected: " << expected);

//         CHECK(vectorized_equal_approx(actual, expected));
//     }

//     SECTION("Sigmoid")
//     {
//         Eigen::VectorXd actual = activation(z, ACTIVATION_FUNCTION::SIGMOID);
//         Eigen::VectorXd expected(3);
//         expected << 0.00005, 0.5, 0.99995;

//         INFO("Epsilon: " << std::numeric_limits<float>::epsilon() * 1000);
//         INFO("Actual: " << actual);
//         INFO("Expected: " << expected);

//         CHECK(vectorized_equal_approx(actual, expected));
//     }

//     SECTION("Tanh")
//     {
//         Eigen::VectorXd actual = activation(z, ACTIVATION_FUNCTION::TANH);
//         Eigen::VectorXd expected(3);
//         expected << -0.99999999587, 0, 0.99999999587;

//         INFO("Epsilon: " << std::numeric_limits<float>::epsilon() * 1000);
//         INFO("Actual: " << actual);
//         INFO("Expected: " << expected);

//         CHECK(vectorized_equal_approx(actual, expected));
//     }

//     SECTION("Relu")
//     {
//         Eigen::VectorXd actual = activation(z, ACTIVATION_FUNCTION::RELU);
//         Eigen::VectorXd expected(3);
//         expected << 0, 0, 10;

//         INFO("Actual: " << actual);
//         INFO("Expected: " << expected);

//         CHECK(vectorized_equal_approx(actual, expected));
//     }
// }