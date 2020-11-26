#include "catch2/catch.hpp"
#include <Eigen/Dense>
#include "loss.h"

using namespace Catch::literals;
// TEST_CASE("Mean squared error function", "[mse]")
// {
//   Eigen::VectorXd x(3);
//   Eigen::VectorXd y(3);

//   x << 1, 2, 3;
//   y << 4, 5, 6;

//   double actual = mse(x, y);
//   double expected = std::pow(1 - 4, 2) + std::pow(2 - 5, 2) + std::pow(3 - 6, 2);

//   REQUIRE(actual == Approx(expected));
// }

// TEST_CASE("Cross entropy loss function", "[cross_entropy_loss]")
// {
//   double x = 0.5;
//   double y = 0.1;

//   double actual = cross_entropy_loss(x, y);
//   double expected = 0.6931471806;

//   REQUIRE(actual == Approx(expected));
// }