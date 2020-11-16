#include "catch2/catch.hpp"
#include <Eigen/Dense>
#include "propagation.h"

using namespace Catch::literals;
TEST_CASE("Preactivation function", "[preactivation]")
{
  Eigen::VectorXd x(3);
  Eigen::VectorXd w(3);

  x << 1, 2, 3;
  w << 4, 5, 6;

  double b = 7;

  double actual = preactivation(x, w, b);
  double expected = 1 * 4 + 2 * 5 + 3 * 6 + 7;

  REQUIRE(actual == expected);
}

TEST_CASE("Activation function", "[activation]")
{
  double a = 0.5;

  SECTION("Linear")
  {
    double actual = activation(a, ACTIVATION_FUNCTION::LINEAR);
    double expected = a;

    REQUIRE(actual == expected);
  }

  SECTION("Sigmoid")
  {
    double actual = activation(a, ACTIVATION_FUNCTION::SIGMOID);
    Catch::Detail::Approx expected = 0.3775406688_a;

    REQUIRE(actual == expected);
  }

  SECTION("Sigmoid")
  {
    double actual = activation(a, ACTIVATION_FUNCTION::TANH);
    Catch::Detail::Approx expected = 0.4621171572_a;

    REQUIRE(actual == expected);
  }
}