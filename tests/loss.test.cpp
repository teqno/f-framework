#include "catch2/catch.hpp"
#include <Eigen/Dense>
#include "loss.h"
#include "utils.h"

using namespace Catch::literals;
TEST_CASE("Mean squared error function", "[mse]")
{
    Eigen::VectorXd y_hat(3);
    Eigen::VectorXd y(3);

    y_hat << 1, 2, 3;
    y << 6, 5, 4;

    double actual = mse(y_hat, y);
    double expected = (25.0 + 9.0 + 1.0) / 2.0;

    CHECK(actual == Approx(expected));
}

TEST_CASE("Mean squared prime error function", "[mse_prime]")
{
    Eigen::VectorXd y_hat(3);
    Eigen::VectorXd y(3);

    y_hat << 1, 2, 3;
    y << 6, 5, 4;

    Eigen::VectorXd actual = mse_prime(y_hat, y);
    Eigen::VectorXd expected(3);
    expected << 5, 3, 1;

    CHECK(actual.isApprox(expected));
}