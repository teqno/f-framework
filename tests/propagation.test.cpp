#include "catch2/catch.hpp"
#include <Eigen/Dense>
#include "propagation.h"
#include "utils.h"

using namespace Catch::literals;
TEST_CASE("Preactivation function", "[preactivation]")
{
    Eigen::VectorXd x(3);
    Eigen::MatrixXd w(3, 3);

    x << 1, 2, 3;
    w << 1, 2, 3, 1, 2, 3, 1, 2, 3;

    Eigen::VectorXd b(3);
    b << 5, 6, 7;

    Eigen::VectorXd actual = preactivation(w, x, b);
    Eigen::VectorXd expected(3);
    expected << 14 + 5, 14 + 6, 14 + 7;

    CHECK(actual.isApprox(expected));
}

TEST_CASE("Activation function", "[activation]")
{
    Eigen::VectorXd z(3);
    z << -10, 0, 10;

    SECTION("Linear")
    {
        Eigen::VectorXd actual = activation(z, ACTIVATION_FUNCTION::LINEAR);
        Eigen::VectorXd expected(3);
        expected << -10, 0, 10;

        INFO("Actual: " << actual);
        INFO("Expected: " << expected);

        CHECK(actual.isApprox(expected));
    }

    // SECTION("Sigmoid")
    // {
    //     Eigen::VectorXd actual = activation(z, ACTIVATION_FUNCTION::SIGMOID);
    //     Eigen::VectorXd expected(3);
    //     expected << 0.00005, 0.5, 0.99995;

    //     INFO("Actual: " << actual);
    //     INFO("Expected: " << expected);

    //     CHECK(actual.isApprox(expected, 0.1));
    // }

    SECTION("Tanh")
    {
        Eigen::VectorXd actual = activation(z, ACTIVATION_FUNCTION::TANH);
        Eigen::VectorXd expected(3);
        expected << -0.99999999587, 0, 0.99999999587;

        INFO("Actual: " << actual);
        INFO("Expected: " << expected);

        CHECK(actual.isApprox(expected, 0.00000000001));
    }

    SECTION("Relu")
    {
        Eigen::VectorXd actual = activation(z, ACTIVATION_FUNCTION::RELU);
        Eigen::VectorXd expected(3);
        expected << 0, 0, 10;

        INFO("Actual: " << actual);
        INFO("Expected: " << expected);

        CHECK(actual.isApprox(expected));
    }
}