#include "loss.h"
#include <iostream>

double mse(const Eigen::VectorXd &y_hat, const Eigen::VectorXd &y)
{
    Eigen::VectorXd squaredDifference = (y - y_hat).array().pow(2);
    return squaredDifference.sum();
}

Eigen::VectorXd mse_prime(const Eigen::VectorXd &y_hat, const Eigen::VectorXd &y)
{
    return 2.0 * (y - y_hat);
}
