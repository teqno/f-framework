#include "loss.h"
#include <iostream>

double mse(const Eigen::VectorXd &y_hat, const Eigen::VectorXd &y)
{
    Eigen::VectorXd squaredDifference = (y - y_hat).array().pow(2);
    return squaredDifference.sum() / 2.0;
}

Eigen::VectorXd mse_prime(const Eigen::VectorXd &y_hat, const Eigen::VectorXd &y)
{
    return y - y_hat;
}
