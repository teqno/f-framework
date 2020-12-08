#include "loss.h"
#include <iostream>

double mse(const Eigen::MatrixXd &y_hat, const Eigen::MatrixXd &y)
{
    Eigen::MatrixXd squaredDifference = (y - y_hat).array().pow(2);
    return squaredDifference.sum();
}

double cross_entropy_loss(double a, double y)
{
    return -(y * std::log(a) + (1.0 - y) * std::log(1.0 - a));
}

Eigen::MatrixXd mse_prime(Eigen::MatrixXd y_hat, Eigen::MatrixXd y)
{
    return 2.0 * (y - y_hat);
}

Eigen::VectorXd cross_entropy_loss_prime(Eigen::VectorXd a, Eigen::VectorXd y)
{
    return -(y.array() / a.array()) + (-y.array() + 1) / (-a.array() + 1);
}
