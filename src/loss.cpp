#include "loss.h"

double mse(Eigen::VectorXd &x, Eigen::VectorXd &y)
{
    Eigen::VectorXd squaredDifference = (x - y).array().pow(2);
    return squaredDifference.sum();
}

double cross_entropy_loss(double a, double y)
{
    return -(y * std::log(a) + (1 - y) * std::log(1 - a));
}

Eigen::VectorXd mse_prime(Eigen::VectorXd a, Eigen::VectorXd y)
{
    return 2 * (a - y);
}

Eigen::VectorXd cross_entropy_loss_prime(Eigen::VectorXd a, Eigen::VectorXd y)
{
    return -(y.array() / a.array()) + (-y.array() + 1) / (-a.array() + 1);
}
