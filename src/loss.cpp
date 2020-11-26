#include "loss.h"

double mse(Eigen::MatrixXd &y_hat, Eigen::MatrixXd &y)
{
    Eigen::MatrixXd squaredDifference = (y_hat - y).array().pow(2);
    return squaredDifference.sum();
}

double cross_entropy_loss(double a, double y)
{
    return -(y * std::log(a) + (1 - y) * std::log(1 - a));
}

Eigen::MatrixXd mse_prime(Eigen::MatrixXd y_hat, Eigen::MatrixXd y)
{
    return 2 * (y_hat - y);
}

Eigen::VectorXd cross_entropy_loss_prime(Eigen::VectorXd a, Eigen::VectorXd y)
{
    return -(y.array() / a.array()) + (-y.array() + 1) / (-a.array() + 1);
}
