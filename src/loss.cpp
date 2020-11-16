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
