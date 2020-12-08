#include <math.h>

#include "activations.h"

double sigmoid(double a)
{
    return 1.0 / (1.0 + exp(a));
}

Eigen::MatrixXd sigmoid(Eigen::MatrixXd a)
{
    return 1.0 / (1.0 + a.array().exp());
}

Eigen::MatrixXd sigmoid_prime(Eigen::MatrixXd z)
{
    return sigmoid(z).array() * (Eigen::MatrixXd::Ones(z.rows(), z.cols()).array() - sigmoid(z).array());
}

Eigen::MatrixXd tanh_prime(Eigen::MatrixXd z)
{
    return 1.0 - z.array().tanh().pow(2);
}