#include <math.h>

#include "activations.h"

double sigmoid(double a)
{
    return 1 / (1 + exp(a));
}

Eigen::MatrixXd sigmoid(Eigen::MatrixXd a)
{
    return 1 / (1 + a.array().exp());
}

Eigen::MatrixXd sigmoid_prime(Eigen::MatrixXd z)
{
    return sigmoid(z).array() * (-sigmoid(z).array() + 1);
}

Eigen::MatrixXd tanh_prime(Eigen::MatrixXd z)
{
    return 1 - z.array().tanh().pow(2);
}