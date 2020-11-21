#include <math.h>

#include "activations.h"

double sigmoid(double a)
{
    return 1 / (1 + exp(a));
}

Eigen::VectorXd sigmoid(Eigen::VectorXd a)
{
    return 1 / (1 + a.array().exp());
}

Eigen::VectorXd sigmoid_prime(Eigen::VectorXd z)
{
    return sigmoid(z).array() * (-sigmoid(z).array() + 1);
}

Eigen::VectorXd tanh_prime(Eigen::VectorXd z)
{
    return 4 / ((-z.array()).exp() + z.array().exp()).pow(2);
}