#include <math.h>
#include <iostream>

#include "activations.h"

Eigen::VectorXd sigmoid(const Eigen::VectorXd &z)
{
    return z.array() / (1.0 + z.array().abs());
}

Eigen::VectorXd relu(const Eigen::VectorXd &z)
{
    return z.cwiseMax(0);
}

Eigen::VectorXd sigmoid_prime(const Eigen::VectorXd &z)
{
    return 1.0 / (z.array().abs() + 1.0).pow(2);
}

Eigen::VectorXd tanh_prime(const Eigen::VectorXd &z)
{
    return 1.0 / z.array().cosh().pow(2);
}

Eigen::VectorXd relu_prime(const Eigen::VectorXd &z)
{
    Eigen::VectorXd updatedZ(z.size());

    for (int i = 0; i < z.size(); i++)
    {
        updatedZ(i) = z(i) > 0 ? 1 : 0;
    }

    return updatedZ;
}