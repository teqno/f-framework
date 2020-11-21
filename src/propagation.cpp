#include <math.h>

#include "propagation.h"
#include "activations.h"

double preactivation(Eigen::VectorXd &x, Eigen::VectorXd &w, double b)
{
    return x.dot(w.transpose()) + b;
}

double activation(double a, ACTIVATION_FUNCTION activation_function)
{
    switch (activation_function)
    {
    case ACTIVATION_FUNCTION::LINEAR:
        return a;
    case ACTIVATION_FUNCTION::SIGMOID:
        return sigmoid(a);
    case ACTIVATION_FUNCTION::TANH:
        return tanh(a);
    default:
        throw "Unrecognized activation function name!";
        break;
    }
}

Eigen::VectorXd activation_prime(Eigen::VectorXd z, ACTIVATION_FUNCTION activation_function)
{
    switch (activation_function)
    {
    case ACTIVATION_FUNCTION::LINEAR:
        return Eigen::VectorXd::Ones(z.size());
    case ACTIVATION_FUNCTION::SIGMOID:
        return sigmoid_prime(z);
    case ACTIVATION_FUNCTION::TANH:
        return tanh_prime(z);
    default:
        throw "Unrecognized activation function name!";
        break;
    }
}