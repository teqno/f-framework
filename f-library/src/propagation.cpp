#include <math.h>

#include "propagation.h"
#include "activations.h"

Eigen::VectorXd preactivation(const Eigen::MatrixXd &w, const Eigen::VectorXd &activations, const Eigen::VectorXd &b)
{
    return (w * activations).array() + b.array();
}

Eigen::VectorXd activation(const Eigen::VectorXd &z, ACTIVATION_FUNCTION activation_function)
{
    switch (activation_function)
    {
    case ACTIVATION_FUNCTION::LINEAR:
        return z;
    case ACTIVATION_FUNCTION::SIGMOID:
        return sigmoid(z);
    case ACTIVATION_FUNCTION::TANH:
        return z.array().tanh();
    case ACTIVATION_FUNCTION::RELU:
        return relu(z);
    default:
        throw "Unrecognized activation function name!";
        break;
    }
}

Eigen::VectorXd activation_prime(const Eigen::VectorXd &z, ACTIVATION_FUNCTION activation_function)
{
    switch (activation_function)
    {
    case ACTIVATION_FUNCTION::LINEAR:
        return Eigen::VectorXd::Ones(z.size());
    case ACTIVATION_FUNCTION::SIGMOID:
        return sigmoid_prime(z);
    case ACTIVATION_FUNCTION::TANH:
        return tanh_prime(z);
    case ACTIVATION_FUNCTION::RELU:
        return relu_prime(z);
    default:
        throw "Unrecognized activation function name!";
        break;
    }
}