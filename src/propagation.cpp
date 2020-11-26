#include <math.h>

#include "propagation.h"
#include "activations.h"

double preactivation(Eigen::VectorXd &x, Eigen::VectorXd &w, double b)
{
    return x.dot(w.transpose()) + b;
}

Eigen::MatrixXd activation(Eigen::MatrixXd z, ACTIVATION_FUNCTION activation_function)
{
    switch (activation_function)
    {
    case ACTIVATION_FUNCTION::LINEAR:
        return z;
    case ACTIVATION_FUNCTION::SIGMOID:
        return sigmoid(z);
    case ACTIVATION_FUNCTION::TANH:
        return z.array().tanh();
    default:
        throw "Unrecognized activation function name!";
        break;
    }
}

Eigen::MatrixXd activation_prime(Eigen::MatrixXd z, ACTIVATION_FUNCTION activation_function)
{
    switch (activation_function)
    {
    case ACTIVATION_FUNCTION::LINEAR:
        return Eigen::MatrixXd::Ones(z.rows(), z.cols());
    case ACTIVATION_FUNCTION::SIGMOID:
        return sigmoid_prime(z);
    case ACTIVATION_FUNCTION::TANH:
        return tanh_prime(z);
    default:
        throw "Unrecognized activation function name!";
        break;
    }
}