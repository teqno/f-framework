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
        break;
    }

    return 0;
}