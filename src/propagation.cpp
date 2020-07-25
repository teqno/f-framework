#include <vector>

#include "propagation.h"
#include "activations.h"

double preactivation(std::vector<double> &x, std::vector<double> &w, double &b)
{
    auto lambda = [&](double x, double w) { return x * w; };

    double xw = 0;

    for (int i = 0; i < x.size(); i++)
    {
        xw += lambda(x[i], w[i]);
    }

    return xw + b;
}

double activation(double &a, ACTIVATION_FUNCTION activation_function)
{
    switch (activation_function)
    {
    case ACTIVATION_FUNCTION::SIGMOID:
        return sigmoid(a);
    default:
        break;
    }

    return 0;
}