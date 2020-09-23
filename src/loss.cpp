#include "loss.h"

double mse(std::vector<double> &x, std::vector<double> &y)
{
    double sum_of_squares = 0;
    for (int i = 0; i < x.size(); i++)
    {
        sum_of_squares += pow(x.at(i) - y.at(i), 2);
    }
    return 1.0 / (2 * x.size()) * sum_of_squares;
}

double cross_entropy_loss(double a, double y)
{
    return -(y * std::log(a) + (1 - y) * std::log(1 - a));
}
