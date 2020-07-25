#pragma once
#include <vector>

enum ACTIVATION_FUNCTION
{
    LINEAR,
    SIGMOID,
    TANH
};

double preactivation(std::vector<double> &x, std::vector<double> &w, double &b);

double activation(double &a, ACTIVATION_FUNCTION activation_function);