#pragma once
#include <vector>
#include "Eigen/Dense"

enum ACTIVATION_FUNCTION
{
    LINEAR,
    SIGMOID,
    TANH
};

double preactivation(Eigen::VectorXd &x, Eigen::VectorXd &w, double b);

double activation(double a, ACTIVATION_FUNCTION activation_function);