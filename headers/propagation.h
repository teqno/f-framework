#pragma once
#include <vector>
#include "Eigen/Dense"

enum ACTIVATION_FUNCTION
{
    LINEAR,
    SIGMOID,
    TANH,
    RELU
};

double preactivation(const Eigen::VectorXd &x, const Eigen::VectorXd &w, double b);

Eigen::VectorXd activation(const Eigen::VectorXd &z, ACTIVATION_FUNCTION activation_function);

Eigen::VectorXd activation_prime(const Eigen::VectorXd &z, ACTIVATION_FUNCTION activation_function);