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

Eigen::MatrixXd activation(Eigen::MatrixXd z, ACTIVATION_FUNCTION activation_function);

Eigen::MatrixXd activation_prime(Eigen::MatrixXd z, ACTIVATION_FUNCTION activation_function);