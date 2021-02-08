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

Eigen::VectorXd preactivation(const Eigen::MatrixXd &w, const Eigen::VectorXd &activations, const Eigen::VectorXd &b);

Eigen::VectorXd activation(const Eigen::VectorXd &z, ACTIVATION_FUNCTION activation_function);

Eigen::VectorXd activation_prime(const Eigen::VectorXd &z, ACTIVATION_FUNCTION activation_function);