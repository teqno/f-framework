#pragma once

#include "Eigen/Dense"

Eigen::VectorXd sigmoid(const Eigen::VectorXd &z);

Eigen::VectorXd relu(const Eigen::VectorXd &z);

Eigen::VectorXd sigmoid_prime(const Eigen::VectorXd &z);

Eigen::VectorXd tanh_prime(const Eigen::VectorXd &z);

Eigen::VectorXd relu_prime(const Eigen::VectorXd &z);