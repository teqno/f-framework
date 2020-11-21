#pragma once

#include "Eigen/Dense"

double sigmoid(double a);

Eigen::VectorXd sigmoid(Eigen::VectorXd a);

Eigen::VectorXd sigmoid_prime(Eigen::VectorXd z);

Eigen::VectorXd tanh_prime(Eigen::VectorXd z);