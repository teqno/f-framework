#pragma once

#include "Eigen/Dense"

double sigmoid(double a);

Eigen::MatrixXd sigmoid(Eigen::MatrixXd a);

Eigen::MatrixXd sigmoid_prime(Eigen::MatrixXd z);

Eigen::MatrixXd tanh_prime(Eigen::MatrixXd z);