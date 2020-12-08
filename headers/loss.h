#pragma once

#include <vector>
#include <math.h>
#include "Eigen/Dense"

double mse(const Eigen::MatrixXd &y_hat, const Eigen::MatrixXd &y);

double cross_entropy_loss(double a, double y);

Eigen::MatrixXd mse_prime(Eigen::MatrixXd y_hat, Eigen::MatrixXd y);

Eigen::VectorXd cross_entropy_loss_derivative(Eigen::VectorXd a, Eigen::VectorXd y);