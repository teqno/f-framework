#pragma once

#include <vector>
#include <math.h>
#include "Eigen/Dense"

double mse(Eigen::VectorXd &x, Eigen::VectorXd &y);

double cross_entropy_loss(double a, double y);

Eigen::VectorXd mse_prime(Eigen::VectorXd a, Eigen::VectorXd y);

Eigen::VectorXd cross_entropy_loss_derivative(Eigen::VectorXd a, Eigen::VectorXd y);