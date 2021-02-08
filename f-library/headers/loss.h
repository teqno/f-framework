#pragma once

#include <vector>
#include <math.h>
#include "Eigen/Dense"

double mse(const Eigen::VectorXd &y_hat, const Eigen::VectorXd &y);

Eigen::VectorXd mse_prime(const Eigen::VectorXd &y_hat, const Eigen::VectorXd &y);
