#include <vector>
#include <math.h>
#include "Eigen/Dense"

double mse(Eigen::VectorXd &x, Eigen::VectorXd &y);

double cross_entropy_loss(double a, double y);