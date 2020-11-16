#include <vector>
#include <math.h>
#include "Eigen/Dense"

double mse(std::vector<double> &x, std::vector<double> &y);

double cross_entropy_loss(double a, double y);