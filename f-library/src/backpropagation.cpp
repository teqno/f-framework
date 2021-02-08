#include "backpropagation.h"
#include "loss.h"

Eigen::VectorXd calculate_da(const Eigen::VectorXd &activations, const Eigen::VectorXd &y)
{
    return mse_prime(activations, y);
};