#include "utils.h"

bool vectorized_equal_approx(Eigen::VectorXd a, Eigen::VectorXd b)
{
    return ((a.array() - b.array()).array().abs() < (std::numeric_limits<float>::epsilon() * 1000)).all();
}

bool vectorized_equal_approx(Eigen::MatrixXd a, Eigen::MatrixXd b)
{
    return ((a.array() - b.array()).array().abs() < (std::numeric_limits<float>::epsilon() * 1000)).all();
}