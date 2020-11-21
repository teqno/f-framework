#pragma once

#include <vector>
#include "Eigen/Dense"

struct Cache
{
    Eigen::MatrixXd* activations;
    Eigen::MatrixXd* zs;

    Cache(int dim_a1, int dim_a2, int dim_z1, int dim_z2)
    {
        activations = new Eigen::MatrixXd(dim_a1, dim_a2);
        zs = new Eigen::MatrixXd(dim_z1, dim_z2);
    }
};