#pragma once

#include <vector>
#include "Eigen/Dense"


namespace DataTypes
{
    struct NeuronParameters
    {
        Eigen::VectorXd w;
        double b;
    };

    struct LayerParameters
    {
        Eigen::MatrixXd w;
        Eigen::VectorXd b;
    };

    struct LayerCacheResult
    {
        Eigen::VectorXd preactivations;
        Eigen::VectorXd activations;
    };

    struct NetworkCache
    {
        std::vector<Eigen::VectorXd> preactivations;
        std::vector<Eigen::VectorXd> activations;
    };

    struct Deltas
    {
        std::vector<Eigen::MatrixXd> dw;
        std::vector<Eigen::VectorXd> db;
    };
}