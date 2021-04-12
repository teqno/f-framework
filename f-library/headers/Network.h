#pragma once

#include "Layer.h"
#include "loss.h"
#include <optional>
#include <functional>

class Network
{
private:
    std::vector<Layer *> layers;
    DataTypes::NetworkCache cache;

public:
    Network(const std::vector<Layer *> &layers, std::optional<unsigned int> random_seed = std::nullopt);

    std::vector<Layer *> getLayers();

    Eigen::VectorXd forward_prop(const Eigen::VectorXd &input);

    DataTypes::Deltas back_prop(const Eigen::VectorXd &y);

    void updateParameters(const std::vector<Eigen::MatrixXd> &dw, const std::vector<Eigen::VectorXd> &db, double alpha, std::optional<DataTypes::Deltas> prev_deltas, std::optional<double> momentum);

    double calc_cost(const Eigen::VectorXd &x, const Eigen::VectorXd &y);

    Eigen::VectorXd train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y, int epochs, double alpha, double momentum);
};