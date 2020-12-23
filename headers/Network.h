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
    std::map<std::string, std::vector<Eigen::MatrixXd>> retainedGradient;

public:
    Network(const std::vector<Layer *> &layers, std::optional<unsigned int> random_seed = std::nullopt);

    std::vector<Layer *> getLayers();

    Eigen::VectorXd forward_prop(const Eigen::VectorXd &input);

    DataTypes::Deltas back_prop(const Eigen::VectorXd &y);

    void updateParameters(const std::vector<Eigen::MatrixXd> &dw, const std::vector<Eigen::VectorXd> &db, double alpha);

    double calc_cost(const Eigen::VectorXd &x, const Eigen::VectorXd &y);

    Eigen::VectorXd train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y, int epochs, double alpha);
};