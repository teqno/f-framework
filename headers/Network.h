#pragma once

#include "Layer.h"
#include "loss.h"
#include "Cache.h"
#include <optional>
#include <functional>

class Network
{
private:
    std::vector<Layer *> layers;
    std::vector<Eigen::VectorXd> activations;
    std::vector<Eigen::VectorXd> preactivations;
    std::map<std::string, std::vector<Eigen::MatrixXd>> retainedGradient;

public:
    Network(const std::vector<Layer *> &layers, std::optional<unsigned int> random_seed = std::nullopt);

    std::vector<Layer *> getLayers();

    Eigen::VectorXd forward_prop(const Eigen::VectorXd &input);

    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> back_prop(const Eigen::MatrixXd &y);

    void updateParameters(const std::vector<Eigen::MatrixXd> &dw, const std::vector<Eigen::MatrixXd> &db, double alpha, double alphaMomentum);

    double calc_cost(const Eigen::VectorXd &x, const Eigen::VectorXd &y);

    Eigen::VectorXd train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y, int epochs, double alpha, double alphaMomentum = 0);
};