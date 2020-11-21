#pragma once

#include "Layer.h"
#include "loss.h"
#include "Cache.h"

class Network
{
private:
    std::vector<Layer *> layers;
    std::vector<Eigen::VectorXd> activations;
    std::vector<Eigen::VectorXd> preactivations;
    // Cache* cache;

public:
    Network(std::vector<Layer *> &layers);
    std::vector<Layer *> getLayers();
    Eigen::VectorXd forward_prop(Eigen::VectorXd &input);
    double calc_cost(Eigen::MatrixXd &x, Eigen::MatrixXd &y);
    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> back_prop(Eigen::VectorXd y);
    void updateParameters(std::vector<Eigen::MatrixXd> dw, std::vector<Eigen::MatrixXd> db, double alpha);
    void train(Eigen::MatrixXd &x, Eigen::MatrixXd &y, int epochs, double alpha);
};