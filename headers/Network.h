#pragma once

#include "Layer.h"
#include "loss.h"

class Network
{
private:
    std::vector<Layer *> layers;

public:
    Network(std::vector<Layer *> &layers);
    std::vector<Layer *> getLayers();
    Eigen::VectorXd forward_prop(Eigen::VectorXd &input);
    double calc_cost(Eigen::MatrixXd &x, Eigen::VectorXd &y);
    void train(Eigen::MatrixXd &x, Eigen::VectorXd &y, int epochs);
};