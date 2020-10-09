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
    std::vector<double> forward_prop(std::vector<double> &input);
    double calc_loss(std::vector<std::vector<double>> &x, std::vector<double> &y);
    void train(std::vector<std::vector<double>> &x, std::vector<double> &y, int epochs);
};