#pragma once
#include <map>

#include "Neuron.h"

class Layer
{
private:
    int layer_size;
    std::vector<Neuron *> neurons;

public:
    ACTIVATION_FUNCTION activation_function;
    Layer(int layer_size, int input_size, ACTIVATION_FUNCTION activation_function);
    DataTypes::LayerCacheResult forward_prop(const Eigen::VectorXd &x);
    DataTypes::LayerHyperParameters getParams();
    std::vector<Neuron *> getNeurons();
};
