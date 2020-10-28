#pragma once
#include <map>

#include "Neuron.h"

class Layer
{
private:
    int layer_size;
    std::vector<Neuron *> neurons;
    ACTIVATION_FUNCTION activation_function;

public:
    Layer(int layer_size, int input_size, ACTIVATION_FUNCTION activation_function);
    Eigen::VectorXd forward_prop(Eigen::VectorXd &x);
    std::vector<std::pair<Eigen::VectorXd, double>> getParams();
    std::vector<Neuron *> getNeurons();
};
