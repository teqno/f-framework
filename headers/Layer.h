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
    std::map<std::string, Eigen::MatrixXd> forward_prop(Eigen::MatrixXd &x);
    std::map<std::string, Eigen::MatrixXd> getParams();
    std::vector<Neuron *> getNeurons();
};
