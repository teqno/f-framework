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
    std::vector<double> forward_prop(std::vector<double> &x);
    std::vector<std::pair<std::vector<double>, double>> getParams();
};
