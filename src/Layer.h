#pragma once

#include "Neuron.h"

class Layer
{
private:
    std::vector<Neuron*> neurons;
    ACTIVATION_FUNCTION activation_function;
public:
    Layer(int layer_size, int input_size, ACTIVATION_FUNCTION activation_function);
    std::vector<double> forward_prop(std::vector<double> &x);
};
