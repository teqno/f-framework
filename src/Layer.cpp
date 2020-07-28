#include "Layer.h"

Layer::Layer(int layer_size, int input_size, ACTIVATION_FUNCTION activation_function)
{
    this->layer_size = layer_size;
    neurons.reserve(layer_size);
    this->activation_function = activation_function;

    for (int i = 0; i < layer_size; i++)
    {
        Neuron *neuron = new Neuron(input_size);
        neurons.push_back(neuron);
    }
}

std::vector<double> Layer::forward_prop(std::vector<double> &x)
{
    std::vector<double> result;
    result.reserve(layer_size);

    for (auto const &n : neurons)
    {
        double neuron_activation = n->forward_prop(x, activation_function);
        result.push_back(neuron_activation);
    }

    return result;
}

std::vector<std::pair<std::vector<double>, double>> Layer::getParams()
{
    std::vector<std::pair<std::vector<double>, double>> params;
    params.reserve(layer_size);

    for (auto const &n : neurons)
    {
        params.push_back(std::make_pair(n->getW(), n->getB()));
    }

    return params;
}
