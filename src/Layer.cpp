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

Eigen::VectorXd Layer::forward_prop(Eigen::VectorXd &x)
{
    Eigen::VectorXd result(layer_size);

    for (int i = 0; i < neurons.size(); i++)
    {
        double neuron_activation = neurons.at(i)->forward_prop(x, activation_function);
        result(i) = neuron_activation;
    }

    return result;
}

std::vector<std::pair<Eigen::VectorXd, double>> Layer::getParams()
{
    std::vector<std::pair<Eigen::VectorXd, double>> params;
    params.reserve(layer_size);

    for (Neuron* n : neurons)
    {
        params.push_back(std::make_pair(n->getW(), n->getB()));
    }

    return params;
}

std::vector<Neuron *> Layer::getNeurons()
{
    return neurons;
}
