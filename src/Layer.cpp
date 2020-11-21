#include "Layer.h"
#include <iostream>

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

std::map<std::string, Eigen::VectorXd> Layer::forward_prop(Eigen::VectorXd &x)
{
    Eigen::VectorXd preactivations(neurons.size());
    Eigen::VectorXd activations(neurons.size());

    for (int i = 0; i < neurons.size(); i++)
    {
        std::map<std::string, double> neuron_activation = neurons.at(i)->forward_prop(x, activation_function);
        preactivations(i) = neuron_activation["z"];
        activations(i) = neuron_activation["a"];
    }

    return {{"preactivations", preactivations}, {"activations", activations}};
}

std::map<std::string, Eigen::MatrixXd> Layer::getParams()
{
    Eigen::MatrixXd w(neurons.size(), neurons.at(0)->getW().size());
    Eigen::VectorXd b(neurons.size());

    for (int i = 0; i < neurons.size(); i++)
    {
        HyperParameters parameters = neurons.at(i)->getParameters();
        w.row(i) = parameters.w;
        b(i) = parameters.b;
    }

    return {{"w", w}, {"b", b}};
}

std::vector<Neuron *> Layer::getNeurons()
{
    return neurons;
}
