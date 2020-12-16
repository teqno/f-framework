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

std::map<std::string, Eigen::VectorXd> Layer::forward_prop(const Eigen::VectorXd &x)
{
    Eigen::MatrixXd w = getParams()["w"];
    Eigen::VectorXd b = getParams()["b"];

    // std::cout << w << "-------------W\n";
    // std::cout << b << "-------------B\n";

    Eigen::VectorXd preactivations = (w * x).array() + b.array();

    // std::cout << preactivations << "-------------Preactivations\n";

    Eigen::VectorXd activations = activation(preactivations, activation_function);

    // std::cout << activations << "-------------Activations\n";

    return {{"preactivations", preactivations}, {"activations", activations}};
}

std::map<std::string, Eigen::MatrixXd> Layer::getParams()
{
    Eigen::MatrixXd w(neurons.size(), neurons.at(0)->getW().size());
    Eigen::VectorXd b(neurons.size());

    for (std::size_t i = 0; i < neurons.size(); i++)
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
