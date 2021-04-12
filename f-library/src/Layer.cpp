#include "Layer.h"
#include <iostream>
#include "DataTypes.h"

Layer::Layer(int layer_size, int input_size, ACTIVATION_FUNCTION activation_function)
{
    this->layer_size = layer_size;
    this->activation_function = activation_function;

    neurons.reserve(layer_size);

    for (int i = 0; i < layer_size; i++)
    {
        Neuron *neuron = new Neuron(input_size);
        neurons.push_back(neuron);
    }
}

/**
 * Returns activations and preactivations of the neurons of this layer
 * 
 * If current layer contains 3 neurons then:
 * 
 * Eigen::VectorXd preactivations =
 * | * |
 * | * |
 * | * |
 * 
 * Eigen::VectorXd activations =
 * | * |
 * | * |
 * | * |
 */
DataTypes::LayerCacheResult Layer::forward_prop(const Eigen::VectorXd &activations)
{
    DataTypes::LayerParameters layerParams = this->getParams();
    Eigen::MatrixXd w = layerParams.w;
    Eigen::VectorXd b = layerParams.b;

    Eigen::VectorXd layerPreactivations = preactivation(w, activations, b);
    Eigen::VectorXd layerActivations = activation(layerPreactivations, activation_function);

    return {.preactivations = layerPreactivations, .activations = layerActivations};
}

/**
 * Returns hyperparameters of the neurons of this layer
 * 
 * If current layer contains 3 neurons and inputs 2 activations then:
 * 
 * Eigen::MatrixXd w =
 * | * * |
 * | * * |
 * | * * |
 * 
 * Eigen::MatrixXd b =
 * | * |
 * | * |
 * | * |
 */
DataTypes::LayerParameters Layer::getParams()
{
    Eigen::MatrixXd w(neurons.size(), neurons.at(0)->getW().size());
    Eigen::VectorXd b(neurons.size());

    for (std::size_t i = 0; i < neurons.size(); i++)
    {
        DataTypes::NeuronParameters parameters = neurons.at(i)->getParameters();
        w.row(i) = parameters.w;
        b(i) = parameters.b;
    }

    return {.w = w, .b = b};
}

std::vector<Neuron *> Layer::getNeurons()
{
    return neurons;
}

void Layer::setWeights(Eigen::MatrixXd weights)
{
    for (size_t i = 0; i < neurons.size(); i++) {
        neurons.at(i)->setW(weights.row(i));
    }
}
