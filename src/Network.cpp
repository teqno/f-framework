#include "Network.h"
#include "activations.h"
#include <iostream>
#include <vector>

Network::Network(std::vector<Layer *> &layers)
{
    this->layers = layers;
}

std::vector<Layer *> Network::getLayers()
{
    return layers;
}

Eigen::VectorXd Network::forward_prop(Eigen::VectorXd &input)
{
    activations = std::vector<Eigen::VectorXd>();
    preactivations = std::vector<Eigen::VectorXd>();

    // activations size = network depth + input layer
    activations.reserve(layers.size() + 1);
    preactivations.reserve(layers.size());

    Eigen::VectorXd activation = input;

    activations.push_back(activation);

    for (int i = 0; i < layers.size(); i++)
    {
        std::map<std::string, Eigen::VectorXd> layer_output = layers.at(i)->forward_prop(activation);

        Eigen::VectorXd z = layer_output["preactivations"];
        activation = layer_output["activations"];

        preactivations.push_back(z);
        activations.push_back(activation);
    }

    return activation;
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> Network::back_prop(Eigen::VectorXd y)
{
    std::vector<Eigen::MatrixXd> dz;
    std::vector<Eigen::MatrixXd> dw;
    std::vector<Eigen::MatrixXd> db;

    dz.reserve(layers.size());
    dw.reserve(layers.size());
    db.reserve(layers.size());

    dz.push_back(mse_prime(activations.back(), y));
    dw.push_back(dz.at(0).array() * activation_prime(preactivations.back(), layers.back()->activation_function).array() * activations.at(activations.size() - 1).array());
    db.push_back(dz.at(0).array() * activation_prime(preactivations.back(), layers.back()->activation_function).array());

    for (int i = layers.size() - 2; i >= 0; i--)
    {
        Eigen::MatrixXd w_prev = layers.at(i + 1)->getParams()["w"];
        Eigen::MatrixXd z = preactivations.at(i);
        Eigen::MatrixXd a_next = activations.at(i);

        // dz.at(i) = (w_prev.transpose() * dz.at(i + 1)).cwiseProduct(activation_prime(z, layers.at(i)->activation_function));
        dz.push_back((w_prev.transpose() * dz.at(layers.size() - 2 - i)).cwiseProduct(activation_prime(z, layers.at(i)->activation_function)));
        // dw.at(i) = dz.at(i) * a_next.transpose();
        dw.push_back(dz.at(layers.size() - 2 - i + 1) * a_next.transpose());
        // db.at(i) = dz.at(i);
        db.push_back(dz.at(layers.size() - 2 - i + 1));
    }

    return std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>>(dw, db);
}

void Network::updateParameters(std::vector<Eigen::MatrixXd> dw, std::vector<Eigen::MatrixXd> db, double alpha)
{
    for (int i = 0; i < layers.size(); i++) {
        std::vector<Neuron *> neurons = layers.at(i)->getNeurons();

        for (int j = 0; j < neurons.size(); j++) {
            Neuron * neuron = neurons.at(j);
            
            Eigen::VectorXd w = neuron->getW();
            double b = neuron->getB();

            w = w.array() - alpha * dw.at(layers.size() - i - 1).row(j).array();
            b = b - alpha * db.at(layers.size() - i - 1).row(j)(0);

            neuron->setW(w);
            neuron->setB(b);
        }
    }
}

double Network::calc_cost(Eigen::MatrixXd &x, Eigen::MatrixXd &y)
{
    double result = 0.0;
    for (int i = 0; i < x.rows(); i++)
    {
        Eigen::VectorXd trainingExample = x.row(i);
        Eigen::VectorXd ai = forward_prop(trainingExample);
        Eigen::VectorXd yi = y.row(i);
        result += mse(ai, yi);
    }

    return result / x.rows();
}

void Network::train(Eigen::MatrixXd &x, Eigen::MatrixXd &y, int epochs, double alpha)
{
    // repeat for n epochs:
    // activation <- forward_prop
    // cost <- calc_cost
    // delta_ws, delta_bs for all layers <- back_prop
    // void <- update_parameters

    for (int i = 0; i < epochs; i++) {
        double cost = calc_cost(x, y);
        std::cout << cost << std::endl;

        for (int j = 0; j < x.rows(); j++) {
            Eigen::VectorXd x_j = x.row(j);
            forward_prop(x_j);
            auto deltas = back_prop(y.row(j));
            updateParameters(deltas.first, deltas.second, alpha);
        } 
    }
}