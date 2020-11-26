#include "Network.h"
#include "activations.h"
#include <iostream>
#include <vector>
#include <time.h>

Network::Network(std::vector<Layer *> &layers)
{
    srand((unsigned)time(NULL));
    this->layers = layers;
}

std::vector<Layer *> Network::getLayers()
{
    return layers;
}

Eigen::MatrixXd Network::forward_prop(Eigen::MatrixXd &input)
{
    activations = std::vector<Eigen::MatrixXd>();
    preactivations = std::vector<Eigen::MatrixXd>();

    activations.reserve(layers.size());
    preactivations.reserve(layers.size());

    Eigen::MatrixXd activation = input.transpose();

    activations.push_back(activation);

    for (int i = 0; i < layers.size(); i++)
    {
        std::map<std::string, Eigen::MatrixXd> layer_output = layers.at(i)->forward_prop(activation);

        Eigen::MatrixXd z = layer_output["preactivations"];
        activation = layer_output["activations"];

        preactivations.push_back(z);
        activations.push_back(activation);
    }

    return activation.transpose();
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> Network::back_prop(Eigen::MatrixXd y)
{
    std::vector<Eigen::MatrixXd> dz;
    std::vector<Eigen::MatrixXd> dw;
    std::vector<Eigen::MatrixXd> db;

    dz.reserve(layers.size());
    dw.reserve(layers.size());
    db.reserve(layers.size());

    auto current_da = mse_prime(activations.back(), y.transpose());

    Eigen::MatrixXd current_dz = current_da.array() * activation_prime(preactivations.back(), layers.back()->activation_function).array();
    Eigen::MatrixXd current_dw = 1.0 / current_dz.cols() * current_dz * activations.rbegin()[1].transpose();
    Eigen::MatrixXd current_db = 1.0 / current_dz.cols() * current_dz.rowwise().sum();

    // std::cout << "da" << current_da << std::endl;
    // std::cout << "dz" << current_dz << std::endl;
    // std::cout << "dw" << current_dw << std::endl;
    // std::cout << "db" << current_db << std::endl;

    dz.push_back(current_dz);
    dw.push_back(current_dw);
    db.push_back(current_db);

    for (int i = layers.size() - 2; i >= 0; i--)
    {
        Eigen::MatrixXd w_prev = layers.at(i + 1)->getParams()["w"];
        Eigen::MatrixXd z = preactivations.at(i);
        Eigen::MatrixXd a_next = activations.at(i);

        Layer *current_layer = layers.at(i);

        Eigen::MatrixXd current_dz = (w_prev * dz[layers.size() - 2 - i]).array() * activation_prime(z, current_layer->activation_function).array();
        Eigen::MatrixXd current_dw = 1 / current_dz.cols() * current_dz * a_next.transpose();
        Eigen::MatrixXd current_db = 1 / current_dz.cols() * current_dz.rowwise().sum();

        // dz.at(i) = (w_prev.transpose() * dz.at(i + 1)).cwiseProduct(activation_prime(z, layers.at(i)->activation_function));
        dz.push_back(current_dz);
        // dw.at(i) = dz.at(i) * a_next.transpose();
        dw.push_back(current_dw);
        // db.at(i) = dz.at(i);
        db.push_back(current_db);
    }

    return std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>>(dw, db);
}

void Network::updateParameters(std::vector<Eigen::MatrixXd> dw, std::vector<Eigen::MatrixXd> db, double alpha)
{
    for (int i = 0; i < layers.size(); i++)
    {
        std::vector<Neuron *> neurons = layers.at(i)->getNeurons();

        for (int j = 0; j < neurons.size(); j++)
        {
            Neuron *neuron = neurons.at(j);

            Eigen::VectorXd w = neuron->getW();
            double b = neuron->getB();

            w = w.array() - alpha * dw.rbegin()[i].row(j).transpose().array();
            b = b - alpha * db.rbegin()[i].row(j)(0);

            neuron->setW(w);
            neuron->setB(b);
        }
    }
}

double Network::calc_cost(Eigen::MatrixXd &x, Eigen::MatrixXd &y)
{
    Eigen::MatrixXd activation = forward_prop(x);
    double result = mse(activation, y);

    return result / y.size();
}

void Network::train(Eigen::MatrixXd &x, Eigen::MatrixXd &y, int epochs, double alpha)
{
    // repeat for n epochs:
    // activation <- forward_prop
    // cost <- calc_cost
    // delta_ws, delta_bs for all layers <- back_prop
    // void <- update_parameters

    for (int i = 0; i < epochs; i++)
    {
        if (i % 1000 == 0)
        {
            double cost = calc_cost(x, y);
            std::cout << "Epoch " << i << ": " << cost << std::endl;
        }

        forward_prop(x);
        auto deltas = back_prop(y);
        updateParameters(deltas.first, deltas.second, alpha);
    }

    double cost = calc_cost(x, y);
    std::cout << "Epoch " << epochs << ": " << cost << std::endl;
}