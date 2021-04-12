#include "Network.h"
#include "activations.h"
#include "backpropagation.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "debug.h"

Network::Network(const std::vector<Layer *> &layers, std::optional<unsigned int> random_seed)
{
    // initialize random generator with the passed seed
    std::srand(random_seed.value_or(std::time(nullptr)));

    this->layers = layers;
}

std::vector<Layer *> Network::getLayers()
{
    return this->layers;
}

Eigen::VectorXd Network::forward_prop(const Eigen::VectorXd &input)
{
    this->cache.preactivations.reserve(layers.size());
    this->cache.activations.reserve(layers.size() + 1);

    Eigen::VectorXd layerActivations = input;

    cache.activations.push_back(layerActivations);

    for (std::size_t i = 0; i < layers.size(); i++)
    {
        DataTypes::LayerCacheResult layerCacheResult = layers.at(i)->forward_prop(layerActivations);

        Eigen::VectorXd layerPreactivations = layerCacheResult.preactivations;
        layerActivations = layerCacheResult.activations;

        this->cache.preactivations.push_back(layerPreactivations);
        this->cache.activations.push_back(layerActivations);

        debug("Network::forward_prop > layer " + std::to_string(i) + " weights");
        debug(layers.at(i)->getParams().w);

        debug("Network::forward_prop > layer " + std::to_string(i) + " biases");
        debug(layers.at(i)->getParams().b);

        debug("Network::forward_prop > layer " + std::to_string(i) + " preactivations");
        debug(layerPreactivations);

        debug("Network::forward_prop > layer " + std::to_string(i) + " activations");
        debug(layerActivations);
    }

    return layerActivations;
}

DataTypes::Deltas Network::back_prop(const Eigen::VectorXd &y)
{
    debug("Network::back_prop > y");
    debug(y);

    std::vector<Eigen::MatrixXd> dw;
    std::vector<Eigen::VectorXd> db;

    dw.reserve(layers.size());
    db.reserve(layers.size());

    Eigen::VectorXd current_da = mse_prime(this->cache.activations.back(), y);

    debug("Network::back_prop > current_da last");
    debug(current_da);

    Eigen::VectorXd current_dz = current_da.array() * activation_prime(this->cache.preactivations.back(), layers.back()->activation_function).array();
    
    debug("Network::back_prop > current_dz last");
    debug(current_dz);

    for (int i = layers.size() - 1; i >= 0; i--)
    {
        Eigen::MatrixXd current_dw;
        Eigen::VectorXd current_db;

        Layer *current_layer = layers.at(i);

        Eigen::VectorXd a_next = this->cache.activations.at(i);

        debug("Network::back_prop > a_next for layer " + std::to_string(i));
        debug(a_next);

        if (((std::size_t)i) == layers.size() - 1)
        {
            Eigen::MatrixXd current_dz_m = current_dz.matrix();
            Eigen::MatrixXd a_next_m = a_next.matrix();

            current_dw = current_dz_m * a_next_m.transpose();
            current_db = current_dz;
        }
        else
        {
            Layer *prev_layer = layers.at((int)i + 1);

            Eigen::MatrixXd w_prev = prev_layer->getParams().w;
            Eigen::VectorXd z_current = this->cache.preactivations.at(i);

            current_dz = (w_prev.transpose() * current_dz).array() * activation_prime(z_current, current_layer->activation_function).array();

            Eigen::MatrixXd current_dz_m = current_dz.matrix();
            Eigen::MatrixXd a_next_m = a_next.matrix();

            current_dw = current_dz_m * a_next_m.transpose();
            current_db = current_dz;
        }

        dw.push_back(current_dw);
        db.push_back(current_db);
    }

    return {.dw = dw, .db = db};
}

void Network::updateParameters(const std::vector<Eigen::MatrixXd> &dw, const std::vector<Eigen::VectorXd> &db, double alpha, std::optional<DataTypes::Deltas> prev_deltas, std::optional<double> momentum)
{
    Eigen::MatrixXd delta_w;
    double delta_b;

    for (std::size_t i = 0; i < layers.size(); i++)
    {
        std::vector<Neuron *> neurons = layers.at(i)->getNeurons();

        for (std::size_t j = 0; j < neurons.size(); j++)
        {
            Neuron *neuron = neurons.at(j);

            Eigen::VectorXd w = neuron->getW();
            double b = neuron->getB();

            delta_w = dw.at(layers.size() - 1 - i).row(j).transpose().array();
            delta_b = db.at(layers.size() - 1 - i)(j);

            if (prev_deltas.has_value() && momentum.has_value()) {
                delta_w = delta_w.array() + momentum.value() * prev_deltas.value().dw.at(layers.size() - 1 - i).row(j).transpose().array();
                delta_b = delta_b + momentum.value() * prev_deltas.value().db.at(layers.size() - 1 - i)(j);
            }

            w = w.array() + alpha * delta_w.array();
            b = b + alpha * delta_b;

            neuron->setW(w);
            neuron->setB(b);
        }
    }
}

double Network::calc_cost(const Eigen::VectorXd &x, const Eigen::VectorXd &y)
{
    debug("Network::calc_cost > x");
    debug(x);

    debug("Network::calc_cost > y");
    debug(y);

    Eigen::VectorXd activations = forward_prop(x);

    debug("Network::calc_cost > activations");
    debug(activations);

    double result = mse(activations, y);

    debug("Network::calc_cost > mse");
    debug(result);

    return result / y.size();
}

Eigen::VectorXd Network::train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y, int epochs, double alpha, double momentum)
{
    debug("Network::train > x");
    debug(x);

    debug("Network::train > y");
    debug(y);

    debug("Network::train > epochs");
    debug(epochs);

    debug("Network::train > alpha");
    debug(alpha);

    Eigen::VectorXd lossCache(epochs);

    for (int i = 0; i < epochs; i++)
    {
        double cost = 0;
        DataTypes::Deltas deltas;

        DataTypes::Deltas prev_deltas;

        for (int j = 0; j < x.rows(); j++)
        {
            cost += calc_cost(x.row(j), y.row(j));
            
            DataTypes::Deltas current_deltas = back_prop(y.row(j));
            
            if (j == 0) {
                updateParameters(current_deltas.dw, current_deltas.db, alpha, std::nullopt, std::nullopt);
            }
            else {
                updateParameters(current_deltas.dw, current_deltas.db, alpha, prev_deltas, momentum);
            }

            prev_deltas = current_deltas;

            cache.activations.clear();
            cache.preactivations.clear();
        }

        lossCache(i) = cost;

        if (i % (epochs / 20) == 0 || epochs < 20)
        {
            std::cout << "Epoch " << i + 1 << ": " << cost << std::endl;
        }
    }

    std::cout << "Epoch " << epochs << ": " << lossCache((int)epochs - 1) << std::endl;

    return lossCache;
}