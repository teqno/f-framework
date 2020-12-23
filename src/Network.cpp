#include "Network.h"
#include "activations.h"
#include "backpropagation.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

Network::Network(const std::vector<Layer *> &layers, std::optional<unsigned int> random_seed)
{
    // initialize random generator with the passed seed
    std::srand(random_seed.value_or(std::time(nullptr)));

    this->layers = layers;

    this->cache.activations = std::vector<Eigen::VectorXd>();
    this->cache.preactivations = std::vector<Eigen::VectorXd>();
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

    this->cache.activations.push_back(layerActivations);

    for (std::size_t i = 0; i < layers.size(); i++)
    {
        DataTypes::LayerCacheResult layerCacheResult = layers.at(i)->forward_prop(layerActivations);

        Eigen::VectorXd layerPreactivations = layerCacheResult.preactivations;
        layerActivations = layerCacheResult.activations;

        this->cache.preactivations.push_back(layerPreactivations);
        this->cache.activations.push_back(layerActivations);
    }

    return layerActivations;
}

DataTypes::Deltas Network::back_prop(const Eigen::VectorXd &y)
{
    std::vector<Eigen::MatrixXd> dw;
    std::vector<Eigen::VectorXd> db;

    dw.reserve(layers.size());
    db.reserve(layers.size());

    Eigen::VectorXd current_da = mse_prime(this->cache.activations.back(), y);
    Eigen::VectorXd current_dz = current_da.array() * activation_prime(this->cache.preactivations.back(), layers.back()->activation_function).array();
    // Eigen::MatrixXd current_dw = (Eigen::MatrixXd::Ones(current_dz.size(), this->cache.activations.rbegin()[1].transpose().size()).array().colwise() * current_dz.array()).array() * (Eigen::MatrixXd::Ones(current_dz.size(), this->cache.activations.rbegin()[1].transpose().size()).array().rowwise() * this->cache.activations.rbegin()[1].transpose().array()).array();
    // Eigen::VectorXd current_db = current_dz;

    // // std::cout << current_da << "--------------dA\n";
    // // std::cout << current_dz << "--------------dZ\n";
    // // std::cout << current_dw << "--------------dW\n";
    // // std::cout << current_db << "--------------dB\n";

    // dw.push_back(current_dw);
    // db.push_back(current_db);

    for (int i = layers.size() - 1; i >= 0; i--)
    {
        Eigen::MatrixXd current_dw;
        Eigen::VectorXd current_db;

        Layer *current_layer = layers.at(i);

        Eigen::VectorXd a_next = this->cache.activations.at(i);

        if (((std::size_t)i) == layers.size() - 1)
        {
            Eigen::MatrixXd current_dz_m = current_dz.matrix();
            Eigen::MatrixXd a_next_m = a_next.matrix();

            current_dw = current_dz_m * a_next_m.transpose();
            current_db = current_dz;
        }
        else
        {
            Layer *prev_layer = layers.at(i + 1);

            Eigen::MatrixXd w_prev = prev_layer->getParams().w;
            Eigen::VectorXd z_current = this->cache.preactivations.at(i);

            current_dz = (w_prev.transpose() * current_dz).array() * activation_prime(z_current, current_layer->activation_function).array();

            Eigen::MatrixXd current_dz_m = current_dz.matrix();
            Eigen::MatrixXd a_next_m = a_next.matrix();

            current_dw = current_dz_m * a_next_m.transpose();
            current_db = current_dz;

            // std::cout << temp1 << "--------------dZ1\n";
            // std::cout << temp2 << "--------------dZ2\n";
            // std::cout << temp3 << "--------------dZ3\n";
            // std::cout << current_dz << "--------------dZ\n";
            // std::cout << current_dw << "--------------dW\n";
            // std::cout << current_db << "--------------dB\n";
        }

        dw.push_back(current_dw);
        db.push_back(current_db);
    }

    return {.dw = dw, .db = db};
}

void Network::updateParameters(const std::vector<Eigen::MatrixXd> &dw, const std::vector<Eigen::VectorXd> &db, double alpha)
{
    for (std::size_t i = 0; i < layers.size(); i++)
    {
        std::vector<Neuron *> neurons = layers.at(i)->getNeurons();

        for (std::size_t j = 0; j < neurons.size(); j++)
        {
            Neuron *neuron = neurons.at(j);

            Eigen::VectorXd w = neuron->getW();
            double b = neuron->getB();

            // Network::retainedGradient["w"].at(i).row(j) = alphaMomentum * retainedGradient["w"].at(i).row(j).eval().array() + (1 - alphaMomentum) * dw.rbegin()[i].row(j).array();
            // Network::retainedGradient["b"].at(i).row(j) = alphaMomentum * retainedGradient["b"].at(i).row(j).eval().array() + (1 - alphaMomentum) * db.rbegin()[i].row(j).array();

            // w = w.array() + alpha * Network::retainedGradient["w"].at(i).row(j).transpose().array();
            // b = b + alpha * Network::retainedGradient["b"].at(i).row(j)(0);

            w = w.array() + alpha * dw.at(layers.size() - 1 - i).row(j).transpose().array();
            b = b + alpha * db.at(layers.size() - 1 - i)(j);

            neuron->setW(w);
            neuron->setB(b);
        }
    }
}

double Network::calc_cost(const Eigen::VectorXd &x, const Eigen::VectorXd &y)
{
    Eigen::VectorXd activations = forward_prop(x);
    double result = mse(activations, y);

    return result / y.size();
}

Eigen::VectorXd Network::train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y, int epochs, double alpha)
{
    // for (Layer *layer : layers)
    // {
    //     Network::retainedGradient["w"].push_back(Eigen::MatrixXd::Zero(layer->getParams()["w"].rows(), layer->getParams()["w"].cols()));
    //     Network::retainedGradient["b"].push_back(Eigen::MatrixXd::Zero(layer->getParams()["b"].rows(), layer->getParams()["b"].cols()));
    // }

    Eigen::VectorXd lossCache(epochs);

    for (int i = 0; i < epochs; i++)
    {
        double cost = 0;

        for (int j = 0; j < x.rows(); j++)
        {
            cost += calc_cost(x.row(j), y.row(j));
            // forward_prop(x.row(j));
            DataTypes::Deltas deltas = back_prop(y.row(j));
            updateParameters(deltas.dw, deltas.db, alpha);
        }

        lossCache(i) = cost;

        if (i % 1000 == 0)
        {
            std::cout << "Epoch " << i << ": " << cost << std::endl;
        }
    }

    std::cout << "Epoch " << epochs << ": " << lossCache(epochs - 1) << std::endl;

    return lossCache;
}