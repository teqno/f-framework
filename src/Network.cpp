#include "Network.h"
#include "activations.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

Network::Network(const std::vector<Layer *> &layers, std::optional<unsigned int> random_seed)
{
    std::srand(random_seed.value_or(std::time(nullptr)));
    this->layers = layers;
}

std::vector<Layer *> Network::getLayers()
{
    return layers;
}

Eigen::VectorXd Network::forward_prop(const Eigen::VectorXd &input)
{
    activations = std::vector<Eigen::VectorXd>();
    preactivations = std::vector<Eigen::VectorXd>();

    activations.reserve(layers.size() + 1);
    preactivations.reserve(layers.size());

    Eigen::VectorXd activation = input; // (3, 1)

    activations.push_back(activation);

    for (std::size_t i = 0; i < layers.size(); i++)
    {
        std::map<std::string, Eigen::VectorXd> layer_output = layers.at(i)->forward_prop(activation);

        Eigen::VectorXd z = layer_output["preactivations"];
        activation = layer_output["activations"];

        preactivations.push_back(z);
        activations.push_back(activation);
    }

    return activation;
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> Network::back_prop(const Eigen::MatrixXd &y)
{
    std::vector<Eigen::VectorXd> dz;
    std::vector<Eigen::MatrixXd> dw;
    std::vector<Eigen::MatrixXd> db;

    dz.reserve(layers.size());
    dw.reserve(layers.size());
    db.reserve(layers.size());

    Eigen::VectorXd current_da = mse_prime(activations.back(), y.transpose());
    Eigen::VectorXd current_dz = current_da.array() * activation_prime(preactivations.back(), layers.back()->activation_function).array();
    Eigen::MatrixXd current_dw = (Eigen::MatrixXd::Ones(current_dz.size(), activations.rbegin()[1].transpose().size()).array().colwise() * current_dz.array()).array() * (Eigen::MatrixXd::Ones(current_dz.size(), activations.rbegin()[1].transpose().size()).array().rowwise() * activations.rbegin()[1].transpose().array()).array();
    Eigen::MatrixXd current_db = current_dz;

    // std::cout << current_da << "--------------dA\n";
    // std::cout << current_dz << "--------------dZ\n";
    // std::cout << current_dw << "--------------dW\n";
    // std::cout << current_db << "--------------dB\n";

    dz.push_back(current_dz);
    dw.push_back(current_dw);
    db.push_back(current_db);

    for (int i = layers.size() - 2; i >= 0; i--)
    {
        Eigen::MatrixXd w_prev = layers.at(i + 1)->getParams()["w"];
        Eigen::VectorXd z_current = preactivations.at(i);
        Eigen::VectorXd a_next = activations.at(i);

        Layer *current_layer = layers.at(i);

        Eigen::MatrixXd temp1 = w_prev.array().colwise() * current_dz.array();
        Eigen::MatrixXd temp2 = temp1.colwise().sum();
        Eigen::MatrixXd temp3 = temp2.transpose().array() * activation_prime(z_current, current_layer->activation_function).array();
        current_dz = temp3;
        current_dw = (Eigen::MatrixXd::Ones(current_dz.size(), a_next.rows()).array().rowwise() * a_next.transpose().array()).array().colwise() * current_dz.array();
        current_db = current_dz;

        // std::cout << temp1 << "--------------dZ1\n";
        // std::cout << temp2 << "--------------dZ2\n";
        // std::cout << temp3 << "--------------dZ3\n";
        // std::cout << current_dz << "--------------dZ\n";
        // std::cout << current_dw << "--------------dW\n";
        // std::cout << current_db << "--------------dB\n";

        dz.push_back(current_dz);
        dw.push_back(current_dw);
        db.push_back(current_db);
    }

    return std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>>(dw, db);
}

void Network::updateParameters(const std::vector<Eigen::MatrixXd> &dw, const std::vector<Eigen::MatrixXd> &db, double alpha, double alphaMomentum = 0)
{
    for (std::size_t i = 0; i < layers.size(); i++)
    {
        std::vector<Neuron *> neurons = layers.at(i)->getNeurons();

        for (std::size_t j = 0; j < neurons.size(); j++)
        {
            Neuron *neuron = neurons.at(j);

            Eigen::VectorXd w = neuron->getW();
            double b = neuron->getB();

            Network::retainedGradient["w"].at(i).row(j) = alphaMomentum * retainedGradient["w"].at(i).row(j).eval().array() + (1 - alphaMomentum) * dw.rbegin()[i].row(j).array();
            Network::retainedGradient["b"].at(i).row(j) = alphaMomentum * retainedGradient["b"].at(i).row(j).eval().array() + (1 - alphaMomentum) * db.rbegin()[i].row(j).array();

            w = w.array() + alpha * Network::retainedGradient["w"].at(i).row(j).transpose().array();
            b = b + alpha * Network::retainedGradient["b"].at(i).row(j)(0);

            neuron->setW(w);
            neuron->setB(b);
        }
    }
}

double Network::calc_cost(const Eigen::VectorXd &x, const Eigen::VectorXd &y)
{
    Eigen::VectorXd activation = forward_prop(x);
    double result = mse(activation, y.transpose());

    return result / y.size();
}

Eigen::VectorXd Network::train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y, int epochs, double alpha, double alphaMomentum)
{
    for (Layer *layer : layers)
    {
        Network::retainedGradient["w"].push_back(Eigen::MatrixXd::Zero(layer->getParams()["w"].rows(), layer->getParams()["w"].cols()));
        Network::retainedGradient["b"].push_back(Eigen::MatrixXd::Zero(layer->getParams()["b"].rows(), layer->getParams()["b"].cols()));
    }

    Eigen::VectorXd lossCache(epochs);

    for (int i = 0; i < epochs; i++)
    {
        double cost = 0;

        for (int j = 0; j < x.rows(); j++)
        {
            cost += calc_cost(x.row(j), y.row(j));
            // forward_prop(x.row(j));
            auto deltas = back_prop(y.row(j));
            updateParameters(deltas.first, deltas.second, alpha, alphaMomentum);
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