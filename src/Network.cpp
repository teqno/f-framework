#include "Network.h"
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

std::vector<double> Network::forward_prop(std::vector<double> &input)
{
    std::vector<double> result = input;

    for (auto const &l : layers)
    {
        result = l->forward_prop(result);
    }

    return result;
}

double Network::calc_loss(std::vector<std::vector<double>> &x, std::vector<double> &y)
{
    double result = 0.0;
    for (int i = 0; i < x.size(); i++)
    {
        std::vector<double> ai = forward_prop(x.at(i));
        result += cross_entropy_loss(ai.at(0), y.at(i));
    }

    return result / x.size();
}

void Network::train(std::vector<std::vector<double>> &x, std::vector<double> &y, int epochs)
{
    for (int i = 0; i < epochs; i++)
    {
        std::vector<std::vector<double>> activations;
        for (int k = 0; k < x.size(); k++) {
            activations.push_back(forward_prop(x.at(k)));
        }
        // for (int j = layers.size() - 1; j >= 0; j--)
        // {
        //     std::vector<double> dz;
        //     std::vector<double> dw;
        //     std::vector<double> db;
        //     if (i == layers.size() - 1)
        //     {
        //          layers.at(i)->forward_prop();
        //     }
        // }

        double loss = calc_loss(x, y);
        std::cout << "Loss: " << loss << std::endl;
    }
}