#include "Network.h"
#include <iostream>
#include <vector>

using namespace Eigen;

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
    Eigen::VectorXd result = input;

    for (Layer* &l : layers)
    {
        result = l->forward_prop(result);
    }

    return result;
}

double Network::calc_cost(Eigen::MatrixXd &x, Eigen::VectorXd &y)
{
    double result = 0.0;
    for (int i = 0; i < x.size(); i++)
    {
        VectorXd trainingExample = x.row(i);
        Eigen::VectorXd ai = forward_prop(trainingExample);
        result += cross_entropy_loss(ai(0), y(i));
    }

    return result / x.rows();
}

void Network::train(Eigen::MatrixXd &x, Eigen::VectorXd &y, int epochs)
{
    for (int i = 0; i < epochs; i++)
    {
        int m = x.rows();
        double J = 0;

        for (int j = 0; j < m; j++) {
            Eigen::VectorXd trainingExample = x.row(j);
            Eigen::VectorXd predictions = forward_prop(trainingExample);

            std::cout << "Training example " << j << std::endl;
            std::cout << predictions.size() << std::endl;

            double networkPrediction = predictions(0);
            double expectedOutput = y(0);

            double loss = cross_entropy_loss(networkPrediction, expectedOutput);

            std::cout << "Loss: " << loss << std::endl;

            for (int k = layers.size() - 1; k >= 0; k--) {
                Layer* currentLayer = layers.at(k);
                
                std::vector<std::pair<Eigen::VectorXd, double>> nodesParams = currentLayer->getParams();

                std::vector<double> dZ;

                if (k == layers.size() - 1) {
                    std::vector<Neuron *> neurons = currentLayer->getNeurons();
                    
                    std::cout << "This layer size should be 1: " << neurons.size() << std::endl;

                    Neuron* outputNeuron = neurons.at(0);
                    double z = outputNeuron->getZ();

                    std::cout << "z value of output neuron: " << z << std::endl;

                    dZ.push_back(networkPrediction - expectedOutput);
                    break;
                } else {
                    std::vector<Neuron *> neurons = currentLayer->getNeurons();
                    
                    std::cout << "This layer size should be greater than 1: " << neurons.size() << std::endl;

                    for (Neuron* n : neurons) {
                        double z = n->getZ();
                        // double dZi = 
                    }
                }


            }

            J += loss;
        }

        J /= m;

        std::cout << "Average loss: " << J << std::endl;
    }
}