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
    for (int i = 0; i < x.rows(); i++)
    {
        Eigen::VectorXd trainingExample = x.row(i);
        Eigen::VectorXd ai = forward_prop(trainingExample);
        result += cross_entropy_loss(ai(0), y(i));
    }

    return result / x.rows();
}

void Network::train(Eigen::MatrixXd &x, Eigen::VectorXd &y, int epochs)
{
    
}