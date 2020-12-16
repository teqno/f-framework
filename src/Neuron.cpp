#include <algorithm>
#include <cstdlib>
#include <map>
#include <numeric>
#include <random>
#include <string.h>
#include <iostream>

#include "Neuron.h"
#include "activations.h"

Neuron::Neuron(int input_size)
{
    Neuron::parameters.w = Eigen::VectorXd(input_size);

    // std::normal_distribution<double> distribution(0, 1);
    for (int i = 0; i < input_size; i++)
    {
        // parameters.w(i) = distribution(generator);
        Neuron::parameters.w(i) = ((double)std::rand()) / RAND_MAX;
    }

    Neuron::parameters.b = 0;
}

HyperParameters Neuron::getParameters() { return Neuron::parameters; }

Eigen::VectorXd Neuron::getW() { return Neuron::parameters.w; }

void Neuron::setW(const Eigen::VectorXd &w) { Neuron::parameters.w = w; }

double Neuron::getB() { return Neuron::parameters.b; }

void Neuron::setB(double b) { Neuron::parameters.b = b; }
