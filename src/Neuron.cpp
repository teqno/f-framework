#include <algorithm>
#include <cstdlib>
#include <map>
#include <numeric>
#include <random>
#include <string.h>
#include <iostream>

#include "Neuron.h"
#include "activations.h"
#include "random.h"

// static std::default_random_engine generator = initialize_random_seed();

Neuron::Neuron(int input_size)
{
    parameters.w = Eigen::VectorXd(input_size);

    // std::normal_distribution<double> distribution(0, 1);
    for (int i = 0; i < input_size; i++)
    {
        // parameters.w(i) = distribution(generator);
        parameters.w(i) = ((double)rand()) / RAND_MAX;
    }

    parameters.b = 0;
}

HyperParameters Neuron::getParameters() { return parameters; }

Eigen::VectorXd Neuron::getW() { return parameters.w; }

void Neuron::setW(Eigen::VectorXd w) { parameters.w = w; }

double Neuron::getB() { return parameters.b; }

void Neuron::setB(double b) { parameters.b = b; }
