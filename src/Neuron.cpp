#include <cstdlib>
#include <random>
#include <time.h>
#include <algorithm>
#include <numeric>

#include "Neuron.h"
#include "random.h"
#include "activations.h"

static std::default_random_engine generator = initialize_random_seed();

Neuron::Neuron(int input_size)
{
    parameters.w.resize(input_size);
    
    std::normal_distribution<double> distribution(0, 0.2);
    for (int i = 0; i < input_size; i++)
    {
        parameters.w(i) = distribution(generator);
    }
    parameters.b = 0;
}

Eigen::VectorXd Neuron::getW()
{
    return parameters.w;
}

void Neuron::setW(Eigen::VectorXd newW)
{
    parameters.w = newW;
}

double Neuron::getB()
{
    return parameters.b;
}

void Neuron::setB(double newB)
{
    parameters.b = newB;
}

double Neuron::forward_prop(Eigen::VectorXd &x, ACTIVATION_FUNCTION activation_function)
{
    double z = preactivation(x, parameters.w, parameters.b);
    double a = activation(z, activation_function);

    return a;
}
