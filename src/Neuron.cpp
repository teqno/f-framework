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
    w.resize(input_size);
    std::normal_distribution<double> distribution(0, 0.2);
    for (int i = 0; i < input_size; i++)
    {
        double num = distribution(generator);
        w(i) = num;
    }
    b = 0;
}

Eigen::VectorXd Neuron::getW()
{
    return w;
}

void Neuron::setW(Eigen::VectorXd newW)
{
    w = newW;
}

double Neuron::getB()
{
    return b;
}

void Neuron::setB(double newB)
{
    b = newB;
}

double Neuron::getZ()
{
    return z;
}

void Neuron::setZ(double newZ)
{
    z = newZ;
}

double Neuron::forward_prop(Eigen::VectorXd &x, ACTIVATION_FUNCTION activation_function)
{
    z = preactivation(x, w, b);
    double a = activation(z, activation_function);

    return a;
}
