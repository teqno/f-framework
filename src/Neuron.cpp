#include <cstdlib>
#include <random>
#include <time.h>
#include <algorithm>
#include <vector>
#include <numeric>

#include "Neuron.h"
#include "random.h"
#include "propagation.h"
#include "activations.h"

static std::default_random_engine generator = initialize_random_seed();

Neuron::Neuron(double input_size)
{
    std::normal_distribution<double> distribution(0, 0.2);
    for (int i = 0; i < input_size; i++)
    {
        w.push_back(distribution(generator));
    }
    b = (double)0;
}

std::vector<double> Neuron::getW() {
    return w;
}

void Neuron::setW(std::vector<double> newW) {
    w = newW;
}

double Neuron::getB() {
    return b;
}

void Neuron::setB(double newB) {
    b = newB;
}

double Neuron::forward_prop(std::vector<double> x, ACTIVATION_FUNCTION activation_function)
{
    double a = preactivation(x, w, b);
    double h = activation(a, activation_function);

    return h;
}
