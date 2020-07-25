#pragma once
#include <vector>

class Neuron
{
private:
    std::vector<double> w;
    double b;

public:
    Neuron(double input_size);
    double forward_prop(std::vector<double> x);
};