#pragma once
#include <vector>

#include "propagation.h"

class Neuron
{
private:
    std::vector<double> w;
    double b;

public:
    Neuron(double input_size);
    std::vector<double> getW();
    void setW(std::vector<double> newW);
    double forward_prop(std::vector<double> x, ACTIVATION_FUNCTION activation_function);
    double getB();
    void setB(double newB);
};