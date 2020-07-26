#pragma once

#include "propagation.h"

class Neuron
{
private:
    std::vector<double> w;
    double b;

public:
    Neuron(int input_size);
    std::vector<double> getW();
    void setW(std::vector<double> newW);
    double getB();
    void setB(double newB);
    double forward_prop(std::vector<double> &x, ACTIVATION_FUNCTION activation_function);
};