#pragma once

#include <map>
#include "propagation.h"
#include "HyperParameters.h"
#include "Eigen/Dense"

class Neuron
{
private:
    HyperParameters parameters;

public:
    Neuron(int input_size);
    HyperParameters getParameters();
    Eigen::VectorXd getW();
    void setW(const Eigen::VectorXd &newW);
    double getB();
    void setB(double newB);
};