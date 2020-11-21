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
    void setW(Eigen::VectorXd newW);
    double getB();
    void setB(double newB);
    std::map<std::string, double> forward_prop(Eigen::VectorXd &x, ACTIVATION_FUNCTION activation_function);
};