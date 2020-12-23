#pragma once

#include <map>
#include "propagation.h"
#include "DataTypes.h"
#include "Eigen/Dense"

class Neuron
{
private:
    DataTypes::NeuronHyperParameters parameters;

public:
    Neuron(int input_size);
    DataTypes::NeuronHyperParameters getParameters();
    Eigen::VectorXd getW();
    void setW(const Eigen::VectorXd &newW);
    double getB();
    void setB(double newB);
};