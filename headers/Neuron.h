#pragma once

#include "propagation.h"
#include "Eigen/Dense"

class Neuron
{
private:
    Eigen::VectorXd w;
    double b;
    double z;

public:
    Neuron(int input_size);
    Eigen::VectorXd getW();
    void setW(Eigen::VectorXd newW);
    double getB();
    void setB(double newB);
    double getZ();
    void setZ(double newZ);
    double forward_prop(Eigen::VectorXd &x, ACTIVATION_FUNCTION activation_function);
};