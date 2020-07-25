#include <iostream>
#include <vector>
#include <math.h>

#include "Neuron.h"
#include "propagation.h"
#include "activations.h"

int main()
{
    std::vector<double> x = {2, 4, 6, 8, 10};
    std::vector<double> y = {1};

    //TODO: FIX BACKPROP

    Neuron *n11 = new Neuron(5);
    // Neuron *n12 = new Neuron(5);
    // Neuron *n13 = new Neuron(5);
    // Neuron *n14 = new Neuron(5);
    // Neuron *n15 = new Neuron(5);

    // Neuron *n21 = new Neuron(5);
    // Neuron *n22 = new Neuron(5);
    // Neuron *n23 = new Neuron(5);
    // Neuron *n24 = new Neuron(5);
    // Neuron *n25 = new Neuron(5);

    Neuron *n31 = new Neuron(1);

    std::vector<double> x1 = x;

    double h11 = n11->forward_prop(x1, ACTIVATION_FUNCTION::TANH);
    // double h12 = n12->forward_prop(x1, ACTIVATION_FUNCTION::TANH);
    // double h13 = n13->forward_prop(x1, ACTIVATION_FUNCTION::TANH);
    // double h14 = n14->forward_prop(x1, ACTIVATION_FUNCTION::TANH);
    // double h15 = n15->forward_prop(x1, ACTIVATION_FUNCTION::TANH);

    // std::vector<double> x2 = {h11, h12, h13, h14, h15};

    // double h21 = n21->forward_prop(x2, ACTIVATION_FUNCTION::TANH);
    // double h22 = n22->forward_prop(x2, ACTIVATION_FUNCTION::TANH);
    // double h23 = n23->forward_prop(x2, ACTIVATION_FUNCTION::TANH);
    // double h24 = n24->forward_prop(x2, ACTIVATION_FUNCTION::TANH);
    // double h25 = n25->forward_prop(x2, ACTIVATION_FUNCTION::TANH);

    std::vector<double> x3 = {h11};

    double h31 = n31->forward_prop(x3, ACTIVATION_FUNCTION::SIGMOID);

    double y_hat = h31;

    std::cout << "y_hat: " << h31 << std::endl;

    double cost = 1.0 / 2.0 * pow(y[0] - y_hat, 2);

    std::cout << "cost: " << cost << std::endl;

    std::vector<double> w11 = n11->getW();
    double b11 = n11->getB();
    // std::vector<double> w12 = n12->getW();
    // std::vector<double> w13 = n13->getW();
    // std::vector<double> w14 = n14->getW();
    // std::vector<double> w15 = n15->getW();

    // std::vector<double> w21 = n21->getW();
    // std::vector<double> w22 = n22->getW();
    // std::vector<double> w23 = n23->getW();
    // std::vector<double> w24 = n24->getW();
    // std::vector<double> w25 = n25->getW();

    std::vector<double> w31 = n31->getW();
    double b31 = n31->getB();

    auto getSigmoidDerivative = [](double x) { return sigmoid(x) * (1 - sigmoid(x)); };
    auto getTanhDerivative = [](double x) { return 1 - pow(tanh(x), 2); };

    // std::vector<std::vector<double>> w1 = {w11, w12, w13, w14, w15};
    // std::vector<std::vector<double>> w2 = {w21, w22, w23, w24, w25};
    // std::vector<double> &w3 = w31;

    double alpha = 0.05;

    double delta3w = alpha * getSigmoidDerivative(cost);
    
    b31 = b31 - alpha * getSigmoidDerivative(cost);


    w31[0] -= delta3w;
    b31 -= 
    w11[0] -= delta1;
    // for (std::vector<double> &w : w2)
    // {
    //     w = alpha * getTanhDerivative(w3[0]);
    // }

    return 0;
}