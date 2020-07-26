#include <iostream>
#include <vector>
#include <math.h>

// #include "Neuron.h"
#include "Layer.h"
#include "propagation.h"
#include "activations.h"

int main()
{
    std::vector<double> x = {2, 4, 6, 8, 10};
    std::vector<double> y = {1};

    Layer *l = new Layer(15, 5, ACTIVATION_FUNCTION::TANH);

    std::vector<double> acts;
    acts = l->forward_prop(x);

    for (auto const &a : acts)
    {
        printf("%f\n", a);
    }

    system("pause");

    // Neuron *n1 = new Neuron(5);
    // Neuron *n2 = new Neuron(5);
    // Neuron *n3 = new Neuron(5);
    // Neuron *n4 = new Neuron(5);

    // std::cout << n1->forward_prop(x, ACTIVATION_FUNCTION::SIGMOID) << std::endl;
    // std::cout << n2->forward_prop(x, ACTIVATION_FUNCTION::SIGMOID) << std::endl;
    // std::cout << n3->forward_prop(x, ACTIVATION_FUNCTION::SIGMOID) << std::endl;
    // std::cout << n4->forward_prop(x, ACTIVATION_FUNCTION::SIGMOID) << std::endl;

    // Neuron *n31 = new Neuron(1);

    // std::vector<double> x1 = x;

    // double h11 = n11->forward_prop(x1, ACTIVATION_FUNCTION::TANH);

    // std::vector<double> x3 = {h11};

    // double h31 = n31->forward_prop(x3, ACTIVATION_FUNCTION::SIGMOID);

    // double y_hat = h31;

    // std::cout << "y_hat: " << h31 << std::endl;

    // double cost = 1.0 / 2.0 * pow(y[0] - y_hat, 2);

    // std::cout << "cost: " << cost << std::endl;

    // std::vector<double> w11 = n11->getW();
    // double b11 = n11->getB();

    // std::vector<double> w31 = n31->getW();
    // double b31 = n31->getB();

    // auto getSigmoidDerivative = [](double x) { return sigmoid(x) * (1 - sigmoid(x)); };
    // auto getTanhDerivative = [](double x) { return 1 - pow(tanh(x), 2); };

    // double alpha = 0.05;

    // double delta3w = alpha * getSigmoidDerivative(cost);

    // b31 = b31 - alpha * getSigmoidDerivative(cost);

    // w31[0] -= alpha * delta3w;
    // b31 -= alpha * delta3w;
    // w11[0] -= delta1;

    // for (std::vector<double> &w : w2)
    // {
    //     w = alpha * getTanhDerivative(w3[0]);
    // }

    return 0;
}