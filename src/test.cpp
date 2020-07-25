#include <iostream>
#include <vector>
#include "Neuron.h"

int main()
{
    Neuron *n1 = new Neuron(5);
    Neuron *n2 = new Neuron(5);

    std::vector<double> x = {1, 2, 3, 4, 5};
    double y_hat1 = n1->forward_prop(x);
    double y_hat2 = n2->forward_prop(x);

    std::cout << "y_hat1 is: " << y_hat1 << std::endl;
    std::cout << "y_hat2 is: " << y_hat2 << std::endl;
    return 0;
}