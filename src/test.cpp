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

    Layer *l1 = new Layer(3, 5, ACTIVATION_FUNCTION::TANH);
    Layer *l2 = new Layer(5, 3, ACTIVATION_FUNCTION::TANH);
    Layer *l3 = new Layer(5, 5, ACTIVATION_FUNCTION::TANH);
    Layer *l4 = new Layer(1, 5, ACTIVATION_FUNCTION::SIGMOID);

    std::vector<double> acts1 = l1->forward_prop(x);
    std::vector<double> acts2 = l2->forward_prop(acts1);
    std::vector<double> acts3 = l3->forward_prop(acts2);
    std::vector<double> acts4 = l4->forward_prop(acts3);

    for (auto const &a : acts4)
    {
        printf("%f\n", a);
    }

    auto params = l1->getParams();
    for (unsigned int i = 0; i < params.size(); ++i)
    {
        printf("neuron %i {\n", i);
        printf("\tw: ");
        for (auto const &w : params.at(i).first) {
            printf("%f; ", w);    
        }
        printf("\n\tb: ");
        printf("%f\n", params.at(i).second);
        printf("}\n", i);
    }

    system("pause");

    return 0;
}