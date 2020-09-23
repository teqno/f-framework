#include <iostream>
#include <vector>
#include <math.h>

// #include <boost/numeric/ublas/matrix.hpp>
// #include <boost/numeric/ublas/io.hpp>

// #include "Neuron.h"
#include "Layer.h"
#include "Network.h"
#include "propagation.h"
#include "activations.h"

void printNeurons(Layer *&l)
{
    auto params = l->getParams();
    for (unsigned int i = 0; i < params.size(); ++i)
    {
        printf("neuron %i {\n", i);
        printf("\tw: ");
        for (auto const &w : params.at(i).first)
        {
            printf("%f; ", w);
        }
        printf("\n\tb: ");
        printf("%f\n", params.at(i).second);
        printf("}\n", i);
    }
}

int main()
{
    // {
    //     using namespace boost::numeric::ublas;
    //     matrix<double> m(3, 4);
    //     matrix<double> n(4, 1);
    //     for (unsigned i = 0; i < m.size1(); ++i)
    //         for (unsigned j = 0; j < m.size2(); ++j)
    //             m(i, j) = 3 * i + j;

    //     for (unsigned i = 0; i < n.size1(); ++i)
    //         for (unsigned j = 0; j < n.size2(); ++j)
    //             n(i, j) = 3 * i + j;

    //     std::cout << m << std::endl;
    //     std::cout << n << std::endl;
    //     std::cout << prod(m, n) << std::endl;
    // }

    std::vector<std::vector<double>> x;
    std::vector<double> x1 = {1.0, 2.0, 3.0};
    std::vector<double> x2 = {1.0, 2.0, 3.0};
    std::vector<double> x3 = {1.0, 2.0, 3.0};

    x.push_back(x1);
    x.push_back(x2);
    x.push_back(x3);

    std::vector<double> y = {1, 2, 3};

    Layer *l1 = new Layer(3, 5, ACTIVATION_FUNCTION::TANH);
    Layer *l2 = new Layer(5, 3, ACTIVATION_FUNCTION::TANH);
    Layer *l3 = new Layer(5, 5, ACTIVATION_FUNCTION::TANH);
    Layer *l4 = new Layer(1, 5, ACTIVATION_FUNCTION::SIGMOID);

    std::vector<Layer *> layers = {l1, l2, l3, l4};

    Network *nn = new Network(layers);

    // std::vector<double> acts = nn->forward_prop(x);
    double loss = nn->calc_loss(x, y);

    printf("Loss: %f\n", loss);

    // std::vector<double> acts1 = l1->forward_prop(x);
    // std::vector<double> acts2 = l2->forward_prop(acts1);
    // std::vector<double> acts3 = l3->forward_prop(acts2);
    // std::vector<double> acts4 = l4->forward_prop(acts3);

    // for (auto const &a : acts4)
    // {
    //     printf("%f\n", a);
    // }

    // for (auto const &a : acts)
    // {
    //     printf("%f\n", a);
    // }

    for (auto &l : nn->getLayers())
    {
        printNeurons(l);
    }

    system("pause");

    return 0;
}