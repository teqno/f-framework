#include <iostream>
#include <vector>
#include <math.h>

#include "Eigen/Dense"
#include "Layer.h"
#include "Network.h"
#include "propagation.h"
#include "activations.h"

// void printNeurons(Layer *&l)
// {
//     auto params = l->getParams();
//     for (unsigned int i = 0; i < params.size(); ++i)
//     {
//         printf("neuron %i {\n", i);
//         printf("\tw: ");
//         for (auto const &w : params.at(i).first)
//         {
//             printf("%f; ", w);
//         }
//         printf("\n\tb: ");
//         printf("%f\n", params.at(i).second);
//         printf("}\n", i);
//     }
// }

int main()
{
    // Each training example contains 5 features
    // There are 3 training examples in total
    Eigen::MatrixXd x(3, 5);
    x << 1.0, 2.0, 3.0, 4.0, 5.0, 
        6.0, 7.0, 8.0, 9.0, 10.0,
        6.0, 7.0, 8.0, 9.0, 10.0;

    // Each element is expected value of the output of the neural network
    // for the corresponding training example
    Eigen::VectorXd y(3);
    y << 1, 0, 1;

    Layer *l1 = new Layer(3, 5, ACTIVATION_FUNCTION::TANH);
    Layer *l2 = new Layer(5, 3, ACTIVATION_FUNCTION::TANH);
    Layer *l3 = new Layer(5, 5, ACTIVATION_FUNCTION::TANH);
    Layer *l4 = new Layer(1, 5, ACTIVATION_FUNCTION::SIGMOID);

    std::vector<Layer *> layers = {l1, l2, l3, l4};

    Network *nn = new Network(layers);

    double cost = nn->calc_cost(x, y);

    std::cout << cost << std::endl;

#ifdef TEST1
    // std::vector<double> acts = nn->forward_prop(x);
    double loss = nn->calc_cost(x, y);

    printf("Loss: %f\n", loss);

#endif

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

    // for (auto &l : nn->getLayers())
    // {
    //     printNeurons(l);
    // }

    system("pause");

    return 0;
}