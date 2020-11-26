#include <iostream>
#include <vector>
#include <math.h>

#include "Eigen/Dense"
#include "Layer.h"
#include "Network.h"
#include "propagation.h"
#include "activations.h"

int main()
{
    // Each training example contains 5 features
    // There are 3 training examples in total
    Eigen::MatrixXd x(10, 1);
    x << Eigen::VectorXd::LinSpaced(10, 1, 10);

    // Each element is expected value of the output of the neural network
    // for the corresponding training example
    Eigen::MatrixXd y(10, 1);
    y << x.array().pow(2);

#define NET_TEST
#ifdef NET_TEST

    Layer *l1 = new Layer(10, 1, ACTIVATION_FUNCTION::SIGMOID);
    Layer *l2 = new Layer(20, 10, ACTIVATION_FUNCTION::SIGMOID);
    Layer *l3 = new Layer(1, 20, ACTIVATION_FUNCTION::LINEAR);

    std::vector<Layer *> layers = {l1, l2, l3};

    Network *nn = new Network(layers);

    nn->train(x, y, 1000000, 0.1);

    Eigen::MatrixXd input(1, 1);
    input << 12;

    Eigen::VectorXd result = nn->forward_prop(input);

    std::cout << "result: " << result;

#endif

    system("pause");

    return 0;
}