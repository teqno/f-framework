#include <iostream>
#include <vector>
#include <math.h>

#include "Eigen/Dense"
#include "Layer.h"
#include "Network.h"
#include "propagation.h"
#include "activations.h"

#include "gnuplot-iostream.h"

int testPlot(Eigen::MatrixXd x, Eigen::MatrixXd y, int yMin, int yMax)
{
    std::vector<std::pair<double, double>> data;
    for (int i = 0; i < x.rows(); i++)
    {
        data.emplace_back(x.row(i)(0), y.row(i)(0));
    }

    double xMin = x.minCoeff();
    double xMax = x.maxCoeff();
    // double yMin = y.minCoeff();
    // double yMax = y.maxCoeff();

    Gnuplot gp;
    gp << "set xrange [" << xMin << ":" << xMax << "]\nset yrange [" << yMin << ":" << yMax << "]\n";
    gp << "plot sin(x) tit 'sin(x)', '-' tit 'data'\n";
    gp.send1d(data);
    return 0;
}

int main()
{
    // Each training example contains 1 feature
    // There are 10 training examples in total
    Eigen::MatrixXd x(500, 1);
    x << Eigen::VectorXd::LinSpaced(500, 1, 10);

    // Each element is expected value of the output of the neural network
    // for the corresponding training example
    Eigen::MatrixXd y(500, 1);
    y << x.array().sin();

    double xNormCoef = x.maxCoeff();
    double yNormCoef = y.maxCoeff();

    Eigen::MatrixXd xNorm = x.array() / xNormCoef;
    Eigen::MatrixXd yNorm = y.array() / yNormCoef;

#define NET_TEST
#ifdef NET_TEST

    Layer *l1 = new Layer(30, 1, ACTIVATION_FUNCTION::TANH);
    Layer *l2 = new Layer(30, 30, ACTIVATION_FUNCTION::TANH);
    Layer *l3 = new Layer(30, 30, ACTIVATION_FUNCTION::TANH);
    Layer *l4 = new Layer(30, 30, ACTIVATION_FUNCTION::TANH);
    Layer *l5 = new Layer(1, 30, ACTIVATION_FUNCTION::LINEAR);
    
    std::vector<Layer *> layers = {l1, l2, l3, l4, l5};

    Network *nn = new Network(layers);

    int EPOCHS = 10;

    Eigen::VectorXd lossCache = nn->train(xNorm, yNorm, EPOCHS, 0.001, 1.7);

    Eigen::MatrixXd results(y.rows(), y.cols());
    for (int i = 0; i < xNorm.rows(); i++) {
        Eigen::VectorXd result = nn->forward_prop(xNorm.row(i));
        results.row(i) = result;
    }

    std::cout << results << '\n';

    testPlot(x, results * yNormCoef, -1, 1);

    testPlot(x, y, -1, 1);

    Eigen::MatrixXd epochsX(EPOCHS, 1);
    epochsX << Eigen::VectorXd::LinSpaced(EPOCHS, 1, EPOCHS);

    testPlot(epochsX, lossCache, lossCache.minCoeff(), lossCache.maxCoeff());

#endif

    system("pause");

    return 0;
}