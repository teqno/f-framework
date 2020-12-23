#include <iostream>
#include <vector>
#include <math.h>

#include "Eigen/Dense"
#include "Layer.h"
#include "Network.h"
#include "propagation.h"
#include "activations.h"

#include "gnuplot-iostream.h"

int testPlot(Eigen::VectorXd x, Eigen::VectorXd y, double yMin, double yMax)
{
    std::vector<std::pair<double, double>> data;
    for (int i = 0; i < x.rows(); i++)
    {
        data.emplace_back(x(i), y(i));
    }

    double xMin = x.minCoeff();
    double xMax = x.maxCoeff();
    // double yMin = y.minCoeff();
    // double yMax = y.maxCoeff();

    Gnuplot gp;
    gp << "set xrange [" << xMin << ":" << xMax << "]\nset yrange [" << yMin << ":" << yMax << "]\n";
    gp << "plot '-' tit 'data'\n";
    gp.send1d(data);
    return 0;
}

int main()
{
    // Each training example contains 1 feature
    // There are 10 training examples in total
    Eigen::MatrixXd x(50, 1);
    x << Eigen::VectorXd::LinSpaced(50, 1, 10);

    // Each element is expected value of the output of the neural network
    // for the corresponding training example
    Eigen::MatrixXd y(50, 1);
    y << x.array() * 3 + 10;

    double xNormCoef = x.maxCoeff();
    double yNormCoef = y.maxCoeff();

    Eigen::MatrixXd xNorm = x.array() / xNormCoef;
    Eigen::MatrixXd yNorm = y.array() / yNormCoef;

#define NET_TEST
#ifdef NET_TEST

    Layer *l1 = new Layer(1, 1, ACTIVATION_FUNCTION::TANH);
    Layer *l2 = new Layer(2, 1, ACTIVATION_FUNCTION::TANH);
    Layer *l3 = new Layer(1, 2, ACTIVATION_FUNCTION::LINEAR);

    std::vector<Layer *> layers = {l1, l2, l3};

    Network *nn = new Network(layers, 1);

    int EPOCHS = 2000;

    Eigen::VectorXd lossCache = nn->train(xNorm, yNorm, EPOCHS, 0.001);

    Eigen::MatrixXd results(y.rows(), y.cols());

    for (int i = 0; i < xNorm.rows(); i++)
    {
        results.row(i) = nn->forward_prop(xNorm.row(i));
    }

    std::cout << results * yNormCoef << '\n';

    testPlot(xNorm * xNormCoef, results * yNormCoef, 0.0, y.maxCoeff());

    testPlot(x, y, 0.0, y.maxCoeff());

    Eigen::MatrixXd epochsX(EPOCHS, 1);
    epochsX << Eigen::VectorXd::LinSpaced(EPOCHS, 1, EPOCHS);

    testPlot(epochsX, lossCache, 0.0, lossCache.maxCoeff());

#endif

    system("pause");

    return 0;
}