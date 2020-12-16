#include <iostream>
#include <vector>
#include <math.h>

#include "Eigen/Dense"
#include "Layer.h"
#include "Network.h"
#include "propagation.h"
#include "activations.h"

#include "gnuplot-iostream.h"

int testPlot(Eigen::MatrixXd x, Eigen::MatrixXd y, double yMin, double yMax)
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
    Eigen::MatrixXd x(50, 1);
    x << Eigen::VectorXd::LinSpaced(50, 1, 10);

    // Each element is expected value of the output of the neural network
    // for the corresponding training example
    Eigen::MatrixXd y(50, 1);
    y << x.array().sin();

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

    Eigen::VectorXd lossCache = nn->train(xNorm, yNorm, EPOCHS, 0.05, 0);

    Eigen::MatrixXd xTest(10, 1);
    xTest << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
    Eigen::MatrixXd xTestNorm = xTest / xTest.maxCoeff();

    Eigen::MatrixXd results(y.rows(), y.cols());
    for (int i = 0; i < xTestNorm.rows(); i++)
    {
        Eigen::VectorXd result = nn->forward_prop(xNorm.row(i));
        results.row(i) = result;
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