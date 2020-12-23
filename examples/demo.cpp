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
    Eigen::MatrixXd x = Eigen::VectorXd::LinSpaced(50, 1, 10);

    // Each element is expected value of the output of the neural network
    // for the corresponding training example
    Eigen::MatrixXd y = x.array() * 30;

    double xNormCoef = x.array().maxCoeff();
    double yNormCoef = y.array().maxCoeff();

    Eigen::MatrixXd xNorm = x.array() / xNormCoef;
    Eigen::MatrixXd yNorm = y.array() / yNormCoef;

#define NET_TEST
#ifdef NET_TEST

    Layer *l1 = new Layer(20, 1, ACTIVATION_FUNCTION::SIGMOID);
    Layer *l4 = new Layer(1, 20, ACTIVATION_FUNCTION::LINEAR);

    std::vector<Layer *> layers = {l1, l4};

    Network *nn = new Network(layers, 1);

    int EPOCHS = 5000;

    Eigen::VectorXd lossCache = nn->train(x, yNorm, EPOCHS, 0.00001);

    Eigen::MatrixXd results(y.rows(), y.cols());

    for (int i = 0; i < xNorm.rows(); i++)
    {
        results.row(i) = nn->forward_prop(x.row(i));
    }

    std::cout << "xNormCoef: " << xNormCoef << '\n';
    std::cout << "yNormCoef: " << yNormCoef << '\n';

    testPlot(x, results * yNormCoef, 0.0, y.maxCoeff());

    testPlot(x, y, 0.0, y.maxCoeff());

    Eigen::MatrixXd epochsX(EPOCHS, 1);
    epochsX << Eigen::VectorXd::LinSpaced(EPOCHS, 1, EPOCHS);

    testPlot(epochsX, lossCache, 0.0, lossCache.maxCoeff());

#endif

    system("pause");

    return 0;
}