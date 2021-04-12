#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "FLibCore.h"
#include "window.h"

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> trainModel();

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
	setDebugFile();

	std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> result = trainModel();

	//init_window(hInstance, std::get<0>(result), std::get<1>(result), std::get<2>(result), 400, std::get<0>(result).maxCoeff() / std::get<0>(result).size() * 400.0);
	init_window(hInstance, std::get<0>(result), std::get<1>(result), std::get<2>(result), 800, 600);
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> trainModel() {
	Eigen::MatrixXd x(100, 1);
	x << Eigen::VectorXd::LinSpaced(100, -50, 50).array() / 50;

	debug("x:");
	debug(x);

	Eigen::MatrixXd y(100, 1);
	y << (x.array() * 50).pow(3) / 125000.0;

	debug("y:");
	debug(y);

	std::vector<Layer*> layers = {
		new Layer(3, 1, ACTIVATION_FUNCTION::SIGMOID),
		new Layer(1, 3, ACTIVATION_FUNCTION::LINEAR)
	};

	Network network = Network(layers, 0);

	Eigen::VectorXd loss = network.train(x, y, 2000, 0.003, 0.9);

	debug(loss);

	Eigen::VectorXd result(x.rows());

	for (int i = 0; i < x.rows(); i++) {
		result[i] = network.forward_prop(x.row(i))[0] * 125000.0;
	}

	debug("Network result:");
	for (auto item : result) {
		debug(item);
	}

	debug("Expected result:");
	debug(y);

	return std::make_tuple(result, y.col(0) * 125000.0, loss);
}