#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "FLibCore.h"
#include "window.h"

void action();
Eigen::VectorXd trainModel();

Eigen::VectorXd loss;

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
	setDebugFile();

	loss = trainModel();

	init_window(hInstance, action);
}

void action() {
	void* buffer_memory = get_buffer_memory_ref();
	int buffer_height = get_buffer_height();
	int buffer_width = get_buffer_width();
	double coef_x = (double)loss.size() / buffer_width;
	double coef_y = loss.maxCoeff() / buffer_height;

	unsigned int* pixel = (unsigned int*)buffer_memory;
	for (int y = 0; y < buffer_height; y++) {
		for (int x = 0; x < buffer_width; x++) {
			int coeffed_x = coef_x * x;

			if (coeffed_x < loss.size() && y - (int)(loss(coeffed_x) / coef_y) < 0) {
				*pixel = 0xff0000;
			}
			else {
				*pixel = 0xffffff;
			}
			pixel++;
		}
	}

	display_buffer();
}

Eigen::VectorXd trainModel() {
	Eigen::MatrixXd x(10, 1);
	x << Eigen::VectorXd::LinSpaced(10, 0, 1);

	debug("x:");
	debug(x);

	Eigen::MatrixXd y(10, 1);
	y << x.array().sin();

	debug("y:");
	debug(y);

	Layer* layer1 = new Layer(1, 1, ACTIVATION_FUNCTION::SIGMOID);
	Layer* layer2 = new Layer(1, 1, ACTIVATION_FUNCTION::SIGMOID);
	Layer* layer3 = new Layer(1, 1, ACTIVATION_FUNCTION::LINEAR);

	debug("layer 1 weights:");
	debug(layer1->getParams().w);

	debug("layer 2 weights:");
	debug(layer2->getParams().w);

	debug("layer 3 weights:");
	debug(layer3->getParams().w);

	debug("layer 1 biases:");
	debug(layer1->getParams().b);

	debug("layer 2 biases:");
	debug(layer2->getParams().b);

	debug("layer 3 biases:");
	debug(layer3->getParams().b);

	Eigen::MatrixXd weightsLayer1(1, 1);
	weightsLayer1 << 0.4;

	Eigen::MatrixXd weightsLayer2(1, 1);
	weightsLayer2 << 0.5;

	Eigen::MatrixXd weightsLayer3(1, 1);
	weightsLayer3 << 0.6;

	layer1->setWeights(weightsLayer1);
	layer2->setWeights(weightsLayer2);
	layer3->setWeights(weightsLayer3);

	debug("layer 1 updated weights:");
	debug(layer1->getParams().w);

	debug("layer 2 updated weights:");
	debug(layer2->getParams().w);

	debug("layer 3 updated weights:");
	debug(layer3->getParams().w);

	std::vector<Layer*> layers = { layer1, layer2, layer3 };

	Network network = Network(layers, 0);

	Eigen::VectorXd loss = network.train(x, y, 1000, 0.01);

	DataTypes::LayerHyperParameters layer1Params = layer1->getParams();
	DataTypes::LayerHyperParameters layer2Params = layer2->getParams();
	DataTypes::LayerHyperParameters layer3Params = layer3->getParams();

	std::cout << loss << '\n';

	std::cout << "layer 1: \n"
		<< "w: " << layer1Params.w << "\nb: " << layer1Params.b
		<< "\n---------\n";
	std::cout << "layer 2: \n"
		<< "w: " << layer2Params.w << "\nb: " << layer2Params.b
		<< "\n---------\n";
	std::cout << "layer 3: \n"
		<< "w: " << layer3Params.w << "\nb: " << layer3Params.b
		<< "\n---------\n";

	
	std::vector<Eigen::VectorXd> result;

	for (int i = 0; i < x.rows(); i++) {
		result.push_back(network.forward_prop(x.row(i)));
	}

	debug("Network result:");
	for (auto item : result) {
		debug(item);
	}

	debug("Expected result:");
	debug(y);

	return loss;
}