#include <iostream>
#include "Eigen/Dense"
#include "FLibCore.h"
#include "window.h"

void action();
Eigen::VectorXd trainModel();

Eigen::VectorXd loss;

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
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

	Eigen::MatrixXd y(10, 1);
	y << x.array() * 3;

	Layer* layer1 = new Layer(1, 1, ACTIVATION_FUNCTION::LINEAR);
	Layer* layer2 = new Layer(1, 1, ACTIVATION_FUNCTION::LINEAR);
	Layer* layer3 = new Layer(1, 1, ACTIVATION_FUNCTION::LINEAR);

	Eigen::MatrixXd weightsLayer1(1, 1);
	weightsLayer1 << 1;

	Eigen::MatrixXd weightsLayer2(1, 1);
	weightsLayer2 << 2;

	Eigen::MatrixXd weightsLayer3(1, 1);
	weightsLayer3 << 0;

	layer1->setWeights(weightsLayer1);
	layer2->setWeights(weightsLayer2);
	layer3->setWeights(weightsLayer3);

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

	return loss;
}