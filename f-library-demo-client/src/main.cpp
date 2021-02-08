#include "FLibCore.h"
#include "Eigen/Dense"
#include <iostream>

int main() {
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

	Eigen::VectorXd loss = network.train(x, y, 1, 0.01);
	
	DataTypes::LayerHyperParameters layer1Params = layer1->getParams();
	DataTypes::LayerHyperParameters layer2Params = layer2->getParams();
	DataTypes::LayerHyperParameters layer3Params = layer3->getParams();

	std::cout << loss << '\n';

	std::cout << "layer 1: \n" << "w: " << layer1Params.w << "\nb: " << layer1Params.b << "\n---------\n";
	std::cout << "layer 2: \n" << "w: " << layer2Params.w << "\nb: " << layer2Params.b << "\n---------\n";
	std::cout << "layer 3: \n" << "w: " << layer3Params.w << "\nb: " << layer3Params.b << "\n---------\n";
}