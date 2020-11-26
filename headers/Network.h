#pragma once

#include "Layer.h"
#include "loss.h"
#include "Cache.h"

class Network
{
private:
    std::vector<Layer *> layers;
    std::vector<Eigen::MatrixXd> activations;
    std::vector<Eigen::MatrixXd> preactivations;

public:
    Network(std::vector<Layer *> &layers);
    
    std::vector<Layer *> getLayers();
    
    Eigen::MatrixXd forward_prop(Eigen::MatrixXd &input);
    
    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> back_prop(Eigen::MatrixXd y);
    
    void updateParameters(std::vector<Eigen::MatrixXd> dw, std::vector<Eigen::MatrixXd> db, double alpha);
    
    double calc_cost(Eigen::MatrixXd &x, Eigen::MatrixXd &y);
    
    void train(Eigen::MatrixXd &x, Eigen::MatrixXd &y, int epochs, double alpha);
};